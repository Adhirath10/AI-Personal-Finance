import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


REQUIRED_COLUMNS = [
    "user_id",
    "transaction_date",
    "amount",
    "transaction_type",
]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "forecasting"
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "personal_finance.db"


def resolve_database_path(database_path: str | None = None) -> Path:
    """Resolve the SQLite database path from CLI, environment, or default project location."""
    if database_path:
        path = Path(database_path).expanduser()
    else:
        env_path = os.getenv("FINANCE_DB_PATH")
        path = Path(env_path).expanduser() if env_path else DEFAULT_DATABASE_PATH

    resolved = path if path.is_absolute() else (Path.cwd() / path)
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved}")
    return resolved


def build_monthly_expense_data(database_path: str | Path | None = None) -> pd.DataFrame:
    """Aggregate monthly expenses per user from real transaction data while excluding income records."""
    resolved_db_path = resolve_database_path(str(database_path) if database_path is not None else None)
    connection = sqlite3.connect(f"file:{resolved_db_path.as_posix()}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {table_name for (table_name,) in tables}
        if "transactions" not in table_names:
            raise ValueError("Table not found: transactions")

        columns = connection.execute('PRAGMA table_info("transactions")').fetchall()
        available = [column[1] for column in columns]
        missing = [column for column in REQUIRED_COLUMNS if column not in available]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        query = "SELECT user_id, transaction_date, amount, transaction_type FROM transactions WHERE transaction_type = 'Expense'"
        df = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    if df.empty:
        raise ValueError("Empty dataset: no expense transactions available for forecasting.")

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "user_id", "amount"]).copy()
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["user_id", "month"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "total_expense"})
        .sort_values(["user_id", "month"])
        .reset_index(drop=True)
    )
    return monthly[["user_id", "month", "total_expense"]]


LEGACY_FEATURE_COLUMNS = [
    "previous_1",
    "previous_2",
    "previous_3",
    "rolling_3",
    "rolling_6",
    "month_number",
    "spending_change",
]
# Scale-free features: each user's months compared with their own 6-month average,
# so one model works for people who spend Rs 20k and Rs 2 lakh a month alike.
RELATIVE_FEATURE_COLUMNS = [
    "latest_vs_avg6",
    "prev1_vs_avg6",
    "prev2_vs_avg6",
    "prev3_vs_avg6",
    "avg3_vs_avg6",
    "spending_change",
    "month_number",
    "history_months",
]
RELATIVE_TARGET = "ratio_to_rolling_3"
RATIO_LIMITS = (0.6, 1.6)
# Half model, half the user's own 3-month average: with about a year of history
# this is as accurate as either alone and much steadier for unusual users.
MODEL_WEIGHT = 0.5
METADATA_FILE = "forecast_metadata.json"


def _ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator > 0, numerator / denominator, 0.0)
    return np.where(np.isfinite(out), out, 0.0)


def build_forecast_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged forecast features for every user-month, including the latest month.

    Each row describes the history up to and including ``month`` and is used to
    predict the following month. ``target_expense`` is NaN for each user's latest
    month because that next month has not been observed yet; those rows are the
    inputs for an actual next-month forecast.
    """
    if monthly_df.empty:
        raise ValueError("Empty monthly expense dataset.")

    monthly_df = monthly_df.copy().sort_values(["user_id", "month"]).reset_index(drop=True)
    monthly_df["month"] = pd.PeriodIndex(monthly_df["month"], freq="M")
    grouped = monthly_df.groupby("user_id")["total_expense"]
    monthly_df["month_number"] = monthly_df["month"].dt.month
    monthly_df["previous_1"] = grouped.shift(1)
    monthly_df["previous_2"] = grouped.shift(2)
    monthly_df["previous_3"] = grouped.shift(3)
    monthly_df["rolling_3"] = grouped.transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    monthly_df["rolling_6"] = grouped.transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    monthly_df["spending_change"] = grouped.pct_change().fillna(0)
    monthly_df["target_expense"] = grouped.shift(-1)

    for column in ["previous_1", "previous_2", "previous_3", "rolling_3", "rolling_6", "spending_change"]:
        monthly_df[column] = monthly_df[column].replace([np.inf, -np.inf], np.nan).fillna(0)

    monthly_df["history_months"] = np.minimum(monthly_df.groupby("user_id").cumcount() + 1, 12)
    monthly_df["latest_vs_avg6"] = _ratio(monthly_df["total_expense"], monthly_df["rolling_6"])
    for lag in (1, 2, 3):
        monthly_df[f"prev{lag}_vs_avg6"] = _ratio(monthly_df[f"previous_{lag}"], monthly_df["rolling_6"])
    monthly_df["avg3_vs_avg6"] = _ratio(monthly_df["rolling_3"], monthly_df["rolling_6"])
    monthly_df[RELATIVE_TARGET] = _ratio(monthly_df["target_expense"], monthly_df["rolling_3"])
    monthly_df.loc[monthly_df["target_expense"].isna(), RELATIVE_TARGET] = np.nan
    return monthly_df


def create_forecast_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Build a no-leakage feature set for next-month expense prediction using only lagged historical information."""
    features = build_forecast_features(monthly_df)
    return features.dropna(subset=["target_expense"]).copy()


def is_relative(metadata: dict | None, feature_columns: list[str] | None = None) -> bool:
    """True for the personalised model; metadata must describe the loaded model's features."""
    return bool(
        metadata
        and metadata.get("target") == RELATIVE_TARGET
        and (feature_columns is None or list(metadata.get("feature_columns", [])) == list(feature_columns))
    )


def predict_ratio(model, metadata: dict, rows: pd.DataFrame) -> np.ndarray:
    """Blended, limited prediction of next month / own 3-month average."""
    if metadata.get("selected") == "moving_average":
        return np.ones(len(rows))
    weight = float(metadata.get("model_weight", MODEL_WEIGHT))
    raw = np.asarray(model.predict(rows[metadata["feature_columns"]]), dtype=float)
    return np.clip(weight * raw + (1 - weight), *RATIO_LIMITS)


def predict_expense(model, metadata: dict | None, rows: pd.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
    """Predicted next-month spending in rupees for feature rows.

    Older models predict rupees directly; they are kept within the same plausible
    range around the user's own average so one user's scale cannot produce a wild
    number (retrain to get the personalised model).
    """
    average = rows["rolling_3"].to_numpy(dtype=float)
    if is_relative(metadata, feature_columns):
        return predict_ratio(model, metadata, rows) * average
    columns = list(feature_columns or LEGACY_FEATURE_COLUMNS)
    raw = np.asarray(model.predict(rows[columns]), dtype=float)
    return np.where(average > 0, np.clip(raw, average * RATIO_LIMITS[0], average * RATIO_LIMITS[1]), raw)


def load_forecast_metadata(model_dir: str | Path | None) -> dict | None:
    path = Path(model_dir or MODEL_DIR) / METADATA_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def evaluate_forecast(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and MAPE with a safe zero-handling rule."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-9, 1.0, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def chronological_split(dataset: pd.DataFrame, test_months: int = 2):
    """Test on the latest feature months of every user; train on all earlier months."""
    months = sorted(dataset["month"].unique())
    if len(months) <= test_months + 2:
        raise ValueError("Not enough months of history for a chronological evaluation split.")
    cutoff = months[-test_months]
    return dataset[dataset["month"] < cutoff].copy(), dataset[dataset["month"] >= cutoff].copy()


def train_and_evaluate_model(database_path: str | Path | None = None, model_dir: str | Path | None = None):
    """Train the personalised forecaster and compare it with simple baselines on later months."""
    monthly = build_monthly_expense_data(database_path)
    dataset = create_forecast_dataset(monthly)
    dataset = dataset[dataset["rolling_3"] > 0]
    if dataset.empty:
        raise ValueError("No feature rows available after constructing the forecasting dataset.")
    train_df, test_df = chronological_split(dataset)
    y_test = test_df["target_expense"]

    results = {
        "Baseline (Moving Average of last 3 months)": evaluate_forecast(y_test, test_df["rolling_3"]),
        "Baseline (Last month)": evaluate_forecast(y_test, test_df["total_expense"]),
    }
    legacy = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    legacy.fit(train_df[LEGACY_FEATURE_COLUMNS], train_df["target_expense"])
    results["Previous model (Random Forest on rupee amounts)"] = evaluate_forecast(
        y_test, legacy.predict(test_df[LEGACY_FEATURE_COLUMNS])
    )

    def relative_model():
        # Shallow trees with large leaves: monthly spending is noisy and a year of
        # history is short, so a heavily regularised forest generalises best.
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=25, max_depth=4, random_state=42, n_jobs=-1)

    model = relative_model()
    model.fit(train_df[RELATIVE_FEATURE_COLUMNS], train_df[RELATIVE_TARGET])
    metadata = {"target": RELATIVE_TARGET, "feature_columns": RELATIVE_FEATURE_COLUMNS, "selected": "model", "model_weight": MODEL_WEIGHT}
    ml_metrics = evaluate_forecast(y_test, predict_expense(model, metadata, test_df))
    baseline = results["Baseline (Moving Average of last 3 months)"]
    # Keep the model unless the plain average is clearly better (beyond noise).
    selected = "model" if ml_metrics["mae"] <= baseline["mae"] * 1.02 else "moving_average"

    # Final model: refit on every available month.
    final = relative_model()
    final.fit(dataset[RELATIVE_FEATURE_COLUMNS], dataset[RELATIVE_TARGET])
    metadata = {
        "target": RELATIVE_TARGET,
        "feature_columns": RELATIVE_FEATURE_COLUMNS,
        "selected": selected,
        "model_weight": MODEL_WEIGHT,
        "ratio_limits": list(RATIO_LIMITS),
        "test_months": [str(m) for m in sorted(test_df["month"].unique())],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": ml_metrics if selected == "model" else baseline,
    }
    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, target_dir / "forecasting_model.pkl")
    joblib.dump(RELATIVE_FEATURE_COLUMNS, target_dir / "feature_columns.pkl")
    (target_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    chosen = ml_metrics if selected == "model" else baseline
    report = [
        "Expense Forecasting Evaluation Report",
        "=" * 40,
        f"Validation: every user's latest {len(metadata['test_months'])} months held out "
        f"({', '.join(metadata['test_months'])}); trained on earlier months.",
        f"Train rows: {len(train_df)}  Test rows: {len(test_df)}",
        "",
    ]
    for name, metrics in results.items():
        report += [f"{name}:", f"MAE: {metrics['mae']:.4f}", f"RMSE: {metrics['rmse']:.4f}", f"MAPE: {metrics['mape']:.4f}%", ""]
    report += [
        "Personalised Random Forest (next month / own 3-month average, blended 50/50 with that average):",
        f"MAE: {ml_metrics['mae']:.4f}",
        f"RMSE: {ml_metrics['rmse']:.4f}",
        f"MAPE: {ml_metrics['mape']:.4f}%",
        "",
        "Machine Learning Model (used by the app):",
        f"MAE: {chosen['mae']:.4f}",
        f"RMSE: {chosen['rmse']:.4f}",
        f"MAPE: {chosen['mape']:.4f}%",
        "",
        "Selected: " + ("personalised Random Forest" if selected == "model" else "moving average (beat the model on held-out months)"),
    ]
    (target_dir / "evaluation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"results": results, "ml_metrics": ml_metrics, "selected": selected, "metadata": metadata,
            "baseline_metrics": baseline, "feature_columns": RELATIVE_FEATURE_COLUMNS}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a monthly expense forecasting model from the real SQLite dataset.")
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite transactions database.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory to save forecasting model artifacts.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        result = train_and_evaluate_model(database_path=args.database_path, model_dir=args.model_dir)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for name, metrics in list(result["results"].items()) + [("Personalised Random Forest", result["ml_metrics"])]:
        print(f"{name:<52} MAE {metrics['mae']:>12,.0f}  MAPE {metrics['mape']:6.1f}%")
    print(f"Selected for the app: {result['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

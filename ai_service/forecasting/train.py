import argparse
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


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

        query = """
            SELECT user_id, transaction_date, amount, transaction_type
            FROM transactions
            WHERE transaction_type = 'Expense'
        """
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


def create_forecast_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Build a no-leakage feature set for next-month expense prediction using only lagged historical information."""
    if monthly_df.empty:
        raise ValueError("Empty monthly expense dataset.")

    monthly_df = monthly_df.copy().sort_values(["user_id", "month"]).reset_index(drop=True)
    monthly_df["month"] = pd.PeriodIndex(monthly_df["month"], freq="M")
    monthly_df["month_number"] = monthly_df["month"].dt.month
    monthly_df["previous_1"] = monthly_df.groupby("user_id")["total_expense"].shift(1)
    monthly_df["previous_2"] = monthly_df.groupby("user_id")["total_expense"].shift(2)
    monthly_df["previous_3"] = monthly_df.groupby("user_id")["total_expense"].shift(3)
    monthly_df["rolling_3"] = monthly_df.groupby("user_id")["total_expense"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    monthly_df["rolling_6"] = monthly_df.groupby("user_id")["total_expense"].transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    monthly_df["spending_change"] = monthly_df.groupby("user_id")["total_expense"].pct_change().fillna(0)

    monthly_df["target_expense"] = monthly_df.groupby("user_id")["total_expense"].shift(-1)
    monthly_df = monthly_df.dropna(subset=["target_expense"]).copy()

    for column in ["previous_1", "previous_2", "previous_3", "rolling_3", "rolling_6", "spending_change"]:
        monthly_df[column] = monthly_df[column].replace([np.inf, -np.inf], np.nan).fillna(0)

    return monthly_df


def moving_average_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Use the average of recent months as a naive baseline."""
    train_lookup = train_df.groupby("user_id")["total_expense"].agg(list).to_dict()
    predictions = []

    for _, row in test_df.iterrows():
        user_id = row["user_id"]
        values = train_lookup.get(user_id, [])
        if not values:
            predictions.append(0.0)
        else:
            predictions.append(float(np.mean(values[-3:])))
    return pd.Series(predictions, index=test_df.index)


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return MAE, RMSE, and MAPE. MAPE handles zero target values safely."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    abs_pct = np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-9, 1.0, y_true))
    mape = np.mean(abs_pct) * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def train_and_evaluate_model(database_path: str | Path | None = None, model_dir: str | Path | None = None):
    """Train a no-leakage forecasting pipeline with chronological split and compare against moving-average baseline."""
    monthly = build_monthly_expense_data(database_path)
    if monthly.empty:
        raise ValueError("Monthly expense dataset is empty after aggregation.")

    dataset = create_forecast_dataset(monthly)
    if dataset.empty:
        raise ValueError("No forecast rows available after feature creation.")

    feature_columns = [
        "previous_1",
        "previous_2",
        "previous_3",
        "rolling_3",
        "rolling_6",
        "month_number",
        "spending_change",
    ]
    X = dataset[feature_columns]
    y = dataset["target_expense"]

    user_ids = dataset["user_id"].unique()
    train_users = user_ids[:-1]
    test_users = user_ids[-1:]

    if len(train_users) == 0 or len(test_users) == 0:
        raise ValueError("Insufficient users to build a chronological train/test split.")

    train_df = dataset[dataset["user_id"].isin(train_users)].copy()
    test_df = dataset[dataset["user_id"].isin(test_users)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty for forecasting.")

    X_train = train_df[feature_columns]
    y_train = train_df["target_expense"]
    X_test = test_df[feature_columns]
    y_test = test_df["target_expense"]

    baseline_pred = moving_average_baseline(train_df, test_df)
    baseline_metrics = evaluate_forecast(y_test, baseline_pred)

    ml_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )
    ml_model.fit(X_train, y_train)
    ml_pred = ml_model.predict(X_test)
    ml_metrics = evaluate_forecast(y_test, ml_pred)

    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ml_model, target_dir / "forecasting_model.pkl")
    joblib.dump(feature_columns, target_dir / "feature_columns.pkl")

    report_lines = [
        "Expense Forecasting Evaluation Report",
        "=" * 40,
        f"Database: {resolved_database_path if 'resolved_database_path' in globals() else database_path}",
        "",
        "Baseline (Moving Average):",
        f"MAE: {baseline_metrics['mae']:.4f}",
        f"RMSE: {baseline_metrics['rmse']:.4f}",
        f"MAPE: {baseline_metrics['mape']:.4f}%",
        "",
        "Machine Learning Model (Random Forest):",
        f"MAE: {ml_metrics['mae']:.4f}",
        f"RMSE: {ml_metrics['rmse']:.4f}",
        f"MAPE: {ml_metrics['mape']:.4f}%",
        "",
        f"Better model: {'ML model' if ml_metrics['mae'] <= baseline_metrics['mae'] else 'Baseline'}",
    ]
    (target_dir / "evaluation_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "baseline_metrics": baseline_metrics,
        "ml_metrics": ml_metrics,
        "feature_columns": feature_columns,
        "train_users": list(train_users),
        "test_users": list(test_users),
        "model_dir": target_dir,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a monthly expense forecasting model from the real SQLite dataset.")
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite transactions database.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory to save forecasting model artifacts.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        result = train_and_evaluate_model(database_path=args.database_path, model_dir=args.model_dir)
        print(f"Baseline MAE: {result['baseline_metrics']['mae']:.4f}")
        print(f"Baseline RMSE: {result['baseline_metrics']['rmse']:.4f}")
        print(f"Baseline MAPE: {result['baseline_metrics']['mape']:.4f}%")
        print(f"ML MAE: {result['ml_metrics']['mae']:.4f}")
        print(f"ML RMSE: {result['ml_metrics']['rmse']:.4f}")
        print(f"ML MAPE: {result['ml_metrics']['mape']:.4f}%")
        print(f"Better model: {'ML model' if result['ml_metrics']['mae'] <= result['baseline_metrics']['mae'] else 'Baseline'}")
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

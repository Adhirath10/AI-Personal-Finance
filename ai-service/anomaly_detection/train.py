import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "anomaly_detection"
MODEL_PATH = MODEL_DIR / "isolation_forest_models.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "personal_finance.db"

REQUIRED_COLUMNS = [
    "transaction_id",
    "user_id",
    "transaction_date",
    "amount",
    "transaction_type",
    "category",
    "subcategory",
    "payment_mode",
    "account_type",
    "is_recurring",
    "balance_after_transaction",
]
LABEL_COLUMNS = ["is_anomaly", "anomaly_type"]
CATEGORICAL_FEATURES = [
    "category",
    "subcategory",
    "payment_mode",
    "account_type",
]
NUMERICAL_FEATURES = [
    "log_amount",
    "balance_after_transaction",
    "is_recurring",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
N_ESTIMATORS = 100
MIN_USER_TRAIN_ROWS = 50


def resolve_database_path(database_path: str | Path | None = None) -> Path:
    """Resolve the database path from CLI, environment, or project default."""
    if database_path:
        path = Path(database_path).expanduser()
    else:
        env_path = os.getenv("FINANCE_DB_PATH")
        path = Path(env_path).expanduser() if env_path else DEFAULT_DATABASE_PATH

    resolved = path if path.is_absolute() else Path.cwd() / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Database not found: {resolved}")
    return resolved


def load_transaction_data(
    database_path: str | Path | None = None,
    include_labels: bool = False,
    expense_only: bool = False,
) -> pd.DataFrame:
    """Load transactions read-only, optionally selecting expenses only."""
    resolved_path = resolve_database_path(database_path)
    connection = sqlite3.connect(
        f"file:{resolved_path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "transactions" not in tables:
            raise ValueError("Table not found: transactions")

        available_columns = {
            row[1] for row in connection.execute('PRAGMA table_info("transactions")')
        }
        required = REQUIRED_COLUMNS + (LABEL_COLUMNS if include_labels else [])
        missing = [column for column in required if column not in available_columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        selected_columns = REQUIRED_COLUMNS + (LABEL_COLUMNS if include_labels else [])
        query = "SELECT " + ", ".join(f'"{column}"' for column in selected_columns)
        query += ' FROM "transactions"'
        if expense_only:
            query += " WHERE transaction_type = 'Expense'"
        data = pd.read_sql_query(query, connection)
    except sqlite3.Error as exc:
        raise sqlite3.DatabaseError(
            f"Unable to read transactions from {resolved_path}: {exc}"
        ) from exc
    finally:
        connection.close()

    if data.empty:
        if expense_only:
            raise ValueError("No expense transactions are available for training.")
        raise ValueError("No transactions are available.")

    data = data.copy()
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")
    invalid_dates = int(data["transaction_date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates} transactions with invalid dates.")
    if data["user_id"].isna().any():
        raise ValueError("Transactions contain missing user_id values.")

    if include_labels:
        data["is_anomaly"] = pd.to_numeric(data["is_anomaly"], errors="coerce")
        if data["is_anomaly"].isna().any():
            raise ValueError("The is_anomaly evaluation labels contain missing or invalid values.")
        data["is_anomaly"] = data["is_anomaly"].astype(int)
        if not data["is_anomaly"].isin([0, 1]).all():
            raise ValueError("The is_anomaly evaluation labels must contain only 0 or 1.")

    return data


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create numeric and categorical features without using labels or identifiers."""
    features = pd.DataFrame(index=data.index)

    amount = pd.to_numeric(data["amount"], errors="coerce")
    features["log_amount"] = np.sign(amount) * np.log1p(amount.abs())
    features["balance_after_transaction"] = pd.to_numeric(
        data["balance_after_transaction"], errors="coerce"
    )
    features["is_recurring"] = pd.to_numeric(data["is_recurring"], errors="coerce")

    dates = pd.to_datetime(data["transaction_date"], errors="coerce")
    day_angle = 2 * np.pi * dates.dt.day_of_week / 7
    month_angle = 2 * np.pi * (dates.dt.month - 1) / 12
    features["day_of_week_sin"] = np.sin(day_angle)
    features["day_of_week_cos"] = np.cos(day_angle)
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)

    for column in CATEGORICAL_FEATURES:
        features[column] = data[column].fillna("__MISSING__").astype(str)

    numeric_features = features[NUMERICAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    for column in NUMERICAL_FEATURES:
        features[column] = numeric_features[column]

    return features[FEATURE_COLUMNS]


def build_pipeline(contamination: float) -> Pipeline:
    """Build a model pipeline whose imputers and encoders are fit on training rows."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                IsolationForest(
                    n_estimators=N_ESTIMATORS,
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def validate_contamination(contamination: float) -> float:
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")
    return contamination


def train_models(
    data: pd.DataFrame,
    contamination: float = 0.02,
) -> dict:
    """Fit a pooled fallback and personal Isolation Forest pipelines."""
    contamination = validate_contamination(contamination)
    if data.empty:
        raise ValueError("No transaction rows are available for training.")

    features = prepare_features(data)
    if len(features) < 2:
        raise ValueError("At least two transactions are required to train Isolation Forest.")

    pooled_model = build_pipeline(contamination)
    pooled_model.fit(features)

    type_models = {}
    for transaction_type, type_rows in data.groupby("transaction_type", sort=True):
        if len(type_rows) >= 2:
            type_model = build_pipeline(contamination)
            type_model.fit(prepare_features(type_rows))
            type_models[str(transaction_type)] = type_model

    user_models = {}
    for (user_id, transaction_type), user_type_rows in data.groupby(
        ["user_id", "transaction_type"],
        sort=True,
    ):
        if len(user_type_rows) < MIN_USER_TRAIN_ROWS:
            continue
        user_model = build_pipeline(contamination)
        user_model.fit(prepare_features(user_type_rows))
        user_models[(str(user_id), str(transaction_type))] = user_model

    return {
        "pooled_model": pooled_model,
        "type_models": type_models,
        "user_models": user_models,
        "metadata": {
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "contamination": contamination,
            "minimum_user_training_rows": MIN_USER_TRAIN_ROWS,
            "feature_columns": FEATURE_COLUMNS,
            "training_rows": int(len(data)),
            "training_users": int(data["user_id"].nunique()),
            "training_transaction_types": sorted(data["transaction_type"].dropna().astype(str).unique()),
            "expense_only": bool(data["transaction_type"].eq("Expense").all()),
            "transaction_type_models": int(len(type_models)),
            "personal_type_models": int(len(user_models)),
            "training_start": pd.to_datetime(data["transaction_date"]).min().date().isoformat(),
            "training_end": pd.to_datetime(data["transaction_date"]).max().date().isoformat(),
        },
    }


def score_transactions(data: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Return original transaction metadata with anomaly flags and scores."""
    metadata = artifacts.get("metadata", {})
    if metadata.get("expense_only"):
        data = data[data["transaction_type"] == "Expense"].copy()
    if data.empty:
        raise ValueError("No transactions in the model's training scope are available to score.")

    user_models = artifacts.get("user_models", {})
    pooled_model = artifacts.get("pooled_model")
    if pooled_model is None:
        raise ValueError("Model artifacts do not contain a pooled fallback model.")

    result = data.copy()
    result["anomaly_flag"] = 0
    result["anomaly_score"] = np.nan

    type_models = artifacts.get("type_models", {})
    groups = data.groupby(["user_id", "transaction_type"], sort=False, dropna=False)
    for (user_id, transaction_type), row_indexes in groups.groups.items():
        model = user_models.get(
            (str(user_id), str(transaction_type)),
            type_models.get(str(transaction_type), pooled_model),
        )
        rows = data.loc[row_indexes]
        features = prepare_features(rows)
        result.loc[row_indexes, "anomaly_flag"] = (model.predict(features) == -1).astype(int)
        result.loc[row_indexes, "anomaly_score"] = -model.decision_function(features)

    result["anomaly_flag"] = result["anomaly_flag"].astype(int)
    return result


def train_and_save(
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    contamination: float = 0.02,
) -> dict:
    """Fit expense-only models and save model/preprocessing pipelines."""
    data = load_transaction_data(database_path, expense_only=True)
    artifacts = train_models(data, contamination)
    predictions = score_transactions(data, artifacts)
    anomaly_count = int(predictions["anomaly_flag"].sum())
    anomaly_rate = anomaly_count / len(predictions)
    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / MODEL_PATH.name
    metadata_path = target_dir / METADATA_PATH.name
    joblib.dump(artifacts, model_path, compress=3)
    metadata_path.write_text(
        json.dumps(artifacts["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "artifacts": artifacts,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train per-user Isolation Forest models on expense transactions."
    )
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite database.")
    parser.add_argument(
        "--model-dir",
        "--output-dir",
        dest="model_dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory for model and preprocessing artifacts.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,
        help="Expected anomaly fraction, from greater than 0 through 0.5 (default: 0.02).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        result = train_and_save(
            database_path=args.database_path,
            model_dir=args.model_dir,
            contamination=args.contamination,
        )
        metadata = result["artifacts"]["metadata"]
        print("Isolation Forest training completed.")
        print(f"Training rows (expenses): {metadata['training_rows']:,}")
        print(f"Users: {metadata['training_users']:,}")
        print(f"Transaction-type models: {metadata['transaction_type_models']:,}")
        print(f"Personal type models: {metadata['personal_type_models']:,}")
        print(f"Contamination: {metadata['contamination']:.3f}")
        print("Selected features:")
        for feature in metadata["feature_columns"]:
            print(f"  - {feature}")
        print(
            f"Training-set anomalies: {result['anomaly_count']:,} "
            f"({result['anomaly_rate']:.2%})"
        )
        print(f"Model and preprocessing pipelines: {result['model_path']}")
        print(f"Metadata: {result['metadata_path']}")
        print("The source database was opened read-only; no transaction rows were changed.")
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error, OSError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

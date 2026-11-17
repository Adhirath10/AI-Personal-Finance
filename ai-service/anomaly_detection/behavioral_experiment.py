import argparse
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import chronological_split, load_evaluation_data
from train import (
    CATEGORICAL_FEATURES,
    MIN_USER_TRAIN_ROWS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    prepare_features,
    resolve_database_path,
    score_transactions,
    train_models,
    validate_contamination,
)


MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "anomaly_detection_behavioral_experiment"
)
BEHAVIORAL_NUMERICAL_FEATURES = [
    "user_history_mean",
    "user_history_std",
    "user_history_count",
    "user_amount_deviation",
    "user_amount_zscore",
    "category_history_mean",
    "category_history_std",
    "category_history_count",
    "category_history_share",
    "merchant_history_count",
    "merchant_familiar",
    "user_transactions_last_30_days",
    "category_transactions_last_30_days",
]
EXPERIMENT_NUMERICAL_FEATURES = (
    NUMERICAL_FEATURES + BEHAVIORAL_NUMERICAL_FEATURES
)
EXPERIMENT_FEATURE_COLUMNS = EXPERIMENT_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
FEATURE_ENGINEERING_VERSION = "prior-only-v1"


def _normalized_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().casefold()
    return normalized or None


def _history_statistics(total: float, total_squared: float, count: int) -> tuple[float, float]:
    if count == 0:
        return np.nan, np.nan
    mean = total / count
    if count < 2:
        return mean, np.nan
    variance = max(0.0, (total_squared - total * total / count) / (count - 1))
    return mean, math.sqrt(variance)


def engineer_behavioral_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create per-user features using transactions strictly before each timestamp.

    All transactions sharing a user and timestamp receive the same prior state.
    Rows at that timestamp are added to history only after their features are
    computed, so tied timestamps cannot leak into one another.
    """
    required = {
        "user_id",
        "transaction_date",
        "amount",
        "category",
        "merchant",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing behavioral feature columns: {sorted(missing)}")
    if data["user_id"].isna().any():
        raise ValueError("Behavioral features require non-missing user_id values.")

    dates = pd.to_datetime(data["transaction_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Behavioral features require valid transaction dates.")

    ordered = data.assign(
        behavior_date=dates,
        behavior_amount=pd.to_numeric(data["amount"], errors="coerce"),
        behavior_row_id=np.arange(len(data)),
    ).sort_values(
        ["user_id", "behavior_date"],
        kind="mergesort",
    )

    features_by_position: list[dict[str, float] | None] = [None] * len(data)
    user_states: dict[str, dict] = {}

    for (user_id, timestamp), timestamp_rows in ordered.groupby(
        ["user_id", "behavior_date"],
        sort=False,
        dropna=False,
    ):
        user_key = str(user_id)
        state = user_states.setdefault(
            user_key,
            {
                "count": 0,
                "sum": 0.0,
                "sum_squared": 0.0,
                "categories": defaultdict(lambda: [0, 0.0, 0.0]),
                "merchants": Counter(),
                "recent": deque(),
                "recent_category_counts": Counter(),
            },
        )

        recent_cutoff = timestamp - pd.Timedelta(days=30)
        while state["recent"] and state["recent"][0][0] < recent_cutoff:
            old_timestamp, old_category = state["recent"].popleft()
            state["recent_category_counts"][old_category] -= 1
            if state["recent_category_counts"][old_category] == 0:
                del state["recent_category_counts"][old_category]

        user_mean, user_std = _history_statistics(
            state["sum"], state["sum_squared"], state["count"]
        )
        for row in timestamp_rows.itertuples():
            amount = float(row.behavior_amount) if pd.notna(row.behavior_amount) else np.nan
            category_key = _normalized_key(row.category)
            merchant_key = _normalized_key(row.merchant)
            category_state = (
                state["categories"].get(category_key, [0, 0.0, 0.0])
                if category_key is not None
                else [0, 0.0, 0.0]
            )
            category_count, category_sum, category_sum_squared = category_state
            category_mean, category_std = _history_statistics(
                category_sum, category_sum_squared, category_count
            )
            category_share = (
                category_count / state["count"] if state["count"] else np.nan
            )
            merchant_count = (
                state["merchants"][merchant_key] if merchant_key is not None else 0
            )
            features_by_position[int(row.behavior_row_id)] = {
                "user_history_mean": user_mean,
                "user_history_std": user_std,
                "user_history_count": float(state["count"]),
                "user_amount_deviation": amount - user_mean
                if math.isfinite(amount) and pd.notna(user_mean)
                else np.nan,
                "user_amount_zscore": (amount - user_mean) / user_std
                if math.isfinite(amount)
                and pd.notna(user_mean)
                and pd.notna(user_std)
                and user_std > 0
                else np.nan,
                "category_history_mean": category_mean,
                "category_history_std": category_std,
                "category_history_count": float(category_count),
                "category_history_share": category_share,
                "merchant_history_count": float(merchant_count),
                "merchant_familiar": float(merchant_count > 0),
                "user_transactions_last_30_days": float(len(state["recent"])),
                "category_transactions_last_30_days": float(
                    state["recent_category_counts"].get(category_key, 0)
                    if category_key is not None
                    else 0
                ),
            }

        for row in timestamp_rows.itertuples():
            amount = float(row.behavior_amount) if pd.notna(row.behavior_amount) else np.nan
            category_key = _normalized_key(row.category)
            merchant_key = _normalized_key(row.merchant)
            if math.isfinite(amount):
                state["count"] += 1
                state["sum"] += amount
                state["sum_squared"] += amount * amount
                if category_key is not None:
                    category_state = state["categories"][category_key]
                    category_state[0] += 1
                    category_state[1] += amount
                    category_state[2] += amount * amount
            if merchant_key is not None:
                state["merchants"][merchant_key] += 1
            state["recent"].append((timestamp, category_key))
            if category_key is not None:
                state["recent_category_counts"][category_key] += 1

    if any(row is None for row in features_by_position):
        raise RuntimeError("Behavioral feature generation did not produce every input row.")
    result = pd.DataFrame(features_by_position, index=data.index)
    return result[BEHAVIORAL_NUMERICAL_FEATURES].astype(float)


def prepare_experiment_features(data: pd.DataFrame) -> pd.DataFrame:
    base_features = prepare_features(data)
    behavioral_features = engineer_behavioral_features(data)
    return base_features.join(behavioral_features)


def load_behavioral_evaluation_data(
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    data = load_evaluation_data(database_path)
    resolved_path = resolve_database_path(database_path)
    connection = sqlite3.connect(
        f"file:{resolved_path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        columns = {
            row[1] for row in connection.execute('PRAGMA table_info("transactions")')
        }
        if "merchant" not in columns:
            raise ValueError("Missing required behavioral feature column: merchant")
        merchants = pd.read_sql_query(
            'SELECT "transaction_id", "merchant" FROM "transactions" '
            'WHERE "transaction_type" = ?',
            connection,
            params=("Expense",),
        )
    except sqlite3.Error as exc:
        raise sqlite3.DatabaseError(
            f"Unable to read merchant feature values from {resolved_path}: {exc}"
        ) from exc
    finally:
        connection.close()

    if merchants["transaction_id"].duplicated().any():
        raise ValueError("Expense transaction IDs must be unique to join merchant values.")
    data = data.merge(
        merchants,
        on="transaction_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if data["merchant"].isna().all():
        raise ValueError("No merchant values were loaded for expense transactions.")
    return data


def build_experiment_pipeline(contamination: float) -> Pipeline:
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
                EXPERIMENT_NUMERICAL_FEATURES,
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
                    n_estimators=100,
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_experiment_models(
    data: pd.DataFrame,
    contamination: float,
    feature_frame: pd.DataFrame | None = None,
) -> dict:
    contamination = validate_contamination(contamination)
    if data.empty:
        raise ValueError("No transaction rows are available for training.")

    features = (
        prepare_experiment_features(data)
        if feature_frame is None
        else feature_frame.loc[data.index]
    )
    if len(features) < 2:
        raise ValueError("At least two transactions are required to train Isolation Forest.")

    pooled_model = build_experiment_pipeline(contamination).fit(features)
    type_models = {}
    for transaction_type, rows in data.groupby("transaction_type", sort=True):
        if len(rows) >= 2:
            type_models[str(transaction_type)] = build_experiment_pipeline(
                contamination
            ).fit(features.loc[rows.index])

    user_models = {}
    for (user_id, transaction_type), rows in data.groupby(
        ["user_id", "transaction_type"], sort=True
    ):
        if len(rows) >= MIN_USER_TRAIN_ROWS:
            user_models[(str(user_id), str(transaction_type))] = (
                build_experiment_pipeline(contamination).fit(features.loc[rows.index])
            )

    return {
        "pooled_model": pooled_model,
        "type_models": type_models,
        "user_models": user_models,
        "metadata": {
            "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
            "feature_columns": EXPERIMENT_FEATURE_COLUMNS,
            "contamination": contamination,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(data)),
            "training_users": int(data["user_id"].nunique()),
            "minimum_user_training_rows": MIN_USER_TRAIN_ROWS,
            "personal_type_models": int(len(user_models)),
            "training_start": pd.to_datetime(data["transaction_date"]).min().date().isoformat(),
            "training_end": pd.to_datetime(data["transaction_date"]).max().date().isoformat(),
            "label_source_reliability": "unverified",
        },
    }


def score_experiment_transactions(
    data: pd.DataFrame,
    artifacts: dict,
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("No transactions are available to score.")
    features = (
        prepare_experiment_features(data)
        if feature_frame is None
        else feature_frame.loc[data.index]
    )
    user_models = artifacts["user_models"]
    type_models = artifacts["type_models"]
    pooled_model = artifacts["pooled_model"]
    predictions = np.zeros(len(data), dtype=int)
    scores = np.empty(len(data), dtype=float)

    for (user_id, transaction_type), indexes in data.groupby(
        ["user_id", "transaction_type"], sort=False
    ).groups.items():
        model = user_models.get(
            (str(user_id), str(transaction_type)),
            type_models.get(str(transaction_type), pooled_model),
        )
        group_features = features.loc[indexes]
        positions = data.index.get_indexer(indexes)
        predictions[positions] = (model.predict(group_features) == -1).astype(int)
        scores[positions] = -model.decision_function(group_features)

    result = data.copy()
    result["anomaly_flag"] = predictions
    result["anomaly_score"] = scores
    return result


def _metrics(testing: pd.DataFrame, predictions: pd.Series) -> dict:
    valid = testing["label_valid"]
    truth = testing.loc[valid, "is_anomaly"].astype(int).to_numpy()
    predicted = predictions.loc[valid].astype(int).to_numpy()
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    return {
        "evaluated_rows": int(len(truth)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "alert_rate": float(np.mean(predicted)) if len(predicted) else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_experiment(
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    contamination: float = 0.02,
    test_fraction: float = 0.2,
) -> dict:
    contamination = validate_contamination(contamination)
    data = load_behavioral_evaluation_data(database_path)
    training, testing, cutoff = chronological_split(data, test_fraction)
    all_features = prepare_experiment_features(data)
    excluded = ["is_anomaly", "anomaly_type", "label_valid"]
    training_without_labels = training.drop(columns=excluded, errors="ignore")

    baseline_artifacts = train_models(training_without_labels, contamination)
    baseline_predictions = score_transactions(testing, baseline_artifacts)
    baseline_metrics = _metrics(testing, baseline_predictions["anomaly_flag"])

    experiment_artifacts = train_experiment_models(
        training_without_labels,
        contamination,
        feature_frame=all_features,
    )
    experiment_predictions = score_experiment_transactions(
        testing,
        experiment_artifacts,
        feature_frame=all_features,
    )
    experiment_metrics = _metrics(testing, experiment_predictions["anomaly_flag"])

    target_dir = Path(model_dir).expanduser() if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "behavioral_isolation_forest_models.joblib"
    metadata_path = target_dir / "model_metadata.json"
    report_path = target_dir / "evaluation_report.txt"

    experiment_artifacts["metadata"].update(
        {
            "test_fraction": test_fraction,
            "test_cutoff_date": cutoff.date().isoformat(),
            "test_rows": int(len(testing)),
            "evaluated_rows": experiment_metrics["evaluated_rows"],
        }
    )
    joblib.dump(experiment_artifacts, model_path, compress=3)
    metadata_path.write_text(
        json.dumps(experiment_artifacts["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )

    def format_metrics(name: str, metrics: dict) -> list[str]:
        matrix = metrics["confusion_matrix"]
        return [
            f"{name}:",
            f"  Precision: {metrics['precision']:.4f}",
            f"  Recall: {metrics['recall']:.4f}",
            f"  F1-score: {metrics['f1']:.4f}",
            f"  Alert rate: {metrics['alert_rate']:.4%}",
            "  Confusion matrix (rows=stored [normal, anomaly], columns=predicted [normal, anomaly]):",
            f"    {matrix}",
        ]

    report_lines = [
        "Behavioral Feature Isolation Forest Experiment",
        "=" * 48,
        "Stored anomaly-label provenance: UNVERIFIED.",
        "Metrics measure agreement with stored annotations only; they are not",
        "validated anomaly-detection or fraud-detection performance.",
        "",
        "Validation: latest 20% of unique transaction dates held out.",
        f"Training rows: {len(training):,}",
        f"Test rows: {len(testing):,}",
        f"Test window starts: {cutoff.date().isoformat()}",
        f"Contamination: {contamination:.4f}",
        "Stored labels and anomaly types are excluded from all model inputs.",
        "Features are computed from strictly earlier transactions for each user.",
        "Transactions with the same user and timestamp share the same prior state.",
        "Test-period features may use earlier transaction attributes from the",
        "training and test periods, as in chronological online scoring; no",
        "future transaction attributes or labels are used.",
        "Numeric imputation/scaling and categorical encoding are fit on training",
        "features only. Users without a personal model use type/pooled fallback.",
        "",
        *format_metrics("Baseline model", baseline_metrics),
        "",
        *format_metrics("Behavioral feature model", experiment_metrics),
        "",
        "Behavioral features include prior user/category amount mean and sample",
        "standard deviation, prior counts/frequencies, amount deviation and",
        "z-score from user history, merchant prior count/familiarity, and counts",
        "in the preceding 30 days. Early-history statistics are missing and",
        "handled by training-fitted median imputation; count/familiarity features",
        "indicate available history.",
        "",
        f"Experimental model artifact: {model_path}",
        f"Experimental metadata: {metadata_path}",
    ]
    report = "\n".join(report_lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Experimental artifacts saved under: {target_dir}")
    return {
        "cutoff_date": cutoff.date().isoformat(),
        "baseline": baseline_metrics,
        "behavioral": experiment_metrics,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "report_path": report_path,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare baseline and prior-history Isolation Forest features."
    )
    parser.add_argument("--database-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        evaluate_experiment(
            database_path=args.database_path,
            model_dir=args.model_dir,
            contamination=args.contamination,
            test_fraction=args.test_fraction,
        )
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error, OSError) as exc:
        print(f"Behavioral experiment error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

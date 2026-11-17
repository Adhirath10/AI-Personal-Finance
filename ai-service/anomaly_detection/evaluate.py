import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import (
    LABEL_COLUMNS,
    REQUIRED_COLUMNS,
    resolve_database_path,
    score_transactions,
    train_models,
    validate_contamination,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "anomaly_detection"


def load_evaluation_data(database_path: str | Path | None = None) -> pd.DataFrame:
    """Load expense rows and their evaluation labels without requiring complete labels."""
    resolved_path = resolve_database_path(database_path)
    connection = sqlite3.connect(f"file:{resolved_path.as_posix()}?mode=ro", uri=True)
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
        required_columns = REQUIRED_COLUMNS + [LABEL_COLUMNS[0]]
        missing_columns = [
            column for column in required_columns if column not in available_columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required evaluation columns: {missing_columns}")

        selected_columns = REQUIRED_COLUMNS + ["is_anomaly"]
        has_anomaly_type = "anomaly_type" in available_columns
        if has_anomaly_type:
            selected_columns += ["anomaly_type"]
        query = "SELECT " + ", ".join(f'"{column}"' for column in selected_columns)
        query += ' FROM "transactions" WHERE "transaction_type" = ?'
        data = pd.read_sql_query(query, connection, params=("Expense",))
    except sqlite3.Error as exc:
        raise sqlite3.DatabaseError(
            f"Unable to read evaluation data from {resolved_path}: {exc}"
        ) from exc
    finally:
        connection.close()

    if data.empty:
        raise ValueError("No expense transactions are available for evaluation.")

    data = data.copy()
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")
    invalid_dates = int(data["transaction_date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Found {invalid_dates} expense rows with invalid transaction dates.")

    labels = pd.to_numeric(data["is_anomaly"], errors="coerce")
    valid_labels = labels.isin([0, 1])
    data["is_anomaly"] = labels.where(valid_labels)
    data["label_valid"] = valid_labels
    if has_anomaly_type:
        data["anomaly_type"] = data["anomaly_type"].replace(r"^\s*$", np.nan, regex=True)
    else:
        data["anomaly_type"] = np.nan
    return data


def chronological_split(
    data: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Keep the latest dates for evaluation and earlier dates for fitting."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be greater than 0 and less than 1.")

    dates = pd.Index(data["transaction_date"].dropna().unique()).sort_values()
    if len(dates) < 5:
        raise ValueError("At least five distinct transaction dates are required for evaluation.")

    split_index = int(len(dates) * (1 - test_fraction))
    split_index = min(max(split_index, 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[split_index])
    training = data[data["transaction_date"] < cutoff].copy()
    testing = data[data["transaction_date"] >= cutoff].copy()
    if training.empty or testing.empty:
        raise ValueError("Chronological train/test split produced an empty partition.")
    return training, testing, cutoff


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str = "Isolation Forest vs Stored Anomaly Labels",
) -> None:
    """Save a labeled confusion matrix visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis, label="Transaction count")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        xlabel="Predicted label",
        ylabel="Stored label",
        title=title,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_model(
    database_path: str | Path | None = None,
    output_path: str | Path | None = None,
    confusion_matrix_path: str | Path | None = None,
    contamination: float = 0.02,
    test_fraction: float = 0.2,
) -> dict:
    """Train on earlier expenses and evaluate only on later labeled expenses."""
    contamination = validate_contamination(contamination)
    data = load_evaluation_data(database_path)
    training, testing, cutoff = chronological_split(data, test_fraction)

    training_features = training.drop(
        columns=["is_anomaly", "anomaly_type", "label_valid"],
        errors="ignore",
    )
    artifacts = train_models(training_features, contamination)
    predictions = score_transactions(testing, artifacts)

    valid_test_labels = testing["label_valid"]
    evaluated_rows = testing.loc[valid_test_labels].copy()
    evaluated_predictions = predictions.loc[valid_test_labels]

    target_path = (
        Path(output_path).expanduser()
        if output_path
        else MODEL_DIR / "evaluation_report.txt"
    )
    matrix_path = (
        Path(confusion_matrix_path).expanduser()
        if confusion_matrix_path
        else target_path.parent / "confusion_matrix.png"
    )

    metrics: dict = {
        "training_rows": int(len(training)),
        "test_rows": int(len(testing)),
        "evaluated_rows": int(len(evaluated_rows)),
        "unlabeled_test_rows": int((~valid_test_labels).sum()),
        "cutoff_date": cutoff.date().isoformat(),
        "contamination": contamination,
        "label_source_reliability": "unverified",
    }

    if evaluated_rows.empty:
        matrix = np.zeros((2, 2), dtype=int)
        classification_text = "Unavailable: no valid is_anomaly labels in the test window."
        metric_lines = [
            "Accuracy: unavailable",
            "Precision: unavailable",
            "Recall: unavailable",
            "F1-score: unavailable",
        ]
        metrics.update(
            {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "confusion_matrix": matrix.tolist(),
            }
        )
    else:
        truth = evaluated_rows["is_anomaly"].astype(int).to_numpy()
        predicted = evaluated_predictions["anomaly_flag"].astype(int).to_numpy()
        matrix = confusion_matrix(truth, predicted, labels=[0, 1])
        classification_text = classification_report(
            truth,
            predicted,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
        metrics.update(
            {
                "accuracy": float(accuracy_score(truth, predicted)),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "confusion_matrix": matrix.tolist(),
                "label_anomaly_rate": float(np.mean(truth)),
                "predicted_anomaly_rate": float(np.mean(predicted)),
            }
        )
        metric_lines = [
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Precision: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"F1-score: {metrics['f1']:.4f}",
        ]

    label_anomalies_missing_type = int(
        (data["is_anomaly"].eq(1) & data["anomaly_type"].isna()).sum()
    )
    normal_rows_with_type = int(
        (data["is_anomaly"].eq(0) & data["anomaly_type"].notna()).sum()
    )
    metrics["positive_labels_missing_anomaly_type"] = label_anomalies_missing_type
    metrics["normal_labels_with_anomaly_type"] = normal_rows_with_type

    save_confusion_matrix(matrix, matrix_path)

    report_lines = [
        "Isolation Forest Anomaly Detection Evaluation",
        "=" * 48,
        "Validation: chronological holdout by transaction date.",
        "Training rows are earlier than the test window; test rows were not used",
        "to fit models or preprocessing.",
        "Scope: expense transactions only.",
        "Existing is_anomaly and anomaly_type fields are excluded from training",
        "features and used only to describe/score the held-out results.",
        "",
        "Stored-label reliability: UNVERIFIED.",
        "No independent ground-truth provenance for the stored labels was found.",
        "The metrics below measure agreement with stored annotations only;",
        "they are not validated anomaly-detection performance and are not",
        "fraud-detection metrics.",
        "",
        f"Contamination: {contamination:.4f}",
        f"Test window starts: {metrics['cutoff_date']}",
        f"Training expense rows: {metrics['training_rows']:,}",
        f"Test expense rows: {metrics['test_rows']:,}",
        f"Test rows with valid labels: {metrics['evaluated_rows']:,}",
        f"Test rows with missing/invalid labels: {metrics['unlabeled_test_rows']:,}",
        f"Positive labels missing anomaly_type (full expense dataset): {label_anomalies_missing_type:,}",
        f"Normal labels with anomaly_type (full expense dataset): {normal_rows_with_type:,}",
        "",
        *metric_lines,
        "",
        "Confusion Matrix (rows = stored labels; columns = predictions):",
        "                 Predicted Normal   Predicted Anomaly",
        f"Stored Normal    {matrix[0, 0]:16,d}   {matrix[0, 1]:17,d}",
        f"Stored Anomaly   {matrix[1, 0]:16,d}   {matrix[1, 1]:17,d}",
        "",
        "Classification Report:",
        classification_text,
        f"Confusion matrix image: {matrix_path}",
        "",
        "Isolation Forest identifies statistical outliers under the selected",
        "features and contamination threshold. It does not determine whether",
        "a transaction is fraudulent. Fraud detection requires validated fraud",
        "labels and a separate supervised evaluation.",
    ]
    report = "\n".join(report_lines) + "\n"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Evaluation report saved to: {target_path}")
    print(f"Confusion matrix image saved to: {matrix_path}")
    return metrics


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate Isolation Forest against stored labels using a chronological holdout."
    )
    parser.add_argument("--database-path", type=str, default=None, help="Path to the existing SQLite database.")
    parser.add_argument("--output", type=str, default=None, help="Path for the evaluation text report.")
    parser.add_argument("--confusion-matrix", type=str, default=None, help="Path for the confusion-matrix PNG.")
    parser.add_argument("--contamination", type=float, default=0.02, help="Isolation Forest contamination (default: 0.02).")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of latest unique dates held out (default: 0.2).")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        evaluate_model(
            database_path=args.database_path,
            output_path=args.output,
            confusion_matrix_path=args.confusion_matrix,
            contamination=args.contamination,
            test_fraction=args.test_fraction,
        )
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error, OSError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

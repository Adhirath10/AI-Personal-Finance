import argparse
import sqlite3
import sys
from pathlib import Path

import joblib

from train import MODEL_DIR, load_transaction_data, score_transactions


MODEL_FILENAME = "isolation_forest_models.joblib"


def load_model_artifacts(model_dir: str | Path | None = None) -> dict:
    """Load the saved Isolation Forest models and fitted preprocessing pipelines."""
    target_dir = Path(model_dir).expanduser() if model_dir else MODEL_DIR
    model_path = target_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Anomaly model not found at {model_path}. Run train.py first."
        )

    try:
        artifacts = joblib.load(model_path)
    except (OSError, EOFError, ValueError, ImportError) as exc:
        raise ValueError(f"Unable to load model artifacts from {model_path}: {exc}") from exc

    required_keys = {"pooled_model", "type_models", "user_models", "metadata"}
    if not isinstance(artifacts, dict) or not required_keys.issubset(artifacts):
        raise ValueError(f"Invalid anomaly model artifacts in {model_path}.")
    return artifacts


def predict_transactions(
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    user_id: str | None = None,
    transaction_id: str | None = None,
):
    """Predict one transaction or all supported expenses belonging to a user."""
    if (user_id is None) == (transaction_id is None):
        raise ValueError("Provide exactly one of user_id or transaction_id.")

    requested_user_id = str(user_id).strip() if user_id is not None else None
    requested_transaction_id = (
        str(transaction_id).strip() if transaction_id is not None else None
    )
    if requested_user_id == "":
        raise ValueError("User ID cannot be empty.")
    if requested_transaction_id == "":
        raise ValueError("Transaction ID cannot be empty.")

    transactions = load_transaction_data(database_path)
    if requested_transaction_id is not None:
        transactions = transactions[
            transactions["transaction_id"] == requested_transaction_id
        ].copy()
        if transactions.empty:
            raise ValueError(f"Transaction ID not found: {requested_transaction_id}")
        if transactions.iloc[0]["transaction_type"] != "Expense":
            raise ValueError(
                f"Transaction {requested_transaction_id} is not an expense. "
                "The saved model was trained on expenses only."
            )
    else:
        transactions = transactions[transactions["user_id"] == requested_user_id].copy()
        if transactions.empty:
            raise ValueError(f"No transactions found for user ID: {requested_user_id}")
        transactions = transactions[transactions["transaction_type"] == "Expense"].copy()
        if transactions.empty:
            raise ValueError(
                f"No expense transactions found for user ID: {requested_user_id}"
            )

    artifacts = load_model_artifacts(model_dir)
    return score_transactions(transactions, artifacts)


def print_predictions(predictions) -> None:
    """Print each transaction prediction in the requested display format."""
    for row in predictions.itertuples(index=False):
        prediction = "Potential Anomaly" if row.anomaly_flag == 1 else "Normal"
        print(f"Transaction ID: {row.transaction_id}")
        print(f"User ID: {row.user_id}")
        print(f"Amount: ₹{float(row.amount):,.2f}")
        print(f"Category: {row.category}")
        print(f"Prediction: {prediction}")
        print(f"Anomaly Score: {float(row.anomaly_score):.6f}")
        print()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict whether an expense transaction is anomalous."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--user-id",
        type=str,
        help="Predict all expense transactions for this user.",
    )
    selection.add_argument(
        "--transaction-id",
        type=str,
        help="Predict a single transaction by its ID.",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default=None,
        help="Path to the existing SQLite database.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory containing the saved model artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        predictions = predict_transactions(
            database_path=args.database_path,
            model_dir=args.model_dir,
            user_id=args.user_id,
            transaction_id=args.transaction_id,
        )
        print_predictions(predictions)
        print(
            f"Scored {len(predictions):,} expense transaction(s). "
            "The database was opened read-only."
        )
        return 0
    except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

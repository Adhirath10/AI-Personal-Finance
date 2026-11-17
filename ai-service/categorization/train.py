import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ["description", "merchant", "category"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "categorization"
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "personal_finance.db"


def resolve_database_path(database_path: str | None = None) -> Path:
    """Resolve the SQLite database path from environment, CLI arg, or default project path."""
    if database_path:
        path = Path(database_path).expanduser()
    else:
        env_path = os.getenv("FINANCE_DB_PATH")
        path = Path(env_path).expanduser() if env_path else DEFAULT_DATABASE_PATH

    resolved = path if path.is_absolute() else (Path.cwd() / path)
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved}")
    return resolved


def prepare_text_feature(description, merchant):
    """Combine and clean description and merchant text; never uses category or target labels."""
    desc = "" if description is None else str(description)
    merch = "" if merchant is None else str(merchant)

    combined = f"{desc} {merch}".strip()
    if not combined:
        return "transaction"

    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", combined.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "transaction"


def load_transaction_data(database_path: str | Path | None = None) -> pd.DataFrame:
    """Load relevant transaction rows for categorization without leaking target-derived fields."""
    resolved_db_path = resolve_database_path(str(database_path) if database_path is not None else None)

    try:
        connection = sqlite3.connect(f"file:{resolved_db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise sqlite3.OperationalError(f"Unable to open database: {resolved_db_path}") from exc

    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {table_name for (table_name,) in tables}

        if "transactions" not in table_names:
            raise ValueError("Table not found: transactions")

        columns = connection.execute('PRAGMA table_info("transactions")').fetchall()
        available_columns = [column[1] for column in columns]
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in available_columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        query = "SELECT description, merchant, category FROM transactions"
        df = pd.read_sql_query(query, connection)

        if df.empty:
            raise ValueError("Empty dataset: no transactions available for training.")

        df = df.copy()
        df["description"] = df["description"].fillna("").astype(str).str.strip()
        df["merchant"] = df["merchant"].fillna("").astype(str).str.strip()
        df["category"] = df["category"].fillna("").astype(str).str.strip()
        df = df[df["category"] != ""].copy()

        if df.empty:
            raise ValueError("Empty dataset after removing rows with missing category labels.")

        df["text"] = df.apply(lambda row: prepare_text_feature(row["description"], row["merchant"]), axis=1)
        return df[["description", "merchant", "category", "text"]]
    finally:
        connection.close()


def save_confusion_matrix(y_true, y_pred, labels, output_path: Path):
    """Store a confusion matrix image for later model analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    plt.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Transaction Category Confusion Matrix")
    plt.colorbar(label="Count")
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            plt.text(col_index, row_index, confusion[row_index, col_index], ha="center", va="center", color="white" if confusion[row_index, col_index] > confusion.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_predictions(y_true, y_pred, labels=None, output_path: Path | None = None):
    """Compute a full set of metrics and optionally save the report file."""
    if labels is None:
        labels = sorted(set(y_true).union(set(y_pred)))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(
                [
                    "Transaction Categorization Evaluation Report",
                    "=" * 48,
                    f"Accuracy: {accuracy:.4f}",
                    f"Weighted Precision: {precision:.4f}",
                    f"Weighted Recall: {recall:.4f}",
                    f"Weighted F1-score: {f1:.4f}",
                    f"Macro Precision: {macro_precision:.4f}",
                    f"Macro Recall: {macro_recall:.4f}",
                    f"Macro F1-score: {macro_f1:.4f}",
                    "",
                    "Classification Report:",
                    report,
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "classification_report": report,
    }


def train_model(database_path: str | Path | None = None, model_dir: str | Path | None = None):
    """Train the baseline TF-IDF + Logistic Regression categorization model."""
    df = load_transaction_data(database_path)
    if df.empty:
        raise ValueError("Empty dataset after preprocessing. Nothing to train.")

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vec, y_train)
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)

    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, target_dir / "tfidf_vectorizer.pkl")
    joblib.dump(classifier, target_dir / "category_classifier.pkl")

    labels = sorted(set(y_train).union(set(y_test)))
    metrics = evaluate_predictions(y_test, y_pred, labels=labels, output_path=target_dir / "evaluation_report.txt")
    save_confusion_matrix(y_test, y_pred, labels, target_dir / "confusion_matrix.png")

    return {
        "metrics": metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_dir": target_dir,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a transaction categorization model from the project SQLite database.")
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite database file. Defaults to the project data/database/personal_finance.db file.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory where model artifacts and reports are saved.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        result = train_model(database_path=args.database_path, model_dir=args.model_dir)
        print(f"Training completed successfully. Training rows: {result['train_size']}, Test rows: {result['test_size']}")
        print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"Weighted F1: {result['metrics']['weighted_f1']:.4f}")
        print(f"Model files saved to: {result['model_dir']}")
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

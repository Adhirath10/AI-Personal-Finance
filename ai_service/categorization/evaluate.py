import argparse
import sqlite3
import sys
from pathlib import Path

import joblib

from ai_service.categorization.train import (
    MODEL_DIR,
    evaluate_predictions,
    load_transaction_data,
    prepare_text_feature,
    save_confusion_matrix,
    resolve_database_path,
)


def load_saved_model(model_dir: str | Path | None = None):
    """Load the trained TF-IDF vectorizer and classifier from disk."""
    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    vectorizer_path = target_dir / "tfidf_vectorizer.pkl"
    classifier_path = target_dir / "category_classifier.pkl"

    if not vectorizer_path.exists() or not classifier_path.exists():
        raise FileNotFoundError(f"Model files not found in {target_dir}. Train the model first.")

    return joblib.load(vectorizer_path), joblib.load(classifier_path)


def evaluate_saved_model(database_path: str | Path | None = None, model_dir: str | Path | None = None):
    """Evaluate the trained model on the test split from the real transaction data."""
    df = load_transaction_data(database_path)
    if df.empty:
        raise ValueError("Empty dataset: no transaction rows available for evaluation.")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    vectorizer, classifier = load_saved_model(model_dir)
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)

    labels = sorted(set(y_train).union(set(y_test)))
    metrics = evaluate_predictions(y_test, y_pred, labels=labels, output_path=Path(model_dir) / "evaluation_report.txt")
    save_confusion_matrix(y_test, y_pred, labels, Path(model_dir) / "confusion_matrix.png")

    print("Transaction Categorization Evaluation")
    print("=" * 42)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    return metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the saved transaction categorization model.")
    parser.add_argument("--database-path", type=str, default=None, help="Path to the SQLite database file.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory containing trained model files.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        evaluate_saved_model(args.database_path, args.model_dir)
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

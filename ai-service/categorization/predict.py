import argparse
import sys
from pathlib import Path

import joblib

from train import MODEL_DIR, prepare_text_feature


def load_model_artifacts(model_dir: str | Path | None = None):
    """Load the saved TF-IDF vectorizer and logistic regression classifier."""
    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    vectorizer_path = target_dir / "tfidf_vectorizer.pkl"
    classifier_path = target_dir / "category_classifier.pkl"

    if not vectorizer_path.exists() or not classifier_path.exists():
        raise FileNotFoundError(f"Model files not found in {target_dir}. Train the model first.")

    return joblib.load(vectorizer_path), joblib.load(classifier_path)


def predict_category(description: str, merchant: str, model_dir: str | Path | None = None):
    """Predict the transaction category from description and merchant text."""
    if description is None and merchant is None:
        raise ValueError("Invalid input: both description and merchant are empty.")

    cleaned_desc = "" if description is None else str(description).strip()
    cleaned_merchant = "" if merchant is None else str(merchant).strip()

    if not cleaned_desc and not cleaned_merchant:
        raise ValueError("Empty description and merchant. Please provide at least one text field.")

    text = prepare_text_feature(cleaned_desc, cleaned_merchant)
    vectorizer, classifier = load_model_artifacts(model_dir)
    features = vectorizer.transform([text])
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    confidence = float(probabilities.max())
    return prediction, confidence


def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict a transaction category from description and merchant text.")
    parser.add_argument("description", nargs="?", default="", help="Transaction description text.")
    parser.add_argument("merchant", nargs="?", default="", help="Merchant or company name.")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Directory containing the saved model artifacts.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        category, confidence = predict_category(args.description, args.merchant, args.model_dir)
        print(f"Predicted Category: {category}")
        print(f"Confidence: {confidence:.4f}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

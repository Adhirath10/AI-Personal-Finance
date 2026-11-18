import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion

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


# Everyday merchants and words that the dataset may not contain, so a transaction
# typed in the app ("Dominos pizza", "Jio recharge") still gets a sensible
# category. Only categories that exist in the training data are used, and these
# rows carry a small weight so they never outweigh the user's real history.
KNOWLEDGE_BASE = {
    "Food": ["dominos pizza", "pizza hut", "kfc", "mcdonalds", "burger king", "starbucks coffee",
             "cafe coffee day", "subway", "haldiram", "dinner restaurant", "lunch", "breakfast",
             "chai tea stall", "biryani", "bakery", "eatsure", "food order"],
    "Groceries": ["more supermarket", "spencers", "zepto", "jiomart", "nature basket", "kirana store",
                  "vegetables", "fruits", "milk dairy", "swiggy instamart", "supermarket"],
    "Shopping": ["nykaa fashion", "decathlon", "croma electronics", "reliance digital", "ikea",
                 "lifestyle", "westside", "shoppers stop", "meesho", "clothes", "shoes", "electronics",
                 "laptop", "mobile phone purchase", "tata cliq"],
    "Transport": ["rapido bike", "petrol pump", "indian oil", "hp petrol", "bharat petroleum", "diesel",
                  "cab ride", "taxi", "parking fee", "toll fastag", "local train", "bus ticket"],
    "Travel": ["goibibo", "yatra", "cleartrip", "indigo airlines", "air india", "vistara", "oyo rooms",
               "airbnb", "trip holiday", "train ticket", "redbus", "resort"],
    "Bills": ["jio recharge", "airtel recharge", "vi recharge", "bsnl", "tata power", "msedcl electricity",
              "mahanagar gas", "broadband", "act fibernet", "dth recharge", "tata play", "water tax",
              "postpaid bill", "utility bill"],
    "Subscription": ["hotstar", "disney hotstar", "sony liv", "zee5", "apple music", "google one",
                     "icloud", "chatgpt plus", "linkedin premium", "jiocinema", "membership renewal",
                     "monthly subscription"],
    "Entertainment": ["pvr cinemas", "inox", "bookmyshow", "playstation", "steam games", "amusement park",
                      "bowling", "event tickets", "comedy show", "match tickets"],
    "Healthcare": ["apollo pharmacy", "medplus", "netmeds", "pharmeasy", "1mg", "doctor consultation",
                   "dentist", "medicines", "health checkup", "fortis hospital", "eye clinic"],
    "Education": ["byjus", "unacademy", "school fees", "tuition", "exam fee", "stationery",
                  "textbooks", "upgrad", "skillshare", "online course"],
    "Investment": ["zerodha", "groww", "upstox", "ppf deposit", "nps contribution", "recurring deposit",
                   "gold investment", "equity shares", "index fund", "elss"],
    "Personal Care": ["haircut", "barber", "spa", "nykaa beauty", "lakme salon", "urban company",
                      "skincare", "grooming", "fitness membership", "cult fit"],
    "Rent": ["flat rent", "pg rent", "hostel rent", "landlord", "room rent", "apartment rent"],
    "Salary": ["salary", "payroll", "monthly pay", "stipend"],
    "Other Income": ["cashback", "refund", "interest credit", "dividend", "bonus", "consulting income"],
    "Other": ["atm withdrawal", "charity", "temple", "wedding gift", "misc expense", "cash"],
}
KNOWLEDGE_WEIGHT = 0.3
# Each example is added twice (half weight each) so its words pass min_df=2.
KNOWLEDGE_COPIES = 2
LOW_CONFIDENCE = 0.3


def knowledge_rows(categories) -> pd.DataFrame:
    """Built-in examples for the given categories (see KNOWLEDGE_BASE)."""
    rows = [
        {"text": prepare_text_feature(phrase, ""), "category": category, "merchant": f"kb:{phrase}"}
        for category in sorted(set(categories))
        for phrase in KNOWLEDGE_BASE.get(category, [])
        for _ in range(KNOWLEDGE_COPIES)
    ]
    return pd.DataFrame(rows, columns=["text", "category", "merchant"])


def build_vectorizer() -> FeatureUnion:
    """Whole words (readable, used in explanations) plus character pieces of words.

    The character pieces make the model tolerant of spelling mistakes and
    variants ("zomto", "swigy", "amazon.in") that whole words miss.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=20000)),
    ])


def build_classifier() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", C=4.0, max_iter=3000, random_state=42, solver="lbfgs")


def _with_knowledge(texts: pd.Series, labels: pd.Series):
    """Training rows plus the down-weighted built-in examples."""
    extra = knowledge_rows(labels.unique())
    all_text = pd.concat([texts.reset_index(drop=True), extra["text"]], ignore_index=True)
    all_labels = pd.concat([labels.reset_index(drop=True), extra["category"]], ignore_index=True)
    weights = np.concatenate([np.ones(len(texts)), np.full(len(extra), KNOWLEDGE_WEIGHT / KNOWLEDGE_COPIES)])
    return all_text, all_labels, weights


def fit_categorizer(texts: pd.Series, labels: pd.Series, *, use_knowledge: bool = True, vectorizer=None, classifier=None):
    vectorizer = vectorizer if vectorizer is not None else build_vectorizer()
    classifier = classifier if classifier is not None else build_classifier()
    if use_knowledge:
        texts, labels, weights = _with_knowledge(texts, labels)
    else:
        weights = np.ones(len(texts))
    classifier.fit(vectorizer.fit_transform(texts), labels, sample_weight=weights)
    return vectorizer, classifier


def add_typos(text: str, seed: int) -> str:
    """Drop or swap one letter in each longer word: a simple spelling-mistake test."""
    rng = np.random.default_rng(seed)
    words = []
    for word in text.split():
        if len(word) >= 5:
            position = int(rng.integers(1, len(word) - 1))
            if rng.random() < 0.5:
                word = word[:position] + word[position + 1:]
            else:
                word = word[:position - 1] + word[position] + word[position - 1] + word[position + 1:]
        words.append(word)
    return " ".join(words)


def _accuracy(vectorizer, classifier, texts, labels) -> float:
    if len(texts) == 0:
        return float("nan")
    return float(accuracy_score(labels, classifier.predict(vectorizer.transform(list(texts)))))


def robustness_checks(df: pd.DataFrame, X_train, X_test, y_train, y_test) -> dict:
    """Compare the old word-only model with the improved one on harder tests.

    * Spelling mistakes: the test transactions with one letter dropped or swapped.
    * Unseen merchants: whole merchants are held out of training, so the model
      has to recognise them from their words, not from having seen them before.
    """
    def baseline():
        return (
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, solver="lbfgs"),
        )

    typo_text = [add_typos(text, index) for index, text in enumerate(X_test)]
    checks = {"Previous model (words only)": {}, "Improved model (words + character pieces + everyday merchants)": {}}
    models = {
        "Previous model (words only)": fit_categorizer(X_train, y_train, use_knowledge=False, vectorizer=baseline()[0], classifier=baseline()[1]),
        "Improved model (words + character pieces + everyday merchants)": fit_categorizer(X_train, y_train),
    }
    for name, (vectorizer, classifier) in models.items():
        checks[name]["Standard test set"] = _accuracy(vectorizer, classifier, X_test, y_test)
        checks[name]["Test set with spelling mistakes"] = _accuracy(vectorizer, classifier, typo_text, y_test)

    merchants = df["merchant"].str.lower()
    if merchants.nunique() >= 10:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_index, test_index = next(splitter.split(df["text"], df["category"], groups=merchants))
        train_part, test_part = df.iloc[train_index], df.iloc[test_index]
        # Only judge categories the model saw during training.
        test_part = test_part[test_part["category"].isin(set(train_part["category"]))]
        if test_part.empty:
            return checks
        for name, builder in (
            ("Previous model (words only)", lambda: fit_categorizer(train_part["text"], train_part["category"], use_knowledge=False, vectorizer=baseline()[0], classifier=baseline()[1])),
            ("Improved model (words + character pieces + everyday merchants)", lambda: fit_categorizer(train_part["text"], train_part["category"])),
        ):
            vectorizer, classifier = builder()
            checks[name]["Merchants never seen in training"] = _accuracy(vectorizer, classifier, test_part["text"], test_part["category"])
    return checks


def train_model(database_path: str | Path | None = None, model_dir: str | Path | None = None):
    """Train the TF-IDF (words + character pieces) + Logistic Regression categorizer.

    The model is evaluated on a held-out 20% split, then refitted on every
    labelled transaction so the saved model has seen all the data.
    """
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

    vectorizer, classifier = fit_categorizer(X_train, y_train)
    y_pred = classifier.predict(vectorizer.transform(X_test))

    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y_train).union(set(y_test)))
    report_path = target_dir / "evaluation_report.txt"
    metrics = evaluate_predictions(y_test, y_pred, labels=labels, output_path=report_path)
    save_confusion_matrix(y_test, y_pred, labels, target_dir / "confusion_matrix.png")

    checks = robustness_checks(df, X_train, X_test, y_train, y_test)
    lines = ["", "Robustness checks (accuracy)", "-" * 28]
    for name, values in checks.items():
        lines.append(name + ":")
        lines.extend(f"  {test}: {value:.4f}" for test, value in values.items())
    lines += [
        "",
        f"Predictions below {LOW_CONFIDENCE:.0%} confidence are marked 'needs review' in the app.",
        "The saved model is refitted on all labelled transactions after this evaluation.",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    final_vectorizer, final_classifier = fit_categorizer(X, y)
    joblib.dump(final_vectorizer, target_dir / "tfidf_vectorizer.pkl")
    joblib.dump(final_classifier, target_dir / "category_classifier.pkl")

    return {
        "metrics": metrics,
        "robustness": checks,
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
        for name, values in result["robustness"].items():
            print(name)
            for test, value in values.items():
                print(f"  {test}: {value:.4f}")
        print(f"Model files saved to: {result['model_dir']}")
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

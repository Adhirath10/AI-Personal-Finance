from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from explainability.common import (
    AI_SERVICE_DIR,
    deterministic_sample,
    json_number,
    load_source_module,
    output_directory,
    require_shap,
    safe_filename,
    save_waterfall,
    single_output_explanation,
    write_global_importance,
    write_json,
)


DEFAULT_MODEL_DIR = AI_SERVICE_DIR / "models" / "categorization"
_training = load_source_module(
    "_finance_xai_categorization_training",
    AI_SERVICE_DIR / "categorization" / "train.py",
)


def load_categorization_artifacts(
    model_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    target = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    vectorizer_path = target / "tfidf_vectorizer.pkl"
    classifier_path = target / "category_classifier.pkl"
    if not vectorizer_path.is_file() or not classifier_path.is_file():
        raise FileNotFoundError(f"Categorization artifacts were not found in {target}.")

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    if getattr(classifier, "n_features_in_", None) != _feature_count(vectorizer):
        raise ValueError("Categorization vectorizer and classifier do not match.")
    return vectorizer, classifier


def _feature_count(vectorizer: Any) -> int:
    """Number of model inputs for a plain TF-IDF or a words + characters union."""
    try:
        return len(vectorizer.get_feature_names_out())
    except (AttributeError, ValueError):
        return len(getattr(vectorizer, "vocabulary_", {}))


ABSENT_TERMS = "__absent__"


def group_term_contributions(text: str, names, input_values, contributions):
    """Turn per-feature SHAP values into per-word contributions.

    The improved model sees whole words ("word__swiggy") and character pieces
    of words ("char__swig"). Pieces present in the text are added to the word
    they come from, so the explanation reads in whole words and still adds up.
    Features that are absent from this text (they only move the score through
    the model's starting point) are combined into one ABSENT_TERMS row.
    Older word-only vectorizers have no prefixes and are reported per term.
    """
    words = text.split()
    totals: dict[str, float] = {}
    tfidf: dict[str, float] = {}
    for name, value, contribution in zip(names, input_values, contributions):
        contribution = float(contribution)
        if contribution == 0:
            continue
        name = str(name)
        if value == 0:
            totals[ABSENT_TERMS] = totals.get(ABSENT_TERMS, 0.0) + contribution
            continue
        if name.startswith("char__"):
            piece = name[len("char__"):].strip()
            owners = [word for word in dict.fromkeys(words) if piece and piece in word] or [ABSENT_TERMS]
            for owner in owners:
                totals[owner] = totals.get(owner, 0.0) + contribution / len(owners)
            continue
        term = name[len("word__"):] if name.startswith("word__") else name
        totals[term] = totals.get(term, 0.0) + contribution
        tfidf[term] = float(value)
    return [
        {"term": term, "tfidf_value": tfidf.get(term), "shap_value": json_number(total)}
        for term, total in sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)
    ]


def _reference_data(database_path: str | Path | None, background_size: int):
    data = _training.load_transaction_data(database_path)
    x_train, x_test, y_train, _ = train_test_split(
        data["text"],
        data["category"],
        test_size=0.2,
        stratify=data["category"],
        random_state=42,
    )
    return (
        deterministic_sample(
            x_train.to_frame("text"),
            background_size,
        )["text"],
        x_test,
        y_train,
    )


def _validate_reconstructed_training(
    classifier: Any,
    vectorizer: Any,
    training_labels,
) -> None:
    model_classes = list(classifier.classes_)
    source_classes = sorted(training_labels.astype(str).unique())
    if set(model_classes) != set(source_classes):
        raise ValueError(
            "Current database categories do not match the saved classifier classes; "
            "a leak-safe training background cannot be reconstructed."
        )
    if _feature_count(vectorizer) != classifier.n_features_in_:
        raise ValueError("Saved TF-IDF vocabulary differs from classifier inputs.")


def _text_for(description: Any, merchant: Any) -> str:
    return _training.prepare_text_feature(description, merchant)


def _explain_rows(
    text_rows,
    vectorizer: Any,
    classifier: Any,
    background_sparse,
):
    shap = require_shap()
    explainer = shap.LinearExplainer(classifier, background_sparse)
    return explainer(vectorizer.transform(text_rows))


def explain_category(
    description: str | None,
    merchant: str | None,
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    class_label: str | None = None,
    background_size: int = 100,
    max_display: int = 20,
) -> dict[str, Any]:
    if not (description and str(description).strip()) and not (
        merchant and str(merchant).strip()
    ):
        raise ValueError("Provide a non-empty description or merchant.")
    if background_size < 1:
        raise ValueError("background_size must be greater than zero.")

    vectorizer, classifier = load_categorization_artifacts(model_dir)
    background_text, _, training_labels = _reference_data(
        database_path, background_size
    )
    _validate_reconstructed_training(classifier, vectorizer, training_labels)
    background = vectorizer.transform(background_text.tolist())
    text = _text_for(description, merchant)
    explanation = _explain_rows([text], vectorizer, classifier, background)

    predicted_label = str(classifier.predict(vectorizer.transform([text]))[0])
    selected_label = str(class_label or predicted_label)
    classes = list(classifier.classes_)
    if selected_label not in classes:
        raise ValueError(f"Unknown category class: {selected_label}")
    class_index = classes.index(selected_label)
    selected_explanation = single_output_explanation(explanation, class_index)
    base_value = float(np.asarray(selected_explanation.base_values).reshape(-1)[0])
    contributions = np.asarray(selected_explanation.values)[0]
    score = float(
        classifier.decision_function(vectorizer.transform([text]))[0][class_index]
    )
    if not np.isclose(base_value + contributions.sum(), score, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            "Categorization SHAP values do not add up to the class score."
        )
    probabilities = classifier.predict_proba(vectorizer.transform([text]))[0]
    confidence = float(probabilities[classes.index(predicted_label)])

    term_names = list(vectorizer.get_feature_names_out())
    input_values = vectorizer.transform([text]).toarray()[0]
    grouped_terms = group_term_contributions(text, term_names, input_values, contributions)[:max_display]
    target_dir = output_directory(output_dir)
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    stem = target_dir / (f"category_{safe_filename(selected_label)}_{text_digest}")
    waterfall_path = save_waterfall(
        selected_explanation,
        stem.with_name(f"{stem.name}_waterfall.png"),
        max_display=max_display,
        title=f"Category decision score: {selected_label}",
    )
    individual_path = write_json(
        stem.with_suffix(".json"),
        {
            "model": "TF-IDF + LogisticRegression",
            "predicted_category": predicted_label,
            "explained_category": selected_label,
            "confidence": confidence,
            "class_score": score,
            "score_type": "Logistic Regression decision_function score",
            "base_value": base_value,
            "background_source": (
                "reconstructed from the current database using the original "
                "stratified split procedure and seed; the model artifact does "
                "not contain a source-data fingerprint"
            ),
            "top_term_contributions": grouped_terms,
            "visualizations": [str(waterfall_path)],
            "explanation": (
                "Contributions explain this class's decision score, not a change "
                "in predicted probability. Terms are TF-IDF features after the "
                "existing description/merchant cleaner."
            ),
        },
    )
    return {
        "predicted_category": predicted_label,
        "explained_category": selected_label,
        "confidence": confidence,
        "json_path": str(individual_path),
        "visualizations": [str(waterfall_path)],
        "explanation": (
            "Individual text contributions are class-specific TF-IDF n-grams; "
            "they do not explain causality or probability directly."
        ),
    }


def explain_categorization_global(
    *,
    database_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    background_size: int = 100,
    max_rows: int = 200,
    max_display: int = 20,
) -> dict[str, Any]:
    if background_size < 1 or max_rows < 1:
        raise ValueError("background_size and max_rows must be greater than zero.")
    vectorizer, classifier = load_categorization_artifacts(model_dir)
    background_text, test_text, training_labels = _reference_data(
        database_path, background_size
    )
    _validate_reconstructed_training(classifier, vectorizer, training_labels)
    test_text = deterministic_sample(test_text.to_frame("text"), max_rows)["text"]
    background = vectorizer.transform(background_text.tolist())
    explanation = _explain_rows(
        test_text.tolist(),
        vectorizer,
        classifier,
        background,
    )
    values = np.asarray(explanation.values)
    if values.ndim != 3:
        raise ValueError(
            "Expected class-specific SHAP values for the multiclass classifier."
        )
    class_importance = np.abs(values).mean(axis=0)
    overall_importance = class_importance.mean(axis=1)
    feature_names = list(vectorizer.get_feature_names_out())
    target_dir = output_directory(output_dir)
    stem = target_dir / "categorization_global"
    csv_path = write_global_importance(
        stem.with_suffix(".csv"),
        feature_names,
        overall_importance,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_indexes = np.argsort(overall_importance)[-max_display:]
    fig, axis = plt.subplots(figsize=(10, max(5, len(top_indexes) * 0.32)))
    axis.barh(
        [feature_names[index] for index in top_indexes],
        overall_importance[top_indexes],
    )
    axis.set_title("Categorization: global mean absolute SHAP by TF-IDF term")
    axis.set_xlabel("Mean absolute class decision-score contribution")
    fig.tight_layout()
    plot_path = stem.with_suffix(".png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    class_csv_path = stem.with_name("categorization_global_by_class.csv")
    rows = [
        {
            "class": str(class_name),
            "feature": feature_names[feature_index],
            "mean_absolute_shap": float(class_importance[feature_index, class_index]),
        }
        for class_index, class_name in enumerate(classifier.classes_)
        for feature_index in range(len(feature_names))
    ]
    import pandas as pd

    pd.DataFrame(rows).to_csv(class_csv_path, index=False)
    payload = {
        "model": "TF-IDF + LogisticRegression",
        "explained_rows": int(len(test_text)),
        "reference_rows": int(len(background_text)),
        "feature_count": len(feature_names),
        "class_labels": [str(name) for name in classifier.classes_],
        "background_source": (
            "reconstructed training split from current database; the model "
            "artifact does not store a source-data fingerprint"
        ),
        "reference_warning": (
            "The original training row order/snapshot was not saved, so this "
            "reconstruction cannot prove the exact historical split."
        ),
        "global_importance_csv": str(csv_path),
        "class_importance_csv": str(class_csv_path),
        "visualizations": [str(plot_path)],
        "explanation": (
            "Global importance averages absolute class-specific decision-score "
            "attributions over the reconstructed held-out split. The background "
            "uses only the reconstructed training split."
        ),
    }
    json_path = write_json(stem.with_suffix(".json"), payload)
    payload["json_path"] = str(json_path)
    return payload

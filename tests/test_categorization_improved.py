import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from categorization_fixture import write_categorization_db

AI_SERVICE_DIR = Path(__file__).resolve().parents[1] / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))
_spec = importlib.util.spec_from_file_location("_finance_test_categorization_train", AI_SERVICE_DIR / "categorization" / "train.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("categorization")
    database = write_categorization_db(folder / "finance.db")
    result = T.train_model(database, folder / "model")
    return database, folder / "model", result


def _predict(model_dir, text):
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
    classifier = joblib.load(model_dir / "category_classifier.pkl")
    return classifier.predict(vectorizer.transform([T.prepare_text_feature(text, "")]))[0]


def test_improved_model_handles_spelling_mistakes_better(trained):
    _, model_dir, result = trained
    previous = result["robustness"]["Previous model (words only)"]
    improved = result["robustness"]["Improved model (words + character pieces + everyday merchants)"]
    assert improved["Standard test set"] >= 0.98
    assert improved["Test set with spelling mistakes"] > previous["Test set with spelling mistakes"] + 0.2
    assert improved["Merchants never seen in training"] >= previous["Merchants never seen in training"]
    report = (model_dir / "evaluation_report.txt").read_text()
    assert "Robustness checks" in report and "spelling mistakes" in report


@pytest.mark.parametrize(
    ("text", "category"),
    [("swigy", "Food"), ("Zomto order", "Food"), ("Dominos pizza", "Food"), ("Jio recharge", "Bills"),
     ("Zerodha", "Investment"), ("Apollo pharmacy", "Healthcare"), ("Rapido ride", "Transport")],
)
def test_everyday_merchants_and_typos_get_sensible_categories(trained, text, category):
    assert _predict(trained[1], text) == category


def test_explanation_reads_in_whole_words_and_adds_up(trained, tmp_path):
    pytest.importorskip("shap")
    import explainability.categorization as xai

    database, model_dir, _ = trained
    result = xai.explain_category("Swigy order", "", database_path=database, model_dir=model_dir, output_dir=tmp_path)
    details = json.loads(Path(result["json_path"]).read_text())
    terms = {row["term"]: row["shap_value"] for row in details["top_term_contributions"]}
    assert result["predicted_category"] == "Food"
    assert set(terms) <= {"swigy", "order", xai.ABSENT_TERMS}
    assert terms["swigy"] > 0
    assert details["base_value"] + sum(terms.values()) == pytest.approx(details["class_score"], abs=1e-4)


def test_categorize_endpoint_flags_weak_guesses(trained, monkeypatch):
    import api

    monkeypatch.setitem(api.MODEL_DIRS, "categorization", trained[1])
    response = TestClient(api.app).post(
        "/api/categorize",
        json={"items": [{"description": "Netflix Transaction", "merchant": "Netflix"}, {"description": "qwerty", "merchant": ""}]},
    )
    sure, unsure = response.json()["items"]
    assert sure["category"] == "Subscription" and not sure["needs_review"]
    assert len(sure["alternatives"]) == 2
    assert unsure["needs_review"]

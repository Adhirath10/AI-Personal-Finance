import sqlite3
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


AI_SERVICE_DIR = Path(__file__).resolve().parents[1] / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

from explainability.anomaly_detection import (  # noqa: E402
    _explain_rows,
    _score_function,
    load_behavioral_transactions,
)
from explainability.common import single_output_explanation  # noqa: E402
from explainability.forecasting import (  # noqa: E402
    explain_forecast_for_user,
    load_forecast_artifacts,
)
import explainability.forecasting as forecasting_module  # noqa: E402


FORECAST_FEATURE_COLUMNS = [
    "previous_1",
    "previous_2",
    "previous_3",
    "rolling_3",
    "rolling_6",
    "month_number",
    "spending_change",
]


def _forecast_model():
    random = np.random.default_rng(42)
    training = pd.DataFrame(
        random.normal(size=(20, len(FORECAST_FEATURE_COLUMNS))),
        columns=FORECAST_FEATURE_COLUMNS,
    )
    target = training["previous_1"] * 2 + training["month_number"]
    return RandomForestRegressor(n_estimators=5, random_state=42).fit(
        training,
        target,
    )


def test_forecast_artifacts_keep_the_fitted_feature_order(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = _forecast_model()
    joblib.dump(model, model_dir / "forecasting_model.pkl")
    joblib.dump(FORECAST_FEATURE_COLUMNS, model_dir / "feature_columns.pkl")

    loaded_model, feature_columns = load_forecast_artifacts(model_dir)

    assert feature_columns == list(loaded_model.feature_names_in_)
    assert len(feature_columns) == loaded_model.n_features_in_


def test_forecast_explanation_matches_existing_predictor(tmp_path, monkeypatch):
    database_path = tmp_path / "forecast.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE transactions (
                user_id TEXT,
                transaction_date TEXT,
                amount REAL,
                transaction_type TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO transactions VALUES (?, ?, ?, ?)",
            [
                ("U1", "2025-01-05", 100.0, "Expense"),
                ("U1", "2025-02-05", 150.0, "Expense"),
                ("U1", "2025-03-05", 175.0, "Expense"),
                ("U1", "2025-04-05", 220.0, "Expense"),
            ],
        )

    script_dir = str(AI_SERVICE_DIR / "forecasting")
    previous_train = sys.modules.pop("train", None)
    sys.path.insert(0, script_dir)
    try:
        predictor_path = AI_SERVICE_DIR / "forecasting" / "predict.py"
        spec = importlib.util.spec_from_file_location(
            "_finance_test_existing_forecast_predictor",
            predictor_path,
        )
        assert spec is not None and spec.loader is not None
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
    finally:
        sys.path.remove(script_dir)
        sys.modules.pop("train", None)
        if previous_train is not None:
            sys.modules["train"] = previous_train

    model = _forecast_model()
    monkeypatch.setattr(
        forecasting_module,
        "load_forecast_artifacts",
        lambda model_dir=None: (model, FORECAST_FEATURE_COLUMNS),
    )
    monkeypatch.setattr(
        predictor,
        "load_model",
        lambda model_dir=None: (model, FORECAST_FEATURE_COLUMNS),
    )
    expected = predictor.predict_next_month_expense(
        "U1",
        database_path=database_path,
    )
    explanation = explain_forecast_for_user(
        "U1",
        database_path=database_path,
        output_dir=tmp_path / "explanations",
    )

    assert explanation["forecast_month"] == expected[1]
    assert explanation["prediction"] == expected[2]


def test_behavioral_data_loader_uses_feature_columns_without_labels(tmp_path):
    database_path = tmp_path / "finance.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE transactions (
            transaction_id TEXT,
            user_id TEXT,
            transaction_date TEXT,
            amount REAL,
            transaction_type TEXT,
            category TEXT,
            subcategory TEXT,
            payment_mode TEXT,
            account_type TEXT,
            is_recurring INTEGER,
            balance_after_transaction REAL,
            merchant TEXT,
            is_anomaly INTEGER,
            anomaly_type TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO transactions VALUES
        ('T1', 'U1', '2025-11-01', 12.5, 'Expense', 'Food', 'Cafe',
         'Card', 'Checking', 0, 100.0, 'Cafe', 1, 'not a feature')
        """
    )
    connection.commit()
    connection.close()

    loaded = load_behavioral_transactions(database_path)

    assert list(loaded.columns) == [
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
        "merchant",
    ]
    assert len(loaded) == 1
    assert "is_anomaly" not in loaded.columns
    assert "anomaly_type" not in loaded.columns
    with sqlite3.connect(database_path) as verification:
        assert (
            verification.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
        )


def test_anomaly_score_function_uses_negated_pipeline_decision_score():
    class Model:
        def decision_function(self, frame):
            return frame["amount"].to_numpy() / 10

    score = _score_function(Model(), ["amount"])
    result = score(pd.DataFrame({"amount": [10.0, 30.0]}))

    np.testing.assert_array_equal(result, [-1.0, -3.0])


def test_anomaly_explainer_handles_categorical_inputs_without_changing_values():
    class Model:
        def decision_function(self, frame):
            category_score = frame["category"].map({"Food": 1.0, "Bills": 2.0})
            return (frame["amount"] + category_score).to_numpy()

    background = pd.DataFrame({"amount": [10.0, 20.0], "category": ["Food", "Bills"]})
    row = pd.DataFrame({"amount": [15.0], "category": ["Food"]})
    explanation = _explain_rows(Model(), background, row, max_evals=5)

    np.testing.assert_allclose(
        explanation.base_values[0] + explanation.values[0].sum(),
        -16.0,
    )
    assert explanation.data[0, 1] == "Food"


def test_single_output_explanation_selects_requested_class():
    import shap

    explanation = shap.Explanation(
        values=np.array([[[1.0, -2.0], [3.0, 4.0]]]),
        base_values=np.array([[0.5, 1.5]]),
        data=np.array([[5.0, 6.0]]),
        feature_names=["first", "second"],
    )

    selected = single_output_explanation(explanation, output_index=1)

    np.testing.assert_array_equal(selected.values, [[-2.0, 4.0]])
    np.testing.assert_array_equal(selected.base_values, [1.5])
    assert list(selected.feature_names) == ["first", "second"]

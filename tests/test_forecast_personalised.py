import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import importlib.util

AI_SERVICE_DIR = Path(__file__).resolve().parents[1] / "ai-service"
# Several modules are named train.py; load the forecasting one under its own name.
_spec = importlib.util.spec_from_file_location("_finance_test_forecasting_train", AI_SERVICE_DIR / "forecasting" / "train.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

SEASON = [1.0, .95, 1.02, .98, 1.03, .97, .96, 1.0, .99, 1.08, 1.10, 1.06]


def _database(path, users=60, seed=3, per_month=1):
    rng = np.random.default_rng(seed)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE transactions (user_id TEXT, transaction_date TEXT, amount REAL, transaction_type TEXT)")
    rows = []
    for user in range(users):
        base = float(np.exp(rng.uniform(np.log(15000), np.log(300000))))
        for month in range(12):
            total = base * SEASON[month] * rng.lognormal(0, 0.12)
            for part in range(per_month):
                rows.append((f"U{user:03d}", f"2025-{month + 1:02d}-{10 + part:02d}", total / per_month, "Expense"))
    connection.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()


def test_training_saves_personalised_model_and_readable_report(tmp_path):
    database = tmp_path / "finance.db"
    _database(database)

    result = T.train_and_evaluate_model(database, tmp_path / "model")

    metadata = json.loads((tmp_path / "model" / T.METADATA_FILE).read_text())
    assert metadata["target"] == T.RELATIVE_TARGET
    assert metadata["feature_columns"] == T.RELATIVE_FEATURE_COLUMNS
    assert metadata["selected"] in {"model", "moving_average"}
    report = (tmp_path / "model" / "evaluation_report.txt").read_text()
    assert "Machine Learning Model (used by the app):" in report
    assert "Baseline (Moving Average of last 3 months)" in report
    # The moving-average baseline is a real prediction now, not a placeholder 0.
    assert result["results"]["Baseline (Moving Average of last 3 months)"]["mape"] < 50


def test_personalised_model_handles_users_outside_the_training_range(tmp_path):
    database = tmp_path / "finance.db"
    _database(database, users=80)
    dataset = T.create_forecast_dataset(T.build_monthly_expense_data(database))
    level = dataset.groupby("user_id")["total_expense"].transform("mean")
    train_rows, _ = T.chronological_split(dataset[level >= 80000])
    _, test_rows = T.chronological_split(dataset[level < 45000])
    actual = test_rows["target_expense"].to_numpy()

    legacy = RandomForestRegressor(n_estimators=100, random_state=42).fit(
        train_rows[T.LEGACY_FEATURE_COLUMNS], train_rows["target_expense"]
    )
    legacy_error = np.mean(np.abs(legacy.predict(test_rows[T.LEGACY_FEATURE_COLUMNS]) - actual) / actual)
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_depth=4, random_state=42).fit(
        train_rows[T.RELATIVE_FEATURE_COLUMNS], train_rows[T.RELATIVE_TARGET]
    )
    metadata = {"target": T.RELATIVE_TARGET, "feature_columns": T.RELATIVE_FEATURE_COLUMNS, "selected": "model", "model_weight": 0.5}
    new_error = np.mean(np.abs(T.predict_expense(model, metadata, test_rows, T.RELATIVE_FEATURE_COLUMNS) - actual) / actual)

    assert legacy_error > 1.0  # the old rupee model is badly off for these users
    assert new_error < 0.2


def test_predictions_stay_within_limits_and_metadata_must_match_the_model():
    rows = pd.DataFrame({column: [1.0] for column in T.RELATIVE_FEATURE_COLUMNS + T.LEGACY_FEATURE_COLUMNS})
    rows["rolling_3"] = 50000.0

    class Fixed:
        def __init__(self, value):
            self.value = value

        def predict(self, frame):
            return np.full(len(frame), self.value)

    metadata = {"target": T.RELATIVE_TARGET, "feature_columns": T.RELATIVE_FEATURE_COLUMNS, "selected": "model", "model_weight": 0.5}
    assert T.predict_expense(Fixed(9.0), metadata, rows, T.RELATIVE_FEATURE_COLUMNS)[0] == pytest.approx(80000)
    assert T.predict_expense(Fixed(1.1), metadata, rows, T.RELATIVE_FEATURE_COLUMNS)[0] == pytest.approx(52500)
    # An old rupee model far off the user's own level is kept within range.
    assert T.predict_expense(Fixed(200000.0), None, rows)[0] == pytest.approx(80000)
    # Metadata describing other features is ignored (treated as an old model).
    assert not T.is_relative(metadata, T.LEGACY_FEATURE_COLUMNS)


def test_forecast_explanation_adds_up_to_the_personalised_forecast(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    sys.path.insert(0, str(AI_SERVICE_DIR))
    import explainability.forecasting as xai

    database = tmp_path / "finance.db"
    _database(database)
    model_dir = tmp_path / "model"
    T.train_and_evaluate_model(database, model_dir)
    metadata = json.loads((model_dir / T.METADATA_FILE).read_text())
    metadata["selected"] = "model"
    (model_dir / T.METADATA_FILE).write_text(json.dumps(metadata))
    monkeypatch.setattr(xai, "_training", T)

    result = xai.explain_forecast_for_user("U005", database_path=database, model_dir=model_dir, output_dir=tmp_path / "out")
    total = result["base_value"] + sum(row["shap_value"] for row in result["feature_contributions"])

    rows = T.build_forecast_features(T.build_monthly_expense_data(database))
    last = rows[rows["user_id"] == "U005"].tail(1)
    model = __import__("joblib").load(model_dir / "forecasting_model.pkl")
    expected = T.predict_expense(model, metadata, last, T.RELATIVE_FEATURE_COLUMNS)[0]
    assert result["prediction"] == pytest.approx(expected)
    assert total == pytest.approx(result["prediction"], rel=1e-6)
    assert 0.6 * last["rolling_3"].iloc[0] <= result["prediction"] <= 1.6 * last["rolling_3"].iloc[0]


def test_evaluation_tests_every_user_and_ignores_a_stray_entry(tmp_path):
    database = tmp_path / "finance.db"
    _database(database, users=40, per_month=5)
    connection = sqlite3.connect(database)
    # One user with newer months (like generated demo data) and a stray test entry after them.
    connection.executemany(
        "INSERT INTO transactions VALUES (?, ?, ?, ?)",
        [("U000", f"2026-{month:02d}-{day:02d}", 12000.0, "Expense") for month in range(1, 7) for day in range(10, 15)] + [("U000", "2026-07-02", 150.0, "Expense")],
    )
    connection.commit()
    connection.close()

    result = T.train_and_evaluate_model(database, tmp_path / "model")

    assert result["metadata"]["test_rows"] == 80  # 2 months for each of the 40 users
    assert result["metadata"]["incomplete_months_left_out"] == ["U000 2026-07"]
    assert result["ml_metrics"]["mape"] < 50
    assert "Median percentage error" in (tmp_path / "model" / "evaluation_report.txt").read_text()


def test_quiet_months_in_the_middle_are_kept():
    monthly = pd.DataFrame({
        "user_id": ["U1"] * 5,
        "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"],
        "total_expense": [100.0, 5000.0, 300.0, 5000.0, 50.0],
        "transaction_count": [1, 20, 3, 20, 1],
    })
    kept, left_out = T.drop_incomplete_edge_months(monthly)
    assert list(kept["month"]) == ["2025-02", "2025-03", "2025-04"]
    assert list(left_out["month"]) == ["2025-01", "2025-05"]

"""Regression tests for the AI service forecast, forecast explanation and insights.

They run the HTTP adapter against the saved model artifacts with a small
synthetic database whose history ends before the "current" month, which is the
situation of the project's historical dataset.
"""

import importlib.util
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


AI_SERVICE_DIR = Path(__file__).resolve().parents[1] / "ai-service"
if str(AI_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_DIR))

import api  # noqa: E402


MONTHLY_SPENDING = {
    # month: {category: amount}
    "2025-07": {"Food": 4000.0, "Shopping": 3000.0, "Rent": 15000.0},
    "2025-08": {"Food": 4200.0, "Shopping": 2800.0, "Rent": 15000.0},
    "2025-09": {"Food": 3900.0, "Shopping": 3100.0, "Rent": 15000.0},
    "2025-10": {"Food": 4100.0, "Shopping": 3000.0, "Rent": 15000.0},
    "2025-11": {"Food": 4000.0, "Shopping": 2900.0, "Rent": 15000.0},
    "2025-12": {"Food": 6000.0, "Shopping": 6000.0, "Rent": 15000.0},
}


def _create_database(path: Path, extra_rows=()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE users (user_id TEXT PRIMARY KEY, monthly_income REAL);
            CREATE TABLE transactions (
                transaction_id TEXT PRIMARY KEY, user_id TEXT, transaction_date TEXT,
                description TEXT, merchant TEXT, amount REAL, transaction_type TEXT,
                category TEXT, subcategory TEXT, payment_mode TEXT, account_type TEXT,
                is_recurring INTEGER, balance_after_transaction REAL
            );
            """
        )
        connection.execute("INSERT INTO users VALUES ('U1', 60000)")
        rows = []
        for month, categories in MONTHLY_SPENDING.items():
            rows.append((f"{month}-01", "Salary", "Employer", 60000.0, "Income", "Salary"))
            for day, (category, amount) in enumerate(categories.items(), start=5):
                rows.append((f"{month}-{day:02d}", category, category, amount, "Expense", category))
        rows.extend(extra_rows)
        connection.executemany(
            "INSERT INTO transactions VALUES (?, 'U1', ?, ?, ?, ?, ?, ?, ?, 'UPI', 'Savings', 0, 1000)",
            [(f"T{index}", *row[:5], row[5], row[5]) for index, row in enumerate(rows)],
        )
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def service(tmp_path, monkeypatch):
    database = tmp_path / "finance.db"
    _create_database(database)
    monkeypatch.setattr(api, "DATABASE_PATH", database)
    monkeypatch.setattr(api, "EXPLANATION_OUTPUT_DIR", tmp_path / "explanations")

    def use_today(today):
        original = api._recommendation_engine

        def engine_on_fixed_date():
            engine = original()
            engine.today = today
            return engine

        monkeypatch.setattr(api, "_recommendation_engine", engine_on_fixed_date)

    use_today(date(2026, 11, 17))
    client = TestClient(api.app)
    client.database = database
    client.use_today = use_today
    return client


def test_forecast_after_dataset_end_is_returned_as_historical(service):
    response = service.get("/api/forecast", params={"user_id": "U1"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["forecast_month"] == "2026-01"
    assert body["predicted_expense"] is not None and body["predicted_expense"] > 0
    assert body["is_historical"] is True
    assert body["history_end_month"] == "2025-12"
    assert "already passed" in body["notice"]
    assert {"mae", "rmse", "mape"} <= body.keys()


def test_forecast_for_an_upcoming_month_is_not_historical(service):
    service.use_today(date(2026, 1, 5))

    body = service.get("/api/forecast", params={"user_id": "U1"}).json()

    assert body["status"] == "available"
    assert body["forecast_month"] == "2026-01"
    assert body["is_historical"] is False
    assert body["notice"] is None


def test_month_in_progress_is_forecast_from_complete_months(service):
    # On 20 December the December total is partial: forecast December itself.
    service.use_today(date(2025, 12, 20))

    body = service.get("/api/forecast", params={"user_id": "U1"}).json()

    assert body["status"] == "available"
    assert body["forecast_month"] == "2025-12"
    assert body["month_in_progress"] is True
    assert body["history_end_month"] == "2025-11"
    assert body["predicted_expense"] >= body["spent_so_far"] > 0
    assert body["is_historical"] is False


def test_insights_for_a_month_in_progress_compare_with_the_same_day(service):
    service.use_today(date(2025, 12, 15))

    body = service.get("/api/insights", params={"user_id": "U1"}).json()

    shopping = next(item for item in body["items"] if "Shopping" in item["title"])
    values = shopping["supporting_values"]
    assert values["compared_through_day"] == 15
    # Earlier Shopping purchases all fell on or before the 15th: by day 15 they were complete.
    assert values["historical_monthly_average"] == pytest.approx(3000.0)
    assert "so far in December 2025 (day 15 of 31)" in shopping["description"]
    assert "by this day" in shopping["description"]


def test_forecast_explanation_explains_the_returned_forecast(service):
    pytest.importorskip("shap")
    forecast = service.get("/api/forecast", params={"user_id": "U1"}).json()

    explanation = service.get(
        "/api/explanations", params={"user_id": "U1", "model": "forecast"}
    ).json()

    assert explanation["status"] == "available"
    assert explanation["forecast_month"] == forecast["forecast_month"]
    assert explanation["prediction"] == pytest.approx(forecast["predicted_expense"])
    assert explanation["feature_contributions"]
    assert explanation["feature_importance"]


def test_insights_compare_latest_month_with_prior_months(service):
    body = service.get("/api/insights", params={"user_id": "U1"}).json()

    assert body["status"] == "available"
    shopping = next(item for item in body["items"] if "Shopping" in item["title"])
    assert shopping["supporting_values"]["latest_month_spending"] == 6000.0
    assert shopping["supporting_values"]["historical_months_compared"] == 3
    assert shopping["supporting_values"]["historical_monthly_average"] == pytest.approx(3000.0)


def test_insights_use_latest_recorded_months_across_a_gap(tmp_path, monkeypatch):
    database = tmp_path / "gap.db"
    # A transaction added today, long after the imported history ends.
    _create_database(
        database,
        extra_rows=[("2026-11-17", "Lunch", "Cafe", 450.0, "Expense", "Food")],
    )
    monkeypatch.setattr(api, "DATABASE_PATH", database)

    body = TestClient(api.app).get("/api/insights", params={"user_id": "U1"}).json()

    assert body["status"] == "available"
    food = body["items"][0]
    assert food["supporting_values"]["category"] == "Food"
    assert food["supporting_values"]["latest_month_spending"] == 450.0
    assert food["supporting_values"]["historical_months_compared"] == 3


def test_latest_month_with_only_income_uses_latest_expense_month(tmp_path, monkeypatch):
    database = tmp_path / "income-only.db"
    # Salary credited in a new month before any expense is recorded for it.
    _create_database(
        database,
        extra_rows=[("2026-01-01", "Salary", "Employer", 60000.0, "Income", "Salary")],
    )
    monkeypatch.setattr(api, "DATABASE_PATH", database)
    client = TestClient(api.app)

    insights = client.get("/api/insights", params={"user_id": "U1"}).json()
    forecast = client.get("/api/forecast", params={"user_id": "U1"}).json()

    assert insights["status"] == "available"
    assert insights["analysis_period"]["selected_month"] == "2025-12"
    assert insights["analysis_period"]["latest_transaction_month"] == "2026-01"
    assert "2026-01 contain no expenses" in insights["notice"]
    assert insights["items"]
    assert forecast["status"] == "available"
    assert forecast["forecast_month"] == "2026-01"
    assert forecast["predicted_expense"] > 0


def test_insights_do_not_depend_on_other_models(service, monkeypatch):
    original = api._recommendation_engine

    def broken_models():
        engine = original()

        def fail(*args, **kwargs):
            raise AttributeError("incompatible saved model")

        engine._forecast_user_expenses = fail
        engine._score_potential_anomalies = fail
        return engine

    monkeypatch.setattr(api, "_recommendation_engine", broken_models)

    insights = service.get("/api/insights", params={"user_id": "U1"})
    forecast = service.get("/api/forecast", params={"user_id": "U1"})

    assert insights.status_code == 200
    assert insights.json()["status"] == "available"
    assert forecast.status_code == 200
    assert forecast.json()["status"] == "unavailable"
    assert "incompatible saved model" in forecast.json()["reason"]


def test_predict_script_forecasts_month_after_latest_record(service):
    script_dir = str(AI_SERVICE_DIR / "forecasting")
    previous_train = sys.modules.pop("train", None)
    sys.path.insert(0, script_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "_finance_test_forecast_predict", AI_SERVICE_DIR / "forecasting" / "predict.py"
        )
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        _, forecast_month, predicted = predictor.predict_next_month_expense(
            "U1", database_path=service.database
        )
    finally:
        sys.path.remove(script_dir)
        sys.modules.pop("train", None)
        if previous_train is not None:
            sys.modules["train"] = previous_train

    api_forecast = service.get("/api/forecast", params={"user_id": "U1"}).json()
    assert forecast_month == "2026-01"
    assert predicted == pytest.approx(api_forecast["predicted_expense"])


def test_selected_analysis_month_is_used_by_insights_forecast_and_explanation(service):
    params = {"user_id": "U1", "analysis_month": "2025-11"}

    insights = service.get("/api/insights", params=params).json()
    forecast = service.get("/api/forecast", params=params).json()

    assert insights["analysis_period"]["selected_month"] == "2025-11"
    assert insights["notice"] is None
    assert forecast["forecast_month"] == "2025-12"
    assert forecast["history_end_month"] == "2025-11"
    if importlib.util.find_spec("shap") is not None:
        explanation = service.get(
            "/api/explanations", params={**params, "model": "forecast"}
        ).json()
        assert explanation["forecast_month"] == "2025-12"
        assert explanation["prediction"] == pytest.approx(forecast["predicted_expense"])
    assert service.get(
        "/api/insights", params={"user_id": "U1", "analysis_month": "2024-01"}
    ).json()["status"] == "unavailable"


def test_forecast_for_the_current_month_is_not_historical(service):
    service.use_today(date(2026, 1, 15))

    body = service.get("/api/forecast", params={"user_id": "U1"}).json()

    assert body["status"] == "available"
    assert body["forecast_month"] == "2026-01"
    assert body["is_historical"] is False


def test_insight_text_leads_with_rupees_and_uses_times_for_big_jumps():
    row = {"category": "Education", "historical_months_compared": 3}
    big = api._insight_item(1, row, 4305.0, 716.0, "October 2026", 60000.0)
    assert big["title"] == "Education spending is up"
    assert big["description"] == "You spent ₹4,305 on Education in October 2026: ₹3,589 more than usual (about 6.0× your usual ₹716)."
    assert big["priority"] == "Medium"  # ₹3,589 is under 10% of the month's ₹60,000
    small = api._insight_item(2, row, 900.0, 1000.0, "October 2026", 60000.0)
    assert small["description"].endswith("₹100 less than usual (10% below your usual ₹1,000).")
    assert small["priority"] == "Low"
    rent = api._insight_item(3, {"category": "Rent", "historical_months_compared": 3}, 30000.0, 20000.0, "October 2026", 60000.0)
    assert rent["priority"] == "High"
    assert api._rupees(125000) == "₹1,25,000"

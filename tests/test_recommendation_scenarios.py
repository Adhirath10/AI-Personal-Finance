import sqlite3
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
RECOMMENDATION_DIR = ROOT / "ai-service" / "recommendation"
DATABASE_PATH = ROOT / "data" / "database" / "personal_finance.db"
sys.path.insert(0, str(RECOMMENDATION_DIR))

from engine import RecommendationEngine


@pytest.fixture(scope="module")
def engine():
    return RecommendationEngine(DATABASE_PATH, today=date(2026, 11, 11))


def _read_only_query(query, params=()):
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH.resolve().as_posix()}?mode=ro",
        uri=True,
    )
    try:
        return connection.execute(query, params).fetchall()
    finally:
        connection.close()


def _recommendations_for(engine, user_id, recommendation_type):
    result = engine.generate(user_id)
    return [
        item
        for item in result["recommendations"]
        if item["recommendation_type"] == recommendation_type
    ]


def _assert_finite_values(value):
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite_values(nested)
    elif isinstance(value, (int, float, np.number)):
        assert np.isfinite(value)


def test_high_food_expense_recommendation_matches_database(engine):
    user_id = "U007"
    latest_month = _read_only_query(
        "SELECT MAX(substr(transaction_date, 1, 7)) FROM transactions"
    )[0][0]
    stored_amount = _read_only_query(
        """
        SELECT SUM(amount)
        FROM transactions
        WHERE user_id = ? AND transaction_type = 'Expense'
          AND category = 'Food' AND substr(transaction_date, 1, 7) = ?
        """,
        (user_id, latest_month),
    )[0][0]
    assert stored_amount is not None

    analysis = engine.analyze_user(user_id)
    analyzed_amount = next(
        row["amount"]
        for row in analysis["category_spending"]
        if row["category"] == "Food"
    )
    all_recommendations = engine.generate(user_id)["recommendations"]
    food_recommendation = next(
        item
        for item in all_recommendations
        if item["recommendation_type"] == "budget_warning"
        and item["supporting_values"].get("category") == "Food"
    )

    assert analysis["analysis_month"] == latest_month
    assert analyzed_amount == stored_amount
    assert food_recommendation["supporting_values"]["actual_amount"] == stored_amount
    assert food_recommendation["user_id"] == user_id
    _assert_finite_values(food_recommendation["supporting_values"])


def test_over_budget_recommendation_matches_stored_budget(engine):
    user_id, month, category, budget, actual = _read_only_query(
        """
        SELECT user_id, budget_month, category, budget_amount, actual_amount
        FROM monthly_budgets
        WHERE exceeded = 1
        ORDER BY budget_month DESC, user_id
        LIMIT 1
        """
    )[0]
    recommendations = _recommendations_for(engine, user_id, "budget_warning")
    recommendation = next(
        item
        for item in recommendations
        if item["supporting_values"].get("category") == category
        and item["supporting_values"].get("month") == month
    )

    assert recommendation["supporting_values"]["budget_amount"] == budget
    assert recommendation["supporting_values"]["actual_amount"] == actual
    assert recommendation["supporting_values"]["over_budget_amount"] == max(
        actual - budget, 0.0
    )
    assert recommendation["priority"] == "High"
    _assert_finite_values(recommendation["supporting_values"])


def test_increasing_expense_recommendation_uses_recorded_months(engine):
    user_id = "U024"
    rows = _read_only_query(
        """
        SELECT substr(transaction_date, 1, 7) AS month, SUM(amount)
        FROM transactions
        WHERE user_id = ? AND transaction_type = 'Expense'
        GROUP BY month
        ORDER BY month
        """,
        (user_id,),
    )
    monthly = {month: amount for month, amount in rows}
    latest_month = max(monthly)
    current_expense = monthly[latest_month]
    current_period = pd.Period(latest_month, freq="M")
    prior = [
        monthly[str(current_period - offset)]
        for offset in range(1, 4)
        if str(current_period - offset) in monthly
    ]
    assert len(prior) >= 2
    expected_average = float(np.mean(prior))
    assert current_expense > expected_average * 1.1

    analysis = engine.analyze_user(user_id)
    trend = next(
        row
        for row in analysis["category_trends"]
        if row["category"] == "Food"
    )
    assert trend["latest_category_spending"] > (
        trend["historical_average_category_spending"] * 1.1
    )
    recommendations = engine.generate(user_id)["recommendations"]
    food_budget_warning = next(
        item
        for item in recommendations
        if item["recommendation_type"] == "budget_warning"
        and item["supporting_values"].get("category") == "Food"
    )
    assert food_budget_warning["supporting_values"]["actual_amount"] == (
        trend["latest_category_spending"]
    )
    assert not any(
        item["recommendation_type"] == "expense_reduction"
        and item["supporting_values"].get("category") == "Food"
        for item in recommendations
    )

    assert current_expense > expected_average * 1.1
    _assert_finite_values(food_budget_warning["supporting_values"])


def test_goal_progress_recommendation_matches_stored_goal(engine):
    user_id, goal_id, goal_type, target, current, target_date = _read_only_query(
        """
        SELECT user_id, goal_id, goal_type, target_amount, current_amount, target_date
        FROM financial_goals
        WHERE status = 'In Progress' AND current_amount < target_amount
        ORDER BY user_id
        LIMIT 1
        """
    )[0]
    recommendations = _recommendations_for(engine, user_id, "goal_progress")
    recommendation = next(
        item
        for item in recommendations
        if item["supporting_values"].get("goal_id") == goal_id
    )

    values = recommendation["supporting_values"]
    assert values["goal_type"] == goal_type
    assert values["target_amount"] == target
    assert values["current_amount"] == current
    assert values["remaining_amount"] == target - current
    assert values["target_date"] == target_date
    _assert_finite_values(values)


def test_anomaly_review_matches_out_of_sample_model_transactions(engine):
    user_id = "U001"
    analysis = engine.analyze_user(user_id)
    anomaly_data = analysis["potential_anomalies"]
    assert anomaly_data["status"] == "available"
    candidates = anomaly_data["transactions"]
    assert candidates

    recommendations = _recommendations_for(
        engine, user_id, "unusual_transaction_review"
    )
    recommendation = next(
        item
        for item in recommendations
        if item["supporting_values"].get("month") == analysis["analysis_month"]
    )
    flagged = recommendation["supporting_values"]["transactions"]
    assert flagged
    for item in flagged:
        stored = _read_only_query(
            """
            SELECT amount, category, transaction_type, transaction_date
            FROM transactions
            WHERE transaction_id = ? AND user_id = ?
            """,
            (item["transaction_id"], user_id),
        )
        assert len(stored) == 1
        amount, category, transaction_type, transaction_date = stored[0]
        assert amount == item["amount"]
        assert category == item["category"]
        assert transaction_type == "Expense"
        assert transaction_date == item["transaction_date"]
    assert recommendation["supporting_values"]["label_provenance"] == "unverified"
    _assert_finite_values(recommendation["supporting_values"])


def test_actual_user_recommendations_are_deduplicated(engine):
    result = engine.generate("U001")
    recommendations = result["recommendations"]
    assert result["recommendation_count"] == len(recommendations)
    assert result["analysis_period"]["selected_month"] == result["analysis_month"]
    expected_fields = {
        "user_id",
        "recommendation_title",
        "recommendation_description",
        "recommendation_type",
        "priority",
        "reason",
        "supporting_values",
    }
    assert all(expected_fields.issubset(item) for item in recommendations)
    assert all(item["user_id"] == "U001" for item in recommendations)
    assert all(item["priority"] in {"High", "Medium", "Low"} for item in recommendations)
    assert all(item["reason"] for item in recommendations)
    for item in recommendations:
        _assert_finite_values(item["supporting_values"])

    keys = [
        (
            item["user_id"],
            item["recommendation_type"],
            item["supporting_values"].get("month")
            or item["supporting_values"].get("forecast_month"),
            item["supporting_values"].get("category"),
            item["supporting_values"].get("goal_id"),
            tuple(
                sorted(
                    str(transaction.get("transaction_id"))
                    for transaction in item["supporting_values"].get(
                        "transactions", []
                    )
                    if transaction.get("transaction_id") is not None
                )
            ),
            item["recommendation_title"]
            if not any(
                (
                    item["supporting_values"].get("month"),
                    item["supporting_values"].get("forecast_month"),
                    item["supporting_values"].get("category"),
                    item["supporting_values"].get("goal_id"),
                    tuple(item["supporting_values"].get("transactions", [])),
                )
            )
            else None,
        )
        for item in recommendations
    ]

    assert len(keys) == len(set(keys))


def test_selected_analysis_month_filters_later_transactions(engine):
    analysis = engine.analyze_user("U001", analysis_month="2025-11")

    assert analysis["analysis_month"] == "2025-11"
    assert analysis["analysis_period"]["latest_transaction_month"] == "2025-12"
    assert analysis["analysis_period"]["period_status"] == "historical"
    assert analysis["data_quality"]["transactions_after_analysis_month_excluded"] > 0
    assert all(month <= "2025-11" for month in analysis["monthly_totals"])


def test_summary_only_period_does_not_use_later_transaction_data(tmp_path):
    database_path = tmp_path / "summary_only.db"
    _create_sparse_database(database_path, 1)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TABLE monthly_financial_summary (
                user_id TEXT,
                summary_month TEXT,
                total_income REAL,
                total_expense REAL,
                total_savings REAL,
                largest_category TEXT,
                largest_category_amount REAL,
                anomaly_count INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO monthly_financial_summary VALUES
            ('U-sparse', '2024-12', 1200, 500, 700, 'Food', 500, 0)
            """
        )
        connection.commit()
    finally:
        connection.close()

    engine = RecommendationEngine(
        database_path,
        forecast_model_dir=tmp_path / "no_forecast",
        anomaly_model_dir=tmp_path / "no_anomaly",
        categorization_model_dir=tmp_path / "no_categorization",
        today=date(2025, 2, 1),
    )
    analysis = engine.analyze_user("U-sparse", analysis_month="2024-12")

    assert analysis["analysis_month_totals"]["income"] == 1200.0
    assert analysis["analysis_month_totals"]["expense"] == 500.0
    assert analysis["data_quality"]["transaction_rows"] == 0
    assert analysis["data_quality"]["transactions_after_analysis_month_excluded"] == 1
    assert analysis["monthly_totals"] == {
        "2024-12": {
            "income": 1200.0,
            "expense": 500.0,
            "investment_expense": None,
            "discretionary_expense": None,
            "income_source": "monthly_financial_summary",
            "expense_source": "monthly_financial_summary",
        }
    }


def test_stale_forecast_is_reported_as_skipped(engine):
    result = engine.generate("U001")

    assert result["model_status"]["forecast"] == "stale"
    assert not any(
        item["recommendation_type"] == "future_expense_warning"
        for item in result["recommendations"]
    )
    skipped = next(
        item
        for item in result["recommendation_skips"]
        if item["recommendation_type"] == "future_expense_warning"
    )
    assert "not later than the current month" in skipped["reason"]


def _create_sparse_database(path, transaction_count):
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                transaction_date TEXT NOT NULL,
                description TEXT NOT NULL,
                merchant TEXT NOT NULL,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL,
                payment_mode TEXT NOT NULL,
                account_type TEXT NOT NULL,
                is_recurring INTEGER NOT NULL,
                balance_after_transaction REAL NOT NULL
            )
            """
        )
        rows = [
            (
                f"t{index}",
                "U-sparse",
                f"2025-01-{index + 1:02d}",
                "transaction",
                "merchant",
                40.0 + index,
                "Expense",
                "Food",
                "Food",
                "Cash",
                "Checking",
                0,
                100.0,
            )
            for index in range(transaction_count)
        ]
        connection.executemany(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def test_insufficient_history_suppresses_forecast_and_history_rules(tmp_path):
    database_path = tmp_path / "sparse.db"
    _create_sparse_database(database_path, 1)
    engine = RecommendationEngine(
        database_path,
        forecast_model_dir=ROOT / "ai-service" / "models" / "forecasting",
        anomaly_model_dir=tmp_path / "no_anomaly",
        categorization_model_dir=tmp_path / "no_categorization",
        today=date(2025, 2, 1),
    )

    result = engine.generate("U-sparse")
    types = {item["recommendation_type"] for item in result["recommendations"]}

    assert result["data_quality"]["months_with_expense_history"] == 1
    assert result["model_status"]["forecast"] == "insufficient_history"
    assert result["model_status"]["anomaly_detection"] == "unavailable"
    assert "future_expense_warning" not in types
    assert "expense_reduction" not in types
    forecast_skip = next(
        item
        for item in result["recommendation_skips"]
        if item["recommendation_type"] == "future_expense_warning"
    )
    assert "three recorded expense months" in forecast_skip["reason"]


def test_missing_budget_information_does_not_create_budget_recommendations(tmp_path):
    database_path = tmp_path / "no_budget.db"
    _create_sparse_database(database_path, 3)
    engine = RecommendationEngine(
        database_path,
        forecast_model_dir=tmp_path / "no_forecast",
        anomaly_model_dir=tmp_path / "no_anomaly",
        categorization_model_dir=tmp_path / "no_categorization",
        today=date(2025, 2, 1),
    )

    result = engine.generate("U-sparse")

    assert result["data_quality"]["budget_rows_for_analysis_month"] == 0
    assert result["model_status"]["forecast"] == "unavailable"
    forecast_skip = next(
        item
        for item in result["recommendation_skips"]
        if item["recommendation_type"] == "future_expense_warning"
    )
    assert "artifacts are missing" in forecast_skip["reason"]
    assert not any(
        item["recommendation_type"] == "budget_warning"
        for item in result["recommendations"]
    )


def test_unfitted_forecast_model_is_not_used(tmp_path):
    model_dir = tmp_path / "unfitted_forecast"
    model_dir.mkdir()
    joblib.dump(RandomForestRegressor(), model_dir / "forecasting_model.pkl")
    joblib.dump(["previous_1"], model_dir / "feature_columns.pkl")
    engine = RecommendationEngine(
        forecast_model_dir=model_dir,
        today=date(2025, 2, 1),
    )
    transactions = pd.DataFrame(
        {
            "transaction_date": ["2024-10-15", "2024-11-15", "2024-12-15"],
            "amount": [100.0, 120.0, 140.0],
            "transaction_type": ["Expense", "Expense", "Expense"],
        }
    )

    result = engine._forecast_user_expenses(
        transactions, pd.DataFrame(), pd.Period("2024-12", freq="M")
    )

    assert result["status"] == "unavailable"
    assert "not been fitted" in result["skip_reason"]


def test_real_database_is_unchanged_after_recommendation_tests():
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH.resolve().as_posix()}?mode=ro",
        uri=True,
    )
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok"
        assert connection.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0] == 47569
    finally:
        connection.close()

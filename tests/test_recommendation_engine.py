from datetime import date
from pathlib import Path
import sys


RECOMMENDATION_DIR = (
    Path(__file__).resolve().parents[1] / "ai-service" / "recommendation"
)
sys.path.insert(0, str(RECOMMENDATION_DIR))

from engine import RecommendationEngine


def _analysis():
    return {
        "user_id": "U-test",
        "analysis_month": "2025-09",
        "analysis_month_totals": {
            "income": 1000.0,
            "expense": 1250.0,
            "net": -250.0,
        },
        "monthly_totals": {
            "2025-06": {"expense": 700.0},
            "2025-07": {"expense": 800.0},
            "2025-08": {"expense": 900.0},
            "2025-09": {"expense": 1250.0},
        },
        "category_spending": [{"category": "Dining", "amount": 400.0}],
        "category_trends": [
            {
                "category": "Dining",
                "latest_category_spending": 400.0,
                "historical_months_compared": 2,
                "historical_average_category_spending": 200.0,
            }
        ],
        "budgets": [
            {
                "category": "Dining",
                "budget_amount": 300.0,
                "actual_amount": 400.0,
                "exceeded": 1,
            }
        ],
        "goals": [],
        "forecast": {"status": "stale"},
        "potential_anomalies": {"status": "no_out_of_sample_rows"},
    }


def test_generates_prioritized_data_backed_recommendations():
    engine = RecommendationEngine(today=date(2025, 9, 15))

    recommendations = engine.generate_recommendations(_analysis())

    categories = [item["recommendation_type"] for item in recommendations]
    assert "overspending" in categories
    assert "budget_warning" in categories
    assert "expense_reduction" not in categories
    assert "future_expense_warning" not in categories
    assert all(item["user_id"] == "U-test" for item in recommendations)
    assert all(
        {
            "user_id",
            "recommendation_title",
            "recommendation_description",
            "recommendation_type",
            "priority",
            "reason",
            "supporting_values",
        }.issubset(item)
        for item in recommendations
    )
    assert all(item["priority"] in {"High", "Medium", "Low"} for item in recommendations)
    assert all(item["reason"] and item["supporting_values"] for item in recommendations)
    assert next(
        item for item in recommendations if item["recommendation_type"] == "overspending"
    )["priority"] == "High"
    assert next(
        item for item in recommendations if item["recommendation_type"] == "budget_warning"
    )["priority"] == "High"


def test_suppresses_budget_warning_without_valid_budget():
    analysis = _analysis()
    analysis["budgets"] = [
        {
            "category": "Dining",
            "budget_amount": 0.0,
            "actual_amount": 400.0,
            "exceeded": 0,
        }
    ]
    engine = RecommendationEngine(today=date(2025, 9, 15))

    recommendations = engine.generate_recommendations(analysis)

    assert not any(
        item["recommendation_type"] == "budget_warning"
        for item in recommendations
    )


def test_deduplicates_same_category_and_month_recommendations():
    duplicate = {
        "user_id": "U-test",
        "recommendation_title": "Dining budget exceeded",
        "recommendation_description": "Review dining budget.",
        "recommendation_type": "budget_warning",
        "priority": "Medium",
        "reason": "Budget exceeded.",
        "supporting_values": {
            "month": "2025-09",
            "category": "Dining",
            "budget_amount": 300.0,
            "actual_amount": 400.0,
        },
    }

    unique = RecommendationEngine._deduplicate_recommendations(
        [duplicate, {**duplicate, "priority": "High"}]
    )

    assert len(unique) == 1
    assert unique[0]["priority"] == "High"


def test_positive_surplus_generates_saving_recommendation_without_goal():
    analysis = _analysis()
    analysis["analysis_month_totals"] = {
        "income": 1500.0,
        "expense": 1000.0,
        "net": 500.0,
    }
    analysis["goals"] = []

    recommendations = RecommendationEngine(
        today=date(2025, 9, 15)
    ).generate_recommendations(analysis)

    saving = [
        item
        for item in recommendations
        if item["recommendation_type"] == "saving_opportunity"
    ]
    assert len(saving) == 1
    assert saving[0]["supporting_values"]["monthly_surplus"] == 500.0


def test_future_goal_progress_uses_analysis_month_not_computer_date():
    analysis = _analysis()
    analysis["analysis_month_totals"] = {
        "income": 1500.0,
        "expense": 1000.0,
        "net": 500.0,
    }
    analysis["goals"] = [
        {
            "goal_id": "G-future",
            "goal_type": "Emergency Fund",
            "target_amount": 5000.0,
            "current_amount": 1000.0,
            "target_date": "2026-06-05",
            "priority": "High",
            "status": "In Progress",
        }
    ]

    recommendations = RecommendationEngine(
        today=date(2026, 11, 11)
    ).generate_recommendations(analysis)

    goal = next(
        item
        for item in recommendations
        if item["recommendation_type"] == "goal_progress"
    )
    values = goal["supporting_values"]
    assert goal["recommendation_title"] == "Plan contributions toward Emergency Fund"
    assert values["analysis_month"] == "2025-09"
    assert values["remaining_amount"] == 4000.0
    assert values["progress_percentage"] == 20.0
    assert values["months_remaining_including_current"] == 10
    assert values["required_monthly_contribution"] == 400.0


def test_goal_target_month_before_analysis_month_is_overdue():
    analysis = _analysis()
    analysis["goals"] = [
        {
            "goal_id": "G-past",
            "goal_type": "Travel",
            "target_amount": 2000.0,
            "current_amount": 500.0,
            "target_date": "2025-08-31",
            "priority": "Medium",
            "status": "In Progress",
        }
    ]

    recommendations = RecommendationEngine(
        today=date(2026, 11, 11)
    ).generate_recommendations(analysis)

    goal = next(
        item
        for item in recommendations
        if item["recommendation_type"] == "goal_progress"
    )
    assert goal["priority"] == "High"
    assert goal["supporting_values"]["remaining_amount"] == 1500.0
    assert goal["supporting_values"]["analysis_month"] == "2025-09"


def test_investment_budget_does_not_generate_reduction_advice():
    analysis = _analysis()
    analysis["budgets"] = [
        {
            "category": "Investment",
            "budget_amount": 300.0,
            "actual_amount": 400.0,
            "exceeded": 1,
        }
    ]
    analysis["category_trends"] = [
        {
            "category": "Investment",
            "latest_category_spending": 400.0,
            "historical_months_compared": 2,
            "historical_average_category_spending": 200.0,
        }
    ]

    recommendations = RecommendationEngine(
        today=date(2025, 9, 15)
    ).generate_recommendations(analysis)

    assert not any(
        item["recommendation_type"] == "expense_reduction"
        for item in recommendations
    )
    investment_budget = next(
        item
        for item in recommendations
        if item["recommendation_type"] == "budget_warning"
    )
    assert "not a suggestion to reduce investments" in investment_budget[
        "recommendation_description"
    ]

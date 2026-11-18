import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-service" / "recommendation"))

from engine import _assess_expense, _inr  # noqa: E402


def _history():
    rows = []
    for day in range(1, 29):
        rows.append({"transaction_date": f"2026-09-{day:02d}", "merchant": "Swiggy", "category": "Food", "amount": 400.0})
        if day % 4 == 0:
            rows.append({"transaction_date": f"2026-09-{day:02d}", "merchant": "Amazon", "category": "Shopping", "amount": 2000.0})
    return rows


def _assess(new_rows, *, model_flag=False, score=0.0, income=60000.0):
    frame = pd.DataFrame(_history() + new_rows)
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    frame = frame.sort_values("transaction_date", kind="stable")
    index = frame.index[-1]
    return _assess_expense(frame.loc[index], index, frame, income, model_flag, score)


def test_large_purchase_at_new_merchant_is_high_with_reasons():
    result = _assess([{"transaction_date": "2026-10-05", "merchant": "Croma", "category": "Shopping", "amount": 18990.0}])

    assert result["severity"] == "High"
    assert result["detected_by"] == "rules"
    assert result["amount_ratio"] > 9
    assert any("9.5× your usual Shopping purchase (₹2,000)" in reason for reason in result["reasons"])
    assert any("First payment to Croma" in reason for reason in result["reasons"])
    assert any("32% of your typical monthly income" in reason for reason in result["reasons"])


def test_repeated_identical_charge_is_flagged_as_possible_duplicate():
    result = _assess([
        {"transaction_date": "2026-10-03", "merchant": "Amazon", "category": "Shopping", "amount": 2499.0},
        {"transaction_date": "2026-10-04", "merchant": "Amazon", "category": "Shopping", "amount": 2499.0},
    ])

    assert "possible_duplicate" in result["signals"]
    assert any("1 day earlier: possible duplicate charge" in reason for reason in result["reasons"])
    assert result["severity"] in {"Medium", "High"}


def test_model_only_flag_is_low_severity_and_normal_spending_is_not_flagged():
    usual = [{"transaction_date": "2026-10-02", "merchant": "Swiggy", "category": "Food", "amount": 420.0}]

    assert _assess(usual) is None
    flagged = _assess(usual, model_flag=True, score=0.09)
    assert flagged["severity"] == "Low"
    assert flagged["detected_by"] == "model"


def test_rupees_use_indian_digit_grouping():
    assert _inr(125000) == "₹1,25,000"
    assert _inr(15990) == "₹15,990"
    assert _inr(999) == "₹999"
    assert _inr(12345678) == "₹1,23,45,678"


def test_model_only_flag_needs_a_score_unusual_for_this_user():
    usual = [{"transaction_date": "2026-10-02", "merchant": "Swiggy", "category": "Food", "amount": 420.0}]

    # Below the user's own threshold: an everyday purchase is not raised.
    assert _assess(usual, model_flag=True, score=0.04) is not None
    assert _assess_with_threshold(usual, score=0.04, threshold=0.08) is None
    assert _assess_with_threshold(usual, score=0.09, threshold=0.08)["detected_by"] == "model"
    # History checks still count the model's agreement whatever the threshold.
    large = [{"transaction_date": "2026-10-05", "merchant": "Croma", "category": "Shopping", "amount": 18990.0}]
    assert _assess_with_threshold(large, score=0.04, threshold=0.08)["detected_by"] == "model + rules"


def _assess_with_threshold(new_rows, *, score, threshold, income=60000.0):
    frame = pd.DataFrame(_history() + new_rows)
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    frame = frame.sort_values("transaction_date", kind="stable")
    index = frame.index[-1]
    return _assess_expense(frame.loc[index], index, frame, income, True, score, threshold)


def test_recommendations_rank_by_money_involved():
    from engine import _monthly_impact

    assert _monthly_impact({"recommendation_type": "overspending", "supporting_values": {"deficit": 12500.0}}) == 12500.0
    assert _monthly_impact({"recommendation_type": "budget_warning", "supporting_values": {"over_budget_amount": 900.0, "actual_amount": 5900.0}}) == 900.0
    assert _monthly_impact({"recommendation_type": "budget_warning", "supporting_values": {"actual_amount": 5900.0, "remaining_budget": 100.0}}) is None
    assert _monthly_impact({"recommendation_type": "goal_progress", "supporting_values": {"required_monthly_contribution": 8333.33}}) == 8333.33
    # A forecast above income is about the expected shortfall, not the whole forecast.
    assert _monthly_impact({"recommendation_type": "future_expense_warning", "supporting_values": {"predicted_expense": 70000.0, "monthly_income_reference": 65000.0, "recent_average_expense": 60000.0}}) == 5000.0
    assert _monthly_impact({"recommendation_type": "anomaly_review", "supporting_values": {"transactions": [{"amount": 1200.0}, {"amount": 800.0}]}}) == 2000.0
    assert _monthly_impact({"recommendation_type": "saving_opportunity", "supporting_values": {}}) is None


def test_regular_large_payment_like_rent_is_not_flagged_for_share_of_income():
    rent = [{"transaction_date": f"2026-0{month}-01", "merchant": "Landlord", "category": "Rent", "amount": 20000.0} for month in range(3, 9)]
    frame_rows = rent + [{"transaction_date": "2026-10-05", "merchant": "Landlord", "category": "Rent", "amount": 20000.0}]
    assert _assess(frame_rows, income=60000.0) is None
    # Twice the usual rent is unusual: the share of income is mentioned.
    doubled = rent + [{"transaction_date": "2026-10-05", "merchant": "Landlord", "category": "Rent", "amount": 40000.0}]
    assert "large_share_of_income" in _assess(doubled, income=60000.0)["signals"]

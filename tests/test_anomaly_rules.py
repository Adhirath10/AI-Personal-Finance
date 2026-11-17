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

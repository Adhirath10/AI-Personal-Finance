from pathlib import Path
import sys

import pandas as pd


ANOMALY_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "ai-service"
    / "anomaly_detection"
)
sys.path.insert(0, str(ANOMALY_SCRIPT_DIR))

from behavioral_experiment import engineer_behavioral_features


def _transactions(amounts, dates=None):
    if dates is None:
        dates = pd.date_range("2025-01-01", periods=len(amounts), freq="D")
    return pd.DataFrame(
        {
            "user_id": ["U1"] * len(amounts),
            "transaction_date": dates,
            "amount": amounts,
            "category": ["Food"] * len(amounts),
            "merchant": ["Cafe"] * len(amounts),
        }
    )


def test_behavior_features_use_only_strictly_prior_timestamps():
    data = _transactions(
        [10.0, 20.0, 30.0],
        dates=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"]),
    )

    features = engineer_behavioral_features(data)

    assert pd.isna(features.loc[0, "user_history_mean"])
    assert features.loc[0, "user_history_count"] == 0
    assert features.loc[1, "user_history_mean"] == 10.0
    assert features.loc[1, "user_history_count"] == 1
    assert features.loc[1, "merchant_familiar"] == 1
    assert features.loc[1, "category_history_mean"] == 10.0
    assert features.loc[1, "category_history_count"] == 1
    assert features.loc[2, "user_history_mean"] == 10.0
    assert features.loc[2, "user_history_count"] == 1
    assert features.loc[2, "merchant_history_count"] == 1


def test_future_transaction_changes_do_not_affect_earlier_features():
    original = _transactions([10.0, 20.0, 30.0])
    changed_future = _transactions([10.0, 2000.0, 3000.0])

    original_features = engineer_behavioral_features(original)
    changed_features = engineer_behavioral_features(changed_future)

    pd.testing.assert_series_equal(
        original_features.loc[0],
        changed_features.loc[0],
        check_names=False,
    )


def test_history_is_isolated_between_users():
    data = _transactions([10.0, 30.0])
    data.loc[1, "user_id"] = "U2"
    data.loc[1, "transaction_date"] = pd.Timestamp("2025-01-02")

    features = engineer_behavioral_features(data)

    assert features.loc[1, "user_history_count"] == 0
    assert features.loc[1, "merchant_familiar"] == 0

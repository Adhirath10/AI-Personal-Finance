import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ai_service.categorization.train import (
    prepare_text_feature,
    load_transaction_data,
)


def test_prepare_text_feature_handles_missing_values():
    text = prepare_text_feature("Swiggy food order", "Swiggy")
    assert "swiggy" in text.lower()
    assert "food" in text.lower()
    assert "order" in text.lower()

    text_empty = prepare_text_feature(None, "")
    assert isinstance(text_empty, str)
    assert len(text_empty) > 0


def test_load_transaction_data_rejects_missing_required_columns(tmp_path):
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE transactions (transaction_id TEXT, description TEXT, merchant TEXT)"
    )
    conn.execute(
        "INSERT INTO transactions (transaction_id, description, merchant) VALUES ('t1', 'Coffee', 'Cafe')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="Missing required columns"):
        load_transaction_data(db_path)


def test_load_transaction_data_returns_expected_columns(tmp_path):
    db_path = tmp_path / "finance.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE transactions (
            transaction_id TEXT,
            user_id TEXT,
            description TEXT,
            merchant TEXT,
            category TEXT,
            amount REAL,
            transaction_type TEXT,
            subcategory TEXT,
            is_anomaly INTEGER,
            anomaly_type TEXT,
            balance_after_transaction REAL,
            day_of_week INTEGER,
            month INTEGER,
            year INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO transactions VALUES ('T1','U1','Netflix payment','Netflix','Subscription',299.0,'Expense','Streaming',0,NULL,1000.0,1,1,2025)"
    )
    conn.execute(
        "INSERT INTO transactions VALUES ('T2','U1','Grocery basket','Local Store','Groceries',450.0,'Expense','Food',0,NULL,1200.0,2,1,2025)"
    )
    conn.commit()
    conn.close()

    df = load_transaction_data(db_path)
    assert {'description', 'merchant', 'category'}.issubset(df.columns)
    assert len(df) == 2
    assert 'text' in df.columns

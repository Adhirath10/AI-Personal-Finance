import sqlite3

import pytest

from ai_service.forecasting.train import build_monthly_expense_data, create_forecast_dataset


def test_build_monthly_expense_data_aggregates_expenses(tmp_path):
    db_path = tmp_path / "forecast.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE transactions (
            transaction_id TEXT,
            user_id TEXT,
            transaction_date TEXT,
            description TEXT,
            merchant TEXT,
            amount REAL,
            transaction_type TEXT,
            category TEXT,
            subcategory TEXT,
            payment_mode TEXT,
            account_type TEXT,
            is_recurring INTEGER,
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
        "INSERT INTO transactions VALUES ('t1','U1','2025-01-05','coffee','Cafe',120.0,'Expense','Food','Cafe','UPI','Savings',0,0,NULL,250.0,4,1,2025)"
    )
    conn.execute(
        "INSERT INTO transactions VALUES ('t2','U1','2025-01-10','salary','Employer',5000.0,'Income','Salary','Salary','Bank','Savings',0,0,NULL,5200.0,2,1,2025)"
    )
    conn.execute(
        "INSERT INTO transactions VALUES ('t3','U1','2025-02-08','rent','Landlord',1500.0,'Expense','Rent','Rent','Bank','Savings',1,0,NULL,2000.0,4,2,2025)"
    )
    conn.commit()
    conn.close()

    monthly = build_monthly_expense_data(db_path)
    assert list(monthly.columns) == ["user_id", "month", "total_expense"]
    assert monthly.loc[(monthly["user_id"] == "U1") & (monthly["month"] == "2025-01"), "total_expense"].iloc[0] == 120.0
    assert monthly.loc[(monthly["user_id"] == "U1") & (monthly["month"] == "2025-02"), "total_expense"].iloc[0] == 1500.0


def test_create_forecast_dataset_builds_lookback_features():
    df = __import__("pandas").DataFrame(
        {
            "user_id": ["U1", "U1", "U1", "U1", "U1"],
            "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"],
            "total_expense": [100.0, 120.0, 140.0, 160.0, 180.0],
        }
    )

    dataset = create_forecast_dataset(df)
    assert {"previous_1", "previous_2", "previous_3", "rolling_3", "rolling_6", "month_number", "spending_change", "target_expense"}.issubset(dataset.columns)
    assert dataset["target_expense"].notna().any()
    assert dataset[dataset["user_id"] == "U1"].shape[0] >= 2

import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("_finance_test_evaluate_hybrid", ROOT / "ai-service" / "anomaly_detection" / "evaluate_hybrid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _database(path):
    """Three users with a year of everyday spending, then labelled anomalies after the model's training end."""
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE transactions (transaction_id TEXT, user_id TEXT, transaction_date TEXT, description TEXT,
        merchant TEXT, amount REAL, transaction_type TEXT, category TEXT, subcategory TEXT, payment_mode TEXT,
        account_type TEXT, is_recurring INTEGER, is_anomaly INTEGER, anomaly_type TEXT,
        balance_after_transaction REAL, day_of_week INTEGER, month INTEGER, year INTEGER)"""
    )
    rows = []
    shops = [("Swiggy", "Food", 450.0), ("BigBasket", "Groceries", 1800.0), ("Uber", "Transport", 300.0)]
    for user in ("U001", "U002", "U003"):
        for month in range(1, 13):
            rows.append((f"{user}-S{month}", user, f"2025-{month:02d}-01", "Salary", "Employer", 80000.0, "Income", "Salary", "Salary", "Bank Transfer", "Savings", 1, 0, None, 90000.0, 2, month, 2025))
            for day in range(2, 28, 2):
                merchant, category, amount = shops[day % 3]
                amount = amount * (1 + (day % 5) / 20)
                rows.append((f"{user}-{month}-{day}", user, f"2025-{month:02d}-{day:02d}", f"{merchant} Transaction", merchant, amount, "Expense", category, category, "UPI", "Savings", 0, 0, None, 50000.0, day % 7, month, 2025))
    rows.append(("U001-BIG", "U001", "2025-11-15", "Croma Transaction", "Croma", 42000.0, "Expense", "Shopping", "Electronics", "Card", "Savings", 0, 1, "Unusual Amount", 9000.0, 5, 11, 2025))
    rows.append(("U002-DUP", "U002", "2025-11-12", "Swiggy Transaction", "Swiggy", 540.0, "Expense", "Food", "Food", "UPI", "Savings", 0, 1, "Duplicate Transaction", 9000.0, 3, 11, 2025))
    rows.append(("U002-DUP0", "U002", "2025-11-11", "Swiggy Transaction", "Swiggy", 540.0, "Expense", "Food", "Food", "UPI", "Savings", 0, 0, None, 9000.0, 2, 11, 2025))
    connection.executemany(f"INSERT INTO transactions VALUES ({','.join('?' * 18)})", rows)
    connection.commit()
    connection.close()


def test_hybrid_evaluation_reports_every_detector(tmp_path):
    model_dir = ROOT / "ai-service" / "models" / "anomaly_detection_behavioral_experiment"
    if not (model_dir / "behavioral_isolation_forest_models.joblib").is_file():
        pytest.skip("Saved anomaly model is not available.")
    module = _load()
    database = tmp_path / "finance.db"
    _database(database)

    summary = module.evaluate(database, model_dir)
    report = module.format_report(summary)

    assert summary["labelled_anomalies"] == 2
    assert set(summary["results"]) == set(module.VARIANTS)
    checks = summary["results"]["History checks only"]
    assert checks["recall"] == 1.0
    assert summary["results"]["Combined (all flags shown in the app)"]["recall"] == 1.0
    assert "Isolation Forest only" in report and "Duplicate Transaction" in report

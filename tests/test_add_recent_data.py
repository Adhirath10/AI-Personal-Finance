import importlib.util
import sqlite3
from datetime import date
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "add_recent_data.py"
spec = importlib.util.spec_from_file_location("_finance_add_recent_data", SCRIPT)
add_recent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(add_recent)


def _database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE users (user_id TEXT PRIMARY KEY);
        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY, user_id TEXT, transaction_date TEXT,
            description TEXT, merchant TEXT, amount REAL, transaction_type TEXT,
            category TEXT, subcategory TEXT, payment_mode TEXT, account_type TEXT,
            is_recurring INTEGER, is_anomaly INTEGER, anomaly_type TEXT,
            balance_after_transaction REAL, day_of_week INTEGER, month INTEGER, year INTEGER
        );
        CREATE TABLE monthly_budgets (
            budget_id TEXT PRIMARY KEY, user_id TEXT, budget_month TEXT, category TEXT,
            budget_amount REAL, actual_amount REAL, variance_amount REAL, exceeded INTEGER
        );
        INSERT INTO users VALUES ('U001'), ('U002');
        """
    )
    rows = []
    for month in range(1, 13):
        rows.append((f"I{month}", "U001", f"2025-{month:02d}-01", "Salary", "Employer", 50000, "Income", "Salary", 1))
        rows.append((f"R{month}", "U001", f"2025-{month:02d}-03", "Rent", "Landlord", 15000, "Expense", "Rent", 1))
        rows.append((f"F{month}", "U001", f"2025-{month:02d}-31" if month in (1, 3, 5, 7, 8, 10, 12) else f"2025-{month:02d}-28", "Dinner", "Cafe", 900, "Expense", "Food", 0))
    rows.append(("U2", "U002", "2025-12-05", "Other user", "Shop", 100, "Expense", "Food", 0))
    connection.executemany(
        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Other', 'UPI', 'Savings', ?, 0, NULL, 1000, 0, 1, 2025)",
        rows,
    )
    connection.execute("INSERT INTO monthly_budgets VALUES ('B1', 'U001', '2025-03', 'Food', 1000, 900, 100, 0)")
    connection.commit()
    return connection


def _count(connection, query, *params):
    return connection.execute(query, params).fetchone()[0]


def test_adds_complete_recent_months_and_can_be_rerun_and_removed(tmp_path):
    connection = _database(tmp_path / "finance.db")

    summary = add_recent.add_recent_data(connection, "U001", date(2026, 3, 31))
    connection.commit()

    assert summary["months"] == ["2026-01", "2026-02", "2026-03"]
    months = [row[0] for row in connection.execute(
        "SELECT DISTINCT substr(transaction_date, 1, 7) FROM transactions "
        "WHERE transaction_id LIKE 'DEMO-%' ORDER BY 1"
    )]
    assert months == ["2026-01", "2026-02", "2026-03"]
    # Every generated date exists (February has 28 days).
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE transaction_id LIKE 'DEMO-%' AND substr(transaction_date, 1, 7) = '2026-02' AND substr(transaction_date, 9, 2) > '28'") == 0
    # Rent is paid on the same early day every month.
    assert {row[0] for row in connection.execute("SELECT substr(transaction_date, 9, 2) FROM transactions WHERE transaction_id LIKE 'DEMO-%' AND merchant = 'Landlord'")} == {"03"}
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE merchant = 'Croma'") == 1
    assert _count(connection, "SELECT COUNT(*) FROM monthly_budgets WHERE budget_id LIKE 'DEMO-%'") == 3
    assert _count(connection, "SELECT actual_amount > 0 FROM monthly_budgets WHERE budget_id LIKE 'DEMO-%' AND budget_month = '2026-03'") == 1
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE user_id = 'U002'") == 1

    first_total = _count(connection, "SELECT COUNT(*) FROM transactions")
    add_recent.add_recent_data(connection, "U001", date(2026, 3, 31))
    assert _count(connection, "SELECT COUNT(*) FROM transactions") == first_total

    removed = add_recent.remove_generated_rows(connection, "U001")
    assert removed == {"transactions": summary["transactions"], "budgets": 3}
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE transaction_id LIKE 'DEMO-%'") == 0


def test_keeps_income_the_user_already_entered_and_skips_future_days(tmp_path):
    connection = _database(tmp_path / "finance.db")
    connection.execute(
        "INSERT INTO transactions VALUES ('TXN-1', 'U001', '2026-01-01', 'salary', 'company', 50000, "
        "'Income', 'Salary', 'Other', 'Bank', 'Savings', 0, 0, NULL, 60000, 3, 1, 2026)"
    )

    add_recent.add_recent_data(connection, "U001", date(2026, 2, 10))

    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE transaction_type = 'Income' AND substr(transaction_date, 1, 7) = '2026-01'") == 1
    assert _count(connection, "SELECT MAX(transaction_date) FROM transactions WHERE transaction_id LIKE 'DEMO-%'") <= "2026-02-10"


def test_generated_months_look_like_a_real_household_budget(tmp_path):
    connection = _database(tmp_path / "finance.db")
    # A history where spending exceeded income.
    connection.execute("UPDATE transactions SET amount = 12000 WHERE transaction_type = 'Income'")
    connection.execute(
        "INSERT INTO transactions VALUES ('S1', 'U001', '2025-12-05', 'Netflix', 'Netflix', 649, "
        "'Expense', 'Subscription', 'Other', 'Card', 'Savings', 1, 0, NULL, 1000, 0, 12, 2025)"
    )

    summary = add_recent.add_recent_data(connection, "U001", date(2026, 12, 31))

    for month, totals in summary["totals"].items():
        assert totals["income"] > totals["expenses"], month
        assert 0.10 <= 1 - totals["expenses"] / totals["income"] <= 0.45, month
    salaries = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT amount) FROM transactions "
        "WHERE transaction_id LIKE 'DEMO-%' AND transaction_type = 'Income'"
    ).fetchone()
    assert tuple(salaries) == (12, 1)
    assert {row[0] for row in connection.execute(
        "SELECT amount FROM transactions WHERE transaction_id LIKE 'DEMO-%' AND description = 'Netflix'"
    )} == {649}
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE merchant = 'Croma' AND transaction_date LIKE '2026-11%'") == 1


def test_a_test_transaction_in_a_later_month_does_not_block_the_gap(tmp_path):
    connection = _database(tmp_path / "finance.db")
    connection.execute(
        "INSERT INTO transactions VALUES ('TXN-TEST', 'U001', '2026-11-17', 'Tea', 'Chai Point', 523, "
        "'Expense', 'Food', 'Other', 'UPI', 'Savings', 0, 0, NULL, 900, 1, 11, 2026)"
    )

    summary = add_recent.add_recent_data(connection, "U001", date(2026, 10, 31))

    assert summary["months"][0] == "2026-01" and summary["months"][-1] == "2026-10"
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE transaction_id = 'TXN-TEST'") == 1


def test_fills_the_current_month_only_up_to_today(tmp_path):
    connection = _database(tmp_path / "finance.db")

    add_recent.add_recent_data(connection, "U001", date(2026, 3, 2), highlight=False)

    march = [row[0] for row in connection.execute(
        "SELECT transaction_date FROM transactions WHERE transaction_id LIKE 'DEMO-%' AND transaction_date >= '2026-03' ORDER BY 1"
    )]
    assert march and max(march) <= "2026-03-02"
    # Salary (1st) has arrived; rent (3rd) has not been paid yet.
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE transaction_id LIKE 'DEMO-%' AND transaction_date LIKE '2026-03-01%' AND merchant = 'Employer'") == 1
    assert _count(connection, "SELECT COUNT(*) FROM transactions WHERE transaction_id LIKE 'DEMO-%' AND transaction_date >= '2026-03' AND merchant = 'Landlord'") == 0


def test_repeated_purchases_are_spread_through_the_month():
    import random

    rows = [{"merchant": "BigBasket"}] * 5 + [{"merchant": "Netflix"}]
    days = add_recent._spread_days(random.Random(1), rows, 30)
    grocery_days = sorted(days[:5])
    assert all(later - earlier >= 3 for earlier, later in zip(grocery_days, grocery_days[1:]))
    assert all(1 <= day <= 30 for day in days)


def test_rent_day_is_early_in_the_month():
    expenses = [
        {"merchant": "Landlord", "category": "Rent", "is_recurring": 1, "transaction_date": f"2025-{month:02d}-{day:02d}"}
        for month, day in ((1, 20), (2, 24), (3, 22))
    ] + [{"merchant": "Tata Power", "category": "Bills", "is_recurring": 1, "transaction_date": f"2025-0{month}-13"} for month in (1, 2, 3)]
    days = add_recent.recurring_payment_days(expenses)
    assert days == {"landlord": add_recent.RENT_LATEST_DAY, "tata power": 13}

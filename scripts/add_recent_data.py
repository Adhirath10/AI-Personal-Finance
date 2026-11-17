"""Add recent months of transactions for one user so the dashboard is current.

The new months are generated from the user's own recorded history: each new
month copies the transactions and budgets of the same calendar month from the
most recent recorded year, with modest growth and random variation. Every
generated row's ID starts with ``DEMO-`` so it can be removed again.

Safety:
  * A backup of the database is written to data/database/backups/ before any change.
  * Re-running replaces the previously generated DEMO- rows (it never duplicates).
  * ``--remove`` deletes all generated DEMO- rows for the user.

By default it uses the same database and user as the app (DATABASE_PATH and
FINANCE_USER_ID from the environment or the project's .env file).

Examples (run from the project root):
  python scripts/add_recent_data.py --status
  python scripts/add_recent_data.py
  python scripts/add_recent_data.py --user-id U001 --through 2026-11-17
  python scripts/add_recent_data.py --remove
"""

from __future__ import annotations

import argparse
import calendar
import os
import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "database" / "personal_finance.db"
DEMO_PREFIX = "DEMO-"
EXPENSE_GROWTH_PER_YEAR = 1.05
INCOME_GROWTH_PER_YEAR = 1.06


def _app_setting(name: str) -> str | None:
    """Read a setting the way the backend does: environment first, then .env (any case)."""
    for key, value in os.environ.items():
        if key.upper() == name and value.strip():
            return value.strip()
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[7:].strip()
            value = raw.split(" #", 1)[0].strip().strip('"').strip("'")
            if key.upper() == name and value:
                return value
    return None


def _database_path(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
    else:
        configured = _app_setting("DATABASE_PATH")
        path = Path(configured).expanduser() if configured else DEFAULT_DATABASE
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise SystemExit(f"Database not found: {path}")
    return path


def _backup(database: Path) -> Path:
    backup_dir = database.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{database.stem}-{datetime.now():%Y%m%d-%H%M%S}{database.suffix}"
    with sqlite3.connect(database) as source, sqlite3.connect(target) as destination:
        source.backup(destination)
    return target


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def _month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def remove_generated_rows(connection: sqlite3.Connection, user_id: str) -> dict[str, int]:
    removed = {}
    removed["transactions"] = connection.execute(
        "DELETE FROM transactions WHERE user_id = ? AND transaction_id LIKE ?",
        (user_id, f"{DEMO_PREFIX}%"),
    ).rowcount
    if _has_table(connection, "monthly_budgets"):
        removed["budgets"] = connection.execute(
            "DELETE FROM monthly_budgets WHERE user_id = ? AND budget_id LIKE ?",
            (user_id, f"{DEMO_PREFIX}%"),
        ).rowcount
    return removed


def _insert(connection: sqlite3.Connection, table: str, row: dict) -> None:
    columns = list(row)
    placeholders = ", ".join("?" for _ in columns)
    quoted = ", ".join(f'"{column}"' for column in columns)
    connection.execute(
        f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
        [row[column] for column in columns],
    )


def add_recent_data(
    connection: sqlite3.Connection,
    user_id: str,
    through: date,
    *,
    highlight: bool = True,
) -> dict:
    connection.row_factory = sqlite3.Row
    if connection.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
        raise SystemExit(f"User {user_id!r} was not found in the users table.")

    remove_generated_rows(connection, user_id)
    transactions = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM transactions WHERE user_id = ? ORDER BY transaction_date, rowid",
            (user_id,),
        )
    ]
    expenses = [row for row in transactions if str(row["transaction_type"]).casefold() == "expense"]
    if not expenses:
        raise SystemExit(f"User {user_id!r} has no recorded expenses to base new months on.")

    by_month: dict[str, list[dict]] = {}
    for row in transactions:
        by_month.setdefault(str(row["transaction_date"])[:7], []).append(row)
    expense_months = sorted({str(row["transaction_date"])[:7] for row in expenses})

    first_new = _add_months(_month_start(_parse_date(expense_months[-1] + "-01")), 1)
    last_new = _month_start(through)
    if first_new > last_new:
        return {"months": [], "transactions": 0, "budgets": 0}

    columns = set(_columns(connection, "transactions"))
    balance = float(transactions[-1].get("balance_after_transaction") or 0.0)
    budget_rows: dict[str, list[dict]] = {}
    if _has_table(connection, "monthly_budgets"):
        for row in connection.execute(
            "SELECT * FROM monthly_budgets WHERE user_id = ? AND budget_id NOT LIKE ?",
            (user_id, f"{DEMO_PREFIX}%"),
        ):
            budget_rows.setdefault(str(row["budget_month"]), []).append(dict(row))

    summary = {"months": [], "transactions": 0, "budgets": 0}
    month = first_new
    while month <= last_new:
        key = _month_key(month)
        # Same calendar month from the most recent recorded year.
        template_key = max(
            (candidate for candidate in expense_months if candidate[5:] == key[5:]),
            default=expense_months[-1],
        )
        years_ahead = max(1, month.year - int(template_key[:4]))
        expense_growth = EXPENSE_GROWTH_PER_YEAR ** years_ahead
        income_growth = INCOME_GROWTH_PER_YEAR ** years_ahead
        rng = random.Random(f"{user_id}-{key}")
        has_income = any(
            str(row["transaction_type"]).casefold() == "income" for row in by_month.get(key, [])
        )
        last_day = calendar.monthrange(month.year, month.month)[1]
        generated = []
        for row in by_month[template_key]:
            kind = str(row["transaction_type"]).casefold()
            if kind == "income" and has_income:
                continue  # keep income the user already entered for this month
            day = min(_parse_date(row["transaction_date"]).day, last_day)
            new_date = month.replace(day=day)
            if new_date > through:
                continue
            amount = float(row["amount"])
            if kind == "income":
                amount *= income_growth
            elif row.get("is_recurring"):
                amount *= expense_growth
            else:
                amount *= expense_growth * rng.uniform(0.85, 1.18)
            generated.append((new_date, row, round(amount, 2)))

        if highlight and month == last_new and through >= _add_months(month, 1) - timedelta(days=1):
            shopping = [r for r in expenses if str(r["category"]).casefold() == "shopping"] or expenses
            typical = sorted(float(r["amount"]) for r in expenses if not r.get("is_recurring"))
            amount = max(20000.0, round(4 * typical[int(len(typical) * 0.95) - 1], -2))
            template = dict(shopping[-1])
            template.update({"description": "Laptop purchase", "merchant": "Croma", "is_recurring": 0})
            generated.append((month.replace(day=min(18, last_day)), template, amount))

        generated.sort(key=lambda item: item[0])
        for index, (new_date, template, amount) in enumerate(generated, start=1):
            row = dict(template)
            kind = str(row["transaction_type"]).casefold()
            balance += amount if kind == "income" else -amount
            row.update(
                {
                    "transaction_id": f"{DEMO_PREFIX}{user_id}-{key.replace('-', '')}-{index:03d}",
                    "transaction_date": new_date.isoformat() + str(template["transaction_date"])[10:],
                    "amount": amount,
                    "balance_after_transaction": round(balance, 2),
                }
            )
            for name, value in {
                "is_anomaly": 0,
                "anomaly_type": None,
                "day_of_week": new_date.weekday(),
                "month": new_date.month,
                "year": new_date.year,
            }.items():
                if name in columns:
                    row[name] = value
            _insert(connection, "transactions", {k: v for k, v in row.items() if k in columns})
        summary["transactions"] += len(generated)

        templates = budget_rows.get(template_key) or (
            budget_rows[max(budget_rows)] if budget_rows else []
        )
        for index, template in enumerate(templates, start=1):
            row = dict(template)
            row.update(
                {
                    "budget_id": f"{DEMO_PREFIX}BUD-{user_id}-{key.replace('-', '')}-{index:02d}",
                    "budget_month": key,
                    "budget_amount": round(float(template["budget_amount"]) * expense_growth, -2),
                }
            )
            _insert(connection, "monthly_budgets", row)
        if templates:
            _sync_budget_actuals(connection, user_id, key)
        summary["budgets"] += len(templates)
        summary["months"].append(key)
        month = _add_months(month, 1)
    return summary


def _sync_budget_actuals(connection: sqlite3.Connection, user_id: str, month: str) -> None:
    budget_columns = set(_columns(connection, "monthly_budgets"))
    spent = """COALESCE((
        SELECT SUM(t.amount) FROM transactions t
        WHERE t.user_id = monthly_budgets.user_id
          AND substr(t.transaction_date, 1, 7) = monthly_budgets.budget_month
          AND t.category = monthly_budgets.category
          AND lower(t.transaction_type) = 'expense'), 0)"""
    updates = []
    if "actual_amount" in budget_columns:
        updates.append(f"actual_amount = {spent}")
    if "variance_amount" in budget_columns:
        updates.append(f"variance_amount = budget_amount - {spent}")
    if "exceeded" in budget_columns:
        updates.append(f"exceeded = CASE WHEN {spent} > budget_amount THEN 1 ELSE 0 END")
    if updates:
        connection.execute(
            f"UPDATE monthly_budgets SET {', '.join(updates)} "
            "WHERE user_id = ? AND budget_month = ? AND budget_id LIKE ?",
            (user_id, month, f"{DEMO_PREFIX}%"),
        )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user-id", help="User to add data for (default: FINANCE_USER_ID from .env, else U001).")
    parser.add_argument("--database", help="SQLite database path (default: DATABASE_PATH or data/database/personal_finance.db).")
    parser.add_argument(
        "--through",
        type=date.fromisoformat,
        help="Last date to generate (YYYY-MM-DD). Default: the end of last month, so every new month is complete.",
    )
    parser.add_argument("--no-highlight", action="store_true", help="Do not add the one unusual purchase in the latest month.")
    parser.add_argument("--remove", action="store_true", help="Remove previously generated DEMO- rows and exit.")
    parser.add_argument("--status", action="store_true", help="Show which database and user the app uses and their latest data, then exit.")
    return parser.parse_args(argv)


def _latest(connection: sqlite3.Connection, user_id: str) -> tuple:
    return connection.execute(
        """
        SELECT MAX(transaction_date),
               MAX(CASE WHEN lower(transaction_type) = 'expense' THEN transaction_date END),
               SUM(CASE WHEN transaction_id LIKE ? THEN 1 ELSE 0 END),
               COUNT(*)
        FROM transactions WHERE user_id = ?
        """,
        (f"{DEMO_PREFIX}%", user_id),
    ).fetchone()


def print_status(database: Path, user_id: str, app_user: str | None) -> None:
    print(f"Database:         {database}")
    print(f"App user (FINANCE_USER_ID): {app_user or 'not set'}")
    with sqlite3.connect(database) as connection:
        if connection.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
            print(f"User {user_id}: NOT FOUND in the users table")
            return
        latest, latest_expense, generated, total = _latest(connection, user_id)
    print(f"User {user_id}: {total} transactions, {generated} generated (DEMO-)")
    print(f"  latest transaction: {latest}   latest expense: {latest_expense}")


def main(argv=None) -> int:
    args = parse_arguments(argv)
    database = _database_path(args.database)
    app_user = _app_setting("FINANCE_USER_ID")
    args.user_id = args.user_id or app_user or "U001"
    if args.status:
        print_status(database, args.user_id, app_user)
        return 0
    if app_user and app_user != args.user_id:
        print(
            f"Note: the app shows user {app_user} (FINANCE_USER_ID), not {args.user_id}. "
            f"Run without --user-id, or set FINANCE_USER_ID={args.user_id} in .env.",
        )
    through = args.through or (_month_start(date.today()) - timedelta(days=1))
    print(f"Database: {database}")
    print(f"User:     {args.user_id}")
    backup = _backup(database)
    print(f"Backup written to {backup}")
    connection = sqlite3.connect(database)
    try:
        if args.remove:
            removed = remove_generated_rows(connection, args.user_id)
            connection.commit()
            print(f"Removed generated rows for {args.user_id}: {removed}")
            return 0
        summary = add_recent_data(connection, args.user_id, through, highlight=not args.no_highlight)
        connection.commit()
    finally:
        connection.close()
    if not summary["months"]:
        print(f"{args.user_id} already has expenses through {through:%Y-%m}; nothing to add.")
        return 0
    print(
        f"Added {summary['transactions']} transactions and {summary['budgets']} budgets for "
        f"{args.user_id}: {summary['months'][0]} to {summary['months'][-1]} (through {through})."
    )
    with sqlite3.connect(database) as connection:
        latest, latest_expense, generated, _ = _latest(connection, args.user_id)
    print(f"Check: latest expense for {args.user_id} is now {latest_expense} ({generated} generated rows).")
    print("Refresh the browser (Ctrl+Shift+R); the month button should show the new latest month.")
    print(f"To undo: python scripts/add_recent_data.py --user-id {args.user_id} --remove")
    return 0


if __name__ == "__main__":
    sys.exit(main())

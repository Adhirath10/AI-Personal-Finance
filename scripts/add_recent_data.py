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
import math
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


# How much of a month's income goes on spending, by calendar month (1.0 = typical).
# Festival and year-end months run higher; February and June quieter.
SEASONAL_SPENDING = {1: 0.98, 2: 0.95, 3: 1.02, 4: 0.99, 5: 1.03, 6: 0.96,
                     7: 0.97, 8: 1.01, 9: 0.98, 10: 1.06, 11: 1.08, 12: 1.05}
MIN_SPENDING_SHARE = 0.62  # at most 38% saved
MAX_SPENDING_SHARE = 0.78  # at least 22% saved


def _kind(row: dict) -> str:
    return str(row["transaction_type"]).strip().casefold()


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value > 0)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _round_to(value: float, step: int) -> float:
    return float(max(step, round(value / step) * step))


def _profile_income(connection: sqlite3.Connection, user_id: str) -> float | None:
    if "monthly_income" not in _columns(connection, "users"):
        return None
    row = connection.execute("SELECT monthly_income FROM users WHERE user_id = ?", (user_id,)).fetchone()
    try:
        return float(row[0]) if row and row[0] is not None else None
    except (TypeError, ValueError):
        return None


def _plan_month(rng, month, through, template_rows, salary_template, *, income, spending_target,
                expense_growth, has_income, highlight_template=None, highlight_amount=0.0):
    """Return [(date, template_row, amount)] for one month that matches the budget plan."""
    last_day = calendar.monthrange(month.year, month.month)[1]

    def on_day(day: int) -> date:
        return month.replace(day=min(max(1, day), last_day))

    planned = []
    if not has_income:
        planned.append((on_day(_parse_date(salary_template["transaction_date"]).day), salary_template, income))

    expenses = [row for row in template_rows if _kind(row) == "expense"]
    recurring = [row for row in expenses if row.get("is_recurring")]
    everyday = [row for row in expenses if not row.get("is_recurring")]

    # Fixed costs keep their price; large ones (rent, EMIs) rise once a year.
    fixed = []
    for row in recurring:
        amount = float(row["amount"])
        fixed.append(_round_to(amount * expense_growth, 100) if amount >= 5000 else round(amount, 2))
    # If large fixed costs (rent) take more than half the plan, the household lives
    # somewhere cheaper: shrink only those, to round figures; subscriptions keep their price.
    large = sum(amount for amount in fixed if amount >= 5000)
    small = sum(amount for amount in fixed if amount < 5000)
    if large and large + small > 0.5 * spending_target:
        factor = max(0.5 * spending_target - small, 0.0) / large
        fixed = [_round_to(amount * factor, 500) if amount >= 5000 else amount for amount in fixed]
    fixed_total = sum(fixed)
    planned += [(on_day(_parse_date(row["transaction_date"]).day), row, amount) for row, amount in zip(recurring, fixed)]

    # Everyday spending fills the rest of the target: drop or repeat purchases so
    # individual amounts stay close to the user's usual ones. A big one-off purchase
    # means cutting back elsewhere that month.
    available = max(spending_target - fixed_total - highlight_amount, 0.1 * spending_target)
    rows = list(everyday)
    rng.shuffle(rows)
    total = lambda items: sum(float(row["amount"]) for row in items) or 1.0
    while rows and available / total(rows) < 0.8 and len(rows) > 4:
        rows.pop()
    pool = list(everyday) or rows
    while pool and available / total(rows) > 1.25 and len(rows) < 3 * max(1, len(everyday)):
        rows.append({**rng.choice(pool), "_day": rng.randint(1, last_day)})
    scale = available / total(rows) if rows else 0.0
    for row in rows:
        amount = max(10.0, float(round(float(row["amount"]) * scale * rng.uniform(0.88, 1.12))))
        day = row.get("_day") or _parse_date(row["transaction_date"]).day
        planned.append((on_day(day), row, amount))

    if highlight_template is not None:
        planned.append((on_day(18), highlight_template, highlight_amount))
    return [item for item in planned if item[0] <= through]


def add_recent_data(
    connection: sqlite3.Connection,
    user_id: str,
    through: date,
    *,
    highlight: bool = True,
    savings_rate: float | None = None,
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
    expenses = [row for row in transactions if _kind(row) == "expense"]
    if not expenses:
        raise SystemExit(f"User {user_id!r} has no recorded expenses to base new months on.")

    by_month: dict[str, list[dict]] = {}
    income_by_month: dict[str, float] = {}
    expense_by_month: dict[str, float] = {}
    for row in transactions:
        key = str(row["transaction_date"])[:7]
        by_month.setdefault(key, []).append(row)
        totals = income_by_month if _kind(row) == "income" else expense_by_month if _kind(row) == "expense" else None
        if totals is not None:
            totals[key] = totals.get(key, 0.0) + float(row["amount"])
    expense_months = sorted(expense_by_month)

    first_new = _add_months(_month_start(_parse_date(expense_months[-1] + "-01")), 1)
    last_new = _month_start(through)
    if first_new > last_new:
        return {"months": [], "transactions": 0, "budgets": 0, "totals": {}}

    # The household plan: the user's usual income, and a realistic share of it spent.
    typical_expense = _median(list(expense_by_month.values())) or 0.0
    base_income = max(
        [value for value in (_median(list(income_by_month.values())), _profile_income(connection, user_id)) if value]
        or [typical_expense / 0.7]
    )
    if savings_rate is not None:
        spending_share = min(max(1 - savings_rate, 0.2), 0.95)
    else:
        spending_share = min(max(typical_expense / base_income, MIN_SPENDING_SHARE), MAX_SPENDING_SHARE)

    income_rows = [row for row in transactions if _kind(row) == "income"]
    if income_rows:
        latest_income_month = max(str(row["transaction_date"])[:7] for row in income_rows)
        salary_template = dict(max(
            (row for row in income_rows if str(row["transaction_date"])[:7] == latest_income_month),
            key=lambda row: float(row["amount"]),
        ))
    else:
        salary_template = dict(expenses[-1])
        salary_template.update({
            "transaction_type": "Income", "category": "Salary", "subcategory": "Salary",
            "description": "Salary credit", "merchant": "Employer", "is_recurring": 1,
            "transaction_date": str(expenses[-1]["transaction_date"])[:8] + "01",
        })

    columns = set(_columns(connection, "transactions"))
    history_year = int(expense_months[-1][:4])
    highlight_month = _add_months(last_new, -1) if _add_months(last_new, -1) >= first_new else last_new

    plan: dict[str, list] = {}
    month = first_new
    while month <= last_new:
        key = _month_key(month)
        rng = random.Random(f"{user_id}-{key}")
        years_ahead = max(0, month.year - history_year)  # yearly raise / price rise
        income = _round_to(base_income * INCOME_GROWTH_PER_YEAR ** years_ahead, 100)
        spending_target = income * spending_share * SEASONAL_SPENDING[month.month] * rng.uniform(0.96, 1.04)
        # Same calendar month from the most recent recorded year.
        template_key = max((m for m in expense_months if m[5:] == key[5:]), default=expense_months[-1])
        highlight_template = None
        highlight_amount = 0.0
        if highlight and month == highlight_month:
            shopping = [r for r in expenses if str(r["category"]).casefold() == "shopping"] or expenses
            highlight_template = {**shopping[-1], "description": "Electronics purchase", "merchant": "Croma", "is_recurring": 0}
            # A store price (x,990) sized to the household's income.
            highlight_amount = min(_round_to(0.18 * income, 1000), 60000.0) - 10
        plan[key] = _plan_month(
            rng, month, through, by_month[template_key], salary_template,
            income=income, spending_target=spending_target,
            expense_growth=EXPENSE_GROWTH_PER_YEAR ** years_ahead,
            has_income=key in income_by_month,
            highlight_template=highlight_template, highlight_amount=highlight_amount,
        )
        month = _add_months(month, 1)

    balance = float(transactions[-1].get("balance_after_transaction") or 0.0)
    summary = {"months": list(plan), "transactions": 0, "budgets": 0, "totals": {}}
    category_spend: dict[str, list[float]] = {}
    for key, planned in plan.items():
        planned.sort(key=lambda item: item[0])
        month_totals = {"income": 0.0, "expenses": 0.0}
        month_categories: dict[str, float] = {}
        for index, (new_date, template, amount) in enumerate(planned, start=1):
            row = {name: value for name, value in template.items() if not name.startswith("_")}
            kind = _kind(row)
            balance += amount if kind == "income" else -amount
            month_totals["income" if kind == "income" else "expenses"] += amount
            if kind == "expense":
                month_categories[str(row["category"])] = month_categories.get(str(row["category"]), 0.0) + amount
            row.update({
                "transaction_id": f"{DEMO_PREFIX}{user_id}-{key.replace('-', '')}-{index:03d}",
                "transaction_date": new_date.isoformat() + str(template["transaction_date"])[10:],
                "amount": amount,
                "balance_after_transaction": round(balance, 2),
                "is_anomaly": 0,
                "anomaly_type": None,
                "day_of_week": new_date.weekday(),
                "month": new_date.month,
                "year": new_date.year,
            })
            _insert(connection, "transactions", {k: v for k, v in row.items() if k in columns})
        summary["transactions"] += len(planned)
        summary["totals"][key] = month_totals
        for category, amount in month_categories.items():
            category_spend.setdefault(category, []).append(amount)

    # Budgets: one steady monthly limit per category, a little above usual spending.
    if _has_table(connection, "monthly_budgets"):
        existing = [
            dict(row) for row in connection.execute(
                "SELECT * FROM monthly_budgets WHERE user_id = ? AND budget_id NOT LIKE ? ORDER BY budget_month",
                (user_id, f"{DEMO_PREFIX}%"),
            )
        ]
        categories = sorted({str(row["category"]) for row in existing} & set(category_spend)) or sorted(
            category_spend, key=lambda name: -sum(category_spend[name])
        )[:6]
        template = existing[-1] if existing else {}
        budget_columns = set(_columns(connection, "monthly_budgets"))
        for key in plan:
            for index, category in enumerate(categories, start=1):
                usual = (_median(category_spend[category]) or 0.0) * 1.1
                step = 500 if usual >= 2000 else 100
                limit = float(max(step, math.ceil(usual / step) * step))  # round up, like a real limit
                row = {**template, "user_id": user_id, "budget_month": key, "category": category,
                       "budget_id": f"{DEMO_PREFIX}BUD-{user_id}-{key.replace('-', '')}-{index:02d}",
                       "budget_amount": limit, "actual_amount": 0.0, "variance_amount": limit, "exceeded": 0}
                _insert(connection, "monthly_budgets", {k: v for k, v in row.items() if k in budget_columns})
                summary["budgets"] += 1
            _sync_budget_actuals(connection, user_id, key)
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
    parser.add_argument("--savings-rate", type=float, help="Average share of income saved, e.g. 0.3 for 30%%. Default: from the user's history, kept between 25%% and 45%%.")
    parser.add_argument("--no-highlight", action="store_true", help="Do not add the one unusual electronics purchase (in the month before the latest).")
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
        summary = add_recent_data(
            connection, args.user_id, through,
            highlight=not args.no_highlight, savings_rate=args.savings_rate,
        )
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
    print(f"\n{'Month':<9}{'Income':>12}{'Expenses':>12}{'Saved':>8}")
    for key, totals in summary["totals"].items():
        income, spent = totals["income"], totals["expenses"]
        saved = f"{(income - spent) / income:.0%}" if income else "-"
        print(f"{key:<9}{'Rs ' + format(round(income), ','):>12}{'Rs ' + format(round(spent), ','):>12}{saved:>8}")
    print()
    with sqlite3.connect(database) as connection:
        latest, latest_expense, generated, _ = _latest(connection, args.user_id)
    print(f"Check: latest expense for {args.user_id} is now {latest_expense} ({generated} generated rows).")
    print("Refresh the browser (Ctrl+Shift+R); the month button should show the new latest month.")
    print(f"To undo: python scripts/add_recent_data.py --user-id {args.user_id} --remove")
    return 0


if __name__ == "__main__":
    sys.exit(main())

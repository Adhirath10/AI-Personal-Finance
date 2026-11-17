"""Database-backed finance queries and CRUD operations."""

from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config.settings import settings
from ..schemas import (
    BudgetCreate,
    BudgetUpdate,
    GoalCreate,
    GoalUpdate,
    TransactionCreate,
    TransactionUpdate,
)


CHART_COLORS = [
    "#5578f5",
    "#46b998",
    "#f0a94a",
    "#a18cf4",
    "#ef7789",
    "#9aa7bd",
    "#36a3c7",
]

TRANSACTION_SORTS = {
    "date": "transaction_date",
    "amount": "amount",
    "merchant": "merchant",
    "category": "category",
}


def current_user_id(db: Session) -> str:
    user_id = (settings.finance_user_id or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=503,
            detail="Set FINANCE_USER_ID in the backend environment before requesting financial data.",
        )
    exists = db.execute(
        text("SELECT 1 FROM users WHERE user_id = :user_id"),
        {"user_id": user_id},
    ).scalar_one_or_none()
    if exists is None:
        raise HTTPException(status_code=404, detail="Configured finance user was not found.")
    return user_id


def analysis_period(db: Session, user_id: str, month: str | None = None) -> dict | None:
    """Describe the month the dashboard reports on.

    Without an explicit ``month`` this is the latest month with recorded
    expenses. A newer month that only holds income (for example a salary
    credited on the 1st) would otherwise show zero spending everywhere.
    """
    row = db.execute(
        text(
            """
            SELECT MAX(transaction_date) AS latest_transaction_date,
                   MAX(CASE WHEN lower(transaction_type) = 'expense'
                            THEN transaction_date END) AS latest_expense_date
            FROM transactions
            WHERE user_id = :user_id
            """
        ),
        {"user_id": user_id},
    ).mappings().one()
    latest_transaction_date = row["latest_transaction_date"]
    if not latest_transaction_date:
        return None
    latest_transaction_month = latest_transaction_date[:7]
    latest_expense_month = (
        row["latest_expense_date"][:7] if row["latest_expense_date"] else None
    )
    default_month = latest_expense_month or latest_transaction_month
    selected = month or default_month
    year, month_number = (int(value) for value in selected.split("-"))
    if month_number == 12:
        end = date(year + 1, 1, 1).isoformat()
    else:
        end = date(year, month_number + 1, 1).isoformat()
    return {
        "month": selected,
        "start_date": f"{selected}-01",
        "end_date_exclusive": end,
        "latest_transaction_date": latest_transaction_date,
        "latest_transaction_month": latest_transaction_month,
        "latest_expense_month": latest_expense_month,
        "default_month": default_month,
        "is_default": selected == default_month,
        "status": "historical" if selected < date.today().strftime("%Y-%m") else "current",
    }


def dashboard_periods(db: Session, user_id: str) -> dict:
    """List the months that have recorded transactions, newest first."""
    rows = db.execute(
        text(
            """
            SELECT substr(transaction_date, 1, 7) AS month,
              SUM(CASE WHEN lower(transaction_type) = 'income' THEN amount ELSE 0 END) AS income,
              SUM(CASE WHEN lower(transaction_type) = 'expense' THEN amount ELSE 0 END) AS expenses,
              COUNT(*) AS transaction_count
            FROM transactions
            WHERE user_id = :user_id
            GROUP BY substr(transaction_date, 1, 7)
            ORDER BY month DESC
            """
        ),
        {"user_id": user_id},
    ).mappings().all()
    period = analysis_period(db, user_id)
    return {
        "default_month": period["default_month"] if period else None,
        "items": [
            {
                "month": row["month"],
                "income": float(row["income"] or 0),
                "expenses": float(row["expenses"] or 0),
                "transaction_count": int(row["transaction_count"]),
            }
            for row in rows
        ],
    }


def _month_before(month: str) -> str:
    year, month_number = (int(value) for value in month.split("-"))
    if month_number == 1:
        return f"{year - 1:04d}-12"
    return f"{year:04d}-{month_number - 1:02d}"


def _period_totals(db: Session, user_id: str, month: str) -> dict:
    row = db.execute(
        text(
            """
            SELECT
              COALESCE(SUM(CASE WHEN lower(transaction_type) = 'income' THEN amount ELSE 0 END), 0) AS income,
              COALESCE(SUM(CASE WHEN lower(transaction_type) = 'expense' THEN amount ELSE 0 END), 0) AS expenses
            FROM transactions
            WHERE user_id = :user_id AND substr(transaction_date, 1, 7) = :month
            """
        ),
        {"user_id": user_id, "month": month},
    ).mappings().one()
    return {"income": float(row["income"]), "expenses": float(row["expenses"])}


def dashboard_summary(db: Session, user_id: str, month: str | None = None) -> dict:
    period = analysis_period(db, user_id, month)
    if period is None:
        return {
            "analysis_period": None,
            "income": 0.0,
            "expenses": 0.0,
            "savings": 0.0,
            "budget_utilization": None,
            "incomeChange": None,
            "expenseChange": None,
            "savingsChange": None,
        }

    month = period["month"]
    totals = _period_totals(db, user_id, month)
    previous = _period_totals(db, user_id, _month_before(month))
    budget_total = db.execute(
        text(
            """
            SELECT COALESCE(SUM(budget_amount), 0)
            FROM monthly_budgets
            WHERE user_id = :user_id AND budget_month = :month
            """
        ),
        {"user_id": user_id, "month": month},
    ).scalar_one()
    savings = totals["income"] - totals["expenses"]
    previous_savings = previous["income"] - previous["expenses"]

    def change(current: float, earlier: float) -> float | None:
        return round((current - earlier) / abs(earlier) * 100, 2) if earlier else None

    return {
        "analysis_period": period,
        "income": totals["income"],
        "expenses": totals["expenses"],
        "savings": savings,
        "budget_utilization": (
            round(totals["expenses"] / float(budget_total) * 100, 2)
            if budget_total
            else None
        ),
        "incomeChange": change(totals["income"], previous["income"]),
        "expenseChange": change(totals["expenses"], previous["expenses"]),
        "savingsChange": change(savings, previous_savings),
    }


def dashboard_monthly_trends(db: Session, user_id: str, months: int = 12) -> dict:
    period = analysis_period(db, user_id)
    rows = db.execute(
        text(
            """
            SELECT substr(transaction_date, 1, 7) AS month,
              SUM(CASE WHEN lower(transaction_type) = 'income' THEN amount ELSE 0 END) AS income,
              SUM(CASE WHEN lower(transaction_type) = 'expense' THEN amount ELSE 0 END) AS expenses
            FROM transactions
            WHERE user_id = :user_id
            GROUP BY substr(transaction_date, 1, 7)
            ORDER BY month DESC
            LIMIT :months
            """
        ),
        {"user_id": user_id, "months": months},
    ).mappings().all()
    items = [
        {
            "month": row["month"],
            "income": float(row["income"] or 0),
            "expenses": float(row["expenses"] or 0),
        }
        for row in reversed(rows)
    ]
    return {"analysis_period": period, "items": items}


def dashboard_category_breakdown(db: Session, user_id: str, month: str | None = None) -> dict:
    period = analysis_period(db, user_id, month)
    if period is None:
        return {"analysis_period": None, "items": []}
    rows = db.execute(
        text(
            """
            SELECT category AS name, SUM(amount) AS value
            FROM transactions
            WHERE user_id = :user_id
              AND substr(transaction_date, 1, 7) = :month
              AND lower(transaction_type) = 'expense'
            GROUP BY category
            ORDER BY value DESC, name ASC
            """
        ),
        {"user_id": user_id, "month": period["month"]},
    ).mappings().all()
    return {
        "analysis_period": period,
        "items": [
            {
                "name": row["name"],
                "value": float(row["value"]),
                "color": CHART_COLORS[index % len(CHART_COLORS)],
            }
            for index, row in enumerate(rows)
        ],
    }


def transaction_payload(row) -> dict:
    return {
        "id": row["transaction_id"],
        "transaction_id": row["transaction_id"],
        "description": row["description"],
        "merchant": row["merchant"],
        "date": row["transaction_date"],
        "transaction_date": row["transaction_date"],
        "category": row["category"],
        "amount": float(row["amount"]),
        "balance_after_transaction": float(row["balance_after_transaction"]),
        "type": row["transaction_type"],
        "transaction_type": row["transaction_type"],
        "status": "Recorded",
    }


def list_transactions(
    db: Session,
    user_id: str,
    *,
    limit: int = 7,
    offset: int = 0,
    search: str | None = None,
    category: str | None = None,
    transaction_type: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sort_by: str = "date",
    sort_order: str = "desc",
) -> dict:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date.")
    if sort_by not in TRANSACTION_SORTS:
        raise HTTPException(status_code=422, detail="Unsupported transaction sort field.")

    clauses = ["user_id = :user_id"]
    params: dict[str, object] = {"user_id": user_id, "limit": limit, "offset": offset}
    if search:
        clauses.append(
            "(lower(description) LIKE :search ESCAPE '\\' "
            "OR lower(merchant) LIKE :search ESCAPE '\\' "
            "OR lower(category) LIKE :search ESCAPE '\\' "
            "OR lower(transaction_id) LIKE :search ESCAPE '\\')"
        )
        escaped = search.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params["search"] = f"%{escaped}%"
    if category:
        clauses.append("category = :category")
        params["category"] = category
    if transaction_type:
        clauses.append("transaction_type = :transaction_type")
        params["transaction_type"] = transaction_type
    if start_date:
        clauses.append("transaction_date >= :start_date")
        params["start_date"] = start_date.isoformat()
    if end_date:
        clauses.append("transaction_date <= :end_date")
        params["end_date"] = end_date.isoformat()

    where = " AND ".join(clauses)
    total = db.execute(
        text(f"SELECT COUNT(*) FROM transactions WHERE {where}"),
        params,
    ).scalar_one()
    direction = "ASC" if sort_order.lower() == "asc" else "DESC"
    rows = db.execute(
        text(
            f"""
            SELECT transaction_id, description, merchant, transaction_date,
                   category, amount, balance_after_transaction, transaction_type
            FROM transactions
            WHERE {where}
            ORDER BY {TRANSACTION_SORTS[sort_by]} {direction}, transaction_id {direction}
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    ).mappings().all()
    return {
        "items": [transaction_payload(row) for row in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "analysis_period": analysis_period(db, user_id),
    }


def recent_transactions(db: Session, user_id: str, limit: int = 5) -> dict:
    return list_transactions(db, user_id, limit=limit, sort_by="date", sort_order="desc")


def activity_feed(db: Session, user_id: str, limit: int = 10) -> dict:
    result = recent_transactions(db, user_id, limit=limit)
    return {
        "items": result["items"],
        "total": result["total"],
        "as_of": datetime.now(timezone.utc).isoformat(),
        "analysis_period": result["analysis_period"],
    }


def get_transaction(db: Session, user_id: str, transaction_id: str) -> dict:
    row = db.execute(
        text(
            """
            SELECT transaction_id, description, merchant, transaction_date,
                   category, amount, balance_after_transaction, transaction_type
            FROM transactions
            WHERE transaction_id = :transaction_id AND user_id = :user_id
            """
        ),
        {"transaction_id": transaction_id, "user_id": user_id},
    ).mappings().one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction was not found.")
    return transaction_payload(row)


def transaction_categories(db: Session, user_id: str) -> list[str]:
    rows = db.execute(
        text(
            """
            SELECT DISTINCT category FROM transactions
            WHERE user_id = :user_id AND category IS NOT NULL AND category != ''
            ORDER BY category
            """
        ),
        {"user_id": user_id},
    ).scalars().all()
    return list(rows)


def _sync_budget_actuals(db: Session, user_id: str, months: set[str]) -> None:
    for month in months:
        db.execute(
            text(
                """
                UPDATE monthly_budgets
                SET actual_amount = COALESCE((
                      SELECT SUM(t.amount)
                      FROM transactions t
                      WHERE t.user_id = monthly_budgets.user_id
                        AND substr(t.transaction_date, 1, 7) = monthly_budgets.budget_month
                        AND t.category = monthly_budgets.category
                        AND lower(t.transaction_type) = 'expense'
                    ), 0),
                    variance_amount = budget_amount - COALESCE((
                      SELECT SUM(t.amount)
                      FROM transactions t
                      WHERE t.user_id = monthly_budgets.user_id
                        AND substr(t.transaction_date, 1, 7) = monthly_budgets.budget_month
                        AND t.category = monthly_budgets.category
                        AND lower(t.transaction_type) = 'expense'
                    ), 0),
                    exceeded = CASE WHEN COALESCE((
                      SELECT SUM(t.amount)
                      FROM transactions t
                      WHERE t.user_id = monthly_budgets.user_id
                        AND substr(t.transaction_date, 1, 7) = monthly_budgets.budget_month
                        AND t.category = monthly_budgets.category
                        AND lower(t.transaction_type) = 'expense'
                    ), 0) > budget_amount THEN 1 ELSE 0 END
                WHERE user_id = :user_id AND budget_month = :month
                """
            ),
            {"user_id": user_id, "month": month},
        )


def create_transaction(db: Session, user_id: str, payload: TransactionCreate) -> dict:
    txn_id = f"TXN-{uuid4().hex[:12].upper()}"
    txn_date = payload.date
    db.execute(
        text(
            """
            INSERT INTO transactions (
              transaction_id, user_id, transaction_date, description, merchant,
              amount, transaction_type, category, subcategory, payment_mode,
              account_type, is_recurring, is_anomaly, anomaly_type,
              balance_after_transaction, day_of_week, month, year
            ) VALUES (
              :id, :user_id, :date, :description, :merchant, :amount, :type,
              :category, :subcategory, :payment_mode, :account_type,
              :is_recurring, 0, NULL, :balance_after_transaction,
              :day_of_week, :month, :year
            )
            """
        ),
        {
            "id": txn_id,
            "user_id": user_id,
            "date": txn_date.isoformat(),
            "description": payload.description,
            "merchant": payload.merchant,
            "amount": payload.amount,
            "balance_after_transaction": payload.balance_after_transaction,
            "type": payload.type,
            "category": payload.category,
            "subcategory": payload.subcategory,
            "payment_mode": payload.payment_mode,
            "account_type": payload.account_type,
            "is_recurring": int(payload.is_recurring),
            "day_of_week": txn_date.weekday(),
            "month": txn_date.month,
            "year": txn_date.year,
        },
    )
    _sync_budget_actuals(db, user_id, {txn_date.strftime("%Y-%m")})
    db.commit()
    return get_transaction(db, user_id, txn_id)


def update_transaction(
    db: Session,
    user_id: str,
    transaction_id: str,
    payload: TransactionUpdate,
) -> dict:
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=422, detail="Provide at least one transaction field to update.")
    if (
        values.keys() & {"amount", "type", "date"}
        and "balance_after_transaction" not in values
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Include balance_after_transaction when changing amount, type, "
                "or date."
            ),
        )
    exists = db.execute(
        text(
            "SELECT transaction_date FROM transactions "
            "WHERE transaction_id = :id AND user_id = :user_id"
        ),
        {"id": transaction_id, "user_id": user_id},
    ).mappings().one_or_none()
    if exists is None:
        raise HTTPException(status_code=404, detail="Transaction was not found.")
    impacted_months = {exists["transaction_date"][:7]}

    mapping = {
        "description": "description",
        "merchant": "merchant",
        "date": "transaction_date",
        "category": "category",
        "amount": "amount",
        "balance_after_transaction": "balance_after_transaction",
        "type": "transaction_type",
        "subcategory": "subcategory",
        "payment_mode": "payment_mode",
        "account_type": "account_type",
        "is_recurring": "is_recurring",
    }
    assignments = []
    params: dict[str, object] = {"id": transaction_id, "user_id": user_id}
    for key, value in values.items():
        if key == "date":
            params["transaction_date"] = value.isoformat()
            impacted_months.add(value.strftime("%Y-%m"))
            params["day_of_week"] = value.weekday()
            params["month"] = value.month
            params["year"] = value.year
            assignments.extend(
                [
                    "transaction_date = :transaction_date",
                    "day_of_week = :day_of_week",
                    "month = :month",
                    "year = :year",
                ]
            )
        else:
            column = mapping[key]
            params[key] = int(value) if key == "is_recurring" else value
            assignments.append(f"{column} = :{key}")
    db.execute(
        text(
            f"UPDATE transactions SET {', '.join(assignments)} "
            "WHERE transaction_id = :id AND user_id = :user_id"
        ),
        params,
    )
    _sync_budget_actuals(db, user_id, impacted_months)
    db.commit()
    return get_transaction(db, user_id, transaction_id)


def delete_transaction(db: Session, user_id: str, transaction_id: str) -> None:
    row = db.execute(
        text(
            "SELECT transaction_date FROM transactions "
            "WHERE transaction_id = :id AND user_id = :user_id"
        ),
        {"id": transaction_id, "user_id": user_id},
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction was not found.")
    result = db.execute(
        text(
            "DELETE FROM transactions "
            "WHERE transaction_id = :id AND user_id = :user_id"
        ),
        {"id": transaction_id, "user_id": user_id},
    )
    if result.rowcount == 0:
        db.rollback()
        raise HTTPException(status_code=404, detail="Transaction was not found.")
    _sync_budget_actuals(db, user_id, {row.transaction_date[:7]})
    db.commit()


def _budget_item(row, spent: float) -> dict:
    amount = float(row["budget_amount"])
    return {
        "id": row["budget_id"],
        "category": row["category"],
        "amount": amount,
        "spent": spent,
        "budget_month": row["budget_month"],
    }


def list_budgets(db: Session, user_id: str) -> dict:
    month = db.execute(
        text("SELECT MAX(budget_month) FROM monthly_budgets WHERE user_id = :user_id"),
        {"user_id": user_id},
    ).scalar_one()
    if month is None:
        period = analysis_period(db, user_id)
        month = period["month"] if period else date.today().strftime("%Y-%m")
    rows = db.execute(
        text(
            """
            SELECT budget_id, category, budget_month, budget_amount
            FROM monthly_budgets
            WHERE user_id = :user_id AND budget_month = :month
            ORDER BY category
            """
        ),
        {"user_id": user_id, "month": month},
    ).mappings().all()
    spent_rows = db.execute(
        text(
            """
            SELECT category, SUM(amount) AS spent
            FROM transactions
            WHERE user_id = :user_id AND substr(transaction_date, 1, 7) = :month
              AND lower(transaction_type) = 'expense'
            GROUP BY category
            """
        ),
        {"user_id": user_id, "month": month},
    ).mappings().all()
    spent = {row["category"]: float(row["spent"]) for row in spent_rows}
    return {
        "items": [_budget_item(row, spent.get(row["category"], 0.0)) for row in rows],
        "budget_month": month,
        "analysis_period": analysis_period(db, user_id),
    }


def create_budget(db: Session, user_id: str, payload: BudgetCreate) -> dict:
    month = payload.month or date.today().strftime("%Y-%m")
    budget_id = f"BUD-{uuid4().hex[:12].upper()}"
    db.execute(
        text(
            """
            INSERT INTO monthly_budgets (
              budget_id, user_id, budget_month, category, budget_amount,
              actual_amount, variance_amount, exceeded
            ) VALUES (
              :id, :user_id, :month, :category, :amount, 0, :amount, 0
            )
            """
        ),
        {
            "id": budget_id,
            "user_id": user_id,
            "month": month,
            "category": payload.category,
            "amount": payload.amount,
        },
    )
    _sync_budget_actuals(db, user_id, {month})
    db.commit()
    row = db.execute(
        text(
            "SELECT budget_id, category, budget_month, budget_amount "
            "FROM monthly_budgets WHERE budget_id = :id AND user_id = :user_id"
        ),
        {"id": budget_id, "user_id": user_id},
    ).mappings().one()
    spent = db.execute(
        text(
            """
            SELECT COALESCE(SUM(amount), 0) FROM transactions
            WHERE user_id = :user_id AND category = :category
              AND substr(transaction_date, 1, 7) = :month
              AND lower(transaction_type) = 'expense'
            """
        ),
        {"user_id": user_id, "category": payload.category, "month": month},
    ).scalar_one()
    return _budget_item(row, float(spent))


def update_budget(db: Session, user_id: str, budget_id: str, payload: BudgetUpdate) -> dict:
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=422, detail="Provide at least one budget field to update.")
    previous = db.execute(
        text(
            "SELECT budget_month FROM monthly_budgets "
            "WHERE budget_id = :id AND user_id = :user_id"
        ),
        {"id": budget_id, "user_id": user_id},
    ).mappings().one_or_none()
    if previous is None:
        raise HTTPException(status_code=404, detail="Budget was not found.")
    impacted_months = {previous["budget_month"]}
    if "month" in values:
        impacted_months.add(values["month"])
    mapping = {"category": "category", "amount": "budget_amount", "month": "budget_month"}
    assignments = []
    params: dict[str, object] = {"id": budget_id, "user_id": user_id}
    for key, value in values.items():
        assignments.append(f"{mapping[key]} = :{key}")
        params[key] = value
    db.execute(
        text(
            f"UPDATE monthly_budgets SET {', '.join(assignments)} "
            "WHERE budget_id = :id AND user_id = :user_id"
        ),
        params,
    )
    _sync_budget_actuals(db, user_id, impacted_months)
    db.commit()
    row = db.execute(
        text(
            """
            SELECT budget_id, category, budget_month, budget_amount
            FROM monthly_budgets
            WHERE budget_id = :id AND user_id = :user_id
            """
        ),
        {"id": budget_id, "user_id": user_id},
    ).mappings().one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Budget was not found.")
    spent = db.execute(
        text(
            """
            SELECT COALESCE(SUM(amount), 0) FROM transactions
            WHERE user_id = :user_id AND category = :category
              AND substr(transaction_date, 1, 7) = :month
              AND lower(transaction_type) = 'expense'
            """
        ),
        {"user_id": user_id, "category": row["category"], "month": row["budget_month"]},
    ).scalar_one()
    return _budget_item(row, float(spent))


def delete_budget(db: Session, user_id: str, budget_id: str) -> None:
    result = db.execute(
        text("DELETE FROM monthly_budgets WHERE budget_id = :id AND user_id = :user_id"),
        {"id": budget_id, "user_id": user_id},
    )
    if result.rowcount == 0:
        db.rollback()
        raise HTTPException(status_code=404, detail="Budget was not found.")
    db.commit()


def _goal_item(row) -> dict:
    return {
        "id": row["goal_id"],
        "name": row["goal_type"],
        "target": float(row["target_amount"]),
        "current": float(row["current_amount"]),
        "date": row["target_date"],
        "priority": row["priority"],
        "status": row["status"],
    }


def list_goals(db: Session, user_id: str) -> dict:
    rows = db.execute(
        text(
            """
            SELECT goal_id, goal_type, target_amount, current_amount, target_date,
                   priority, status
            FROM financial_goals
            WHERE user_id = :user_id
            ORDER BY target_date, goal_id
            """
        ),
        {"user_id": user_id},
    ).mappings().all()
    return {"items": [_goal_item(row) for row in rows], "analysis_period": analysis_period(db, user_id)}


def create_goal(db: Session, user_id: str, payload: GoalCreate) -> dict:
    goal_id = f"GOAL-{uuid4().hex[:12].upper()}"
    db.execute(
        text(
            """
            INSERT INTO financial_goals (
              goal_id, user_id, goal_type, target_amount, current_amount,
              target_date, priority, status
            ) VALUES (
              :id, :user_id, :name, :target, :current, :date, :priority, :status
            )
            """
        ),
        {
            "id": goal_id,
            "user_id": user_id,
            "name": payload.name,
            "target": payload.target,
            "current": payload.current,
            "date": payload.date.isoformat(),
            "priority": payload.priority,
            "status": payload.status,
        },
    )
    db.commit()
    row = db.execute(
        text(
            """
            SELECT goal_id, goal_type, target_amount, current_amount, target_date,
                   priority, status
            FROM financial_goals
            WHERE goal_id = :id AND user_id = :user_id
            """
        ),
        {"id": goal_id, "user_id": user_id},
    ).mappings().one()
    return _goal_item(row)


def update_goal(db: Session, user_id: str, goal_id: str, payload: GoalUpdate) -> dict:
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=422, detail="Provide at least one goal field to update.")
    mapping = {
        "name": "goal_type",
        "target": "target_amount",
        "current": "current_amount",
        "date": "target_date",
        "priority": "priority",
        "status": "status",
    }
    assignments = []
    params: dict[str, object] = {"id": goal_id, "user_id": user_id}
    for key, value in values.items():
        assignments.append(f"{mapping[key]} = :{key}")
        params[key] = value.isoformat() if key == "date" else value
    result = db.execute(
        text(
            f"UPDATE financial_goals SET {', '.join(assignments)} "
            "WHERE goal_id = :id AND user_id = :user_id"
        ),
        params,
    )
    if result.rowcount == 0:
        db.rollback()
        raise HTTPException(status_code=404, detail="Financial goal was not found.")
    db.commit()
    row = db.execute(
        text(
            """
            SELECT goal_id, goal_type, target_amount, current_amount, target_date,
                   priority, status
            FROM financial_goals WHERE goal_id = :id AND user_id = :user_id
            """
        ),
        {"id": goal_id, "user_id": user_id},
    ).mappings().one()
    return _goal_item(row)


def delete_goal(db: Session, user_id: str, goal_id: str) -> None:
    result = db.execute(
        text("DELETE FROM financial_goals WHERE goal_id = :id AND user_id = :user_id"),
        {"id": goal_id, "user_id": user_id},
    )
    if result.rowcount == 0:
        db.rollback()
        raise HTTPException(status_code=404, detail="Financial goal was not found.")
    db.commit()

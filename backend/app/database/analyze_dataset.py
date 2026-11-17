"""Read-only analysis report for the existing personal finance SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "personal_finance.db"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_report.txt"


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier discovered from the database schema."""
    return '"' + identifier.replace('"', '""') + '"'


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Find a column case-insensitively, preferring the listed candidate order."""
    available = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    return None


def find_table(tables: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Find a table case-insensitively, preferring the listed candidate order."""
    available = {table.casefold(): table for table in tables}
    for candidate in candidates:
        if candidate.casefold() in available:
            return available[candidate.casefold()]
    return None


def format_value(value: object, limit: int = 120) -> str:
    if pd.isna(value):
        return "NULL"
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def append_table_overview(report: list[str], connection: sqlite3.Connection) -> list[str]:
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    report.append("1. DATABASE TABLES")
    report.append(f"Total tables: {len(tables)}")
    report.extend(f"- {table}" for table in tables)
    report.append("")

    report.append("2. TABLE ROW COUNTS, SCHEMA, AND FIRST 5 RECORDS")
    for table in tables:
        table_sql = quote_identifier(table)
        schema = connection.execute(f"PRAGMA table_info({table_sql})").fetchall()
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table_sql}").fetchone()[0]
        report.extend(["", f"TABLE: {table}", f"Rows: {row_count:,}", "Columns:"])
        for _, name, data_type, not_null, default, primary_key in schema:
            constraints = []
            if primary_key:
                constraints.append("PRIMARY KEY")
            if not_null:
                constraints.append("NOT NULL")
            suffix = f" ({', '.join(constraints)})" if constraints else ""
            report.append(f"- {name}: {data_type or 'unspecified'}{suffix}")

        sample = pd.read_sql_query(f"SELECT * FROM {table_sql} LIMIT 5", connection)
        report.append("First 5 records:")
        report.extend("  " + line for line in (sample.to_string(index=False).splitlines() or ["(none)"]))
    report.append("")
    return tables


def append_missing_and_duplicate_analysis(
    report: list[str], tables: list[str], frames: dict[str, pd.DataFrame]
) -> None:
    report.append("3. MISSING VALUES")
    for table in tables:
        frame = frames[table]
        report.append(f"{table}:")
        if frame.empty and len(frame.columns) == 0:
            report.append("  (no columns)")
            continue
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            percentage = missing / len(frame) * 100 if len(frame) else 0
            report.append(f"  - {column}: {missing:,} missing ({percentage:.2f}%)")

    report.extend(["", "4. DUPLICATE RECORDS"])
    total_duplicates = 0
    for table in tables:
        duplicate_count = int(frames[table].duplicated(keep=False).sum())
        total_duplicates += duplicate_count
        report.append(f"{table}: {duplicate_count:,} rows are part of a duplicate record group")
    report.append(f"Duplicate rows across all tables: {total_duplicates:,}")
    report.append("")


def append_transaction_analysis(report: list[str], frames: dict[str, pd.DataFrame], transaction_table: str | None) -> None:
    report.append("5. TRANSACTION ANALYSIS")
    if transaction_table is None:
        report.append("No transactions table was found; transaction metrics are unavailable.")
        report.append("")
        append_optional_field_analysis(report, frames, None)
        return

    transactions = frames[transaction_table].copy()
    columns = list(transactions.columns)
    user_column = find_column(columns, ["user_id", "userid", "user"])
    date_column = find_column(columns, ["date", "transaction_date", "trans_date", "created_at"])
    amount_column = find_column(columns, ["amount", "transaction_amount", "value"])
    type_column = find_column(columns, ["transaction_type", "type", "category_type", "income_expense"])
    category_column = find_column(columns, ["category", "transaction_category", "expense_category"])

    report.append(f"Transaction table: {transaction_table}")
    report.append(f"Total number of transactions: {len(transactions):,}")

    if user_column:
        users = transactions[user_column].dropna()
        report.append(f"Number of unique users: {users.nunique():,}")
        per_user = users.value_counts().sort_index()
        report.append("Transactions per user:")
        report.extend(f"  - {format_value(user)}: {count:,}" for user, count in per_user.items())
        report.append(f"Average transactions per user: {per_user.mean():,.2f}" if len(per_user) else "Average transactions per user: N/A")
    else:
        report.append("User metrics: unavailable (no user identifier column found)")

    if date_column:
        dates = pd.to_datetime(transactions[date_column], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates):
            report.append(f"Transaction date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
        else:
            report.append("Transaction date range: unavailable (no parseable dates)")
    else:
        dates = None
        report.append("Transaction date range: unavailable (no date column found)")

    if amount_column:
        amounts = pd.to_numeric(transactions[amount_column], errors="coerce").dropna()
        report.append(f"Minimum transaction amount: {amounts.min():,.2f}" if len(amounts) else "Minimum transaction amount: N/A")
        report.append(f"Maximum transaction amount: {amounts.max():,.2f}" if len(amounts) else "Maximum transaction amount: N/A")
        report.append(f"Average transaction amount: {amounts.mean():,.2f}" if len(amounts) else "Average transaction amount: N/A")
    else:
        report.append("Amount metrics: unavailable (no amount column found)")

    if type_column:
        type_values = transactions[type_column].astype("string").str.strip().str.casefold()
        income_mask = type_values.str.contains("income|credit|deposit", regex=True, na=False)
        expense_mask = type_values.str.contains("expense|debit|withdrawal|payment", regex=True, na=False)
        report.append(f"Number of income transactions: {int(income_mask.sum()):,}")
        report.append(f"Number of expense transactions: {int(expense_mask.sum()):,}")
        report.append("Transaction types:")
        report.extend(f"  - {format_value(value)}: {count:,}" for value, count in transactions[type_column].value_counts(dropna=False).items())
    else:
        income_mask = expense_mask = pd.Series(False, index=transactions.index)
        report.append("Income and expense transaction counts: unavailable (no transaction type column found)")

    if category_column:
        report.append(f"All unique transaction categories ({transactions[category_column].nunique(dropna=True):,}):")
        report.extend(f"  - {format_value(category)}: {count:,}" for category, count in transactions[category_column].value_counts(dropna=False).items())
    else:
        report.append("Transaction categories: unavailable (no category column found)")

    if date_column and dates is not None:
        dated = transactions.loc[dates.notna()].copy()
        dated["_month"] = dates[dates.notna()].dt.to_period("M").astype(str).values
        report.append("Monthly transaction counts:")
        report.extend(f"  - {month}: {count:,}" for month, count in dated.groupby("_month", sort=True).size().items())
        if amount_column:
            dated["_amount"] = pd.to_numeric(dated[amount_column], errors="coerce").fillna(0)
            report.append("Monthly income:")
            report.extend(f"  - {month}: {dated.loc[income_mask[dated.index],].groupby('_month')['_amount'].sum().get(month, 0):,.2f}" for month in sorted(dated["_month"].unique()))
            report.append("Monthly expenses:")
            report.extend(f"  - {month}: {dated.loc[expense_mask[dated.index],].groupby('_month')['_amount'].sum().get(month, 0):,.2f}" for month in sorted(dated["_month"].unique()))
    else:
        report.append("Monthly transaction analysis: unavailable (no parseable date column found)")

    report.append(f"Duplicate transaction rows: {int(transactions.duplicated(keep=False).sum()):,}")
    report.append("")
    append_optional_field_analysis(report, frames, transaction_table)


def append_optional_field_analysis(report: list[str], frames: dict[str, pd.DataFrame], transaction_table: str | None) -> None:
    report.append("6. OPTIONAL FINANCIAL FIELDS")
    groups = {
        "Recurring-payment related fields": ["recurring", "recurrence", "subscription", "frequency", "recurring_payment"],
        "Budget-related fields": ["budget", "budgeted", "budget_amount", "budget_limit"],
        "Savings-goal fields": ["saving", "savings_goal", "goal_amount", "target_amount"],
        "Anomaly-related fields or labels": ["anomal", "outlier", "fraud", "risk_score", "is_anomaly"],
    }
    for heading, terms in groups.items():
        matches = []
        for table, frame in frames.items():
            for column in frame.columns:
                if any(term in column.casefold() for term in terms):
                    matches.append(f"{table}.{column}")
        report.append(f"{heading}: {', '.join(matches) if matches else 'not available'}")
    report.append("")


def generate_report() -> str:
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    database_uri = f"file:{DATABASE_PATH.as_posix()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        report = [
            "PERSONAL FINANCE DATASET REPORT",
            "=" * 80,
            f"Database: {DATABASE_PATH}",
            "Access mode: SQLite read-only",
            "",
        ]
        tables = append_table_overview(report, connection)
        frames = {
            table: pd.read_sql_query(f"SELECT * FROM {quote_identifier(table)}", connection)
            for table in tables
        }
        append_missing_and_duplicate_analysis(report, tables, frames)
        transaction_table = find_table(tables, ["transactions", "transaction", "expenses", "expense"])
        append_transaction_analysis(report, frames, transaction_table)
        report.extend(["7. ANALYSIS COMPLETE", "No database records or schema were modified."])

    output = "\n".join(report) + "\n"
    REPORT_PATH.write_text(output, encoding="utf-8")
    return output


def main() -> None:
    print(generate_report())
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from datetime import date
from datetime import datetime

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.config.settings import settings
from backend.app.database.connection import get_db, verify_database_connection
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    database = tmp_path / "finance-test.db"
    test_engine = create_engine(
        f"sqlite:///{database.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    with test_engine.begin() as connection:
        connection.execute(text("CREATE TABLE users (user_id TEXT PRIMARY KEY)"))
        connection.execute(
            text(
                """
                CREATE TABLE transactions (
                  transaction_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                  transaction_date TEXT NOT NULL, description TEXT NOT NULL,
                  merchant TEXT NOT NULL, amount REAL NOT NULL,
                  transaction_type TEXT NOT NULL, category TEXT NOT NULL,
                  subcategory TEXT NOT NULL, payment_mode TEXT NOT NULL,
                  account_type TEXT NOT NULL, is_recurring INTEGER NOT NULL,
                  is_anomaly INTEGER NOT NULL, anomaly_type TEXT,
                  balance_after_transaction REAL NOT NULL, day_of_week INTEGER NOT NULL,
                  month INTEGER NOT NULL, year INTEGER NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE monthly_budgets (
                  budget_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                  budget_month TEXT NOT NULL, category TEXT NOT NULL,
                  budget_amount REAL NOT NULL, actual_amount REAL NOT NULL,
                  variance_amount REAL NOT NULL, exceeded INTEGER NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE financial_goals (
                  goal_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                  goal_type TEXT NOT NULL, target_amount REAL NOT NULL,
                  current_amount REAL NOT NULL, target_date TEXT NOT NULL,
                  priority TEXT NOT NULL, status TEXT NOT NULL
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO users (user_id) VALUES ('U001'), ('U002')")
        )
        connection.execute(
            text(
                """
                INSERT INTO transactions VALUES
                ('T-1','U001','2025-12-01','Salary','Employer',5000,'Income','Income','Other','Bank','Savings',0,0,NULL,5000,0,12,2025),
                ('T-2','U001','2025-12-03','Groceries','Market',1500,'Expense','Food','Other','Card','Checking',0,0,NULL,3500,2,12,2025),
                ('T-3','U002','2025-12-04','Private','Merchant',900,'Expense','Other','Other','Card','Checking',0,0,NULL,100,3,12,2025)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO monthly_budgets VALUES
                ('B-1','U001','2025-12','Food',2000,1500,500,0)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO financial_goals VALUES
                ('G-1','U001','Emergency fund',10000,3000,'2026-12-31','High','active')
                """
            )
        )

    factory = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(settings, "finance_user_id", "U001")
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    test_engine.dispose()


def test_database_dashboard_uses_recorded_period_and_user_scope(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["database"] == "connected"

    summary = client.get("/api/dashboard/summary")
    assert summary.status_code == 200
    assert summary.json()["analysis_period"]["month"] == "2025-12"
    assert summary.json()["analysis_period"]["status"] == "historical"
    assert summary.json()["income"] == 5000
    assert summary.json()["expenses"] == 1500
    assert summary.json()["budget_utilization"] == 75

    trends = client.get("/api/dashboard/monthly-trends").json()
    assert trends["items"] == [{"month": "2025-12", "income": 5000, "expenses": 1500}]
    categories = client.get("/api/dashboard/category-breakdown").json()
    assert categories["items"][0]["name"] == "Food"
    assert client.get("/api/dashboard/recent-transactions").json()["items"][0]["id"] == "T-2"
    assert client.get("/api/transactions/T-3").status_code == 404


def test_activity_feed_is_recent_scoped_and_timestamped(client):
    response = client.get("/api/dashboard/activity", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [item["id"] for item in payload["items"]] == ["T-2"]
    assert payload["analysis_period"]["latest_transaction_date"] == "2025-12-03"
    assert datetime.fromisoformat(payload["as_of"]).utcoffset() is not None
    assert client.get("/api/dashboard/activity", params={"limit": 51}).status_code == 422


def test_transaction_filters_and_crud_are_scoped(client):
    filtered = client.get(
        "/api/transactions",
        params={
            "search": "market",
            "category": "Food",
            "transaction_type": "Expense",
            "start_date": "2025-12-01",
            "end_date": "2025-12-31",
            "limit": 1,
            "offset": 0,
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["id"] == "T-2"
    assert client.put(
        "/api/transactions/T-2",
        json={"amount": 1600},
    ).status_code == 422

    created = client.post(
        "/api/transactions",
        json={
            "description": "Bus fare",
            "merchant": "Transit",
            "date": "2025-12-10",
            "category": "Food",
            "amount": 25,
            "balance_after_transaction": 3475,
            "type": "Expense",
        },
    )
    assert created.status_code == 201
    transaction_id = created.json()["id"]
    assert client.get("/api/budgets").json()["items"][0]["spent"] == 1525
    updated = client.put(
        f"/api/transactions/{transaction_id}",
        json={
            "amount": 30,
            "balance_after_transaction": 3470,
            "date": "2025-12-11",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["amount"] == 30
    assert updated.json()["date"] == "2025-12-11"
    assert client.get("/api/budgets").json()["items"][0]["spent"] == 1530
    assert client.delete(f"/api/transactions/{transaction_id}").json()["deleted"]
    assert client.get("/api/budgets").json()["items"][0]["spent"] == 1500
    assert client.get(f"/api/transactions/{transaction_id}").status_code == 404
    assert client.post(
        "/api/transactions",
        json={
            "description": "Invalid",
            "merchant": "Test",
            "date": date.today().isoformat(),
            "category": "Other",
            "amount": 0,
            "type": "Expense",
        },
    ).status_code == 422


def test_budgets_and_goals_crud(client):
    budgets = client.get("/api/budgets")
    assert budgets.status_code == 200
    assert budgets.json()["items"][0]["spent"] == 1500

    budget = client.post(
        "/api/budgets",
        json={"category": "Food", "amount": 500, "month": "2025-12"},
    )
    assert budget.status_code == 201
    assert budget.json()["spent"] == 1500
    budget_id = budget.json()["id"]
    assert client.put(
        f"/api/budgets/{budget_id}", json={"amount": 600}
    ).json()["amount"] == 600
    assert client.put(
        f"/api/budgets/{budget_id}", json={"amount": None}
    ).status_code == 422
    assert client.delete(f"/api/budgets/{budget_id}").json()["deleted"]

    goals = client.get("/api/goals")
    assert goals.status_code == 200
    assert goals.json()["items"][0]["name"] == "Emergency fund"
    goal = client.post(
        "/api/goals",
        json={
            "name": "Travel",
            "target": 20000,
            "current": 500,
            "date": "2027-01-01",
        },
    )
    assert goal.status_code == 201
    goal_id = goal.json()["id"]
    assert client.put(
        f"/api/goals/{goal_id}", json={"current": 1000}
    ).json()["current"] == 1000
    assert client.put(
        f"/api/goals/{goal_id}", json={"current": None}
    ).status_code == 422
    assert client.delete(f"/api/goals/{goal_id}").json()["deleted"]


def test_missing_database_path_is_not_created(tmp_path, monkeypatch):
    missing_database = tmp_path / "does-not-exist.db"
    monkeypatch.setattr(settings, "database_path", missing_database)

    with pytest.raises(HTTPException) as error:
        verify_database_connection()

    assert error.value.status_code == 503
    assert not missing_database.exists()

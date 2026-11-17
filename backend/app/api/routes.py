"""REST routes for dashboard, transactions, budgets, goals, and AI operations."""

from datetime import date
import logging

import httpx
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..config.settings import settings
from ..database.connection import get_db
from ..schemas import (
    BudgetCreate,
    BudgetUpdate,
    CategorizeRequest,
    ExplainRequest,
    GoalCreate,
    GoalUpdate,
    TransactionCreate,
    TransactionUpdate,
)
from ..services import finance


logger = logging.getLogger(__name__)
router = APIRouter()


def _user_id(db: Session = Depends(get_db)) -> str:
    return finance.current_user_id(db)


async def _ai_request(
    method: str,
    path: str,
    *,
    user_id: str | None = None,
    params: dict | None = None,
    payload: dict | None = None,
) -> dict:
    query = dict(params or {})
    if user_id is not None:
        query["user_id"] = user_id
    try:
        async with httpx.AsyncClient(
            base_url=settings.resolved_ai_service_url,
            timeout=httpx.Timeout(settings.ai_service_timeout_seconds),
        ) as client:
            response = await client.request(method, path, params=query, json=payload)
        if response.status_code >= 400:
            logger.warning("AI service returned HTTP %s for %s.", response.status_code, path)
            raise httpx.HTTPStatusError(
                "AI service request failed",
                request=response.request,
                response=response,
            )
        try:
            response_body = response.json()
        except ValueError as exc:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=502,
                detail="AI service returned an invalid response.",
            ) from exc
        if not isinstance(response_body, dict):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=502,
                detail="AI service returned an invalid response.",
            )
        return response_body
    except httpx.TimeoutException as exc:
        logger.warning("AI service timed out for %s.", path)
        from fastapi import HTTPException

        raise HTTPException(status_code=504, detail="AI service timed out.") from exc
    except httpx.HTTPStatusError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=(
                404 if exc.response.status_code == 404 else 502
            ),
            detail=(
                "AI service resource was not found."
                if exc.response.status_code == 404
                else "AI service returned an error."
            ),
        ) from exc
    except (httpx.RequestError, ValueError) as exc:
        logger.warning("AI service is unavailable for %s: %s", path, type(exc).__name__)
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="AI service is unavailable.") from exc


@router.get("/dashboard/summary")
def get_dashboard_summary(
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.dashboard_summary(db, user_id)


@router.get("/dashboard/monthly-trends")
def get_monthly_trends(
    months: int = Query(default=12, ge=1, le=60),
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.dashboard_monthly_trends(db, user_id, months)


@router.get("/dashboard/category-breakdown")
def get_category_breakdown(
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.dashboard_category_breakdown(db, user_id)


@router.get("/dashboard/recent-transactions")
def get_recent_transactions(
    limit: int = Query(default=5, ge=1, le=50),
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.recent_transactions(db, user_id, limit)


@router.get("/dashboard/activity")
def get_dashboard_activity(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.activity_feed(db, user_id, limit)


@router.get("/transactions")
def get_transactions(
    limit: int = Query(default=7, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=200),
    category: str | None = Query(default=None, max_length=100),
    transaction_type: str | None = Query(default=None, pattern="^(Income|Expense)$"),
    start_date: date | None = None,
    end_date: date | None = None,
    sort_by: str = Query(default="date", pattern="^(date|amount|merchant|category)$"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.list_transactions(
        db,
        user_id,
        limit=limit,
        offset=offset,
        search=search,
        category=category,
        transaction_type=transaction_type,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.get("/transactions/categories")
def get_transaction_categories(
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return {"items": finance.transaction_categories(db, user_id)}


@router.get("/transactions/{transaction_id}")
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.get_transaction(db, user_id, transaction_id)


@router.post("/transactions", status_code=201)
def post_transaction(
    payload: TransactionCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.create_transaction(db, user_id, payload)


@router.put("/transactions/{transaction_id}")
def put_transaction(
    transaction_id: str,
    payload: TransactionUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.update_transaction(db, user_id, transaction_id, payload)


@router.delete("/transactions/{transaction_id}")
def remove_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    finance.delete_transaction(db, user_id, transaction_id)
    return {"deleted": True, "id": transaction_id}


@router.get("/budgets")
def get_budgets(
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.list_budgets(db, user_id)


@router.post("/budgets", status_code=201)
def post_budget(
    payload: BudgetCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.create_budget(db, user_id, payload)


@router.put("/budgets/{budget_id}")
def put_budget(
    budget_id: str,
    payload: BudgetUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.update_budget(db, user_id, budget_id, payload)


@router.delete("/budgets/{budget_id}")
def remove_budget(
    budget_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    finance.delete_budget(db, user_id, budget_id)
    return {"deleted": True, "id": budget_id}


@router.get("/goals")
def get_goals(
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.list_goals(db, user_id)


@router.post("/goals", status_code=201)
def post_goal(
    payload: GoalCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.create_goal(db, user_id, payload)


@router.put("/goals/{goal_id}")
def put_goal(
    goal_id: str,
    payload: GoalUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    return finance.update_goal(db, user_id, goal_id, payload)


@router.delete("/goals/{goal_id}")
def remove_goal(
    goal_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(_user_id),
) -> dict:
    finance.delete_goal(db, user_id, goal_id)
    return {"deleted": True, "id": goal_id}


@router.post("/ai/categorize")
async def categorize(payload: CategorizeRequest) -> dict:
    return await _ai_request(
        "POST",
        "/api/categorize",
        payload=payload.model_dump(),
    )


@router.get("/ai/forecast")
async def get_forecast(user_id: str = Depends(_user_id)) -> dict:
    return await _ai_request("GET", "/api/forecast", user_id=user_id)


@router.get("/ai/anomalies")
async def get_anomalies(user_id: str = Depends(_user_id)) -> dict:
    return await _ai_request("GET", "/api/anomalies", user_id=user_id)


@router.get("/ai/recommendations")
async def get_recommendations(user_id: str = Depends(_user_id)) -> dict:
    return await _ai_request("GET", "/api/recommendations", user_id=user_id)


@router.get("/ai/insights")
async def get_insights(user_id: str = Depends(_user_id)) -> dict:
    return await _ai_request("GET", "/api/insights", user_id=user_id)


@router.get("/ai/explanations")
async def get_explanations(
    model: str = Query(default="forecast", pattern="^(forecast|categorization|anomaly)$"),
    transaction_id: str | None = Query(default=None, max_length=100),
    description: str | None = Query(default=None, max_length=500),
    merchant: str | None = Query(default=None, max_length=200),
    user_id: str = Depends(_user_id),
) -> dict:
    return await _ai_request(
        "GET",
        "/api/explanations",
        user_id=user_id,
        params={
            "model": model,
            "transaction_id": transaction_id,
            "description": description,
            "merchant": merchant,
        },
    )

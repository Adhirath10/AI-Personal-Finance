# FinAI API Documentation

The project uses the existing FastAPI backend as the only frontend-facing REST
API. It reads and writes the configured SQLite database; AI requests are
proxied to the separate `ai-service/` process. The frontend never opens SQLite
or loads model files.

## Services and startup

| Service | Default address | Start from the project root |
|---|---|---|
| FastAPI backend | `http://127.0.0.1:8000` | `python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000` |
| Python AI service | `http://127.0.0.1:8001` | `python -m uvicorn api:app --app-dir ai-service --host 127.0.0.1 --port 8001` |
| React + Vite frontend | `http://localhost:5173` | `npm --prefix frontend run dev` |

Create `.env` from `.env.example` and set `FINANCE_USER_ID` to an existing
`users.user_id` in the selected database. The API intentionally does not choose
a profile automatically or accept a user ID from the browser. The checked-in
dataset covers January–December 2025; responses include an `analysis_period`
and mark old records as historical.

Install backend packages from the root `requirements.txt`; install AI-service
packages from `ai-service/requirements.txt`. For the frontend, run
`npm --prefix frontend install`, copy `frontend/.env.example` to
`frontend/.env`, and set `VITE_API_BASE_URL` if the backend is not at its
default address.

## Environment variables

| Variable | Used by | Default / purpose |
|---|---|---|
| `DATABASE_PATH` | Backend and AI service | `data/database/personal_finance.db`; relative paths resolve from the project root. The database must already exist. |
| `FINANCE_USER_ID` | Backend | Required; scope all financial API and AI operations to one existing profile. |
| `API_HOST` | Backend | `127.0.0.1`; do not expose this unauthenticated single-profile API publicly. |
| `API_PORT` | Backend | `8000`. |
| `AI_SERVICE_URL` | Backend | `http://127.0.0.1:8001`. |
| `AI_SERVICE_TIMEOUT_SECONDS` | Backend | `120`; bounds model/SHAP operations that may load saved artifacts or compute explanations. Health checks use a shorter timeout. |
| `CORS_ORIGINS` | Backend | JSON list containing the local Vite origins (`localhost:5173` and `127.0.0.1:5173`). |
| `FINANCE_EXPLANATION_OUTPUT_DIR` | AI service | `data/processed/explainability`; generated explanation artifacts are written here, not into the source database. |
| `VITE_API_BASE_URL` | Frontend | `http://localhost:8000`; set in `frontend/.env`. |

The AI service loads the project-root `.env` as well. If a custom database
location is configured, use the same path for both services.

## Common responses and errors

Successful requests return JSON. Request validation errors use HTTP `422`;
unknown or out-of-profile transaction, budget, or goal IDs use `404` (including
transaction IDs supplied to SHAP explanations);
duplicate/conflicting database writes use `409`; unavailable database/service
errors use `503`; AI timeouts use `504`; and upstream AI errors use `502`.
FastAPI validation errors include a `detail` array; other errors use a
`detail` string. Database and AI-service failures are reported independently.

The backend health check reports the database and AI sidecar separately:

```json
{
  "status": "ok",
  "database": "connected",
  "ai_service": "connected"
}
```

`ai_service` may be `"unavailable"` while the database-backed API remains
healthy. `GET /` is a basic backend liveness check; `GET /api/health` verifies
database connectivity and checks the AI sidecar.

## Dashboard

All dashboard data is scoped to `FINANCE_USER_ID`.

| Method and route | Parameters | Response |
|---|---|---|
| `GET /api/dashboard/summary` | None | Latest recorded month, totals, savings, and budget utilization. |
| `GET /api/dashboard/monthly-trends` | `months` (1–60; default 12) | `{ "analysis_period": ..., "items": [{ "month", "income", "expenses" }] }` |
| `GET /api/dashboard/category-breakdown` | None | Latest-month expense totals grouped by category. |
| `GET /api/dashboard/recent-transactions` | `limit` (1–50; default 5) | `{ "analysis_period": ..., "items": [...] }` |
| `GET /api/dashboard/activity` | `limit` (1–50; default 10) | Latest user-scoped transactions with `total`, `analysis_period`, and an ISO-8601 UTC `as_of` refresh timestamp. |

The dashboard polls this activity feed and the related summary, trends,
category breakdown, and budgets every 30 seconds while the browser tab is
visible. Returning to the tab also triggers a refresh. This reflects updates
written to the configured database; it does not connect to a bank or fetch
transactions from an external financial institution.

Summary response shape:

```json
{
  "analysis_period": {
    "month": "YYYY-MM",
    "start_date": "YYYY-MM-01",
    "end_date_exclusive": "YYYY-MM-01",
    "latest_transaction_date": "YYYY-MM-DD",
    "status": "historical"
  },
  "income": 0.0,
  "expenses": 0.0,
  "savings": 0.0,
  "budget_utilization": null,
  "incomeChange": null,
  "expenseChange": null,
  "savingsChange": null
}
```

Numeric values above describe the response shape only. `null` change values
mean no comparable prior-month baseline or no applicable budget.

## Transactions

| Method and route | Parameters / body | Response |
|---|---|---|
| `GET /api/transactions` | `limit` (1–200), `offset` (0+), `search`, `category`, `transaction_type` (`Income` or `Expense`), `start_date`, `end_date`, `sort_by` (`date`, `amount`, `merchant`, `category`), `sort_order` (`asc`, `desc`) | `{ "items": [...], "total": n, "limit": n, "offset": n, "analysis_period": ... }` |
| `GET /api/transactions/categories` | None | `{ "items": ["category", ...] }` |
| `GET /api/transactions/{id}` | Path transaction ID | One transaction or `404`. |
| `POST /api/transactions` | Required: `description`, `merchant`, ISO `date`, `category`, positive `amount`, `balance_after_transaction`, `type`. Optional: `subcategory`, `payment_mode`, `account_type`, `is_recurring`. | Created transaction (`201`). |
| `PUT /api/transactions/{id}` | Any non-empty subset of the create fields. Include `balance_after_transaction` when changing `amount`, `type`, or `date`. | Updated transaction. A supplied date also refreshes stored date-derived columns. |
| `DELETE /api/transactions/{id}` | Path transaction ID | `{ "deleted": true, "id": "..." }` |

Transaction response fields include `id`, `transaction_id`, `description`,
`merchant`, `date`, `transaction_date`, `category`, `amount`,
`balance_after_transaction`, `type`, `transaction_type`, and `status`. The
explicit balance field avoids inventing a placeholder for a model feature.
Search text is escaped and SQL values are parameterized; only sort columns from
the allowlist are interpolated.

## Budgets

| Method and route | Parameters / body | Response |
|---|---|---|
| `GET /api/budgets` | None | `{ "items": [...], "budget_month": "YYYY-MM", "analysis_period": ... }` |
| `POST /api/budgets` | `{ "category": "...", "amount": 1000, "month": "YYYY-MM" }`; `month` is optional and must be valid `YYYY-MM`. | Created budget (`201`). |
| `PUT /api/budgets/{id}` | Non-empty subset of `category`, positive `amount`, and `month`. | Updated budget. |
| `DELETE /api/budgets/{id}` | Path budget ID | `{ "deleted": true, "id": "..." }` |

Each budget item contains `id`, `category`, `amount`, `spent`, and
`budget_month`. `spent` is derived from expense transactions for that
user/category/month. Transaction create/update/delete operations also
recalculate the affected month's stored budget actuals, variance, and exceeded
flag so recommendation inputs remain consistent with the transaction ledger.

## Financial goals

| Method and route | Parameters / body | Response |
|---|---|---|
| `GET /api/goals` | None | `{ "items": [...], "analysis_period": ... }` |
| `POST /api/goals` | Required: `name`, positive `target`, ISO `date`. Optional: non-negative `current`, `priority`, `status`. | Created goal (`201`). |
| `PUT /api/goals/{id}` | Non-empty subset of the create fields. | Updated goal. |
| `DELETE /api/goals/{id}` | Path goal ID | `{ "deleted": true, "id": "..." }` |

Each goal item contains `id`, `name`, `target`, `current`, `date`, `priority`,
and `status`.

## AI operations

The backend proxies AI requests to the configured sidecar. The sidecar uses
the existing saved artifacts; it does not train models at startup. `user_id`
is derived from backend configuration and is not accepted from the frontend.

| Method and route | Parameters / body | Result |
|---|---|---|
| `POST /api/ai/categorize` | `{ "items": [{ "id": "...", "description": "...", "merchant": "..." }] }`; 1–100 items. | `status` and model-returned category/confidence items. Suggestions are not persisted to transaction rows. |
| `GET /api/ai/forecast` | None | Saved Random Forest status, forecast period and metrics when available. A stale prediction has its amount withheld. |
| `GET /api/ai/anomalies` | None | Eligible post-training-cutoff transaction scores, counts, review statuses, and the notice that unusual does not mean fraudulent. Stored anomaly labels are not model input features. |
| `GET /api/ai/recommendations` | None | Rule-based recommendations computed from the selected profile's recorded data and available saved model outputs. |
| `GET /api/ai/insights` | None | Category spending comparisons from the recorded analysis period. |
| `GET /api/ai/explanations` | Optional `model` (`forecast`, `categorization`, `anomaly`), `transaction_id`, `description`, `merchant` | Real SHAP outputs only when the corresponding model, data, and explanation are available. |

AI results return explicit `status` values such as `available`, `stale`,
`unavailable`, `insufficient_history`, or `no_out_of_sample_rows`. A stale
forecast is never presented as a current prediction. Anomaly scores are
statistical outlier scores, not fraud probabilities. The explanations endpoint
may write plots/JSON to `FINANCE_EXPLANATION_OUTPUT_DIR`; it never modifies the
SQLite source data or saved model artifacts.

## Data safety and current limitations

- The API reads and writes the existing SQLite file and does not create it if
  the configured path is missing.
- Every financial query and mutation is scoped to the configured profile.
- This local single-profile API does not provide authentication. Keep both
  backend and AI service bound to loopback unless authentication and production
  deployment controls are added.
- The database is historical synthetic data through December 2025. It is not
  treated as current financial activity.
- SHAP responses are omitted when model artifacts or compatible explanation
  inputs are unavailable. Explanation values describe model attributions, not
  causal effects.

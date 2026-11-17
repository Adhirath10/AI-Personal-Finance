# Personalized Financial Recommendation Engine

The recommendation engine combines read-only SQLite analysis with the existing
transaction categorization, Random Forest expense forecasting, and behavioral
Isolation Forest artifacts. It does not retrain models or write to the source
database.

Run commands from the project root:

```powershell
python ai-service\recommendation\analyze.py --user-id U001
python ai-service\recommendation\generate.py --user-id U001
python ai-service\recommendation\evaluate.py
python ai-service\recommendation\generate.py --user-id U001 --analysis-month 2025-11
```

`analyze.py` emits the user's financial snapshot, data coverage, and status of
each optional model. `generate.py` emits an individual user's prioritized
recommendations in JSON. `evaluate.py` summarizes recommendation-type coverage
and priorities across users; it is an operational coverage report, not a measure of
recommendation effectiveness because no verified outcome labels are available.
The optional `--analysis-month YYYY-MM` selects a period explicitly. Without it,
the latest available transaction month is used. Transactions after the selected
month are excluded from trend, category, anomaly, and forecast inputs; budgets
are selected for the same month. Output includes the selected month, latest
available data month, and whether the selected period is historical or current.
The engine never describes the latest stored month as current financial status
without showing this period information.

## Financial analysis and recommendations

The engine analyzes recorded monthly income and expenses, category-level
expense totals, historical trends, budget actuals/utilization, and in-progress
financial goals. It only creates recommendations supported by the rows and
amounts present. Rule thresholds are explicit in `RecommendationEngine`:
category trends are compared with available prior three-month history using a
10% default threshold, and budget warnings start at 80% utilization. An
overspending alert requires positive recorded income and expense exceeding
income. Investment categories are separated from tracked discretionary
categories; investment trends do not produce spending-reduction suggestions,
and investment budget alerts are phrased as allocation checks. Category-level
budget alerts suppress overlapping category trend suggestions; overall trend
advice is emitted only when no more specific budget or category advice applies.
Insufficient history or invalid values suppress the affected rule and are
reported or logged.

Every recommendation includes `user_id`, `recommendation_title`,
`recommendation_description`, `recommendation_type`, priority (`High`,
`Medium`, or `Low`), reason, and `supporting_values`. Duplicate recommendations
for the same user, type, period, and supported category/goal/transactions are
collapsed, keeping the highest priority.

## Existing AI model outputs

- **Transaction categorization:** the saved TF-IDF and Logistic Regression
  artifacts are used only when a transaction has no usable category. Existing
  stored categories are never overwritten.
- **Expense forecasting:** the saved Random Forest model and feature-column
  artifact are loaded without retraining. The engine verifies that the model is
  fitted and its feature metadata is usable, then builds a lagged feature row
  from transactions through the selected month. A future-expense recommendation
  is generated only when the forecast target month is later than the current
  month. Stale, unavailable, or insufficient forecasts are not used; the JSON
  output reports a `recommendation_skips` reason for the forecast recommendation.
- **Anomaly detection:** the saved behavioral Isolation Forest experiment is
  loaded without retraining. Its training-end metadata is used to score only
  later transactions through the selected analysis month; its behavioral
  feature builder uses preceding transactions. A flagged transaction is
  presented only as a statistical outlier for review, not confirmed fraud.

Financial-goal deadlines are compared with the selected analysis month, not the
machine's current date. Goal recommendations include the stored target/current
amount, calculated remaining amount and progress percentage, target date, and
analysis month. The recommendation count in generated JSON is the count of the
final deduplicated recommendation list.

The existing `is_anomaly` and `anomaly_type` columns are not loaded by the
recommendation engine. Stored anomaly-label provenance has not been verified.

## Read-only behavior and missing data

SQLite is opened with `mode=ro`. The engine never creates tables or writes
records. The core transaction table is required. Missing optional tables or
model artifacts are logged and their related recommendations are omitted.
Invalid transaction dates/amounts are excluded from aggregates and counted in
`data_quality`; invalid budgets and insufficient histories do not produce
fabricated values. Model-loading errors are surfaced with actionable messages.

Model artifact directories default to:

- `ai-service\models\categorization`
- `ai-service\models\forecasting`
- `ai-service\models\anomaly_detection_behavioral_experiment`

They can be overridden when constructing `RecommendationEngine`.

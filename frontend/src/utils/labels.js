// Plain-language names for values the AI service returns.

const FIELD_LABELS = {
  month: "Month",
  category: "Category",
  budget_amount: "Budget",
  actual_amount: "Spent",
  over_budget_amount: "Over by",
  remaining_budget: "Left",
  utilization: "Used",
  latest_month_spending: "This month",
  historical_monthly_average: "Usual month",
  latest_category_spending: "This month",
  historical_average_category_spending: "Usual month",
  increase_amount: "Increase",
  increase_rate: "Increase",
  latest_discretionary_expense: "Non-essentials this month",
  historical_average_discretionary_expense: "Usual non-essentials",
  forecast_month: "Forecast for",
  predicted_expense: "Predicted spending",
  monthly_income_reference: "Monthly income",
  recent_average_expense: "Recent average spending",
  income: "Income",
  expense: "Spending",
  deficit: "Shortfall",
  monthly_surplus: "Left over each month",
  latest_monthly_surplus: "Left over this month",
  expense_to_income_ratio: "Spending vs income",
  investment_expense: "Investments",
  goal_type: "Goal",
  target_amount: "Target",
  current_amount: "Saved so far",
  remaining_amount: "Still needed",
  progress_rate: "Progress",
  progress_percentage: "Progress",
  target_date: "Target date",
  required_monthly_contribution: "Needed per month",
  remaining_months_including_current: "Months left",
  flagged_transaction_count: "Transactions to review",
  high_severity_count: "High severity",
  medium_severity_count: "Medium severity",
};

// Internal bookkeeping that means nothing to a reader.
const HIDDEN_FIELDS = new Set([
  "goal_id", "label_provenance", "model_training_end", "forecast_model_history_months",
  "historical_months_compared", "transactions", "analysis_month", "priority", "interpretation",
  "investment_expense_excluded", "available", "user_id", "progress_percentage",
]);

export function fieldLabel(name) {
  return FIELD_LABELS[name] || name.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase());
}

export function visibleFields(values = {}) {
  return Object.entries(values).filter(([name, value]) => !HIDDEN_FIELDS.has(name) && value !== null && value !== undefined && value !== "");
}

const FEATURE_LABELS = {
  // Expense forecast (the latest recorded month is the input month)
  previous_1: "Spending 1 month earlier",
  previous_2: "Spending 2 months earlier",
  previous_3: "Spending 3 months earlier",
  rolling_3: "Average of the last 3 months",
  rolling_6: "Average of the last 6 months",
  month_number: "Time of year",
  spending_change: "Change from the month before",
  latest_vs_avg6: "Latest month vs your 6-month average",
  prev1_vs_avg6: "1 month earlier vs your 6-month average",
  prev2_vs_avg6: "2 months earlier vs your 6-month average",
  prev3_vs_avg6: "3 months earlier vs your 6-month average",
  avg3_vs_avg6: "Last 3 months vs your 6-month average",
  history_months: "Months of history",
  safety_limit: "Safety limit (0.6×–1.6× of your average)",
  average_fallback: "Recent 3-month average used (it beat the model in testing)",
  // Anomaly model
  log_amount: "Amount",
  amount: "Amount",
  balance_after_transaction: "Balance after the payment",
  is_recurring: "Recurring payment",
  day_of_week_sin: "Day of the week",
  day_of_week_cos: "Day of the week",
  month_sin: "Time of year",
  month_cos: "Time of year",
  user_history_mean: "Your average purchase",
  user_history_std: "How much your purchases vary",
  user_history_count: "Number of earlier purchases",
  user_amount_deviation: "Difference from your average",
  user_amount_zscore: "How unusual the amount is for you",
  category_history_mean: "Usual amount in this category",
  category_history_std: "Variation within this category",
  category_history_count: "Earlier purchases in this category",
  category_history_share: "How often you buy in this category",
  merchant_history_count: "Earlier payments to this merchant",
  merchant_familiar: "Known merchant",
  user_transactions_last_30_days: "Your purchases in the last 30 days",
  category_transactions_last_30_days: "Category purchases in the last 30 days",
  category: "Category",
  subcategory: "Subcategory",
  payment_mode: "Payment method",
  account_type: "Account type",
};

export function featureLabel(name) {
  const raw = String(name);
  if (FEATURE_LABELS[raw]) return FEATURE_LABELS[raw];
  // One-hot encoded columns such as "categorical__category_Food".
  const cleaned = raw.replace(/^[a-z]+__/, "");
  for (const base of ["subcategory", "category", "payment_mode", "account_type"]) {
    if (cleaned.startsWith(`${base}_`)) return `${FEATURE_LABELS[base]}: ${cleaned.slice(base.length + 1)}`;
  }
  if (FEATURE_LABELS[cleaned]) return FEATURE_LABELS[cleaned];
  return cleaned.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase());
}

export const FORECAST_MONEY_FEATURES = new Set(["previous_1", "previous_2", "previous_3", "rolling_3", "rolling_6"]);

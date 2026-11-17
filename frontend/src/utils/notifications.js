import { currency } from "./format";

const DAY_MS = 24 * 60 * 60 * 1000;

function monthName(month) {
  if (!month) return "";
  return new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric" }).format(new Date(`${month}-01T00:00:00`));
}

// Alerts derived from data the dashboard has already loaded, most urgent first.
export function buildNotifications(data) {
  const items = [];
  for (const budget of data.budgets || []) {
    const amount = Number(budget.amount);
    const spent = Number(budget.spent);
    if (!amount) continue;
    if (spent > amount) {
      items.push({
        id: `budget-over-${budget.id}`,
        tone: "danger",
        title: `${budget.category} budget exceeded`,
        body: `${currency(spent)} spent of ${currency(amount)} in ${monthName(budget.budget_month)}.`,
        to: "/budgets",
      });
    } else if (spent >= amount * 0.9) {
      items.push({
        id: `budget-near-${budget.id}`,
        tone: "warning",
        title: `${budget.category} budget almost used`,
        body: `${Math.round((spent / amount) * 100)}% used in ${monthName(budget.budget_month)}.`,
        to: "/budgets",
      });
    }
  }

  const anomalyCount = data.anomalies?.status === "available" ? Number(data.anomalies.anomaly_count || 0) : 0;
  if (anomalyCount > 0) {
    items.push({
      id: "anomalies",
      tone: "warning",
      title: `${anomalyCount} unusual transaction${anomalyCount === 1 ? "" : "s"} to review`,
      body: "Flagged by the anomaly model as outside your usual pattern.",
      to: "/anomalies",
    });
  }

  const forecast = data.forecast;
  if (forecast?.status === "available" && !forecast.is_historical && forecast.predicted_expense != null) {
    items.push({
      id: "forecast",
      tone: "info",
      title: `Forecast for ${monthName(forecast.forecast_month)}: ${currency(forecast.predicted_expense)}`,
      body: "Predicted total expenses from your recent spending.",
      to: "/forecasting",
    });
  }

  const highPriority = Array.isArray(data.recommendations)
    ? data.recommendations.filter((item) => item.priority === "High").length
    : 0;
  if (highPriority > 0) {
    items.push({
      id: "recommendations",
      tone: "info",
      title: `${highPriority} high-priority recommendation${highPriority === 1 ? "" : "s"}`,
      body: "Suggested actions based on your recorded finances.",
      to: "/recommendations",
    });
  }

  const today = new Date();
  for (const goal of data.goals || []) {
    const target = Number(goal.target);
    const current = Number(goal.current);
    const due = goal.date ? new Date(`${String(goal.date).slice(0, 10)}T00:00:00`) : null;
    if (!target || current >= target || !due || Number.isNaN(due.getTime())) continue;
    const days = Math.ceil((due - today) / DAY_MS);
    if (days <= 120) {
      items.push({
        id: `goal-${goal.id}`,
        tone: days < 0 ? "danger" : "warning",
        title: days < 0 ? `${goal.name} is past its target date` : `${goal.name} is due in ${days} days`,
        body: `${Math.round((current / target) * 100)}% saved · ${currency(target - current)} to go.`,
        to: "/goals",
      });
    }
  }

  const order = { danger: 0, warning: 1, info: 2 };
  return items.sort((left, right) => order[left.tone] - order[right.tone]);
}

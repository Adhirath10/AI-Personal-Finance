export const demoData = {
  overview: {
    income: 184500,
    expenses: 112840,
    savings: 71660,
    budgetUtilization: 72,
    incomeChange: 8.2,
    expenseChange: -3.4,
    savingsChange: 14.8,
  },
  monthly: [
    { month: "Jun", income: 158000, expenses: 101400, forecast: null },
    { month: "Jul", income: 172000, expenses: 118200, forecast: null },
    { month: "Aug", income: 168000, expenses: 109800, forecast: null },
    { month: "Sep", income: 179000, expenses: 121500, forecast: null },
    { month: "Oct", income: 176500, expenses: 116900, forecast: null },
    { month: "Nov", income: 184500, expenses: 112840, forecast: 118300 },
  ],
  categories: [
    { name: "Housing", value: 34500, color: "#5578f5" },
    { name: "Food & dining", value: 22840, color: "#46b998" },
    { name: "Transport", value: 14500, color: "#f0a94a" },
    { name: "Shopping", value: 18300, color: "#a18cf4" },
    { name: "Utilities", value: 12700, color: "#ef7789" },
    { name: "Other", value: 10000, color: "#9aa7bd" },
  ],
  transactions: [
    { id: "TXN-2048", description: "Monthly rent", merchant: "Urban Nest", date: "2026-11-12", category: "Housing", amount: 28500, type: "Expense", status: "Completed" },
    { id: "TXN-2047", description: "Salary credited", merchant: "Acme Technologies", date: "2026-11-11", category: "Income", amount: 145000, type: "Income", status: "Completed" },
    { id: "TXN-2046", description: "Grocery run", merchant: "Nature's Basket", date: "2026-11-10", category: "Food & dining", amount: 3240, type: "Expense", status: "Completed" },
    { id: "TXN-2045", description: "Ride to office", merchant: "Uber", date: "2026-11-10", category: "Transport", amount: 486, type: "Expense", status: "Completed" },
    { id: "TXN-2044", description: "Monthly subscription", merchant: "Netflix", date: "2026-11-09", category: "Entertainment", amount: 649, type: "Expense", status: "Completed" },
    { id: "TXN-2043", description: "Online purchase", merchant: "Myntra", date: "2026-11-08", category: "Shopping", amount: 4299, type: "Expense", status: "Completed" },
    { id: "TXN-2042", description: "Electricity bill", merchant: "BESCOM", date: "2026-11-07", category: "Utilities", amount: 2180, type: "Expense", status: "Completed" },
    { id: "TXN-2041", description: "Restaurant dinner", merchant: "Olive Bar & Kitchen", date: "2026-11-06", category: "Food & dining", amount: 3650, type: "Expense", status: "Completed" },
    { id: "TXN-2040", description: "Freelance project", merchant: "Client payment", date: "2026-11-04", category: "Income", amount: 39500, type: "Income", status: "Completed" },
  ],
  budgets: [
    { id: "b-housing", category: "Housing", amount: 42000, spent: 34500 },
    { id: "b-food", category: "Food & dining", amount: 28000, spent: 22840 },
    { id: "b-transport", category: "Transport", amount: 18000, spent: 14500 },
    { id: "b-shopping", category: "Shopping", amount: 16000, spent: 18300 },
    { id: "b-utilities", category: "Utilities", amount: 15000, spent: 12700 },
  ],
  goals: [
    { id: "g-1", name: "Emergency fund", target: 300000, current: 198000, date: "2027-06-30", icon: "shield" },
    { id: "g-2", name: "Japan vacation", target: 180000, current: 76500, date: "2027-10-01", icon: "plane" },
    { id: "g-3", name: "New laptop", target: 120000, current: 92000, date: "2027-02-15", icon: "laptop" },
  ],
  recommendations: [
    { id: "r-1", title: "Shopping budget exceeded", description: "You have spent ₹2,300 over your monthly shopping budget. Consider pausing non-essential purchases for the rest of the month.", priority: "High", reason: "Shopping is at 114% of its monthly budget.", type: "Budget warning", icon: "shopping" },
    { id: "r-2", title: "Your savings are on track", description: "At your current pace, you could add ₹18,400 to your emergency fund this month.", priority: "Medium", reason: "Monthly savings are up 14.8% compared with last month.", type: "Saving opportunity", icon: "trend" },
    { id: "r-3", title: "Review dining spend", description: "Dining expenses rose over the last two weeks. A weekly meal plan may help bring this category back in line.", priority: "Low", reason: "Recent dining activity is above your 3-month average.", type: "Spending pattern", icon: "utensils" },
  ],
  anomalies: [],
  forecast: null,
  insights: null,
  explainability: null,
  settings: {
    name: "Aarav Sharma",
    email: "aarav.sharma@email.com",
    currency: "INR",
    notifications: true,
  },
};

export const demoForecast = {
  predicted_expense: 118300,
  forecast_month: "2026-12",
  mae: null,
  rmse: null,
  mape: null,
  explanation: null,
  feature_importance: [],
};

export const demoAnomalies = {
  total_analyzed: 0,
  normal_count: 0,
  anomaly_count: 0,
  alert_rate: 0,
  items: [],
};

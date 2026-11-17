import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  Banknote,
  BellRing,
  BrainCircuit,
  CalendarDays,
  Check,
  ChevronDown,
  CircleAlert,
  CircleDollarSign,
  Clock3,
  Download,
  Filter,
  HeartPulse,
  Lightbulb,
  LockKeyhole,
  PieChart,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Wallet,
} from "lucide-react";
import {
  AreaChart,
  Area,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BudgetProgress,
  Button,
  ChartCard,
  EmptyAction,
  EmptyState,
  PageHeading,
  StatusBadge,
  SummaryCard,
  TooltipNote,
} from "../components/UI";
import TransactionTable from "../components/TransactionTable";
import { ExpenseDonut, FeatureImportanceChart, IncomeExpenseChart, SpendingLineChart } from "../components/Charts";
import { currency, percent, shortDate } from "../utils/format";

function ChartLegend({ items }) {
  return <div className="chart-legend">{items.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}</div>;
}

function MiniMetric({ icon: Icon, label, value, tone = "blue" }) {
  return <div className="mini-metric"><span className={`mini-metric-icon ${tone}`}><Icon size={16} /></span><span>{label}</span><strong>{value}</strong></div>;
}

export function OverviewPage({ data, mode, refreshing, onRefresh, onNavigate }) {
  const [trendMonths, setTrendMonths] = useState(6);
  const overview = data.overview || {};
  const monthlyData = (data.monthly || []).slice(-trendMonths);
  const recent = data.recentTransactions || [];
  const categoryTotal = (data.categories || []).reduce((sum, item) => sum + Number(item.value || 0), 0);
  const modeIsLive = mode === "live";
  const period = overview.analysis_period?.month;
  return (
    <div className="page-stack">
      <PageHeading eyebrow={period ? `Recorded period · ${period}` : "Recorded financial data"} title="Financial overview" description="A database-derived view of your recorded income, expenses and plans." actions={<><button className="button button-outline" onClick={() => window.print()}><Download size={15} />Export report</button><button className="button button-primary" onClick={() => onNavigate("/transactions")}><Plus size={16} />Add transaction</button></>} />
      {period && overview.analysis_period?.status === "historical" && <div className="info-strip"><CircleAlert size={16} /><span>Showing historical records through {overview.analysis_period.latest_transaction_date}. This is not current live financial activity.</span></div>}
      {mode === "live" && data.overviewError && <div className="inline-warning"><CircleAlert size={16} />{data.overviewError}</div>}
      <div className="summary-grid">
        <SummaryCard label="Recorded income" value={overview.income ?? "—"} change={overview.incomeChange ?? undefined} icon={ArrowDownRight} tone="blue" helper={period ? `for ${period}` : "no transaction history"} />
        <SummaryCard label="Recorded expenses" value={overview.expenses ?? "—"} change={overview.expenseChange ?? undefined} icon={ArrowUpRight} tone="rose" helper={period ? `for ${period}` : "no transaction history"} />
        <SummaryCard label="Recorded surplus" value={overview.savings ?? "—"} change={overview.savingsChange ?? undefined} icon={Wallet} tone="mint" helper="income minus expenses" />
        <SummaryCard label="Budget utilization" value={overview.budget_utilization == null ? "—" : percent(overview.budget_utilization)} icon={PieChart} tone="amber" helper="of recorded monthly budget" />
      </div>
      <div className="dashboard-grid">
        <ChartCard title="Income vs. expenses" subtitle="Your monthly cash flow" className="span-7" action={<div className="chart-tools"><ChartLegend items={[{ label: "Income", color: "#5578f5" }, { label: "Expenses", color: "#c6d2fb" }]} /><select className="select-button" aria-label="Chart period" value={trendMonths} onChange={(event) => setTrendMonths(Number(event.target.value))}><option value={3}>Last 3 months</option><option value={6}>Last 6 months</option><option value={12}>Last 12 months</option></select></div>}>
          <div className="chart-area"><IncomeExpenseChart data={monthlyData} /></div>
        </ChartCard>
        <ChartCard title="Expense breakdown" subtitle="Where your money goes" className="span-5">
          {categoryTotal > 0 ? <><div className="donut-container"><ExpenseDonut data={data.categories || []} /></div><div className="category-legend">{(data.categories || []).slice(0, 5).map((category) => <div key={category.name}><span><i style={{ background: category.color }} />{category.name}</span><strong>{Math.round(category.value / categoryTotal * 100)}%</strong></div>)}</div></> : <EmptyState title="No category data" description="Category breakdown will appear when expense data is available." />}
        </ChartCard>
        <ChartCard title="Spending trends" subtitle="Monthly expense activity" className="span-7" action={<span className="select-button">Same period as cash flow</span>}>
          <div className="chart-area chart-area-short"><SpendingLineChart data={monthlyData} /></div>
        </ChartCard>
        <ChartCard title="Budget overview" subtitle="Your monthly category limits" className="span-5" action={<button className="text-button" onClick={() => onNavigate("/budgets")}>View budgets <ArrowRight size={14} /></button>}>
          <div className="budget-list">{(data.budgets || []).slice(0, 4).map((budget) => <BudgetProgress key={budget.id} budget={budget} compact />)}</div>
          <div className="budget-total"><span>Total monthly budget</span><strong>{currency((data.budgets || []).reduce((s, item) => s + Number(item.amount), 0))}</strong></div>
        </ChartCard>
        <ChartCard title="Recent transactions" subtitle={`Configured backend activity · auto-refreshes every 30 seconds${data.activity?.as_of ? ` · checked ${new Date(data.activity.as_of).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : ""}`} className="span-12" action={<div className="heading-actions"><button className="text-button" onClick={onRefresh} disabled={refreshing || mode !== "live"}><RefreshCw size={14} className={refreshing ? "spin" : ""} />{refreshing ? "Refreshing" : "Refresh"}</button><button className="text-button" onClick={() => onNavigate("/transactions")}>All transactions <ArrowRight size={14} /></button></div>}>
          {recent.length ? <TransactionTable rows={recent} compact /> : <EmptyState title="No transactions yet" description="Transactions from your connected backend appear here." />}
        </ChartCard>
        <section className="ai-spotlight span-12">
          <div className="ai-spotlight-head"><div className="ai-spark"><Sparkles size={18} /></div><div><h2>Your AI financial pulse</h2><p>AI results appear only after a connected model endpoint returns them.</p></div><StatusBadge tone="neutral">{modeIsLive ? "Run on request" : "Unavailable"}</StatusBadge></div>
          <div className="pulse-grid">
            <div className="pulse-item"><span className="pulse-kicker"><TrendingDown size={14} />Expense forecast</span><strong>{data.forecast?.status === "available" ? currency(data.forecast.predicted_expense) : data.forecast?.status ? "Unavailable" : "Not run"}</strong><p>{data.forecast?.status === "available" ? `For ${data.forecast.forecast_month}${data.forecast.is_historical ? ` · from records through ${data.forecast.history_end_month}` : ""}` : data.forecast?.reason || data.forecast?.skip_reason || "Run forecasting to check model availability."}</p><button className="text-button" onClick={() => onNavigate("/forecasting")}>Explore forecast <ArrowRight size={13} /></button></div>
            <div className="pulse-item"><span className="pulse-kicker"><ShieldAlert size={14} />Unusual activity</span><strong>{data.anomalies?.status === "available" ? (data.anomalies.anomaly_count ?? "—") : "Not run"}</strong><p>{data.anomalies?.analysis_period?.selected_month ? `Recorded period ${data.anomalies.analysis_period.selected_month}` : "Run the saved model to review potential outliers."}</p><button className="text-button" onClick={() => onNavigate("/anomalies")}>Review activity <ArrowRight size={13} /></button></div>
            <div className="pulse-item"><span className="pulse-kicker"><Lightbulb size={14} />Smart suggestions</span><strong>{Array.isArray(data.recommendations) ? data.recommendations.length : "Not run"}</strong><p>{data.recommendations?.analysis_period?.selected_month ? `Analysis period ${data.recommendations.analysis_period.selected_month}` : "Request data-backed recommendations."}</p><button className="text-button" onClick={() => onNavigate("/recommendations")}>See suggestions <ArrowRight size={13} /></button></div>
          </div>
        </section>
      </div>
    </div>
  );
}

function hasMeaningfulNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatRecommendationSupportValue(name, value) {
  if (value == null) return "—";
  const label = name.replace(/_/g, " ");

  if (Array.isArray(value)) {
    if (!value.length) return "None";
    return value
      .map((item) => {
        if (typeof item === "object" && item !== null) {
          const parts = [];
          const merchant = item.merchant || item.name;
          if (merchant) parts.push(`Merchant: ${merchant}`);
          if (item.amount != null) parts.push(`Amount: ${currency(item.amount)}`);
          const date = item.transaction_date || item.date;
          if (date) parts.push(`Date: ${shortDate(date)}`);
          if (item.description) parts.push(`Description: ${item.description}`);
          if (item.category) parts.push(`Category: ${item.category}`);
          if (item.anomaly_score != null) parts.push(`Score: ${Number(item.anomaly_score).toFixed(3)}`);
          return parts.join(" • ");
        }
        return String(item);
      })
      .join(" • ");
  }

  if (typeof value === "object") {
    const parts = [];
    const merchant = value.merchant || value.name;
    if (merchant) parts.push(`Merchant: ${merchant}`);
    if (value.amount != null) parts.push(`Amount: ${currency(value.amount)}`);
    const date = value.transaction_date || value.date;
    if (date) parts.push(`Date: ${shortDate(date)}`);
    if (value.description) parts.push(`Description: ${value.description}`);
    if (value.category) parts.push(`Category: ${value.category}`);
    if (value.anomaly_score != null) parts.push(`Score: ${Number(value.anomaly_score).toFixed(3)}`);
    return parts.join(" • ") || label;
  }

  if (/(amount|budget|actual|target|current|remaining|required|deficit|expense|income|cost|spent)/i.test(name) && hasMeaningfulNumber(value)) {
    return currency(value);
  }

  if (/(utilization|ratio|rate|percentage|progress|change)/i.test(name) && hasMeaningfulNumber(value)) {
    return percent(value);
  }

  return String(value);
}

export function TransactionsPage({
  data,
  mode,
  transactionMeta,
  transactionLoading,
  onFetchTransactions,
  onSave,
  onDelete,
  onRunModel,
}) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState("All types");
  const [category, setCategory] = useState("All categories");
  const [sort, setSort] = useState("date-new");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState(null);
  const pageSize = 7;
  const rows = data.transactions || [];
  const total = transactionMeta?.total || 0;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const categories = data.transactionCategories || [];
  const sortBy = sort.startsWith("amount") ? "amount" : "date";
  const sortOrder = sort.endsWith("old") || sort.endsWith("low") ? "asc" : "desc";
  useEffect(() => {
    const timer = window.setTimeout(() => {
      onFetchTransactions({
        limit: pageSize,
        offset: (page - 1) * pageSize,
        search: query.trim() || undefined,
        category: category === "All categories" ? undefined : category,
        transaction_type: type === "All types" ? undefined : type,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
    }, query ? 250 : 0);
    return () => window.clearTimeout(timer);
  }, [page, query, category, type, startDate, endDate, sortBy, sortOrder, onFetchTransactions]);

  const submit = async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = { description: form.get("description"), merchant: form.get("merchant"), date: form.get("date"), category: form.get("category"), amount: Number(form.get("amount")), balance_after_transaction: Number(form.get("balance_after_transaction")), type: form.get("type") };
    const result = await onSave("transactions", payload, editing?.id);
    if (result) {
      setEditing(null);
      onFetchTransactions({
        limit: pageSize,
        offset: (page - 1) * pageSize,
        search: query.trim() || undefined,
        category: category === "All categories" ? undefined : category,
        transaction_type: type === "All types" ? undefined : type,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
    }
  };
  return (
    <div className="page-stack">
      <PageHeading eyebrow={data.overview?.analysis_period?.month ? `Stored activity · ${data.overview.analysis_period.month}` : "Your activity"} title="Transactions" description="Track, search and organize your recorded income and spending." actions={<><Button variant="outline" icon={Sparkles} disabled={!rows.length || transactionLoading || mode !== "live" || data.modelLoading?.categorize} onClick={() => onRunModel("categorize", { items: rows.map((row) => ({ id: row.id, description: row.description, merchant: row.merchant })) }, "Transaction categorization")}>{data.modelLoading?.categorize ? "Categorizing..." : "Auto categorize"}</Button><Button icon={Plus} onClick={() => setEditing({})}>Add transaction</Button></>} />
      {data.transactionsError && <div className="inline-warning"><CircleAlert size={16} />{data.transactionsError}</div>}
      <div className="summary-grid summary-grid-small">
        <SummaryCard label="Matching transactions" value={total} format="number" icon={Activity} tone="blue" helper="for selected filters" />
        <SummaryCard label="Income in latest period" value={data.overview?.income ?? "—"} icon={ArrowDownRight} tone="mint" helper={data.overview?.analysis_period?.month || "stored data"} />
        <SummaryCard label="Expenses in latest period" value={data.overview?.expenses ?? "—"} icon={ArrowUpRight} tone="rose" helper={data.overview?.analysis_period?.month || "stored data"} />
      </div>
      <section className="panel transactions-panel">
        <div className="filter-toolbar">
          <label className="table-search"><Search size={16} /><input placeholder="Search merchant or description" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} /></label>
          <div className="filter-controls"><select aria-label="Transaction type" value={type} onChange={(event) => { setType(event.target.value); setPage(1); }}><option>All types</option><option>Income</option><option>Expense</option></select><select aria-label="Category" value={category} onChange={(event) => { setCategory(event.target.value); setPage(1); }}><option>All categories</option>{categories.map((item) => <option key={item}>{item}</option>)}</select><select aria-label="Sort transactions" value={sort} onChange={(event) => setSort(event.target.value)}><option value="date-new">Newest first</option><option value="date-old">Oldest first</option><option value="amount-high">Amount: high to low</option><option value="amount-low">Amount: low to high</option></select><input aria-label="Start date" type="date" value={startDate} onChange={(event) => { setStartDate(event.target.value); setPage(1); }} /><input aria-label="End date" type="date" value={endDate} onChange={(event) => { setEndDate(event.target.value); setPage(1); }} /></div>
        </div>
        {transactionLoading ? <div className="empty-table">Loading transactions…</div> : rows.length ? <TransactionTable rows={rows} onEdit={setEditing} onDelete={onDelete ? (row) => onDelete("transactions", row.id) : undefined} /> : <EmptyState title="No matching transactions" description={data.transactionsError || "Try adjusting your search or filters."} />}
        <div className="table-footer"><span>Showing {total ? (page - 1) * pageSize + 1 : 0}–{Math.min((page - 1) * pageSize + rows.length, total)} of {total} transactions</span><div className="pagination"><button disabled={page <= 1 || transactionLoading} onClick={() => setPage((current) => current - 1)}>Previous</button><span>{page} / {totalPages}</span><button disabled={page >= totalPages || transactionLoading} onClick={() => setPage((current) => current + 1)}>Next</button></div></div>
      </section>
      {editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal-card" onSubmit={submit}><header><div><h2>{editing.id ? "Edit transaction" : "Add transaction"}</h2><p>Keep your financial activity up to date.</p></div><button type="button" className="icon-btn" onClick={() => setEditing(null)}>×</button></header><div className="form-grid"><label>Description<input name="description" required defaultValue={editing.description || ""} /></label><label>Merchant<input name="merchant" required defaultValue={editing.merchant || ""} /></label><label>Amount (₹)<input name="amount" type="number" min="0.01" step="0.01" required defaultValue={editing.amount || ""} /></label><label>Balance after transaction (₹)<input name="balance_after_transaction" type="number" step="0.01" required defaultValue={editing.balance_after_transaction ?? ""} /></label><label>Date<input name="date" type="date" required defaultValue={editing.date || new Date().toISOString().slice(0, 10)} /></label><label>Category<select name="category" defaultValue={editing.category || categories[0] || "Other"}>{categories.map((item) => <option key={item}>{item}</option>)}</select></label><label>Type<select name="type" defaultValue={editing.type || "Expense"}><option>Expense</option><option>Income</option></select></label></div><footer><Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button type="submit">{editing.id ? "Save changes" : "Add transaction"}</Button></footer></form></div>}
    </div>
  );
}

export function InsightsPage({ data, mode, onRunModel }) {
  const insights = data.insights;
  return <div className="page-stack"><PageHeading eyebrow={insights?.analysis_period?.selected_month ? `Recorded period · ${insights.analysis_period.selected_month}` : "Patterns & opportunities"} title="AI insights" description="Patterns derived from this user’s stored transaction history." actions={<Button variant="outline" icon={RefreshCw} disabled={mode !== "live" || data.modelLoading?.insights} onClick={() => onRunModel("insights", {}, "AI insights")}>{data.modelLoading?.insights ? "Analyzing..." : "Refresh insights"}</Button>} />{insights?.items?.length ? <div className="insight-grid">{insights.items.map((item, index) => <InsightCard key={item.id || index} item={item} />)}</div> : <EmptyState title={insights?.status === "unavailable" ? "Insights unavailable" : "No AI insights available"} description={insights?.reason || data.insightsError || "Request analysis to check for transaction patterns. No insights are fabricated when the service is unavailable."} action={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.insights} onClick={() => onRunModel("insights", {}, "AI insights")}>Request insights</Button>} />}<div className="info-strip"><BrainCircuit size={17} /><span>Insights are based on recorded transactions. Their analysis period may be historical.</span></div></div>;
}

function InsightCard({ item }) {
  return <article className="panel insight-card"><div className="insight-symbol"><Sparkles size={18} /></div><StatusBadge tone={item.priority === "High" ? "danger" : "success"}>{item.priority || "Insight"}</StatusBadge><h3>{item.title}</h3><p>{item.description}</p>{item.supporting_values && <div className="support-values">{Object.entries(item.supporting_values).map(([name, value]) => <span key={name}>{name.replaceAll("_", " ")} <strong>{formatInsightValue(name, value)}</strong></span>)}</div>}</article>;
}

function formatInsightValue(name, value) {
  if (/(spending|average)/i.test(name) && hasMeaningfulNumber(value)) return currency(value);
  return formatRecommendationSupportValue(name, value);
}

export function ForecastPage({ data, mode, onRunModel, navigate }) {
  const forecast = data.forecast;
  const explanation = data.explainability?.forecast;
  const usable = forecast?.status === "available" && forecast.predicted_expense != null;
  return <div className="page-stack"><PageHeading eyebrow="Predictive finance" title="Expense forecasting" description="Run the saved Random Forest model against available recorded history." actions={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.forecast} onClick={() => onRunModel("forecast", {}, "Expense forecast")}>{data.modelLoading?.forecast ? "Running forecast..." : "Run forecast"}</Button>} />{usable ? <>{forecast.is_historical ? <div className="info-strip info-warning" role="status"><CircleAlert size={16} /><span>{forecast.notice || `Recorded history ends in ${forecast.history_end_month}, so the forecast month ${forecast.forecast_month} has already passed.`}</span></div> : <div className="info-strip"><CircleAlert size={16} /><span>Forecast for the month after your latest recorded expenses ({forecast.history_end_month || forecast.analysis_period?.selected_month}).</span></div>}<div className="forecast-hero"><div className="forecast-main"><div className="forecast-kicker"><Sparkles size={15} />FORECAST <TooltipNote>Model output from the connected forecasting API.</TooltipNote></div><strong>{currency(forecast.predicted_expense)}</strong><span>Predicted expenses for {forecast.forecast_month}</span>{forecast.is_historical ? <StatusBadge tone="warning">Based on historical records</StatusBadge> : <StatusBadge tone="success">Model response received</StatusBadge>}</div><div className="forecast-metrics"><MiniMetric icon={Activity} label="MAE" value={forecast.mae == null ? "—" : currency(forecast.mae)} /><MiniMetric icon={Activity} label="RMSE" value={forecast.rmse == null ? "—" : currency(forecast.rmse)} /><MiniMetric icon={Activity} label="MAPE" value={forecast.mape == null ? "—" : `${Number(forecast.mape).toFixed(1)}%`} /></div></div><ChartCard title="Historical spending" subtitle="Recorded monthly expenses"><div className="chart-area chart-area-tall"><SpendingLineChart data={data.monthly || []} /></div></ChartCard>{explanation?.feature_importance?.length ? <ChartCard title="What drives this forecast?" subtitle="Global mean absolute SHAP contribution"><FeatureImportanceChart data={explanation.feature_importance} /></ChartCard> : <EmptyState title={explanation?.status && explanation.status !== "available" ? "SHAP explanation unavailable" : "Explain this forecast"} description={explanation?.reason || "See which recent spending features pushed this prediction up or down."} action={<div className="heading-actions"><Button icon={BrainCircuit} disabled={mode !== "live" || data.modelLoading?.explainability} onClick={() => onRunModel("explainability", { model: "forecast" }, "Forecast explanation")}>{data.modelLoading?.explainability ? "Explaining..." : "Explain this forecast"}</Button><Button variant="outline" onClick={() => navigate("/explainability")}>Open Explainable AI</Button></div>} />}{explanation?.feature_contributions?.length > 0 && <FeatureTable rows={explanation.feature_contributions} title="This prediction explained" />}</> : <EmptyState title={forecast?.status === "insufficient_history" ? "Not enough history to forecast" : forecast?.status === "unavailable" ? "Forecast unavailable" : "Forecasting has not run"} description={forecast?.reason || forecast?.skip_reason || data.forecastError || "Run the saved forecasting model on your recorded monthly expenses."} action={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.forecast} onClick={() => onRunModel("forecast", {}, "Expense forecast")}>{data.modelLoading?.forecast ? "Checking..." : "Check forecast status"}</Button>} />}</div>;
}

export function AnomaliesPage({ data, mode, onRunModel }) {
  const result = data.anomalies;
  const items = result?.items || result?.transactions || [];
  return <div className="page-stack"><PageHeading eyebrow={result?.analysis_period?.selected_month ? `Recorded period · ${result.analysis_period.selected_month}` : "Transaction safety"} title="Anomaly detection" description="Review out-of-pattern transactions from the saved Isolation Forest model." actions={<Button icon={ShieldAlert} disabled={mode !== "live" || data.modelLoading?.anomalies} onClick={() => onRunModel("anomalies", {}, "Anomaly detection")}>{data.modelLoading?.anomalies ? "Analyzing..." : "Analyze transactions"}</Button>} /><div className="info-strip info-warning"><ShieldAlert size={17} /><span>An unusual transaction is a statistical outlier requiring review, not confirmed fraud. Results are limited to data after the model’s training cutoff.</span></div>{result?.status === "available" ? <><div className="summary-grid"><SummaryCard label="Transactions analyzed" value={result.total_analyzed ?? "—"} format="number" icon={Activity} tone="blue" /><SummaryCard label="Normal transactions" value={result.normal_count ?? "—"} format="number" icon={BadgeCheck} tone="mint" /><SummaryCard label="Potential anomalies" value={result.anomaly_count ?? items.length} format="number" icon={ShieldAlert} tone="rose" /><SummaryCard label="Alert rate" value={result.alert_rate == null ? "—" : result.alert_rate * 100} format="percent" icon={CircleAlert} tone="amber" /></div><section className="panel"><div className="panel-heading"><div><h2>Transactions to review</h2><p>Scored after {result.training_end || "the model training cutoff"}</p></div></div>{items.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Transaction</th><th>Date</th><th>Category</th><th>Anomaly score</th><th>Review status</th></tr></thead><tbody>{items.map((item) => <tr key={item.transaction_id || item.id}><td><div className="transaction-cell"><span className="merchant-logo">{(item.merchant || "TX").slice(0, 2).toUpperCase()}</span><span><strong>{item.merchant || item.description || "Transaction"}</strong><small>{item.transaction_id || item.id}</small></span></div></td><td>{shortDate(item.transaction_date || item.date)}</td><td>{item.category}</td><td><strong>{Number(item.anomaly_score).toFixed(4)}</strong></td><td><StatusBadge tone="warning">{item.review_status || "Needs review"}</StatusBadge></td></tr>)}</tbody></table></div> : <EmptyState title="No potential anomalies returned" description="The model did not flag any of the eligible, out-of-sample expense transactions." />}</section></> : <EmptyState title={result?.status === "no_out_of_sample_rows" ? "No eligible out-of-sample transactions" : "Anomaly analysis unavailable"} description={result?.reason || data.anomaliesError || `Model status: ${result?.status || "not run"}. Run the saved model to inspect eligible expenses.`} action={<Button icon={ShieldAlert} disabled={mode !== "live" || data.modelLoading?.anomalies} onClick={() => onRunModel("anomalies", {}, "Anomaly detection")}>{data.modelLoading?.anomalies ? "Analyzing..." : "Check model status"}</Button>} />}</div>;
}

export function RecommendationsPage({ data, mode, onRunModel }) {
  const recommendations = data.recommendations || [];
  const priorityCounts = { High: 0, Medium: 0, Low: 0 };
  recommendations.forEach((item) => {
    const priority = item.priority || "Low";
    if (priorityCounts[priority] !== undefined) {
      priorityCounts[priority] += 1;
    } else {
      priorityCounts.Low += 1;
    }
  });

  return <div className="page-stack"><PageHeading eyebrow={data.recommendationsPeriod?.selected_month ? `Analysis period · ${data.recommendationsPeriod.selected_month}` : "Your next best moves"} title="Recommendations" description="Data-backed actions based on the configured user’s recorded financial history." actions={<Button icon={RefreshCw} variant="outline" disabled={mode !== "live" || data.modelLoading?.recommendations} onClick={() => onRunModel("recommendations", {}, "Recommendations")}>{data.modelLoading?.recommendations ? "Analyzing..." : "Refresh"}</Button>} />{recommendations.length ? <><div className="summary-grid"><SummaryCard label="Total recommendations" value={recommendations.length} format="number" icon={Lightbulb} tone="blue" /><SummaryCard label="High-priority recommendations" value={priorityCounts.High} format="number" icon={ShieldAlert} tone="rose" /><SummaryCard label="Medium-priority recommendations" value={priorityCounts.Medium} format="number" icon={CircleAlert} tone="amber" /><SummaryCard label="Low-priority recommendations" value={priorityCounts.Low} format="number" icon={BadgeCheck} tone="mint" /></div><div className="recommendation-list">{recommendations.map((item) => <RecommendationCard key={item.id || item.title} item={item} />)}</div></> : <EmptyState title="No recommendations available" description={data.recommendationsError || "Request analysis. No suggestions are fabricated when the AI service or required artifacts are unavailable."} action={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.recommendations} onClick={() => onRunModel("recommendations", {}, "Recommendations")}>Get recommendations</Button>} />}</div>;
}

function RecommendationCard({ item }) {
  const tone = item.priority === "High" ? "danger" : item.priority === "Medium" ? "warning" : "success";
  return <article className="panel recommendation-card"><div className={`recommendation-icon ${tone}`}><Lightbulb size={19} /></div><div className="recommendation-content"><div className="recommendation-top"><StatusBadge tone={tone}>{item.priority || "Priority"}</StatusBadge><span className="recommendation-type">{item.type || item.recommendation_type || "Recommendation"}</span></div><h3>{item.title || item.recommendation_title}</h3><p>{item.description || item.recommendation_description}</p>{item.reason && <div className="reason-line"><InfoText />{item.reason}</div>}{item.supporting_values && <div className="support-values">{Object.entries(item.supporting_values).map(([name, value]) => <span key={name}>{name.replaceAll("_", " ")} <strong>{formatRecommendationSupportValue(name, value)}</strong></span>)}</div>}</div><span className="subtle-icon" aria-hidden="true"><ArrowRight size={17} /></span></article>;
}
function InfoText() { return <CircleAlert size={14} />; }

export function BudgetsPage({ data, mode, onSave, onDelete }) {
  const [editing, setEditing] = useState(null);
  const budgets = data.budgets || [];
  const total = budgets.reduce((sum, item) => sum + Number(item.amount || 0), 0);
  const spent = budgets.reduce((sum, item) => sum + Number(item.spent || 0), 0);
  const save = async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = { category: form.get("category"), amount: Number(form.get("amount")), month: form.get("month") };
    if (await onSave("budgets", payload, editing?.id)) setEditing(null);
  };
  return <div className="page-stack"><PageHeading eyebrow={budgets[0]?.budget_month ? `Recorded budget · ${budgets[0].budget_month}` : "Plan with confidence"} title="Budgets" description="Set monthly limits and compare them with recorded expenses." actions={<Button icon={Plus} onClick={() => setEditing({})}>Create budget</Button>} /><div className="summary-grid"><SummaryCard label="Monthly budget" value={total} icon={Wallet} tone="blue" /><SummaryCard label="Recorded spending" value={spent} icon={ArrowUpRight} tone="rose" /><SummaryCard label="Remaining" value={Math.max(0, total - spent)} icon={CircleDollarSign} tone="mint" /><SummaryCard label="Overall utilization" value={percent(total ? spent / total * 100 : 0)} icon={PieChart} tone="amber" /></div><div className="budget-page-grid">{budgets.length ? budgets.map((budget) => <article className="panel budget-card" key={budget.id}><div className="budget-card-top"><div><span className="budget-category-icon"><Wallet size={17} /></span><h3>{budget.category}</h3></div><details className="row-menu"><summary aria-label="Budget actions">···</summary><div className="row-menu-pop"><button onClick={() => setEditing(budget)}>Edit budget</button><button className="danger-action" onClick={() => onDelete("budgets", budget.id)}>Delete budget</button></div></details></div><div className="budget-amount">{currency(budget.spent)} <small>spent of {currency(budget.amount)}</small></div><BudgetProgress budget={budget} /><div className="budget-card-bottom"><span>{budget.budget_month || "Month not provided"}</span><StatusBadge tone={budget.spent > budget.amount ? "danger" : "success"}>{budget.spent > budget.amount ? "Over budget" : "On track"}</StatusBadge></div></article>) : <EmptyState title="No budgets yet" description="Create a monthly budget to keep spending on track." action={<Button icon={Plus} onClick={() => setEditing({})}>Create budget</Button>} />}</div>{editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal-card" onSubmit={save}><header><div><h2>{editing.id ? "Edit budget" : "Create budget"}</h2><p>Choose a category and month for your limit.</p></div><button type="button" className="icon-btn" onClick={() => setEditing(null)}>×</button></header><div className="form-grid"><label>Category<select name="category" defaultValue={editing.category || "Other"}>{(data.transactionCategories || []).map((item) => <option key={item}>{item}</option>)}</select></label><label>Monthly amount (₹)<input name="amount" type="number" min="1" required defaultValue={editing.amount || ""} /></label><label>Month<input name="month" type="month" required defaultValue={editing.budget_month || new Date().toISOString().slice(0, 7)} /></label></div><footer><Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button type="submit">Save budget</Button></footer></form></div>}</div>;
}

export function GoalsPage({ data, mode, onSave, onDelete }) {
  const [editing, setEditing] = useState(null);
  const save = async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = { name: form.get("name"), target: Number(form.get("target")), current: Number(form.get("current")), date: form.get("date") };
    if (await onSave("goals", payload, editing?.id)) setEditing(null);
  };
  return <div className="page-stack"><PageHeading eyebrow="Stored financial goals" title="Financial goals" description="Track progress against goals stored for the configured user." actions={<Button icon={Plus} onClick={() => setEditing({})}>Create a goal</Button>} /><div className="goal-grid">{(data.goals || []).map((goal) => { const progress = goal.target ? Math.min(100, Math.round(goal.current / goal.target * 100)) : 0; return <article className="panel goal-card" key={goal.id}><div className="goal-card-top"><span className="goal-symbol"><Target size={19} /></span><details className="row-menu"><summary aria-label="Goal actions">···</summary><div className="row-menu-pop"><button onClick={() => setEditing(goal)}>Update goal</button><button className="danger-action" onClick={() => onDelete("goals", goal.id)}>Delete goal</button></div></details></div><h3>{goal.name}</h3><p>Target date · {shortDate(goal.date)}</p><div className="goal-progress"><div className="goal-progress-label"><strong>{currency(goal.current)}</strong><span>{currency(goal.target)} goal</span></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><div className="goal-progress-label"><span>{progress}% completed</span><span>{currency(Math.max(goal.target - goal.current, 0))} to go</span></div></div><button className="button button-outline button-full" onClick={() => setEditing(goal)}>Update progress</button></article>; })}{!data.goals?.length && <EmptyState title="Start your first goal" description="Create a savings goal and track progress over time." action={<Button icon={Plus} onClick={() => setEditing({})}>Create a goal</Button>} />}</div>{editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal-card" onSubmit={save}><header><div><h2>{editing.id ? "Update financial goal" : "Create a financial goal"}</h2><p>Make progress visible, one step at a time.</p></div><button type="button" className="icon-btn" onClick={() => setEditing(null)}>×</button></header><div className="form-grid"><label>Goal name<input name="name" required defaultValue={editing.name || ""} /></label><label>Target amount (₹)<input name="target" type="number" min="1" required defaultValue={editing.target || ""} /></label><label>Current savings (₹)<input name="current" type="number" min="0" required defaultValue={editing.current || 0} /></label><label>Target date<input name="date" type="date" required defaultValue={editing.date || ""} /></label></div><footer><Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button type="submit">Save goal</Button></footer></form></div>}</div>;
}

export function SimulatorPage({ data, mode }) {
  const [income, setIncome] = useState(0);
  const [expenses, setExpenses] = useState({});
  useEffect(() => {
    if (mode === "live") {
      setIncome(Number(data.overview?.income || 0));
      setExpenses(Object.fromEntries(
        (data.categories || []).map((item) => [item.name, Number(item.value || 0)]),
      ));
    }
  }, [mode, data.overview?.income, data.categories]);
  const values = Object.values(expenses);
  const total = values.reduce((sum, value) => sum + Number(value), 0);
  const balance = Number(income) - total;
  const updateExpense = (category, value) => setExpenses((current) => ({ ...current, [category]: Number(value) }));
  return <div className="page-stack"><PageHeading eyebrow={data.overview?.analysis_period?.month ? `Starting values · ${data.overview.analysis_period.month}` : "Explore the possibilities"} title="What-if simulator" description="Adjust the recorded baseline and compare a hypothetical monthly cash flow." actions={<Button variant="outline" icon={RefreshCw} onClick={() => { setIncome(Number(data.overview?.income || 0)); setExpenses(Object.fromEntries((data.categories || []).map((item) => [item.name, Number(item.value || 0)]))); }}>Reset scenario</Button>} /><div className="simulator-layout"><section className="panel simulator-controls"><div className="panel-heading"><div><h2>Monthly assumptions</h2><p>Changes stay in this browser session; baseline uses recorded values.</p></div></div><label className="sim-input-label">Monthly income <strong>{currency(income)}</strong><input type="range" min="0" max="1000000" step="5000" value={income} onChange={(event) => setIncome(Number(event.target.value))} /><div className="range-values"><span>₹0</span><span>₹10L</span></div></label><div className="sim-category-list">{Object.entries(expenses).map(([category, amount]) => <label key={category} className="sim-category"><span>{category}</span><div><span>₹</span><input type="number" min="0" step="500" value={amount} onChange={(event) => updateExpense(category, event.target.value)} /></div></label>)}{!Object.keys(expenses).length && <EmptyState title="No expense baseline" description="Transaction category values will appear here when the backend returns recorded expenses." />}</div></section><section className="simulator-results"><article className="sim-balance-card"><span>Hypothetical monthly balance</span><strong className={balance < 0 ? "text-danger" : ""}>{currency(balance)}</strong><p>{balance < 0 ? "Expenses exceed income in this scenario." : "This is what remains after planned expenses."}</p><div className="sim-balance-bar"><span style={{ width: `${Math.min(100, Math.max(0, balance / Math.max(income, 1) * 100))}%` }} /></div></article><div className="panel sim-breakdown"><h3>Scenario breakdown</h3><MiniMetric icon={Banknote} label="Monthly income" value={currency(income)} tone="blue" /><MiniMetric icon={ArrowUpRight} label="Total expenses" value={currency(total)} tone="rose" /><MiniMetric icon={Wallet} label="Potential savings" value={currency(Math.max(balance, 0))} tone="mint" /><MiniMetric icon={PieChart} label="Savings rate" value={percent(income ? Math.max(balance, 0) / income * 100 : 0)} tone="amber" /><div className="sim-disclaimer"><InfoText />Illustrative calculation only. This simulator does not call an AI model or alter your budget.</div></div></section></div></div>;
}

function FeatureTable({ rows, title }) {
  return <ChartCard title={title} subtitle="Signed SHAP contributions for this prediction"><div className="table-scroll"><table className="data-table"><thead><tr><th>Feature</th><th>Input value</th><th className="align-right">SHAP contribution</th></tr></thead><tbody>{rows.map((row, index) => <tr key={`${row.feature}-${index}`}><td><strong>{row.feature}</strong></td><td>{String(row.value ?? "—")}</td><td className={`align-right ${Number(row.shap_value) >= 0 ? "income-text" : "expense-text"}`}>{Number(row.shap_value) >= 0 ? "+" : ""}{Number(row.shap_value).toFixed(4)}</td></tr>)}</tbody></table></div></ChartCard>;
}

function WaterfallChart({ rows, baseValue, predictionValue }) {
  const base = Number(baseValue);
  const contributions = rows
    .filter((row) => Number.isFinite(Number(row.shap_value)))
    .sort((left, right) => Math.abs(Number(right.shap_value)) - Math.abs(Number(left.shap_value)));
  let running = base;
  const steps = contributions.map((row) => {
    const start = running;
    running += Number(row.shap_value);
    return { ...row, start, end: running };
  });
  const output = Number.isFinite(Number(predictionValue)) ? Number(predictionValue) : running;
  const bounds = [base, output, ...steps.flatMap((step) => [step.start, step.end])];
  const minimum = Math.min(...bounds);
  const span = Math.max(Math.max(...bounds) - minimum, 1e-9);
  return (
    <div className="shap-waterfall">
      <div className="waterfall-summary"><span>Baseline <strong>{base.toFixed(3)}</strong></span><span>Model output <strong>{output.toFixed(3)}</strong></span></div>
      <div className="waterfall-list">
        {steps.map((step, index) => {
          const delta = Number(step.shap_value);
          const left = (Math.min(step.start, step.end) - minimum) / span * 100;
          const width = Math.max(Math.abs(delta) / span * 100, delta === 0 ? 0 : 0.6);
          const baseline = (base - minimum) / span * 100;
          return (
            <div className="waterfall-row" key={`${step.feature}-${index}`}>
              <span className="waterfall-feature" title={step.feature}>{step.feature}</span>
              <div className="waterfall-track">
                <i className="waterfall-baseline" style={{ left: `${baseline}%` }} />
                <i className={`waterfall-bar ${delta >= 0 ? "positive" : "negative"}`} style={{ left: `${left}%`, width: `${width}%` }} />
              </div>
              <strong className="waterfall-value">{delta >= 0 ? "+" : ""}{delta.toFixed(3)}</strong>
              <span className="sr-only">{delta >= 0 ? "increases" : "decreases"} the model output</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ExplainabilityPage({ data, mode, onRunModel }) {
  const explanation = data.explainability;
  const [selected, setSelected] = useState("forecast");
  const result = explanation?.[selected];
  const ready = result?.status === "available";
  const featureRows = result?.feature_contributions || result?.individual || [];
  const importance = result?.feature_importance || result?.global_importance || [];
  const requestExplanation = () => {
    const transaction = selected === "categorization"
      ? data.recentTransactions?.[0]
      : selected === "anomaly"
        ? data.anomalies?.items?.[0]
        : null;
    onRunModel("explainability", {
      model: selected,
      transaction_id: transaction?.transaction_id || transaction?.id,
      description: transaction?.description,
      merchant: transaction?.merchant,
    }, "SHAP explanation");
  };
  return <div className="page-stack"><PageHeading eyebrow="Inside the model" title="Explainable AI" description="View SHAP outputs only when the existing model and explanation pipeline return them." actions={<select className="select-button" value={selected} onChange={(event) => setSelected(event.target.value)}><option value="forecast">Expense forecast</option><option value="categorization">Transaction category</option><option value="anomaly">Anomaly score</option></select>} />{ready ? <>{result.is_historical && result.notice && <div className="info-strip info-warning" role="status"><CircleAlert size={16} /><span>{result.notice}</span></div>}<div className="info-strip"><BrainCircuit size={17} /><span>{result.explanation || "SHAP values explain model outputs relative to the selected reference data. They are not causal effects or proof of fraud."}</span></div>{featureRows.length > 0 && result.base_value != null && Number.isFinite(Number(result.base_value)) && <ChartCard title="SHAP waterfall" subtitle={result.score_type || "Cumulative feature contributions from baseline to model output"}><WaterfallChart rows={featureRows} baseValue={result.base_value} predictionValue={result.prediction_value ?? result.prediction ?? result.anomaly_score} /></ChartCard>}<div className="explainability-grid"><ChartCard title="Global feature importance" subtitle="Mean absolute SHAP value">{importance.length ? <FeatureImportanceChart data={importance} /> : <EmptyState title="No global SHAP values" description="This explanation returned local contributions only." />}</ChartCard><ChartCard title="Individual prediction" subtitle={result.prediction_label || result.forecast_month || "Local explanation"}>{featureRows.length ? <div className="contribution-list">{featureRows.slice(0, 8).map((row, index) => <div className="contribution-row" key={`${row.feature}-${index}`}><span>{row.feature}</span><div className="contribution-track"><i className={Number(row.shap_value) >= 0 ? "positive" : "negative"} style={{ width: `${Math.min(100, Math.abs(row.shap_value) * 30 + 8)}%` }} /></div><strong className={Number(row.shap_value) >= 0 ? "income-text" : "expense-text"}>{Number(row.shap_value) >= 0 ? "+" : ""}{Number(row.shap_value).toFixed(3)}</strong></div>)}</div> : <EmptyState title="No individual explanation" description="The API returned no local SHAP values for this request." />}</ChartCard></div>{featureRows.length > 0 && <FeatureTable rows={featureRows} title="Feature contribution details" />}</> : <EmptyState title={result?.status === "stale" ? "Explanation is stale" : result?.status === "unavailable" ? "SHAP explanation unavailable" : "No SHAP output available"} description={result?.reason || data.explainabilityError || "Request an explanation from a supported model. Stale predictions are not explained as current."} action={<Button icon={Activity} disabled={mode !== "live" || data.modelLoading?.explainability} onClick={requestExplanation}>{data.modelLoading?.explainability ? "Explaining..." : "Request explanation"}</Button>} />}</div>;
}

export function SettingsPage({ data, mode, theme, onTheme, onSave, toast }) {
  const [savedSettings] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("finai-settings") || "{}");
    } catch {
      return {};
    }
  });
  const settings = data.settings || savedSettings;
  const [name, setName] = useState(settings.name || "");
  const [email, setEmail] = useState(settings.email || "");
  const [notifications, setNotifications] = useState(settings.notifications ?? true);
  const save = async (event) => {
    event.preventDefault();
    if (await onSave("settings", { ...settings, name, email, notifications })) {
      toast("Preferences saved in this browser; profile details are not synced to an account.");
    }
  };
  return (
    <div className="page-stack">
      <PageHeading eyebrow="Make FinAI yours" title="Settings" description="Manage local preferences; profile and security APIs are not configured." />
      <div className="settings-layout">
        <form className="panel settings-panel" onSubmit={save}>
          <div className="panel-heading"><div><h2>Local profile preferences</h2><p>These details are stored only in this browser.</p></div></div>
          <div className="settings-profile"><span className="profile-avatar large">PF</span><div><strong>{name || "Your profile"}</strong><span>Profile photo upload is not connected</span></div><button className="button button-outline" type="button" onClick={() => toast("Profile photo upload requires an account API.")}>Change photo</button></div>
          <div className="form-grid">
            <label>Full name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label>Email address<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
            <label>Currency<select defaultValue="INR"><option value="INR">Indian Rupee (₹ INR)</option></select></label>
            <label>Region<select defaultValue="en-IN"><option value="en-IN">India</option></select></label>
          </div>
          <div className="settings-divider" />
          <div className="setting-row"><div><strong>Notifications</strong><span>Get updates about budgets, goals and account activity.</span></div><button type="button" role="switch" aria-checked={notifications} className={`toggle ${notifications ? "on" : ""}`} onClick={() => setNotifications((value) => !value)}><i /></button></div>
          <div className="setting-row"><div><strong>Appearance</strong><span>Choose your preferred dashboard theme.</span></div><button type="button" className="button button-outline" onClick={onTheme}>{theme === "light" ? "Switch to dark" : "Switch to light"}</button></div>
          <footer><Button type="submit">Save preferences</Button></footer>
        </form>
        <div className="settings-aside">
          <section className="panel settings-panel"><div className="setting-aside-icon"><LockKeyhole size={18} /></div><h3>Account security</h3><p>Authentication and security APIs are not implemented in this backend.</p><Button variant="outline" onClick={() => toast("Account security requires an authentication API.")}>Manage security</Button></section>
          <section className="panel settings-panel"><div className="setting-aside-icon green"><BellRing size={18} /></div><h3>Data connection</h3><p>{mode === "live" ? "Connected to the configured backend API." : mode === "connecting" ? "Checking backend availability." : "Backend connection is unavailable."}</p><StatusBadge tone={mode === "live" ? "success" : "warning"}>{mode === "live" ? "API connected" : mode === "connecting" ? "Connecting" : "Unavailable"}</StatusBadge></section>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { errorMessage, exportTransactionsCsv } from "../services/api";
import { clearSettings, resizeImage } from "../utils/profile";
import { FORECAST_MONEY_FEATURES, featureLabel, fieldLabel, visibleFields } from "../utils/labels";
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
  Avatar,
  BudgetProgress,
  Button,
  ChartCard,
  EmptyAction,
  EmptyState,
  LoadingSkeleton,
  PageHeading,
  StatusBadge,
  SummaryCard,
  TooltipNote,
} from "../components/UI";
import TransactionTable from "../components/TransactionTable";
import GoalPlanner from "../components/GoalPlanner";
import { ExpenseDonut, FeatureImportanceChart, IncomeExpenseChart, SpendingLineChart } from "../components/Charts";
import { currency, percent, ratioPercent, shortDate } from "../utils/format";

function ChartLegend({ items }) {
  return <div className="chart-legend">{items.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}</div>;
}

function MiniMetric({ icon: Icon, label, value, tone = "blue" }) {
  return <div className="mini-metric"><span className={`mini-metric-icon ${tone}`}><Icon size={16} /></span><span>{label}</span><strong>{value}</strong></div>;
}

export function OverviewPage({ data, mode, refreshing, onRefresh, onNavigate, profile = {}, onRunAll, runningAll = false }) {
  const [trendMonths, setTrendMonths] = useState(6);
  const overview = data.overview || {};
  const monthlyData = (data.monthly || []).slice(-trendMonths);
  const recent = data.recentTransactions || [];
  const categoryTotal = (data.categories || []).reduce((sum, item) => sum + Number(item.value || 0), 0);
  const modeIsLive = mode === "live";
  const period = overview.analysis_period?.month;
  return (
    <div className="page-stack">
      <WelcomeHero overview={overview} profile={profile} mode={mode} runningAll={runningAll} onRunAll={onRunAll} onAdd={() => onNavigate("/transactions", { state: { addTransaction: true } })} />
      {period && <PeriodNote period={overview.analysis_period} />}
      {mode === "live" && data.overviewError && <div className="inline-warning"><CircleAlert size={16} />{data.overviewError}</div>}
      <div className="summary-grid">
        <SummaryCard label="Income" value={overview.income ?? "—"} change={overview.incomeChange ?? undefined} icon={ArrowDownRight} tone="blue" helper={period ? `for ${period}` : "no transaction history"} />
        <SummaryCard label="Spending" value={overview.expenses ?? "—"} change={overview.expenseChange ?? undefined} icon={ArrowUpRight} tone="rose" helper={period ? `for ${period}` : "no transaction history"} />
        <SummaryCard label="Saved" value={overview.savings ?? "—"} change={overview.savingsChange ?? undefined} icon={Wallet} tone="mint" helper="income minus expenses" />
        <SummaryCard label="Budget used" value={overview.budget_utilization == null ? "—" : percent(overview.budget_utilization)} icon={PieChart} tone="amber" helper="of recorded monthly budget" />
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
          <div className="ai-spotlight-head"><div className="ai-spark"><Sparkles size={18} /></div><div><h2>Your AI financial pulse</h2><p>Press “Run AI analysis” above to fill these in with the saved models.</p></div><StatusBadge tone={modeIsLive ? "success" : "neutral"}>{modeIsLive ? "Models ready" : "Unavailable"}</StatusBadge></div>
          <div className="pulse-grid">
            <div className="pulse-item"><span className="pulse-kicker"><TrendingDown size={14} />Expense forecast</span><strong>{data.forecast?.status === "available" ? currency(data.forecast.predicted_expense) : data.forecast?.status ? "Unavailable" : "Not run"}</strong><p>{data.forecast?.status === "available" ? `For ${data.forecast.forecast_month}${data.forecast.is_historical ? ` · from records through ${data.forecast.history_end_month}` : ""}` : data.forecast?.reason || data.forecast?.skip_reason || "Run forecasting to check model availability."}</p><button className="text-button" onClick={() => onNavigate("/forecasting")}>Explore forecast <ArrowRight size={13} /></button></div>
            <div className="pulse-item"><span className="pulse-kicker"><ShieldAlert size={14} />Unusual activity</span><strong>{data.anomalies?.status === "available" ? (data.anomalies.anomaly_count ?? "—") : "Not run"}</strong><p>{data.anomalies?.analysis_period?.selected_month ? `High or medium severity in ${monthTitle(data.anomalies.analysis_period.selected_month)}` : "Run the saved model to review potential outliers."}</p><button className="text-button" onClick={() => onNavigate("/anomalies")}>Review activity <ArrowRight size={13} /></button></div>
            <div className="pulse-item"><span className="pulse-kicker"><Lightbulb size={14} />Smart suggestions</span><strong>{Array.isArray(data.recommendations) ? data.recommendations.length : "Not run"}</strong><p>{data.recommendationsPeriod?.selected_month ? `Analysis period ${data.recommendationsPeriod.selected_month}` : "Request data-backed recommendations."}</p><button className="text-button" onClick={() => onNavigate("/recommendations")}>See suggestions <ArrowRight size={13} /></button></div>
            <div className="pulse-item"><span className="pulse-kicker"><Sparkles size={14} />Spending insights</span><strong>{data.insights?.items ? data.insights.items.length : "Not run"}</strong><p>{data.insights?.items?.[0] ? `${data.insights.items[0].title} (${insightChangeLabel(data.insights.items[0])})` : "Compare each category with earlier months."}</p><button className="text-button" onClick={() => onNavigate("/insights")}>View insights <ArrowRight size={13} /></button></div>
          </div>
        </section>
      </div>
    </div>
  );
}

// "Groceries budget exceeded in 2026-10" -> "... in October 2026"
function humanizeMonths(text) {
  return String(text ?? "").replace(/\b\d{4}-(0[1-9]|1[0-2])\b/g, (month) => monthTitle(month));
}

function greeting() {
  const hour = new Date().getHours();
  return hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";
}

function monthTitle(month) {
  return month ? new Intl.DateTimeFormat("en-IN", { month: "long", year: "numeric" }).format(new Date(`${month}-01T00:00:00`)) : "";
}

function WelcomeHero({ overview, profile, mode, runningAll, onRunAll, onAdd }) {
  const month = overview.analysis_period?.month;
  const income = Number(overview.income || 0);
  const expenses = Number(overview.expenses || 0);
  const savingsRate = income > 0 ? Math.max(income - expenses, 0) / income : null;
  const firstName = String(profile.name || "").trim().split(/\s+/)[0];
  return (
    <section className="welcome-hero">
      <div className="welcome-copy">
        <span className="welcome-kicker"><Sparkles size={14} />{greeting()}{firstName ? `, ${firstName}` : ""}</span>
        <h1>{month ? `Your ${monthTitle(month)} snapshot` : "Your financial overview"}</h1>
        <p>{month
          ? <>You earned <strong>{currency(income)}</strong> and spent <strong>{currency(expenses)}</strong>{savingsRate != null ? <> — a savings rate of <strong>{ratioPercent(savingsRate)}</strong>.</> : "."}</>
          : "Add transactions to see your monthly snapshot."}</p>
        <div className="welcome-chips">
          {overview.budget_utilization != null && <span><PieChart size={13} />Budget used {percent(overview.budget_utilization)}</span>}
          {overview.expenseChange != null && <span>{overview.expenseChange > 0 ? <TrendingUp size={13} /> : <TrendingDown size={13} />}Spending {overview.expenseChange > 0 ? "up" : "down"} {percent(Math.abs(overview.expenseChange))} vs last month</span>}
        </div>
      </div>
      <div className="welcome-actions">
        <button className="button welcome-primary" disabled={mode !== "live" || runningAll} onClick={onRunAll}><BrainCircuit size={16} />{runningAll ? "Analysing..." : "Run AI analysis"}</button>
        <button className="button welcome-secondary" onClick={onAdd}><Plus size={16} />Add transaction</button>
        <button className="button welcome-secondary" onClick={() => window.print()}><Download size={15} />Export report</button>
      </div>
    </section>
  );
}

// Run a page's AI model when the page opens (and again after the month changes),
// once per month, so people see results instead of an empty page.
function useAutoLoad({ mode, result, loading, month, load }) {
  const attempted = useRef(null);
  useEffect(() => {
    const key = month || "default";
    if (mode !== "live" || result !== undefined || loading || attempted.current === key) return;
    attempted.current = key;
    load();
  }, [mode, result, loading, month, load]);
}

function AiLoading({ text }) {
  return <section className="panel ai-loading" aria-live="polite"><div className="ai-loading-head"><span className="ai-spark"><Sparkles size={16} /></span><strong>{text}</strong></div><LoadingSkeleton rows={3} /></section>;
}

function PeriodNote({ period }) {
  const newerIncomeOnly = period.is_default && period.latest_transaction_month && period.latest_transaction_month !== period.month;
  const now = new Date();
  const [year, monthNumber] = String(period.month).split("-").map(Number);
  const monthsAgo = (now.getFullYear() - year) * 12 + (now.getMonth() + 1 - monthNumber);
  if (!newerIncomeOnly && monthsAgo < 2) return null;
  return <div className="info-strip"><CircleAlert size={16} /><span>{newerIncomeOnly
    ? `Showing ${period.month}, the latest month with recorded expenses. Transactions in ${period.latest_transaction_month} contain no expenses yet. Use the month picker at the top to switch periods.`
    : `Showing recorded data for ${period.month}. This is not current live financial activity.`}</span></div>;
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

  if (/percentage/i.test(name) && hasMeaningfulNumber(value)) {
    return percent(value);
  }

  // The recommendation engine reports rates, ratios and utilization as fractions.
  if (/(utilization|ratio|rate|progress|change)/i.test(name) && hasMeaningfulNumber(value)) {
    return ratioPercent(value);
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
  const location = useLocation();
  const navigate = useNavigate();
  const searchInput = useRef(null);
  const [query, setQuery] = useState(() => location.state?.search || "");
  const [type, setType] = useState("All types");
  const [category, setCategory] = useState("All categories");
  const [sort, setSort] = useState("date-new");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState(() => (location.state?.addTransaction ? {} : null));
  const pageSize = 7;
  // Apply a search or "add" request sent from another page, then clear it so a
  // browser refresh does not repeat it.
  useEffect(() => {
    if (!location.state) return;
    if (location.state.search !== undefined) {
      setQuery(location.state.search);
      setPage(1);
      window.setTimeout(() => searchInput.current?.focus(), 0);
    }
    if (location.state.addTransaction) setEditing({});
    navigate(location.pathname, { replace: true, state: null });
  }, [location.state, location.pathname, navigate]);
  const latestBalance = Number((data.recentTransactions || [])[0]?.balance_after_transaction);
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
    const amount = Number(form.get("amount"));
    const enteredBalance = String(form.get("balance_after_transaction") || "").trim();
    // Left blank: continue from the latest recorded balance.
    const balance = enteredBalance !== ""
      ? Number(enteredBalance)
      : (Number.isFinite(latestBalance) ? latestBalance : 0) + (form.get("type") === "Income" ? amount : -amount);
    const payload = { description: form.get("description"), merchant: form.get("merchant"), date: form.get("date"), category: form.get("category"), amount, balance_after_transaction: Math.round(balance * 100) / 100, type: form.get("type") };
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
      <PageHeading eyebrow={data.overview?.analysis_period?.month ? `All transactions · totals for ${monthTitle(data.overview.analysis_period.month)}` : "Your activity"} title="Transactions" description="Track, search and organize your recorded income and spending." actions={<><Button variant="outline" icon={Sparkles} disabled={!rows.length || transactionLoading || mode !== "live" || data.modelLoading?.categorize} onClick={() => onRunModel("categorize", { items: rows.map((row) => ({ id: row.id, description: row.description, merchant: row.merchant })) }, "Transaction categorization")}>{data.modelLoading?.categorize ? "Categorizing..." : "Auto categorize"}</Button><Button icon={Plus} onClick={() => setEditing({})}>Add transaction</Button></>} />
      {data.transactionsError && <div className="inline-warning"><CircleAlert size={16} />{data.transactionsError}</div>}
      <div className="summary-grid summary-grid-small">
        <SummaryCard label="Matching transactions" value={total} format="number" icon={Activity} tone="blue" helper="for selected filters" />
        <SummaryCard label="Income in period" value={data.overview?.income ?? "—"} icon={ArrowDownRight} tone="mint" helper={data.overview?.analysis_period?.month || "stored data"} />
        <SummaryCard label="Expenses in period" value={data.overview?.expenses ?? "—"} icon={ArrowUpRight} tone="rose" helper={data.overview?.analysis_period?.month || "stored data"} />
      </div>
      <section className="panel transactions-panel">
        <div className="filter-toolbar">
          <label className="table-search"><Search size={16} /><input ref={searchInput} placeholder="Search merchant or description" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} /></label>
          <div className="filter-controls"><select aria-label="Transaction type" value={type} onChange={(event) => { setType(event.target.value); setPage(1); }}><option>All types</option><option>Income</option><option>Expense</option></select><select aria-label="Category" value={category} onChange={(event) => { setCategory(event.target.value); setPage(1); }}><option>All categories</option>{categories.map((item) => <option key={item}>{item}</option>)}</select><select aria-label="Sort transactions" value={sort} onChange={(event) => setSort(event.target.value)}><option value="date-new">Newest first</option><option value="date-old">Oldest first</option><option value="amount-high">Amount: high to low</option><option value="amount-low">Amount: low to high</option></select><input aria-label="Start date" type="date" value={startDate} onChange={(event) => { setStartDate(event.target.value); setPage(1); }} /><input aria-label="End date" type="date" value={endDate} onChange={(event) => { setEndDate(event.target.value); setPage(1); }} /></div>
        </div>
        {transactionLoading ? <div className="empty-table">Loading transactions…</div> : rows.length ? <TransactionTable rows={rows} onEdit={setEditing} onDelete={onDelete ? (row) => onDelete("transactions", row.id) : undefined} /> : <EmptyState title="No matching transactions" description={data.transactionsError || "Try adjusting your search or filters."} />}
        <div className="table-footer"><span>Showing {total ? (page - 1) * pageSize + 1 : 0}–{Math.min((page - 1) * pageSize + rows.length, total)} of {total} transactions</span><div className="pagination"><button disabled={page <= 1 || transactionLoading} onClick={() => setPage((current) => current - 1)}>Previous</button><span>{page} / {totalPages}</span><button disabled={page >= totalPages || transactionLoading} onClick={() => setPage((current) => current + 1)}>Next</button></div></div>
      </section>
      {editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal-card" onSubmit={submit}><header><div><h2>{editing.id ? "Edit transaction" : "Add transaction"}</h2><p>Keep your financial activity up to date.</p></div><button type="button" className="icon-btn" onClick={() => setEditing(null)}>×</button></header><div className="form-grid"><label>Description<input name="description" required defaultValue={editing.description || ""} /></label><label>Merchant<input name="merchant" required defaultValue={editing.merchant || ""} /></label><label>Amount (₹)<input name="amount" type="number" min="0.01" step="0.01" required defaultValue={editing.amount || ""} /></label><label>Balance after transaction (₹)<input name="balance_after_transaction" type="number" step="0.01" placeholder={Number.isFinite(latestBalance) ? `Auto from latest balance ${currency(latestBalance)}` : "Optional"} defaultValue={editing.balance_after_transaction ?? ""} /></label><label>Date<input name="date" type="date" required defaultValue={editing.date || new Date().toISOString().slice(0, 10)} /></label><label>Category<select name="category" defaultValue={editing.category || categories[0] || "Other"}>{categories.map((item) => <option key={item}>{item}</option>)}</select></label><label>Type<select name="type" defaultValue={editing.type || "Expense"}><option>Expense</option><option>Income</option></select></label></div><footer><Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button type="submit">{editing.id ? "Save changes" : "Add transaction"}</Button></footer></form></div>}
    </div>
  );
}

export function InsightsPage({ data, mode, onRunModel }) {
  const insights = data.insights;
  useAutoLoad({ mode, result: insights, loading: data.modelLoading?.insights, month: data.overview?.analysis_period?.month, load: () => onRunModel("insights", {}, "AI insights", { quiet: true, reportErrors: true }) });
  return <div className="page-stack"><PageHeading eyebrow={insights?.analysis_period?.selected_month ? monthTitle(insights.analysis_period.selected_month) : "Patterns & opportunities"} title="AI insights" description="How this month’s spending in each category compares with your usual month." actions={<Button variant="outline" icon={RefreshCw} disabled={mode !== "live" || data.modelLoading?.insights} onClick={() => onRunModel("insights", {}, "AI insights")}>{data.modelLoading?.insights ? "Analyzing..." : "Refresh insights"}</Button>} />{insights?.notice && <div className="info-strip" role="status"><CircleAlert size={16} /><span>{insights.notice}</span></div>}{!insights && data.modelLoading?.insights ? <AiLoading text="Comparing this month’s spending with earlier months…" /> : insights?.items?.length ? <div className="insight-grid">{insights.items.map((item, index) => <InsightCard key={item.id || index} item={item} />)}</div> : <EmptyState title={insights?.status === "unavailable" ? "Insights unavailable" : "No AI insights available"} description={insights?.reason || data.insightsError || "Request analysis to check for transaction patterns. No insights are fabricated when the service is unavailable."} action={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.insights} onClick={() => onRunModel("insights", {}, "AI insights")}>Request insights</Button>} />}<div className="info-strip"><BrainCircuit size={17} /><span>Insights compare the month chosen at the top with the three months before it.</span></div></div>;
}

function insightChange(item) {
  const values = item.supporting_values || {};
  const latest = Number(values.latest_month_spending);
  const average = Number(values.historical_monthly_average);
  return Number.isFinite(latest) && Number.isFinite(average) && average ? (latest - average) / Math.abs(average) : null;
}

function insightChangeLabel(item) {
  const change = insightChange(item);
  if (change == null) return "";
  if (Math.abs(change) < 0.005) return "No change";
  return `${change >= 0 ? "+" : "−"}${ratioPercent(Math.abs(change))}`;
}

function InsightCard({ item }) {
  const change = insightChange(item);
  const direction = change == null || Math.abs(change) < 0.005 ? "flat" : change > 0 ? "up" : "down";
  return <article className={`panel insight-card insight-${direction}`}><div className="insight-card-top"><div className="insight-symbol"><Sparkles size={18} /></div>{change != null && <span className={`insight-change ${direction}`}>{direction === "up" ? <TrendingUp size={15} /> : direction === "down" ? <TrendingDown size={15} /> : null}{insightChangeLabel(item)}</span>}</div><StatusBadge tone={item.priority === "High" ? "danger" : item.priority === "Medium" ? "warning" : "success"}>{item.priority || "Insight"}</StatusBadge><h3>{item.title}</h3><p>{humanizeMonths(item.description)}</p>{item.supporting_values?.latest_month_spending != null && <div className="insight-compare"><span>This month <strong>{currency(item.supporting_values.latest_month_spending)}</strong></span><span>Usual month <strong>{currency(item.supporting_values.historical_monthly_average)}</strong></span></div>}</article>;
}

export function ForecastPage({ data, mode, onRunModel, navigate }) {
  const forecast = data.forecast;
  const explanation = data.explainability?.forecast;
  const usable = forecast?.status === "available" && forecast.predicted_expense != null;
  useAutoLoad({ mode, result: forecast, loading: data.modelLoading?.forecast, month: data.overview?.analysis_period?.month, load: () => onRunModel("forecast", {}, "Expense forecast", { quiet: true, reportErrors: true }) });
  const drivers = explanationRows("forecast", explanation?.feature_contributions || []);
  const driverImportance = explanationRows("forecast", (explanation?.feature_importance || []).map((item) => ({ ...item, shap_value: Math.abs(item.importance ?? 0) })))
    .map((item) => ({ feature: item.label, importance: item.shap_value }));
  return <div className="page-stack"><PageHeading eyebrow="Predictive finance" title="Expense forecasting" description="The AI predicts your total spending for the coming month from your recent months." actions={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.forecast} onClick={() => onRunModel("forecast", {}, "Expense forecast")}>{data.modelLoading?.forecast ? "Updating..." : forecast ? "Update forecast" : "Run forecast"}</Button>} />{!forecast && data.modelLoading?.forecast ? <AiLoading text="Predicting next month’s spending…" /> : usable ? <>{forecast.is_historical ? <div className="info-strip info-warning" role="status"><CircleAlert size={16} /><span>{forecast.notice || `Recorded history ends in ${forecast.history_end_month}, so the forecast month ${forecast.forecast_month} has already passed.`}</span></div> : <div className="info-strip"><CircleAlert size={16} /><span>Forecast for the month after your latest recorded expenses ({forecast.history_end_month || forecast.analysis_period?.selected_month}).</span></div>}<div className="forecast-hero"><div className="forecast-main"><div className="forecast-kicker"><Sparkles size={15} />FORECAST <TooltipNote>Predicted by a Random Forest model trained on monthly spending patterns.</TooltipNote></div><strong>{currency(forecast.predicted_expense)}</strong><span>Predicted spending for {monthTitle(forecast.forecast_month)}</span>{forecast.is_historical ? <StatusBadge tone="warning">Based on older records</StatusBadge> : <StatusBadge tone="success">Live forecast</StatusBadge>}</div><div className="forecast-metrics"><MiniMetric icon={Activity} label="Typical error" value={forecast.mae == null ? "—" : `± ${currency(forecast.mae)}`} /><MiniMetric icon={Activity} label="Usually within" value={forecast.mape == null ? "—" : `± ${Math.round(Number(forecast.mape))}%`} /><MiniMetric icon={Activity} label="Based on" value={forecast.history_months ? `${forecast.history_months} months` : "—"} /></div></div><ChartCard title="Your monthly spending" subtitle="Recorded months up to the one selected at the top"><div className="chart-area chart-area-tall"><SpendingLineChart data={data.monthly || []} /></div></ChartCard>{driverImportance.length ? <ChartCard title="What drives this forecast?" subtitle="Average influence of each factor"><FeatureImportanceChart data={driverImportance} valueFormatter={(value) => currency(value)} /></ChartCard> : <EmptyState title={explanation?.status && explanation.status !== "available" ? "Explanation unavailable" : "Why this amount?"} description={explanation?.reason || "See which parts of your recent spending pushed this prediction up or down."} action={<div className="heading-actions"><Button icon={BrainCircuit} disabled={mode !== "live" || data.modelLoading?.explainability} onClick={() => onRunModel("explainability", { model: "forecast" }, "Forecast explanation")}>{data.modelLoading?.explainability ? "Explaining..." : "Explain this forecast"}</Button><Button variant="outline" onClick={() => navigate("/explainability")}>Open Explainable AI</Button></div>} />}{drivers.length > 0 && <FeatureTable rows={drivers} model="forecast" title="This prediction explained" />}</> : <EmptyState title={forecast?.status === "insufficient_history" ? "Not enough history to forecast" : forecast?.status === "unavailable" ? "Forecast unavailable" : "Forecast not loaded yet"} description={forecast?.reason || forecast?.skip_reason || data.forecastError || "Predict next month’s spending from your recent months."} action={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.forecast} onClick={() => onRunModel("forecast", {}, "Expense forecast")}>{data.modelLoading?.forecast ? "Checking..." : "Check forecast status"}</Button>} />}</div>;
}

const REVIEWED_KEY = "finai-reviewed-anomalies";

function readReviewed() {
  try {
    return new Set(JSON.parse(localStorage.getItem(REVIEWED_KEY) || "[]"));
  } catch {
    return new Set();
  }
}

const SEVERITY_TONE = { High: "danger", Medium: "warning", Low: "neutral" };
const ANOMALY_FILTERS = [
  ["important", "High & medium"],
  ["High", "High"],
  ["Medium", "Medium"],
  ["Low", "Low (pattern only)"],
  ["reviewed", "Reviewed"],
];

export function AnomaliesPage({ data, mode, onRunModel, navigate }) {
  const result = data.anomalies;
  const items = result?.items || result?.transactions || [];
  const [filter, setFilter] = useState("important");
  const [reviewed, setReviewed] = useState(readReviewed);
  const counts = result?.severity_counts || { High: 0, Medium: 0, Low: 0 };
  useAutoLoad({ mode, result, loading: data.modelLoading?.anomalies, month: data.overview?.analysis_period?.month, load: () => onRunModel("anomalies", {}, "Anomaly detection", { quiet: true, reportErrors: true }) });
  const month = result?.scope_month || result?.analysis_period?.selected_month;
  const toggleReviewed = (id) => {
    setReviewed((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      try { localStorage.setItem(REVIEWED_KEY, JSON.stringify([...next])); } catch { /* browser storage unavailable */ }
      return next;
    });
  };
  const visible = items.filter((item) => {
    const id = item.transaction_id || item.id;
    if (filter === "reviewed") return reviewed.has(id);
    if (reviewed.has(id)) return false;
    if (filter === "important") return item.severity !== "Low";
    return item.severity === filter;
  });
  const countFor = (key) => key === "reviewed"
    ? items.filter((item) => reviewed.has(item.transaction_id || item.id)).length
    : items.filter((item) => !reviewed.has(item.transaction_id || item.id) && (key === "important" ? item.severity !== "Low" : item.severity === key)).length;
  const analyze = () => onRunModel("anomalies", {}, "Anomaly detection");
  const explain = (item) => {
    navigate("/explainability", { state: { model: "anomaly" } });
    onRunModel("explainability", { model: "anomaly", transaction_id: item.transaction_id || item.id }, "Anomaly explanation");
  };

  return <div className="page-stack">
    <PageHeading eyebrow={month ? monthTitle(month) : "Transaction safety"} title="Anomaly detection" description="Transactions that don’t fit your usual spending, with the reasons why." actions={<Button icon={ShieldAlert} disabled={mode !== "live" || data.modelLoading?.anomalies} onClick={analyze}>{data.modelLoading?.anomalies ? "Analyzing..." : result ? "Re-check" : "Analyze transactions"}</Button>} />
    <div className="info-strip info-warning"><ShieldAlert size={17} /><span>Flags combine the trained Isolation Forest with checks against your own history: amount versus your usual purchase in that category, first payments to a merchant, share of income and repeated identical charges. A flag means “worth a look”, not confirmed fraud.</span></div>
    {!result && data.modelLoading?.anomalies ? <AiLoading text="Checking this month’s transactions…" /> : result?.status === "available" ? <>
      <div className="summary-grid">
        <SummaryCard label="Expenses checked" value={result.total_analyzed ?? 0} format="number" icon={Activity} tone="blue" helper={month ? `in ${month}` : undefined} />
        <SummaryCard label="High severity" value={counts.High ?? 0} format="number" icon={ShieldAlert} tone="rose" helper="clear outliers" />
        <SummaryCard label="Medium severity" value={counts.Medium ?? 0} format="number" icon={CircleAlert} tone="amber" helper="worth a look" />
        <SummaryCard label="Low · pattern only" value={counts.Low ?? 0} format="number" icon={BadgeCheck} tone="mint" helper="model-only, usually fine" />
      </div>
      <div className="filter-chips" role="tablist" aria-label="Filter by severity">
        {ANOMALY_FILTERS.map(([key, label]) => <button key={key} role="tab" aria-selected={filter === key} className={`filter-chip ${filter === key ? "active" : ""}`} onClick={() => setFilter(key)}>{label}<span>{countFor(key)}</span></button>)}
      </div>
      {visible.length ? <div className="anomaly-list">{visible.map((item) => {
        const id = item.transaction_id || item.id;
        const isReviewed = reviewed.has(id);
        return <article key={id} className={`panel anomaly-card severity-${String(item.severity || "Low").toLowerCase()} ${isReviewed ? "is-reviewed" : ""}`}>
          <div className="anomaly-main">
            <span className="merchant-logo">{(item.merchant || "TX").slice(0, 2).toUpperCase()}</span>
            <div className="anomaly-title"><strong>{item.merchant || item.description || "Transaction"}</strong><small>{item.description && item.description !== item.merchant ? `${item.description} · ` : ""}{shortDate(item.transaction_date || item.date)} · {item.category}</small></div>
            <div className="anomaly-amount"><strong>{currency(item.amount)}</strong>{item.usual_category_amount ? <small>usual {currency(item.usual_category_amount)}</small> : null}</div>
          </div>
          <div className="anomaly-tags"><StatusBadge tone={SEVERITY_TONE[item.severity] || "neutral"}>{item.severity || "Low"} severity</StatusBadge><span className="anomaly-source">Detected by {item.detected_by || "model"}</span>{isReviewed && <StatusBadge tone="success">Reviewed</StatusBadge>}</div>
          {item.reasons?.length ? <ul className="anomaly-reasons">{item.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : null}
          <div className="anomaly-actions">
            <span className="anomaly-score">{item.model_flag ? `Model score ${Number(item.anomaly_score).toFixed(3)}` : "Model saw a normal pattern; caught by the checks above"}</span>
            <button className="text-button" onClick={() => navigate("/transactions", { state: { search: item.merchant || item.description || "" } })}>View in transactions</button>
            <button className="text-button" disabled={mode !== "live"} onClick={() => explain(item)}>Why was this flagged? <ArrowRight size={13} /></button>
            <button className={`button ${isReviewed ? "button-outline" : "button-primary"} button-small`} onClick={() => toggleReviewed(id)}>{isReviewed ? "Undo review" : <><Check size={14} />Mark reviewed</>}</button>
          </div>
        </article>;
      })}</div> : <EmptyState title={filter === "reviewed" ? "Nothing reviewed yet" : filter === "important" ? "Nothing stands out this month" : "No transactions in this group"} description={filter === "important" && counts.Low ? `Only ${counts.Low} pattern-only flag(s) from the model; see “Low (pattern only)”.` : "Try another filter or choose a different month at the top."} />}
    </> : <EmptyState title={result?.status === "no_out_of_sample_rows" ? "No expenses to check in this month" : result ? "Anomaly analysis unavailable" : "Check this month’s transactions"} description={result?.status === "no_out_of_sample_rows" ? `The model was trained on data up to ${result.training_end}; choose a later month at the top.` : result?.reason || data.anomaliesError || "Run the anomaly model on the selected month’s expenses."} action={<Button icon={ShieldAlert} disabled={mode !== "live" || data.modelLoading?.anomalies} onClick={analyze}>{data.modelLoading?.anomalies ? "Analyzing..." : "Analyze transactions"}</Button>} />}
  </div>;
}

export function RecommendationsPage({ data, mode, onRunModel, navigate }) {
  const recommendations = data.recommendations || [];
  useAutoLoad({ mode, result: data.recommendations, loading: data.modelLoading?.recommendations, month: data.overview?.analysis_period?.month, load: () => onRunModel("recommendations", {}, "Recommendations", { quiet: true, reportErrors: true }) });
  const priorityCounts = { High: 0, Medium: 0, Low: 0 };
  recommendations.forEach((item) => {
    const priority = item.priority || "Low";
    if (priorityCounts[priority] !== undefined) {
      priorityCounts[priority] += 1;
    } else {
      priorityCounts.Low += 1;
    }
  });

  return <div className="page-stack"><PageHeading eyebrow={data.recommendationsPeriod?.selected_month ? `Based on ${monthTitle(data.recommendationsPeriod.selected_month)}` : "Your next best moves"} title="Recommendations" description="Personal suggestions based on your budgets, spending trends, forecast and goals." actions={<Button icon={RefreshCw} variant="outline" disabled={mode !== "live" || data.modelLoading?.recommendations} onClick={() => onRunModel("recommendations", {}, "Recommendations")}>{data.modelLoading?.recommendations ? "Analyzing..." : "Refresh"}</Button>} />{!data.recommendations && data.modelLoading?.recommendations ? <AiLoading text="Working out your next best moves…" /> : recommendations.length ? <><div className="summary-grid"><SummaryCard label="Total recommendations" value={recommendations.length} format="number" icon={Lightbulb} tone="blue" /><SummaryCard label="High-priority recommendations" value={priorityCounts.High} format="number" icon={ShieldAlert} tone="rose" /><SummaryCard label="Medium-priority recommendations" value={priorityCounts.Medium} format="number" icon={CircleAlert} tone="amber" /><SummaryCard label="Low-priority recommendations" value={priorityCounts.Low} format="number" icon={BadgeCheck} tone="mint" /></div><div className="recommendation-list">{recommendations.map((item) => <RecommendationCard key={item.id || item.title} item={item} onOpen={navigate} />)}</div></> : <EmptyState title="No recommendations available" description={data.recommendationsError || "Request analysis. No suggestions are fabricated when the AI service or required artifacts are unavailable."} action={<Button icon={Sparkles} disabled={mode !== "live" || data.modelLoading?.recommendations} onClick={() => onRunModel("recommendations", {}, "Recommendations")}>Get recommendations</Button>} />}</div>;
}

const RECOMMENDATION_LINKS = {
  budget_warning: ["/budgets", "Review budgets"],
  expense_reduction: ["/insights", "See spending insights"],
  overspending: ["/simulator", "Plan in the simulator"],
  saving_opportunity: ["/simulator", "Plan your savings"],
  future_expense_warning: ["/forecasting", "View forecast"],
  goal_progress: ["/goals", "Open goals"],
  unusual_transaction_review: ["/anomalies", "Review transactions"],
};

function RecommendationCard({ item, onOpen }) {
  const tone = item.priority === "High" ? "danger" : item.priority === "Medium" ? "warning" : "success";
  const [to, label] = RECOMMENDATION_LINKS[item.type || item.recommendation_type] || ["/insights", "See insights"];
  return <article className="panel recommendation-card"><div className={`recommendation-icon ${tone}`}><Lightbulb size={19} /></div><div className="recommendation-content"><div className="recommendation-top"><StatusBadge tone={tone}>{item.priority || "Priority"}</StatusBadge><span className="recommendation-type">{String(item.type || item.recommendation_type || "Recommendation").replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase())}</span></div><h3>{humanizeMonths(item.title || item.recommendation_title)}</h3><p>{humanizeMonths(item.description || item.recommendation_description)}</p>{item.reason && <div className="reason-line"><InfoText />{humanizeMonths(item.reason)}</div>}{item.supporting_values && <div className="support-values">{visibleFields(item.supporting_values).slice(0, 6).map(([name, value]) => <span key={name}>{fieldLabel(name)} <strong>{name === "month" || name === "forecast_month" ? monthTitle(value) : formatRecommendationSupportValue(name, value)}</strong></span>)}</div>}</div><button className="recommendation-link" onClick={() => onOpen(to)} title={label}><span>{label}</span><ArrowRight size={15} /></button></article>;
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
  return <div className="page-stack"><PageHeading eyebrow={budgets[0]?.budget_month ? monthTitle(budgets[0].budget_month) : "Plan with confidence"} title="Budgets" description="Set monthly limits and compare them with recorded expenses." actions={<Button icon={Plus} onClick={() => setEditing({})}>Create budget</Button>} /><div className="summary-grid"><SummaryCard label="Monthly budget" value={total} icon={Wallet} tone="blue" /><SummaryCard label="Spent so far" value={spent} icon={ArrowUpRight} tone="rose" /><SummaryCard label="Remaining" value={Math.max(0, total - spent)} icon={CircleDollarSign} tone="mint" /><SummaryCard label="Overall utilization" value={percent(total ? spent / total * 100 : 0)} icon={PieChart} tone="amber" /></div><div className="budget-page-grid">{budgets.length ? budgets.map((budget) => <article className="panel budget-card" key={budget.id}><div className="budget-card-top"><div><span className="budget-category-icon"><Wallet size={17} /></span><h3>{budget.category}</h3></div><details className="row-menu"><summary aria-label="Budget actions">···</summary><div className="row-menu-pop"><button onClick={() => setEditing(budget)}>Edit budget</button><button className="danger-action" onClick={() => onDelete("budgets", budget.id)}>Delete budget</button></div></details></div><div className="budget-amount">{currency(budget.spent)} <small>spent of {currency(budget.amount)}</small></div><BudgetProgress budget={budget} hideMeta /><div className="budget-card-bottom"><span>{budget.budget_month || "Month not provided"}</span><StatusBadge tone={budget.spent > budget.amount ? "danger" : "success"}>{budget.spent > budget.amount ? "Over budget" : "On track"}</StatusBadge></div></article>) : <EmptyState title="No budgets yet" description="Create a monthly budget to keep spending on track." action={<Button icon={Plus} onClick={() => setEditing({})}>Create budget</Button>} />}</div>{editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal-card" onSubmit={save}><header><div><h2>{editing.id ? "Edit budget" : "Create budget"}</h2><p>Choose a category and month for your limit.</p></div><button type="button" className="icon-btn" onClick={() => setEditing(null)}>×</button></header><div className="form-grid"><label>Category<select name="category" defaultValue={editing.category || "Other"}>{(data.transactionCategories || []).map((item) => <option key={item}>{item}</option>)}</select></label><label>Monthly amount (₹)<input name="amount" type="number" min="1" required defaultValue={editing.amount || ""} /></label><label>Month<input name="month" type="month" required defaultValue={editing.budget_month || budgets[0]?.budget_month || data.overview?.analysis_period?.month || new Date().toISOString().slice(0, 7)} /></label></div><footer><Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button type="submit">Save budget</Button></footer></form></div>}</div>;
}

export function GoalsPage({ data, mode, onSave, onDelete }) {
  const [editing, setEditing] = useState(null);
  const save = async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = { name: form.get("name"), target: Number(form.get("target")), current: Number(form.get("current")), date: form.get("date") };
    if (await onSave("goals", payload, editing?.id)) setEditing(null);
  };
  return <div className="page-stack"><PageHeading eyebrow="Plan ahead" title="Financial goals" description="Track progress towards your savings goals." actions={<Button icon={Plus} onClick={() => setEditing({})}>Create a goal</Button>} /><div className="goal-grid">{(data.goals || []).map((goal) => { const progress = goal.target ? Math.min(100, Math.round(goal.current / goal.target * 100)) : 0; return <article className="panel goal-card" key={goal.id}><div className="goal-card-top"><span className="goal-symbol"><Target size={19} /></span><details className="row-menu"><summary aria-label="Goal actions">···</summary><div className="row-menu-pop"><button onClick={() => setEditing(goal)}>Update goal</button><button className="danger-action" onClick={() => onDelete("goals", goal.id)}>Delete goal</button></div></details></div><h3>{goal.name}</h3><p>Target date · {shortDate(goal.date)}</p><div className="goal-progress"><div className="goal-progress-label"><strong>{currency(goal.current)}</strong><span>{currency(goal.target)} goal</span></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div><div className="goal-progress-label"><span>{progress}% completed</span><span>{currency(Math.max(goal.target - goal.current, 0))} to go</span></div></div><button className="button button-outline button-full" onClick={() => setEditing(goal)}>Update progress</button></article>; })}{!data.goals?.length && <EmptyState title="Start your first goal" description="Create a savings goal and track progress over time." action={<Button icon={Plus} onClick={() => setEditing({})}>Create a goal</Button>} />}</div>{editing && <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setEditing(null)}><form className="modal-card" onSubmit={save}><header><div><h2>{editing.id ? "Update financial goal" : "Create a financial goal"}</h2><p>Make progress visible, one step at a time.</p></div><button type="button" className="icon-btn" onClick={() => setEditing(null)}>×</button></header><div className="form-grid"><label>Goal name<input name="name" required defaultValue={editing.name || ""} /></label><label>Target amount (₹)<input name="target" type="number" min="1" required defaultValue={editing.target || ""} /></label><label>Current savings (₹)<input name="current" type="number" min="0" required defaultValue={editing.current || 0} /></label><label>Target date<input name="date" type="date" required defaultValue={editing.date || ""} /></label></div><footer><Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button type="submit">Save goal</Button></footer></form></div>}</div>;
}

export function SimulatorPage({ data, mode, onSave, toast }) {
  const [income, setIncome] = useState(0);
  const [expenses, setExpenses] = useState({});
  const [edited, setEdited] = useState(false);
  const baselineMonth = data.overview?.analysis_period?.month || null;
  const baselineIncome = Number(data.overview?.income || 0);
  // Compare by value: the dashboard refreshes every 30 seconds and on window
  // focus, which creates new arrays even when nothing changed.
  const baselineKey = JSON.stringify([
    baselineMonth,
    Math.round(baselineIncome),
    (data.categories || []).map((item) => [item.name, Math.round(Number(item.value || 0))]),
  ]);
  const applyBaseline = () => {
    const [, nextIncome, categoryValues] = JSON.parse(baselineKey);
    setIncome(nextIncome);
    setExpenses(Object.fromEntries(categoryValues));
    setEdited(false);
  };
  const appliedBaseline = useRef({ key: null, month: null });
  useEffect(() => {
    if (mode !== "live" || appliedBaseline.current.key === baselineKey) return;
    // Keep the user's scenario when the same month's data merely refreshes;
    // start over when a different month is selected.
    if (!edited || appliedBaseline.current.month !== baselineMonth) applyBaseline();
    appliedBaseline.current = { key: baselineKey, month: baselineMonth };
  }, [mode, baselineKey]);
  const values = Object.values(expenses);
  const total = values.reduce((sum, value) => sum + Number(value), 0);
  const balance = Number(income) - total;
  const updateExpense = (category, value) => {
    setEdited(true);
    setExpenses((current) => ({ ...current, [category]: Math.max(0, Number(value) || 0) }));
  };
  return <div className="page-stack"><PageHeading eyebrow={data.overview?.analysis_period?.month ? `Starting from ${monthTitle(data.overview.analysis_period.month)}` : "Explore the possibilities"} title="What-if simulator" description="Adjust the recorded baseline and compare a hypothetical monthly cash flow." actions={<Button variant="outline" icon={RefreshCw} onClick={applyBaseline}>Reset scenario</Button>} /><div className="simulator-layout"><section className="panel simulator-controls"><div className="panel-heading"><div><h2>Monthly assumptions</h2><p>Changes stay in this browser session; baseline uses recorded values.</p></div></div><label className="sim-input-label">Monthly income <strong>{currency(income)}</strong><input type="range" min="0" max="1000000" step="5000" value={income} onChange={(event) => { setEdited(true); setIncome(Number(event.target.value)); }} /><div className="range-values"><span>₹0</span><span>₹10L</span></div></label><div className="sim-category-list">{Object.entries(expenses).map(([category, amount]) => <label key={category} className="sim-category"><span>{category}</span><div><span>₹</span><input type="number" min="0" step="500" value={amount} onChange={(event) => updateExpense(category, event.target.value)} /></div></label>)}{!Object.keys(expenses).length && <EmptyState title="No expense baseline" description="Transaction category values will appear here when the backend returns recorded expenses." />}</div></section><section className="simulator-results"><article className="sim-balance-card"><span>Hypothetical monthly balance</span><strong className={balance < 0 ? "text-danger" : ""}>{currency(balance)}</strong><p>{balance < 0 ? "Expenses exceed income in this scenario." : "This is what remains after planned expenses."}</p><div className="sim-balance-bar"><span style={{ width: `${Math.min(100, Math.max(0, balance / Math.max(income, 1) * 100))}%` }} /></div></article><div className="panel sim-breakdown"><h3>Scenario breakdown</h3><MiniMetric icon={Banknote} label="Monthly income" value={currency(income)} tone="blue" /><MiniMetric icon={ArrowUpRight} label="Total expenses" value={currency(total)} tone="rose" /><MiniMetric icon={Wallet} label="Potential savings" value={currency(Math.max(balance, 0))} tone="mint" /><MiniMetric icon={PieChart} label="Savings rate" value={ratioPercent(income ? Math.max(balance, 0) / income : 0)} tone="amber" /><div className="sim-disclaimer"><InfoText />Illustrative calculation only. This simulator does not call an AI model or alter your budget.</div></div></section></div><GoalPlanner income={income} expenses={expenses} mode={mode} onApplyCuts={(cuts) => { setEdited(true); setExpenses((current) => ({ ...current, ...cuts })); window.scrollTo({ top: 0, behavior: "smooth" }); toast?.("Suggested cuts applied — see the new monthly balance at the top."); }} onSaveGoal={(payload) => onSave("goals", payload)} /></div>;
}

function FeatureTable({ rows, title, model = "forecast" }) {
  return <ChartCard title={title} subtitle="Each factor’s value and how much it moved the result"><div className="table-scroll"><table className="data-table"><thead><tr><th>Factor</th><th>Value</th><th className="align-right">Effect</th></tr></thead><tbody>{rows.map((row, index) => { const effect = Number(row.shap_value); return <tr key={`${row.label || row.feature}-${index}`}><td><strong>{row.label || featureLabel(row.feature)}</strong></td><td>{factorValue(model, row)}</td><td className={`align-right ${effect >= 0 ? "income-text" : "expense-text"}`}>{effectText(model, effect)}</td></tr>; })}</tbody></table></div></ChartCard>;
}

function WaterfallChart({ rows, baseValue, predictionValue, format = (value) => Number(value).toFixed(3), effect = (value) => `${value >= 0 ? "+" : ""}${Number(value).toFixed(3)}` }) {
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
      <div className="waterfall-summary"><span>Starting point <strong>{format(base)}</strong></span><span>Result <strong>{format(output)}</strong></span></div>
      <div className="waterfall-list">
        {steps.map((step, index) => {
          const delta = Number(step.shap_value);
          const left = (Math.min(step.start, step.end) - minimum) / span * 100;
          const width = Math.max(Math.abs(delta) / span * 100, delta === 0 ? 0 : 0.6);
          const baseline = (base - minimum) / span * 100;
          return (
            <div className="waterfall-row" key={`${step.feature}-${index}`}>
              <span className="waterfall-feature" title={step.label || step.feature}>{step.label || step.feature}</span>
              <div className="waterfall-track">
                <i className="waterfall-baseline" style={{ left: `${baseline}%` }} />
                <i className={`waterfall-bar ${delta >= 0 ? "positive" : "negative"}`} style={{ left: `${left}%`, width: `${width}%` }} />
              </div>
              <strong className="waterfall-value">{effect(delta)}</strong>
              <span className="sr-only">{delta >= 0 ? "increases" : "decreases"} the result</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const EXPLAIN_MODELS = {
  forecast: {
    label: "Expense forecast",
    intro: "Why the AI predicts this spending: each factor pushed the prediction up or down from a typical starting point.",
  },
  categorization: {
    label: "Transaction category",
    intro: "Which words in a transaction made the AI pick its category.",
  },
  anomaly: {
    label: "Unusual transaction",
    intro: "What made a transaction look unusual, or normal, to the anomaly model.",
  },
};

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function explanationRows(model, rows) {
  const merged = new Map();
  for (const row of rows) {
    const label = model === "categorization" ? `“${row.feature}”` : featureLabel(row.feature);
    const existing = merged.get(label);
    if (existing) existing.shap_value += Number(row.shap_value) || 0;
    else merged.set(label, { ...row, label, shap_value: Number(row.shap_value) || 0 });
  }
  return [...merged.values()].sort((left, right) => Math.abs(right.shap_value) - Math.abs(left.shap_value));
}

function effectText(model, value) {
  if (model === "forecast") return `${value >= 0 ? "+" : "−"}${currency(Math.abs(value))}`;
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
}

function factorValue(model, row) {
  const value = row.value;
  if (value == null || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (model === "forecast") {
    if (FORECAST_MONEY_FEATURES.has(row.feature)) return currency(number);
    if (row.feature === "spending_change") return `${number >= 0 ? "+" : ""}${ratioPercent(number)}`;
    if (String(row.feature).endsWith("_vs_avg6")) return `${number.toFixed(2)}×`;
    if (row.feature === "month_number") return MONTH_NAMES[Math.round(number) - 1] || String(number);
  }
  if (model === "categorization") return number.toFixed(3);
  if (row.feature === "is_recurring" || row.feature === "merchant_familiar") return number ? "Yes" : "No";
  return Number.isInteger(number) ? String(number) : number.toFixed(2);
}

function summarySentence(model, row, result) {
  const amount = effectText(model, row.shap_value).slice(1);
  if (model === "forecast") return `${row.label} ${row.shap_value >= 0 ? "raised" : "lowered"} the forecast by ${amount}.`;
  if (model === "categorization") return `The word ${row.label} ${row.shap_value >= 0 ? "pointed towards" : "pointed away from"} ${result.prediction_label}.`;
  return `${row.label} made it look ${row.shap_value >= 0 ? "more" : "less"} unusual.`;
}

export function ExplainabilityPage({ data, mode, onRunModel }) {
  const location = useLocation();
  const explanation = data.explainability;
  const [selected, setSelected] = useState(() => location.state?.model || "forecast");
  const result = explanation?.[selected];
  const ready = result?.status === "available";
  const rows = explanationRows(selected, result?.feature_contributions || result?.individual || []);
  const importance = explanationRows(selected, (result?.feature_importance || result?.global_importance || []).map((item) => ({ ...item, shap_value: Math.abs(item.importance ?? item.value ?? item.mean_absolute_shap ?? 0) })))
    .map((item) => ({ feature: item.label, importance: item.shap_value }));
  const info = EXPLAIN_MODELS[selected];
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
    }, "Explanation");
  };
  const isMoney = selected === "forecast";
  const outcome = selected === "forecast"
    ? result?.prediction != null && `Predicted spending for ${monthTitle(result.forecast_month)}: ${currency(result.prediction)}`
    : selected === "categorization"
      ? result?.prediction_label && `Chosen category: ${result.prediction_label}${result.confidence != null ? ` (${Math.round(result.confidence * 100)}% sure)` : ""}`
      : result?.prediction_label && `Model verdict: ${result.prediction_label}`;

  return <div className="page-stack">
    <PageHeading eyebrow="Why did the AI decide this?" title="Explainable AI" description={info.intro} actions={<div className="heading-actions"><select className="select-button" aria-label="What to explain" value={selected} onChange={(event) => setSelected(event.target.value)}>{Object.entries(EXPLAIN_MODELS).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}</select><Button variant="outline" icon={RefreshCw} disabled={mode !== "live" || data.modelLoading?.explainability} onClick={requestExplanation}>{data.modelLoading?.explainability ? "Explaining..." : "Explain"}</Button></div>} />
    {ready ? <>
      {result.is_historical && result.notice && <div className="info-strip info-warning" role="status"><CircleAlert size={16} /><span>{result.notice}</span></div>}
      {rows.length > 0 && <section className="panel xai-summary">
        <div className="xai-summary-head"><span className="insight-symbol"><BrainCircuit size={18} /></span><div><h2>What the AI decided</h2>{outcome && <p className="xai-outcome">{outcome}</p>}</div></div>
        <ol className="xai-reasons">{rows.slice(0, 3).map((row) => <li key={row.label} className={row.shap_value >= 0 ? "up" : "down"}>{summarySentence(selected, row, result)}</li>)}</ol>
        <small>These show how the model reached its answer from your data; they are not proof of cause{selected === "anomaly" ? " or of fraud" : ""}.</small>
      </section>}
      {rows.length > 0 && result.base_value != null && Number.isFinite(Number(result.base_value)) && <ChartCard title="Step by step" subtitle={isMoney ? "From a typical starting prediction to yours" : "From the model's starting score to this result"}><WaterfallChart rows={rows} baseValue={result.base_value} predictionValue={result.prediction_value ?? result.prediction ?? result.anomaly_score} format={(value) => isMoney ? currency(value) : Number(value).toFixed(3)} effect={(value) => effectText(selected, value)} /></ChartCard>}
      <div className="explainability-grid">
        <ChartCard title="What matters most overall" subtitle="Average influence across your data">{importance.length ? <FeatureImportanceChart data={importance} valueFormatter={(value) => isMoney ? currency(value) : Number(value).toFixed(3)} /> : <EmptyState title="Shown for single predictions only" description="This explanation covers one transaction; see the factors on the right." />}</ChartCard>
        <ChartCard title="This prediction" subtitle={result.prediction_label || (result.forecast_month ? monthTitle(result.forecast_month) : "Single result")}>{rows.length ? <div className="contribution-list">{rows.slice(0, 8).map((row) => { const max = Math.max(...rows.map((item) => Math.abs(item.shap_value)), 1e-9); return <div className="contribution-row" key={row.label}><span title={row.feature}>{row.label}</span><div className="contribution-track"><i className={row.shap_value >= 0 ? "positive" : "negative"} style={{ width: `${Math.max(6, Math.abs(row.shap_value) / max * 100)}%` }} /></div><strong className={row.shap_value >= 0 ? "income-text" : "expense-text"}>{effectText(selected, row.shap_value)}</strong></div>; })}</div> : <EmptyState title="No single-prediction factors" description="Choose another model or request the explanation again." />}</ChartCard>
      </div>
      {rows.length > 0 && <FeatureTable rows={rows} model={selected} title="All factors" />}
    </> : <EmptyState title={result?.status === "unavailable" ? "Explanation unavailable" : `Explain the ${info.label.toLowerCase()}`} description={result?.reason || data.explainabilityError || info.intro} action={<Button icon={BrainCircuit} disabled={mode !== "live" || data.modelLoading?.explainability} onClick={requestExplanation}>{data.modelLoading?.explainability ? "Explaining..." : "Explain"}</Button>} />}
  </div>;
}

export function SettingsPage({ settings = {}, mode, theme, onTheme, onSave, toast, onReset }) {
  const [name, setName] = useState(settings.name || "");
  const [email, setEmail] = useState(settings.email || "");
  const [notifications, setNotifications] = useState(settings.notifications ?? true);
  const [exporting, setExporting] = useState(false);
  const photoInput = useRef(null);
  const current = () => ({ ...settings, name: name.trim(), email: email.trim(), notifications });
  const save = async (event) => {
    event.preventDefault();
    if (await onSave("settings", current())) toast("Preferences saved.");
  };
  const changePhoto = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const photo = await resizeImage(file);
      if (await onSave("settings", { ...current(), photo })) toast("Profile photo updated.");
    } catch (error) {
      toast(error.message, "error");
    }
  };
  const removePhoto = async () => {
    const { photo, ...rest } = current();
    if (await onSave("settings", rest)) toast("Profile photo removed.");
  };
  const exportCsv = async () => {
    setExporting(true);
    try {
      const count = await exportTransactionsCsv();
      toast(`Exported ${count} transactions to a CSV file.`);
    } catch (error) {
      toast(`Export failed: ${errorMessage(error)}`, "error");
    } finally {
      setExporting(false);
    }
  };
  const resetPreferences = () => {
    if (!window.confirm("Reset your name, photo, notification and theme preferences in this browser?")) return;
    clearSettings();
    localStorage.removeItem("finai-theme");
    setName("");
    setEmail("");
    setNotifications(true);
    onReset();
    toast("Preferences reset.");
  };
  return (
    <div className="page-stack">
      <PageHeading eyebrow="Make FinAI yours" title="Settings" description="Your profile, notifications, appearance and data." />
      <div className="settings-layout">
        <form className="panel settings-panel" onSubmit={save}>
          <div className="panel-heading"><div><h2>Profile</h2><p>Shown in the sidebar and top bar of this browser.</p></div></div>
          <div className="settings-profile">
            <Avatar name={name} photo={settings.photo} large />
            <div><strong>{name || "Your profile"}</strong><span>{email || "Add your name and email below"}</span></div>
            <input ref={photoInput} type="file" accept="image/*" hidden onChange={changePhoto} />
            <div className="heading-actions">
              {settings.photo && <button className="button button-ghost" type="button" onClick={removePhoto}>Remove</button>}
              <button className="button button-outline" type="button" onClick={() => photoInput.current?.click()}>Change photo</button>
            </div>
          </div>
          <div className="form-grid">
            <label>Full name<input value={name} placeholder="e.g. Priya Sharma" onChange={(event) => setName(event.target.value)} /></label>
            <label>Email address<input type="email" value={email} placeholder="you@example.com" onChange={(event) => setEmail(event.target.value)} /></label>
            <label>Currency<select defaultValue="INR"><option value="INR">Indian Rupee (₹ INR)</option></select></label>
            <label>Region<select defaultValue="en-IN"><option value="en-IN">India</option></select></label>
          </div>
          <div className="settings-divider" />
          <div className="setting-row"><div><strong>Notifications</strong><span>Show budget, goal and AI alerts under the bell icon.</span></div><button type="button" role="switch" aria-checked={notifications} aria-label="Notifications" className={`toggle ${notifications ? "on" : ""}`} onClick={() => setNotifications((value) => !value)}><i /></button></div>
          <div className="setting-row"><div><strong>Appearance</strong><span>Choose your preferred dashboard theme.</span></div><button type="button" className="button button-outline" onClick={onTheme}>{theme === "light" ? "Switch to dark" : "Switch to light"}</button></div>
          <footer><Button type="submit">Save preferences</Button></footer>
        </form>
        <div className="settings-aside">
          <section className="panel settings-panel"><div className="setting-aside-icon"><LockKeyhole size={18} /></div><h3>Privacy &amp; data</h3><p>Your financial data stays on this computer: FinAI only talks to the local API and database.</p><div className="settings-actions"><Button variant="outline" icon={Download} disabled={mode !== "live" || exporting} onClick={exportCsv}>{exporting ? "Exporting..." : "Export transactions (CSV)"}</Button><Button variant="ghost" icon={RefreshCw} onClick={resetPreferences}>Reset preferences</Button></div></section>
          <section className="panel settings-panel"><div className="setting-aside-icon green"><BellRing size={18} /></div><h3>Data connection</h3><p>{mode === "live" ? "Connected to the FinAI backend and AI model service." : mode === "connecting" ? "Checking backend availability." : "Backend connection is unavailable."}</p><StatusBadge tone={mode === "live" ? "success" : "warning"}>{mode === "live" ? "API connected" : mode === "connecting" ? "Connecting" : "Unavailable"}</StatusBadge></section>
        </div>
      </div>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router-dom";
import {
  AnomaliesPage,
  BudgetsPage,
  ExplainabilityPage,
  ForecastPage,
  GoalsPage,
  InsightsPage,
  OverviewPage,
  RecommendationsPage,
  SettingsPage,
  SimulatorPage,
  TransactionsPage,
} from "./pages/Pages.jsx";
import Navbar from "./components/Navbar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import HelpGuide from "./components/HelpGuide.jsx";
import { EmptyState, LoadingSkeleton } from "./components/UI.jsx";
import {
  checkBackend,
  createResource,
  deleteResource,
  errorMessage,
  getResource,
  runModel,
  updateResource,
} from "./services/api.js";
import { Check, CircleAlert, X } from "lucide-react";
import { buildNotifications } from "./utils/notifications";
import { currency } from "./utils/format";
import { readSettings, writeSettings } from "./utils/profile";

const resourceNames = [
  "overview",
  "transactions",
  "transactionCategories",
  "budgets",
  "goals",
  "periods",
];

// AI results belong to the month they were computed for.
const monthScopedResults = {
  forecast: undefined,
  insights: undefined,
  anomalies: undefined,
  recommendations: undefined,
  recommendationsPeriod: undefined,
  explainability: undefined,
};

function overviewFields(result) {
  return {
    overview: result,
    monthly: result.monthly || [],
    categories: result.categories || [],
    recentTransactions: result.transactions || [],
    activity: result.activity || null,
  };
}

const MONTH_KEY = "finai-selected-month";

function readSelectedMonth() {
  try {
    return localStorage.getItem(MONTH_KEY) || null;
  } catch {
    return null;
  }
}

function storeSelectedMonth(month) {
  try {
    if (month) localStorage.setItem(MONTH_KEY, month);
    else localStorage.removeItem(MONTH_KEY);
  } catch {
    // Browser storage unavailable: the choice lasts for this visit only.
  }
}

// Dashboard resources that follow the month picker.
function periodParams(name, month) {
  return month && (name === "overview" || name === "budgets") ? { month } : {};
}

function normalizeResource(name, payload) {
  if (name === "transactions" || name === "budgets" || name === "goals" || name === "recommendations" || name === "insights") {
    if (Array.isArray(payload)) return payload;
    return payload?.items || payload?.[name] || [];
  }
  return payload;
}

function titleFor(pathname) {
  if (pathname === "/") return "Overview";
  return pathname
    .split("/")
    .filter(Boolean)[0]
    ?.replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase()) || "Overview";
}

export default function App() {
  const [mode, setMode] = useState("connecting");
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(true);
  const [resourceLoading, setResourceLoading] = useState({});
  const [resourceErrors, setResourceErrors] = useState({});
  const [theme, setTheme] = useState(() => localStorage.getItem("finai-theme") || "light");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [search, setSearch] = useState("");
  const [transactionMeta, setTransactionMeta] = useState({ total: 0, limit: 7, offset: 0 });
  const [transactionLoading, setTransactionLoading] = useState(false);
  const transactionQuery = useRef({ limit: 7, offset: 0 });
  const transactionRequest = useRef(0);
  const refreshInFlight = useRef(false);
  const [refreshing, setRefreshing] = useState(false);
  const [toast, setToast] = useState(null);
  const [toastKind, setToastKind] = useState("success");
  // The month picked in the top bar is remembered across reloads.
  const [selectedMonth, setSelectedMonth] = useState(readSelectedMonth);
  const selectedMonthRef = useRef(selectedMonth);
  const [settings, setSettings] = useState(readSettings);
  const [helpOpen, setHelpOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const notify = useCallback((message, kind = "success") => {
    setToastKind(kind);
    setToast(message);
    window.clearTimeout(notify.timeout);
    notify.timeout = window.setTimeout(() => setToast(null), 4200);
  }, []);

  const loadData = useCallback(async () => {
    setLoading(true);
    setMode("connecting");
    setResourceErrors({});
    try {
      await checkBackend();
      setMode("live");
      const results = await Promise.allSettled(resourceNames.map((name) => getResource(
        name,
        periodParams(name, selectedMonthRef.current),
      )));
      const nextData = {};
      const errors = {};
      results.forEach((result, index) => {
        const name = resourceNames[index];
        if (result.status === "fulfilled") {
          if (name === "overview") {
            Object.assign(nextData, overviewFields(result.value));
          } else if (name === "transactions") {
            nextData.transactions = normalizeResource(name, result.value);
            nextData.transactionMeta = {
              total: result.value.total || 0,
              limit: result.value.limit || 7,
              offset: result.value.offset || 0,
            };
          } else if (name === "transactionCategories") {
            nextData.transactionCategories = normalizeResource(name, result.value);
          } else {
            nextData[name] = normalizeResource(name, result.value);
          }
        } else {
          errors[`${name}Error`] = `${name} endpoint unavailable: ${errorMessage(result.reason)}`;
        }
      });
      setData(nextData);
      setTransactionMeta(nextData.transactionMeta || { total: 0, limit: 7, offset: 0 });
      setResourceErrors(errors);
    } catch (error) {
      setMode("unavailable");
      setData({});
      setResourceErrors({
        connectionError: `Backend connection failed: ${errorMessage(error)}`,
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("finai-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 4200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const currentData = useMemo(
    () => ({ ...data, ...resourceErrors, modelLoading: resourceLoading }),
    [data, resourceErrors, resourceLoading],
  );
  const overviewData = useMemo(() => {
    if (!search) return currentData;
    const term = search.toLowerCase();
    return {
      ...currentData,
      recentTransactions: (currentData.recentTransactions || []).filter((row) =>
        `${row.description} ${row.merchant} ${row.category} ${row.id}`.toLowerCase().includes(term),
      ),
    };
  }, [currentData, search]);

  const fetchTransactions = useCallback(async (params = {}) => {
    transactionQuery.current = params;
    const requestId = ++transactionRequest.current;
    setTransactionLoading(true);
    try {
      const result = await getResource("transactions", params);
      if (requestId !== transactionRequest.current) return false;
      setData((current) => ({ ...current, transactions: result.items || [] }));
      setTransactionMeta({
        total: result.total || 0,
        limit: result.limit || params.limit || 7,
        offset: result.offset || params.offset || 0,
      });
      setResourceErrors((current) => {
        const { transactionsError, ...rest } = current;
        return rest;
      });
      return true;
    } catch (error) {
      if (requestId !== transactionRequest.current) return false;
      setResourceErrors((current) => ({
        ...current,
        transactionsError: `Transactions unavailable: ${errorMessage(error)}`,
      }));
      return false;
    } finally {
      if (requestId === transactionRequest.current) setTransactionLoading(false);
    }
  }, []);

  const refreshLiveData = useCallback(async () => {
    if (mode !== "live" || loading || refreshInFlight.current) return;
    refreshInFlight.current = true;
    setRefreshing(true);
    try {
      const [overviewResult, budgetsResult, periodsResult] = await Promise.allSettled([
        getResource("overview", periodParams("overview", selectedMonthRef.current)),
        getResource("budgets", periodParams("budgets", selectedMonthRef.current)),
        getResource("periods"),
      ]);
      setResourceErrors((current) => {
        const next = { ...current };
        if (overviewResult.status === "fulfilled") {
          delete next.overviewError;
        } else {
          next.overviewError = `Activity refresh failed: ${errorMessage(overviewResult.reason)}`;
        }
        if (budgetsResult.status === "fulfilled") {
          delete next.budgetsError;
        } else {
          next.budgetsError = `Budgets refresh failed: ${errorMessage(budgetsResult.reason)}`;
        }
        return next;
      });
      if (overviewResult.status === "fulfilled") {
        setData((current) => ({ ...current, ...overviewFields(overviewResult.value) }));
      }
      if (periodsResult.status === "fulfilled") {
        setData((current) => ({ ...current, periods: periodsResult.value }));
      }
      if (budgetsResult.status === "fulfilled") {
        setData((current) => ({
          ...current,
          budgets: normalizeResource("budgets", budgetsResult.value),
        }));
      }
      if (location.pathname === "/transactions") {
        await fetchTransactions(transactionQuery.current);
      }
    } finally {
      refreshInFlight.current = false;
      setRefreshing(false);
    }
  }, [fetchTransactions, loading, location.pathname, mode]);

  useEffect(() => {
    if (mode !== "live" || loading) return undefined;
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") refreshLiveData();
    };
    const interval = window.setInterval(refreshWhenVisible, 30000);
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [loading, mode, refreshLiveData]);

  const changeMonth = useCallback(async (month) => {
    const next = month || null;
    if (next === selectedMonthRef.current) return;
    selectedMonthRef.current = next;
    setSelectedMonth(next);
    storeSelectedMonth(next);
    setData((current) => ({ ...current, ...monthScopedResults }));
    try {
      const [result, budgets] = await Promise.all([
        getResource("overview", periodParams("overview", next)),
        getResource("budgets", periodParams("budgets", next)),
      ]);
      if (selectedMonthRef.current !== next) return;
      setData((current) => ({
        ...current,
        ...overviewFields(result),
        budgets: normalizeResource("budgets", budgets),
      }));
      setResourceErrors((current) => {
        const { overviewError, ...rest } = current;
        return rest;
      });
    } catch (error) {
      if (selectedMonthRef.current !== next) return;
      setResourceErrors((current) => ({
        ...current,
        overviewError: `Period change failed: ${errorMessage(error)}`,
      }));
    }
  }, []);

  // Forget a remembered month that no longer has data (e.g. after removing generated rows).
  useEffect(() => {
    const items = data.periods?.items;
    if (selectedMonth && items && !items.some((item) => item.month === selectedMonth)) changeMonth(null);
  }, [data.periods, selectedMonth, changeMonth]);

  const saveResource = useCallback(async (resource, payload, id) => {
    if (resource === "settings") {
      try {
        writeSettings(payload);
        setSettings(payload);
        return true;
      } catch (error) {
        notify(`Preferences could not be saved in this browser: ${errorMessage(error)}`, "error");
        return false;
      }
    }
    try {
      if (mode !== "live") throw new Error("Connect to the backend before saving financial data.");
      const response = id
        ? await updateResource(resource, id, payload)
        : await createResource(resource, payload);
      if (resource === "budgets" || resource === "goals") {
        try {
          const refreshed = await getResource(resource, periodParams(resource, selectedMonthRef.current));
          setData((current) => ({
            ...current,
            [resource]: normalizeResource(resource, refreshed),
          }));
          setResourceErrors((current) => {
            const { [`${resource}Error`]: _error, ...rest } = current;
            return rest;
          });
          notify("Changes saved.");
        } catch (refreshError) {
          const savedItem = response?.item || response;
          setData((current) => {
            const existing = current[resource] || [];
            const updated = id
              ? existing.map((item) => item.id === id ? { ...item, ...savedItem } : item)
              : [savedItem, ...existing];
            return { ...current, [resource]: updated };
          });
          setResourceErrors((current) => ({
            ...current,
            [`${resource}Error`]: `Saved successfully, but the list could not be refreshed: ${errorMessage(refreshError)}`,
          }));
          notify("Saved successfully, but the list could not be refreshed.", "error");
        }
        return true;
      }
      setData((current) => {
        const existing = current[resource] || [];
        const savedItem = response?.item || response || payload;
        const updated = id
          ? existing.map((item) => item.id === id ? { ...item, ...savedItem } : item)
          : [savedItem, ...existing];
        return { ...current, [resource]: updated };
      });
      notify("Changes saved.");
      return true;
    } catch (error) {
      notify(errorMessage(error), "error");
      return false;
    }
  }, [mode, notify]);

  const removeResource = useCallback(async (resource, id) => {
    if (mode !== "live") {
      notify("Connect to the backend before deleting financial data.", "error");
      return false;
    }
    try {
      await deleteResource(resource, id);
      setData((current) => ({
        ...current,
        [resource]: (current[resource] || []).filter((item) => item.id !== id),
      }));
      notify("Item deleted.");
      return true;
    } catch (error) {
      notify(errorMessage(error), "error");
      return false;
    }
  }, [mode, notify]);

  const executeModel = useCallback(async (modelName, payload = {}, label = "AI model", { quiet = false, reportErrors = false } = {}) => {
    if (mode !== "live") {
      notify(`${label} did not run: connect a live backend first.`, "error");
      return null;
    }
    setResourceLoading((current) => ({ ...current, [modelName]: true }));
    try {
      const month = selectedMonthRef.current;
      const result = await runModel(
        modelName,
        modelName === "categorize" || !month ? payload : { ...payload, month },
      );
      setData((current) => {
        if (modelName === "forecast") {
          return {
            ...current,
            forecast: result.forecast || result,
            explainability: {
              ...(current.explainability || {}),
              forecast: result.explanation || result.shap || current.explainability?.forecast,
            },
          };
        }
        if (modelName === "anomalies") return { ...current, anomalies: result };
        if (modelName === "recommendations") return { ...current, recommendations: result.items || result.recommendations || [], recommendationsPeriod: result.analysis_period };
        if (modelName === "explainability") {
          const model = payload.model || "forecast";
          return {
            ...current,
            explainability: { ...(current.explainability || {}), [model]: result },
          };
        }
        if (modelName === "insights") return { ...current, insights: result };
        if (modelName === "categorize") {
          const replacements = result.items || result.transactions || [];
          return {
            ...current,
            transactions: (current.transactions || []).map((row) => {
              const replacement = replacements.find((item) => item.id === row.id || item.transaction_id === row.id);
              return replacement ? { ...row, suggested_category: replacement.category, category_confidence: replacement.confidence, category_needs_review: Boolean(replacement.needs_review) } : row;
            }),
          };
        }
        return current;
      });
      const reason = result.reason || result.skip_reason;
      if (quiet) {
        // The caller reports a combined summary.
      } else if (result.status === "unavailable") {
        notify(`${label} unavailable: ${reason || "The model did not return results."}`, "error");
      } else if (result.status === "insufficient_history") {
        notify(`${label} needs more history: ${reason || "Not enough recorded months."}`, "error");
      } else if (result.status === "stale") {
        notify(`${label} is stale: ${reason || "no current prediction is shown."}`, "error");
      } else if (result.status === "no_insights") {
        notify(`${label}: ${reason || "No spending changes were found for the latest recorded month."}`);
      } else if (modelName === "categorize") {
        notify("Category suggestions received; they have not been saved to your transactions.");
      } else if (result.is_historical) {
        notify(`${label} updated using records through ${result.history_end_month || result.analysis_period?.selected_month}.`);
      } else {
        notify(`${label} updated.`);
      }
      return result;
    } catch (error) {
      if (!quiet || reportErrors) notify(`${label} failed: ${errorMessage(error)}`, "error");
      return null;
    } finally {
      setResourceLoading((current) => ({ ...current, [modelName]: false }));
    }
  }, [mode, notify]);

  const [runningAll, setRunningAll] = useState(false);
  const runAllModels = useCallback(async () => {
    if (mode !== "live") {
      notify("AI analysis did not run: connect a live backend first.", "error");
      return;
    }
    setRunningAll(true);
    try {
      const [forecast, insights, anomalies, recommendations] = await Promise.all([
        executeModel("forecast", {}, "Expense forecast", { quiet: true }),
        executeModel("insights", {}, "AI insights", { quiet: true }),
        executeModel("anomalies", {}, "Anomaly detection", { quiet: true }),
        executeModel("recommendations", {}, "Recommendations", { quiet: true }),
      ]);
      const parts = [];
      const forecastMonth = forecast?.forecast_month
        ? new Intl.DateTimeFormat("en-IN", { month: "long", year: "numeric" }).format(new Date(`${forecast.forecast_month}-01T00:00:00`))
        : "";
      if (forecast?.status === "available") parts.push(`forecast ${currency(forecast.predicted_expense)} for ${forecastMonth}`);
      if (insights?.items?.length) parts.push(`${insights.items.length} insights`);
      if (anomalies?.status === "available") parts.push(`${anomalies.anomaly_count} transactions to review`);
      if (recommendations?.items?.length) parts.push(`${recommendations.items.length} recommendations`);
      const failed = [forecast, insights, anomalies, recommendations].filter((result) => !result || result.status === "unavailable").length;
      notify(
        parts.length
          ? `AI analysis complete: ${parts.join(" · ")}.${failed ? ` ${failed} model(s) were unavailable.` : ""}`
          : "AI analysis finished, but no model returned results. Check that the AI service is running.",
        parts.length ? "success" : "error",
      );
    } finally {
      setRunningAll(false);
    }
  }, [executeModel, mode, notify]);

  const handleDelete = async (resource, id) => {
    if (window.confirm("Delete this item? This action cannot be undone.")) {
      const removed = await removeResource(resource, id);
      if (removed && resource === "transactions") {
        await fetchTransactions(transactionQuery.current);
      }
    }
  };

  const handleProfile = () => navigate("/settings");
  const handleSearchSubmit = (value) => navigate("/transactions", { state: { search: value.trim() } });
  const notifications = useMemo(() => buildNotifications(data), [data]);

  // Close open "…" row menus after choosing an action or clicking elsewhere.
  useEffect(() => {
    const closeMenus = (event) => {
      document.querySelectorAll("details.row-menu[open]").forEach((menu) => {
        if (!menu.contains(event.target) || event.target.closest(".row-menu-pop button")) menu.open = false;
      });
    };
    document.addEventListener("click", closeMenus);
    return () => document.removeEventListener("click", closeMenus);
  }, []);
  const themeToggle = () => setTheme((value) => value === "light" ? "dark" : "light");
  const recommendationCount = Array.isArray(currentData.recommendations) ? currentData.recommendations.length : 0;

  return (
    <div className="app-shell">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        collapsed={sidebarCollapsed}
        onCollapse={() => setSidebarCollapsed((value) => !value)}
        recommendationCount={recommendationCount}
        profile={settings}
        onWorkspace={handleProfile}
        onHelp={() => setHelpOpen(true)}
      />
      <div className={`app-main ${sidebarCollapsed ? "main-expanded" : ""}`}>
        <Navbar
          onMenu={() => setSidebarOpen(true)}
          onSearch={setSearch}
          onSearchSubmit={handleSearchSubmit}
          onTheme={themeToggle}
          theme={theme}
          mode={mode}
          notifications={notifications}
          notificationsEnabled={settings.notifications ?? true}
          onNavigate={navigate}
          onProfile={handleProfile}
          profile={settings}
          analysisPeriod={currentData.overview?.analysis_period?.month}
          periods={currentData.periods}
          selectedMonth={selectedMonth}
          onSelectMonth={changeMonth}
        />
        <main className="main-content">
          <div className="content-inner">
            {mode === "unavailable" && <div className="inline-warning" role="alert"><CircleAlert size={16} />{currentData.connectionError}<button className="text-button" onClick={loadData}>Retry</button></div>}
            {mode === "live" && Object.keys(resourceErrors).length > 0 && <div className="inline-warning" role="status"><CircleAlert size={16} />Some backend data is unavailable: {Object.values(resourceErrors).join(" · ")}</div>}
            {loading ? <LoadingSkeleton rows={5} /> : (
              <Routes>
                <Route path="/" element={<OverviewPage data={overviewData} mode={mode} refreshing={refreshing} onRefresh={refreshLiveData} onNavigate={navigate} profile={settings} onRunAll={runAllModels} runningAll={runningAll} />} />
                <Route path="/transactions" element={<TransactionsPage data={currentData} mode={mode} transactionMeta={transactionMeta} transactionLoading={transactionLoading} onFetchTransactions={fetchTransactions} onSave={saveResource} onDelete={handleDelete} onRunModel={executeModel} toast={notify} />} />
                <Route path="/insights" element={<InsightsPage data={currentData} mode={mode} onRunModel={executeModel} />} />
                <Route path="/forecasting" element={<ForecastPage data={currentData} mode={mode} onRunModel={executeModel} navigate={navigate} />} />
                <Route path="/anomalies" element={<AnomaliesPage data={currentData} mode={mode} onRunModel={executeModel} navigate={navigate} />} />
                <Route path="/recommendations" element={<RecommendationsPage data={currentData} mode={mode} onRunModel={executeModel} navigate={navigate} />} />
                <Route path="/budgets" element={<BudgetsPage data={currentData} mode={mode} onSave={saveResource} onDelete={handleDelete} />} />
                <Route path="/goals" element={<GoalsPage data={currentData} mode={mode} onSave={saveResource} onDelete={handleDelete} />} />
                <Route path="/simulator" element={<SimulatorPage data={currentData} mode={mode} onSave={saveResource} toast={notify} />} />
                <Route path="/explainability" element={<ExplainabilityPage data={currentData} mode={mode} onRunModel={executeModel} />} />
                <Route path="/settings" element={<SettingsPage data={currentData} settings={settings} mode={mode} theme={theme} onTheme={themeToggle} onSave={saveResource} toast={notify} onReset={() => { setSettings({}); setTheme("light"); }} />} />
                <Route path="*" element={<EmptyState title="Page not found" description="This FinAI page doesn’t exist." action={<button className="button button-primary" onClick={() => navigate("/")}>Go to overview</button>} />} />
              </Routes>
            )}
          </div>
        </main>
      </div>
      {helpOpen && <HelpGuide onClose={() => setHelpOpen(false)} onNavigate={navigate} />}
      {toast && <div className={`toast ${toastKind}`} role="status">{toastKind === "error" ? <CircleAlert size={17} /> : <Check size={17} />}{toast}<button onClick={() => setToast(null)} aria-label="Dismiss notification"><X size={15} /></button></div>}
      <div className="sr-only" aria-live="polite">{titleFor(location.pathname)}</div>
    </div>
  );
}

import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 20000,
  headers: { "Content-Type": "application/json" },
});

const aiRequestTimeout = 130000;

export const resourcePaths = {
  transactions: "/api/transactions",
  insights: "/api/ai/insights",
  forecast: "/api/ai/forecast",
  anomalies: "/api/ai/anomalies",
  recommendations: "/api/ai/recommendations",
  budgets: "/api/budgets",
  goals: "/api/goals",
  explainability: "/api/ai/explanations",
  periods: "/api/dashboard/periods",
};

export async function checkBackend() {
  const { data } = await api.get("/api/health");
  return data;
}

export async function getResource(name, params = {}) {
  if (name === "overview") {
    const periodParams = params.month ? { month: params.month } : {};
    const [summary, monthly, categories, transactions] = await Promise.all([
      api.get("/api/dashboard/summary", { params: periodParams }),
      api.get("/api/dashboard/monthly-trends"),
      api.get("/api/dashboard/category-breakdown", { params: periodParams }),
      api.get("/api/dashboard/activity"),
    ]);
    const activity = transactions.data;
    return {
      ...summary.data,
      monthly: monthly.data.items || [],
      categories: categories.data.items || [],
      activity,
      transactions: activity.items || [],
    };
  }
  if (name === "transactionCategories") {
    const { data } = await api.get("/api/transactions/categories");
    return data.items || [];
  }
  const path = resourcePaths[name];
  if (!path) throw new Error(`Unknown API resource: ${name}`);
  const { data } = await api.get(path, { params });
  return data;
}

export async function createResource(name, payload) {
  const path = resourcePaths[name];
  if (!path) throw new Error(`Unknown API resource: ${name}`);
  const { data } = await api.post(path, payload);
  return data;
}

export async function updateResource(name, id, payload) {
  const path = resourcePaths[name];
  if (!path) throw new Error(`Unknown API resource: ${name}`);
  const { data } = await api.put(
    `${path}/${encodeURIComponent(id)}`,
    payload,
  );
  return data;
}

export async function deleteResource(name, id) {
  const path = resourcePaths[name];
  if (!path) throw new Error(`Unknown API resource: ${name}`);
  const { data } = await api.delete(`${path}/${encodeURIComponent(id)}`);
  return data;
}

export async function runModel(modelName, payload = {}) {
  const endpoints = {
    categorize: "/api/ai/categorize",
    forecast: "/api/ai/forecast",
    anomalies: "/api/ai/anomalies",
    recommendations: "/api/ai/recommendations",
    explainability: "/api/ai/explanations",
    insights: "/api/ai/insights",
  };
  const endpoint = endpoints[modelName];
  if (!endpoint) throw new Error(`Unknown AI action: ${modelName}`);
  const response = modelName === "categorize"
    ? await api.post(endpoint, payload, { timeout: aiRequestTimeout })
    : await api.get(endpoint, { params: payload, timeout: aiRequestTimeout });
  return response.data;
}

export function errorMessage(error) {
  const detail = error?.response?.data?.detail || error?.response?.data?.message;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg).filter(Boolean).join("; ");
  }
  return error?.message || "The request could not be completed.";
}

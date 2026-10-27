const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  const bodyText = await response.text();
  let body = null;
  try { body = bodyText ? JSON.parse(bodyText) : null; } catch { body = bodyText; }
  if (!response.ok) {
    const message = body?.error?.message || body?.message || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return body;
}

export const getHealth = () => apiRequest("/health");
export const getMeta = () => apiRequest("/meta");
export const getExamples = () => apiRequest("/examples");
export const getExperiments = () => apiRequest("/experiments");

// Keep the legacy endpoint available for compatibility and direct API comparisons.
export const runLegacyBacktest = (payload) => apiRequest("/backtest", {
  method: "POST",
  body: JSON.stringify(payload)
});

// The job endpoint persists the run lifecycle but still executes synchronously.
// Unwrap its result so existing UI components keep consuming the same result shape.
export const runBacktest = async (payload) => {
  const job = await apiRequest("/execution-jobs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  if (job?.status === "succeeded" && job.result) return job.result;
  if (job?.status === "failed") {
    throw new Error(job.error?.message || "Backtest execution failed.");
  }
  throw new Error(job?.error?.message || "Execution job returned an unexpected response.");
};

export const generateAlphaIdeas = (payload) => apiRequest("/generate-alpha", { method: "POST", body: JSON.stringify(payload) });

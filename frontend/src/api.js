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
export const getExecutionJobs = (limit = 30) => apiRequest(`/execution-jobs?limit=${encodeURIComponent(limit)}`);
export const getExecutionJob = (jobId) => apiRequest(`/execution-jobs/${encodeURIComponent(jobId)}`);
export const cancelExecutionJob = (jobId) => apiRequest(`/execution-jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
export const submitExecutionJob = (payload) => apiRequest("/execution-jobs", { method: "POST", body: JSON.stringify(payload) });

// Keep the legacy endpoint available for compatibility and direct API comparisons.
export const runLegacyBacktest = (payload) => apiRequest("/backtest", {
  method: "POST",
  body: JSON.stringify(payload)
});

// Submit a durable background job, then poll its persisted lifecycle.
export const runBacktest = async (payload) => {
  let job = await submitExecutionJob(payload);
  const deadline = Date.now() + 30 * 60 * 1000;
  while (job?.status === "queued" || job?.status === "running") {
    if (Date.now() >= deadline) {
      throw new Error(`Backtest is still ${job.status}. Check Runs & history for job ${job.job_id}.`);
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
    job = await getExecutionJob(job.job_id);
  }
  if (job?.status === "succeeded" && job.result) return job.result;
  if (job?.status === "failed") throw new Error(job.error?.message || "Backtest execution failed.");
  if (job?.status === "cancelled") throw new Error("Backtest execution was cancelled.");
  throw new Error(job?.error?.message || "Execution job returned an unexpected response.");
};

export const generateAlphaIdeas = (payload) => apiRequest("/generate-alpha", { method: "POST", body: JSON.stringify(payload) });

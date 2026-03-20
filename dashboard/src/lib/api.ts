const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch<T>(path: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API}${path}`, { cache: "no-store" });
    if (!res.ok) return fallback;
    return res.json();
  } catch {
    return fallback;
  }
}

// Portfolio
export const fetchPortfolioSummary = () => apiFetch<any>("/api/portfolio/summary", null);
export const fetchPositions = () => apiFetch<any[]>("/api/portfolio/positions", []);
export const fetchPortfolioHistory = () => apiFetch<any[]>("/api/portfolio/history", []);
export const fetchAllocation = () => apiFetch<any>("/api/portfolio/allocation", { bySector: [], byStrategy: [], byDirection: [] });

// Agents
export const fetchAgents = () => apiFetch<any[]>("/api/agents", []);
export const fetchAgent = (id: string) => apiFetch<any>(`/api/agents/${id}`, null);
export const fetchAgentNetwork = () => apiFetch<{ nodes: any[]; edges: any[] }>("/api/agents/network", { nodes: [], edges: [] });
export const fetchLeaderboard = () => apiFetch<any[]>("/api/agents/leaderboard", []);

// Pipeline
export const fetchPipelineStatus = () => apiFetch<any>("/api/pipeline/status", {});
export const fetchPipelineRecent = () => apiFetch<any[]>("/api/pipeline/recent", []);
export const fetchPipelineFunnel = () => apiFetch<any>("/api/pipeline/funnel", { scanned: 0, surfaced: 0, voted: 0, approved: 0 });
export const triggerScan = () =>
  fetch(`${API}/api/pipeline/scan`, { method: "POST" }).then((r) => r.json()).catch(() => null);
export const fetchScanStatus = (id: string) => apiFetch<any>(`/api/pipeline/scan/${id}`, { status: "unknown" });

// Risk
export const fetchRiskSummary = () => apiFetch<any>("/api/risk/summary", null);
export const fetchAlerts = () => apiFetch<any[]>("/api/risk/alerts", []);
export const fetchControls = () => apiFetch<any>("/api/risk/controls", {});
export const toggleKillSwitch = (activate: boolean) =>
  fetch(`${API}/api/risk/killswitch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ activate, operator: "dashboard_user" }),
  }).then((r) => r.json()).catch(() => null);
export const resolveAlert = (id: number) =>
  fetch(`${API}/api/risk/alerts/${id}/resolve`, { method: "POST" }).then((r) => r.json()).catch(() => null);

// Market
export const fetchMarketStatus = () => apiFetch<any>("/api/market/status", { ibkrConnected: false, marketOpen: false });
export const fetchRegime = () => apiFetch<any>("/api/market/regime", { regime: "unknown" });

// Health
export const fetchHealth = () => apiFetch<any>("/api/health", { status: "offline" });

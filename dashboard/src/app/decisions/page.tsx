"use client";

import { useState, useMemo } from "react";
import {
  Search,
  Trophy,
  Loader2,
} from "lucide-react";
import { Badge } from "@/components/shared/Badge";
import { PipelineFunnel } from "@/components/decisions/PipelineFunnel";
import { AgentFlowViz } from "@/components/decisions/AgentFlowViz";
import { AgentCard } from "@/components/decisions/AgentCard";
import { AgentDetail } from "@/components/decisions/AgentDetail";
import {
  fetchPipelineFunnel,
  fetchAgents,
  fetchLeaderboard,
  fetchPipelineRecent,
} from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import type { AgentProfile, Strategy } from "@/types";

// ============================================================
// Tab definitions
// ============================================================

type Tab = "flow" | "agents" | "leaderboard";

const tabs: { key: Tab; label: string }[] = [
  { key: "flow", label: "Flow" },
  { key: "agents", label: "Agents" },
  { key: "leaderboard", label: "Leaderboard" },
];

// ============================================================
// Flow Tab
// ============================================================

function FlowTab() {
  const { data: funnel, loading: loadingFunnel } = useApi(fetchPipelineFunnel, 15000);
  const { data: agents, loading: loadingAgents } = useApi(fetchAgents, 30000);
  const { data: recentRuns } = useApi(fetchPipelineRecent, 15000);

  const funnelData = funnel || { scanned: 0, surfaced: 0, voted: 0, approved: 0 };

  // Transform funnel to the shape PipelineFunnel expects
  const pipelineFunnel = {
    universeScanned: funnelData.scanned || 0,
    shortlisted: funnelData.surfaced || 0,
    highConviction: funnelData.surfaced || 0,
    specialistReviewed: funnelData.voted || 0,
    approved: funnelData.approved || 0,
    rejected: Math.max(0, (funnelData.voted || 0) - (funnelData.approved || 0)),
    noTrade: 0,
    downsized: 0,
  };

  // Build peer groups from agents
  const peerGroups = useMemo(() => {
    if (!agents || agents.length === 0) return [];
    const groupMap: Record<string, any> = {};
    for (const a of agents) {
      const pg = a.peerGroup || "ungrouped";
      if (!groupMap[pg]) {
        groupMap[pg] = {
          name: pg,
          agentCount: 0,
          agents: [],
          specialty: a.primaryStrategy || "",
          avgScore: 0,
          scannedCount: 0,
          surfacedCount: 0,
          passedCount: 0,
          avgConviction: 0,
          recentHitRate: 0,
        };
      }
      groupMap[pg].agentCount++;
      groupMap[pg].agents.push(a.agentId);
      groupMap[pg].avgScore += (a.score?.compositeWeight || 0);
    }
    return Object.values(groupMap).map((g: any) => ({
      ...g,
      avgScore: g.agentCount > 0 ? g.avgScore / g.agentCount : 0,
    }));
  }, [agents]);

  // Build agent profiles for flow viz
  const agentProfiles = useMemo(() => {
    if (!agents) return [];
    return agents.map((a: any) => ({
      ...a,
      health: {
        agentId: a.agentId,
        status: a.score?.status || "healthy",
        reason: "",
        compositeWeight: a.score?.compositeWeight || 1,
        regimeMismatch: false,
        trueDecay: false,
        isRedundant: false,
        redundantWith: "",
        nOutcomes: a.score?.nOutcomes || 0,
        daysSinceLastCorrect: null,
        recommendation: "",
      },
      recentProposals: [],
      approvedCount: 0,
      rejectedCount: 0,
      profitableCount: 0,
      losingCount: 0,
    }));
  }, [agents]);

  if (loadingFunnel && loadingAgents) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
        <span className="ml-2 text-sm text-slate-400">Loading pipeline data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PipelineFunnel funnel={pipelineFunnel} />

      {agentProfiles.length > 0 && (
        <AgentFlowViz
          groups={peerGroups}
          agents={agentProfiles}
          proposals={[]}
          decisions={[]}
        />
      )}

      {/* Recent Pipeline Runs */}
      {recentRuns && recentRuns.length > 0 && (
        <div>
          <h3
            className="mb-3 text-xs font-semibold uppercase tracking-wider"
            style={{ color: "var(--text-muted)" }}
          >
            Recent Pipeline Runs
          </h3>
          <div className="space-y-2">
            {recentRuns.map((run: any, i: number) => (
              <div
                key={run.scan_id || i}
                className="flex items-center justify-between rounded-xl border px-4 py-3 shadow-sm"
                style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
              >
                <div className="flex items-center gap-4">
                  <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                    {run.scan_id || `Run #${i + 1}`}
                  </span>
                  <Badge variant={run.status === "completed" ? "success" : run.status === "running" ? "info" : "neutral"} size="sm">
                    {run.status}
                  </Badge>
                  {run.regime && (
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {run.regime.replace(/_/g, " ")}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-4 text-xs" style={{ color: "var(--text-muted)" }}>
                  {run.proposals_generated != null && <span>{run.proposals_generated} proposals</span>}
                  {run.approved != null && <span>{run.approved} approved</span>}
                  {run.started_at && <span>{new Date(run.started_at).toLocaleString()}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(!recentRuns || recentRuns.length === 0) && !loadingFunnel && (
        <div className="flex items-center justify-center rounded-xl border border-slate-100 bg-white py-12 shadow-sm">
          <p className="text-sm text-slate-400">No pipeline runs yet. Trigger a scan to see decisions flow through.</p>
        </div>
      )}
    </div>
  );
}

// ============================================================
// Agents Tab
// ============================================================

const allStrategies: Strategy[] = [
  "momentum", "mean_reversion", "value", "growth",
  "event_driven", "volatility", "sentiment", "breakout",
];

function AgentsTab() {
  const { data: agents, loading } = useApi(fetchAgents, 30000);
  const [selectedAgent, setSelectedAgent] = useState<AgentProfile | null>(null);
  const [filterStrategy, setFilterStrategy] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterGroup, setFilterGroup] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const agentProfiles: AgentProfile[] = useMemo(() => {
    if (!agents) return [];
    return agents.map((a: any) => ({
      ...a,
      health: {
        agentId: a.agentId,
        status: a.score?.status || "healthy",
        reason: "",
        compositeWeight: a.score?.compositeWeight || 1,
        regimeMismatch: false,
        trueDecay: false,
        isRedundant: false,
        redundantWith: "",
        nOutcomes: a.score?.nOutcomes || 0,
        daysSinceLastCorrect: null,
        recommendation: "",
      },
      recentProposals: [],
      approvedCount: 0,
      rejectedCount: 0,
      profitableCount: Math.round((a.score?.hitRate || 0) * (a.score?.nOutcomes || 0)),
      losingCount: Math.round((1 - (a.score?.hitRate || 0)) * (a.score?.nOutcomes || 0)),
    }));
  }, [agents]);

  const peerGroupNames = useMemo(
    () => [...new Set(agentProfiles.map((a) => a.peerGroup))],
    [agentProfiles]
  );

  const filteredAgents = useMemo(() => {
    return agentProfiles.filter((agent) => {
      if (filterStrategy !== "all" && agent.primaryStrategy !== filterStrategy) return false;
      if (filterStatus !== "all" && agent.health.status !== filterStatus) return false;
      if (filterGroup !== "all" && agent.peerGroup !== filterGroup) return false;
      if (searchQuery && !agent.displayName.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });
  }, [agentProfiles, filterStrategy, filterStatus, filterGroup, searchQuery]);

  const selectStyle = {
    borderColor: "var(--border)",
    background: "var(--bg-card)",
    color: "var(--text-secondary)",
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
        <span className="ml-2 text-sm text-slate-400">Loading 121 agents...</span>
      </div>
    );
  }

  return (
    <div>
      {/* Filter bar */}
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2"
            style={{ color: "var(--text-muted)" }}
          />
          <input
            type="text"
            placeholder="Search agents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-8 rounded-lg border pl-8 pr-3 text-xs focus:outline-none"
            style={{
              borderColor: "var(--border)",
              background: "var(--bg-card)",
              color: "var(--text-primary)",
            }}
          />
        </div>

        <select
          value={filterStrategy}
          onChange={(e) => setFilterStrategy(e.target.value)}
          className="h-8 rounded-lg border px-2 text-xs focus:outline-none"
          style={selectStyle}
        >
          <option value="all">All Strategies</option>
          {allStrategies.map((s) => (
            <option key={s} value={s}>{s.replace("_", " ")}</option>
          ))}
        </select>

        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="h-8 rounded-lg border px-2 text-xs focus:outline-none"
          style={selectStyle}
        >
          <option value="all">All Statuses</option>
          <option value="healthy">Healthy</option>
          <option value="warning">Warning</option>
          <option value="underperforming">Underperforming</option>
          <option value="replace">Replace</option>
        </select>

        <select
          value={filterGroup}
          onChange={(e) => setFilterGroup(e.target.value)}
          className="h-8 rounded-lg border px-2 text-xs focus:outline-none"
          style={selectStyle}
        >
          <option value="all">All Groups</option>
          {peerGroupNames.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>

        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
          {filteredAgents.length} of {agentProfiles.length} agents
        </span>
      </div>

      {/* Agent grid */}
      <div className="grid grid-cols-3 gap-4 xl:grid-cols-4 2xl:grid-cols-5">
        {filteredAgents.map((agent) => (
          <AgentCard
            key={agent.agentId}
            agent={agent}
            onClick={() => setSelectedAgent(agent)}
          />
        ))}
      </div>

      {filteredAgents.length === 0 && (
        <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-muted)" }}>
          No agents match the selected filters.
        </div>
      )}

      <AgentDetail
        agent={selectedAgent}
        open={selectedAgent !== null}
        onClose={() => setSelectedAgent(null)}
      />
    </div>
  );
}

// ============================================================
// Leaderboard Tab
// ============================================================

function LeaderboardTab() {
  const { data: agents, loading } = useApi(fetchLeaderboard, 30000);

  const sorted = useMemo(() => {
    if (!agents || agents.length === 0) return [];
    return [...agents].sort(
      (a: any, b: any) => (b.score?.compositeWeight || 0) - (a.score?.compositeWeight || 0)
    );
  }, [agents]);

  const bottom = sorted.slice(-5).reverse();

  const rankAccent = (rank: number) => {
    if (rank === 1) return "#FFD700";
    if (rank === 2) return "#C0C0C0";
    if (rank === 3) return "#CD7F32";
    return "var(--text-muted)";
  };

  function metricColor(value: number, invert = false) {
    const v = invert ? 1 - value : value;
    if (v >= 0.75) return "#10B981";
    if (v >= 0.6) return "#4F46E5";
    if (v >= 0.45) return "#F59E0B";
    return "#EF4444";
  }

  const statusVariant: Record<string, "success" | "warning" | "danger" | "neutral"> = {
    healthy: "success",
    warning: "warning",
    underperforming: "danger",
    replace: "danger",
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
        <span className="ml-2 text-sm text-slate-400">Loading leaderboard...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div
        className="rounded-xl border shadow-sm overflow-auto"
        style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
      >
        <table className="w-full text-xs">
          <thead className="sticky top-0 z-10" style={{ background: "var(--bg-card)" }}>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Rank", "Agent", "Strategy", "Score", "Weight", "Hit Rate", "Calibration", "Uniqueness", "Regime Fit", "Drawdown", "Stability", "Status"].map((h, i) => (
                <th
                  key={h}
                  className={`px-3 py-2.5 font-semibold uppercase tracking-wider ${
                    i >= 3 && i <= 10 ? "text-right" : i === 11 ? "text-center" : "text-left"
                  }`}
                  style={{ color: "var(--text-muted)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((agent: any, i: number) => {
              const rank = i + 1;
              const score = agent.score || {};
              const hitRate = score.hitRate || 0;
              const isTop3 = rank <= 3;

              return (
                <tr
                  key={agent.agentId}
                  className="transition-colors"
                  style={{
                    borderBottom: "1px solid var(--border)",
                    background: isTop3 ? "rgba(248, 250, 252, 0.5)" : "transparent",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-card-hover)")}
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background = isTop3
                      ? "rgba(248, 250, 252, 0.5)"
                      : "transparent")
                  }
                >
                  <td className="px-3 py-2.5 font-bold tabular-nums" style={{ color: rankAccent(rank) }}>
                    {rank <= 3 ? (
                      <span className="flex items-center gap-1">
                        <Trophy size={12} />
                        {rank}
                      </span>
                    ) : (
                      rank
                    )}
                  </td>
                  <td className="px-3 py-2.5 font-semibold" style={{ color: "var(--text-primary)" }}>
                    {agent.displayName}
                  </td>
                  <td className="px-3 py-2.5">
                    <Badge variant="neutral" size="sm">
                      {(agent.primaryStrategy || "").replace("_", " ")}
                    </Badge>
                  </td>
                  <td className="px-3 py-2.5 text-right font-bold tabular-nums" style={{ color: metricColor(score.riskAdjustedReturn || 0) }}>
                    {((score.riskAdjustedReturn || 0) * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-semibold tabular-nums" style={{ color: "var(--text-primary)" }}>
                    {((score.compositeWeight || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5 text-right font-semibold tabular-nums" style={{ color: metricColor(hitRate) }}>
                    {(hitRate * 100).toFixed(0)}%
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(score.calibrationQuality || 0) }}>
                    {((score.calibrationQuality || 0) * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(score.uniqueness || 0) }}>
                    {((score.uniqueness || 0) * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(score.regimeEffectiveness || 0) }}>
                    {((score.regimeEffectiveness || 0) * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(score.drawdownBehavior || 0) }}>
                    {((score.drawdownBehavior || 0) * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(score.stability || 0) }}>
                    {((score.stability || 0) * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    <Badge
                      variant={statusVariant[score.status || "healthy"] || "neutral"}
                      size="sm"
                    >
                      {score.status || "healthy"}
                    </Badge>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {sorted.length === 0 && (
        <div className="flex items-center justify-center rounded-xl border border-slate-100 bg-white py-12 shadow-sm">
          <p className="text-sm text-slate-400">No agent data yet. Agents will appear after the first pipeline run.</p>
        </div>
      )}

      {/* Bottom Performers */}
      {bottom.length > 0 && (
        <div>
          <h3
            className="mb-3 text-xs font-semibold uppercase tracking-wider"
            style={{ color: "var(--negative)" }}
          >
            Bottom Performers
          </h3>
          <div
            className="rounded-xl border shadow-sm overflow-auto"
            style={{ borderColor: "rgba(239, 68, 68, 0.2)", background: "var(--bg-card)" }}
          >
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Agent", "Strategy", "Weight", "Score", "Status"].map((h) => (
                    <th
                      key={h}
                      className={`px-3 py-2.5 font-semibold uppercase tracking-wider ${
                        h === "Weight" || h === "Score" ? "text-right" : h === "Status" ? "text-center" : "text-left"
                      }`}
                      style={{ color: "var(--text-muted)" }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bottom.map((agent: any) => {
                  const score = agent.score || {};
                  return (
                    <tr
                      key={agent.agentId}
                      className="transition-colors"
                      style={{ borderBottom: "1px solid var(--border)" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-card-hover)")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    >
                      <td className="px-3 py-2.5 font-semibold" style={{ color: "var(--text-primary)" }}>
                        {agent.displayName}
                      </td>
                      <td className="px-3 py-2.5">
                        <Badge variant="neutral" size="sm">
                          {(agent.primaryStrategy || "").replace("_", " ")}
                        </Badge>
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: "var(--negative)" }}>
                        {((score.compositeWeight || 0) * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: "var(--negative)" }}>
                        {((score.riskAdjustedReturn || 0) * 100).toFixed(0)}
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        <Badge
                          variant={statusVariant[score.status || "healthy"] || "neutral"}
                          size="sm"
                        >
                          {score.status || "healthy"}
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// Main Page
// ============================================================

export default function DecisionsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("flow");

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
          Decision Intelligence
        </h1>
        <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
          How the system discovers, evaluates, and approves trading opportunities
        </p>
      </div>

      <div className="mb-6 flex items-center gap-1" style={{ borderBottom: "1px solid var(--border)" }}>
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider transition-colors"
            style={{
              color: activeTab === tab.key ? "var(--text-primary)" : "var(--text-muted)",
              borderBottom: activeTab === tab.key ? "2px solid var(--accent)" : "2px solid transparent",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "flow" && <FlowTab />}
      {activeTab === "agents" && <AgentsTab />}
      {activeTab === "leaderboard" && <LeaderboardTab />}
    </div>
  );
}

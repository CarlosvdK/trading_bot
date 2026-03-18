"use client";

import { useState, useMemo } from "react";
import {
  ChevronDown,
  ChevronUp,
  Search,
  Trophy,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import { Badge } from "@/components/shared/Badge";
import { ProgressBar } from "@/components/shared/ProgressBar";
import { PipelineFunnel } from "@/components/decisions/PipelineFunnel";
import { AgentFlowViz } from "@/components/decisions/AgentFlowViz";
import { AgentCard } from "@/components/decisions/AgentCard";
import { AgentDetail } from "@/components/decisions/AgentDetail";
import {
  mockPipelineFunnel,
  mockPeerGroups,
  mockProposals,
  mockDecisionOutputs,
  mockAgentProfiles,
} from "@/data/mock";
import { formatDateTime, formatCurrency } from "@/lib/utils";
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
// Helper components
// ============================================================

function stageBadge(stage: string) {
  const map: Record<string, { variant: "success" | "warning" | "danger" | "info" | "neutral"; label: string }> = {
    approved: { variant: "success", label: "Approved" },
    portfolio: { variant: "success", label: "Portfolio" },
    global_vote: { variant: "info", label: "Global Vote" },
    specialist: { variant: "warning", label: "Specialist" },
    proposal: { variant: "neutral", label: "Proposal" },
    rejected: { variant: "danger", label: "Rejected" },
  };
  const config = map[stage] || { variant: "neutral" as const, label: stage };
  return <Badge variant={config.variant} size="sm">{config.label}</Badge>;
}

// ============================================================
// Flow Tab
// ============================================================

function FlowTab() {
  const [expandedDecision, setExpandedDecision] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      {/* Funnel overview */}
      <PipelineFunnel funnel={mockPipelineFunnel} />

      {/* n8n-style flow visualization */}
      <AgentFlowViz
        groups={mockPeerGroups}
        agents={mockAgentProfiles}
        proposals={mockProposals}
        decisions={mockDecisionOutputs}
      />

      {/* Recent Decisions */}
      <div>
        <h3
          className="mb-3 text-xs font-semibold uppercase tracking-wider"
          style={{ color: "var(--text-muted)" }}
        >
          Recent Decisions
        </h3>
        <div className="space-y-2">
          {mockDecisionOutputs.map((dec) => {
            const isExpanded = expandedDecision === dec.decisionId;
            return (
              <div
                key={dec.decisionId}
                className="rounded-xl border shadow-sm"
                style={{
                  borderColor: "var(--border)",
                  background: "var(--bg-card)",
                }}
              >
                {/* Summary row */}
                <button
                  className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors rounded-xl"
                  style={{ background: "transparent" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-card-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  onClick={() =>
                    setExpandedDecision(isExpanded ? null : dec.decisionId)
                  }
                >
                  <div className="flex items-center gap-4">
                    <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                      {dec.asset}
                    </span>
                    <Badge variant={dec.direction === "long" ? "success" : "danger"} size="sm">
                      {dec.direction.toUpperCase()}
                    </Badge>
                    <Badge variant="default" size="sm">
                      {dec.investmentType.replace("_", " ")}
                    </Badge>
                    <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {formatDateTime(dec.timestamp)}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-xs font-semibold tabular-nums" style={{ color: "var(--text-secondary)" }}>
                      {(dec.confidence * 100).toFixed(0)}% conf
                    </span>
                    <span className="text-xs tabular-nums" style={{ color: "var(--text-muted)" }}>
                      {formatCurrency(dec.capitalAllocated)}
                    </span>
                    {isExpanded ? (
                      <ChevronUp size={14} style={{ color: "var(--text-muted)" }} />
                    ) : (
                      <ChevronDown size={14} style={{ color: "var(--text-muted)" }} />
                    )}
                  </div>
                </button>

                {/* Expanded detail */}
                {isExpanded && (
                  <div className="px-4 py-4" style={{ borderTop: "1px solid var(--border)" }}>
                    <div className="grid grid-cols-3 gap-6">
                      {/* Thesis & Logic */}
                      <div>
                        <h4
                          className="text-[10px] font-semibold uppercase tracking-wider mb-2"
                          style={{ color: "var(--text-muted)" }}
                        >
                          Thesis
                        </h4>
                        <p className="text-xs leading-relaxed mb-3" style={{ color: "var(--text-secondary)" }}>
                          {dec.thesis}
                        </p>
                        <div className="space-y-1.5 text-[11px]">
                          <div>
                            <span style={{ color: "var(--text-muted)" }}>Catalyst: </span>
                            <span style={{ color: "var(--text-secondary)" }}>{dec.catalyst}</span>
                          </div>
                          <div>
                            <span style={{ color: "var(--text-muted)" }}>Entry: </span>
                            <span style={{ color: "var(--text-secondary)" }}>{dec.entryLogic}</span>
                          </div>
                          <div>
                            <span style={{ color: "var(--text-muted)" }}>Exit: </span>
                            <span style={{ color: "var(--text-secondary)" }}>{dec.exitLogic}</span>
                          </div>
                          <div>
                            <span style={{ color: "var(--text-muted)" }}>Stop: </span>
                            <span style={{ color: "var(--negative)" }}>{dec.stopLogic}</span>
                          </div>
                          <div>
                            <span style={{ color: "var(--text-muted)" }}>Target: </span>
                            <span style={{ color: "var(--positive)" }}>{dec.targetLogic}</span>
                          </div>
                        </div>
                      </div>

                      {/* Voting & Confidence */}
                      <div>
                        <h4
                          className="text-[10px] font-semibold uppercase tracking-wider mb-2"
                          style={{ color: "var(--text-muted)" }}
                        >
                          Voting
                        </h4>
                        <div className="space-y-2">
                          <ProgressBar
                            label="Weighted Vote"
                            value={dec.weightedVoteResult}
                            showValue
                            size="md"
                            color="#10B981"
                          />
                          <ProgressBar
                            label="Specialist Confidence"
                            value={dec.specialistConfidence}
                            showValue
                            size="md"
                            color="#4F46E5"
                          />
                          <ProgressBar
                            label="Overall Confidence"
                            value={dec.confidence}
                            showValue
                            size="md"
                            color="#8B5CF6"
                          />
                        </div>
                        {dec.dissentSummary && (
                          <div className="mt-3">
                            <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Dissent:</span>
                            <p className="text-[11px] mt-0.5" style={{ color: "var(--warning)" }}>
                              {dec.dissentSummary}
                            </p>
                          </div>
                        )}
                        {dec.mainObjections.length > 0 && (
                          <div className="mt-2">
                            <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Objections:</span>
                            <ul className="mt-0.5 space-y-0.5">
                              {dec.mainObjections.map((obj, i) => (
                                <li key={i} className="text-[11px]" style={{ color: "var(--negative)" }}>
                                  - {obj}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Risk & Vehicle */}
                      <div>
                        <h4
                          className="text-[10px] font-semibold uppercase tracking-wider mb-2"
                          style={{ color: "var(--text-muted)" }}
                        >
                          Risk & Sizing
                        </h4>
                        <div className="space-y-1.5 text-[11px]">
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Position Size</span>
                            <span className="font-medium tabular-nums" style={{ color: "var(--text-secondary)" }}>
                              {dec.positionSizePct}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Capital</span>
                            <span className="font-medium tabular-nums" style={{ color: "var(--text-secondary)" }}>
                              {formatCurrency(dec.capitalAllocated)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Max Loss</span>
                            <span className="font-medium tabular-nums" style={{ color: "var(--negative)" }}>
                              {dec.maxLossPct}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Annual Vol</span>
                            <span className="font-medium tabular-nums" style={{ color: "var(--text-secondary)" }}>
                              {(dec.expectedAnnualVol * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Vehicle</span>
                            <span className="font-medium" style={{ color: "var(--accent)" }}>
                              {dec.selectedVehicle.replace("_", " ")}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Regime</span>
                            <span style={{ color: "var(--text-secondary)" }}>
                              {dec.currentRegime.replace(/_/g, " ")}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span style={{ color: "var(--text-muted)" }}>Regime Fit</span>
                            <Badge
                              variant={
                                dec.regimeFit === "strong"
                                  ? "success"
                                  : dec.regimeFit === "moderate"
                                  ? "warning"
                                  : "danger"
                              }
                              size="sm"
                            >
                              {dec.regimeFit}
                            </Badge>
                          </div>
                        </div>
                        {dec.keyRisks.length > 0 && (
                          <div className="mt-3">
                            <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Key Risks:</span>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {dec.keyRisks.map((risk, i) => (
                                <Badge key={i} variant="neutral" size="sm">
                                  {risk}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
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
  const [selectedAgent, setSelectedAgent] = useState<AgentProfile | null>(null);
  const [filterStrategy, setFilterStrategy] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterGroup, setFilterGroup] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const peerGroupNames = useMemo(
    () => [...new Set(mockAgentProfiles.map((a) => a.peerGroup))],
    []
  );

  const filteredAgents = useMemo(() => {
    return mockAgentProfiles.filter((agent) => {
      if (filterStrategy !== "all" && agent.primaryStrategy !== filterStrategy) return false;
      if (filterStatus !== "all" && agent.health.status !== filterStatus) return false;
      if (filterGroup !== "all" && agent.peerGroup !== filterGroup) return false;
      if (searchQuery && !agent.displayName.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });
  }, [filterStrategy, filterStatus, filterGroup, searchQuery]);

  const selectStyle = {
    borderColor: "var(--border)",
    background: "var(--bg-card)",
    color: "var(--text-secondary)",
  };

  return (
    <div>
      {/* Filter bar */}
      <div className="mb-5 flex flex-wrap items-center gap-3">
        {/* Search */}
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

        {/* Strategy filter */}
        <select
          value={filterStrategy}
          onChange={(e) => setFilterStrategy(e.target.value)}
          className="h-8 rounded-lg border px-2 text-xs focus:outline-none"
          style={selectStyle}
        >
          <option value="all">All Strategies</option>
          {allStrategies.map((s) => (
            <option key={s} value={s}>
              {s.replace("_", " ")}
            </option>
          ))}
        </select>

        {/* Status filter */}
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

        {/* Group filter */}
        <select
          value={filterGroup}
          onChange={(e) => setFilterGroup(e.target.value)}
          className="h-8 rounded-lg border px-2 text-xs focus:outline-none"
          style={selectStyle}
        >
          <option value="all">All Groups</option>
          {peerGroupNames.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>

        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
          {filteredAgents.length} of {mockAgentProfiles.length} agents
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

      {/* Agent detail slide-over */}
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
  const sorted = useMemo(
    () =>
      [...mockAgentProfiles].sort(
        (a, b) => b.score.compositeWeight - a.score.compositeWeight
      ),
    []
  );

  const top = sorted.slice(0, sorted.length);
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

  return (
    <div className="space-y-6">
      {/* Full leaderboard */}
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
            {top.map((agent, i) => {
              const rank = i + 1;
              const hitRate =
                agent.profitableCount /
                Math.max(agent.profitableCount + agent.losingCount, 1);
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
                      {agent.primaryStrategy.replace("_", " ")}
                    </Badge>
                  </td>
                  <td className="px-3 py-2.5 text-right font-bold tabular-nums" style={{ color: metricColor(agent.score.riskAdjustedReturn) }}>
                    {(agent.score.riskAdjustedReturn * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-semibold tabular-nums" style={{ color: "var(--text-primary)" }}>
                    {(agent.score.compositeWeight * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5 text-right font-semibold tabular-nums" style={{ color: metricColor(hitRate) }}>
                    {(hitRate * 100).toFixed(0)}%
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(agent.score.calibrationQuality) }}>
                    {(agent.score.calibrationQuality * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(agent.score.uniqueness) }}>
                    {(agent.score.uniqueness * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(agent.score.regimeEffectiveness) }}>
                    {(agent.score.regimeEffectiveness * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(agent.score.drawdownBehavior) }}>
                    {(agent.score.drawdownBehavior * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: metricColor(agent.score.stability) }}>
                    {(agent.score.stability * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    <Badge
                      variant={statusVariant[agent.health.status] || "neutral"}
                      size="sm"
                    >
                      {agent.health.status}
                    </Badge>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Bottom Performers */}
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
                {["Agent", "Strategy", "Weight", "Score", "Status", "Recommendation"].map((h) => (
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
              {bottom.map((agent) => (
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
                      {agent.primaryStrategy.replace("_", " ")}
                    </Badge>
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: "var(--negative)" }}>
                    {(agent.score.compositeWeight * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums" style={{ color: "var(--negative)" }}>
                    {(agent.score.riskAdjustedReturn * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    <Badge
                      variant={statusVariant[agent.health.status] || "neutral"}
                      size="sm"
                    >
                      {agent.health.status}
                    </Badge>
                  </td>
                  <td className="px-3 py-2.5 max-w-[240px] truncate" style={{ color: "var(--text-secondary)" }}>
                    {agent.health.recommendation}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
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
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
          Decision Intelligence
        </h1>
        <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
          How the system discovers, evaluates, and approves trading opportunities
        </p>
      </div>

      {/* Tabs */}
      <div className="mb-6 flex items-center gap-1" style={{ borderBottom: "1px solid var(--border)" }}>
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider transition-colors"
            style={{
              color:
                activeTab === tab.key
                  ? "var(--text-primary)"
                  : "var(--text-muted)",
              borderBottom:
                activeTab === tab.key
                  ? "2px solid var(--accent)"
                  : "2px solid transparent",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "flow" && <FlowTab />}
      {activeTab === "agents" && <AgentsTab />}
      {activeTab === "leaderboard" && <LeaderboardTab />}
    </div>
  );
}

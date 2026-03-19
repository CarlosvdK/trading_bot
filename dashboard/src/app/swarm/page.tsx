"use client";

import { useState, useMemo, useCallback } from "react";
import {
  Search, X, Loader2, Filter, LayoutGrid, Network,
  Trophy, TrendingUp, TrendingDown, RefreshCw, Zap,
  BarChart3, Target, Shield,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, Legend,
} from "recharts";
import { useApi } from "@/hooks/useApi";
import { fetchAgents, fetchAgent, fetchPipelineFunnel } from "@/lib/api";
import { SwarmCanvas } from "@/components/swarm/SwarmCanvas";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface AgentSummary {
  agentId: string;
  displayName: string;
  primarySectors: string[];
  primaryStrategy: string;
  secondaryStrategy?: string | null;
  peerGroup: string;
  holdingPeriod: string;
  lookbackDays: number;
  riskAppetite: number;
  contrarianFactor: number;
  convictionStyle: number;
  regimeSensitivity: number;
  compositeWeight: number;
  status: string;
  hitRate?: number;
  nOutcomes?: number;
  riskAdjustedReturn?: number;
  calibrationQuality?: number;
  reasoningQuality?: number;
  uniqueness?: number;
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function SwarmPage() {
  const { data: agents, loading } = useApi(fetchAgents, 15000);
  const { data: funnel } = useApi(fetchPipelineFunnel, 30000);

  const [view, setView] = useState<"network" | "grid">("network");
  const [search, setSearch] = useState("");
  const [strategyFilter, setStrategyFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"weight" | "name" | "hitRate">("weight");
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  const agentList: AgentSummary[] = useMemo(() => {
    if (!agents || !Array.isArray(agents)) return [];
    return agents.map((a: any) => ({
      agentId: a.agentId || a.agent_id || "",
      displayName: a.displayName || a.display_name || a.agentId || "",
      primarySectors: a.primarySectors || a.primary_sectors || [],
      primaryStrategy: a.primaryStrategy || a.primary_strategy || "unknown",
      secondaryStrategy: a.secondaryStrategy || a.secondary_strategy || null,
      peerGroup: a.peerGroup || a.peer_group || "",
      holdingPeriod: a.holdingPeriod || a.holding_period || "",
      lookbackDays: a.lookbackDays || a.lookback_days || 0,
      riskAppetite: a.riskAppetite || a.risk_appetite || 0.5,
      contrarianFactor: a.contrarianFactor || a.contrarian_factor || 0,
      convictionStyle: a.convictionStyle || a.conviction_style || 0.5,
      regimeSensitivity: a.regimeSensitivity || a.regime_sensitivity || 0.5,
      compositeWeight: a.compositeWeight || a.composite_weight || 0.5,
      status: a.status || "healthy",
      hitRate: a.hitRate || a.hit_rate || 0,
      nOutcomes: a.nOutcomes || a.n_outcomes || 0,
      riskAdjustedReturn: a.riskAdjustedReturn || a.risk_adjusted_return || 0.5,
      calibrationQuality: a.calibrationQuality || a.calibration_quality || 0.5,
      reasoningQuality: a.reasoningQuality || a.reasoning_quality || 0.5,
      uniqueness: a.uniqueness || 0.5,
    }));
  }, [agents]);

  const strategies = useMemo(() => {
    const set = new Set(agentList.map((a) => a.primaryStrategy));
    return Array.from(set).sort();
  }, [agentList]);

  const filtered = useMemo(() => {
    let list = [...agentList];
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (a) =>
          a.displayName.toLowerCase().includes(q) ||
          a.agentId.toLowerCase().includes(q) ||
          a.primarySectors.some((s) => s.toLowerCase().includes(q)) ||
          a.primaryStrategy.toLowerCase().includes(q)
      );
    }
    if (strategyFilter !== "all") list = list.filter((a) => a.primaryStrategy === strategyFilter);
    if (statusFilter !== "all") list = list.filter((a) => a.status === statusFilter);
    list.sort((a, b) => {
      if (sortBy === "weight") return b.compositeWeight - a.compositeWeight;
      if (sortBy === "hitRate") return (b.hitRate || 0) - (a.hitRate || 0);
      return a.displayName.localeCompare(b.displayName);
    });
    return list;
  }, [agentList, search, strategyFilter, statusFilter, sortBy]);

  const statusCounts = useMemo(() => {
    const c = { healthy: 0, warning: 0, underperforming: 0, replace: 0 };
    agentList.forEach((a) => { const s = a.status as keyof typeof c; if (s in c) c[s]++; });
    return c;
  }, [agentList]);

  /* ---- Derived data for charts ---- */
  const strategyDistribution = useMemo(() => {
    const map: Record<string, number> = {};
    agentList.forEach((a) => { map[a.primaryStrategy] = (map[a.primaryStrategy] || 0) + 1; });
    return Object.entries(map)
      .map(([name, value]) => ({ name: name.replace(/_/g, " "), value }))
      .sort((a, b) => b.value - a.value);
  }, [agentList]);

  const scoreDistribution = useMemo(() => {
    const buckets = [
      { range: "0-20", min: 0, max: 0.2, count: 0 },
      { range: "20-40", min: 0.2, max: 0.4, count: 0 },
      { range: "40-60", min: 0.4, max: 0.6, count: 0 },
      { range: "60-80", min: 0.6, max: 0.8, count: 0 },
      { range: "80-100", min: 0.8, max: 1.01, count: 0 },
    ];
    agentList.forEach((a) => {
      const b = buckets.find((b) => a.compositeWeight >= b.min && a.compositeWeight < b.max);
      if (b) b.count++;
    });
    return buckets.map((b) => ({ range: b.range, agents: b.count }));
  }, [agentList]);

  const avgMetrics = useMemo(() => {
    if (!agentList.length) return [];
    const sum = (fn: (a: AgentSummary) => number) => agentList.reduce((s, a) => s + fn(a), 0) / agentList.length;
    return [
      { metric: "Risk-Adj Return", value: Math.round(sum((a) => a.riskAdjustedReturn || 0.5) * 100) },
      { metric: "Calibration", value: Math.round(sum((a) => a.calibrationQuality || 0.5) * 100) },
      { metric: "Reasoning", value: Math.round(sum((a) => a.reasoningQuality || 0.5) * 100) },
      { metric: "Uniqueness", value: Math.round(sum((a) => a.uniqueness || 0.5) * 100) },
      { metric: "Composite", value: Math.round(sum((a) => a.compositeWeight) * 100) },
    ];
  }, [agentList]);

  const leaderboard = useMemo(() => {
    return [...agentList].sort((a, b) => b.compositeWeight - a.compositeWeight);
  }, [agentList]);

  const underperformers = useMemo(() => {
    return agentList.filter(
      (a) => a.status === "underperforming" || a.status === "replace" || a.compositeWeight < 0.3
    );
  }, [agentList]);

  const [replacing, setReplacing] = useState(false);

  const handleReplace = async () => {
    setReplacing(true);
    // Simulate replacement — in production this would call POST /api/agents/replace
    await new Promise((r) => setTimeout(r, 2000));
    setReplacing(false);
    alert(`${underperformers.length} agents flagged for replacement. New agents will be generated on next pipeline cycle.`);
  };

  const PIE_COLORS = ["#3B82F6", "#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#22C55E", "#14B8A6", "#EF4444", "#64748B", "#0EA5E9"];

  if (loading && !agents) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--accent)]" />
      </div>
    );
  }

  return (
    <div className="px-6 py-8">
      {/* Header */}
      <div className="animate-fade-in flex items-end justify-between mb-8">
        <div>
          <h1 className="text-[42px] font-black tracking-tight leading-none text-[var(--text-primary)]">
            Agent Swarm
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-2">
            {agentList.length} agents monitoring the market
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Replace Underperformers */}
          {underperformers.length > 0 && (
            <button
              onClick={handleReplace}
              disabled={replacing}
              className="flex items-center gap-2 rounded-xl bg-[var(--negative)] hover:bg-red-600 text-white px-4 py-2.5 text-xs font-semibold transition-all shadow-sm disabled:opacity-60"
            >
              {replacing ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Zap size={14} />
              )}
              Replace {underperformers.length} Underperformers
            </button>
          )}

          {/* View Toggle */}
          <div className="flex items-center bg-white rounded-xl border border-[var(--border)] p-1">
            <button
              onClick={() => setView("network")}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all ${
                view === "network"
                  ? "bg-[var(--accent)] text-white shadow-sm"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              <Network size={14} />
              Network
            </button>
            <button
              onClick={() => setView("grid")}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-all ${
                view === "grid"
                  ? "bg-[var(--accent)] text-white shadow-sm"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              <LayoutGrid size={14} />
              Grid
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        {[
          { label: "Total Agents", value: agentList.length, color: "var(--accent)", icon: Shield },
          { label: "Healthy", value: statusCounts.healthy, color: "var(--positive)", icon: Target },
          { label: "Warning", value: statusCounts.warning, color: "var(--warning)", icon: TrendingDown },
          { label: "Scanned Today", value: funnel?.scanned || 0, color: "var(--text-secondary)", icon: BarChart3 },
          { label: "Approved Today", value: funnel?.approved || 0, color: "var(--positive)", icon: TrendingUp },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={stat.label} className={`animate-fade-in stagger-${i + 1} card px-5 py-4`}>
              <div className="flex items-center justify-between mb-1">
                <p className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">
                  {stat.label}
                </p>
                <Icon size={14} style={{ color: stat.color }} className="opacity-50" />
              </div>
              <p className="text-2xl font-black tabular-nums" style={{ color: stat.color }}>
                {stat.value}
              </p>
            </div>
          );
        })}
      </div>

      {/* ============ NETWORK VIEW ============ */}
      {view === "network" && (
        <div className="animate-fade-in stagger-3">
          <SwarmCanvas
            agents={agentList}
            onSelectAgent={(id) => setSelectedAgent(id)}
          />
        </div>
      )}

      {/* ============ GRID VIEW ============ */}
      {view === "grid" && (
        <>
          {/* Filters */}
          <div className="animate-fade-in stagger-3 flex items-center gap-3 mb-6">
            <div className="relative flex-1 max-w-sm">
              <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
              <input
                type="text"
                placeholder="Search agents, sectors, strategies..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full rounded-xl border border-[var(--border)] bg-white pl-10 pr-4 py-2.5 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] outline-none transition-all focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent)]/20"
              />
              {search && (
                <button onClick={() => setSearch("")} className="absolute right-3 top-1/2 -translate-y-1/2">
                  <X size={14} className="text-[var(--text-muted)]" />
                </button>
              )}
            </div>
            <select
              value={strategyFilter}
              onChange={(e) => setStrategyFilter(e.target.value)}
              className="rounded-xl border border-[var(--border)] bg-white px-4 py-2.5 text-sm text-[var(--text-secondary)] outline-none cursor-pointer"
            >
              <option value="all">All Strategies</option>
              {strategies.map((s) => (
                <option key={s} value={s}>{s.replace(/_/g, " ")}</option>
              ))}
            </select>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-xl border border-[var(--border)] bg-white px-4 py-2.5 text-sm text-[var(--text-secondary)] outline-none cursor-pointer"
            >
              <option value="all">All Status</option>
              <option value="healthy">Healthy</option>
              <option value="warning">Warning</option>
              <option value="underperforming">Underperforming</option>
            </select>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="rounded-xl border border-[var(--border)] bg-white px-4 py-2.5 text-sm text-[var(--text-secondary)] outline-none cursor-pointer"
            >
              <option value="weight">Sort: Score</option>
              <option value="hitRate">Sort: Hit Rate</option>
              <option value="name">Sort: Name</option>
            </select>
            <span className="text-xs text-[var(--text-muted)] ml-2">
              {filtered.length} of {agentList.length}
            </span>
          </div>

          {/* Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-10">
            {filtered.map((agent, i) => (
              <AgentCard
                key={agent.agentId}
                agent={agent}
                index={i}
                isSelected={selectedAgent === agent.agentId}
                onSelect={() => setSelectedAgent(selectedAgent === agent.agentId ? null : agent.agentId)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="col-span-full card py-16 text-center">
                <Filter size={24} className="mx-auto text-[var(--text-muted)] mb-2 opacity-40" />
                <p className="text-sm text-[var(--text-muted)]">No agents match your filters</p>
              </div>
            )}
          </div>
        </>
      )}

      {/* ============ ANALYTICS SECTION (always visible) ============ */}
      <div className="mt-10 space-y-8">
        {/* Charts Row */}
        <div className="grid grid-cols-3 gap-6 animate-fade-in stagger-4">
          {/* Score Distribution Histogram */}
          <div className="card p-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-4">
              Score Distribution
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={scoreDistribution} barSize={32}>
                <XAxis dataKey="range" tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#fff", border: "1px solid #DBEAFE", borderRadius: 12, fontSize: 12 }}
                  formatter={(v) => [`${v} agents`, "Count"]}
                />
                <Bar dataKey="agents" fill="#3B82F6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Strategy Breakdown Pie */}
          <div className="card p-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-4">
              Strategy Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={strategyDistribution.slice(0, 8)}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={75}
                  innerRadius={40}
                  paddingAngle={2}
                  strokeWidth={0}
                >
                  {strategyDistribution.slice(0, 8).map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: "#fff", border: "1px solid #DBEAFE", borderRadius: 12, fontSize: 12 }}
                  formatter={(v) => [`${v} agents`, "Count"]}
                />
                <Legend
                  iconSize={8}
                  wrapperStyle={{ fontSize: 10, color: "#94A3B8" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Avg Metrics Radar */}
          <div className="card p-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-4">
              Swarm Avg Metrics
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={avgMetrics} outerRadius={65}>
                <PolarGrid stroke="#DBEAFE" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "#94A3B8" }} />
                <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
                <Radar dataKey="value" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.2} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Leaderboard */}
        <div className="card p-6 animate-fade-in stagger-5">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <Trophy size={18} className="text-[var(--accent)]" />
              <h3 className="text-lg font-black text-[var(--text-primary)]">Agent Leaderboard</h3>
            </div>
            <div className="flex items-center gap-4">
              {underperformers.length > 0 && (
                <span className="text-xs font-semibold text-[var(--negative)]">
                  {underperformers.length} flagged for replacement
                </span>
              )}
              <button
                onClick={handleReplace}
                disabled={replacing || underperformers.length === 0}
                className="flex items-center gap-1.5 rounded-lg bg-[var(--accent-light)] hover:bg-blue-100 text-[var(--accent)] px-3 py-1.5 text-[11px] font-semibold transition-all disabled:opacity-40"
              >
                <RefreshCw size={12} className={replacing ? "animate-spin" : ""} />
                Auto-Replace
              </button>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-[var(--border)]">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[var(--accent-light)]">
                  <th className="text-left px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">#</th>
                  <th className="text-left px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Agent</th>
                  <th className="text-left px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Strategy</th>
                  <th className="text-left px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Peer Group</th>
                  <th className="text-center px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Score</th>
                  <th className="text-center px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Risk-Adj</th>
                  <th className="text-center px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Calibration</th>
                  <th className="text-center px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Status</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.slice(0, 20).map((agent, i) => {
                  const score = Math.round(agent.compositeWeight * 100);
                  const isBottom = agent.compositeWeight < 0.3 || agent.status === "underperforming" || agent.status === "replace";
                  const statusColor = {
                    healthy: "text-[var(--positive)] bg-[var(--positive-light)]",
                    warning: "text-[var(--warning)] bg-[var(--warning-light)]",
                    underperforming: "text-[var(--negative)] bg-[var(--negative-light)]",
                    replace: "text-[var(--negative)] bg-[var(--negative-light)]",
                  }[agent.status] || "text-[var(--text-muted)] bg-[var(--bg-hover)]";
                  return (
                    <tr
                      key={agent.agentId}
                      onClick={() => setSelectedAgent(agent.agentId)}
                      className={`border-t border-[var(--border)] cursor-pointer transition-colors hover:bg-[var(--bg-hover)] ${isBottom ? "bg-red-50/50" : ""}`}
                    >
                      <td className="px-4 py-3">
                        <span className={`text-xs font-black tabular-nums ${
                          i < 3 ? "text-[var(--accent)]" : "text-[var(--text-muted)]"
                        }`}>
                          {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-sm font-bold text-[var(--text-primary)]">{agent.displayName}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex rounded-full bg-[var(--accent-light)] px-2 py-0.5 text-[10px] font-semibold text-[var(--accent)] capitalize">
                          {agent.primaryStrategy.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-xs text-[var(--text-muted)]">{agent.peerGroup || "—"}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={`text-sm font-black tabular-nums ${
                          score >= 70 ? "text-[var(--positive)]" : score >= 40 ? "text-[var(--accent)]" : "text-[var(--negative)]"
                        }`}>{score}</span>
                      </td>
                      <td className="px-4 py-3 text-center text-xs font-semibold tabular-nums text-[var(--text-secondary)]">
                        {Math.round((agent.riskAdjustedReturn || 0.5) * 100)}
                      </td>
                      <td className="px-4 py-3 text-center text-xs font-semibold tabular-nums text-[var(--text-secondary)]">
                        {Math.round((agent.calibrationQuality || 0.5) * 100)}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold capitalize ${statusColor}`}>
                          {agent.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {leaderboard.length > 20 && (
              <div className="border-t border-[var(--border)] px-4 py-3 text-center">
                <span className="text-xs text-[var(--text-muted)]">
                  Showing top 20 of {leaderboard.length} agents
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Agent Detail Drawer */}
      {selectedAgent && (
        <AgentDetailPanel agentId={selectedAgent} onClose={() => setSelectedAgent(null)} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Agent Card                                                          */
/* ------------------------------------------------------------------ */

function AgentCard({
  agent, index, isSelected, onSelect,
}: {
  agent: AgentSummary; index: number; isSelected: boolean; onSelect: () => void;
}) {
  const statusColor = {
    healthy: "bg-[var(--positive)]",
    warning: "bg-[var(--warning)]",
    underperforming: "bg-[var(--negative)]",
    replace: "bg-[var(--negative)]",
  }[agent.status] || "bg-[var(--text-muted)]";

  const weight = Math.round(agent.compositeWeight * 100);

  return (
    <div
      onClick={onSelect}
      className={`card card-interactive p-5 cursor-pointer animate-fade-in ${
        isSelected ? "ring-2 ring-[var(--accent)]" : ""
      }`}
      style={{ animationDelay: `${Math.min(index * 0.02, 0.4)}s` }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <span className={`h-2 w-2 rounded-full ${statusColor} shrink-0`} />
          <h3 className="text-sm font-bold text-[var(--text-primary)] leading-tight truncate max-w-[160px]">
            {agent.displayName}
          </h3>
        </div>
        <span className="text-lg font-black tabular-nums text-[var(--accent)]">{weight}</span>
      </div>
      <div className="flex items-center gap-2 mb-3">
        <span className="inline-flex rounded-full bg-[var(--accent-light)] px-2.5 py-0.5 text-[10px] font-semibold text-[var(--accent)] capitalize">
          {agent.primaryStrategy.replace(/_/g, " ")}
        </span>
        {agent.secondaryStrategy && (
          <span className="inline-flex rounded-full bg-[var(--bg-hover)] px-2.5 py-0.5 text-[10px] font-medium text-[var(--text-muted)] capitalize">
            {agent.secondaryStrategy.replace(/_/g, " ")}
          </span>
        )}
      </div>
      <p className="text-[11px] text-[var(--text-muted)] mb-3 truncate">
        {agent.primarySectors.length > 0 ? agent.primarySectors.slice(0, 3).join(", ") : "All sectors"}
      </p>
      <div className="space-y-2">
        <ScoreBar label="Risk-Adj Return" value={agent.riskAdjustedReturn || 0.5} />
        <ScoreBar label="Calibration" value={agent.calibrationQuality || 0.5} />
        <ScoreBar label="Reasoning" value={agent.reasoningQuality || 0.5} />
      </div>
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-[var(--border)]">
        <span className="text-[10px] text-[var(--text-muted)]">{agent.peerGroup || "—"}</span>
        <span className="text-[10px] text-[var(--text-muted)]">{agent.holdingPeriod || "swing"} · {agent.lookbackDays}d</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Score Bar                                                           */
/* ------------------------------------------------------------------ */

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 70 ? "bg-[var(--positive)]" :
    pct >= 50 ? "bg-[var(--accent)]" :
    "bg-[var(--negative)]";

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-[var(--text-muted)] w-[90px] shrink-0 truncate">{label}</span>
      <div className="score-bar flex-1">
        <div className={`score-bar-fill ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-semibold tabular-nums text-[var(--text-secondary)] w-[28px] text-right">{pct}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Agent Detail Panel                                                  */
/* ------------------------------------------------------------------ */

function AgentDetailPanel({ agentId, onClose }: { agentId: string; onClose: () => void }) {
  const { data: detail, loading } = useApi(
    useCallback(() => fetchAgent(agentId), [agentId]),
    0
  );

  const agent = detail?.agent || detail;
  const trades = detail?.trades || [];
  const lessons = detail?.lessons || [];

  return (
    <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
      <div className="absolute inset-0 bg-blue-950/10 backdrop-blur-[2px]" />
      <div
        className="relative w-full max-w-[520px] h-full bg-white shadow-2xl overflow-y-auto animate-slide-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 z-10 bg-white border-b border-[var(--border)] px-8 py-5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-black text-[var(--text-primary)]">
              {agent?.displayName || agent?.display_name || agentId}
            </h2>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">{agent?.agentId || agentId}</p>
          </div>
          <button onClick={onClose} className="flex h-8 w-8 items-center justify-center rounded-lg hover:bg-[var(--bg-hover)] transition-colors">
            <X size={18} className="text-[var(--text-muted)]" />
          </button>
        </div>

        {loading ? (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="h-5 w-5 animate-spin text-[var(--accent)]" />
          </div>
        ) : agent ? (
          <div className="px-8 py-6 space-y-8">
            <Section title="Identity">
              <div className="grid grid-cols-2 gap-3">
                <InfoItem label="Strategy" value={agent.primaryStrategy || agent.primary_strategy || "—"} />
                <InfoItem label="Secondary" value={agent.secondaryStrategy || agent.secondary_strategy || "—"} />
                <InfoItem label="Peer Group" value={agent.peerGroup || agent.peer_group || "—"} />
                <InfoItem label="Holding" value={agent.holdingPeriod || agent.holding_period || "—"} />
                <InfoItem label="Lookback" value={`${agent.lookbackDays || agent.lookback_days || 0} days`} />
                <InfoItem label="Max Picks" value={String(agent.maxPicksPerScan || agent.max_picks_per_scan || 0)} />
              </div>
            </Section>

            <Section title="Sectors">
              <div className="flex flex-wrap gap-2">
                {(agent.primarySectors || agent.primary_sectors || []).map((s: string) => (
                  <span key={s} className="rounded-full bg-[var(--accent-light)] px-3 py-1 text-xs font-medium text-[var(--accent)]">{s}</span>
                ))}
                {(agent.primarySectors || agent.primary_sectors || []).length === 0 && (
                  <span className="text-xs text-[var(--text-muted)]">All sectors</span>
                )}
              </div>
            </Section>

            <Section title="Personality">
              <div className="space-y-3">
                <PersonalityBar label="Risk Appetite" value={agent.riskAppetite || agent.risk_appetite || 0.5} low="Conservative" high="Aggressive" />
                <PersonalityBar label="Contrarian Factor" value={agent.contrarianFactor || agent.contrarian_factor || 0} low="Consensus" high="Contrarian" />
                <PersonalityBar label="Conviction Style" value={agent.convictionStyle || agent.conviction_style || 0.5} low="Diversified" high="Concentrated" />
                <PersonalityBar label="Regime Sensitivity" value={agent.regimeSensitivity || agent.regime_sensitivity || 0.5} low="Ignores" high="Adapts" />
              </div>
            </Section>

            <Section title="Performance Scores">
              <div className="space-y-2.5">
                {[
                  { label: "Composite Weight", value: agent.compositeWeight || agent.composite_weight || 0.5 },
                  { label: "Risk-Adj Return", value: agent.riskAdjustedReturn || agent.risk_adjusted_return || 0.5 },
                  { label: "Calibration", value: agent.calibrationQuality || agent.calibration_quality || 0.5 },
                  { label: "Reasoning", value: agent.reasoningQuality || agent.reasoning_quality || 0.5 },
                  { label: "Uniqueness", value: agent.uniqueness || 0.5 },
                  { label: "Regime Effectiveness", value: agent.regimeEffectiveness || agent.regime_effectiveness || 0.5 },
                  { label: "Stability", value: agent.stability || 0.5 },
                ].map((s) => <ScoreBar key={s.label} label={s.label} value={s.value} />)}
              </div>
            </Section>

            <Section title={`Trade History (${trades.length})`}>
              {trades.length > 0 ? (
                <div className="space-y-2">
                  {trades.slice(0, 10).map((t: any, i: number) => (
                    <div key={i} className="flex items-center justify-between py-2 border-b border-[var(--border)] last:border-0">
                      <div>
                        <span className="text-sm font-bold">{t.symbol}</span>
                        <span className={`ml-2 text-[10px] font-semibold uppercase ${
                          t.direction === "long" ? "text-[var(--positive)]" : "text-[var(--negative)]"
                        }`}>{t.direction}</span>
                      </div>
                      <div className="text-right">
                        {t.pnl !== undefined && t.pnl !== null ? (
                          <span className={`text-sm font-semibold tabular-nums ${t.pnl >= 0 ? "text-[var(--positive)]" : "text-[var(--negative)]"}`}>
                            {t.pnl >= 0 ? "+" : ""}{(t.pnl * 100).toFixed(1)}%
                          </span>
                        ) : (
                          <span className="text-xs text-[var(--text-muted)]">Open</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-[var(--text-muted)]">No trade history yet</p>
              )}
            </Section>

            {lessons.length > 0 && (
              <Section title={`Lessons Learned (${lessons.length})`}>
                <div className="space-y-3">
                  {lessons.slice(0, 5).map((l: any, i: number) => (
                    <div key={i} className="rounded-xl bg-[var(--accent-light)] p-3">
                      <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                        {l.lesson || l.content || JSON.stringify(l)}
                      </p>
                    </div>
                  ))}
                </div>
              </Section>
            )}
          </div>
        ) : (
          <div className="flex h-64 items-center justify-center">
            <p className="text-sm text-[var(--text-muted)]">Agent not found</p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-3">{title}</h3>
      {children}
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] text-[var(--text-muted)] mb-0.5">{label}</p>
      <p className="text-sm font-semibold text-[var(--text-primary)] capitalize">{value.replace(/_/g, " ")}</p>
    </div>
  );
}

function PersonalityBar({ label, value, low, high }: { label: string; value: number; low: string; high: string }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-medium text-[var(--text-secondary)]">{label}</span>
        <span className="text-[11px] font-bold tabular-nums text-[var(--accent)]">{pct}%</span>
      </div>
      <div className="relative h-1.5 rounded-full bg-[#E0E7FF]">
        <div className="absolute left-0 top-0 h-full rounded-full bg-[var(--accent)] transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <div className="flex items-center justify-between mt-0.5">
        <span className="text-[9px] text-[var(--text-muted)]">{low}</span>
        <span className="text-[9px] text-[var(--text-muted)]">{high}</span>
      </div>
    </div>
  );
}

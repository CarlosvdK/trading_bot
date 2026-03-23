"use client";

import { useState, useMemo } from "react";
import {
  Activity, Brain, Newspaper, Target, Vote, Zap,
  BookOpen, RefreshCw, TrendingUp, Eye, Radio,
  Filter, Search,
} from "lucide-react";
import { useApi } from "@/hooks/useApi";
import { fetchActivity } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Activity type config                                                */
/* ------------------------------------------------------------------ */

const ACTIVITY_CONFIG: Record<string, { icon: any; color: string; bg: string; label: string }> = {
  news_scan:    { icon: Newspaper,  color: "#3B82F6", bg: "#EFF6FF",  label: "News Scan" },
  analysis:     { icon: Eye,        color: "#8B5CF6", bg: "#F5F3FF",  label: "Analysis" },
  thesis:       { icon: Brain,      color: "#EC4899", bg: "#FDF2F8",  label: "Thesis" },
  proposal:     { icon: Target,     color: "#F59E0B", bg: "#FFFBEB",  label: "Proposal" },
  vote:         { icon: Vote,       color: "#6366F1", bg: "#EEF2FF",  label: "Vote" },
  execution:    { icon: Zap,        color: "#22C55E", bg: "#F0FDF4",  label: "Execution" },
  monitoring:   { icon: Activity,   color: "#06B6D4", bg: "#ECFEFF",  label: "Monitoring" },
  learning:     { icon: BookOpen,   color: "#14B8A6", bg: "#F0FDFA",  label: "Learning" },
  retraining:   { icon: RefreshCw,  color: "#F97316", bg: "#FFF7ED",  label: "Retraining" },
  playbook:     { icon: TrendingUp, color: "#10B981", bg: "#ECFDF5",  label: "Playbook" },
  regime:       { icon: Radio,      color: "#EF4444", bg: "#FEF2F2",  label: "Regime" },
};

const MODE_LABELS: Record<string, string> = {
  weekend: "Weekend",
  premarket: "Pre-Market",
  market: "Market Hours",
  overnight: "Overnight",
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function ActivityPage() {
  const { data: events, loading } = useApi(fetchActivity, 5000);
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [search, setSearch] = useState("");

  const types = useMemo(() => {
    const s = new Set<string>();
    (events || []).forEach((e: any) => s.add(e.activity_type));
    return Array.from(s).sort();
  }, [events]);

  const filtered = useMemo(() => {
    let list = events || [];
    if (typeFilter !== "all") {
      list = list.filter((e: any) => e.activity_type === typeFilter);
    }
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (e: any) =>
          (e.summary || "").toLowerCase().includes(q) ||
          (e.agent_id || "").toLowerCase().includes(q) ||
          (e.symbol || "").toLowerCase().includes(q)
      );
    }
    return list;
  }, [events, typeFilter, search]);

  // Stats
  const stats = useMemo(() => {
    const all = events || [];
    const last5m = all.filter(
      (e: any) => Date.now() - new Date(e.created_at).getTime() < 5 * 60 * 1000
    );
    const last1h = all.filter(
      (e: any) => Date.now() - new Date(e.created_at).getTime() < 60 * 60 * 1000
    );
    const uniqueAgents = new Set(all.map((e: any) => e.agent_id)).size;
    const latestMode = all[0]?.market_mode || "unknown";
    return { last5m: last5m.length, last1h: last1h.length, uniqueAgents, latestMode };
  }, [events]);

  return (
    <div className="p-8 space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-black text-[var(--text-primary)]">Live Activity</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Real-time feed of what every agent is doing, thinking, and planning
        </p>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Last 5 min", value: stats.last5m, color: "var(--positive)" },
          { label: "Last hour", value: stats.last1h, color: "var(--accent)" },
          { label: "Active agents", value: stats.uniqueAgents, color: "#8B5CF6" },
          { label: "Mode", value: MODE_LABELS[stats.latestMode] || stats.latestMode, color: "#F59E0B" },
        ].map((s) => (
          <div key={s.label} className="card px-5 py-4">
            <p className="text-[11px] text-[var(--text-muted)] mb-1">{s.label}</p>
            <p className="text-xl font-black" style={{ color: s.color }}>
              {s.value}
            </p>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
          <input
            type="text"
            placeholder="Search agents, symbols..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-[13px] rounded-xl border border-[var(--border)] bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <Filter size={14} className="text-[var(--text-muted)]" />
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="text-[13px] px-3 py-2 rounded-xl border border-[var(--border)] bg-white focus:outline-none"
          >
            <option value="all">All Types</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {ACTIVITY_CONFIG[t]?.label || t}
              </option>
            ))}
          </select>
        </div>
        {loading && (
          <div className="flex items-center gap-1.5 text-[var(--text-muted)]">
            <RefreshCw size={12} className="animate-spin" />
            <span className="text-[11px]">Updating...</span>
          </div>
        )}
      </div>

      {/* Activity Feed */}
      <div className="space-y-2">
        {filtered.length === 0 && !loading && (
          <div className="card px-8 py-16 text-center">
            <Activity size={32} className="mx-auto mb-3 text-[var(--text-muted)]" />
            <p className="text-sm text-[var(--text-muted)]">
              No activity yet. Agents will start logging events as they scan, analyze, and trade.
            </p>
          </div>
        )}

        {filtered.map((event: any, i: number) => {
          const cfg = ACTIVITY_CONFIG[event.activity_type] || ACTIVITY_CONFIG.monitoring;
          const Icon = cfg.icon;
          return (
            <div
              key={event.id || i}
              className="card px-5 py-3.5 flex items-start gap-4 animate-fade-in"
              style={{ animationDelay: `${Math.min(i * 20, 300)}ms` }}
            >
              {/* Icon */}
              <div
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl"
                style={{ backgroundColor: cfg.bg }}
              >
                <Icon size={16} style={{ color: cfg.color }} />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span
                    className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
                    style={{ color: cfg.color, backgroundColor: cfg.bg }}
                  >
                    {cfg.label}
                  </span>
                  {event.symbol && (
                    <span className="text-[11px] font-bold text-[var(--text-primary)] bg-[var(--bg-hover)] px-2 py-0.5 rounded-md">
                      {event.symbol}
                    </span>
                  )}
                  {event.market_mode && (
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {MODE_LABELS[event.market_mode] || event.market_mode}
                    </span>
                  )}
                </div>
                <p className="text-[13px] text-[var(--text-primary)] leading-relaxed">
                  {event.summary}
                </p>
                <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                  {event.agent_id?.replace(/_/g, " ")}
                </p>
              </div>

              {/* Time */}
              <span className="text-[11px] text-[var(--text-muted)] shrink-0 tabular-nums">
                {event.created_at ? timeAgo(event.created_at) : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

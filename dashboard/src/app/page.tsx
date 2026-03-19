"use client";

import { useMemo, useState } from "react";
import {
  ArrowUpRight, ArrowDownRight, Search,
  TrendingUp, TrendingDown, Loader2, Play,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { useApi } from "@/hooks/useApi";
import {
  fetchPortfolioSummary, fetchPositions, fetchPortfolioHistory,
  fetchRiskSummary, fetchControls, toggleKillSwitch,
  fetchPipelineFunnel, fetchPipelineRecent, triggerScan,
} from "@/lib/api";
import { formatCurrency, formatPct, getPnlColor } from "@/lib/utils";
import { KillSwitch } from "@/components/risk/KillSwitch";

export default function PortfolioPage() {
  const { data: summary, loading: loadingSummary } = useApi(fetchPortfolioSummary, 10000);
  const { data: positions } = useApi(fetchPositions, 10000);
  const { data: history } = useApi(fetchPortfolioHistory, 60000);
  const { data: risk } = useApi(fetchRiskSummary, 15000);
  const { data: controls, refresh: refreshControls } = useApi(fetchControls, 10000);
  const { data: funnel } = useApi(fetchPipelineFunnel, 30000);
  const { data: pipelineRecent } = useApi(fetchPipelineRecent, 30000);
  const [scanning, setScanning] = useState(false);

  const s = summary || {
    totalValue: 0, cashBalance: 0, dailyPnl: 0, unrealizedPnl: 0,
    grossExposure: 0, realizedPnl: 0, openPositions: 0,
  };

  const totalValue = s.totalValue || 0;
  const dailyPnl = s.dailyPnl || 0;
  const dailyPnlPct = totalValue > 0 ? dailyPnl / totalValue : 0;
  const positionList: any[] = positions || [];
  const killActive = risk?.killSwitchActive || controls?.killSwitchActive || false;

  const chartData = useMemo(() => {
    if (!history || !Array.isArray(history) || history.length === 0) return [];
    return history.map((h: any) => ({
      date: h.timestamp || h.date || "",
      value: h.totalValue || h.total_value || 0,
    }));
  }, [history]);

  const sortedByPnl = useMemo(
    () => [...positionList].sort((a, b) => (b.unrealizedPnl || 0) - (a.unrealizedPnl || 0)),
    [positionList]
  );
  const topMovers = sortedByPnl.slice(0, 3);
  const bottomMovers = sortedByPnl.slice(-3).reverse();

  const handleKillSwitch = async (activate: boolean) => {
    await toggleKillSwitch(activate);
    refreshControls();
  };

  const handleScan = async () => {
    setScanning(true);
    await triggerScan();
    setTimeout(() => setScanning(false), 3000);
  };

  if (loadingSummary && !summary) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--accent)]" />
      </div>
    );
  }

  return (
    <div className="px-10 py-8 max-w-[1400px] mx-auto">
      {/* Page Header */}
      <div className="animate-fade-in flex items-end justify-between mb-10">
        <div>
          <h1 className="text-[42px] font-black tracking-tight leading-none text-[var(--text-primary)]">
            Portfolio
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-2">
            {s.source === "ibkr_live" ? "Live from Interactive Brokers" : "Real-time overview"}
          </p>
        </div>
        <button
          onClick={handleScan}
          disabled={scanning}
          className="flex items-center gap-2 rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-blue-200 transition-all hover:bg-[var(--accent-dark)] disabled:opacity-50"
        >
          {scanning ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
          {scanning ? "Scanning..." : "Run Scan"}
        </button>
      </div>

      {/* Hero Card — Value + Chart */}
      <div className="animate-fade-in stagger-1 card p-8 mb-6">
        <div className="flex items-start justify-between mb-8">
          {/* Left: Value */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)] mb-2">
              Total Value
            </p>
            <h2 className="text-[52px] font-black tracking-tight leading-none tabular-nums text-[var(--text-primary)]">
              {formatCurrency(totalValue)}
            </h2>
            <div className={`flex items-center gap-2 mt-3 ${dailyPnl >= 0 ? "text-[var(--positive)]" : "text-[var(--negative)]"}`}>
              {dailyPnl >= 0 ? <ArrowUpRight size={20} strokeWidth={2.5} /> : <ArrowDownRight size={20} strokeWidth={2.5} />}
              <span className="text-xl font-bold tabular-nums">
                {formatCurrency(Math.abs(dailyPnl))}
              </span>
              <span className="text-sm font-semibold tabular-nums opacity-70">
                {formatPct(dailyPnlPct)}
              </span>
              <span className="text-xs text-[var(--text-muted)] ml-1">today</span>
            </div>
          </div>

          {/* Right: Key Stats */}
          <div className="text-right space-y-2">
            <StatPill label="Cash" value={formatCurrency(s.cashBalance || 0)} />
            <StatPill
              label="Unrealized"
              value={formatCurrency(s.unrealizedPnl || 0)}
              color={getPnlColor(s.unrealizedPnl || 0)}
            />
            <StatPill
              label="Exposure"
              value={formatPct(
                typeof s.grossExposure === "number" && s.grossExposure <= 1
                  ? s.grossExposure
                  : totalValue > 0 ? (s.grossExposure || 0) / totalValue : 0
              )}
            />
            <StatPill label="Positions" value={String(positionList.length)} />
          </div>
        </div>

        {/* Chart */}
        {chartData.length > 1 ? (
          <div className="h-[220px] -mx-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="valueGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={dailyPnl >= 0 ? "#22C55E" : "#EF4444"} stopOpacity={0.1} />
                    <stop offset="100%" stopColor={dailyPnl >= 0 ? "#22C55E" : "#EF4444"} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" hide />
                <YAxis hide domain={["dataMin", "dataMax"]} />
                <Tooltip
                  contentStyle={{
                    background: "#fff",
                    border: "none",
                    borderRadius: 12,
                    boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                    fontSize: 13,
                    padding: "8px 14px",
                  }}
                  formatter={(v) => [formatCurrency(Number(v)), "Value"]}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={dailyPnl >= 0 ? "#22C55E" : "#EF4444"}
                  strokeWidth={2.5}
                  fill="url(#valueGrad)"
                  dot={false}
                  animationDuration={1200}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-[220px] flex items-center justify-center">
            <p className="text-sm text-[var(--text-muted)]">
              Chart will appear after your first trading day
            </p>
          </div>
        )}
      </div>

      {/* Quick Stats Row */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { label: "Open Positions", value: positionList.length, delay: "stagger-2" },
          { label: "Stocks Scanned", value: funnel?.scanned || 0, delay: "stagger-3" },
          { label: "Approved Today", value: funnel?.approved || 0, delay: "stagger-4" },
          { label: "Pipeline Runs", value: pipelineRecent?.length || 0, delay: "stagger-5" },
        ].map((stat) => (
          <div key={stat.label} className={`animate-fade-in ${stat.delay} card px-6 py-5`}>
            <p className="text-[11px] font-semibold uppercase tracking-widest text-[var(--text-muted)] mb-1">
              {stat.label}
            </p>
            <p className="text-3xl font-black tabular-nums text-[var(--text-primary)]">
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      {/* Open Positions */}
      <div className="animate-fade-in stagger-3 mb-8">
        <h2 className="text-2xl font-black text-[var(--text-primary)] mb-5">
          Open Positions
        </h2>
        {positionList.length > 0 ? (
          <div className="card overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  {["Symbol", "Side", "Qty", "Entry", "Current", "Value", "P&L", "P&L %", "Strategy"].map((h) => (
                    <th
                      key={h}
                      className="px-5 py-4 text-left text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positionList.map((p: any, i: number) => (
                  <tr
                    key={p.tradeId || p.symbol || i}
                    className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--bg-hover)] transition-colors duration-150 cursor-default"
                  >
                    <td className="px-5 py-4">
                      <span className="text-sm font-bold text-[var(--text-primary)]">{p.symbol}</span>
                    </td>
                    <td className="px-5 py-4">
                      <span
                        className={`inline-flex rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase ${
                          p.direction === "long"
                            ? "bg-[var(--positive-light)] text-[var(--positive)]"
                            : "bg-[var(--negative-light)] text-[var(--negative)]"
                        }`}
                      >
                        {(p.direction || "long").toUpperCase()}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-sm tabular-nums text-[var(--text-secondary)]">{p.quantity}</td>
                    <td className="px-5 py-4 text-sm tabular-nums text-[var(--text-muted)]">${(p.avgEntryPrice || 0).toFixed(2)}</td>
                    <td className="px-5 py-4 text-sm tabular-nums text-[var(--text-secondary)]">${(p.currentPrice || 0).toFixed(2)}</td>
                    <td className="px-5 py-4 text-sm tabular-nums text-[var(--text-secondary)]">{formatCurrency(p.marketValue || 0)}</td>
                    <td className={`px-5 py-4 text-sm font-bold tabular-nums ${getPnlColor(p.unrealizedPnl || 0)}`}>
                      {formatCurrency(p.unrealizedPnl || 0)}
                    </td>
                    <td className={`px-5 py-4 text-sm tabular-nums ${getPnlColor(p.unrealizedPnlPct || 0)}`}>
                      {formatPct(p.unrealizedPnlPct || 0)}
                    </td>
                    <td className="px-5 py-4 text-sm text-[var(--text-muted)] capitalize">
                      {(p.strategy || "—").replace(/_/g, " ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="card py-20 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[var(--bg-hover)]">
              <TrendingUp size={20} className="text-[var(--accent)]" />
            </div>
            <p className="text-sm font-medium text-[var(--text-secondary)]">No open positions</p>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Run a pipeline scan or connect IBKR to see positions
            </p>
          </div>
        )}
      </div>

      {/* Top Movers — only show if positions exist */}
      {positionList.length > 0 && (
        <div className="grid grid-cols-2 gap-4 mb-8 animate-fade-in stagger-4">
          <div className="card p-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-4 flex items-center gap-2">
              <TrendingUp size={14} className="text-[var(--positive)]" />
              Top Winners
            </h3>
            <div className="space-y-3">
              {topMovers.filter((p: any) => (p.unrealizedPnl || 0) > 0).map((p: any) => (
                <div key={p.symbol} className="flex items-center justify-between">
                  <span className="text-sm font-bold">{p.symbol}</span>
                  <div className="text-right">
                    <span className="text-sm font-bold text-[var(--positive)] tabular-nums">
                      {formatCurrency(p.unrealizedPnl || 0)}
                    </span>
                    <span className="ml-2 text-xs text-[var(--positive)] tabular-nums opacity-60">
                      {formatPct(p.unrealizedPnlPct || 0)}
                    </span>
                  </div>
                </div>
              ))}
              {topMovers.filter((p: any) => (p.unrealizedPnl || 0) > 0).length === 0 && (
                <p className="text-xs text-[var(--text-muted)]">No winners yet</p>
              )}
            </div>
          </div>
          <div className="card p-6">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-4 flex items-center gap-2">
              <TrendingDown size={14} className="text-[var(--negative)]" />
              Top Losers
            </h3>
            <div className="space-y-3">
              {bottomMovers.filter((p: any) => (p.unrealizedPnl || 0) < 0).map((p: any) => (
                <div key={p.symbol} className="flex items-center justify-between">
                  <span className="text-sm font-bold">{p.symbol}</span>
                  <div className="text-right">
                    <span className="text-sm font-bold text-[var(--negative)] tabular-nums">
                      {formatCurrency(p.unrealizedPnl || 0)}
                    </span>
                    <span className="ml-2 text-xs text-[var(--negative)] tabular-nums opacity-60">
                      {formatPct(p.unrealizedPnlPct || 0)}
                    </span>
                  </div>
                </div>
              ))}
              {bottomMovers.filter((p: any) => (p.unrealizedPnl || 0) < 0).length === 0 && (
                <p className="text-xs text-[var(--text-muted)]">No losers yet</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Agent Plans */}
      <div className="animate-fade-in stagger-4 mb-8">
        <h2 className="text-2xl font-black text-[var(--text-primary)] mb-5">
          Agent Plans
        </h2>
        <div className="card p-6">
          {pipelineRecent && pipelineRecent.length > 0 ? (
            <div className="space-y-0">
              {pipelineRecent.slice(0, 6).map((run: any, i: number) => (
                <div
                  key={run.id || i}
                  className="flex items-center gap-4 py-4 border-b border-[var(--border)] last:border-0"
                >
                  <div
                    className={`h-2.5 w-2.5 rounded-full shrink-0 ${
                      run.status === "completed"
                        ? "bg-[var(--positive)]"
                        : run.status === "running"
                        ? "bg-[var(--warning)] animate-pulse-dot"
                        : "bg-[var(--text-muted)]"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-[var(--text-primary)]">
                      Pipeline Run
                      {run.approved_count > 0 && (
                        <span className="ml-2 text-xs font-bold text-[var(--positive)]">
                          {run.approved_count} approved
                        </span>
                      )}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                      {run.scanned_count || run.scanned || 0} scanned
                      {" · "}
                      {run.status || "unknown"}
                    </p>
                  </div>
                  <span className="text-[11px] tabular-nums text-[var(--text-muted)] shrink-0">
                    {run.started_at || run.timestamp
                      ? new Date(run.started_at || run.timestamp).toLocaleString("en-US", {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : ""}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-12 text-center">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[var(--bg-hover)]">
                <Search size={20} className="text-[var(--accent)]" />
              </div>
              <p className="text-sm font-medium text-[var(--text-secondary)]">No recent scans</p>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                Click "Run Scan" to have agents analyze the market
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Emergency Controls */}
      <div className="animate-fade-in stagger-5 mb-8">
        <h2 className="text-2xl font-black text-[var(--text-primary)] mb-5">
          Emergency Controls
        </h2>
        <KillSwitch
          active={killActive}
          onActivate={() => handleKillSwitch(true)}
          onDeactivate={() => handleKillSwitch(false)}
        />
      </div>
    </div>
  );
}

/* Small helper component */
function StatPill({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-end gap-3">
      <span className="text-[11px] text-[var(--text-muted)]">{label}</span>
      <span className={`text-sm font-bold tabular-nums ${color || "text-[var(--text-secondary)]"}`}>
        {value}
      </span>
    </div>
  );
}

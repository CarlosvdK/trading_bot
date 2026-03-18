"use client";

import { useState, useCallback } from "react";
import {
  ShieldAlert,
  Activity,
  TrendingDown,
  DollarSign,
  BarChart3,
  GitBranch,
  Layers,
  Wifi,
  WifiOff,
  AlertTriangle,
  XOctagon,
  CheckCircle2,
  Ban,
} from "lucide-react";
import { MetricCard } from "@/components/shared/MetricCard";
import { Badge } from "@/components/shared/Badge";
import { ProgressBar } from "@/components/shared/ProgressBar";
import { KillSwitch } from "@/components/risk/KillSwitch";
import { AlertPanel } from "@/components/risk/AlertPanel";
import { ControlPanel } from "@/components/risk/ControlPanel";
import {
  mockRiskSummary,
  mockAlerts,
  mockControlState,
  mockRegimeInfo,
  mockAgentHealth,
  mockPositionRisks,
  mockExecutionHealth,
} from "@/data/mock/risk";
import {
  formatCurrency,
  formatPct,
  formatDateTime,
  getPnlColor,
} from "@/lib/utils";
import type { ControlState, AgentStatus } from "@/types";

// --- Agent status config ---
const agentStatusConfig: Record<AgentStatus, { color: string; badgeVariant: "success" | "warning" | "danger" | "neutral" }> = {
  healthy: { color: "text-[var(--positive)]", badgeVariant: "success" },
  warning: { color: "text-[var(--warning)]", badgeVariant: "warning" },
  underperforming: { color: "text-[var(--negative)]", badgeVariant: "danger" },
  replace: { color: "text-[var(--critical)]", badgeVariant: "danger" },
};

// --- Risk score color ---
function getRiskScoreColor(score: number): string {
  if (score >= 7) return "text-[#DC2626]";
  if (score >= 5) return "text-[#F59E0B]";
  return "text-[#10B981]";
}

function getRiskScoreBg(score: number): string {
  if (score >= 7) return "bg-[#DC2626]/15";
  if (score >= 5) return "bg-[#F59E0B]/15";
  return "bg-[#10B981]/15";
}

export default function RiskPage() {
  const [killSwitchActive, setKillSwitchActive] = useState(mockRiskSummary.killSwitchActive);
  const [controlState, setControlState] = useState<ControlState>(mockControlState);
  const [disabledAgents, setDisabledAgents] = useState<Set<string>>(new Set());

  const handleControlChange = useCallback((key: string, value: boolean | number) => {
    setControlState((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleKillActivate = useCallback(() => {
    setKillSwitchActive(true);
    setControlState((prev) => ({ ...prev, tradingPaused: true }));
  }, []);

  const handleKillDeactivate = useCallback(() => {
    setKillSwitchActive(false);
  }, []);

  const handleDisableAgent = useCallback((agentId: string) => {
    setDisabledAgents((prev) => {
      const next = new Set(prev);
      if (next.has(agentId)) {
        next.delete(agentId);
      } else {
        next.add(agentId);
      }
      return next;
    });
  }, []);

  // Compute alert counts
  const criticalCount = mockAlerts.filter((a) => a.severity === "critical" && !a.acknowledged).length;
  const totalUnack = mockAlerts.filter((a) => !a.acknowledged).length;

  // Agent health stats
  const healthyAgents = mockAgentHealth.filter((a) => a.status === "healthy").length;
  const warningAgents = mockAgentHealth.filter((a) => a.status === "warning").length;
  const underperformingAgents = mockAgentHealth.filter((a) => a.status === "underperforming").length;
  const replaceAgents = mockAgentHealth.filter((a) => a.status === "replace").length;
  const flaggedAgents = mockAgentHealth.filter((a) => a.status !== "healthy");

  return (
    <div className="space-y-6">
      {/* ================================================================ */}
      {/* 1. PAGE HEADER WITH STATUS STRIP */}
      {/* ================================================================ */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <ShieldAlert size={22} className="text-[var(--text-primary)]" />
            <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">
              Risk & Control Center
            </h1>
          </div>

          {/* Kill switch status */}
          <div className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
            killSwitchActive
              ? "bg-red-50 text-red-600 border border-red-200"
              : "bg-emerald-50 text-emerald-600 border border-emerald-200"
          }`}>
            <span className={`h-2 w-2 rounded-full ${
              killSwitchActive ? "bg-red-500 animate-pulse" : "bg-emerald-500"
            }`} />
            {killSwitchActive ? "KILLED" : "ACTIVE"}
          </div>

          {/* Regime badge */}
          <Badge variant="info" size="md">
            {mockRegimeInfo.current}
          </Badge>
        </div>

        {/* Alert count */}
        <div className="flex items-center gap-2">
          {criticalCount > 0 && (
            <div className="flex items-center gap-1.5 rounded-full bg-red-50 border border-red-200 px-3 py-1 text-xs font-semibold text-red-600">
              <AlertTriangle size={12} />
              {criticalCount} Critical
            </div>
          )}
          {totalUnack > 0 && (
            <div className="flex items-center gap-1.5 rounded-md bg-slate-100 px-3 py-1 text-xs text-[var(--text-muted)]">
              {totalUnack} unacknowledged
            </div>
          )}
        </div>
      </div>

      {/* ================================================================ */}
      {/* 2. RISK SUMMARY CARDS */}
      {/* ================================================================ */}
      <div className="grid grid-cols-6 gap-3">
        {/* Current Drawdown */}
        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Current Drawdown
          </div>
          <div className={`mt-1 text-xl font-bold ${
            mockRiskSummary.currentDrawdown > 0.05 ? "text-[#DC2626]" :
            mockRiskSummary.currentDrawdown > 0.03 ? "text-[#F59E0B]" :
            "text-[var(--text-primary)]"
          }`}>
            {(mockRiskSummary.currentDrawdown * 100).toFixed(1)}%
          </div>
          <ProgressBar
            value={mockRiskSummary.currentDrawdown * 100}
            max={mockRiskSummary.maxDrawdownLimit * 100}
            warningThreshold={50}
            dangerThreshold={80}
            className="mt-2"
          />
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            Limit: {(mockRiskSummary.maxDrawdownLimit * 100).toFixed(0)}%
          </div>
        </div>

        {/* Daily PnL */}
        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Daily PnL
          </div>
          <div className={`mt-1 text-xl font-bold ${getPnlColor(mockRiskSummary.dailyLoss)}`}>
            {formatCurrency(mockRiskSummary.dailyLoss)}
          </div>
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            Limit: {formatCurrency(-mockRiskSummary.dailyLossLimit)}
          </div>
        </div>

        {/* Gross Exposure */}
        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Gross Exposure
          </div>
          <div className="mt-1 text-xl font-bold text-[var(--text-primary)]">
            {(mockRiskSummary.grossExposure * 100).toFixed(0)}%
          </div>
          <ProgressBar
            value={mockRiskSummary.grossExposure * 100}
            max={mockRiskSummary.grossExposureLimit * 100}
            warningThreshold={60}
            dangerThreshold={85}
            className="mt-2"
          />
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            Limit: {(mockRiskSummary.grossExposureLimit * 100).toFixed(0)}%
          </div>
        </div>

        {/* Net Exposure */}
        <MetricCard
          label="Net Exposure"
          value={`${(mockRiskSummary.netExposure * 100).toFixed(0)}%`}
          subValue={`Leverage: ${mockRiskSummary.leverage.toFixed(2)}x`}
          trend="neutral"
        />

        {/* Concentration */}
        <MetricCard
          label="Concentration"
          value={`${(mockRiskSummary.concentrationRisk * 100).toFixed(0)}%`}
          subValue="Top position %"
          trend={mockRiskSummary.concentrationRisk > 0.15 ? "down" : "neutral"}
        />

        {/* Correlation Risk */}
        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
            Correlation Risk
          </div>
          <div className={`mt-1 text-xl font-bold ${
            mockRiskSummary.correlationRisk > 0.7 ? "text-[#DC2626]" :
            mockRiskSummary.correlationRisk > 0.5 ? "text-[#F59E0B]" :
            "text-[var(--positive)]"
          }`}>
            {mockRiskSummary.correlationRisk.toFixed(2)}
          </div>
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            {mockRiskSummary.correlationRisk > 0.7 ? "Severely correlated" :
             mockRiskSummary.correlationRisk > 0.5 ? "Moderately correlated" :
             "Well diversified"}
          </div>
        </div>
      </div>

      {/* ================================================================ */}
      {/* 3 & 4. ALERTS + CONTROL PANEL */}
      {/* ================================================================ */}
      <div className="grid grid-cols-3 gap-4" style={{ minHeight: 420 }}>
        <div className="col-span-2">
          <AlertPanel alerts={mockAlerts} />
        </div>
        <div className="col-span-1">
          <ControlPanel state={controlState} onChange={handleControlChange} />
        </div>
      </div>

      {/* ================================================================ */}
      {/* 5. KILL SWITCH */}
      {/* ================================================================ */}
      <KillSwitch
        active={killSwitchActive}
        onActivate={handleKillActivate}
        onDeactivate={handleKillDeactivate}
      />

      {/* ================================================================ */}
      {/* 6. POSITION RISK TABLE */}
      {/* ================================================================ */}
      <div className="rounded-xl border border-[var(--border)] bg-white shadow-sm">
        <div className="border-b border-[var(--border)] px-4 py-3">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">Position Risk</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border)] bg-slate-50">
                <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Symbol</th>
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Direction</th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Size</th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">PnL%</th>
                <th className="px-3 py-2.5 text-right text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Dist to Stop</th>
                <th className="px-3 py-2.5 text-center text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Risk Score</th>
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Sector</th>
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Flags</th>
              </tr>
            </thead>
            <tbody>
              {[...mockPositionRisks]
                .sort((a, b) => b.riskScore - a.riskScore)
                .map((pos) => (
                  <tr
                    key={pos.symbol}
                    className="border-b border-slate-100 transition-colors hover:bg-slate-50"
                  >
                    <td className="px-4 py-2.5">
                      <span className="font-semibold text-[var(--text-primary)]">{pos.symbol}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <Badge
                        variant={pos.direction === "long" ? "success" : "danger"}
                        size="sm"
                      >
                        {pos.direction.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-[var(--text-secondary)]">
                      {formatPct(pos.sizePct)}
                    </td>
                    <td className={`px-3 py-2.5 text-right tabular-nums font-medium ${getPnlColor(pos.pnlPct)}`}>
                      {formatPct(pos.pnlPct)}
                    </td>
                    <td className={`px-3 py-2.5 text-right tabular-nums font-medium ${
                      pos.distanceToStop < 0.02 ? "text-[#DC2626]" :
                      pos.distanceToStop < 0.04 ? "text-[#F59E0B]" :
                      "text-[var(--text-secondary)]"
                    }`}>
                      {formatPct(pos.distanceToStop)}
                    </td>
                    <td className="px-3 py-2.5 text-center">
                      <span className={`inline-flex items-center justify-center rounded px-2 py-0.5 font-mono font-bold ${getRiskScoreColor(pos.riskScore)} ${getRiskScoreBg(pos.riskScore)}`}>
                        {pos.riskScore.toFixed(1)}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-[var(--text-muted)]">{pos.sector}</td>
                    <td className="px-3 py-2.5">
                      <div className="flex flex-wrap gap-1">
                        {pos.flags.map((flag) => (
                          <Badge
                            key={flag}
                            variant={
                              flag.includes("Stop") || flag.includes("Short") ? "danger" :
                              flag.includes("Mismatch") || flag.includes("Low") ? "warning" :
                              "neutral"
                            }
                            size="sm"
                          >
                            {flag}
                          </Badge>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ================================================================ */}
      {/* 7 & 8. AGENT HEALTH + EXECUTION HEALTH */}
      {/* ================================================================ */}
      <div className="grid grid-cols-2 gap-4">
        {/* Agent Health */}
        <div className="rounded-xl border border-[var(--border)] bg-white shadow-sm">
          <div className="border-b border-[var(--border)] px-4 py-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">Agent Health</h3>
              <div className="flex items-center gap-3 text-[10px]">
                <span className="flex items-center gap-1 text-[var(--positive)]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--positive)]" />
                  {healthyAgents} Healthy
                </span>
                <span className="flex items-center gap-1 text-[var(--warning)]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--warning)]" />
                  {warningAgents} Warning
                </span>
                <span className="flex items-center gap-1 text-[var(--negative)]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--negative)]" />
                  {underperformingAgents} Under
                </span>
                <span className="flex items-center gap-1 text-[var(--critical)]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--critical)]" />
                  {replaceAgents} Replace
                </span>
              </div>
            </div>
          </div>

          <div className="divide-y divide-slate-100">
            {flaggedAgents.map((agent) => {
              const config = agentStatusConfig[agent.status];
              const isDisabled = disabledAgents.has(agent.agentId);

              return (
                <div
                  key={agent.agentId}
                  className={`flex items-center justify-between px-4 py-3 ${isDisabled ? "opacity-40" : ""}`}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-medium text-[var(--text-primary)]">
                        {agent.agentId}
                      </span>
                      <Badge variant={config.badgeVariant} size="sm">
                        {agent.status}
                      </Badge>
                      {agent.regimeMismatch && (
                        <Badge variant="warning" size="sm">Regime Mismatch</Badge>
                      )}
                      {agent.isRedundant && (
                        <Badge variant="neutral" size="sm">Redundant</Badge>
                      )}
                    </div>
                    <p className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                      {agent.reason}
                    </p>
                    <p className="text-[10px] text-[var(--text-muted)]">
                      Weight: {(agent.compositeWeight * 100).toFixed(0)}%
                      {agent.daysSinceLastCorrect != null && (
                        <> &middot; {agent.daysSinceLastCorrect}d since last correct</>
                      )}
                    </p>
                  </div>
                  <button
                    onClick={() => handleDisableAgent(agent.agentId)}
                    className={`shrink-0 rounded-md border px-3 py-1 text-[10px] font-semibold transition-colors ${
                      isDisabled
                        ? "border-[var(--positive)]/30 text-[var(--positive)] hover:bg-[var(--positive)]/10"
                        : "border-[var(--negative)]/30 text-[var(--negative)] hover:bg-[var(--negative)]/10"
                    }`}
                  >
                    {isDisabled ? "Enable" : "Disable"}
                  </button>
                </div>
              );
            })}
            {flaggedAgents.length === 0 && (
              <div className="flex items-center justify-center py-8 text-sm text-[var(--text-muted)]">
                All agents healthy
              </div>
            )}
          </div>
        </div>

        {/* Execution Health */}
        <div className="rounded-xl border border-[var(--border)] bg-white shadow-sm">
          <div className="border-b border-[var(--border)] px-4 py-3">
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">Execution Health</h3>
          </div>

          <div className="p-4 space-y-4">
            {/* Broker Connection */}
            <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="flex items-center gap-3">
                {mockExecutionHealth.brokerConnected ? (
                  <Wifi size={18} className="text-[var(--positive)]" />
                ) : (
                  <WifiOff size={18} className="text-[var(--negative)]" />
                )}
                <div>
                  <p className="text-xs font-medium text-[var(--text-primary)]">
                    {mockExecutionHealth.brokerName}
                  </p>
                  <p className="text-[10px] text-[var(--text-muted)]">
                    Last heartbeat: {formatDateTime(mockExecutionHealth.lastHeartbeat)}
                  </p>
                </div>
              </div>
              <Badge
                variant={mockExecutionHealth.brokerConnected ? "success" : "danger"}
                size="md"
              >
                {mockExecutionHealth.brokerConnected ? "Connected" : "Disconnected"}
              </Badge>
            </div>

            {/* Metrics grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Fill Rate
                </div>
                <div className={`mt-1 text-lg font-bold ${
                  mockExecutionHealth.orderFillRate >= 0.9 ? "text-[var(--positive)]" :
                  mockExecutionHealth.orderFillRate >= 0.8 ? "text-[var(--warning)]" :
                  "text-[var(--negative)]"
                }`}>
                  {(mockExecutionHealth.orderFillRate * 100).toFixed(0)}%
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  {mockExecutionHealth.filledToday}/{mockExecutionHealth.totalOrdersToday} today
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Avg Slippage
                </div>
                <div className={`mt-1 text-lg font-bold ${
                  mockExecutionHealth.avgSlippageBps <= 3 ? "text-[var(--positive)]" :
                  mockExecutionHealth.avgSlippageBps <= 5 ? "text-[var(--warning)]" :
                  "text-[var(--negative)]"
                }`}>
                  {mockExecutionHealth.avgSlippageBps.toFixed(1)} bps
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  Target: &lt;3.0 bps
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Failed Orders
                </div>
                <div className={`mt-1 text-lg font-bold ${
                  mockExecutionHealth.failedOrders === 0 ? "text-[var(--positive)]" :
                  mockExecutionHealth.failedOrders <= 2 ? "text-[var(--warning)]" :
                  "text-[var(--negative)]"
                }`}>
                  {mockExecutionHealth.failedOrders}
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  Today
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                  Stale Orders
                </div>
                <div className={`mt-1 text-lg font-bold ${
                  mockExecutionHealth.staleOrders === 0 ? "text-[var(--positive)]" :
                  "text-[var(--warning)]"
                }`}>
                  {mockExecutionHealth.staleOrders}
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  Pending: {mockExecutionHealth.pendingToday}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

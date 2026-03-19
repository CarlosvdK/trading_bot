"use client";

import { useState, useCallback, useMemo } from "react";
import {
  ShieldAlert,
  Wifi,
  WifiOff,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { MetricCard } from "@/components/shared/MetricCard";
import { Badge } from "@/components/shared/Badge";
import { ProgressBar } from "@/components/shared/ProgressBar";
import { KillSwitch } from "@/components/risk/KillSwitch";
import { AlertPanel } from "@/components/risk/AlertPanel";
import { ControlPanel } from "@/components/risk/ControlPanel";
import {
  fetchRiskSummary,
  fetchAlerts,
  fetchControls,
  fetchRegime,
  fetchAgents,
  fetchHealth,
  toggleKillSwitch,
} from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import {
  formatCurrency,
  formatPct,
  formatDateTime,
  getPnlColor,
} from "@/lib/utils";
import type { ControlState, AgentStatus } from "@/types";

const agentStatusConfig: Record<AgentStatus, { color: string; badgeVariant: "success" | "warning" | "danger" | "neutral" }> = {
  healthy: { color: "text-[var(--positive)]", badgeVariant: "success" },
  warning: { color: "text-[var(--warning)]", badgeVariant: "warning" },
  underperforming: { color: "text-[var(--negative)]", badgeVariant: "danger" },
  replace: { color: "text-[var(--critical)]", badgeVariant: "danger" },
};

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
  const { data: riskSummary, loading: loadingRisk, refresh: refreshRisk } = useApi(fetchRiskSummary, 10000);
  const { data: alerts, refresh: refreshAlerts } = useApi(fetchAlerts, 10000);
  const { data: controls } = useApi(fetchControls, 10000);
  const { data: regime } = useApi(fetchRegime, 30000);
  const { data: agents } = useApi(fetchAgents, 30000);
  const { data: health } = useApi(fetchHealth, 5000);

  const risk = riskSummary || {
    currentDrawdown: 0, maxDrawdownLimit: 0.15,
    dailyLoss: 0, dailyLossLimit: 0.03,
    grossExposure: 0, grossExposureLimit: 1.0,
    netExposure: 0, leverage: 1.0,
    concentrationRisk: 0, correlationRisk: 0,
    liquidityRisk: 0, regimeMismatch: false,
    killSwitchActive: false, killSwitchReason: "",
  };

  const [controlState, setControlState] = useState<ControlState>({
    tradingPaused: false,
    entriesPaused: false,
    exitsOnly: false,
    riskOffMode: false,
    manualApprovalMode: false,
    manualOnlyMode: false,
    maxPositionSizePct: 5,
    maxExposurePct: 100,
    disabledGroups: [],
    disabledAgents: [],
  });
  const [disabledAgents, setDisabledAgents] = useState<Set<string>>(new Set());

  const killSwitchActive = risk.killSwitchActive || controls?.killSwitchActive || false;

  const handleControlChange = useCallback((key: string, value: boolean | number) => {
    setControlState((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleKillActivate = useCallback(async () => {
    await toggleKillSwitch(true);
    refreshRisk();
  }, [refreshRisk]);

  const handleKillDeactivate = useCallback(async () => {
    await toggleKillSwitch(false);
    refreshRisk();
  }, [refreshRisk]);

  const handleDisableAgent = useCallback((agentId: string) => {
    setDisabledAgents((prev) => {
      const next = new Set(prev);
      if (next.has(agentId)) next.delete(agentId);
      else next.add(agentId);
      return next;
    });
  }, []);

  // Alerts from API
  const alertList = Array.isArray(alerts) ? alerts.map((a: any, i: number) => ({
    id: String(a.id || i),
    severity: a.severity || "info",
    category: a.event_type || "",
    title: a.event_type || "Risk Event",
    message: a.message || "",
    timestamp: a.created_at || "",
    acknowledged: a.resolved || false,
    relatedSymbol: a.symbol || "",
    relatedAgent: a.agent_id || "",
  })) : [];

  const criticalCount = alertList.filter((a) => a.severity === "critical" && !a.acknowledged).length;
  const totalUnack = alertList.filter((a) => !a.acknowledged).length;

  // Agent health from real agents
  const agentHealthList = useMemo(() => {
    if (!agents) return [];
    return agents.map((a: any) => ({
      agentId: a.agentId,
      status: (a.score?.status || "healthy") as AgentStatus,
      reason: "",
      compositeWeight: a.score?.compositeWeight || 1,
      regimeMismatch: false,
      isRedundant: false,
      daysSinceLastCorrect: null as number | null,
    }));
  }, [agents]);

  const healthyAgents = agentHealthList.filter((a) => a.status === "healthy").length;
  const warningAgents = agentHealthList.filter((a) => a.status === "warning").length;
  const underperformingAgents = agentHealthList.filter((a) => a.status === "underperforming").length;
  const replaceAgents = agentHealthList.filter((a) => a.status === "replace").length;
  const flaggedAgents = agentHealthList.filter((a) => a.status !== "healthy");

  const ibkrConnected = health?.ibkr_connected || health?.ibkrConnected || false;
  const regimeLabel = regime?.regime?.replace(/_/g, " ") || "unknown";

  if (loadingRisk) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-slate-400" />
        <span className="ml-3 text-sm text-slate-400">Loading risk data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 1. PAGE HEADER */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <ShieldAlert size={22} className="text-[var(--text-primary)]" />
            <h1 className="text-lg font-bold tracking-tight text-[var(--text-primary)]">
              Risk & Control Center
            </h1>
          </div>

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

          <Badge variant="info" size="md">
            {regimeLabel}
          </Badge>
        </div>

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

      {/* 2. RISK SUMMARY CARDS */}
      <div className="grid grid-cols-6 gap-3">
        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Current Drawdown</div>
          <div className={`mt-1 text-xl font-bold ${
            risk.currentDrawdown > 0.05 ? "text-[#DC2626]" :
            risk.currentDrawdown > 0.03 ? "text-[#F59E0B]" :
            "text-[var(--text-primary)]"
          }`}>
            {(risk.currentDrawdown * 100).toFixed(1)}%
          </div>
          <ProgressBar
            value={risk.currentDrawdown * 100}
            max={risk.maxDrawdownLimit * 100}
            warningThreshold={50}
            dangerThreshold={80}
            className="mt-2"
          />
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            Limit: {(risk.maxDrawdownLimit * 100).toFixed(0)}%
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Daily PnL</div>
          <div className={`mt-1 text-xl font-bold ${getPnlColor(risk.dailyLoss)}`}>
            {formatCurrency(risk.dailyLoss)}
          </div>
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            Limit: {formatCurrency(-risk.dailyLossLimit)}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Gross Exposure</div>
          <div className="mt-1 text-xl font-bold text-[var(--text-primary)]">
            {(risk.grossExposure * 100).toFixed(0)}%
          </div>
          <ProgressBar
            value={risk.grossExposure * 100}
            max={risk.grossExposureLimit * 100}
            warningThreshold={60}
            dangerThreshold={85}
            className="mt-2"
          />
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            Limit: {(risk.grossExposureLimit * 100).toFixed(0)}%
          </div>
        </div>

        <MetricCard
          label="Net Exposure"
          value={`${(risk.netExposure * 100).toFixed(0)}%`}
          subValue={`Leverage: ${risk.leverage.toFixed(2)}x`}
          trend="neutral"
        />

        <MetricCard
          label="Concentration"
          value={`${(risk.concentrationRisk * 100).toFixed(0)}%`}
          subValue="Top position %"
          trend={risk.concentrationRisk > 0.15 ? "down" : "neutral"}
        />

        <div className="rounded-xl border border-[var(--border)] bg-white px-4 py-3 shadow-sm">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Correlation Risk</div>
          <div className={`mt-1 text-xl font-bold ${
            risk.correlationRisk > 0.7 ? "text-[#DC2626]" :
            risk.correlationRisk > 0.5 ? "text-[#F59E0B]" :
            "text-[var(--positive)]"
          }`}>
            {risk.correlationRisk.toFixed(2)}
          </div>
          <div className="mt-1 text-[10px] text-[var(--text-muted)]">
            {risk.correlationRisk > 0.7 ? "Severely correlated" :
             risk.correlationRisk > 0.5 ? "Moderately correlated" :
             "Well diversified"}
          </div>
        </div>
      </div>

      {/* 3 & 4. ALERTS + CONTROL PANEL */}
      <div className="grid grid-cols-3 gap-4" style={{ minHeight: 420 }}>
        <div className="col-span-2">
          <AlertPanel alerts={alertList} />
        </div>
        <div className="col-span-1">
          <ControlPanel state={controlState} onChange={handleControlChange} />
        </div>
      </div>

      {/* 5. KILL SWITCH */}
      <KillSwitch
        active={killSwitchActive}
        onActivate={handleKillActivate}
        onDeactivate={handleKillDeactivate}
      />

      {/* 6 & 7. AGENT HEALTH + EXECUTION HEALTH */}
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
                    <p className="text-[10px] text-[var(--text-muted)]">
                      Weight: {(agent.compositeWeight * 100).toFixed(0)}%
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
            <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="flex items-center gap-3">
                {ibkrConnected ? (
                  <Wifi size={18} className="text-[var(--positive)]" />
                ) : (
                  <WifiOff size={18} className="text-[var(--negative)]" />
                )}
                <div>
                  <p className="text-xs font-medium text-[var(--text-primary)]">
                    IBKR TWS Paper
                  </p>
                  <p className="text-[10px] text-[var(--text-muted)]">
                    Port 4002 &middot; Account DUP272334
                  </p>
                </div>
              </div>
              <Badge
                variant={ibkrConnected ? "success" : "danger"}
                size="md"
              >
                {ibkrConnected ? "Connected" : "Disconnected"}
              </Badge>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Total Agents</div>
                <div className="mt-1 text-lg font-bold text-[var(--text-primary)]">
                  {agents?.length || 0}
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  {healthyAgents} healthy
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Risk Events</div>
                <div className={`mt-1 text-lg font-bold ${
                  alertList.length === 0 ? "text-[var(--positive)]" :
                  alertList.length <= 2 ? "text-[var(--warning)]" :
                  "text-[var(--negative)]"
                }`}>
                  {alertList.length}
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  Active unresolved
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Kill Switch</div>
                <div className={`mt-1 text-lg font-bold ${
                  killSwitchActive ? "text-[var(--negative)]" : "text-[var(--positive)]"
                }`}>
                  {killSwitchActive ? "ACTIVE" : "OFF"}
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  {risk.killSwitchReason || "No issues"}
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">Database</div>
                <div className={`mt-1 text-lg font-bold ${
                  health?.supabase_connected || health?.supabaseConnected ? "text-[var(--positive)]" : "text-[var(--negative)]"
                }`}>
                  {health?.supabase_connected || health?.supabaseConnected ? "Online" : "Offline"}
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                  Supabase
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

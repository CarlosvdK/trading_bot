"use client";

import { X, AlertTriangle, CheckCircle, XCircle, Activity } from "lucide-react";
import type { AgentProfile } from "@/types";
import { Badge } from "@/components/shared/Badge";
import { ProgressBar } from "@/components/shared/ProgressBar";

interface AgentDetailProps {
  agent: AgentProfile | null;
  open: boolean;
  onClose: () => void;
}

const statusConfig: Record<string, { variant: "success" | "warning" | "danger" | "neutral"; label: string }> = {
  healthy: { variant: "success", label: "Healthy" },
  warning: { variant: "warning", label: "Warning" },
  underperforming: { variant: "danger", label: "Underperforming" },
  replace: { variant: "danger", label: "Replace" },
};

function dimensionColor(value: number): string {
  if (value >= 0.75) return "#10B981";
  if (value >= 0.6) return "#4F46E5";
  if (value >= 0.45) return "#F59E0B";
  return "#EF4444";
}

export function AgentDetail({ agent, open, onClose }: AgentDetailProps) {
  if (!open || !agent) return null;

  const status = statusConfig[agent.health.status] || statusConfig.healthy;

  const dimensions: { label: string; key: keyof typeof agent.score; invert?: boolean }[] = [
    { label: "Risk-Adj Return", key: "riskAdjustedReturn" },
    { label: "Calibration", key: "calibrationQuality" },
    { label: "Reasoning Quality", key: "reasoningQuality" },
    { label: "Uniqueness", key: "uniqueness" },
    { label: "Regime Effectiveness", key: "regimeEffectiveness" },
    { label: "Drawdown Behavior", key: "drawdownBehavior" },
    { label: "Stability", key: "stability" },
    { label: "False Positive Rate", key: "falsePositiveRate", invert: true },
    { label: "False Negative Rate", key: "falseNegativeRate", invert: true },
    { label: "Peer Vote Accuracy", key: "peerVoteAccuracy" },
  ];

  const totalProposals = agent.approvedCount + agent.rejectedCount;
  const hitRate = agent.profitableCount / Math.max(agent.profitableCount + agent.losingCount, 1);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Slide-over panel */}
      <div
        className="fixed right-0 top-0 z-50 flex h-full w-[480px] flex-col shadow-2xl"
        style={{
          background: "var(--bg-card)",
          borderLeft: "1px solid var(--border)",
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg" style={{ background: "#EEF2FF" }}>
              <Activity size={16} style={{ color: "#4F46E5" }} />
            </div>
            <div>
              <h2 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                {agent.displayName}
              </h2>
              <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                {agent.agentId}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 transition-colors"
            style={{ color: "var(--text-muted)" }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">
          {/* Profile Section */}
          <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
            <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Profile
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Primary Strategy</span>
                <div className="mt-0.5">
                  <Badge variant="default" size="md">
                    {agent.primaryStrategy.replace("_", " ")}
                  </Badge>
                </div>
              </div>
              {agent.secondaryStrategy && (
                <div>
                  <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Secondary</span>
                  <div className="mt-0.5">
                    <Badge variant="neutral" size="md">
                      {agent.secondaryStrategy.replace("_", " ")}
                    </Badge>
                  </div>
                </div>
              )}
              <div>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Peer Group</span>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{agent.peerGroup}</p>
              </div>
              <div>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Holding Period</span>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{agent.holdingPeriod}</p>
              </div>
              <div>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Primary Sectors</span>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  {agent.primarySectors.join(", ")}
                </p>
              </div>
              <div>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Lookback</span>
                <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{agent.lookbackDays} days</p>
              </div>
            </div>

            {/* Personality parameters */}
            <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 rounded-lg p-3" style={{ background: "#F8FAFC" }}>
              <ProgressBar label="Risk Appetite" value={agent.riskAppetite} showValue size="sm" color="#EF4444" />
              <ProgressBar label="Contrarian Factor" value={agent.contrarianFactor} showValue size="sm" color="#8B5CF6" />
              <ProgressBar label="Conviction Style" value={agent.convictionStyle} showValue size="sm" color="#F59E0B" />
              <ProgressBar label="Regime Sensitivity" value={agent.regimeSensitivity} showValue size="sm" color="#0EA5E9" />
            </div>
          </div>

          {/* Score Breakdown */}
          <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                Score Breakdown
              </h3>
              <div className="flex items-baseline gap-1">
                <span className="text-lg font-bold tabular-nums" style={{ color: "var(--text-primary)" }}>
                  {(agent.score.compositeWeight * 100).toFixed(1)}%
                </span>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>composite</span>
              </div>
            </div>
            <div className="flex flex-col gap-2 rounded-lg p-3" style={{ background: "#F8FAFC" }}>
              {dimensions.map((dim) => {
                const raw = agent.score[dim.key] as number;
                const displayValue = dim.invert ? (1 - raw) : raw;
                const color = dimensionColor(displayValue);
                return (
                  <ProgressBar
                    key={dim.key}
                    label={dim.label}
                    value={raw}
                    showValue
                    size="sm"
                    color={color}
                  />
                );
              })}
            </div>
            <div className="mt-2 text-right text-[10px]" style={{ color: "var(--text-muted)" }}>
              Based on {agent.score.nOutcomes} outcomes
            </div>
          </div>

          {/* Health Report */}
          <div className="px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
            <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Health Report
            </h3>
            <div className="rounded-xl border p-3" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
              <div className="flex items-center justify-between mb-2">
                <Badge variant={status.variant} size="md">{status.label}</Badge>
                {agent.health.daysSinceLastCorrect !== null && (
                  <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                    Last correct: {agent.health.daysSinceLastCorrect}d ago
                  </span>
                )}
              </div>
              <p className="text-xs mb-2" style={{ color: "var(--text-secondary)" }}>
                {agent.health.reason}
              </p>
              <div className="flex flex-wrap gap-2 mb-2">
                {agent.health.regimeMismatch && (
                  <Badge variant="warning" size="sm">Regime Mismatch</Badge>
                )}
                {agent.health.trueDecay && (
                  <Badge variant="danger" size="sm">True Decay</Badge>
                )}
                {agent.health.isRedundant && (
                  <Badge variant="neutral" size="sm">
                    Redundant w/ {agent.health.redundantWith}
                  </Badge>
                )}
              </div>
              <div className="mt-2 flex items-start gap-2 rounded-lg px-3 py-2" style={{ background: "#F8FAFC" }}>
                {agent.health.status === "healthy" ? (
                  <CheckCircle size={14} className="mt-0.5 shrink-0" style={{ color: "var(--positive)" }} />
                ) : agent.health.status === "replace" ? (
                  <XCircle size={14} className="mt-0.5 shrink-0" style={{ color: "var(--negative)" }} />
                ) : (
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" style={{ color: "var(--warning)" }} />
                )}
                <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                  {agent.health.recommendation}
                </span>
              </div>
            </div>
          </div>

          {/* Performance Stats */}
          <div className="px-5 py-4">
            <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Performance
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Total Proposals</span>
                <p className="text-lg font-bold tabular-nums" style={{ color: "var(--text-primary)" }}>{totalProposals}</p>
              </div>
              <div className="rounded-xl border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Approval Rate</span>
                <p className="text-lg font-bold tabular-nums" style={{ color: "var(--text-primary)" }}>
                  {totalProposals > 0 ? ((agent.approvedCount / totalProposals) * 100).toFixed(0) : 0}%
                </p>
              </div>
              <div className="rounded-xl border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Hit Rate</span>
                <p className={`text-lg font-bold tabular-nums`} style={{ color: hitRate >= 0.6 ? "var(--positive)" : hitRate >= 0.45 ? "var(--warning)" : "var(--negative)" }}>
                  {(hitRate * 100).toFixed(0)}%
                </p>
              </div>
              <div className="rounded-xl border px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>Win / Loss</span>
                <p className="text-lg font-bold tabular-nums">
                  <span style={{ color: "var(--positive)" }}>{agent.profitableCount}</span>
                  {" / "}
                  <span style={{ color: "var(--negative)" }}>{agent.losingCount}</span>
                </p>
              </div>
            </div>

            {/* Approved vs Rejected bar */}
            <div className="mt-4">
              <div className="flex items-center justify-between text-[10px] mb-1">
                <span style={{ color: "var(--text-muted)" }}>Approved vs Rejected</span>
                <span style={{ color: "var(--text-secondary)" }}>
                  {agent.approvedCount} / {agent.rejectedCount}
                </span>
              </div>
              <div className="flex h-2 w-full overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
                <div
                  className="h-full rounded-l-full"
                  style={{
                    width: `${(agent.approvedCount / Math.max(totalProposals, 1)) * 100}%`,
                    background: "var(--positive)",
                  }}
                />
                <div
                  className="h-full rounded-r-full"
                  style={{
                    width: `${(agent.rejectedCount / Math.max(totalProposals, 1)) * 100}%`,
                    background: "var(--negative)",
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

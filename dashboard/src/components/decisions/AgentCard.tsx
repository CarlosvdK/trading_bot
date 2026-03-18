import type { AgentProfile } from "@/types";
import { Badge } from "@/components/shared/Badge";
import { ProgressBar } from "@/components/shared/ProgressBar";

interface AgentCardProps {
  agent: AgentProfile;
  onClick: () => void;
}

const statusColors: Record<string, string> = {
  healthy: "#10B981",
  warning: "#F59E0B",
  underperforming: "#EF4444",
  replace: "#DC2626",
};

const strategyVariant: Record<string, "default" | "success" | "warning" | "danger" | "info" | "neutral"> = {
  momentum: "default",
  breakout: "info",
  mean_reversion: "warning",
  value: "neutral",
  growth: "success",
  event_driven: "warning",
  volatility: "info",
  sentiment: "neutral",
};

export function AgentCard({ agent, onClick }: AgentCardProps) {
  const statusColor = statusColors[agent.health.status] || "#94A3B8";
  const topDimensions = [
    { label: "Risk-Adj Return", value: agent.score.riskAdjustedReturn, color: "#4F46E5" },
    { label: "Calibration", value: agent.score.calibrationQuality, color: "#0EA5E9" },
    { label: "Reasoning", value: agent.score.reasoningQuality, color: "#8B5CF6" },
  ];

  return (
    <button
      onClick={onClick}
      className="w-full rounded-xl border bg-white p-4 text-left transition-all hover:shadow-md"
      style={{
        borderColor: "var(--border)",
        background: "var(--bg-card)",
      }}
    >
      {/* Header: Name + Status */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full shrink-0"
            style={{ backgroundColor: statusColor }}
          />
          <span className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>
            {agent.displayName}
          </span>
        </div>
        <Badge variant={strategyVariant[agent.primaryStrategy] || "neutral"} size="sm">
          {agent.primaryStrategy.replace("_", " ")}
        </Badge>
      </div>

      {/* Composite Weight - Large */}
      <div className="mb-3 flex items-baseline gap-2">
        <span className="text-3xl font-bold tabular-nums" style={{ color: "var(--text-primary)" }}>
          {(agent.score.compositeWeight * 100).toFixed(1)}
        </span>
        <span className="text-[10px] font-medium uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          weight %
        </span>
      </div>

      {/* Peer Group + Sectors */}
      <div className="mb-3 flex items-center gap-2">
        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>{agent.peerGroup}</span>
        <span style={{ color: "var(--border)" }}>|</span>
        <span className="text-[10px] truncate" style={{ color: "var(--text-muted)" }}>
          {agent.primarySectors.slice(0, 2).join(", ")}
        </span>
      </div>

      {/* Mini score bars */}
      <div className="flex flex-col gap-1.5 rounded-lg p-2.5" style={{ background: "#F8FAFC" }}>
        {topDimensions.map((dim) => (
          <ProgressBar
            key={dim.label}
            label={dim.label}
            value={dim.value}
            showValue
            size="sm"
            color={dim.color}
          />
        ))}
      </div>
    </button>
  );
}

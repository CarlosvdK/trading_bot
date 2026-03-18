import type { PipelineFunnel as PipelineFunnelType } from "@/types";
import { formatNumber } from "@/lib/utils";

interface PipelineFunnelProps {
  funnel: PipelineFunnelType;
}

interface FunnelStage {
  label: string;
  count: number;
  color: string;
  bgColor: string;
}

export function PipelineFunnel({ funnel }: PipelineFunnelProps) {
  const stages: FunnelStage[] = [
    {
      label: "Scanned",
      count: funnel.universeScanned,
      color: "#4F46E5",
      bgColor: "#EEF2FF",
    },
    {
      label: "Shortlisted",
      count: funnel.shortlisted,
      color: "#0EA5E9",
      bgColor: "#F0F9FF",
    },
    {
      label: "Specialist Review",
      count: funnel.specialistReviewed,
      color: "#F59E0B",
      bgColor: "#FFFBEB",
    },
    {
      label: "Global Vote",
      count: funnel.highConviction,
      color: "#8B5CF6",
      bgColor: "#F5F3FF",
    },
    {
      label: "Approved",
      count: funnel.approved,
      color: "#10B981",
      bgColor: "#ECFDF5",
    },
  ];

  const maxCount = stages[0].count;

  return (
    <div
      className="rounded-xl border shadow-sm p-5"
      style={{
        borderColor: "var(--border)",
        background: "var(--bg-card)",
      }}
    >
      <div className="mb-4 flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: "var(--text-muted)" }}
        >
          Decision Pipeline
        </h3>
        <div className="flex items-center gap-4 text-[10px]" style={{ color: "var(--text-muted)" }}>
          <span>
            Rejected:{" "}
            <span className="font-semibold" style={{ color: "var(--negative)" }}>
              {funnel.rejected}
            </span>
          </span>
          <span>
            No Trade:{" "}
            <span className="font-semibold" style={{ color: "var(--text-secondary)" }}>
              {formatNumber(funnel.noTrade)}
            </span>
          </span>
          <span>
            Downsized:{" "}
            <span className="font-semibold" style={{ color: "var(--warning)" }}>
              {funnel.downsized}
            </span>
          </span>
        </div>
      </div>

      {/* Funnel bars - horizontal flow */}
      <div className="flex items-end gap-2">
        {stages.map((stage, i) => {
          const widthPct = Math.max(
            ((stage.count / maxCount) * 100), 8
          );
          const conversionPct =
            i === 0
              ? 100
              : ((stage.count / stages[i - 1].count) * 100);

          return (
            <div key={stage.label} className="flex flex-1 flex-col items-center gap-2">
              {/* Count */}
              <span
                className="text-lg font-bold tabular-nums"
                style={{ color: stage.color }}
              >
                {formatNumber(stage.count)}
              </span>

              {/* Bar */}
              <div
                className="w-full rounded-lg transition-all duration-500"
                style={{
                  height: `${Math.max(widthPct * 0.6, 8)}px`,
                  backgroundColor: stage.bgColor,
                  borderLeft: `3px solid ${stage.color}`,
                }}
              />

              {/* Label */}
              <span
                className="text-[10px] font-medium text-center leading-tight"
                style={{ color: "var(--text-muted)" }}
              >
                {stage.label}
              </span>

              {/* Conversion rate */}
              {i > 0 && (
                <span
                  className="text-[10px] font-semibold tabular-nums"
                  style={{ color: stage.color }}
                >
                  {conversionPct.toFixed(1)}%
                </span>
              )}
              {i === 0 && (
                <span className="text-[10px] text-transparent">-</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Connecting arrows */}
      <div className="mt-2 flex items-center justify-center gap-1">
        {stages.slice(0, -1).map((_, i) => (
          <div key={i} className="flex flex-1 items-center justify-center">
            <div className="h-px flex-1" style={{ background: "var(--border)" }} />
            <svg
              width="8"
              height="8"
              viewBox="0 0 8 8"
              className="mx-1"
              style={{ color: "var(--text-muted)" }}
            >
              <path d="M0 0 L8 4 L0 8 Z" fill="currentColor" opacity="0.4" />
            </svg>
            <div className="h-px flex-1" style={{ background: "var(--border)" }} />
          </div>
        ))}
        <div className="flex-1" />
      </div>
    </div>
  );
}

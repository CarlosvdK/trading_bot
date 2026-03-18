"use client";

import { X } from "lucide-react";
import type { Position } from "@/types";
import { formatCurrency, formatPct, formatDateTime, getPnlColor, getStatusColor, getConfidenceColor } from "@/lib/utils";
import { Badge } from "@/components/shared/Badge";

interface PositionDetailProps {
  position: Position | null;
  open: boolean;
  onClose: () => void;
}

export function PositionDetail({ position, open, onClose }: PositionDetailProps) {
  if (!position) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 z-40 bg-black/20 backdrop-blur-sm transition-opacity duration-300 ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={`fixed right-0 top-0 z-50 flex h-full w-[480px] flex-col border-l border-[var(--border)] bg-white shadow-2xl transition-transform duration-300 ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[var(--border)] px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="text-lg font-bold text-[var(--text-primary)]">
              {position.symbol}
            </span>
            <Badge
              label={position.direction}
              color={position.direction === "long" ? "green" : "red"}
            />
            <Badge
              label={position.status.replace("_", " ")}
              color={getStatusColor(position.status) as "green" | "yellow" | "red" | "gray"}
            />
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-[var(--text-muted)] transition-colors hover:bg-slate-100 hover:text-[var(--text-secondary)]"
          >
            <X size={18} />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {/* PnL Summary */}
          <div className="mb-6 grid grid-cols-2 gap-4">
            <div>
              <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Unrealized PnL
              </span>
              <p className={`font-mono text-lg font-bold ${getPnlColor(position.unrealizedPnl)}`}>
                {formatCurrency(position.unrealizedPnl)}
              </p>
              <p className={`font-mono text-xs ${getPnlColor(position.unrealizedPnlPct)}`}>
                {formatPct(position.unrealizedPnlPct)}
              </p>
            </div>
            <div>
              <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                Market Value
              </span>
              <p className="font-mono text-lg font-bold text-[var(--text-primary)]">
                {formatCurrency(position.marketValue)}
              </p>
              <p className="font-mono text-xs text-[var(--text-muted)]">
                {position.quantity} shares @ {formatCurrency(position.currentPrice)}
              </p>
            </div>
          </div>

          {/* Thesis & Rationale */}
          <Section title="Thesis">
            <p className="text-xs leading-relaxed text-[var(--text-secondary)]">{position.thesis}</p>
            <div className="mt-2 space-y-1">
              {position.keyReasons.map((reason, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-[var(--text-muted)]">
                  <span className="mt-0.5 text-emerald-600">+</span>
                  <span>{reason}</span>
                </div>
              ))}
            </div>
            {position.dissentNotes && (
              <div className="mt-2 rounded-lg border border-red-200 bg-[var(--negative-light)] px-3 py-2">
                <span className="text-[10px] font-semibold uppercase text-red-500">Dissent</span>
                <p className="mt-0.5 text-xs text-[var(--text-muted)]">{position.dissentNotes}</p>
              </div>
            )}
          </Section>

          {/* Vehicle Selection */}
          <Section title="Vehicle Selection">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Type</span>
                <Badge label={position.investmentType.replace("_", " ")} color="blue" />
              </div>
              <p className="text-xs leading-relaxed text-[var(--text-muted)]">{position.vehicleRationale}</p>
            </div>
          </Section>

          {/* Vote Results */}
          <Section title="Vote Results">
            <div className="grid grid-cols-2 gap-3">
              {position.specialistResult && (
                <div className="rounded-lg border border-[var(--border)] bg-slate-50 p-3">
                  <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">
                    Specialist Confidence
                  </span>
                  <p className={`mt-0.5 font-mono text-sm font-bold ${getConfidenceColor(position.specialistResult.confidenceScore)}`}>
                    {(position.specialistResult.confidenceScore * 100).toFixed(0)}%
                  </p>
                  <p className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                    {position.specialistResult.approvalCount}A / {position.specialistResult.rejectCount}R / {position.specialistResult.modifyCount}M
                  </p>
                </div>
              )}
              <div className="rounded-lg border border-[var(--border)] bg-slate-50 p-3">
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">
                  Global Approval
                </span>
                <p className={`mt-0.5 font-mono text-sm font-bold ${getConfidenceColor(position.globalVoteResult)}`}>
                  {(position.globalVoteResult * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            {position.supportingAgents.length > 0 && (
              <div className="mt-3">
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Supporting</span>
                <div className="mt-1 flex flex-wrap gap-1">
                  {position.supportingAgents.map((a) => (
                    <Badge key={a} label={a} color="green" />
                  ))}
                </div>
              </div>
            )}
            {position.dissentingAgents.length > 0 && (
              <div className="mt-2">
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Dissenting</span>
                <div className="mt-1 flex flex-wrap gap-1">
                  {position.dissentingAgents.map((a) => (
                    <Badge key={a} label={a} color="red" />
                  ))}
                </div>
              </div>
            )}
          </Section>

          {/* Risk */}
          <Section title="Risk">
            <div className="grid grid-cols-3 gap-3">
              <MiniStat label="Stop" value={formatCurrency(position.stopLevel)} color="text-red-500" />
              <MiniStat label="Target" value={formatCurrency(position.targetLevel)} color="text-emerald-600" />
              <MiniStat
                label="Confidence"
                value={`${(position.confidence * 100).toFixed(0)}%`}
                color={getConfidenceColor(position.confidence)}
              />
            </div>
            {position.riskFactors.length > 0 && (
              <div className="mt-3 space-y-1">
                {position.riskFactors.map((rf, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs text-[var(--text-muted)]">
                    <span className="mt-0.5 text-amber-500">!</span>
                    <span>{rf}</span>
                  </div>
                ))}
              </div>
            )}
          </Section>

          {/* Execution */}
          <Section title="Execution">
            <div className="space-y-2">
              <div>
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Entry</span>
                <p className="text-xs text-[var(--text-muted)]">{position.executionNotes}</p>
              </div>
              <div>
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Exit Logic</span>
                <p className="text-xs text-[var(--text-muted)]">{position.plannedExitCondition}</p>
              </div>
              <div className="flex gap-4">
                <div>
                  <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">
                    Holding Category
                  </span>
                  <p className="text-xs text-[var(--text-secondary)]">{position.holdingCategory}</p>
                </div>
                <div>
                  <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">
                    Planned Exit
                  </span>
                  <p className="text-xs text-[var(--text-secondary)]">{position.plannedExitDate}</p>
                </div>
              </div>
            </div>
          </Section>

          {/* Review */}
          <Section title="Review">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Last Review</span>
                <p className="font-mono text-xs text-[var(--text-secondary)]">
                  {formatDateTime(position.lastReviewTime)}
                </p>
              </div>
              <div>
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">Next Review</span>
                <p className="font-mono text-xs text-[var(--text-secondary)]">
                  {formatDateTime(position.nextReviewTime)}
                </p>
              </div>
            </div>
          </Section>

          {/* Agent Attribution */}
          <Section title="Agent Attribution">
            <div className="space-y-2">
              <div>
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">
                  Originating Group
                </span>
                <p className="text-xs text-[var(--text-secondary)]">{position.originAgentGroup}</p>
              </div>
              <div>
                <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">
                  Voting Round
                </span>
                <p className="font-mono text-xs text-[var(--text-secondary)]">{position.votingRound}</p>
              </div>
            </div>
          </Section>
        </div>
      </div>
    </>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-5 border-t border-slate-100 pt-4">
      <h4 className="mb-2.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
        {title}
      </h4>
      {children}
    </div>
  );
}

function MiniStat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-slate-50 p-2.5 text-center">
      <span className="text-[10px] font-medium uppercase text-[var(--text-muted)]">{label}</span>
      <p className={`mt-0.5 font-mono text-sm font-bold ${color}`}>{value}</p>
    </div>
  );
}

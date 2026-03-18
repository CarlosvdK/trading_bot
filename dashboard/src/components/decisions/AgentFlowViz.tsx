"use client";

import { useMemo } from "react";
import type { AgentProfile, PeerGroup, Proposal, DecisionOutput } from "@/types";
import { Badge } from "@/components/shared/Badge";

interface AgentFlowVizProps {
  groups: PeerGroup[];
  agents: AgentProfile[];
  proposals: Proposal[];
  decisions: DecisionOutput[];
}

// ============================================================
// Layout constants
// ============================================================

const CANVAS_W = 1200;
const CANVAS_H = 720;

// Stage X positions
const STAGE_1_X = 40;
const STAGE_2_X = 440;
const STAGE_3_X = 740;
const STAGE_4_X = 1000;

// Agent node dimensions
const AGENT_W = 110;
const AGENT_H = 36;

// Group box padding
const GROUP_PAD_X = 12;
const GROUP_PAD_Y = 28;
const GROUP_GAP = 8;

// Specialist node
const SPEC_W = 130;
const SPEC_H = 50;

// Roundtable
const RT_CX = STAGE_3_X + 80;
const RT_CY = CANVAS_H / 2;
const RT_R = 64;

// Rejected bin
const BIN_W = 100;
const BIN_H = 36;
const BIN_X = STAGE_2_X + 50;
const BIN_Y = CANVAS_H - 60;

// Decision card
const DEC_W = 120;
const DEC_H = 50;

// ============================================================
// Helper: compute layout positions for groups & agents
// ============================================================

interface AgentNode {
  id: string;
  name: string;
  score: number;
  status: string;
  x: number;
  y: number;
  groupName: string;
}

interface GroupBox {
  name: string;
  specialty: string;
  x: number;
  y: number;
  w: number;
  h: number;
  agents: AgentNode[];
  outputX: number;
  outputY: number;
  surfacedCount: number;
}

interface SpecNode {
  groupName: string;
  x: number;
  y: number;
  passedCount: number;
  totalCount: number;
}

function computeLayout(
  groups: PeerGroup[],
  agents: AgentProfile[]
): { groupBoxes: GroupBox[]; specNodes: SpecNode[] } {
  const totalAgents = groups.reduce((s, g) => s + g.agents.length, 0);
  const totalGroupHeight =
    groups.length * GROUP_PAD_Y +
    totalAgents * (AGENT_H + GROUP_GAP) +
    (groups.length - 1) * 16;
  const startY = Math.max(20, (CANVAS_H - totalGroupHeight) / 2);

  let curY = startY;
  const groupBoxes: GroupBox[] = [];
  const specNodes: SpecNode[] = [];

  groups.forEach((group) => {
    const groupAgents = group.agents.map((aid) =>
      agents.find((a) => a.agentId === aid)
    ).filter(Boolean) as AgentProfile[];

    const innerH = groupAgents.length * (AGENT_H + GROUP_GAP) - GROUP_GAP;
    const boxH = innerH + GROUP_PAD_Y + 12;
    const boxW = AGENT_W + GROUP_PAD_X * 2;

    const agentNodes: AgentNode[] = groupAgents.map((a, i) => ({
      id: a.agentId,
      name: a.displayName,
      score: Math.round(a.score.riskAdjustedReturn * 100),
      status: a.health.status,
      x: STAGE_1_X + GROUP_PAD_X,
      y: curY + GROUP_PAD_Y + i * (AGENT_H + GROUP_GAP),
      groupName: group.name,
    }));

    const box: GroupBox = {
      name: group.name,
      specialty: group.specialty,
      x: STAGE_1_X,
      y: curY,
      w: boxW,
      h: boxH,
      agents: agentNodes,
      outputX: STAGE_1_X + boxW,
      outputY: curY + boxH / 2,
      surfacedCount: group.surfacedCount,
    };

    groupBoxes.push(box);

    specNodes.push({
      groupName: group.name,
      x: STAGE_2_X,
      y: curY + boxH / 2 - SPEC_H / 2,
      passedCount: group.passedCount,
      totalCount: group.surfacedCount,
    });

    curY += boxH + 16;
  });

  return { groupBoxes, specNodes };
}

// ============================================================
// SVG path helpers
// ============================================================

function bezierH(x1: number, y1: number, x2: number, y2: number): string {
  const cpOffset = Math.abs(x2 - x1) * 0.45;
  return `M ${x1} ${y1} C ${x1 + cpOffset} ${y1}, ${x2 - cpOffset} ${y2}, ${x2} ${y2}`;
}

// ============================================================
// Status color
// ============================================================

const statusDotColor: Record<string, string> = {
  healthy: "#10B981",
  warning: "#F59E0B",
  underperforming: "#EF4444",
  replace: "#DC2626",
};

// ============================================================
// Component
// ============================================================

export function AgentFlowViz({ groups, agents, proposals, decisions }: AgentFlowVizProps) {
  const { groupBoxes, specNodes } = useMemo(
    () => computeLayout(groups, agents),
    [groups, agents]
  );

  // Compute summary stats
  const totalScanned = groups.reduce((s, g) => s + g.scannedCount, 0);
  const totalSurfaced = groups.reduce((s, g) => s + g.surfacedCount, 0);
  const specialistReviewed = proposals.filter(
    (p) => p.specialistReviewPassed !== undefined
  ).length;
  const votedCount = proposals.filter(
    (p) =>
      p.pipelineStage === "global_vote" ||
      p.pipelineStage === "approved" ||
      p.pipelineStage === "rejected"
  ).length;
  const approvedCount = proposals.filter(
    (p) => p.pipelineStage === "approved"
  ).length;
  const rejectedCount = proposals.filter(
    (p) => p.pipelineStage === "rejected"
  ).length;

  // Build decision cards layout
  const approvedDecisions = decisions.slice(0, 4);
  const rejectedProposals = proposals
    .filter((p) => p.pipelineStage === "rejected")
    .slice(0, 2);

  return (
    <div className="space-y-4">
      {/* Flow visualization */}
      <div
        className="rounded-xl border border-[var(--border)] bg-white shadow-sm overflow-x-auto"
        style={{ background: "var(--bg-card)" }}
      >
        <div className="relative" style={{ minWidth: CANVAS_W, height: CANVAS_H }}>
          {/* ---- SVG Layer (connection lines + animated dots) ---- */}
          <svg
            className="absolute inset-0 pointer-events-none"
            width={CANVAS_W}
            height={CANVAS_H}
            style={{ zIndex: 1 }}
          >
            <defs>
              {/* Animated dot keyframes via SVG */}
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Group output -> Specialist node connections */}
            {groupBoxes.map((box, i) => {
              const spec = specNodes[i];
              const path = bezierH(
                box.outputX,
                box.outputY,
                spec.x,
                spec.y + SPEC_H / 2
              );
              return (
                <g key={`conn-gs-${i}`}>
                  <path
                    d={path}
                    stroke="#CBD5E1"
                    strokeWidth={1.5}
                    fill="none"
                  />
                  {/* Animated dot */}
                  <circle r={3} fill="#4F46E5" filter="url(#glow)">
                    <animateMotion
                      dur="3s"
                      repeatCount="indefinite"
                      begin={`${i * 0.5}s`}
                      path={path}
                    />
                  </circle>
                  {/* Second dot staggered */}
                  <circle r={2.5} fill="#818CF8" opacity={0.7}>
                    <animateMotion
                      dur="3s"
                      repeatCount="indefinite"
                      begin={`${i * 0.5 + 1.5}s`}
                      path={path}
                    />
                  </circle>
                </g>
              );
            })}

            {/* Specialist -> Roundtable connections (passed only) */}
            {specNodes.map((spec, i) => {
              if (groupBoxes[i] && specNodes[i].passedCount > 0) {
                const path = bezierH(
                  spec.x + SPEC_W,
                  spec.y + SPEC_H / 2,
                  RT_CX - RT_R,
                  RT_CY
                );
                return (
                  <g key={`conn-sr-${i}`}>
                    <path
                      d={path}
                      stroke="#CBD5E1"
                      strokeWidth={1.5}
                      fill="none"
                    />
                    <circle r={3} fill="#10B981">
                      <animateMotion
                        dur="4s"
                        repeatCount="indefinite"
                        begin={`${i * 0.6 + 0.3}s`}
                        path={path}
                      />
                    </circle>
                  </g>
                );
              }
              return null;
            })}

            {/* Specialist -> Rejected bin connections (failed) */}
            {specNodes.map((spec, i) => {
              const failedCount =
                specNodes[i].totalCount - specNodes[i].passedCount;
              if (failedCount > 0) {
                const path = bezierH(
                  spec.x + SPEC_W / 2,
                  spec.y + SPEC_H,
                  BIN_X + BIN_W / 2,
                  BIN_Y
                );
                return (
                  <g key={`conn-sb-${i}`}>
                    <path
                      d={path}
                      stroke="#FCA5A5"
                      strokeWidth={1}
                      fill="none"
                      strokeDasharray="4 3"
                      opacity={0.5}
                    />
                    <circle r={2} fill="#EF4444" opacity={0.6}>
                      <animateMotion
                        dur="4s"
                        repeatCount="indefinite"
                        begin={`${i * 0.8 + 1}s`}
                        path={path}
                      />
                    </circle>
                  </g>
                );
              }
              return null;
            })}

            {/* Roundtable -> Approved decisions */}
            {approvedDecisions.map((dec, i) => {
              const targetY = 100 + i * (DEC_H + 16);
              const path = bezierH(
                RT_CX + RT_R,
                RT_CY,
                STAGE_4_X,
                targetY + DEC_H / 2
              );
              return (
                <g key={`conn-rd-${i}`}>
                  <path
                    d={path}
                    stroke="#A7F3D0"
                    strokeWidth={1.5}
                    fill="none"
                  />
                  <circle r={3} fill="#10B981">
                    <animateMotion
                      dur="3.5s"
                      repeatCount="indefinite"
                      begin={`${i * 0.7 + 0.5}s`}
                      path={path}
                    />
                  </circle>
                </g>
              );
            })}

            {/* Roundtable -> Rejected proposals exit */}
            {rejectedProposals.map((prop, i) => {
              const targetY = 100 + (approvedDecisions.length + i) * (DEC_H + 16) + 20;
              const path = bezierH(
                RT_CX + RT_R,
                RT_CY,
                STAGE_4_X,
                Math.min(targetY + DEC_H / 2, CANVAS_H - 80)
              );
              return (
                <g key={`conn-rr-${i}`}>
                  <path
                    d={path}
                    stroke="#FECACA"
                    strokeWidth={1}
                    fill="none"
                    strokeDasharray="4 3"
                    opacity={0.6}
                  />
                </g>
              );
            })}

            {/* Roundtable pulse ring */}
            <circle
              cx={RT_CX}
              cy={RT_CY}
              r={RT_R + 4}
              fill="none"
              stroke="#4F46E5"
              strokeWidth={2}
              opacity={0.3}
            >
              <animate
                attributeName="r"
                values={`${RT_R + 2};${RT_R + 14};${RT_R + 2}`}
                dur="3s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0.4;0;0.4"
                dur="3s"
                repeatCount="indefinite"
              />
            </circle>
          </svg>

          {/* ---- HTML Layer (nodes) ---- */}
          <div className="absolute inset-0" style={{ zIndex: 2 }}>
            {/* Stage labels */}
            <div
              className="absolute text-[10px] font-semibold uppercase tracking-widest"
              style={{ left: STAGE_1_X, top: 6, color: "var(--text-muted)" }}
            >
              Agent Clusters
            </div>
            <div
              className="absolute text-[10px] font-semibold uppercase tracking-widest"
              style={{ left: STAGE_2_X, top: 6, color: "var(--text-muted)" }}
            >
              Specialist Review
            </div>
            <div
              className="absolute text-[10px] font-semibold uppercase tracking-widest"
              style={{ left: STAGE_3_X + 20, top: 6, color: "var(--text-muted)" }}
            >
              Global Vote
            </div>
            <div
              className="absolute text-[10px] font-semibold uppercase tracking-widest"
              style={{ left: STAGE_4_X, top: 6, color: "var(--text-muted)" }}
            >
              Decisions
            </div>

            {/* Group boxes with agent nodes */}
            {groupBoxes.map((box) => (
              <div key={box.name}>
                {/* Group container */}
                <div
                  className="absolute rounded-lg border border-dashed"
                  style={{
                    left: box.x,
                    top: box.y,
                    width: box.w,
                    height: box.h,
                    borderColor: "#CBD5E1",
                    background: "rgba(248, 250, 252, 0.6)",
                  }}
                >
                  <div
                    className="absolute -top-0 left-2 px-1.5 text-[9px] font-semibold uppercase tracking-wider"
                    style={{
                      color: "var(--text-secondary)",
                      background: "var(--bg-card)",
                      transform: "translateY(-50%)",
                    }}
                  >
                    {box.name}
                  </div>
                </div>

                {/* Agent nodes inside group */}
                {box.agents.map((agent) => (
                  <div
                    key={agent.id}
                    className="absolute flex items-center gap-1.5 rounded-lg border bg-white shadow-sm px-2.5"
                    style={{
                      left: agent.x,
                      top: agent.y,
                      width: AGENT_W,
                      height: AGENT_H,
                      borderColor: "#E2E8F0",
                    }}
                  >
                    <span
                      className="inline-block h-2 w-2 rounded-full shrink-0"
                      style={{
                        backgroundColor:
                          statusDotColor[agent.status] || "#94A3B8",
                      }}
                    />
                    <span
                      className="text-[10px] font-medium truncate"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {agent.name}
                    </span>
                    <span
                      className="ml-auto text-[10px] font-bold tabular-nums shrink-0"
                      style={{ color: "var(--accent)" }}
                    >
                      {agent.score}
                    </span>
                  </div>
                ))}
              </div>
            ))}

            {/* Specialist review nodes */}
            {specNodes.map((spec, i) => {
              const passed = spec.passedCount;
              const total = spec.totalCount;
              const pct = total > 0 ? Math.round((passed / total) * 100) : 0;
              return (
                <div
                  key={`spec-${i}`}
                  className="absolute flex flex-col items-center justify-center rounded-lg border shadow-sm"
                  style={{
                    left: spec.x,
                    top: spec.y,
                    width: SPEC_W,
                    height: SPEC_H,
                    borderColor: "#93C5FD",
                    background: "white",
                  }}
                >
                  <div className="flex items-center gap-1">
                    <span
                      className="text-[10px] font-semibold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {groupBoxes[i]?.name.split(" ")[0]}
                    </span>
                    <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                      Review
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="flex items-center gap-0.5 text-[10px] font-bold" style={{ color: "#10B981" }}>
                      <svg width="10" height="10" viewBox="0 0 10 10">
                        <path d="M2 5 L4 7 L8 3" stroke="#10B981" strokeWidth="1.5" fill="none" />
                      </svg>
                      {passed}
                    </span>
                    <span className="flex items-center gap-0.5 text-[10px] font-bold" style={{ color: "#EF4444" }}>
                      <svg width="10" height="10" viewBox="0 0 10 10">
                        <path d="M3 3 L7 7 M7 3 L3 7" stroke="#EF4444" strokeWidth="1.5" fill="none" />
                      </svg>
                      {total - passed}
                    </span>
                    <span className="text-[9px] font-medium tabular-nums" style={{ color: "var(--text-muted)" }}>
                      {pct}%
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Roundtable node */}
            <div
              className="absolute flex flex-col items-center justify-center rounded-full shadow-lg"
              style={{
                left: RT_CX - RT_R,
                top: RT_CY - RT_R,
                width: RT_R * 2,
                height: RT_R * 2,
                background: "linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 50%, #C7D2FE 100%)",
                border: "2.5px solid #4F46E5",
              }}
            >
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "#4F46E5" }}>
                Roundtable
              </span>
              <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                Vote
              </span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs font-bold" style={{ color: "#10B981" }}>
                  {approvedCount}
                </span>
                <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>/</span>
                <span className="text-xs font-bold" style={{ color: "#EF4444" }}>
                  {rejectedCount + (votedCount - approvedCount - rejectedCount)}
                </span>
              </div>
              <div className="mt-0.5 text-[8px] font-medium" style={{ color: "var(--text-muted)" }}>
                70% threshold
              </div>
            </div>

            {/* Approved decision cards */}
            {approvedDecisions.map((dec, i) => {
              const targetY = 100 + i * (DEC_H + 16);
              return (
                <div
                  key={dec.decisionId}
                  className="absolute flex items-center gap-2 rounded-lg border shadow-sm px-3"
                  style={{
                    left: STAGE_4_X,
                    top: targetY,
                    width: DEC_W,
                    height: DEC_H,
                    borderColor: "#A7F3D0",
                    background: "#F0FDF4",
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" className="shrink-0">
                    <circle cx="7" cy="7" r="6" fill="#10B981" />
                    <path d="M4 7 L6 9 L10 5" stroke="white" strokeWidth="1.5" fill="none" />
                  </svg>
                  <div className="min-w-0">
                    <div className="text-[11px] font-bold" style={{ color: "var(--text-primary)" }}>
                      {dec.asset}
                    </div>
                    <div className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                      {(dec.confidence * 100).toFixed(0)}% conf
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Rejected proposal cards */}
            {rejectedProposals.map((prop, i) => {
              const targetY =
                100 +
                (approvedDecisions.length + i) * (DEC_H + 16) +
                20;
              return (
                <div
                  key={prop.proposalId}
                  className="absolute flex items-center gap-2 rounded-lg border shadow-sm px-3 opacity-60"
                  style={{
                    left: STAGE_4_X,
                    top: Math.min(targetY, CANVAS_H - 80),
                    width: DEC_W,
                    height: DEC_H,
                    borderColor: "#FECACA",
                    background: "#FEF2F2",
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" className="shrink-0">
                    <circle cx="7" cy="7" r="6" fill="#EF4444" />
                    <path d="M5 5 L9 9 M9 5 L5 9" stroke="white" strokeWidth="1.5" fill="none" />
                  </svg>
                  <div className="min-w-0">
                    <div className="text-[11px] font-bold line-through" style={{ color: "var(--text-muted)" }}>
                      {prop.symbol}
                    </div>
                    <div className="text-[9px]" style={{ color: "var(--negative)" }}>
                      Rejected
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Rejected bin */}
            <div
              className="absolute flex items-center justify-center gap-1.5 rounded-md border shadow-sm"
              style={{
                left: BIN_X,
                top: BIN_Y,
                width: BIN_W,
                height: BIN_H,
                borderColor: "#FECACA",
                background: "#FEF2F2",
              }}
            >
              <svg width="12" height="12" viewBox="0 0 12 12">
                <path d="M3 3 L9 9 M9 3 L3 9" stroke="#EF4444" strokeWidth="1.5" fill="none" />
              </svg>
              <span className="text-[10px] font-semibold" style={{ color: "#EF4444" }}>
                Rejected Bin
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Summary bar */}
      <div
        className="rounded-xl border border-[var(--border)] bg-white shadow-sm px-6 py-3 flex items-center gap-3 text-xs"
        style={{ background: "var(--bg-card)" }}
      >
        <SummaryStep
          count={totalScanned}
          label="scanned"
          color="#4F46E5"
        />
        <ArrowSep />
        <SummaryStep
          count={totalSurfaced}
          label="surfaced"
          color="#0EA5E9"
        />
        <ArrowSep />
        <SummaryStep
          count={specialistReviewed}
          label="specialist reviewed"
          color="#F59E0B"
        />
        <ArrowSep />
        <SummaryStep
          count={votedCount}
          label="voted"
          color="#8B5CF6"
        />
        <ArrowSep />
        <SummaryStep
          count={approvedCount}
          label="approved"
          color="#10B981"
        />
        <span className="mx-1 text-[var(--text-muted)]">|</span>
        <SummaryStep
          count={rejectedCount}
          label="rejected"
          color="#EF4444"
        />
      </div>
    </div>
  );
}

function SummaryStep({
  count,
  label,
  color,
}: {
  count: number;
  label: string;
  color: string;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="font-bold tabular-nums" style={{ color }}>
        {count}
      </span>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
    </span>
  );
}

function ArrowSep() {
  return (
    <svg width="16" height="12" viewBox="0 0 16 12" className="shrink-0">
      <path
        d="M0 6 L12 6 M9 3 L12 6 L9 9"
        stroke="#CBD5E1"
        strokeWidth="1.5"
        fill="none"
      />
    </svg>
  );
}

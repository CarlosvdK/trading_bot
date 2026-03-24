"use client";

import { useRef, useState, useMemo, useEffect } from "react";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface Agent {
  agentId: string;
  displayName: string;
  primarySectors: string[];
  primaryStrategy: string;
  secondaryStrategy?: string | null;
  peerGroup: string;
  compositeWeight: number;
  status: string;
  holdingPeriod?: string;
  lookbackDays?: number;
  riskAppetite?: number;
  riskAdjustedReturn?: number;
  calibrationQuality?: number;
  reasoningQuality?: number;
}

interface ClusterData {
  id: string;
  name: string;
  agents: Agent[];
  shape: string;
  color: string;
  x: number;
  y: number;
  angle: number;
}

interface AgentPosition {
  agent: Agent;
  x: number;
  y: number;
  cluster: ClusterData;
}

interface Trade {
  trade_id?: string;
  agent_id?: string;
  symbol?: string;
  direction?: string;
  confidence?: number;
  strategy_used?: string;
  reasoning?: string;
  vote_result?: string;
  approval_pct?: number;
  num_voters?: number;
  outcome?: string;
  pnl?: number;
  sector?: string;
  supporting_agents?: string[];
  dissenting_agents?: string[];
  proposed_at?: string;
}

interface Props {
  agents: Agent[];
  trades?: Trade[];
  onSelectAgent?: (id: string) => void;
}

/* ------------------------------------------------------------------ */
/* Shape + Color mapping                                               */
/* ------------------------------------------------------------------ */

const CLUSTER_CONFIG: Record<string, { shape: string; color: string; label: string }> = {
  tech_cluster:        { shape: "circle",   color: "#6366F1", label: "Technology" },
  healthcare_cluster:  { shape: "diamond",  color: "#EC4899", label: "Healthcare" },
  consumer_cluster:    { shape: "triangle", color: "#8B5CF6", label: "Consumer" },
  financials_cluster:  { shape: "square",   color: "#10B981", label: "Financials" },
  energy_cluster:      { shape: "hexagon",  color: "#F59E0B", label: "Energy" },
  industrials_cluster: { shape: "octagon",  color: "#F97316", label: "Industrials" },
  momentum_cluster:    { shape: "pentagon", color: "#3B82F6", label: "Momentum" },
  value_cluster:       { shape: "star",     color: "#06B6D4", label: "Value" },
  mean_rev_cluster:    { shape: "diamond",  color: "#14B8A6", label: "Mean Reversion" },
  wildcard_cluster:    { shape: "cross",    color: "#EF4444", label: "Wild Cards" },
  sniper_cluster:      { shape: "triangle", color: "#D946EF", label: "Snipers" },
  guardian_cluster:    { shape: "octagon",  color: "#64748B", label: "Guardians" },
  regime_cluster:      { shape: "hexagon",  color: "#22D3EE", label: "Regime" },
  flow_cluster:        { shape: "pentagon", color: "#2563EB", label: "Volume Flow" },
  contrarian_cluster:  { shape: "star",     color: "#EA580C", label: "Contrarian" },
  comms_cluster:       { shape: "circle",   color: "#3B82F6", label: "Comms" },
  materials_cluster:   { shape: "star",     color: "#84CC16", label: "Materials" },
  real_estate_cluster: { shape: "diamond",  color: "#0D9488", label: "Real Estate" },
  blender_cluster:     { shape: "pentagon", color: "#7C3AED", label: "Blender" },
};

const DEFAULT_CONFIG = { shape: "circle", color: "#94A3B8", label: "Other" };

/* ------------------------------------------------------------------ */
/* SVG Shape Renderer                                                  */
/* ------------------------------------------------------------------ */

function getShapePoints(shape: string, s: number): string {
  switch (shape) {
    case "square":
      return `${-s},${-s} ${s},${-s} ${s},${s} ${-s},${s}`;
    case "diamond":
      return `0,${-s} ${s},0 0,${s} ${-s},0`;
    case "triangle":
      return `0,${-s} ${s * 0.87},${s * 0.5} ${-s * 0.87},${s * 0.5}`;
    case "hexagon": {
      const pts = [];
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i - Math.PI / 2;
        pts.push(`${(Math.cos(a) * s).toFixed(1)},${(Math.sin(a) * s).toFixed(1)}`);
      }
      return pts.join(" ");
    }
    case "pentagon": {
      const pts = [];
      for (let i = 0; i < 5; i++) {
        const a = (Math.PI * 2 / 5) * i - Math.PI / 2;
        pts.push(`${(Math.cos(a) * s).toFixed(1)},${(Math.sin(a) * s).toFixed(1)}`);
      }
      return pts.join(" ");
    }
    case "octagon": {
      const pts = [];
      for (let i = 0; i < 8; i++) {
        const a = (Math.PI / 4) * i - Math.PI / 8;
        pts.push(`${(Math.cos(a) * s).toFixed(1)},${(Math.sin(a) * s).toFixed(1)}`);
      }
      return pts.join(" ");
    }
    case "star": {
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const a = (Math.PI / 5) * i - Math.PI / 2;
        const r = i % 2 === 0 ? s : s * 0.45;
        pts.push(`${(Math.cos(a) * r).toFixed(1)},${(Math.sin(a) * r).toFixed(1)}`);
      }
      return pts.join(" ");
    }
    case "cross": {
      const w = s * 0.35;
      return `${-w},${-s} ${w},${-s} ${w},${-w} ${s},${-w} ${s},${w} ${w},${w} ${w},${s} ${-w},${s} ${-w},${w} ${-s},${w} ${-s},${-w} ${-w},${-w}`;
    }
    default:
      return "";
  }
}

function ShapeNode({
  shape, size, fill, stroke, strokeWidth = 2,
}: {
  shape: string; size: number; fill: string; stroke: string; strokeWidth?: number;
}) {
  if (shape === "circle") {
    return <circle r={size} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />;
  }
  const points = getShapePoints(shape, size);
  return <polygon points={points} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />;
}

function curvedPath(x1: number, y1: number, x2: number, y2: number): string {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const cx1 = x1 + dx * 0.3;
  const cy1 = y1 + dy * 0.05;
  const cx2 = x1 + dx * 0.7;
  const cy2 = y1 + dy * 0.95;
  return `M${x1},${y1} C${cx1},${cy1} ${cx2},${cy2} ${x2},${y2}`;
}

/* ------------------------------------------------------------------ */
/* Main Canvas                                                         */
/* ------------------------------------------------------------------ */

export function SwarmCanvas({ agents, trades = [], onSelectAgent }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 1200, height: 800 });
  const [hoveredCluster, setHoveredCluster] = useState<string | null>(null);
  const [hoveredAgent, setHoveredAgent] = useState<AgentPosition | null>(null);
  const [hoveredCenter, setHoveredCenter] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      setSize({ width: Math.max(800, width), height: 800 });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const cx = size.width / 2;
  const cy = size.height / 2;
  const outerRadius = Math.min(size.width * 0.4, size.height * 0.38);

  // Group into clusters
  const clusters: ClusterData[] = useMemo(() => {
    const groups: Record<string, Agent[]> = {};
    agents.forEach((a) => {
      const key = a.peerGroup || "other";
      if (!groups[key]) groups[key] = [];
      groups[key].push(a);
    });

    const mainKeys = Object.keys(groups).filter((k) => groups[k].length >= 2);
    const otherAgents = Object.keys(groups)
      .filter((k) => groups[k].length < 2)
      .flatMap((k) => groups[k]);
    if (otherAgents.length > 0) {
      groups["other"] = [...(groups["other"] || []), ...otherAgents.filter((a) => a.peerGroup !== "other")];
      if (!mainKeys.includes("other") && groups["other"]?.length >= 1) mainKeys.push("other");
    }

    return mainKeys.map((key, i) => {
      const angle = (2 * Math.PI * i) / mainKeys.length - Math.PI / 2;
      const conf = CLUSTER_CONFIG[key] || DEFAULT_CONFIG;
      return {
        id: key,
        name: conf.label || key.replace(/_/g, " "),
        agents: groups[key],
        shape: conf.shape,
        color: conf.color,
        x: cx + Math.cos(angle) * outerRadius,
        y: cy + Math.sin(angle) * outerRadius,
        angle,
      };
    });
  }, [agents, cx, cy, outerRadius]);

  // Pre-compute all agent dot positions for hit detection
  const agentPositions: AgentPosition[] = useMemo(() => {
    const positions: AgentPosition[] = [];
    clusters.forEach((cluster) => {
      const agentRadius = Math.min(60, 28 + cluster.agents.length * 2.5);
      cluster.agents.forEach((agent, i) => {
        const spread = Math.min(0.28, 1.8 / cluster.agents.length);
        const aAngle = cluster.angle + ((i - (cluster.agents.length - 1) / 2) * spread);
        positions.push({
          agent,
          x: cluster.x + Math.cos(aAngle) * agentRadius,
          y: cluster.y + Math.sin(aAngle) * agentRadius,
          cluster,
        });
      });
    });
    return positions;
  }, [clusters]);

  // Group trades by sector for cluster tooltips
  const tradesBySector = useMemo(() => {
    const map: Record<string, Trade[]> = {};
    trades.forEach((t) => {
      const sector = (t.sector || "other").toLowerCase();
      if (!map[sector]) map[sector] = [];
      map[sector].push(t);
    });
    return map;
  }, [trades]);

  // Map trades to clusters
  const tradesByCluster = useMemo(() => {
    const map: Record<string, Trade[]> = {};
    clusters.forEach((cluster) => {
      const sectorKeys = Array.from(new Set(cluster.agents.flatMap((a) => a.primarySectors)));
      const clusterTrades: Trade[] = [];
      sectorKeys.forEach((s) => {
        (tradesBySector[s.toLowerCase()] || []).forEach((t) => {
          if (!clusterTrades.find((ct) => ct.trade_id === t.trade_id)) clusterTrades.push(t);
        });
      });
      // Also match by agent_id
      const agentIds = new Set(cluster.agents.map((a) => a.agentId));
      trades.forEach((t) => {
        if (t.agent_id && agentIds.has(t.agent_id) && !clusterTrades.find((ct) => ct.trade_id === t.trade_id)) {
          clusterTrades.push(t);
        }
      });
      map[cluster.id] = clusterTrades.sort((a, b) => (b.proposed_at || "").localeCompare(a.proposed_at || ""));
    });
    return map;
  }, [clusters, trades, tradesBySector]);

  // Trades by agent
  const tradesByAgent = useMemo(() => {
    const map: Record<string, Trade[]> = {};
    trades.forEach((t) => {
      if (t.agent_id) {
        if (!map[t.agent_id]) map[t.agent_id] = [];
        map[t.agent_id].push(t);
      }
    });
    return map;
  }, [trades]);

  // Pipeline summary stats
  const pipelineStats = useMemo(() => {
    const approved = trades.filter((t) => t.vote_result === "approved");
    const rejected = trades.filter((t) => t.vote_result === "rejected");
    const executed = trades.filter((t) => t.outcome === "open" || t.outcome === "profitable" || t.outcome === "losing");
    const profitable = trades.filter((t) => t.outcome === "profitable");
    const strategies = new Set(trades.map((t) => t.strategy_used).filter(Boolean));
    return { approved: approved.length, rejected: rejected.length, executed: executed.length, profitable: profitable.length, total: trades.length, strategies: Array.from(strategies) };
  }, [trades]);

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
  };

  const hoveredClusterData = clusters.find((c) => c.id === hoveredCluster);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[800px] rounded-3xl bg-white overflow-hidden"
      style={{ boxShadow: "0 1px 3px rgba(59,130,246,0.06), 0 0 0 1px rgba(59,130,246,0.04)" }}
      onMouseMove={handleMouseMove}
    >
      <svg width={size.width} height={size.height} viewBox={`0 0 ${size.width} ${size.height}`} className="absolute inset-0">
        <defs>
          <filter id="glow"><feGaussianBlur stdDeviation="4" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="4" floodOpacity="0.08" /></filter>
          <radialGradient id="centerGlow"><stop offset="0%" stopColor="#3B82F6" stopOpacity="0.08" /><stop offset="100%" stopColor="#3B82F6" stopOpacity="0" /></radialGradient>
        </defs>

        {/* Background orbital rings */}
        <circle cx={cx} cy={cy} r={outerRadius + 90} fill="none" stroke="#E0E7FF" strokeWidth="1" strokeDasharray="4 8" opacity="0.5" />
        <circle cx={cx} cy={cy} r={outerRadius * 0.5} fill="none" stroke="#E0E7FF" strokeWidth="1" strokeDasharray="4 8" opacity="0.3" />

        {/* Connection paths */}
        {clusters.map((cluster) => {
          const pathId = `path-${cluster.id}`;
          const d = curvedPath(cluster.x, cluster.y, cx, cy);
          const active = hoveredCluster === cluster.id || hoveredAgent?.cluster.id === cluster.id;
          return (
            <g key={`conn-${cluster.id}`}>
              <path
                id={pathId} d={d} fill="none"
                stroke={active ? cluster.color : "#DBEAFE"}
                strokeWidth={active ? 2.5 : 1.5}
                opacity={hoveredCluster && !active ? 0.15 : 0.6}
                style={{ transition: "all 0.3s ease" }}
              />
              {[0, 1, 2].map((idx) => (
                <circle key={idx} r="3" fill={cluster.color} opacity="0">
                  <animateMotion dur={`${2.5 + idx * 0.5}s`} repeatCount="indefinite" begin={`${idx * 0.8}s`}>
                    <mpath href={`#${pathId}`} />
                  </animateMotion>
                  <animate attributeName="opacity" values="0;0.6;0.6;0" dur={`${2.5 + idx * 0.5}s`} repeatCount="indefinite" begin={`${idx * 0.8}s`} />
                </circle>
              ))}
            </g>
          );
        })}

        {/* Inter-cluster lines */}
        {clusters.map((c1, i) =>
          clusters.slice(i + 1).map((c2) => {
            if (Math.hypot(c1.x - c2.x, c1.y - c2.y) > outerRadius * 1.2) return null;
            return <line key={`l-${c1.id}-${c2.id}`} x1={c1.x} y1={c1.y} x2={c2.x} y2={c2.y} stroke="#E0E7FF" strokeWidth="0.5" opacity="0.4" />;
          })
        )}

        {/* Individual agent dots */}
        {agentPositions.map((ap) => {
          const isThisHovered = hoveredAgent?.agent.agentId === ap.agent.agentId;
          const clusterActive = hoveredCluster === ap.cluster.id;
          const statusColor =
            ap.agent.status === "healthy" ? "#22C55E" :
            ap.agent.status === "warning" ? "#F59E0B" : "#EF4444";

          return (
            <g key={ap.agent.agentId}>
              {/* Hover ring */}
              {isThisHovered && (
                <circle cx={ap.x} cy={ap.y} r="14" fill="none" stroke={ap.cluster.color} strokeWidth="1.5" opacity="0.4">
                  <animate attributeName="r" values="12;16;12" dur="1.2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.5;0.1;0.5" dur="1.2s" repeatCount="indefinite" />
                </circle>
              )}
              {/* Dot */}
              <circle
                cx={ap.x} cy={ap.y}
                r={isThisHovered ? 8 : clusterActive ? 6 : 4.5}
                fill={isThisHovered ? ap.cluster.color : clusterActive ? ap.cluster.color : `${ap.cluster.color}50`}
                stroke={isThisHovered ? "#fff" : clusterActive ? "#fff" : "none"}
                strokeWidth={isThisHovered ? 2.5 : clusterActive ? 1.5 : 0}
                style={{ transition: "all 0.2s ease", cursor: "pointer" }}
                onMouseEnter={() => { setHoveredAgent(ap); setHoveredCluster(null); }}
                onMouseLeave={() => setHoveredAgent(null)}
                onClick={() => onSelectAgent?.(ap.agent.agentId)}
              />
              {/* Status indicator on hovered dot */}
              {isThisHovered && (
                <circle cx={ap.x + 6} cy={ap.y - 6} r="3.5" fill={statusColor} stroke="#fff" strokeWidth="1.5" />
              )}
            </g>
          );
        })}

        {/* Cluster hub nodes */}
        {clusters.map((cluster) => {
          const isHovered = hoveredCluster === cluster.id;
          const agentInClusterHovered = hoveredAgent?.cluster.id === cluster.id;
          const active = isHovered || agentInClusterHovered;
          const hubSize = active ? 34 : 28;
          return (
            <g
              key={`hub-${cluster.id}`}
              transform={`translate(${cluster.x}, ${cluster.y})`}
              onMouseEnter={() => { setHoveredCluster(cluster.id); setHoveredAgent(null); }}
              onMouseLeave={() => setHoveredCluster(null)}
              style={{ cursor: "pointer" }}
            >
              {active && (
                <circle r={hubSize + 8} fill="none" stroke={cluster.color} strokeWidth="2" opacity="0.2">
                  <animate attributeName="r" values={`${hubSize + 6};${hubSize + 14};${hubSize + 6}`} dur="1.5s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.3;0.05;0.3" dur="1.5s" repeatCount="indefinite" />
                </circle>
              )}
              <ShapeNode shape={cluster.shape} size={hubSize} fill={`${cluster.color}18`} stroke={cluster.color} strokeWidth={active ? 2.5 : 2} />
              <g transform={`translate(${hubSize * 0.65}, ${-hubSize * 0.65})`}>
                <circle r="12" fill={cluster.color} />
                <text textAnchor="middle" dominantBaseline="central" fill="white" fontSize="10" fontWeight="800">{cluster.agents.length}</text>
              </g>
              <text y={hubSize + 18} textAnchor="middle" fill="#475569" fontSize="11" fontWeight="600">{cluster.name}</text>
            </g>
          );
        })}

        {/* Center consensus node */}
        <g
          transform={`translate(${cx}, ${cy})`}
          onMouseEnter={() => setHoveredCenter(true)}
          onMouseLeave={() => setHoveredCenter(false)}
          style={{ cursor: "pointer" }}
        >
          <circle r={outerRadius * 0.25} fill="url(#centerGlow)" />
          <circle r="48" fill="none" stroke="#3B82F6" strokeWidth="1.5" opacity="0.15">
            <animate attributeName="r" values="48;65;48" dur="3s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.2;0;0.2" dur="3s" repeatCount="indefinite" />
          </circle>
          <circle r="48" fill="none" stroke="#3B82F6" strokeWidth="1" opacity="0.1">
            <animate attributeName="r" values="48;72;48" dur="3s" repeatCount="indefinite" begin="1s" />
            <animate attributeName="opacity" values="0.15;0;0.15" dur="3s" repeatCount="indefinite" begin="1s" />
          </circle>
          <circle r={hoveredCenter ? 50 : 44} fill={hoveredCenter ? "#3B82F608" : "#F0F4FF"} stroke="#3B82F6" strokeWidth={hoveredCenter ? 3 : 2} style={{ transition: "all 0.3s ease" }} filter="url(#shadow)" />
          <circle r="28" fill="#3B82F6" opacity="0.08" />
          <text textAnchor="middle" dy="-6" fill="#3B82F6" fontSize="11" fontWeight="800" letterSpacing="1">CONSENSUS</text>
          <text textAnchor="middle" dy="10" fill="#3B82F6" fontSize="20" fontWeight="900">{agents.length}</text>
          <text textAnchor="middle" dy="24" fill="#94A3B8" fontSize="9" fontWeight="500">AGENTS VOTING</text>
        </g>
      </svg>

      {/* ====== TOOLTIP: Individual Agent ====== */}
      {hoveredAgent && (
        <div
          className="canvas-tooltip absolute z-30 pointer-events-none"
          style={{
            left: Math.min(mousePos.x + 16, size.width - 300),
            top: Math.max(mousePos.y - 120, 10),
          }}
        >
          <div className="bg-white rounded-2xl shadow-xl border border-[var(--border)] p-4 w-[270px]">
            {/* Header */}
            <div className="flex items-center gap-2.5 mb-3">
              <div className="h-8 w-8 rounded-xl flex items-center justify-center" style={{ background: `${hoveredAgent.cluster.color}18` }}>
                <div className="h-3.5 w-3.5 rounded-full" style={{ background: hoveredAgent.cluster.color }} />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-black text-[var(--text-primary)] truncate">{hoveredAgent.agent.displayName}</h4>
                <p className="text-[10px] text-[var(--text-muted)]">{hoveredAgent.cluster.name}</p>
              </div>
              <span className={`h-2 w-2 rounded-full ${
                hoveredAgent.agent.status === "healthy" ? "bg-[var(--positive)]" :
                hoveredAgent.agent.status === "warning" ? "bg-[var(--warning)]" : "bg-[var(--negative)]"
              }`} />
            </div>

            {/* Strategy + Sectors */}
            <div className="flex flex-wrap gap-1.5 mb-3">
              <span className="rounded-full px-2 py-0.5 text-[9px] font-bold capitalize" style={{ background: `${hoveredAgent.cluster.color}15`, color: hoveredAgent.cluster.color }}>
                {hoveredAgent.agent.primaryStrategy.replace(/_/g, " ")}
              </span>
              {hoveredAgent.agent.secondaryStrategy && (
                <span className="rounded-full bg-[var(--bg-hover)] px-2 py-0.5 text-[9px] font-medium text-[var(--text-muted)] capitalize">
                  {hoveredAgent.agent.secondaryStrategy.replace(/_/g, " ")}
                </span>
              )}
              {hoveredAgent.agent.primarySectors.slice(0, 2).map((s) => (
                <span key={s} className="rounded-full bg-[var(--accent-light)] px-2 py-0.5 text-[9px] font-medium text-[var(--accent)] capitalize">{s}</span>
              ))}
            </div>

            {/* Score bars */}
            <div className="space-y-1.5 mb-3">
              {[
                { label: "Composite Score", value: hoveredAgent.agent.compositeWeight },
                { label: "Risk-Adj Return", value: hoveredAgent.agent.riskAdjustedReturn || 0.5 },
                { label: "Calibration", value: hoveredAgent.agent.calibrationQuality || 0.5 },
                { label: "Reasoning", value: hoveredAgent.agent.reasoningQuality || 0.5 },
              ].map((s) => {
                const pct = Math.round(s.value * 100);
                const barColor = pct >= 70 ? "#22C55E" : pct >= 50 ? "#3B82F6" : "#EF4444";
                return (
                  <div key={s.label} className="flex items-center gap-2">
                    <span className="text-[9px] text-[var(--text-muted)] w-[85px] shrink-0">{s.label}</span>
                    <div className="flex-1 h-[3px] rounded-full bg-[#E0E7FF] overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${pct}%`, background: barColor, transition: "width 0.4s" }} />
                    </div>
                    <span className="text-[9px] font-bold tabular-nums w-[22px] text-right" style={{ color: barColor }}>{pct}</span>
                  </div>
                );
              })}
            </div>

            {/* Agent's recent trades */}
            {(() => {
              const agentTrades = tradesByAgent[hoveredAgent.agent.agentId] || [];
              if (agentTrades.length > 0) {
                return (
                  <div className="mb-2">
                    <p className="text-[9px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-1">Recent Picks</p>
                    <div className="space-y-1 max-h-[80px] overflow-y-auto">
                      {agentTrades.slice(0, 3).map((t, i) => (
                        <div key={t.trade_id || i} className={`flex items-center justify-between px-2 py-1 rounded-md ${t.vote_result === "approved" ? "bg-[var(--positive-light)]" : "bg-[var(--negative-light)]"}`}>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[9px] font-bold">{t.direction?.toUpperCase()} {t.symbol}</span>
                            <span className="text-[8px] px-1 py-0.5 rounded bg-[var(--bg-hover)] text-[var(--text-muted)]">{t.strategy_used?.replace(/_/g, " ") || "swing"}</span>
                          </div>
                          <span className={`text-[9px] font-bold ${t.vote_result === "approved" ? "text-[var(--positive)]" : "text-[var(--negative)]"}`}>
                            {t.vote_result === "approved" ? "✓" : "✗"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              }
              return null;
            })()}

            {/* Quick stats */}
            <div className="flex items-center justify-between pt-2 border-t border-[var(--border)]">
              <span className="text-[9px] text-[var(--text-muted)]">
                <span className="font-bold capitalize">{hoveredAgent.agent.holdingPeriod || "swing"}</span> · {hoveredAgent.agent.lookbackDays || 0}d lookback
              </span>
              <span className="text-[9px] font-bold text-[var(--accent)]">Risk: {Math.round((hoveredAgent.agent.riskAppetite || 0.5) * 100)}%</span>
            </div>

            {/* Click hint */}
            <p className="text-[8px] text-[var(--text-muted)] text-center mt-2 opacity-60">Click for full profile</p>
          </div>
        </div>
      )}

      {/* ====== TOOLTIP: Cluster ====== */}
      {hoveredClusterData && !hoveredAgent && (() => {
        const clTrades = tradesByCluster[hoveredClusterData.id] || [];
        const accepted = clTrades.filter((t) => t.vote_result === "approved");
        const rejected = clTrades.filter((t) => t.vote_result === "rejected");
        return (
          <div
            className="canvas-tooltip absolute z-20 pointer-events-none"
            style={{
              left: Math.min(mousePos.x + 16, size.width - 320),
              top: Math.min(mousePos.y - 10, size.height - 400),
            }}
          >
            <div className="bg-white rounded-2xl shadow-xl border border-[var(--border)] p-5 w-[310px]">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-3 w-3 rounded-full" style={{ background: hoveredClusterData.color }} />
                <h4 className="text-sm font-black text-[var(--text-primary)]">{hoveredClusterData.name}</h4>
                <span className="ml-auto text-xs font-bold text-[var(--text-muted)]">{hoveredClusterData.agents.length} agents</span>
              </div>

              {/* Accepted trades */}
              {accepted.length > 0 && (
                <div className="mb-3">
                  <p className="text-[10px] font-bold text-[var(--positive)] uppercase tracking-wider mb-1.5">Accepted ({accepted.length})</p>
                  <div className="space-y-1.5 max-h-[120px] overflow-y-auto">
                    {accepted.slice(0, 5).map((t, i) => (
                      <div key={t.trade_id || i} className="rounded-lg bg-[var(--positive-light)] px-3 py-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-bold text-[var(--text-primary)]">{t.direction?.toUpperCase()} {t.symbol}</span>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-[var(--positive)] text-white">{t.strategy_used?.replace(/_/g, " ") || "swing"}</span>
                            <span className="text-[10px] font-bold text-[var(--positive)]">{Math.round((t.approval_pct || 0) * 100)}%</span>
                          </div>
                        </div>
                        {t.reasoning && <p className="text-[9px] text-[var(--text-muted)] mt-1 line-clamp-2">{t.reasoning}</p>}
                      </div>
                    ))}
                    {accepted.length > 5 && <p className="text-[9px] text-[var(--text-muted)]">+{accepted.length - 5} more</p>}
                  </div>
                </div>
              )}

              {/* Rejected trades */}
              {rejected.length > 0 && (
                <div className="mb-3">
                  <p className="text-[10px] font-bold text-[var(--negative)] uppercase tracking-wider mb-1.5">Rejected ({rejected.length})</p>
                  <div className="space-y-1.5 max-h-[100px] overflow-y-auto">
                    {rejected.slice(0, 4).map((t, i) => (
                      <div key={t.trade_id || i} className="rounded-lg bg-[var(--negative-light)] px-3 py-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-medium text-[var(--text-secondary)]">{t.direction?.toUpperCase()} {t.symbol}</span>
                          <span className="text-[10px] text-[var(--negative)]">{Math.round((t.approval_pct || 0) * 100)}% approval</span>
                        </div>
                        {t.reasoning && <p className="text-[9px] text-[var(--text-muted)] mt-1 line-clamp-1">{t.reasoning}</p>}
                      </div>
                    ))}
                    {rejected.length > 4 && <p className="text-[9px] text-[var(--text-muted)]">+{rejected.length - 4} more</p>}
                  </div>
                </div>
              )}

              {/* No trades yet */}
              {clTrades.length === 0 && (
                <div className="space-y-1.5 max-h-[180px] overflow-y-auto mb-3">
                  {hoveredClusterData.agents.slice(0, 8).map((a) => (
                    <div key={a.agentId} className="flex items-center justify-between py-1">
                      <span className="text-[11px] font-medium text-[var(--text-secondary)] truncate max-w-[140px]">{a.displayName}</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-hover)] text-[var(--text-muted)] capitalize">{a.primaryStrategy.replace(/_/g, " ")}</span>
                        <span className={`h-1.5 w-1.5 rounded-full ${
                          a.status === "healthy" ? "bg-[var(--positive)]" : a.status === "warning" ? "bg-[var(--warning)]" : "bg-[var(--negative)]"
                        }`} />
                      </div>
                    </div>
                  ))}
                  {hoveredClusterData.agents.length > 8 && (
                    <p className="text-[10px] text-[var(--text-muted)]">+{hoveredClusterData.agents.length - 8} more agents</p>
                  )}
                  <p className="text-[9px] text-[var(--text-muted)] italic pt-1">No trades proposed yet — waiting for market scan</p>
                </div>
              )}

              <div className="pt-2 border-t border-[var(--border)] flex gap-2 flex-wrap">
                {Array.from(new Set(hoveredClusterData.agents.flatMap((a) => a.primarySectors))).slice(0, 4).map((sector) => (
                  <span key={sector} className="rounded-full px-2 py-0.5 text-[9px] font-semibold capitalize" style={{ background: `${hoveredClusterData.color}15`, color: hoveredClusterData.color }}>{sector}</span>
                ))}
              </div>
            </div>
          </div>
        );
      })()}

      {/* ====== TOOLTIP: Center ====== */}
      {hoveredCenter && (
        <div className="canvas-tooltip absolute z-20 pointer-events-none" style={{ left: cx + 60, top: cy - 120 }}>
          <div className="bg-white rounded-2xl shadow-xl border border-[var(--border)] p-5 w-[300px]">
            <h4 className="text-sm font-black text-[var(--accent)] mb-2">Global Consensus</h4>
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed mb-3">
              All {agents.length} agents vote on stocks that pass their group review. Weighted approval required.
            </p>
            <div className="grid grid-cols-4 gap-2 text-center mb-3">
              <div className="rounded-xl bg-[var(--accent-light)] px-2 py-2">
                <p className="text-base font-black text-[var(--accent)]">{clusters.length}</p>
                <p className="text-[8px] font-semibold text-[var(--text-muted)]">GROUPS</p>
              </div>
              <div className="rounded-xl bg-[var(--accent-light)] px-2 py-2">
                <p className="text-base font-black text-[var(--accent)]">{pipelineStats.total}</p>
                <p className="text-[8px] font-semibold text-[var(--text-muted)]">PROPOSED</p>
              </div>
              <div className="rounded-xl bg-[var(--positive-light)] px-2 py-2">
                <p className="text-base font-black text-[var(--positive)]">{pipelineStats.approved}</p>
                <p className="text-[8px] font-semibold text-[var(--text-muted)]">APPROVED</p>
              </div>
              <div className="rounded-xl bg-[var(--negative-light)] px-2 py-2">
                <p className="text-base font-black text-[var(--negative)]">{pipelineStats.rejected}</p>
                <p className="text-[8px] font-semibold text-[var(--text-muted)]">REJECTED</p>
              </div>
            </div>

            {/* Recent approved trades */}
            {pipelineStats.approved > 0 && (
              <div className="mb-3">
                <p className="text-[10px] font-bold text-[var(--positive)] uppercase tracking-wider mb-1.5">Latest Approved</p>
                <div className="space-y-1 max-h-[120px] overflow-y-auto">
                  {trades.filter((t) => t.vote_result === "approved").slice(0, 5).map((t, i) => (
                    <div key={t.trade_id || i} className="flex items-center justify-between py-1 px-2 rounded-lg bg-[var(--bg-hover)]">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-black text-[var(--text-primary)]">{t.direction?.toUpperCase()} {t.symbol}</span>
                        <span className="text-[8px] font-bold px-1.5 py-0.5 rounded bg-[var(--accent)] text-white">{t.strategy_used?.replace(/_/g, " ") || "swing"}</span>
                      </div>
                      <span className="text-[10px] font-bold text-[var(--positive)]">{Math.round((t.confidence || 0) * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Trade types breakdown */}
            {pipelineStats.strategies.length > 0 && (
              <div className="pt-2 border-t border-[var(--border)]">
                <p className="text-[9px] font-bold text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Trade Types</p>
                <div className="flex gap-1.5 flex-wrap">
                  {pipelineStats.strategies.map((s) => (
                    <span key={s} className="rounded-full px-2 py-0.5 text-[9px] font-semibold capitalize bg-[var(--accent-light)] text-[var(--accent)]">{(s || "").replace(/_/g, " ")}</span>
                  ))}
                </div>
              </div>
            )}

            {pipelineStats.total === 0 && (
              <p className="text-[9px] text-[var(--text-muted)] italic">No trades proposed yet — currently in overnight scanning mode</p>
            )}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex flex-wrap gap-x-4 gap-y-1.5 max-w-[600px]">
        {clusters.map((c) => (
          <div key={c.id} className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-sm" style={{ background: c.color }} />
            <span className="text-[10px] text-[var(--text-muted)] font-medium">{c.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

import type {
  RiskSummary,
  Alert,
  ControlState,
  RegimeInfo,
  AgentHealthReport,
} from "@/types";

export const mockRiskSummary: RiskSummary = {
  currentDrawdown: -0.038,
  maxDrawdownLimit: 0.15,
  dailyLoss: -0.008,
  dailyLossLimit: 0.03,
  grossExposure: 0.82,
  grossExposureLimit: 1.0,
  netExposure: 0.64,
  leverage: 1.0,
  concentrationRisk: 0.18,
  correlationRisk: 0.35,
  liquidityRisk: 0.12,
  regimeMismatch: false,
  killSwitchActive: false,
  killSwitchReason: "",
};

export const mockAlerts: Alert[] = [
  {
    id: "a1",
    severity: "critical",
    category: "drawdown",
    title: "Drawdown Warning",
    message: "Portfolio drawdown at -3.8%, approaching 50% of kill-switch level (-7.5%)",
    timestamp: "2026-03-18T14:23:00Z",
    acknowledged: false,
    relatedSymbol: "",
    relatedAgent: "",
  },
  {
    id: "a2",
    severity: "high",
    category: "position",
    title: "Position Near Stop",
    message: "TSLA short position within 1.2% of stop-loss level",
    timestamp: "2026-03-18T14:18:00Z",
    acknowledged: false,
    relatedSymbol: "TSLA",
    relatedAgent: "",
  },
  {
    id: "a3",
    severity: "high",
    category: "concentration",
    title: "Sector Concentration",
    message: "Technology sector exposure at 28%, approaching 30% limit",
    timestamp: "2026-03-18T13:45:00Z",
    acknowledged: false,
    relatedSymbol: "",
    relatedAgent: "",
  },
  {
    id: "a4",
    severity: "medium",
    category: "correlation",
    title: "Correlated Positions",
    message: "NVDA and AMD positions have 0.82 rolling correlation",
    timestamp: "2026-03-18T13:30:00Z",
    acknowledged: true,
    relatedSymbol: "NVDA",
    relatedAgent: "",
  },
  {
    id: "a5",
    severity: "medium",
    category: "agent",
    title: "Agent Drift",
    message: "contrarian_broad scoring below 0.3 for 15 consecutive days",
    timestamp: "2026-03-18T12:00:00Z",
    acknowledged: false,
    relatedSymbol: "",
    relatedAgent: "contrarian_broad",
  },
  {
    id: "a6",
    severity: "low",
    category: "execution",
    title: "Slippage Above Average",
    message: "Last 5 fills averaged 2.3 bps slippage vs 1.0 bps target",
    timestamp: "2026-03-18T11:30:00Z",
    acknowledged: true,
    relatedSymbol: "",
    relatedAgent: "",
  },
  {
    id: "a7",
    severity: "medium",
    category: "regime",
    title: "Regime Shift Detected",
    message: "Market regime shifting from low_vol_trending_up to low_vol_choppy",
    timestamp: "2026-03-18T10:00:00Z",
    acknowledged: false,
    relatedSymbol: "",
    relatedAgent: "",
  },
  {
    id: "a8",
    severity: "low",
    category: "review",
    title: "Overdue Exit Review",
    message: "XOM position has exceeded planned holding period by 3 days",
    timestamp: "2026-03-18T09:00:00Z",
    acknowledged: false,
    relatedSymbol: "XOM",
    relatedAgent: "",
  },
];

export const mockControlState: ControlState = {
  tradingPaused: false,
  entriesPaused: false,
  exitsOnly: false,
  riskOffMode: false,
  manualApprovalMode: false,
  manualOnlyMode: false,
  maxPositionSizePct: 0.05,
  maxExposurePct: 1.0,
  disabledGroups: [],
  disabledAgents: [],
};

export const mockRegimeInfo: RegimeInfo = {
  current: "low_vol_trending_up",
  approvalThreshold: 0.5,
  minVoters: 4,
  minConfidence: 0.55,
  positionSizeMultiplier: 1.2,
  preferredVehicles: ["shares_spot", "call_option", "leaps"],
  maxNewPositions: 7,
  requireStopLoss: false,
};

export const mockAgentHealth: AgentHealthReport[] = [
  {
    agentId: "contrarian_broad",
    status: "underperforming",
    reason: "Below average — scoring 0.28 composite",
    compositeWeight: 0.28,
    regimeMismatch: true,
    trueDecay: false,
    isRedundant: false,
    redundantWith: "",
    nOutcomes: 67,
    daysSinceLastCorrect: 15,
    recommendation: "Regime mismatch — bench until regime shifts",
  },
  {
    agentId: "random_walk",
    status: "warning",
    reason: "High false positive rate",
    compositeWeight: 0.45,
    regimeMismatch: false,
    trueDecay: false,
    isRedundant: false,
    redundantWith: "",
    nOutcomes: 52,
    daysSinceLastCorrect: 8,
    recommendation: "Monitor for 20 more outcomes",
  },
  {
    agentId: "insurance_value",
    status: "underperforming",
    reason: "Redundant with financial_value",
    compositeWeight: 0.32,
    regimeMismatch: false,
    trueDecay: false,
    isRedundant: true,
    redundantWith: "financial_value",
    nOutcomes: 55,
    daysSinceLastCorrect: 12,
    recommendation: "Replace with gap-filler agent",
  },
];

export interface PositionRisk {
  symbol: string;
  direction: string;
  sizePct: number;
  pnlPct: number;
  distanceToStop: number;
  riskScore: number;
  sector: string;
  flags: string[];
}

export const mockPositionRisks: PositionRisk[] = [
  { symbol: "TSLA", direction: "short", sizePct: 0.042, pnlPct: -0.023, distanceToStop: 0.012, riskScore: 0.85, sector: "Consumer Disc", flags: ["Near stop", "High vol"] },
  { symbol: "NVDA", direction: "long", sizePct: 0.048, pnlPct: 0.058, distanceToStop: 0.082, riskScore: 0.45, sector: "Technology", flags: ["Correlated w/ AMD"] },
  { symbol: "AMD", direction: "long", sizePct: 0.035, pnlPct: 0.032, distanceToStop: 0.065, riskScore: 0.42, sector: "Technology", flags: ["Correlated w/ NVDA"] },
  { symbol: "XOM", direction: "long", sizePct: 0.038, pnlPct: -0.015, distanceToStop: 0.035, riskScore: 0.55, sector: "Energy", flags: ["Overdue review"] },
  { symbol: "JPM", direction: "long", sizePct: 0.045, pnlPct: 0.018, distanceToStop: 0.058, riskScore: 0.30, sector: "Financials", flags: [] },
  { symbol: "AAPL", direction: "long", sizePct: 0.050, pnlPct: 0.042, distanceToStop: 0.095, riskScore: 0.20, sector: "Technology", flags: [] },
];

export interface ExecutionHealth {
  brokerConnected: boolean;
  brokerName: string;
  brokerLatencyMs: number;
  lastHeartbeat: string;
  orderFillRate: number;
  fillRate: number;
  avgSlippageBps: number;
  averageSlippage: number;
  failedOrders: number;
  rejectedOrders: number;
  staleOrders: number;
  lastOrderTime: string;
  dailyOrderCount: number;
  dailyOrders: number;
  filledToday: number;
  totalOrdersToday: number;
  pendingToday: number;
}

export const mockExecutionHealth: ExecutionHealth = {
  brokerConnected: true,
  brokerName: "Interactive Brokers",
  brokerLatencyMs: 12,
  lastHeartbeat: "2026-03-18T14:22:45Z",
  orderFillRate: 0.97,
  fillRate: 0.97,
  avgSlippageBps: 1.2,
  averageSlippage: 1.2,
  failedOrders: 1,
  rejectedOrders: 0,
  staleOrders: 0,
  lastOrderTime: "2026-03-18T14:15:32Z",
  dailyOrderCount: 8,
  dailyOrders: 8,
  filledToday: 7,
  totalOrdersToday: 8,
  pendingToday: 1,
};

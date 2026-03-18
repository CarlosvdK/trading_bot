// ============================================================
// Core TypeScript interfaces matching Python data structures
// ============================================================

// --- Enums ---

export type InvestmentType =
  | "intraday" | "swing_trade" | "position_trade" | "long_term_investment"
  | "shares_spot" | "call_option" | "put_option" | "call_spread" | "put_spread"
  | "covered_call" | "protective_put" | "leaps" | "pairs_trade"
  | "volatility_trade" | "no_trade";

export type ThesisType =
  | "technical" | "event_driven" | "macro" | "valuation" | "structural"
  | "momentum" | "mean_reversion" | "sentiment" | "volatility";

export type Direction = "long" | "short";

export type HoldingCategory = "scalp" | "swing" | "position" | "macro";

export type PositionStatus = "healthy" | "warning" | "near_stop" | "near_target" | "exit_pending" | "stopped_out";

export type AgentStatus = "healthy" | "warning" | "underperforming" | "replace";

export type AlertSeverity = "critical" | "high" | "medium" | "low" | "info";

export type SpecialistRole = "supportive" | "skeptical" | "risk";

export type VoteOutcome = "approve" | "reject" | "modify";

export type Strategy =
  | "momentum" | "mean_reversion" | "value" | "growth"
  | "event_driven" | "volatility" | "sentiment" | "breakout";

// --- Portfolio & Positions ---

export interface PortfolioSummary {
  totalValue: number;
  cashBalance: number;
  unrealizedPnl: number;
  realizedPnl: number;
  dailyPnl: number;
  dailyPnlPct: number;
  totalExposure: number;
  netExposure: number;
  grossExposure: number;
  openPositions: number;
  pendingOrders: number;
  capitalDeployed: number;
  capitalDeployedPct: number;
  peakNav: number;
  currentDrawdown: number;
  dayStartNav: number;
}

export interface Position {
  id: string;
  symbol: string;
  assetType: string;
  direction: Direction;
  quantity: number;
  avgEntryPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  unrealizedPnlPct: number;
  realizedPnl: number;
  strategy: string;
  investmentType: InvestmentType;
  holdingCategory: HoldingCategory;
  openTimestamp: string;
  plannedExitCondition: string;
  plannedExitDate: string;
  stopLevel: number;
  targetLevel: number;
  confidence: number;
  status: PositionStatus;
  originAgentGroup: string;
  votingRound: string;
  sector: string;
  // Expanded detail
  thesis: string;
  keyReasons: string[];
  dissentNotes: string;
  executionNotes: string;
  vehicleRationale: string;
  supportingAgents: string[];
  dissentingAgents: string[];
  specialistResult: SubgroupReviewResult | null;
  globalVoteResult: number;
  riskFactors: string[];
  lastReviewTime: string;
  nextReviewTime: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: string;
  type: string;
  quantity: number;
  limitPrice: number | null;
  stopPrice: number | null;
  status: string;
  timestamp: string;
  reason: string;
}

// --- Allocation / Exposure ---

export interface AllocationBreakdown {
  label: string;
  value: number;
  pct: number;
  color: string;
}

// --- Decision Pipeline ---

export interface Proposal {
  proposalId: string;
  agentId: string;
  timestamp: string;
  symbol: string;
  sector: string;
  subIndustry: string;
  direction: Direction;
  investmentType: InvestmentType;
  thesisType: ThesisType;
  thesis: string;
  strategyUsed: Strategy;
  catalyst: string;
  hasDefinedCatalyst: boolean;
  confidence: number;
  expectedEdgeBps: number;
  rawScore: number;
  timeHorizonDays: number;
  holdingPeriodCategory: HoldingCategory;
  suggestedPositionPct: number;
  expectedAnnualVol: number;
  impliedVol: number;
  realizedVol: number;
  avgDailyVolume: number;
  slippageSensitivity: number;
  risks: ProposalRisks;
  entryLogic: string;
  exitLogic: string;
  stopLossPct: number;
  takeProfitPct: number;
  vehicleCandidates: VehicleCandidate[];
  selectedVehicle: VehicleCandidate | null;
  vehicleRationale: string;
  // Review stages
  peerApproved: boolean;
  peerApprovalPct: number;
  specialistReviewPassed: boolean;
  specialistConfidence: number;
  specialistObjections: string[];
  specialistModifications: string[];
  globalVotePassed: boolean;
  globalApprovalPct: number;
  globalWeightedConfidence: number;
  dissentSummary: string;
  portfolioApproved: boolean;
  finalPositionPct: number;
  capitalAllocated: number;
  // Computed
  pipelineStage: "proposal" | "specialist" | "global_vote" | "portfolio" | "approved" | "rejected";
}

export interface ProposalRisks {
  keyRisks: string[];
  invalidationCriteria: string[];
  maxLossPct: number;
  correlationRisk: string;
  executionRisk: string;
  tailRisk: string;
  crowdedTradeScore: number;
  macroContradiction: string;
}

export interface VehicleCandidate {
  vehicleType: InvestmentType;
  description: string;
  expectedReturn: number;
  expectedRisk: number;
  payoffAsymmetry: number;
  timingSensitivity: number;
  volatilitySensitivity: number;
  liquidityQuality: number;
  implementationSimplicity: number;
  robustnessToError: number;
  thetaDecayCost: number;
  transactionCostBps: number;
  compositeScore: number;
}

export interface SubgroupReviewResult {
  passed: boolean;
  confidenceScore: number;
  approvalCount: number;
  rejectCount: number;
  modifyCount: number;
  totalReviewers: number;
  mainSupportingReasons: string[];
  mainObjections: string[];
  recommendedModifications: string[];
  preferredVehicle: string;
  preferredHorizonDays: number;
  verdicts: SpecialistVerdict[];
}

export interface SpecialistVerdict {
  agentId: string;
  vote: VoteOutcome;
  confidence: number;
  supportingReasons: string[];
  objections: string[];
  recommendedModifications: string[];
  preferredVehicle: string;
  preferredHorizonDays: number;
  role: SpecialistRole;
}

export interface DecisionOutput {
  decisionId: string;
  timestamp: string;
  asset: string;
  sector: string;
  direction: Direction;
  investmentType: string;
  investmentTypeRationale: string;
  holdingHorizonDays: number;
  holdingCategory: HoldingCategory;
  entryLogic: string;
  exitLogic: string;
  stopLogic: string;
  targetLogic: string;
  invalidationLogic: string;
  confidence: number;
  weightedVoteResult: number;
  specialistConfidence: number;
  dissentSummary: string;
  mainObjections: string[];
  positionSizePct: number;
  capitalAllocated: number;
  selectedVehicle: string;
  vehicleRationale: string;
  alternativeVehicles: { type: string; description: string; score: number }[];
  maxLossPct: number;
  expectedAnnualVol: number;
  keyRisks: string[];
  currentRegime: string;
  regimeFit: string;
  thesis: string;
  catalyst: string;
}

// --- Agents ---

export interface AgentDNA {
  agentId: string;
  displayName: string;
  primarySectors: string[];
  secondarySectors: string[];
  primaryStrategy: Strategy;
  secondaryStrategy: Strategy | null;
  riskAppetite: number;
  contrarianFactor: number;
  convictionStyle: number;
  regimeSensitivity: number;
  lookbackDays: number;
  holdingPeriod: HoldingCategory;
  minConfidence: number;
  maxPicksPerScan: number;
  peerGroup: string;
}

export interface DimensionalScore {
  agentId: string;
  riskAdjustedReturn: number;
  calibrationQuality: number;
  reasoningQuality: number;
  uniqueness: number;
  regimeEffectiveness: number;
  drawdownBehavior: number;
  stability: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  peerVoteAccuracy: number;
  compositeWeight: number;
  nOutcomes: number;
}

export interface AgentHealthReport {
  agentId: string;
  status: AgentStatus;
  reason: string;
  compositeWeight: number;
  regimeMismatch: boolean;
  trueDecay: boolean;
  isRedundant: boolean;
  redundantWith: string;
  nOutcomes: number;
  daysSinceLastCorrect: number | null;
  recommendation: string;
}

export interface AgentProfile extends AgentDNA {
  score: DimensionalScore;
  health: AgentHealthReport;
  recentProposals: Proposal[];
  approvedCount: number;
  rejectedCount: number;
  profitableCount: number;
  losingCount: number;
}

export interface PeerGroup {
  name: string;
  agentCount: number;
  agents: string[];
  specialty: string;
  avgScore: number;
  scannedCount: number;
  surfacedCount: number;
  passedCount: number;
  avgConviction: number;
  recentHitRate: number;
}

// --- Pipeline Funnel ---

export interface PipelineFunnel {
  universeScanned: number;
  shortlisted: number;
  highConviction: number;
  specialistReviewed: number;
  approved: number;
  rejected: number;
  noTrade: number;
  downsized: number;
}

// --- Risk / Control Center ---

export interface RiskSummary {
  currentDrawdown: number;
  maxDrawdownLimit: number;
  dailyLoss: number;
  dailyLossLimit: number;
  grossExposure: number;
  grossExposureLimit: number;
  netExposure: number;
  leverage: number;
  concentrationRisk: number;
  correlationRisk: number;
  liquidityRisk: number;
  regimeMismatch: boolean;
  killSwitchActive: boolean;
  killSwitchReason: string;
}

export interface Alert {
  id: string;
  severity: AlertSeverity;
  category: string;
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  relatedSymbol: string;
  relatedAgent: string;
}

export interface ControlState {
  tradingPaused: boolean;
  entriesPaused: boolean;
  exitsOnly: boolean;
  riskOffMode: boolean;
  manualApprovalMode: boolean;
  manualOnlyMode: boolean;
  maxPositionSizePct: number;
  maxExposurePct: number;
  disabledGroups: string[];
  disabledAgents: string[];
}

export interface RegimeInfo {
  current: string;
  approvalThreshold: number;
  minVoters: number;
  minConfidence: number;
  positionSizeMultiplier: number;
  preferredVehicles: string[];
  maxNewPositions: number;
  requireStopLoss: boolean;
}

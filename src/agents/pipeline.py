"""Full Multi-Stage Pipeline — orchestrates all 4 stages of trade approval.

Stage A: Proposal generation (agents scan independently)
Stage B: Specialist subgroup review (expert pre-screening)
Stage C: Global weighted vote (full pool with adaptive thresholds)
Stage D: Portfolio/risk layer (exposure, correlation, sizing)
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.agents.agent_dna import AgentDNA
from src.agents.trading_agent import TradingAgent, TradePick
from src.agents.proposal import Proposal, InvestmentType, proposal_from_trade_pick
from src.agents.specialist_review import SpecialistReview, SubgroupReviewResult
from src.agents.voting_engine import VotingEngine
from src.agents.vehicle_engine import VehicleSelectionEngine
from src.agents.regime_adapter import RegimeAdapter
from src.agents.enhanced_scoring import EnhancedScoring
from src.agents.idea_bucket import IdeaBucket
from src.agents.decision_output import DecisionOutput, build_decision_output

logger = logging.getLogger(__name__)


class TradingPipeline:
    """
    Full 4-stage multi-agent trading pipeline.

    Replaces the simpler AgentPool with a rigorous multi-stage process:
    1. Stage A: Each agent scans and produces enhanced Proposals
    2. Stage B: Specialist subgroup reviews each proposal
    3. Stage C: Surviving proposals go to global weighted vote
    4. Stage D: Portfolio/risk layer makes final sizing/approval

    Thresholds are adaptive to regime, not fixed.
    Vehicle selection determines optimal implementation.
    Anti-herd measures prevent consensus-for-its-own-sake.
    """

    def __init__(
        self,
        agents: List[TradingAgent],
        scoring: Optional[EnhancedScoring] = None,
        vehicle_engine: Optional[VehicleSelectionEngine] = None,
        regime_adapter: Optional[RegimeAdapter] = None,
        risk_governor=None,
        max_new_positions_per_day: int = 5,
        max_portfolio_gross_exposure: float = 1.0,
        max_sector_concentration: float = 0.30,
        max_correlation_threshold: float = 0.70,
    ):
        self.agents = agents
        self.scoring = scoring or EnhancedScoring()
        self.vehicle_engine = vehicle_engine or VehicleSelectionEngine()
        self.regime_adapter = regime_adapter or RegimeAdapter()
        self.risk_governor = risk_governor

        # Portfolio constraints
        self.max_new_positions_per_day = max_new_positions_per_day
        self.max_portfolio_gross_exposure = max_portfolio_gross_exposure
        self.max_sector_concentration = max_sector_concentration
        self.max_correlation_threshold = max_correlation_threshold

        # Build lookup maps
        self._agent_map: Dict[str, TradingAgent] = {
            a.dna.agent_id: a for a in agents
        }
        self._dna_map: Dict[str, AgentDNA] = {
            a.dna.agent_id: a.dna for a in agents
        }

        # Build peer groups for Stage A pre-voting
        self._peer_groups: Dict[str, List[str]] = defaultdict(list)
        for a in agents:
            if a.dna.peer_group:
                self._peer_groups[a.dna.peer_group].append(a.dna.agent_id)

        # Specialist review engine
        self._specialist_review = SpecialistReview(self._dna_map)

        # Idea bucket for Stage C
        self._idea_bucket = IdeaBucket()

    def run_daily(
        self,
        universe_data: Dict[str, pd.DataFrame],
        index_df: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        portfolio_state: Optional[dict] = None,
    ) -> List[DecisionOutput]:
        """Execute the full 4-stage pipeline.

        Args:
            universe_data: {symbol: OHLCV DataFrame}.
            index_df: Market index OHLCV.
            current_date: Date to evaluate at.
            portfolio_state: Current portfolio state dict.

        Returns:
            List of DecisionOutput for each approved trade.
        """
        self._idea_bucket.clear()

        # ---- Detect regime and get adaptive thresholds ---- #
        regime = ""
        if index_df is not None:
            regime = self.regime_adapter.detect_regime(index_df, current_date)
        thresholds = self.regime_adapter.get_thresholds(regime)

        logger.info(
            f"Regime: {regime} | Thresholds: approval={thresholds.approval_threshold}, "
            f"min_voters={thresholds.min_voters}"
        )

        # ============================================== #
        #  STAGE A: Proposal Generation                  #
        # ============================================== #
        raw_proposals = self._stage_a_proposals(
            universe_data, index_df, current_date, regime
        )
        logger.info(
            f"Stage A: {len(raw_proposals)} proposals from {len(self.agents)} agents"
        )

        if not raw_proposals:
            return []

        # Adapt thresholds based on signal quality
        avg_conf = (
            np.mean([p.confidence for p in raw_proposals]) if raw_proposals else 0
        )
        thresholds = self.regime_adapter.adjust_for_signal_quality(
            thresholds, avg_conf, len(raw_proposals)
        )

        # ============================================== #
        #  STAGE B: Specialist Subgroup Review           #
        # ============================================== #
        specialist_passed = self._stage_b_specialist_review(
            raw_proposals, thresholds
        )
        logger.info(
            f"Stage B: {len(specialist_passed)}/{len(raw_proposals)} passed specialist review"
        )

        if not specialist_passed:
            return []

        # ============================================== #
        #  STAGE C: Global Weighted Vote                 #
        # ============================================== #
        global_approved = self._stage_c_global_vote(
            specialist_passed, thresholds
        )
        logger.info(
            f"Stage C: {len(global_approved)}/{len(specialist_passed)} passed global vote"
        )

        if not global_approved:
            return []

        # ============================================== #
        #  LLM Reasoning Audit (between C and D)        #
        # ============================================== #
        global_approved = self._llm_audit(global_approved, regime)

        # ============================================== #
        #  Vehicle Selection                             #
        # ============================================== #
        for proposal in global_approved:
            self.vehicle_engine.select_vehicle(proposal, regime, portfolio_state)

        # Filter out no-trade selections
        tradeable = [
            p for p in global_approved
            if p.selected_vehicle is None
            or p.selected_vehicle.vehicle_type != InvestmentType.NO_TRADE
        ]

        # ============================================== #
        #  STAGE D: Portfolio/Risk Layer                 #
        # ============================================== #
        final_approved = self._stage_d_portfolio_risk(
            tradeable, thresholds, portfolio_state, regime
        )
        logger.info(
            f"Stage D: {len(final_approved)}/{len(tradeable)} passed portfolio/risk layer"
        )

        # Build decision outputs
        outputs = [build_decision_output(p, regime) for p in final_approved]

        return outputs

    # ================================================================== #
    #  STAGE A: Proposal Generation                                       #
    # ================================================================== #

    def _stage_a_proposals(
        self,
        universe_data: Dict[str, pd.DataFrame],
        index_df: Optional[pd.DataFrame],
        current_date: Optional[pd.Timestamp],
        regime: str,
    ) -> List[Proposal]:
        """Each agent scans and produces proposals."""
        all_proposals: List[Proposal] = []
        all_raw_picks: Dict[str, List[TradePick]] = {}

        for agent in self.agents:
            try:
                picks = agent.scan(universe_data, index_df, current_date)
                if picks:
                    all_raw_picks[agent.dna.agent_id] = picks
            except Exception as e:
                logger.warning(f"Agent {agent.dna.agent_id} scan failed: {e}")

        # Peer pre-vote
        for agent_id, picks in all_raw_picks.items():
            peer_group = self._dna_map[agent_id].peer_group
            peers = self._peer_groups.get(peer_group, [])
            peer_agents = [p for p in peers if p != agent_id]

            for pick in picks:
                # Peer pre-vote
                if len(peer_agents) >= 2:
                    votes_for = sum(
                        1 for pid in peer_agents
                        if any(
                            pp.symbol == pick.symbol
                            and pp.direction == pick.direction
                            for pp in all_raw_picks.get(pid, [])
                        )
                    )
                    pick.peer_approved = votes_for / len(peer_agents) >= 0.3
                    pick.peer_approval_pct = votes_for / len(peer_agents)

                # Convert to enhanced Proposal
                proposal = proposal_from_trade_pick(pick)
                proposal = self._enrich_proposal(
                    proposal, universe_data, index_df, current_date
                )
                all_proposals.append(proposal)

        return all_proposals

    def _enrich_proposal(
        self,
        proposal: Proposal,
        universe_data: Dict[str, pd.DataFrame],
        index_df: Optional[pd.DataFrame],
        current_date: Optional[pd.Timestamp],
    ) -> Proposal:
        """Enrich a proposal with volatility, liquidity, and risk data."""
        df = universe_data.get(proposal.symbol)
        if df is None or df.empty:
            return proposal

        idx = len(df) - 1
        if current_date is not None and current_date in df.index:
            idx = df.index.get_loc(current_date)

        if idx < 21:
            return proposal

        close = df["close"]
        log_ret = np.log(close / close.shift(1)).dropna()

        # Volatility
        vol_21 = log_ret.iloc[max(0, idx - 21):idx + 1].std() * np.sqrt(252)
        proposal.realized_vol = round(float(vol_21 * 100), 2)
        proposal.expected_annual_vol = proposal.realized_vol

        # Liquidity
        avg_vol = df["volume"].iloc[max(0, idx - 21):idx].mean()
        proposal.avg_daily_volume = float(avg_vol)

        # Slippage sensitivity
        if avg_vol > 0:
            proposal.slippage_sensitivity = round(
                min(1.0, 1_000_000 / avg_vol), 4
            )

        # Edge estimate (confidence * volatility-adjusted)
        proposal.expected_edge_bps = round(
            proposal.confidence * vol_21 * 100 * 50, 1
        )

        # Stop loss from ATR
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - close.shift(1)).abs(),
                (df["low"] - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.iloc[max(0, idx - 14):idx + 1].mean()
        if close.iloc[idx] > 0:
            proposal.stop_loss_pct = round(
                float(atr * 2 / close.iloc[idx]), 4
            )
            proposal.take_profit_pct = round(
                float(atr * 3 / close.iloc[idx]), 4
            )

        # Risks
        proposal.risks.max_loss_pct = proposal.stop_loss_pct
        if proposal.stop_loss_pct > 0.08:
            proposal.risks.key_risks.append("Wide stop loss (>8%)")
            proposal.risks.execution_risk = "medium"

        # Entry/exit logic
        proposal.entry_logic = "Enter at market on signal confirmation"
        proposal.exit_logic = (
            f"Exit at TP={proposal.take_profit_pct:.1%} or "
            f"SL={proposal.stop_loss_pct:.1%}"
        )

        return proposal

    # ================================================================== #
    #  STAGE B: Specialist Subgroup Review                                #
    # ================================================================== #

    def _stage_b_specialist_review(
        self,
        proposals: List[Proposal],
        thresholds,
    ) -> List[Proposal]:
        """Run specialist subgroup review on each proposal."""
        passed = []

        for proposal in proposals:
            result = self._specialist_review.review(proposal)

            proposal.specialist_review_passed = result.passed
            proposal.specialist_confidence = result.confidence_score
            proposal.specialist_objections = result.main_objections
            proposal.specialist_modifications = result.recommended_modifications

            if (
                result.passed
                and result.confidence_score >= thresholds.specialist_pass_threshold
            ):
                # Apply recommended modifications
                if result.preferred_horizon_days > 0:
                    proposal.time_horizon_days = result.preferred_horizon_days
                passed.append(proposal)

        return passed

    # ================================================================== #
    #  STAGE C: Global Weighted Vote                                      #
    # ================================================================== #

    def _stage_c_global_vote(
        self,
        proposals: List[Proposal],
        thresholds,
    ) -> List[Proposal]:
        """Run global weighted vote with anti-herd measures."""
        weights = self.scoring.get_all_weights()
        # Ensure all agents have a weight
        for aid in self._dna_map:
            if aid not in weights:
                weights[aid] = 1.0

        # Group proposals by symbol
        by_symbol: Dict[str, List[Proposal]] = defaultdict(list)
        for p in proposals:
            by_symbol[p.symbol].append(p)

        approved = []

        for symbol, symbol_proposals in by_symbol.items():
            # Get all agent IDs that proposed this symbol
            proposing_agents = {p.agent_id for p in symbol_proposals}

            # Count eligible voters (agents with sector knowledge)
            sector = symbol_proposals[0].sector
            eligible_weight = 0.0
            for aid, dna in self._dna_map.items():
                if dna.knows_sector(sector) or "all" in dna.primary_sectors:
                    eligible_weight += weights.get(aid, 1.0)

            if eligible_weight == 0:
                continue

            # Weighted approval
            supporting_weight = sum(
                weights.get(p.agent_id, 1.0) for p in symbol_proposals
            )
            approval_pct = supporting_weight / eligible_weight

            # Anti-herd: penalize if all supporters use same strategy
            strategies_used = set(p.strategy_used for p in symbol_proposals)
            if len(strategies_used) == 1 and len(symbol_proposals) > 3:
                approval_pct *= 0.85  # penalize monoculture

            # Uniqueness bonus: reward proposals with diverse supporter base
            if len(strategies_used) >= 3:
                approval_pct *= 1.05

            # Weighted confidence
            total_w = sum(
                weights.get(p.agent_id, 1.0) for p in symbol_proposals
            )
            weighted_conf = (
                sum(
                    p.confidence * weights.get(p.agent_id, 1.0)
                    for p in symbol_proposals
                )
                / total_w
                if total_w > 0
                else 0
            )

            # Exceptional override: high edge + tight risk can pass with
            # lower consensus
            exceptional = (
                weighted_conf > 0.85
                and any(
                    p.risks.max_loss_pct < 0.03 for p in symbol_proposals
                )
                and len(symbol_proposals) >= 3
            )

            passes = (
                approval_pct >= thresholds.approval_threshold
                and len(symbol_proposals) >= thresholds.min_voters
            ) or exceptional

            if passes:
                # Take the highest-confidence proposal as representative
                best = max(symbol_proposals, key=lambda p: p.confidence)
                best.global_vote_passed = True
                best.global_approval_pct = round(approval_pct, 4)
                best.global_weighted_confidence = round(weighted_conf, 4)

                # Dissent summary
                n_eligible = sum(
                    1
                    for dna in self._dna_map.values()
                    if dna.knows_sector(sector) or "all" in dna.primary_sectors
                )
                n_dissent = n_eligible - len(symbol_proposals)
                best.dissent_summary = (
                    f"{n_dissent}/{n_eligible} eligible agents did not propose "
                    f"{symbol}. Strategies: {', '.join(strategies_used)}"
                )

                approved.append(best)

        return approved

    # ================================================================== #
    #  LLM Reasoning Audit                                                 #
    # ================================================================== #

    def _llm_audit(
        self,
        proposals: List[Proposal],
        regime: str,
    ) -> List[Proposal]:
        """Run LLM reasoning audit on approved proposals, adjust confidence."""
        try:
            from src.agents.reasoning_auditor import audit_decision
        except ImportError:
            return proposals

        audited = []
        for proposal in proposals:
            try:
                # Collect supporting reasons from all agents that proposed this symbol
                supporting_reasons = [proposal.reasoning] if hasattr(proposal, 'reasoning') else []
                strategies = [proposal.strategy_used] if hasattr(proposal, 'strategy_used') else []

                result = audit_decision(
                    symbol=proposal.symbol,
                    direction=proposal.direction,
                    approval_pct=getattr(proposal, 'global_approval_pct', 0.5),
                    confidence=getattr(proposal, 'global_weighted_confidence', proposal.confidence),
                    supporting_reasons=supporting_reasons,
                    n_dissent=0,
                    strategies=strategies,
                    regime=regime,
                    sector=getattr(proposal, 'sector', 'unknown'),
                )

                # Apply confidence adjustment from audit
                if hasattr(proposal, 'global_weighted_confidence'):
                    proposal.global_weighted_confidence = max(
                        0.0,
                        min(1.0, proposal.global_weighted_confidence + result.confidence_adjustment)
                    )

                # Store audit result on proposal for logging
                proposal.audit_quality = result.quality_score
                proposal.audit_note = result.review_note

                # Reject if auditor says reject and quality is very low
                if result.recommendation == "reject" and result.quality_score < 0.3:
                    logger.info(
                        f"LLM audit rejected {proposal.symbol}: {result.review_note}"
                    )
                    continue

                audited.append(proposal)

            except Exception as e:
                logger.debug(f"LLM audit skipped for {proposal.symbol}: {e}")
                audited.append(proposal)  # Pass through on error

        logger.info(f"LLM Audit: {len(audited)}/{len(proposals)} passed reasoning review")
        return audited

    # ================================================================== #
    #  STAGE D: Portfolio/Risk Layer                                       #
    # ================================================================== #

    def _stage_d_portfolio_risk(
        self,
        proposals: List[Proposal],
        thresholds,
        portfolio_state: Optional[dict],
        regime: str,
    ) -> List[Proposal]:
        """Final portfolio/risk checks: exposure, correlation, timing, sizing."""
        approved = []
        sectors_allocated: Dict[str, float] = {}
        n_approved = 0

        # Sort by weighted confidence (best first)
        proposals.sort(
            key=lambda p: p.global_weighted_confidence, reverse=True
        )

        max_new = min(
            thresholds.max_new_positions,
            self.max_new_positions_per_day,
        )

        for proposal in proposals:
            if n_approved >= max_new:
                break

            # Check sector concentration
            sector = proposal.sector
            current_sector_pct = sectors_allocated.get(sector, 0.0)
            new_pct = proposal.suggested_position_pct
            if current_sector_pct + new_pct > self.max_sector_concentration:
                proposal.portfolio_approved = False
                continue

            # Position sizing: scale by regime multiplier and confidence
            base_size = proposal.suggested_position_pct
            regime_mult = thresholds.position_size_multiplier
            conf_scale = proposal.global_weighted_confidence
            final_size = base_size * regime_mult * conf_scale

            # Cap at 5% per position
            final_size = min(0.05, final_size)

            # Require stop loss in volatile regimes
            if thresholds.require_stop_loss and proposal.stop_loss_pct == 0:
                proposal.risks.key_risks.append(
                    "No stop loss in volatile regime"
                )
                final_size *= 0.5  # reduce size if no stop

            # Risk governor check if available
            if self.risk_governor:
                try:
                    from src.risk_management.risk_governor import (
                        OrderSide,
                        PortfolioState,
                    )

                    side = (
                        OrderSide.BUY
                        if proposal.direction == "long"
                        else OrderSide.SHORT
                    )
                    ps = portfolio_state or {}
                    if isinstance(ps, dict) and "nav" in ps:
                        notional = final_size * ps["nav"]
                        state = PortfolioState(
                            nav=ps["nav"],
                            peak_nav=ps.get("peak_nav", ps["nav"]),
                            cash=ps.get("cash", ps["nav"] * 0.1),
                        )
                        allowed, reason = self.risk_governor.pre_trade_check(
                            proposal.symbol,
                            side,
                            notional,
                            "swing",
                            state,
                            sector=sector,
                        )
                        if not allowed:
                            proposal.portfolio_approved = False
                            proposal.risks.key_risks.append(
                                f"Risk governor: {reason}"
                            )
                            continue
                except Exception:
                    pass

            proposal.portfolio_approved = True
            proposal.final_position_pct = round(final_size, 4)
            sectors_allocated[sector] = current_sector_pct + final_size
            approved.append(proposal)
            n_approved += 1

        return approved

    # ================================================================== #
    #  Outcome Recording                                                   #
    # ================================================================== #

    def record_outcomes(self, fills: List[dict], regime: str = "") -> None:
        """Record trade outcomes for scoring.

        Args:
            fills: List of fill dicts with symbol, actual_return, actual_days.
            regime: Current market regime string.
        """
        for fill in fills:
            symbol = fill["symbol"]
            actual_return = fill["actual_return"]
            actual_days = fill["actual_days"]

            # Find all agents that proposed this symbol
            for pick in self._idea_bucket.all_picks:
                if pick.symbol == symbol:
                    self.scoring.record_outcome(
                        agent_id=pick.agent_id,
                        symbol=symbol,
                        direction=pick.direction,
                        confidence=pick.confidence,
                        actual_return=actual_return,
                        actual_days=actual_days,
                        regime=regime,
                        peer_approved=pick.peer_approved,
                    )

    def get_leaderboard(self) -> pd.DataFrame:
        """Get agent leaderboard from enhanced scoring."""
        return self.scoring.get_leaderboard()

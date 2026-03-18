"""Vehicle Selection Engine — determines optimal investment type for each thesis."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.agents.proposal import (
    Proposal, VehicleCandidate, InvestmentType, ThesisType,
)


# Regime -> preferred vehicle adjustments
REGIME_VEHICLE_PREFS: Dict[str, Dict[str, float]] = {
    "low_vol_trending_up": {
        "shares_bias": 0.3, "options_bias": 0.1, "spread_bias": -0.1,
        "leaps_bias": 0.2, "no_trade_penalty": 0.3,
    },
    "high_vol_trending_up": {
        "shares_bias": 0.1, "options_bias": -0.1, "spread_bias": 0.2,
        "leaps_bias": 0.0, "no_trade_penalty": 0.1,
    },
    "low_vol_choppy": {
        "shares_bias": 0.0, "options_bias": -0.2, "spread_bias": 0.1,
        "leaps_bias": 0.0, "no_trade_penalty": 0.2,
    },
    "high_vol_choppy": {
        "shares_bias": -0.1, "options_bias": -0.3, "spread_bias": 0.1,
        "leaps_bias": -0.2, "no_trade_penalty": 0.4,
    },
    "high_vol_trending_down": {
        "shares_bias": -0.2, "options_bias": -0.2, "spread_bias": 0.2,
        "leaps_bias": -0.1, "no_trade_penalty": 0.3,
    },
    "low_vol_trending_down": {
        "shares_bias": -0.1, "options_bias": 0.0, "spread_bias": 0.1,
        "leaps_bias": 0.0, "no_trade_penalty": 0.2,
    },
}


class VehicleSelectionEngine:
    """
    Ranks the best expression for each trade thesis.

    For every opportunity, compares at least 3 candidate implementations
    and selects the highest-scoring one. Always includes "no_trade" as
    a candidate so weak signals get rejected at the vehicle level.

    Selection considers: time horizon, catalyst structure, volatility,
    risk profile, liquidity, conviction vs precision, portfolio role,
    and cost/carry.
    """

    def __init__(
        self,
        options_available: bool = False,
        min_option_oi: int = 100,
        max_option_spread_pct: float = 0.05,
    ):
        """
        Args:
            options_available: Whether options data/trading is available.
            min_option_oi: Minimum open interest for options to be considered.
            max_option_spread_pct: Max bid-ask spread as % for options.
        """
        self.options_available = options_available
        self.min_option_oi = min_option_oi
        self.max_option_spread_pct = max_option_spread_pct

    def select_vehicle(
        self,
        proposal: Proposal,
        regime: str = "",
        portfolio_context: Optional[dict] = None,
    ) -> Proposal:
        """
        Generate vehicle candidates and select the best one.

        Mutates proposal in place: sets vehicle_candidates, selected_vehicle,
        vehicle_rationale, and possibly adjusts investment_type.

        Args:
            proposal: The proposal to evaluate.
            regime: Current market regime string.
            portfolio_context: Optional dict with current portfolio state.

        Returns:
            The proposal with vehicle selection populated.
        """
        candidates = self._generate_candidates(proposal, regime)

        # Score each candidate
        for c in candidates:
            c.compute_composite()

        # Apply regime adjustments
        if regime:
            candidates = self._apply_regime_adjustments(candidates, regime)

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        proposal.vehicle_candidates = candidates

        if candidates:
            best = candidates[0]
            proposal.selected_vehicle = best
            proposal.investment_type = best.vehicle_type
            proposal.vehicle_rationale = self._explain_selection(
                best, candidates, proposal
            )

        return proposal

    def _generate_candidates(
        self, proposal: Proposal, regime: str
    ) -> List[VehicleCandidate]:
        """Generate candidate implementations for a proposal."""
        candidates = []

        # Always generate shares/spot candidate
        candidates.append(self._build_shares_candidate(proposal))

        # Generate options candidates if available
        if self.options_available:
            if proposal.direction == "long":
                candidates.append(self._build_call_option_candidate(proposal))
                candidates.append(self._build_call_spread_candidate(proposal))
                if proposal.time_horizon_days > 90:
                    candidates.append(self._build_leaps_candidate(proposal))
                # Covered call if already holding or low conviction
                candidates.append(self._build_covered_call_candidate(proposal))
            else:
                candidates.append(self._build_put_option_candidate(proposal))
                candidates.append(self._build_put_spread_candidate(proposal))
                candidates.append(self._build_protective_put_candidate(proposal))

        # Always include no-trade candidate
        candidates.append(self._build_no_trade_candidate(proposal))

        # Pairs trade if there's a hedge symbol
        if proposal.direction in ("long", "short"):
            candidates.append(self._build_pairs_candidate(proposal))

        return candidates

    def _build_shares_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for plain shares/stock position."""
        # Shares: no theta decay, simple, robust to timing
        expected_return = self._estimate_return(p, leverage=1.0)
        return VehicleCandidate(
            vehicle_type=InvestmentType.SHARES_SPOT,
            description=f"{'Buy' if p.direction == 'long' else 'Short'} {p.symbol} shares",
            expected_return=expected_return,
            expected_risk=min(1.0, p.expected_annual_vol / 100) if p.expected_annual_vol > 0 else 0.3,
            payoff_asymmetry=0.5,  # symmetric for shares
            timing_sensitivity=0.2 if p.time_horizon_days > 20 else 0.5,
            volatility_sensitivity=0.3,
            liquidity_quality=min(1.0, p.avg_daily_volume / 5_000_000) if p.avg_daily_volume > 0 else 0.7,
            implementation_simplicity=1.0,
            robustness_to_error=0.8 if p.time_horizon_days > 10 else 0.5,
            theta_decay_cost=0.0,
            transaction_cost_bps=2.0,
        )

    def _build_call_option_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for buying call options."""
        leverage = 5.0  # rough leverage estimate
        expected_return = self._estimate_return(p, leverage=leverage)

        # IV vs RV assessment
        iv_expensive = p.implied_vol > p.realized_vol * 1.2 if p.implied_vol > 0 and p.realized_vol > 0 else False

        return VehicleCandidate(
            vehicle_type=InvestmentType.CALL_OPTION,
            description=f"Buy {p.symbol} ATM call, ~{p.time_horizon_days}d expiry",
            expected_return=expected_return * (0.6 if iv_expensive else 1.0),
            expected_risk=0.8,  # can lose entire premium
            payoff_asymmetry=0.8 if not iv_expensive else 0.5,
            timing_sensitivity=0.8,
            volatility_sensitivity=0.9,
            liquidity_quality=0.5,  # generally lower than shares
            implementation_simplicity=0.6,
            robustness_to_error=0.3,  # fragile to timing
            theta_decay_cost=self._estimate_theta_cost(p),
            transaction_cost_bps=10.0,
        )

    def _build_call_spread_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for bull call spread (defined risk)."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.CALL_SPREAD,
            description=f"Bull call spread on {p.symbol}",
            expected_return=self._estimate_return(p, leverage=3.0) * 0.7,
            expected_risk=0.6,  # capped at spread width
            payoff_asymmetry=0.65,
            timing_sensitivity=0.6,
            volatility_sensitivity=0.4,  # reduced vs naked options
            liquidity_quality=0.4,
            implementation_simplicity=0.4,
            robustness_to_error=0.45,
            theta_decay_cost=self._estimate_theta_cost(p) * 0.5,
            transaction_cost_bps=15.0,
        )

    def _build_put_option_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for buying put options."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.PUT_OPTION,
            description=f"Buy {p.symbol} ATM put, ~{p.time_horizon_days}d expiry",
            expected_return=self._estimate_return(p, leverage=5.0),
            expected_risk=0.8,
            payoff_asymmetry=0.8,
            timing_sensitivity=0.8,
            volatility_sensitivity=0.9,
            liquidity_quality=0.5,
            implementation_simplicity=0.6,
            robustness_to_error=0.3,
            theta_decay_cost=self._estimate_theta_cost(p),
            transaction_cost_bps=10.0,
        )

    def _build_put_spread_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for bear put spread."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.PUT_SPREAD,
            description=f"Bear put spread on {p.symbol}",
            expected_return=self._estimate_return(p, leverage=3.0) * 0.7,
            expected_risk=0.6,
            payoff_asymmetry=0.65,
            timing_sensitivity=0.6,
            volatility_sensitivity=0.4,
            liquidity_quality=0.4,
            implementation_simplicity=0.4,
            robustness_to_error=0.45,
            theta_decay_cost=self._estimate_theta_cost(p) * 0.5,
            transaction_cost_bps=15.0,
        )

    def _build_leaps_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for LEAPS (long-term options > 9 months)."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.LEAPS,
            description=f"Buy {p.symbol} LEAPS call, 9-12 month expiry",
            expected_return=self._estimate_return(p, leverage=3.0),
            expected_risk=0.7,
            payoff_asymmetry=0.75,
            timing_sensitivity=0.3,  # long duration = more forgiving
            volatility_sensitivity=0.6,
            liquidity_quality=0.3,
            implementation_simplicity=0.5,
            robustness_to_error=0.6,  # more time to be right
            theta_decay_cost=self._estimate_theta_cost(p) * 0.3,
            transaction_cost_bps=12.0,
        )

    def _build_covered_call_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for covered call (income/hedge)."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.COVERED_CALL,
            description=f"Buy {p.symbol} shares + sell OTM call",
            expected_return=self._estimate_return(p, leverage=1.0) * 0.6,
            expected_risk=0.4,  # reduced by premium
            payoff_asymmetry=0.3,  # capped upside
            timing_sensitivity=0.3,
            volatility_sensitivity=0.4,
            liquidity_quality=0.6,
            implementation_simplicity=0.5,
            robustness_to_error=0.7,
            theta_decay_cost=-0.02,  # you COLLECT theta
            transaction_cost_bps=8.0,
        )

    def _build_protective_put_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for protective put (downside hedge)."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.PROTECTIVE_PUT,
            description=f"Short {p.symbol} shares + buy OTM put for protection",
            expected_return=self._estimate_return(p, leverage=1.0) * 0.8,
            expected_risk=0.3,  # defined max loss
            payoff_asymmetry=0.7,
            timing_sensitivity=0.4,
            volatility_sensitivity=0.5,
            liquidity_quality=0.5,
            implementation_simplicity=0.4,
            robustness_to_error=0.65,
            theta_decay_cost=self._estimate_theta_cost(p) * 0.4,
            transaction_cost_bps=12.0,
        )

    def _build_pairs_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build candidate for pairs/relative value trade."""
        return VehicleCandidate(
            vehicle_type=InvestmentType.PAIRS_TRADE,
            description=f"Pairs trade: {p.direction} {p.symbol} vs sector hedge",
            expected_return=self._estimate_return(p, leverage=1.0) * 0.5,
            expected_risk=0.25,  # market-neutral reduces risk
            payoff_asymmetry=0.55,
            timing_sensitivity=0.3,
            volatility_sensitivity=0.2,  # hedged
            liquidity_quality=0.5,
            implementation_simplicity=0.3,
            robustness_to_error=0.7,
            theta_decay_cost=0.0,
            transaction_cost_bps=5.0,
        )

    def _build_no_trade_candidate(self, p: Proposal) -> VehicleCandidate:
        """Build the no-trade baseline candidate."""
        # No-trade score based on signal weakness
        no_trade_score = max(0.1, 1.0 - p.confidence)
        return VehicleCandidate(
            vehicle_type=InvestmentType.NO_TRADE,
            description=f"Pass on {p.symbol} — wait for better setup",
            expected_return=0.0,
            expected_risk=0.0,
            payoff_asymmetry=0.5,
            timing_sensitivity=0.0,
            volatility_sensitivity=0.0,
            liquidity_quality=1.0,
            implementation_simplicity=1.0,
            robustness_to_error=1.0,
            theta_decay_cost=0.0,
            transaction_cost_bps=0.0,
        )

    def _apply_regime_adjustments(
        self, candidates: List[VehicleCandidate], regime: str
    ) -> List[VehicleCandidate]:
        """Apply regime-specific biases to candidate scores."""
        prefs = REGIME_VEHICLE_PREFS.get(regime, {})
        if not prefs:
            return candidates

        for c in candidates:
            if c.vehicle_type == InvestmentType.SHARES_SPOT:
                c.composite_score += prefs.get("shares_bias", 0)
            elif c.vehicle_type in (InvestmentType.CALL_OPTION, InvestmentType.PUT_OPTION):
                c.composite_score += prefs.get("options_bias", 0)
            elif c.vehicle_type in (InvestmentType.CALL_SPREAD, InvestmentType.PUT_SPREAD):
                c.composite_score += prefs.get("spread_bias", 0)
            elif c.vehicle_type == InvestmentType.LEAPS:
                c.composite_score += prefs.get("leaps_bias", 0)
            elif c.vehicle_type == InvestmentType.NO_TRADE:
                c.composite_score += prefs.get("no_trade_penalty", 0)

        return candidates

    def _estimate_return(self, p: Proposal, leverage: float) -> float:
        """Estimate expected return scaled by confidence and leverage."""
        base = p.confidence * p.expected_edge_bps / 10000 if p.expected_edge_bps > 0 else p.confidence * 0.02
        return min(1.0, max(0.0, base * leverage))

    def _estimate_theta_cost(self, p: Proposal) -> float:
        """Estimate annualized theta decay cost as fraction."""
        if p.time_horizon_days <= 0:
            return 0.5
        # Theta accelerates as expiry approaches
        if p.time_horizon_days < 7:
            return 0.4
        elif p.time_horizon_days < 30:
            return 0.2
        elif p.time_horizon_days < 90:
            return 0.1
        else:
            return 0.05

    def _explain_selection(
        self,
        best: VehicleCandidate,
        all_candidates: List[VehicleCandidate],
        proposal: Proposal,
    ) -> str:
        """Generate human-readable explanation of vehicle selection."""
        lines = [f"Selected: {best.description} (score={best.composite_score:.3f})"]

        # Why this over alternatives
        if best.vehicle_type == InvestmentType.NO_TRADE:
            lines.append("Reason: Signal quality insufficient for any implementation.")
            return " | ".join(lines)

        if best.vehicle_type == InvestmentType.SHARES_SPOT:
            reasons = []
            if proposal.time_horizon_days > 20:
                reasons.append("long horizon favors patience of shares")
            if best.robustness_to_error > 0.6:
                reasons.append("robust to timing errors")
            if not reasons:
                reasons.append("simplest implementation with acceptable risk/reward")
            lines.append(f"Reason: {'; '.join(reasons)}")

        elif best.vehicle_type in (InvestmentType.CALL_OPTION, InvestmentType.PUT_OPTION):
            lines.append("Reason: high conviction + defined catalyst + acceptable IV")

        elif best.vehicle_type in (InvestmentType.CALL_SPREAD, InvestmentType.PUT_SPREAD):
            lines.append("Reason: defined risk preferred due to elevated IV or uncertain timing")

        elif best.vehicle_type == InvestmentType.LEAPS:
            lines.append("Reason: long-horizon fundamental thesis benefits from low theta decay")

        elif best.vehicle_type == InvestmentType.PAIRS_TRADE:
            lines.append("Reason: market-neutral implementation reduces systematic risk")

        # Add runner-up comparison
        if len(all_candidates) >= 2:
            runner = all_candidates[1]
            lines.append(
                f"vs {runner.description}: score={runner.composite_score:.3f} "
                f"(delta={best.composite_score - runner.composite_score:.3f})"
            )

        return " | ".join(lines)

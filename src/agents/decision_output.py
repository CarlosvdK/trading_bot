"""Decision Output — structured final output for approved trades."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.agents.proposal import Proposal, InvestmentType, VehicleCandidate


@dataclass
class DecisionOutput:
    """Complete structured output for an approved trade.

    This is what the system produces for each opportunity that passes
    all 4 pipeline stages. It contains everything needed to execute,
    monitor, and evaluate the trade.
    """

    # Identity
    decision_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Asset
    asset: str = ""
    sector: str = ""

    # Direction and type
    direction: str = ""  # long | short
    investment_type: str = ""  # from InvestmentType enum
    investment_type_rationale: str = ""

    # Holding horizon
    holding_horizon_days: int = 0
    holding_category: str = ""  # scalp | swing | position | macro

    # Entry/exit logic
    entry_logic: str = ""
    exit_logic: str = ""
    stop_logic: str = ""
    target_logic: str = ""
    invalidation_logic: str = ""

    # Conviction
    confidence: float = 0.0
    weighted_vote_result: float = 0.0
    specialist_confidence: float = 0.0

    # Dissent
    dissent_summary: str = ""
    main_objections: List[str] = field(default_factory=list)

    # Capital allocation
    position_size_pct: float = 0.0
    capital_allocated: float = 0.0
    suggested_shares: int = 0

    # Vehicle comparison
    selected_vehicle: str = ""
    vehicle_rationale: str = ""
    alternative_vehicles: List[Dict] = field(default_factory=list)

    # Risk
    max_loss_pct: float = 0.0
    expected_annual_vol: float = 0.0
    key_risks: List[str] = field(default_factory=list)

    # Regime context
    current_regime: str = ""
    regime_fit: str = ""  # good | moderate | poor

    # Full thesis
    thesis: str = ""
    catalyst: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "sector": self.sector,
            "direction": self.direction,
            "investment_type": self.investment_type,
            "investment_type_rationale": self.investment_type_rationale,
            "holding_horizon_days": self.holding_horizon_days,
            "holding_category": self.holding_category,
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "stop_logic": self.stop_logic,
            "target_logic": self.target_logic,
            "invalidation_logic": self.invalidation_logic,
            "confidence": self.confidence,
            "weighted_vote_result": self.weighted_vote_result,
            "specialist_confidence": self.specialist_confidence,
            "dissent_summary": self.dissent_summary,
            "main_objections": self.main_objections,
            "position_size_pct": self.position_size_pct,
            "capital_allocated": self.capital_allocated,
            "selected_vehicle": self.selected_vehicle,
            "vehicle_rationale": self.vehicle_rationale,
            "alternative_vehicles": self.alternative_vehicles,
            "max_loss_pct": self.max_loss_pct,
            "expected_annual_vol": self.expected_annual_vol,
            "key_risks": self.key_risks,
            "current_regime": self.current_regime,
            "regime_fit": self.regime_fit,
            "thesis": self.thesis,
            "catalyst": self.catalyst,
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.direction.upper()} {self.asset} via {self.investment_type} | "
            f"Conf={self.confidence:.0%} Vote={self.weighted_vote_result:.0%} | "
            f"Size={self.position_size_pct:.1%} | Hold={self.holding_horizon_days}d | "
            f"Regime={self.current_regime}"
        )


def build_decision_output(proposal: Proposal, regime: str = "") -> DecisionOutput:
    """Build a DecisionOutput from a fully-approved Proposal.

    Args:
        proposal: A Proposal that has passed all 4 pipeline stages.
        regime: Current market regime string.

    Returns:
        DecisionOutput with all fields populated from the proposal.
    """
    # Classify holding category
    if proposal.time_horizon_days <= 3:
        hold_cat = "scalp"
    elif proposal.time_horizon_days <= 15:
        hold_cat = "swing"
    elif proposal.time_horizon_days <= 90:
        hold_cat = "position"
    else:
        hold_cat = "macro"

    # Regime fit assessment
    regime_fit = "moderate"
    if regime in ("low_vol_trending_up", "risk_on") and proposal.direction == "long":
        regime_fit = "good"
    elif regime in ("high_vol_trending_down", "liquidity_stressed"):
        regime_fit = "poor"
    elif regime in ("high_vol_choppy",) and proposal.direction == "long":
        regime_fit = "poor"

    # Build alternative vehicles summary
    alt_vehicles = []
    if proposal.vehicle_candidates:
        for vc in proposal.vehicle_candidates[:5]:
            if (
                proposal.selected_vehicle
                and vc.vehicle_type == proposal.selected_vehicle.vehicle_type
            ):
                continue
            alt_vehicles.append({
                "type": vc.vehicle_type.value,
                "description": vc.description,
                "score": round(vc.composite_score, 3),
            })

    inv_type = ""
    inv_rationale = ""
    if proposal.selected_vehicle:
        inv_type = proposal.selected_vehicle.vehicle_type.value
        inv_rationale = proposal.vehicle_rationale
    else:
        inv_type = proposal.investment_type.value
        inv_rationale = "Default vehicle selection"

    return DecisionOutput(
        decision_id=proposal.proposal_id,
        timestamp=proposal.timestamp,
        asset=proposal.symbol,
        sector=proposal.sector,
        direction=proposal.direction,
        investment_type=inv_type,
        investment_type_rationale=inv_rationale,
        holding_horizon_days=proposal.time_horizon_days,
        holding_category=hold_cat,
        entry_logic=proposal.entry_logic,
        exit_logic=proposal.exit_logic,
        stop_logic=(
            f"Stop at {proposal.stop_loss_pct:.1%} from entry"
            if proposal.stop_loss_pct > 0
            else "No stop set"
        ),
        target_logic=(
            f"Target at {proposal.take_profit_pct:.1%} from entry"
            if proposal.take_profit_pct > 0
            else "No target set"
        ),
        invalidation_logic=(
            "; ".join(proposal.risks.invalidation_criteria)
            if proposal.risks.invalidation_criteria
            else "Price through stop level"
        ),
        confidence=proposal.confidence,
        weighted_vote_result=proposal.global_approval_pct,
        specialist_confidence=proposal.specialist_confidence,
        dissent_summary=proposal.dissent_summary,
        main_objections=proposal.specialist_objections,
        position_size_pct=proposal.final_position_pct,
        capital_allocated=proposal.capital_allocated,
        selected_vehicle=inv_type,
        vehicle_rationale=inv_rationale,
        alternative_vehicles=alt_vehicles,
        max_loss_pct=proposal.risks.max_loss_pct,
        expected_annual_vol=proposal.expected_annual_vol,
        key_risks=proposal.risks.key_risks,
        current_regime=regime,
        regime_fit=regime_fit,
        thesis=proposal.thesis,
        catalyst=proposal.catalyst,
    )

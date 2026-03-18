"""Voting Engine — weighted consensus voting for trade approval."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.agents.trading_agent import TradePick
from src.agents.agent_dna import AgentDNA, ALL_SECTORS
from src.agents.sector_mapping import get_sector


@dataclass
class ApprovedTrade:
    """A trade that passed the voting threshold."""

    symbol: str
    direction: str
    approval_pct: float          # what % of weighted votes approved
    weighted_confidence: float   # weight-averaged confidence
    num_voters: int
    supporting_agents: List[str]
    dissenting_agents: List[str]
    consensus_hold_days: int     # weighted average of suggested holds
    sector: str = ""
    avg_raw_score: float = 0.0
    peer_pre_vote_bonus: float = 0.0  # bonus from peer pre-voting


class VotingEngine:
    """
    Weighted consensus voting system.

    For each symbol in the idea bucket, calculates approval percentage
    based on weighted votes from agents with sector knowledge.
    Only agents that know the symbol's sector can vote.
    """

    def __init__(
        self,
        approval_threshold: float = 0.7,
        min_voters: int = 5,
        peer_bonus_weight: float = 0.1,
    ):
        """
        Args:
            approval_threshold: Minimum weighted approval % to pass (0-1).
            min_voters: Minimum number of voting agents required.
            peer_bonus_weight: Extra weight for picks that passed peer pre-vote.
        """
        self.approval_threshold = approval_threshold
        self.min_voters = min_voters
        self.peer_bonus_weight = peer_bonus_weight

    def run_vote(
        self,
        candidates: Dict[str, List[TradePick]],
        weights: Dict[str, float],
        all_agent_dnas: Optional[Dict[str, AgentDNA]] = None,
    ) -> List[ApprovedTrade]:
        """
        Run weighted consensus vote on all candidate symbols.

        Args:
            candidates: {symbol: [TradePicks]} from IdeaBucket.
            weights: {agent_id: weight} from AgentScorekeeper.
            all_agent_dnas: {agent_id: AgentDNA} for sector eligibility checks.

        Returns:
            List of ApprovedTrade sorted by weighted_confidence descending.
        """
        approved: List[ApprovedTrade] = []

        for symbol, picks in candidates.items():
            sector = get_sector(symbol) or ""

            # Determine eligible voters (agents that know this sector)
            eligible_weight_sum = 0.0
            eligible_agents: Set[str] = set()

            if all_agent_dnas:
                for agent_id, dna in all_agent_dnas.items():
                    if dna.knows_sector(sector) or not dna.primary_sectors or "all" in (dna.primary_sectors + dna.secondary_sectors):
                        eligible_agents.add(agent_id)
                        eligible_weight_sum += weights.get(agent_id, 1.0)
            else:
                # If no DNA info, all agents are eligible
                for pick in picks:
                    eligible_agents.add(pick.agent_id)
                eligible_weight_sum = sum(weights.get(p.agent_id, 1.0) for p in picks)

            if len(eligible_agents) == 0:
                continue

            # Calculate approval
            approval_pct = self._calculate_approval(
                picks, weights, eligible_weight_sum
            )

            # Check direction consensus
            direction = self._consensus_direction(picks, weights)

            # Calculate weighted confidence
            weighted_conf = self._aggregate_confidence(picks, weights)

            # Peer pre-vote bonus
            peer_bonus = self._calculate_peer_bonus(picks)

            # Determine supporters vs dissenters
            supporting = [p.agent_id for p in picks]
            dissenting = [
                aid for aid in eligible_agents if aid not in set(supporting)
            ]

            # Consensus hold days (weighted average)
            hold_days = self._consensus_hold_days(picks, weights)

            # Average raw score
            avg_raw = sum(p.raw_score for p in picks) / len(picks) if picks else 0

            num_voters = len(picks)

            # Apply peer bonus to approval
            effective_approval = approval_pct + (peer_bonus * self.peer_bonus_weight)

            if (
                effective_approval >= self.approval_threshold
                and num_voters >= self.min_voters
            ):
                approved.append(ApprovedTrade(
                    symbol=symbol,
                    direction=direction,
                    approval_pct=round(effective_approval, 4),
                    weighted_confidence=round(weighted_conf, 4),
                    num_voters=num_voters,
                    supporting_agents=supporting,
                    dissenting_agents=dissenting,
                    consensus_hold_days=hold_days,
                    sector=sector,
                    avg_raw_score=round(avg_raw, 4),
                    peer_pre_vote_bonus=round(peer_bonus, 4),
                ))

        # Sort by weighted confidence
        approved.sort(key=lambda t: t.weighted_confidence, reverse=True)
        return approved

    def _calculate_approval(
        self,
        picks: List[TradePick],
        weights: Dict[str, float],
        eligible_weight_sum: float,
    ) -> float:
        """Calculate weighted approval percentage."""
        if eligible_weight_sum <= 0:
            return 0.0

        supporting_weight = sum(
            weights.get(p.agent_id, 1.0) for p in picks
        )
        return supporting_weight / eligible_weight_sum

    def _aggregate_confidence(
        self,
        picks: List[TradePick],
        weights: Dict[str, float],
    ) -> float:
        """Calculate weight-averaged confidence."""
        if not picks:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for pick in picks:
            w = weights.get(pick.agent_id, 1.0)
            weighted_sum += pick.confidence * w
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _consensus_direction(
        self,
        picks: List[TradePick],
        weights: Dict[str, float],
    ) -> str:
        """Determine consensus direction by weighted vote."""
        long_weight = sum(
            weights.get(p.agent_id, 1.0) for p in picks if p.direction == "long"
        )
        short_weight = sum(
            weights.get(p.agent_id, 1.0) for p in picks if p.direction == "short"
        )
        return "long" if long_weight >= short_weight else "short"

    def _consensus_hold_days(
        self,
        picks: List[TradePick],
        weights: Dict[str, float],
    ) -> int:
        """Calculate weighted average hold days."""
        if not picks:
            return 10

        total_weight = 0.0
        weighted_sum = 0.0
        for pick in picks:
            w = weights.get(pick.agent_id, 1.0)
            weighted_sum += pick.suggested_hold_days * w
            total_weight += w

        return int(round(weighted_sum / total_weight)) if total_weight > 0 else 10

    def _calculate_peer_bonus(self, picks: List[TradePick]) -> float:
        """
        Calculate bonus from peer pre-voting.

        Picks that were approved by their peer group get a bonus
        proportional to their peer approval percentage.
        """
        if not picks:
            return 0.0

        peer_approved_picks = [p for p in picks if p.peer_approved]
        if not peer_approved_picks:
            return 0.0

        avg_peer_approval = sum(
            p.peer_approval_pct for p in peer_approved_picks
        ) / len(peer_approved_picks)

        # Bonus scaled by fraction of picks that were peer-approved
        fraction_peer_approved = len(peer_approved_picks) / len(picks)
        return avg_peer_approval * fraction_peer_approved

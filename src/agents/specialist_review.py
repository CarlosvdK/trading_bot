"""Specialist Review — Stage B of multi-stage pipeline.

Routes each proposal to a subgroup of relevant experts for pre-screening.
The subgroup includes supportive, skeptical, and risk-oriented reviewers.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.agents.agent_dna import AgentDNA
from src.agents.proposal import Proposal
from src.agents.sector_mapping import get_sector

logger = logging.getLogger(__name__)


@dataclass
class SpecialistVerdict:
    """Result from a single specialist reviewer."""
    agent_id: str
    vote: str  # "approve" | "reject" | "modify"
    confidence: float
    supporting_reasons: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    recommended_modifications: List[str] = field(default_factory=list)
    preferred_vehicle: str = ""
    preferred_horizon_days: int = 0
    role: str = ""  # "supportive" | "skeptical" | "risk"


@dataclass
class SubgroupReviewResult:
    """Aggregated result from specialist subgroup review."""
    passed: bool
    confidence_score: float
    approval_count: int
    reject_count: int
    modify_count: int
    total_reviewers: int
    main_supporting_reasons: List[str] = field(default_factory=list)
    main_objections: List[str] = field(default_factory=list)
    recommended_modifications: List[str] = field(default_factory=list)
    preferred_vehicle: str = ""
    preferred_horizon_days: int = 0
    verdicts: List[SpecialistVerdict] = field(default_factory=list)


class SpecialistReview:
    """
    Selects and runs specialist subgroup review for each proposal.

    The subgroup is composed of:
    - 2-3 supportive specialists (agents with expertise in the proposal's sector/strategy)
    - 1-2 skeptical specialists (contrarian or different-strategy agents with sector knowledge)
    - 1 risk-oriented reviewer (guardian/conservative agent)

    Minimum 4, maximum 7 reviewers per subgroup.
    """

    def __init__(
        self,
        all_dnas: Dict[str, AgentDNA],
        min_reviewers: int = 4,
        max_reviewers: int = 7,
        pass_threshold: float = 0.5,
    ):
        """
        Args:
            all_dnas: {agent_id: AgentDNA} for all agents.
            min_reviewers: Minimum specialist reviewers per proposal.
            max_reviewers: Maximum specialist reviewers per proposal.
            pass_threshold: Fraction of reviewers that must approve/modify.
        """
        self.all_dnas = all_dnas
        self.min_reviewers = min_reviewers
        self.max_reviewers = max_reviewers
        self.pass_threshold = pass_threshold

        # Pre-categorize agents
        self._sector_experts: Dict[str, List[str]] = defaultdict(list)
        self._strategy_experts: Dict[str, List[str]] = defaultdict(list)
        self._contrarians: List[str] = []
        self._risk_reviewers: List[str] = []

        for aid, dna in all_dnas.items():
            for s in dna.primary_sectors + dna.secondary_sectors:
                self._sector_experts[s].append(aid)
            self._strategy_experts[dna.primary_strategy].append(aid)
            if dna.contrarian_factor > 0.5:
                self._contrarians.append(aid)
            if dna.risk_appetite < 0.2 or "guardian" in aid:
                self._risk_reviewers.append(aid)

    def review(
        self,
        proposal: Proposal,
        agent_scores: Optional[Dict[str, float]] = None,
    ) -> SubgroupReviewResult:
        """
        Run specialist subgroup review on a single proposal.

        Args:
            proposal: The proposal to review.
            agent_scores: Optional {agent_id: confidence_score} from actual scans.

        Returns:
            SubgroupReviewResult with pass/reject and detailed feedback.
        """
        # Select the subgroup
        subgroup = self._select_subgroup(proposal)

        if len(subgroup) < self.min_reviewers:
            # Not enough specialists, auto-pass with low confidence
            return SubgroupReviewResult(
                passed=True, confidence_score=0.3,
                approval_count=0, reject_count=0, modify_count=0,
                total_reviewers=0,
                main_objections=["Insufficient specialist reviewers available"],
            )

        # Each specialist evaluates the proposal
        verdicts = []
        for agent_id, role in subgroup:
            dna = self.all_dnas[agent_id]
            verdict = self._evaluate_proposal(proposal, dna, agent_id, role, agent_scores)
            verdicts.append(verdict)

        # Aggregate verdicts
        return self._aggregate_verdicts(verdicts, proposal)

    def _select_subgroup(
        self, proposal: Proposal
    ) -> List[Tuple[str, str]]:
        """
        Select specialist subgroup for a proposal.

        Returns list of (agent_id, role) tuples.
        """
        selected: List[Tuple[str, str]] = []
        used: Set[str] = {proposal.agent_id}  # exclude the proposer

        sector = proposal.sector or get_sector(proposal.symbol) or ""
        strategy = proposal.strategy_used

        # 1. Supportive specialists: same sector + same/similar strategy
        sector_pool = [
            aid for aid in self._sector_experts.get(sector, [])
            if aid not in used
        ]
        strategy_pool = [
            aid for aid in self._strategy_experts.get(strategy, [])
            if aid not in used
        ]

        # Prefer agents that match BOTH sector and strategy
        both = [aid for aid in sector_pool if aid in strategy_pool]
        for aid in both[:2]:
            selected.append((aid, "supportive"))
            used.add(aid)

        # Fill remaining supportive from sector pool
        for aid in sector_pool:
            if aid not in used and len([s for s in selected if s[1] == "supportive"]) < 3:
                selected.append((aid, "supportive"))
                used.add(aid)

        # 2. Skeptical specialists: contrarians with sector knowledge
        skeptic_pool = [
            aid for aid in self._contrarians
            if aid not in used and (
                self.all_dnas[aid].knows_sector(sector)
                or "all" in self.all_dnas[aid].primary_sectors
            )
        ]
        for aid in skeptic_pool[:2]:
            selected.append((aid, "skeptical"))
            used.add(aid)

        # If no contrarians available, use agents with different strategy
        if not skeptic_pool:
            diff_strategy = [
                aid for aid in sector_pool
                if aid not in used
                and self.all_dnas[aid].primary_strategy != strategy
            ]
            for aid in diff_strategy[:1]:
                selected.append((aid, "skeptical"))
                used.add(aid)

        # 3. Risk reviewer: guardian/conservative agent
        risk_pool = [aid for aid in self._risk_reviewers if aid not in used]
        if risk_pool:
            selected.append((risk_pool[0], "risk"))
            used.add(risk_pool[0])

        return selected[:self.max_reviewers]

    def _evaluate_proposal(
        self,
        proposal: Proposal,
        reviewer_dna: AgentDNA,
        reviewer_id: str,
        role: str,
        agent_scores: Optional[Dict[str, float]] = None,
    ) -> SpecialistVerdict:
        """Have a single specialist evaluate the proposal."""
        supporting = []
        objections = []
        modifications = []

        # Base confidence from the proposal's own confidence
        conf = proposal.confidence

        # Sector expertise check
        sector = proposal.sector or ""
        if reviewer_dna.knows_sector(sector):
            sector_w = reviewer_dna.sector_weight(sector)
            if sector_w == 1.0:
                supporting.append(f"Deep {sector} expertise confirms thesis")
                conf *= 1.1
            else:
                supporting.append(f"Secondary {sector} knowledge supports")
        else:
            objections.append(f"Limited {sector} expertise — review may lack depth")
            conf *= 0.8

        # Strategy alignment check
        if reviewer_dna.primary_strategy == proposal.strategy_used:
            supporting.append(f"Strategy alignment: {proposal.strategy_used}")
            conf *= 1.05
        elif role == "skeptical":
            objections.append(f"Different strategy lens ({reviewer_dna.primary_strategy}) questions approach")
            conf *= 0.9

        # Risk appetite alignment
        if role == "risk":
            if proposal.risks.max_loss_pct > 0.05:
                objections.append(f"Max loss {proposal.risks.max_loss_pct:.1%} exceeds conservative threshold")
                conf *= 0.85
            if proposal.confidence < 0.7:
                objections.append(f"Confidence {proposal.confidence:.2f} below risk reviewer threshold")
                modifications.append("Reduce position size by 50%")
                conf *= 0.9
            if proposal.risks.crowded_trade_score > 0.5:
                objections.append(f"Crowded trade risk: {proposal.risks.crowded_trade_score:.2f}")
                conf *= 0.85

        # Contrarian check
        if reviewer_dna.contrarian_factor > 0.6:
            if proposal.confidence > 0.8:
                objections.append("High confidence often precedes reversals — contrarian caution")
                conf *= 0.9

        # Holding period alignment
        reviewer_min, reviewer_max = reviewer_dna.holding_days_range
        if proposal.time_horizon_days < reviewer_min:
            modifications.append(f"Consider extending horizon to {reviewer_min}+ days")
        elif proposal.time_horizon_days > reviewer_max:
            modifications.append(f"Consider shortening horizon to {reviewer_max} days")

        # Use agent_scores if available
        if agent_scores and reviewer_id in agent_scores:
            peer_conf = agent_scores[reviewer_id]
            conf = (conf + peer_conf) / 2

        # Determine vote based on role
        conf = max(0.0, min(1.0, conf))
        if role == "skeptical":
            vote = "approve" if conf > 0.6 else ("modify" if conf > 0.4 else "reject")
        elif role == "risk":
            vote = "approve" if conf > 0.65 else ("modify" if conf > 0.45 else "reject")
        else:
            vote = "approve" if conf > 0.5 else ("modify" if conf > 0.3 else "reject")

        return SpecialistVerdict(
            agent_id=reviewer_id,
            vote=vote,
            confidence=round(conf, 4),
            supporting_reasons=supporting,
            objections=objections,
            recommended_modifications=modifications,
            preferred_vehicle=proposal.investment_type.value if vote != "reject" else "",
            preferred_horizon_days=proposal.time_horizon_days,
            role=role,
        )

    def _aggregate_verdicts(
        self, verdicts: List[SpecialistVerdict], proposal: Proposal
    ) -> SubgroupReviewResult:
        """Aggregate individual verdicts into a subgroup result."""
        approve_count = sum(1 for v in verdicts if v.vote == "approve")
        reject_count = sum(1 for v in verdicts if v.vote == "reject")
        modify_count = sum(1 for v in verdicts if v.vote == "modify")
        total = len(verdicts)

        # Pass if enough approve or modify (not reject)
        pass_count = approve_count + modify_count
        passed = (pass_count / total) >= self.pass_threshold if total > 0 else False

        # Weighted confidence
        confidences = [v.confidence for v in verdicts]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        # Collect reasons
        all_supporting = []
        all_objections = []
        all_modifications = []
        for v in verdicts:
            all_supporting.extend(v.supporting_reasons)
            all_objections.extend(v.objections)
            all_modifications.extend(v.recommended_modifications)

        # Deduplicate
        all_supporting = list(dict.fromkeys(all_supporting))
        all_objections = list(dict.fromkeys(all_objections))
        all_modifications = list(dict.fromkeys(all_modifications))

        # Consensus vehicle preference
        vehicles = [v.preferred_vehicle for v in verdicts if v.preferred_vehicle]
        preferred_vehicle = max(set(vehicles), key=vehicles.count) if vehicles else ""

        horizons = [v.preferred_horizon_days for v in verdicts if v.preferred_horizon_days > 0]
        preferred_horizon = int(sum(horizons) / len(horizons)) if horizons else proposal.time_horizon_days

        return SubgroupReviewResult(
            passed=passed,
            confidence_score=round(avg_conf, 4),
            approval_count=approve_count,
            reject_count=reject_count,
            modify_count=modify_count,
            total_reviewers=total,
            main_supporting_reasons=all_supporting[:5],
            main_objections=all_objections[:5],
            recommended_modifications=all_modifications[:5],
            preferred_vehicle=preferred_vehicle,
            preferred_horizon_days=preferred_horizon,
            verdicts=verdicts,
        )

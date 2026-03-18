"""Agent Evolution — replacement framework for underperforming agents.

Distinguishes between:
1. Temporary regime mismatch (agent is fine, just wrong regime)
2. True model decay (agent's edge has eroded)
3. Redundancy (agent duplicates a stronger agent)

Maintains a healthy mix of proven agents and experimental challengers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from src.agents.agent_dna import AgentDNA, VALID_STRATEGIES, ALL_SECTORS
from src.agents.enhanced_scoring import EnhancedScoring, DimensionalScore


@dataclass
class AgentHealthReport:
    """Health assessment for a single agent."""
    agent_id: str
    status: str  # "healthy" | "warning" | "underperforming" | "replace"
    reason: str
    composite_weight: float
    regime_mismatch: bool = False
    true_decay: bool = False
    is_redundant: bool = False
    redundant_with: str = ""
    n_outcomes: int = 0
    days_since_last_correct: Optional[int] = None
    recommendation: str = ""


@dataclass
class EvolutionAction:
    """Action to take on an agent."""
    action: str  # "keep" | "probation" | "replace" | "retire"
    agent_id: str
    reason: str
    replacement_dna: Optional[AgentDNA] = None


class AgentEvolution:
    """Manages agent lifecycle: monitoring, probation, replacement, and introduction.

    Rules:
        - Agents need 50+ outcomes before they can be flagged for replacement.
        - Probation period: 30 outcomes after first flag before actual replacement.
        - Regime-mismatched agents are NOT replaced -- they're benched until regime fits.
        - Truly decayed agents are replaced with novel-logic agents.
        - Redundant agents are replaced with agents covering gaps in the ensemble.
        - At least 15% of agents should be "challengers" (< 50 outcomes).
    """

    MIN_OUTCOMES_FOR_EVAL = 50
    PROBATION_OUTCOMES = 30
    MIN_CHALLENGER_PCT = 0.15
    REPLACEMENT_WEIGHT_THRESHOLD = 0.3
    REDUNDANCY_THRESHOLD = 0.7  # Jaccard similarity threshold

    def __init__(self, scoring: EnhancedScoring):
        """Initialize agent evolution.

        Args:
            scoring: EnhancedScoring instance used to evaluate agent performance.
        """
        self.scoring = scoring
        self._probation: Dict[str, int] = {}  # agent_id -> outcomes_at_probation_start
        self._retired: Set[str] = set()
        self._regime_benched: Set[str] = set()

    def evaluate_all(
        self,
        all_dnas: Dict[str, AgentDNA],
        current_regime: str = "",
    ) -> List[AgentHealthReport]:
        """Evaluate health of all agents.

        Args:
            all_dnas: Dict mapping agent_id to AgentDNA.
            current_regime: Current market regime label.

        Returns:
            List of AgentHealthReport, one per agent.
        """
        reports = []
        for agent_id, dna in all_dnas.items():
            report = self._evaluate_agent(agent_id, dna, all_dnas, current_regime)
            reports.append(report)
        return reports

    def recommend_actions(
        self,
        reports: List[AgentHealthReport],
        all_dnas: Dict[str, AgentDNA],
    ) -> List[EvolutionAction]:
        """Generate recommended actions based on health reports.

        Args:
            reports: List of AgentHealthReport from evaluate_all().
            all_dnas: Dict mapping agent_id to AgentDNA.

        Returns:
            List of EvolutionAction describing what to do with each agent.
        """
        actions: List[EvolutionAction] = []

        n_total = len(reports)
        n_challengers = sum(1 for r in reports if r.n_outcomes < self.MIN_OUTCOMES_FOR_EVAL)
        challenger_pct = n_challengers / n_total if n_total > 0 else 0

        for report in reports:
            if report.status == "healthy":
                actions.append(EvolutionAction("keep", report.agent_id, "Performing well"))

            elif report.status == "warning":
                if report.agent_id not in self._probation:
                    self._probation[report.agent_id] = report.n_outcomes
                actions.append(EvolutionAction(
                    "probation", report.agent_id, report.reason
                ))

            elif report.status == "replace":
                if report.regime_mismatch:
                    self._regime_benched.add(report.agent_id)
                    actions.append(EvolutionAction(
                        "keep", report.agent_id,
                        "Regime mismatch — benched, not replaced"
                    ))
                elif report.is_redundant:
                    replacement = self._generate_gap_filler(all_dnas)
                    actions.append(EvolutionAction(
                        "replace", report.agent_id,
                        f"Redundant with {report.redundant_with}",
                        replacement_dna=replacement,
                    ))
                elif report.true_decay:
                    replacement = self._generate_novel_agent(all_dnas)
                    actions.append(EvolutionAction(
                        "replace", report.agent_id,
                        "True performance decay",
                        replacement_dna=replacement,
                    ))
                else:
                    actions.append(EvolutionAction(
                        "retire", report.agent_id, report.reason
                    ))

            elif report.status == "underperforming":
                actions.append(EvolutionAction(
                    "probation", report.agent_id, report.reason
                ))

        # Ensure minimum challenger percentage
        if challenger_pct < self.MIN_CHALLENGER_PCT:
            n_needed = int(n_total * self.MIN_CHALLENGER_PCT) - n_challengers
            if n_needed > 0:
                for _ in range(n_needed):
                    new_dna = self._generate_novel_agent(all_dnas)
                    actions.append(EvolutionAction(
                        "replace", "",
                        f"Need more challengers ({challenger_pct:.0%} < {self.MIN_CHALLENGER_PCT:.0%})",
                        replacement_dna=new_dna,
                    ))

        return actions

    def _evaluate_agent(
        self,
        agent_id: str,
        dna: AgentDNA,
        all_dnas: Dict[str, AgentDNA],
        current_regime: str,
    ) -> AgentHealthReport:
        """Evaluate a single agent's health.

        Args:
            agent_id: Unique agent identifier.
            dna: Agent's DNA configuration.
            all_dnas: All agents for redundancy checks.
            current_regime: Current market regime label.

        Returns:
            AgentHealthReport with status and diagnostics.
        """
        score = self.scoring.compute_score(agent_id)
        n = score.n_outcomes

        if n < self.MIN_OUTCOMES_FOR_EVAL:
            return AgentHealthReport(
                agent_id=agent_id, status="healthy",
                reason="Challenger (insufficient data)",
                composite_weight=score.composite_weight,
                n_outcomes=n,
            )

        # Check for true decay
        true_decay = (
            score.composite_weight < self.REPLACEMENT_WEIGHT_THRESHOLD
            and score.stability < 0.3
            and score.calibration_quality < 0.3
        )

        # Check for regime mismatch
        regime_mismatch = (
            score.composite_weight < self.REPLACEMENT_WEIGHT_THRESHOLD
            and score.regime_effectiveness < 0.3
            and score.stability > 0.5  # stable in general, just not now
        )

        # Check for redundancy
        is_redundant, redundant_with = self._check_redundancy(agent_id, all_dnas)

        # Determine status
        if score.composite_weight >= 1.0:
            status = "healthy"
            reason = "Strong performer"
        elif score.composite_weight >= 0.5:
            status = "healthy"
            reason = "Adequate performer"
        elif score.composite_weight >= self.REPLACEMENT_WEIGHT_THRESHOLD:
            status = "warning"
            reason = "Below average — monitoring"
        else:
            # Check if in probation
            if agent_id in self._probation:
                probation_start = self._probation[agent_id]
                if n - probation_start >= self.PROBATION_OUTCOMES:
                    status = "replace"
                    reason = "Failed probation period"
                else:
                    status = "underperforming"
                    reason = f"On probation ({n - probation_start}/{self.PROBATION_OUTCOMES})"
            else:
                status = "underperforming"
                reason = "First flag — entering probation"

        return AgentHealthReport(
            agent_id=agent_id,
            status=status,
            reason=reason,
            composite_weight=score.composite_weight,
            regime_mismatch=regime_mismatch,
            true_decay=true_decay,
            is_redundant=is_redundant,
            redundant_with=redundant_with,
            n_outcomes=n,
        )

    def _check_redundancy(
        self, agent_id: str, all_dnas: Dict[str, AgentDNA]
    ) -> Tuple[bool, str]:
        """Check if agent is redundant with a stronger agent.

        Args:
            agent_id: Agent to check.
            all_dnas: All agents for comparison.

        Returns:
            Tuple of (is_redundant, redundant_with_agent_id).
        """
        my_outcomes = self.scoring._outcomes.get(agent_id, [])
        if not my_outcomes:
            return False, ""

        my_symbols = set(o.symbol for o in my_outcomes)
        my_weight = self.scoring.get_weight(agent_id)

        for other_id in self.scoring._outcomes:
            if other_id == agent_id:
                continue
            other_outcomes = self.scoring._outcomes[other_id]
            other_symbols = set(o.symbol for o in other_outcomes)

            if not other_symbols:
                continue

            union = len(my_symbols | other_symbols)
            if union == 0:
                continue

            jaccard = len(my_symbols & other_symbols) / union
            if jaccard > self.REDUNDANCY_THRESHOLD:
                other_weight = self.scoring.get_weight(other_id)
                if other_weight > my_weight:
                    return True, other_id

        return False, ""

    def _generate_novel_agent(self, existing_dnas: Dict[str, AgentDNA]) -> AgentDNA:
        """Generate a new agent with novel characteristics.

        Picks the least-represented strategy and sector among existing agents
        to maximize ensemble diversity.

        Args:
            existing_dnas: Currently active agents.

        Returns:
            New AgentDNA with underrepresented strategy/sector combination.
        """
        # Find underrepresented strategies and sectors
        strategy_counts: Dict[str, int] = {}
        sector_counts: Dict[str, int] = {}
        for dna in existing_dnas.values():
            strategy_counts[dna.primary_strategy] = strategy_counts.get(dna.primary_strategy, 0) + 1
            for s in dna.primary_sectors:
                sector_counts[s] = sector_counts.get(s, 0) + 1

        # Pick least-represented strategy
        min_strategy = min(VALID_STRATEGIES, key=lambda s: strategy_counts.get(s, 0))

        # Pick least-represented sector
        all_secs = ALL_SECTORS + ["all"]
        min_sector = min(all_secs, key=lambda s: sector_counts.get(s, 0))

        # Random personality variation
        rng = np.random.default_rng()
        agent_id = f"evolved_{min_strategy}_{min_sector}_{rng.integers(1000, 9999)}"

        return AgentDNA(
            agent_id=agent_id,
            display_name=f"Evolved {min_strategy.title()} ({min_sector})",
            primary_sectors=[min_sector],
            primary_strategy=min_strategy,
            risk_appetite=float(rng.uniform(0.2, 0.8)),
            contrarian_factor=float(rng.uniform(0.0, 0.5)),
            conviction_style=float(rng.uniform(0.3, 0.7)),
            regime_sensitivity=float(rng.uniform(0.2, 0.8)),
            lookback_days=int(rng.choice([20, 30, 50, 60, 90, 120])),
            holding_period=str(rng.choice(["scalp", "swing", "position"])),
            min_confidence=float(rng.uniform(0.55, 0.75)),
            max_picks_per_scan=int(rng.integers(3, 7)),
            peer_group=f"{min_strategy}_cluster",
        )

    def _generate_gap_filler(self, existing_dnas: Dict[str, AgentDNA]) -> AgentDNA:
        """Generate an agent that fills a gap in the current ensemble.

        Finds the sector-strategy combination with the fewest existing agents
        and creates a new agent targeting that niche.

        Args:
            existing_dnas: Currently active agents.

        Returns:
            New AgentDNA targeting an underserved sector-strategy combination.
        """
        # Count existing sector-strategy combinations
        combos: Dict[Tuple[str, str], int] = {}
        for dna in existing_dnas.values():
            for s in dna.primary_sectors:
                key = (s, dna.primary_strategy)
                combos[key] = combos.get(key, 0) + 1

        # Generate all possible combos and find the least represented
        all_combos: Dict[Tuple[str, str], int] = {}
        for sector in ALL_SECTORS:
            for strategy in VALID_STRATEGIES:
                key = (sector, strategy)
                all_combos[key] = combos.get(key, 0)

        # Pick the least represented
        min_combo = min(all_combos, key=lambda k: all_combos[k])
        sector, strategy = min_combo

        rng = np.random.default_rng()
        agent_id = f"gapfill_{sector}_{strategy}_{rng.integers(1000, 9999)}"

        return AgentDNA(
            agent_id=agent_id,
            display_name=f"Gap-Fill {strategy.title()} ({sector})",
            primary_sectors=[sector],
            primary_strategy=strategy,
            risk_appetite=float(rng.uniform(0.3, 0.7)),
            contrarian_factor=float(rng.uniform(0.0, 0.3)),
            conviction_style=0.5,
            regime_sensitivity=float(rng.uniform(0.2, 0.6)),
            lookback_days=int(rng.choice([30, 50, 60, 90])),
            holding_period=str(rng.choice(["swing", "position"])),
            min_confidence=0.6,
            max_picks_per_scan=5,
            peer_group=f"{strategy}_cluster",
        )

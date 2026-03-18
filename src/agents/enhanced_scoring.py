"""Enhanced Scoring Model — multi-dimensional agent evaluation.

Scores agents on 10 dimensions beyond simple win/loss:
1. Risk-adjusted return contribution
2. Calibration quality (predicted confidence vs actual outcomes)
3. Reasoning quality proxy (confidence-weighted accuracy)
4. Uniqueness / marginal predictive value
5. Regime-specific effectiveness
6. Drawdown behavior
7. Stability across time
8. False positive rate
9. False negative rate
10. Peer pre-vote accuracy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class DimensionalScore:
    """Scores for a single agent across all dimensions."""
    agent_id: str
    risk_adjusted_return: float = 0.0
    calibration_quality: float = 0.0
    reasoning_quality: float = 0.0
    uniqueness: float = 0.0
    regime_effectiveness: float = 0.0
    drawdown_behavior: float = 0.0
    stability: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    peer_vote_accuracy: float = 0.0
    composite_weight: float = 1.0
    n_outcomes: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "risk_adjusted_return": round(self.risk_adjusted_return, 4),
            "calibration_quality": round(self.calibration_quality, 4),
            "reasoning_quality": round(self.reasoning_quality, 4),
            "uniqueness": round(self.uniqueness, 4),
            "regime_effectiveness": round(self.regime_effectiveness, 4),
            "drawdown_behavior": round(self.drawdown_behavior, 4),
            "stability": round(self.stability, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "peer_vote_accuracy": round(self.peer_vote_accuracy, 4),
            "composite_weight": round(self.composite_weight, 4),
            "n_outcomes": self.n_outcomes,
        }


# Dimension weights for composite score
DIMENSION_WEIGHTS = {
    "risk_adjusted_return": 0.20,
    "calibration_quality": 0.12,
    "reasoning_quality": 0.10,
    "uniqueness": 0.15,
    "regime_effectiveness": 0.10,
    "drawdown_behavior": 0.10,
    "stability": 0.08,
    "false_positive_rate": 0.05,
    "false_negative_rate": 0.05,
    "peer_vote_accuracy": 0.05,
}


@dataclass
class OutcomeRecord:
    """Single outcome record for enhanced scoring."""
    agent_id: str
    symbol: str
    direction: str
    confidence: float
    actual_return: float
    actual_days: int
    was_correct: bool
    timestamp: datetime
    regime_at_time: str = ""
    peer_approved: bool = False
    was_peer_vote_correct: bool = False


class EnhancedScoring:
    """Multi-dimensional agent scoring system.

    Evaluates agents on 10 dimensions and produces a composite weight
    that determines their influence in the voting process.

    Minimum 30 outcomes required before scoring kicks in.
    Weight range: [0.1, 5.0].
    """

    MIN_OUTCOMES = 30
    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 5.0

    def __init__(self, decay_halflife_days: int = 60):
        """Initialize enhanced scoring.

        Args:
            decay_halflife_days: Half-life in days for exponential recency decay.
        """
        self._decay_lambda = np.log(2) / decay_halflife_days
        self._outcomes: Dict[str, List[OutcomeRecord]] = {}
        self._all_outcomes: List[OutcomeRecord] = []

    def record_outcome(
        self,
        agent_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        actual_return: float,
        actual_days: int,
        regime: str = "",
        peer_approved: bool = False,
    ) -> None:
        """Record a single resolved outcome for an agent.

        Args:
            agent_id: Unique agent identifier.
            symbol: Ticker symbol.
            direction: "long" or "short".
            confidence: Agent's confidence at time of pick (0-1).
            actual_return: Realized return of the trade.
            actual_days: Days the trade was held.
            regime: Market regime label at time of pick.
            peer_approved: Whether the pick was approved by peer pre-vote.
        """
        was_correct = (actual_return > 0) if direction == "long" else (actual_return < 0)

        record = OutcomeRecord(
            agent_id=agent_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            actual_return=actual_return,
            actual_days=actual_days,
            was_correct=was_correct,
            timestamp=datetime.now(),
            regime_at_time=regime,
            peer_approved=peer_approved,
            was_peer_vote_correct=peer_approved and was_correct,
        )

        if agent_id not in self._outcomes:
            self._outcomes[agent_id] = []
        self._outcomes[agent_id].append(record)
        self._all_outcomes.append(record)

    def compute_score(self, agent_id: str) -> DimensionalScore:
        """Compute full multi-dimensional score for an agent.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            DimensionalScore with all 10 dimensions and composite weight.
        """
        outcomes = self._outcomes.get(agent_id, [])
        n = len(outcomes)

        score = DimensionalScore(agent_id=agent_id, n_outcomes=n)

        if n < self.MIN_OUTCOMES:
            score.composite_weight = 1.0
            return score

        score.risk_adjusted_return = self._risk_adjusted_return(outcomes)
        score.calibration_quality = self._calibration_quality(outcomes)
        score.reasoning_quality = self._reasoning_quality(outcomes)
        score.uniqueness = self._uniqueness(agent_id)
        score.regime_effectiveness = self._regime_effectiveness(outcomes)
        score.drawdown_behavior = self._drawdown_behavior(outcomes)
        score.stability = self._stability(outcomes)
        score.false_positive_rate = self._false_positive_rate(outcomes)
        score.false_negative_rate = self._false_negative_rate(outcomes)
        score.peer_vote_accuracy = self._peer_vote_accuracy(outcomes)

        # Composite
        raw_composite = sum(
            getattr(score, dim) * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )

        # Normalize to weight range
        recency = self._recency_factor(outcomes)
        weight = raw_composite * 4.0 * recency
        score.composite_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weight))

        return score

    def get_weight(self, agent_id: str) -> float:
        """Get composite weight for an agent.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            Composite weight in [MIN_WEIGHT, MAX_WEIGHT].
        """
        return self.compute_score(agent_id).composite_weight

    def get_all_weights(self) -> Dict[str, float]:
        """Get weights for all tracked agents.

        Returns:
            Dict mapping agent_id to composite weight.
        """
        return {aid: self.get_weight(aid) for aid in self._outcomes}

    def get_leaderboard(self) -> pd.DataFrame:
        """Full leaderboard with all dimension scores.

        Returns:
            DataFrame sorted by composite_weight descending.
        """
        rows = [self.compute_score(aid).to_dict() for aid in self._outcomes]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("composite_weight", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Dimension computations                                              #
    # ------------------------------------------------------------------ #

    def _risk_adjusted_return(self, outcomes: List[OutcomeRecord]) -> float:
        """Sharpe-like risk-adjusted return contribution."""
        returns = [o.actual_return for o in outcomes]
        if len(returns) < 2:
            return 0.5
        avg = np.mean(returns)
        std = np.std(returns)
        sharpe = avg / std if std > 0 else 0
        return float(np.clip(sharpe / 3 + 0.5, 0, 1))

    def _calibration_quality(self, outcomes: List[OutcomeRecord]) -> float:
        """How well does predicted confidence match actual win rate?"""
        if len(outcomes) < 10:
            return 0.5

        # Bin by confidence and check calibration
        bins = [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        total_error = 0.0
        n_bins = 0

        for low, high in bins:
            bin_outcomes = [o for o in outcomes if low <= o.confidence < high]
            if len(bin_outcomes) < 3:
                continue
            predicted = np.mean([o.confidence for o in bin_outcomes])
            actual = np.mean([1.0 if o.was_correct else 0.0 for o in bin_outcomes])
            total_error += abs(predicted - actual)
            n_bins += 1

        if n_bins == 0:
            return 0.5

        avg_error = total_error / n_bins
        # Lower error = better calibration
        return max(0.0, 1.0 - avg_error * 2)

    def _reasoning_quality(self, outcomes: List[OutcomeRecord]) -> float:
        """Proxy: confidence-weighted accuracy (high confidence + correct = good reasoning)."""
        if not outcomes:
            return 0.5

        weighted_correct = sum(
            o.confidence * (1.0 if o.was_correct else -0.5) for o in outcomes
        )
        total_weight = sum(o.confidence for o in outcomes)

        if total_weight == 0:
            return 0.5

        return float(np.clip(weighted_correct / total_weight + 0.5, 0, 1))

    def _uniqueness(self, agent_id: str) -> float:
        """How unique are this agent's picks vs the crowd? (1 - mean Jaccard similarity)."""
        my_outcomes = self._outcomes.get(agent_id, [])
        if not my_outcomes:
            return 1.0

        my_symbols = set(o.symbol for o in my_outcomes)
        similarities: List[float] = []

        for other_id, other_outcomes in self._outcomes.items():
            if other_id == agent_id:
                continue
            other_symbols = set(o.symbol for o in other_outcomes)
            if not other_symbols:
                continue
            jaccard = len(my_symbols & other_symbols) / len(my_symbols | other_symbols)
            similarities.append(jaccard)

        if not similarities:
            return 1.0

        return 1.0 - float(np.mean(similarities))

    def _regime_effectiveness(self, outcomes: List[OutcomeRecord]) -> float:
        """How well does the agent perform in the current regime (proxied by recent outcomes)?"""
        recent = outcomes[-20:] if len(outcomes) >= 20 else outcomes
        if not recent:
            return 0.5
        hit_rate = sum(1 for o in recent if o.was_correct) / len(recent)
        return hit_rate

    def _drawdown_behavior(self, outcomes: List[OutcomeRecord]) -> float:
        """How bad are the agent's worst streaks?"""
        if not outcomes:
            return 0.5

        # Max consecutive losses
        max_streak = 0
        current_streak = 0
        for o in outcomes:
            if not o.was_correct:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Also check worst return
        returns = [o.actual_return for o in outcomes]
        worst = min(returns) if returns else 0

        # Score: fewer consecutive losses + smaller worst loss = better
        streak_score = max(0.0, 1.0 - max_streak / 10.0)
        loss_score = max(0.0, 1.0 + worst * 5.0)  # worst=-0.1 -> 0.5

        return streak_score * 0.6 + loss_score * 0.4

    def _stability(self, outcomes: List[OutcomeRecord]) -> float:
        """How consistent is the agent's performance over time?"""
        if len(outcomes) < 10:
            return 0.5

        # Split into halves and compare
        mid = len(outcomes) // 2
        first_half = outcomes[:mid]
        second_half = outcomes[mid:]

        hr1 = sum(1 for o in first_half if o.was_correct) / len(first_half)
        hr2 = sum(1 for o in second_half if o.was_correct) / len(second_half)

        # Small difference = stable
        diff = abs(hr1 - hr2)
        return max(0.0, 1.0 - diff * 3.0)

    def _false_positive_rate(self, outcomes: List[OutcomeRecord]) -> float:
        """Rate of high-confidence picks that were wrong (lower = better, inverted to score)."""
        high_conf = [o for o in outcomes if o.confidence > 0.7]
        if not high_conf:
            return 0.5

        fp_rate = sum(1 for o in high_conf if not o.was_correct) / len(high_conf)
        return max(0.0, 1.0 - fp_rate)  # invert so higher = better

    def _false_negative_rate(self, outcomes: List[OutcomeRecord]) -> float:
        """Proxy: fraction of low-confidence picks that turned out correct.

        Since we can't measure what wasn't proposed, use low-confidence correct
        picks as a proxy -- agents that frequently pass on winners get penalized.
        """
        low_conf = [o for o in outcomes if o.confidence < 0.6]
        if not low_conf:
            return 0.5

        # Low confidence but still correct = agent is too conservative
        fn_rate = sum(1 for o in low_conf if o.was_correct) / len(low_conf)
        return max(0.0, 1.0 - fn_rate * 0.5)  # mild penalty

    def _peer_vote_accuracy(self, outcomes: List[OutcomeRecord]) -> float:
        """How accurate were picks that received peer pre-vote approval?"""
        peer_voted = [o for o in outcomes if o.peer_approved]
        if not peer_voted:
            return 0.5
        return sum(1 for o in peer_voted if o.was_correct) / len(peer_voted)

    def _recency_factor(self, outcomes: List[OutcomeRecord]) -> float:
        """Exponential decay based on time since last correct prediction."""
        correct = [o for o in outcomes if o.was_correct]
        if not correct:
            return 0.5
        last = max(o.timestamp for o in correct)
        days = (datetime.now() - last).days
        return float(np.exp(-self._decay_lambda * days))

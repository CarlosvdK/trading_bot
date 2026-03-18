"""Agent Scorekeeper — adaptive scoring with recency decay and independence bonus."""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.agents.trading_agent import TradePick


@dataclass
class AgentOutcome:
    """Record of an agent's pick and its actual outcome."""

    agent_id: str
    symbol: str
    direction: str
    confidence: float
    predicted_hold_days: int
    actual_return: float
    actual_days: int
    timestamp: datetime
    was_correct: bool
    peer_pre_vote_correct: bool = False  # did peer pre-vote help accuracy?


class AgentScorekeeper:
    """
    Tracks agent performance and computes adaptive weights.

    Score formula per agent:
        raw_score = (hit_rate * 0.3) + (avg_return * 0.3) + (sharpe * 0.2) + (independence * 0.2)
        weight = raw_score * exp(-lambda * days_since_last_correct)

    New agents start with weight = 1.0. After 20+ outcomes, actual score kicks in.
    Weight is clamped to [0.1, 5.0].
    """

    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 5.0
    MIN_OUTCOMES_FOR_SCORING = 20

    def __init__(self, score_decay_halflife_days: int = 60):
        """
        Args:
            score_decay_halflife_days: Half-life in days for recency decay.
        """
        self.score_decay_halflife_days = score_decay_halflife_days
        self._decay_lambda = np.log(2) / score_decay_halflife_days
        self._outcomes: Dict[str, List[AgentOutcome]] = {}
        self._peer_pre_vote_outcomes: Dict[str, List[bool]] = {}  # tracks peer pre-vote accuracy

    def record_outcome(
        self,
        agent_id: str,
        pick: TradePick,
        actual_return: float,
        actual_days: int,
    ) -> None:
        """
        Record the outcome of a closed trade for an agent.

        Args:
            agent_id: The agent that made the pick.
            pick: The original TradePick.
            actual_return: Realized return (signed, e.g. 0.05 = 5%).
            actual_days: How many days the position was held.
        """
        # A pick is "correct" if direction matches return sign
        if pick.direction == "long":
            was_correct = actual_return > 0
        else:
            was_correct = actual_return < 0

        outcome = AgentOutcome(
            agent_id=agent_id,
            symbol=pick.symbol,
            direction=pick.direction,
            confidence=pick.confidence,
            predicted_hold_days=pick.suggested_hold_days,
            actual_return=actual_return,
            actual_days=actual_days,
            timestamp=pick.timestamp,
            was_correct=was_correct,
            peer_pre_vote_correct=pick.peer_approved and was_correct,
        )

        if agent_id not in self._outcomes:
            self._outcomes[agent_id] = []
        self._outcomes[agent_id].append(outcome)

        # Track peer pre-vote accuracy separately
        if pick.peer_approved:
            if agent_id not in self._peer_pre_vote_outcomes:
                self._peer_pre_vote_outcomes[agent_id] = []
            self._peer_pre_vote_outcomes[agent_id].append(was_correct)

    def record_peer_pre_vote_outcome(
        self,
        agent_id: str,
        was_correct: bool,
    ) -> None:
        """Record whether a peer pre-vote decision was ultimately correct."""
        if agent_id not in self._peer_pre_vote_outcomes:
            self._peer_pre_vote_outcomes[agent_id] = []
        self._peer_pre_vote_outcomes[agent_id].append(was_correct)

    def get_weight(self, agent_id: str) -> float:
        """
        Get the current voting weight for an agent.

        Returns 1.0 for agents with < 20 outcomes (warm-up period).
        """
        outcomes = self._outcomes.get(agent_id, [])
        if len(outcomes) < self.MIN_OUTCOMES_FOR_SCORING:
            return 1.0

        raw_score = self._compute_raw_score(agent_id)
        recency_factor = self._recency_decay(outcomes)

        # Peer pre-vote bonus
        peer_bonus = self._peer_pre_vote_bonus(agent_id)

        weight = (raw_score + peer_bonus) * recency_factor
        return max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weight))

    def get_all_weights(self) -> Dict[str, float]:
        """Get weights for all known agents."""
        agents = set(self._outcomes.keys())
        return {aid: self.get_weight(aid) for aid in agents}

    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get a leaderboard DataFrame with all agent stats.

        Returns:
            DataFrame with columns: agent_id, weight, hit_rate, avg_return,
            sharpe, independence, n_outcomes, last_correct_days_ago,
            peer_pre_vote_accuracy.
        """
        rows = []
        for agent_id in self._outcomes:
            stats = self.get_agent_stats(agent_id)
            stats["agent_id"] = agent_id
            rows.append(stats)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        return df

    def get_agent_stats(self, agent_id: str) -> dict:
        """
        Get detailed stats for a single agent.

        Returns:
            Dict with weight, hit_rate, avg_return, sharpe, independence,
            n_outcomes, last_correct_days_ago, peer_pre_vote_accuracy.
        """
        outcomes = self._outcomes.get(agent_id, [])
        n = len(outcomes)

        if n == 0:
            return {
                "weight": 1.0,
                "hit_rate": 0.0,
                "avg_return": 0.0,
                "sharpe": 0.0,
                "independence": 1.0,
                "n_outcomes": 0,
                "last_correct_days_ago": None,
                "peer_pre_vote_accuracy": 0.0,
            }

        hit_rate = sum(1 for o in outcomes if o.was_correct) / n
        returns = [o.actual_return for o in outcomes]
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1.0
        sharpe = avg_return / std_return if std_return > 0 else 0.0
        independence = self._compute_independence(agent_id)

        # Days since last correct
        correct_outcomes = [o for o in outcomes if o.was_correct]
        if correct_outcomes:
            last_correct = max(o.timestamp for o in correct_outcomes)
            days_ago = (datetime.now() - last_correct).days
        else:
            days_ago = None

        # Peer pre-vote accuracy
        peer_outcomes = self._peer_pre_vote_outcomes.get(agent_id, [])
        peer_accuracy = sum(peer_outcomes) / len(peer_outcomes) if peer_outcomes else 0.0

        return {
            "weight": self.get_weight(agent_id),
            "hit_rate": round(hit_rate, 4),
            "avg_return": round(avg_return, 6),
            "sharpe": round(sharpe, 4),
            "independence": round(independence, 4),
            "n_outcomes": n,
            "last_correct_days_ago": days_ago,
            "peer_pre_vote_accuracy": round(peer_accuracy, 4),
        }

    def save(self, path: str) -> None:
        """Save scorekeeper state to JSON."""
        data = {
            "score_decay_halflife_days": self.score_decay_halflife_days,
            "outcomes": {},
            "peer_pre_vote_outcomes": {},
        }
        for agent_id, outcomes in self._outcomes.items():
            data["outcomes"][agent_id] = [
                {
                    "agent_id": o.agent_id,
                    "symbol": o.symbol,
                    "direction": o.direction,
                    "confidence": o.confidence,
                    "predicted_hold_days": o.predicted_hold_days,
                    "actual_return": o.actual_return,
                    "actual_days": o.actual_days,
                    "timestamp": o.timestamp.isoformat(),
                    "was_correct": o.was_correct,
                    "peer_pre_vote_correct": o.peer_pre_vote_correct,
                }
                for o in outcomes
            ]
        for agent_id, results in self._peer_pre_vote_outcomes.items():
            data["peer_pre_vote_outcomes"][agent_id] = results

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load scorekeeper state from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        self.score_decay_halflife_days = data.get(
            "score_decay_halflife_days", 60
        )
        self._decay_lambda = np.log(2) / self.score_decay_halflife_days

        self._outcomes = {}
        for agent_id, outcomes in data.get("outcomes", {}).items():
            self._outcomes[agent_id] = [
                AgentOutcome(
                    agent_id=o["agent_id"],
                    symbol=o["symbol"],
                    direction=o["direction"],
                    confidence=o["confidence"],
                    predicted_hold_days=o["predicted_hold_days"],
                    actual_return=o["actual_return"],
                    actual_days=o["actual_days"],
                    timestamp=datetime.fromisoformat(o["timestamp"]),
                    was_correct=o["was_correct"],
                    peer_pre_vote_correct=o.get("peer_pre_vote_correct", False),
                )
                for o in outcomes
            ]

        self._peer_pre_vote_outcomes = {}
        for agent_id, results in data.get("peer_pre_vote_outcomes", {}).items():
            self._peer_pre_vote_outcomes[agent_id] = results

    # ------------------------------------------------------------------ #
    #  Internal scoring                                                    #
    # ------------------------------------------------------------------ #

    def _compute_raw_score(self, agent_id: str) -> float:
        """
        Compute raw score from components.

        raw_score = (hit_rate * 0.3) + (avg_return * 0.3) +
                    (sharpe * 0.2) + (independence * 0.2)
        """
        outcomes = self._outcomes.get(agent_id, [])
        if not outcomes:
            return 1.0

        n = len(outcomes)
        hit_rate = sum(1 for o in outcomes if o.was_correct) / n

        returns = [o.actual_return for o in outcomes]
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1.0
        sharpe = avg_return / std_return if std_return > 0 else 0.0

        independence = self._compute_independence(agent_id)

        # Normalize components to roughly 0-1 scale
        norm_hit = hit_rate  # already 0-1
        norm_return = np.clip(avg_return * 10 + 0.5, 0, 1)  # center at 0.5
        norm_sharpe = np.clip(sharpe / 3 + 0.5, 0, 1)  # center at 0.5
        norm_independence = independence  # already 0-1

        raw = (
            norm_hit * 0.3
            + norm_return * 0.3
            + norm_sharpe * 0.2
            + norm_independence * 0.2
        )

        # Scale to weight range (roughly 0.5 to 3.0 before decay)
        return raw * 4.0

    def _recency_decay(self, outcomes: List[AgentOutcome]) -> float:
        """Apply exponential decay based on days since last correct pick."""
        correct = [o for o in outcomes if o.was_correct]
        if not correct:
            return 0.5  # penalty for never being correct

        last_correct = max(o.timestamp for o in correct)
        days_since = (datetime.now() - last_correct).days
        return np.exp(-self._decay_lambda * days_since)

    def _compute_independence(self, agent_id: str) -> float:
        """
        Compute independence score: 1 - avg_correlation_with_other_agents.

        Agents that pick different symbols from the crowd get higher scores.
        """
        my_outcomes = self._outcomes.get(agent_id, [])
        if not my_outcomes:
            return 1.0

        my_symbols = set(o.symbol for o in my_outcomes)
        if not my_symbols:
            return 1.0

        correlations = []
        for other_id, other_outcomes in self._outcomes.items():
            if other_id == agent_id:
                continue
            other_symbols = set(o.symbol for o in other_outcomes)
            if not other_symbols:
                continue

            # Jaccard similarity
            intersection = len(my_symbols & other_symbols)
            union = len(my_symbols | other_symbols)
            if union > 0:
                correlations.append(intersection / union)

        if not correlations:
            return 1.0

        avg_corr = np.mean(correlations)
        return 1.0 - avg_corr

    def _peer_pre_vote_bonus(self, agent_id: str) -> float:
        """
        Compute bonus weight for agents whose peer pre-vote decisions
        were ultimately correct.
        """
        outcomes = self._peer_pre_vote_outcomes.get(agent_id, [])
        if not outcomes or len(outcomes) < 5:
            return 0.0

        accuracy = sum(outcomes) / len(outcomes)
        # Bonus scales from 0 (50% accuracy) to 0.5 (100% accuracy)
        if accuracy > 0.5:
            return (accuracy - 0.5) * 1.0
        return 0.0

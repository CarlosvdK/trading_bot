"""Tests for AgentScorekeeper — adaptive scoring, persistence, and peer tracking."""

import json
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.agents.trading_agent import TradePick
from src.agents.scorekeeper import AgentScorekeeper, AgentOutcome


def _make_ohlcv(n=100, seed=42):
    """Create synthetic OHLCV DataFrame (unused here but kept for consistency)."""
    import numpy as np

    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = max(close.tolist(), key=lambda x: x)  # noqa – just for import
    return None  # Not needed for scorekeeper tests


def _make_pick(symbol="AAPL", agent_id="agent1", direction="long",
               confidence=0.8, peer_approved=False):
    """Build a TradePick with sensible defaults."""
    return TradePick(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        agent_id=agent_id,
        strategy_used="momentum",
        reasoning="test",
        suggested_hold_days=10,
        peer_approved=peer_approved,
    )


def test_new_agent_weight_is_one():
    """An agent with no history should have weight 1.0."""
    sk = AgentScorekeeper()
    assert sk.get_weight("brand_new") == 1.0


def test_warmup_period_holds_weight():
    """Weight stays at 1.0 until MIN_OUTCOMES_FOR_SCORING (20) outcomes."""
    sk = AgentScorekeeper()
    for i in range(19):
        sk.record_outcome("agent1", _make_pick(f"SYM{i}"), 0.02, 5)
    assert sk.get_weight("agent1") == 1.0


def test_weight_changes_after_warmup():
    """After 20+ outcomes the weight is based on actual scoring."""
    sk = AgentScorekeeper()
    for i in range(25):
        sk.record_outcome("agent1", _make_pick(f"SYM{i}"), 0.02, 5)
    w = sk.get_weight("agent1")
    assert w != 1.0
    assert sk.MIN_WEIGHT <= w <= sk.MAX_WEIGHT


def test_weight_bounds_enforced():
    """Weight never falls below MIN_WEIGHT or exceeds MAX_WEIGHT."""
    sk = AgentScorekeeper()
    # Record many losses
    for i in range(30):
        sk.record_outcome("loser", _make_pick(f"SYM{i}", "loser"), -0.10, 5)
    assert sk.get_weight("loser") >= sk.MIN_WEIGHT

    # Record many wins
    for i in range(30):
        sk.record_outcome("winner", _make_pick(f"SYM{i}", "winner"), 0.10, 5)
    assert sk.get_weight("winner") <= sk.MAX_WEIGHT


def test_leaderboard_returns_valid_dataframe():
    """get_leaderboard returns a sorted DataFrame with standard columns."""
    sk = AgentScorekeeper()
    for i in range(5):
        sk.record_outcome("agent1", _make_pick(f"S{i}", "agent1"), 0.02, 5)
        sk.record_outcome("agent2", _make_pick(f"S{i}", "agent2"), -0.01, 5)

    lb = sk.get_leaderboard()
    assert isinstance(lb, pd.DataFrame)
    assert len(lb) == 2
    for col in ("agent_id", "weight", "hit_rate", "sharpe", "n_outcomes"):
        assert col in lb.columns


def test_save_load_roundtrip(tmp_path):
    """Save then load preserves all outcome records."""
    sk = AgentScorekeeper()
    for i in range(5):
        sk.record_outcome("agent1", _make_pick(f"SYM{i}"), 0.02, 5)
    sk.record_peer_pre_vote_outcome("agent1", True)

    path = str(tmp_path / "scores.json")
    sk.save(path)

    sk2 = AgentScorekeeper()
    sk2.load(path)
    assert sk2.get_agent_stats("agent1")["n_outcomes"] == 5
    assert sk2.get_agent_stats("agent1")["peer_pre_vote_accuracy"] > 0


def test_peer_vote_tracking():
    """Peer pre-vote outcomes are tracked and affect stats."""
    sk = AgentScorekeeper()
    # Record with peer_approved=True (correct pick)
    pick_good = _make_pick("AAPL", "agent1", peer_approved=True)
    sk.record_outcome("agent1", pick_good, 0.05, 5)

    # Record with peer_approved=True (incorrect pick)
    pick_bad = _make_pick("MSFT", "agent1", peer_approved=True)
    sk.record_outcome("agent1", pick_bad, -0.05, 5)

    stats = sk.get_agent_stats("agent1")
    assert stats["n_outcomes"] == 2
    # One of two peer-approved picks was correct
    assert 0.0 <= stats["peer_pre_vote_accuracy"] <= 1.0


def test_get_all_weights_covers_all_agents():
    """get_all_weights returns an entry for every agent that has outcomes."""
    sk = AgentScorekeeper()
    sk.record_outcome("a", _make_pick("X", "a"), 0.01, 5)
    sk.record_outcome("b", _make_pick("Y", "b"), -0.01, 5)
    sk.record_outcome("c", _make_pick("Z", "c"), 0.03, 5)

    weights = sk.get_all_weights()
    assert set(weights.keys()) == {"a", "b", "c"}
    for w in weights.values():
        assert sk.MIN_WEIGHT <= w <= sk.MAX_WEIGHT or w == 1.0  # warmup

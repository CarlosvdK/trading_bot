"""Tests for VotingEngine — weighted consensus voting and peer bonuses."""

import pytest
from datetime import datetime

from src.agents.agent_dna import AgentDNA
from src.agents.trading_agent import TradePick
from src.agents.voting_engine import VotingEngine, ApprovedTrade


def _make_ohlcv(n=100, seed=42):
    """Create synthetic OHLCV DataFrame (unused here but kept for consistency)."""
    import numpy as np
    import pandas as pd

    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)


def _make_pick(symbol="AAPL", agent_id="agent1", direction="long",
               confidence=0.8, peer_approved=False, peer_pct=0.0):
    """Build a TradePick with sensible defaults."""
    return TradePick(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        agent_id=agent_id,
        strategy_used="momentum",
        reasoning="test signal",
        suggested_hold_days=10,
        peer_approved=peer_approved,
        peer_approval_pct=peer_pct,
    )


def _make_all_sector_dnas(n=10):
    """Build {agent_id: AgentDNA} for n agents that cover all sectors."""
    return {
        f"agent{i}": AgentDNA(
            agent_id=f"agent{i}",
            display_name=f"Agent {i}",
            primary_sectors=["all"],
            primary_strategy="momentum",
        )
        for i in range(n)
    }


def test_approves_strong_consensus():
    """High consensus with enough voters leads to approval."""
    engine = VotingEngine(approval_threshold=0.5, min_voters=3)
    candidates = {
        "AAPL": [_make_pick("AAPL", f"agent{i}") for i in range(5)],
    }
    weights = {f"agent{i}": 1.0 for i in range(10)}
    approved = engine.run_vote(candidates, weights)

    assert len(approved) == 1
    assert approved[0].symbol == "AAPL"
    assert approved[0].num_voters == 5


def test_rejects_low_consensus():
    """Low approval relative to eligible voters leads to rejection."""
    engine = VotingEngine(approval_threshold=0.8, min_voters=3)
    candidates = {
        "AAPL": [_make_pick("AAPL", f"agent{i}") for i in range(2)],
    }
    weights = {f"agent{i}": 1.0 for i in range(10)}
    dnas = _make_all_sector_dnas(10)
    approved = engine.run_vote(candidates, weights, dnas)

    assert len(approved) == 0


def test_min_voters_enforced():
    """Trades with fewer voters than min_voters are rejected."""
    engine = VotingEngine(approval_threshold=0.1, min_voters=5)
    candidates = {
        "AAPL": [_make_pick("AAPL", f"agent{i}") for i in range(3)],
    }
    weights = {f"agent{i}": 1.0 for i in range(3)}
    approved = engine.run_vote(candidates, weights)

    assert len(approved) == 0


def test_weighted_confidence_favors_heavy_weight():
    """Higher-weighted agents pull confidence toward their value."""
    engine = VotingEngine(approval_threshold=0.3, min_voters=2)
    candidates = {
        "AAPL": [
            _make_pick("AAPL", "heavy", confidence=0.9),
            _make_pick("AAPL", "light", confidence=0.5),
        ],
    }
    weights = {"heavy": 5.0, "light": 1.0}
    approved = engine.run_vote(candidates, weights)

    assert len(approved) == 1
    # Weighted confidence must be closer to 0.9 than 0.5
    assert approved[0].weighted_confidence > 0.8


def test_direction_consensus_by_weight():
    """Direction is determined by the heavier side of the weighted vote."""
    engine = VotingEngine(approval_threshold=0.3, min_voters=2)
    candidates = {
        "AAPL": [
            _make_pick("AAPL", "bull1", direction="long"),
            _make_pick("AAPL", "bull2", direction="long"),
            _make_pick("AAPL", "bear1", direction="short"),
        ],
    }
    weights = {"bull1": 1.0, "bull2": 1.0, "bear1": 1.0}
    approved = engine.run_vote(candidates, weights)

    assert len(approved) == 1
    assert approved[0].direction == "long"


def test_peer_pre_vote_bonus_applied():
    """Peer-approved picks contribute a bonus to approval_pct."""
    engine = VotingEngine(approval_threshold=0.3, min_voters=2, peer_bonus_weight=0.2)
    candidates = {
        "AAPL": [
            _make_pick("AAPL", "a1", peer_approved=True, peer_pct=0.8),
            _make_pick("AAPL", "a2", peer_approved=True, peer_pct=0.9),
            _make_pick("AAPL", "a3", peer_approved=False),
        ],
    }
    weights = {"a1": 1.0, "a2": 1.0, "a3": 1.0}
    approved = engine.run_vote(candidates, weights)

    assert len(approved) == 1
    assert approved[0].peer_pre_vote_bonus > 0


def test_approved_trade_fields_populated():
    """ApprovedTrade dataclass contains all expected fields."""
    trade = ApprovedTrade(
        symbol="AAPL",
        direction="long",
        approval_pct=0.85,
        weighted_confidence=0.78,
        num_voters=7,
        supporting_agents=["a1", "a2"],
        dissenting_agents=["a3"],
        consensus_hold_days=12,
    )
    assert trade.symbol == "AAPL"
    assert trade.approval_pct == 0.85
    assert len(trade.supporting_agents) == 2
    assert trade.avg_raw_score == 0.0
    assert trade.sector == ""


def test_multiple_symbols_independently_voted():
    """Each symbol is voted on independently; one can pass while another fails."""
    engine = VotingEngine(approval_threshold=0.3, min_voters=2)
    candidates = {
        "AAPL": [_make_pick("AAPL", f"a{i}") for i in range(5)],
        "RARE": [_make_pick("RARE", "lonely")],
    }
    weights = {f"a{i}": 1.0 for i in range(5)}
    weights["lonely"] = 1.0
    approved = engine.run_vote(candidates, weights)

    symbols = [t.symbol for t in approved]
    assert "AAPL" in symbols
    assert "RARE" not in symbols  # only 1 voter, needs min 2

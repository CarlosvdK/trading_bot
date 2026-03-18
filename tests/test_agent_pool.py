"""Tests for AgentPool — orchestration, peer pre-voting, and state persistence."""

import numpy as np
import pandas as pd
import pytest

from src.agents.agent_dna import AgentDNA
from src.agents.trading_agent import TradingAgent
from src.agents.agent_pool import AgentPool
from src.agents.voting_engine import VotingEngine
from src.agents.scorekeeper import AgentScorekeeper


def _make_ohlcv(n=100, seed=42):
    """Create synthetic OHLCV DataFrame."""
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


def _make_agents(n=10):
    """Create n test agents with rotating strategies and a peer group."""
    strategies = [
        "momentum", "mean_reversion", "value", "growth",
        "event_driven", "volatility", "sentiment", "breakout",
    ]
    agents = []
    for i in range(n):
        dna = AgentDNA(
            agent_id=f"test_agent_{i}",
            display_name=f"Test Agent {i}",
            primary_sectors=["all"],
            primary_strategy=strategies[i % len(strategies)],
            min_confidence=0.0,
            max_picks_per_scan=3,
            peer_group="test_peers" if i < 5 else "",
        )
        agents.append(TradingAgent(dna))
    return agents


def test_daily_scan_returns_list():
    """daily_scan returns a list of ApprovedTrade objects."""
    pool = AgentPool(
        _make_agents(10),
        VotingEngine(approval_threshold=0.1, min_voters=2),
    )
    data = {f"SYM{i}": _make_ohlcv(seed=i) for i in range(5)}
    approved = pool.daily_scan(data)
    assert isinstance(approved, list)


def test_peer_groups_built_correctly():
    """Peer groups are populated from agent DNA peer_group field."""
    pool = AgentPool(_make_agents(10))
    assert "test_peers" in pool._peer_groups
    assert len(pool._peer_groups["test_peers"]) == 5


def test_record_outcomes_flows_to_scorekeeper():
    """Outcome recording updates the scorekeeper for relevant agents."""
    pool = AgentPool(
        _make_agents(5),
        VotingEngine(approval_threshold=0.01, min_voters=1),
    )
    data = {"AAPL": _make_ohlcv()}
    pool.daily_scan(data)

    fills = [{"symbol": "AAPL", "actual_return": 0.05, "actual_days": 5}]
    pool.record_outcomes(fills)
    # Should not raise; scorekeeper will have recorded if any agent picked AAPL


def test_leaderboard_returns_dataframe():
    """get_leaderboard returns a pandas DataFrame."""
    pool = AgentPool(_make_agents(5))
    lb = pool.get_leaderboard()
    assert isinstance(lb, pd.DataFrame)


def test_save_load_state_roundtrip(tmp_path):
    """State can be saved and loaded without error."""
    pool = AgentPool(
        _make_agents(5),
        VotingEngine(approval_threshold=0.01, min_voters=1),
    )
    data = {"AAPL": _make_ohlcv()}
    pool.daily_scan(data)

    path = str(tmp_path / "pool_state.json")
    pool.save_state(path)
    pool.load_state(path)


def test_scan_stats_populated_after_scan():
    """get_scan_stats returns counts after a successful scan."""
    pool = AgentPool(
        _make_agents(10),
        VotingEngine(approval_threshold=0.01, min_voters=1),
    )
    data = {f"SYM{i}": _make_ohlcv(seed=i) for i in range(5)}
    pool.daily_scan(data)

    stats = pool.get_scan_stats()
    assert "total_picks" in stats
    assert "unique_symbols" in stats
    assert "by_strategy" in stats
    assert stats["total_picks"] > 0


def test_empty_universe_returns_empty():
    """An empty universe produces no approved trades."""
    pool = AgentPool(_make_agents(5))
    approved = pool.daily_scan({})
    assert approved == []

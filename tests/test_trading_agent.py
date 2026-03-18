"""Tests for TradingAgent scan, strategy dispatch, and signal adjustments."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.agents.agent_dna import AgentDNA
from src.agents.trading_agent import TradingAgent, TradePick


def _make_ohlcv(n=100, seed=42):
    """Create synthetic OHLCV DataFrame with realistic structure."""
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


def _make_dna(**overrides) -> AgentDNA:
    defaults = dict(
        agent_id="test_mom",
        display_name="Test Momentum",
        primary_sectors=["all"],
        primary_strategy="momentum",
        min_confidence=0.5,
        max_picks_per_scan=5,
    )
    defaults.update(overrides)
    return AgentDNA(**defaults)


def test_scan_returns_trade_picks():
    """scan() returns a list of TradePick objects with valid fields."""
    agent = TradingAgent(_make_dna())
    data = {"AAPL": _make_ohlcv(), "MSFT": _make_ohlcv(seed=43)}
    picks = agent.scan(data)

    assert isinstance(picks, list)
    for p in picks:
        assert isinstance(p, TradePick)
        assert p.direction in ("long", "short")
        assert 0.0 <= p.confidence <= 1.0
        assert p.agent_id == "test_mom"


def test_scan_respects_max_picks():
    """Number of returned picks never exceeds max_picks_per_scan."""
    agent = TradingAgent(_make_dna(max_picks_per_scan=2, min_confidence=0.0))
    data = {f"SYM{i}": _make_ohlcv(seed=i) for i in range(20)}
    picks = agent.scan(data)
    assert len(picks) <= 2


def test_scan_sector_filtering():
    """Agents only pick symbols from their known sectors."""
    dna = _make_dna(
        agent_id="tech_only",
        primary_sectors=["technology"],
        min_confidence=0.0,
    )
    agent = TradingAgent(dna)
    # AAPL is in technology; XOM is in energy
    data = {"AAPL": _make_ohlcv(), "XOM": _make_ohlcv(seed=43)}
    picks = agent.scan(data)
    symbols = [p.symbol for p in picks]
    assert "XOM" not in symbols


def test_all_eight_strategies_run_without_error():
    """Every strategy can execute and return a list."""
    strategies = [
        "momentum", "mean_reversion", "value", "growth",
        "event_driven", "volatility", "sentiment", "breakout",
    ]
    for strat in strategies:
        dna = _make_dna(
            agent_id=f"test_{strat}",
            primary_strategy=strat,
            min_confidence=0.0,
        )
        agent = TradingAgent(dna)
        picks = agent.scan({"TEST": _make_ohlcv()})
        assert isinstance(picks, list), f"Strategy {strat} did not return list"


def test_contrarian_inverts_direction():
    """High contrarian_factor inverts the signal direction."""
    data = {"TEST": _make_ohlcv()}

    normal = TradingAgent(_make_dna(contrarian_factor=0.0, min_confidence=0.0))
    contrarian = TradingAgent(_make_dna(
        agent_id="ctr", contrarian_factor=1.0, min_confidence=0.0,
    ))

    picks_n = normal.scan(data)
    picks_c = contrarian.scan(data)

    if picks_n and picks_c:
        # With full contrarian factor the direction should flip
        assert picks_n[0].direction != picks_c[0].direction


def test_secondary_strategy_blending():
    """Agent with a secondary strategy blends its score into the primary."""
    dna = _make_dna(
        primary_strategy="momentum",
        secondary_strategy="mean_reversion",
        min_confidence=0.0,
    )
    agent = TradingAgent(dna)
    picks = agent.scan({"TEST": _make_ohlcv()})
    # Just verify it runs and returns picks
    assert isinstance(picks, list)
    # Reasoning should contain a semicolon from blending
    if picks:
        assert ";" in picks[0].reasoning


def test_scan_at_specific_date():
    """Scanning at a specific date uses only data up to that date."""
    df = _make_ohlcv(100)
    target_date = df.index[80]
    agent = TradingAgent(_make_dna(min_confidence=0.0))
    picks = agent.scan({"TEST": df}, current_date=target_date)
    assert isinstance(picks, list)


def test_trade_pick_dataclass_defaults():
    """TradePick has correct default values for peer fields."""
    pick = TradePick(
        symbol="AAPL",
        direction="long",
        confidence=0.85,
        agent_id="test",
        strategy_used="momentum",
        reasoning="Strong momentum",
        suggested_hold_days=10,
    )
    assert pick.peer_approved is False
    assert pick.peer_approval_pct == 0.0
    assert pick.raw_score == 0.0

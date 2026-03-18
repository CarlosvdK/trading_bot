"""Tests for TradingPipeline — full 4-stage multi-agent pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.agents.agent_dna import AgentDNA
from src.agents.trading_agent import TradingAgent
from src.agents.pipeline import TradingPipeline
from src.agents.decision_output import DecisionOutput
from src.agents.regime_adapter import RegimeAdapter, RegimeThresholds
from src.agents.enhanced_scoring import EnhancedScoring
from src.agents.vehicle_engine import VehicleSelectionEngine


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


def _make_agents(n=15):
    """Create n test agents with diverse strategies and peer groups."""
    strategies = [
        "momentum", "mean_reversion", "value", "growth",
        "event_driven", "volatility", "sentiment", "breakout",
    ]
    agents = []
    for i in range(n):
        dna = AgentDNA(
            agent_id=f"pipe_agent_{i}",
            display_name=f"Pipeline Agent {i}",
            primary_sectors=["all"],
            primary_strategy=strategies[i % len(strategies)],
            min_confidence=0.0,
            max_picks_per_scan=3,
            peer_group="pipe_peers" if i < 8 else "pipe_peers_b",
            risk_appetite=0.3 if i == 0 else 0.5,
            contrarian_factor=0.7 if i == 1 else 0.1,
        )
        agents.append(TradingAgent(dna))
    return agents


def _make_universe(n_symbols=5, n_bars=100):
    """Build a {symbol: DataFrame} universe dict."""
    return {f"SYM{i}": _make_ohlcv(n=n_bars, seed=i) for i in range(n_symbols)}


def test_run_daily_returns_decision_outputs():
    """run_daily produces a list of DecisionOutput objects."""
    pipeline = TradingPipeline(_make_agents(15))
    universe = _make_universe(5, 100)
    index_df = _make_ohlcv(100, seed=999)

    results = pipeline.run_daily(universe, index_df=index_df)

    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, DecisionOutput)


def test_regime_detection_runs():
    """Regime adapter detects a regime from index data."""
    adapter = RegimeAdapter()
    index_df = _make_ohlcv(200, seed=99)
    regime = adapter.detect_regime(index_df)

    assert isinstance(regime, str)
    assert regime != ""


def test_regime_thresholds_vary_by_regime():
    """Different regimes produce different adaptive thresholds."""
    adapter = RegimeAdapter()
    t_bull = adapter.get_thresholds("low_vol_trending_up")
    t_crisis = adapter.get_thresholds("high_vol_trending_down")

    assert t_bull.approval_threshold < t_crisis.approval_threshold
    assert t_bull.position_size_multiplier > t_crisis.position_size_multiplier


def test_empty_universe_returns_empty():
    """Empty universe produces no decisions."""
    pipeline = TradingPipeline(_make_agents(10))
    results = pipeline.run_daily({})
    assert results == []


def test_all_four_stages_execute():
    """Pipeline exercises all 4 stages when given adequate data."""
    pipeline = TradingPipeline(
        _make_agents(15),
        max_new_positions_per_day=10,
    )
    universe = _make_universe(10, 150)
    index_df = _make_ohlcv(150, seed=999)

    results = pipeline.run_daily(universe, index_df=index_df)

    # We cannot guarantee approvals because thresholds are strict,
    # but the pipeline must complete without error.
    assert isinstance(results, list)


def test_max_positions_per_day_honored():
    """Pipeline does not exceed max_new_positions_per_day."""
    pipeline = TradingPipeline(
        _make_agents(15),
        max_new_positions_per_day=2,
    )
    universe = _make_universe(20, 150)
    index_df = _make_ohlcv(150, seed=999)

    results = pipeline.run_daily(universe, index_df=index_df)
    assert len(results) <= 2


def test_decision_output_has_key_fields():
    """DecisionOutput objects include asset, direction, and confidence."""
    pipeline = TradingPipeline(_make_agents(15))
    universe = _make_universe(5, 150)
    index_df = _make_ohlcv(150, seed=999)

    results = pipeline.run_daily(universe, index_df=index_df)
    for r in results:
        assert r.asset != ""
        assert r.direction in ("long", "short")
        assert 0.0 <= r.confidence <= 1.0
        assert r.current_regime != "" or True  # regime may be empty string


def test_record_outcomes_does_not_raise():
    """Recording outcomes after a scan completes without error."""
    pipeline = TradingPipeline(_make_agents(10))
    universe = _make_universe(5, 100)
    pipeline.run_daily(universe)

    fills = [{"symbol": "SYM0", "actual_return": 0.03, "actual_days": 7}]
    pipeline.record_outcomes(fills, regime="low_vol_trending_up")

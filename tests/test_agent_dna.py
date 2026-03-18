"""Tests for AgentDNA configuration, validation, and the full 121-agent roster."""

import pytest
from src.agents.agent_dna import AgentDNA, VALID_STRATEGIES, VALID_HOLDING_PERIODS, ALL_SECTORS
from src.agents.agent_definitions import ALL_AGENTS, build_all_agents


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


def _make_dna(**overrides) -> AgentDNA:
    """Build an AgentDNA with sensible defaults, easy to override."""
    defaults = dict(
        agent_id="test_agent",
        display_name="Test Agent",
        primary_sectors=["technology"],
        secondary_sectors=["healthcare"],
        primary_strategy="momentum",
        risk_appetite=0.5,
        contrarian_factor=0.1,
        conviction_style=0.5,
        regime_sensitivity=0.3,
        lookback_days=50,
        holding_period="swing",
        min_confidence=0.6,
        max_picks_per_scan=5,
        peer_group="test_cluster",
    )
    defaults.update(overrides)
    return AgentDNA(**defaults)


def test_agent_dna_creation_defaults():
    """Basic creation uses correct default values."""
    dna = _make_dna()
    assert dna.agent_id == "test_agent"
    assert dna.risk_appetite == 0.5
    assert dna.holding_period == "swing"
    assert dna.min_confidence == 0.6
    assert dna.max_picks_per_scan == 5


def test_knows_sector_primary_and_secondary():
    """knows_sector returns True for primary and secondary sectors, False otherwise."""
    dna = _make_dna(primary_sectors=["technology"], secondary_sectors=["energy"])
    assert dna.knows_sector("technology") is True
    assert dna.knows_sector("energy") is True
    assert dna.knows_sector("financials") is False


def test_sector_weight_correct_tiers():
    """sector_weight returns 1.0 for primary, 0.5 for secondary, 0.0 for unknown."""
    dna = _make_dna(primary_sectors=["technology"], secondary_sectors=["financials"])
    assert dna.sector_weight("technology") == 1.0
    assert dna.sector_weight("financials") == 0.5
    assert dna.sector_weight("energy") == 0.0


def test_holding_days_range_all_periods():
    """Each holding_period maps to the correct (min, max) day range."""
    expected = {
        "scalp": (1, 3),
        "swing": (5, 15),
        "position": (30, 90),
        "macro": (90, 252),
    }
    for period, days_range in expected.items():
        lb = 10 if period == "scalp" else 50
        dna = _make_dna(holding_period=period, lookback_days=lb)
        assert dna.holding_days_range == days_range, f"Mismatch for {period}"


def test_all_sectors_property_combines_primary_secondary():
    """all_sectors returns the union of primary and secondary sectors."""
    dna = _make_dna(
        primary_sectors=["technology", "healthcare"],
        secondary_sectors=["energy"],
    )
    assert set(dna.all_sectors) == {"technology", "healthcare", "energy"}


def test_validation_rejects_invalid_params():
    """Post-init validation catches out-of-range personality parameters."""
    with pytest.raises(AssertionError):
        _make_dna(risk_appetite=1.5)
    with pytest.raises(AssertionError):
        _make_dna(contrarian_factor=-0.1)
    with pytest.raises(AssertionError):
        _make_dna(primary_strategy="invalid_strategy")
    with pytest.raises(AssertionError):
        _make_dna(lookback_days=3)
    with pytest.raises(AssertionError):
        _make_dna(holding_period="ultra_short")


def test_all_121_agents_load_correctly():
    """All 121 agents in the roster build without error and have unique IDs."""
    agents = ALL_AGENTS
    assert len(agents) == 121

    ids = [a.agent_id for a in agents]
    assert len(ids) == len(set(ids)), "Duplicate agent_id found"

    for dna in agents:
        assert dna.primary_strategy in VALID_STRATEGIES
        assert dna.holding_period in VALID_HOLDING_PERIODS
        assert 0.0 <= dna.risk_appetite <= 1.0
        assert 5 <= dna.lookback_days <= 252


def test_valid_strategies_complete():
    """All 8 documented strategies exist in VALID_STRATEGIES."""
    assert len(VALID_STRATEGIES) == 8
    for s in ["momentum", "mean_reversion", "value", "growth",
              "event_driven", "volatility", "sentiment", "breakout"]:
        assert s in VALID_STRATEGIES

"""Tests for VehicleSelectionEngine — candidate generation, scoring, and regime adjustments."""

import numpy as np
import pandas as pd
import pytest

from src.agents.proposal import (
    Proposal,
    VehicleCandidate,
    InvestmentType,
    ProposalRisks,
)
from src.agents.vehicle_engine import VehicleSelectionEngine, REGIME_VEHICLE_PREFS


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


def _make_proposal(**overrides) -> Proposal:
    """Build a Proposal with sensible defaults."""
    defaults = dict(
        proposal_id="test_proposal",
        agent_id="test_agent",
        symbol="AAPL",
        sector="technology",
        direction="long",
        confidence=0.75,
        expected_edge_bps=50.0,
        time_horizon_days=10,
        expected_annual_vol=25.0,
        avg_daily_volume=5_000_000.0,
        realized_vol=20.0,
        implied_vol=0.0,
    )
    defaults.update(overrides)
    return Proposal(**defaults)


def test_generates_candidates():
    """select_vehicle populates vehicle_candidates on the proposal."""
    engine = VehicleSelectionEngine()
    proposal = _make_proposal()
    result = engine.select_vehicle(proposal)

    assert len(result.vehicle_candidates) >= 3
    assert result.selected_vehicle is not None


def test_selects_best_candidate():
    """Selected vehicle is the candidate with the highest composite score."""
    engine = VehicleSelectionEngine()
    proposal = _make_proposal()
    result = engine.select_vehicle(proposal)

    scores = [c.composite_score for c in result.vehicle_candidates]
    assert result.selected_vehicle.composite_score == max(scores)


def test_no_trade_for_weak_signal():
    """Low confidence should make no_trade the winning candidate."""
    engine = VehicleSelectionEngine()
    proposal = _make_proposal(confidence=0.1, expected_edge_bps=2.0)
    result = engine.select_vehicle(proposal)

    assert result.selected_vehicle is not None
    assert result.selected_vehicle.vehicle_type == InvestmentType.NO_TRADE


def test_always_includes_no_trade_candidate():
    """The no_trade candidate is always among the generated candidates."""
    engine = VehicleSelectionEngine()
    proposal = _make_proposal()
    result = engine.select_vehicle(proposal)

    types = [c.vehicle_type for c in result.vehicle_candidates]
    assert InvestmentType.NO_TRADE in types


def test_options_candidates_when_enabled():
    """With options_available=True, option candidates are generated."""
    engine = VehicleSelectionEngine(options_available=True)
    proposal = _make_proposal(direction="long")
    result = engine.select_vehicle(proposal)

    types = {c.vehicle_type for c in result.vehicle_candidates}
    assert InvestmentType.CALL_OPTION in types
    assert InvestmentType.CALL_SPREAD in types


def test_regime_adjustments_change_scores():
    """Different regimes produce different composite scores for the same proposal."""
    engine = VehicleSelectionEngine()
    proposal_bull = _make_proposal()
    proposal_bear = _make_proposal()

    result_bull = engine.select_vehicle(proposal_bull, regime="low_vol_trending_up")
    result_bear = engine.select_vehicle(proposal_bear, regime="high_vol_trending_down")

    # Scores should differ because regime biases are different
    scores_bull = {
        c.vehicle_type: c.composite_score for c in result_bull.vehicle_candidates
    }
    scores_bear = {
        c.vehicle_type: c.composite_score for c in result_bear.vehicle_candidates
    }
    # At least shares_spot should have different adjusted scores
    assert scores_bull.get(InvestmentType.SHARES_SPOT) != scores_bear.get(
        InvestmentType.SHARES_SPOT
    )


def test_vehicle_rationale_populated():
    """A human-readable rationale is set after vehicle selection."""
    engine = VehicleSelectionEngine()
    proposal = _make_proposal(confidence=0.8, expected_edge_bps=100)
    result = engine.select_vehicle(proposal)

    assert result.vehicle_rationale != ""
    assert "Selected:" in result.vehicle_rationale


def test_pairs_trade_candidate_included():
    """Pairs trade candidate is always generated for long or short proposals."""
    engine = VehicleSelectionEngine()
    for direction in ("long", "short"):
        proposal = _make_proposal(direction=direction)
        result = engine.select_vehicle(proposal)
        types = [c.vehicle_type for c in result.vehicle_candidates]
        assert InvestmentType.PAIRS_TRADE in types

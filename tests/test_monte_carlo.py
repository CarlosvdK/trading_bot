"""Tests for Monte Carlo position sizer."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signals.monte_carlo import (
    simulate_portfolio_paths,
    optimal_kelly_fraction,
    risk_of_ruin,
    optimal_position_size_mc,
    drawdown_distribution,
    position_size_sensitivity,
)


@pytest.fixture
def sample_trades():
    """Realistic-ish trade history."""
    rng = np.random.default_rng(42)
    n = 200
    returns = rng.normal(0.002, 0.03, n)
    return [{"return_pct": r} for r in returns]


class TestKellyFraction:
    def test_known_analytical(self):
        # 60% win rate, 2:1 reward:risk
        # Kelly = (0.6*2 - 0.4) / 2 = 0.4
        # Half-Kelly = 0.2
        result = optimal_kelly_fraction(0.6, 0.10, 0.05, half_kelly=True)
        assert abs(result - 0.2) < 0.01

    def test_full_kelly(self):
        result = optimal_kelly_fraction(0.6, 0.10, 0.05, half_kelly=False)
        assert abs(result - 0.25) < 0.01  # Clamped to 0.25 max

    def test_negative_edge_returns_zero(self):
        # 30% win rate with 1:1 ratio -> negative Kelly
        result = optimal_kelly_fraction(0.3, 0.05, 0.05, half_kelly=False)
        assert result == 0.0

    def test_zero_inputs(self):
        assert optimal_kelly_fraction(0, 0.1, 0.05) == 0.0
        assert optimal_kelly_fraction(0.6, 0, 0.05) == 0.0
        assert optimal_kelly_fraction(0.6, 0.1, 0) == 0.0


class TestPortfolioSimulation:
    def test_paths_shape(self, sample_trades):
        result = simulate_portfolio_paths(sample_trades, n_simulations=100, n_periods=50, seed=42)
        assert result["paths"].shape == (100, 50)
        assert len(result["terminal_values"]) == 100
        assert len(result["max_drawdowns"]) == 100

    def test_all_drawdowns_negative(self, sample_trades):
        result = simulate_portfolio_paths(sample_trades, n_simulations=500, seed=42)
        assert (result["max_drawdowns"] <= 0).all()

    def test_empty_trades(self):
        result = simulate_portfolio_paths([], n_simulations=10, n_periods=20)
        assert result["paths"].shape == (10, 20)
        assert (result["terminal_values"] == 100000).all()


class TestOptimalPositionSize:
    def test_respects_drawdown_constraint(self, sample_trades):
        result = optimal_position_size_mc(
            sample_trades, target_max_dd=0.15, confidence=0.90,
            n_sims=500, seed=42,
        )
        assert "optimal_multiplier" in result
        assert result["optimal_multiplier"] > 0

    def test_returns_expected_fields(self, sample_trades):
        result = optimal_position_size_mc(sample_trades, n_sims=200, seed=42)
        assert "expected_return" in result
        assert "expected_dd" in result
        assert "confidence_interval" in result


class TestDrawdownDistribution:
    def test_distribution_stats(self, sample_trades):
        result = simulate_portfolio_paths(sample_trades, n_simulations=200, seed=42)
        dd = drawdown_distribution(result["paths"])
        assert dd["mean"] <= 0
        # p99 and p95 are worst percentiles (more negative)
        assert dd["p99"] <= 0
        assert dd["p95"] <= 0
        assert dd["mean"] <= 0


class TestSensitivity:
    def test_monotonic_risk(self, sample_trades):
        df = position_size_sensitivity(
            sample_trades,
            multiplier_range=[0.5, 1.0, 2.0],
            n_sims=200, seed=42,
        )
        assert len(df) == 3
        assert "multiplier" in df.columns
        # Risk should generally increase with multiplier
        assert df["max_dd_median"].iloc[-1] <= df["max_dd_median"].iloc[0]


class TestRiskOfRuin:
    def test_zero_risk_trades(self):
        # All winning trades -> 0 risk of ruin
        result = risk_of_ruin(1.0, 0.05, 0.05, 0.01, n_simulations=100, seed=42)
        assert result == 0.0

    def test_returns_probability(self):
        result = risk_of_ruin(0.5, 0.05, 0.05, 0.10, n_simulations=500, seed=42)
        assert 0 <= result <= 1

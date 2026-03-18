"""Tests for ensemble signal framework."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signals.ensemble import (
    mean_reversion_candidates,
    volume_anomaly_candidates,
    gap_and_go_candidates,
    relative_strength_candidates,
    EnsembleSignalGenerator,
)


def _make_price_df(n=100, base=100, seed=42):
    """Create a simple OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = base * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "high": close * (1 + abs(rng.normal(0, 0.01, n))),
        "low": close * (1 - abs(rng.normal(0, 0.01, n))),
        "close": close,
        "volume": rng.integers(500000, 2000000, n),
    }, index=dates)


class TestSignalFormat:
    """All signal generators should produce valid output format."""

    def test_mean_reversion_format(self):
        # Create data with a drop then recovery
        prices = {"AAPL": _make_price_df(100, seed=42)}
        config = {"mr_lookback_days": 10, "mr_drop_threshold": -0.15, "mr_volume_surge": 0.5}
        date = prices["AAPL"].index[50]
        result = mean_reversion_candidates(prices, date, config)
        assert isinstance(result, list)
        for r in result:
            assert "symbol" in r
            assert "signal_type" in r
            assert "direction" in r

    def test_volume_anomaly_format(self):
        prices = {"AAPL": _make_price_df(100, seed=42)}
        config = {"va_volume_threshold": 0.5, "va_price_range_pct": 1.0}
        date = prices["AAPL"].index[50]
        result = volume_anomaly_candidates(prices, date, config)
        assert isinstance(result, list)

    def test_gap_and_go_format(self):
        prices = {"AAPL": _make_price_df(100, seed=42)}
        config = {"gg_min_gap_pct": 0.001, "gg_volume_surge": 0.5}
        date = prices["AAPL"].index[50]
        result = gap_and_go_candidates(prices, date, config)
        assert isinstance(result, list)

    def test_relative_strength_format(self):
        prices = {"AAPL": _make_price_df(100, seed=42)}
        index_df = _make_price_df(100, seed=99)
        config = {"rs_window": 21, "rs_accel_window": 5, "rs_min_outperformance": -1.0}
        date = prices["AAPL"].index[50]
        result = relative_strength_candidates(prices, index_df, date, config)
        assert isinstance(result, list)


class TestEnsembleScorer:
    def test_deduplication(self):
        """Same symbol from multiple signals should be deduplicated."""
        config = {
            "momentum_threshold_pct": 0.001,
            "volume_surge_min": 0.1,
            "vol_expansion_ratio": 0.1,
        }
        ensemble = EnsembleSignalGenerator(config)
        prices = {"AAPL": _make_price_df(100, seed=42)}
        date = prices["AAPL"].index[50]
        results = ensemble.generate(prices, date)
        symbols = [r["symbol"] for r in results]
        # No duplicate symbols
        assert len(symbols) == len(set(symbols))

    def test_ensemble_score_present(self):
        config = {"momentum_threshold_pct": 0.001, "volume_surge_min": 0.1}
        ensemble = EnsembleSignalGenerator(config)
        prices = {"AAPL": _make_price_df(100, seed=42)}
        date = prices["AAPL"].index[50]
        results = ensemble.generate(prices, date)
        for r in results:
            assert "ensemble_score" in r
            assert 0 <= r["ensemble_score"] <= 1

    def test_empty_data(self):
        config = {}
        ensemble = EnsembleSignalGenerator(config)
        results = ensemble.generate({}, pd.Timestamp("2023-06-01"))
        assert results == []

    def test_sorted_by_score(self):
        config = {"momentum_threshold_pct": 0.001, "volume_surge_min": 0.1}
        prices = {f"SYM{i}": _make_price_df(100, seed=i) for i in range(5)}
        ensemble = EnsembleSignalGenerator(config)
        date = list(prices.values())[0].index[50]
        results = ensemble.generate(prices, date)
        if len(results) > 1:
            scores = [r["ensemble_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

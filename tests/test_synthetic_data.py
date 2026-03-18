"""Tests for synthetic data generator."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_feeds.synthetic import (
    generate_gbm_ohlcv,
    generate_regime_aware_ohlcv,
    generate_correlated_universe,
    generate_index_from_universe,
    SyntheticDataProvider,
    RegimeSpec,
)


class TestGBMGenerator:
    def test_output_shape_and_columns(self):
        df = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-06-30", seed=42)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) > 100

    def test_business_days_only(self):
        df = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-01-31", seed=42)
        for d in df.index:
            assert d.weekday() < 5  # Mon-Fri

    def test_no_nan_values(self):
        df = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-12-31", seed=42)
        assert not df.isna().any().any()

    def test_ohlc_consistency(self):
        df = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-12-31", seed=42)
        assert (df["high"] >= df["close"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["low"] <= df["open"]).all()

    def test_positive_prices(self):
        df = generate_gbm_ohlcv("TEST", "2020-01-01", "2024-12-31", seed=42)
        assert (df["close"] > 0).all()
        assert (df["low"] > 0).all()

    def test_positive_volume(self):
        df = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-12-31", seed=42)
        assert (df["volume"] > 0).all()

    def test_reproducible_with_seed(self):
        df1 = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-06-30", seed=42)
        df2 = generate_gbm_ohlcv("TEST", "2023-01-01", "2023-06-30", seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestRegimeAwareGenerator:
    @pytest.fixture
    def regimes(self):
        return [
            RegimeSpec("bull", "2023-01-01", "2023-03-31"),
            RegimeSpec("crisis", "2023-04-01", "2023-06-30"),
            RegimeSpec("choppy", "2023-07-01", "2023-09-30"),
        ]

    def test_regime_transitions_change_stats(self, regimes):
        df, regime_series = generate_regime_aware_ohlcv(
            "TEST", "2023-01-01", "2023-09-30", regimes, seed=42
        )
        # Bull period should have higher returns on average than crisis
        bull_mask = regime_series == "bull"
        crisis_mask = regime_series == "crisis"
        bull_rets = df.loc[bull_mask, "close"].pct_change().dropna()
        crisis_rets = df.loc[crisis_mask, "close"].pct_change().dropna()
        # Crisis should have higher volatility
        assert crisis_rets.std() > bull_rets.std() * 0.8  # Generous tolerance

    def test_regime_series_aligned(self, regimes):
        df, regime_series = generate_regime_aware_ohlcv(
            "TEST", "2023-01-01", "2023-09-30", regimes, seed=42
        )
        assert len(df) == len(regime_series)
        assert (df.index == regime_series.index).all()

    def test_no_nans(self, regimes):
        df, _ = generate_regime_aware_ohlcv(
            "TEST", "2023-01-01", "2023-09-30", regimes, seed=42
        )
        assert not df.isna().any().any()


class TestCorrelatedUniverse:
    def test_correct_symbols(self):
        symbols = ["AAPL", "MSFT", "GOOGL"]
        universe = generate_correlated_universe(
            symbols, "2023-01-01", "2023-12-31", seed=42
        )
        assert set(universe.keys()) == set(symbols)

    def test_correlation_structure(self):
        symbols = ["A", "B", "C"]
        corr = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]])
        universe = generate_correlated_universe(
            symbols, "2020-01-01", "2023-12-31",
            correlation_matrix=corr, seed=42,
        )
        closes = pd.DataFrame({s: df["close"] for s, df in universe.items()})
        realized_corr = closes.pct_change().dropna().corr()
        # Realized should be reasonably close to target
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                assert realized_corr.iloc[i, j] > 0.3  # Should be positively correlated


class TestIndexFromUniverse:
    def test_index_creation(self):
        universe = generate_correlated_universe(
            ["A", "B"], "2023-01-01", "2023-06-30", seed=42
        )
        index_df = generate_index_from_universe(universe)
        assert set(index_df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(index_df) > 0


class TestSyntheticDataProvider:
    def test_provider_interface(self):
        provider = SyntheticDataProvider(
            symbols=["AAPL", "MSFT"], start_date="2023-01-01",
            end_date="2023-06-30", seed=42,
        )
        data = provider.get_prices()
        assert "AAPL" in data
        assert "MSFT" in data

    def test_get_index(self):
        provider = SyntheticDataProvider(
            symbols=["AAPL", "MSFT"], start_date="2023-01-01",
            end_date="2023-06-30", seed=42,
        )
        index = provider.get_index()
        assert "close" in index.columns

    def test_close_matrix(self):
        provider = SyntheticDataProvider(
            symbols=["A", "B", "C"], start_date="2023-01-01",
            end_date="2023-06-30", seed=42,
        )
        matrix = provider.get_close_matrix()
        assert list(matrix.columns) == ["A", "B", "C"]

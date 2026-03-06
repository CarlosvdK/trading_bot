"""Tests for sector & breadth features — no future leakage."""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import build_features


def make_df(n=300, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.003),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(100_000, 10_000_000, n),
        },
        index=dates,
    )


def make_universe_closes(n=300, n_stocks=10) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = {}
    for i in range(n_stocks):
        np.random.seed(100 + i)
        c = 50 + np.cumsum(np.random.randn(n) * 0.3)
        closes[f"SYM{i}"] = np.maximum(c, 5)
    return pd.DataFrame(closes, index=dates)


class TestSectorFeatures:
    def test_sector_features_present_when_data_provided(self):
        sym = make_df()
        idx = make_df(seed=10)
        sector = make_df(seed=20)
        feats = build_features(sym, idx, {}, sector_etf_df=sector)
        assert "sector_ret_21d" in feats.columns
        assert "sector_rel_ret" in feats.columns

    def test_sector_features_absent_when_no_data(self):
        sym = make_df()
        idx = make_df(seed=10)
        feats = build_features(sym, idx, {})
        assert "sector_ret_21d" not in feats.columns
        assert "sector_rel_ret" not in feats.columns

    def test_no_future_leakage_in_sector_features(self):
        sym = make_df(300)
        idx = make_df(300, seed=10)
        sector = make_df(300, seed=20)

        feats_orig = build_features(sym, idx, {}, sector_etf_df=sector)

        # Modify sector data after index 200
        sector_mod = sector.copy()
        sector_mod.iloc[200:, sector_mod.columns.get_loc("close")] *= 2

        feats_mod = build_features(sym, idx, {}, sector_etf_df=sector_mod)

        # Features at index 199 should be identical
        check_date = sym.index[199]
        pd.testing.assert_series_equal(
            feats_orig.loc[check_date].dropna(),
            feats_mod.loc[check_date].dropna(),
            check_names=False,
        )


class TestBreadthFeatures:
    def test_breadth_features_present_when_data_provided(self):
        sym = make_df()
        idx = make_df(seed=10)
        uc = make_universe_closes()
        feats = build_features(sym, idx, {}, universe_closes=uc)
        assert "breadth_pct_above_ma50" in feats.columns
        assert "breadth_advance_decline" in feats.columns

    def test_breadth_features_absent_when_no_data(self):
        sym = make_df()
        idx = make_df(seed=10)
        feats = build_features(sym, idx, {})
        assert "breadth_pct_above_ma50" not in feats.columns
        assert "breadth_advance_decline" not in feats.columns

    def test_breadth_pct_above_ma50_bounded(self):
        sym = make_df()
        idx = make_df(seed=10)
        uc = make_universe_closes()
        feats = build_features(sym, idx, {}, universe_closes=uc)
        vals = feats["breadth_pct_above_ma50"].dropna()
        # Should be between 0 and 1 (percentage)
        assert vals.min() >= 0
        assert vals.max() <= 1

    def test_no_future_leakage_in_breadth_features(self):
        sym = make_df(300)
        idx = make_df(300, seed=10)
        uc = make_universe_closes(300)

        feats_orig = build_features(sym, idx, {}, universe_closes=uc)

        # Modify universe closes after index 200
        uc_mod = uc.copy()
        uc_mod.iloc[200:] *= 3

        feats_mod = build_features(sym, idx, {}, universe_closes=uc_mod)

        # Features at index 199 should be identical
        check_date = sym.index[199]
        pd.testing.assert_series_equal(
            feats_orig.loc[check_date].dropna(),
            feats_mod.loc[check_date].dropna(),
            check_names=False,
        )


class TestCombinedFeatures:
    def test_all_features_together(self):
        sym = make_df()
        idx = make_df(seed=10)
        sector = make_df(seed=20)
        uc = make_universe_closes()

        feats = build_features(
            sym, idx, {},
            sector_etf_df=sector,
            universe_closes=uc,
        )

        # Original features still present
        assert "ret_5d" in feats.columns
        assert "rsi_14" in feats.columns

        # New features present
        assert "sector_ret_21d" in feats.columns
        assert "sector_rel_ret" in feats.columns
        assert "breadth_pct_above_ma50" in feats.columns
        assert "breadth_advance_decline" in feats.columns

        # Total feature count: 22 original + 4 new = 26
        assert len(feats.columns) >= 26

    def test_backward_compatible_without_new_data(self):
        """Calling with no sector/breadth data produces same output as before."""
        sym = make_df()
        idx = make_df(seed=10)
        feats = build_features(sym, idx, {})
        # Should have original features only, no sector/breadth
        assert "sector_ret_21d" not in feats.columns
        assert "breadth_pct_above_ma50" not in feats.columns
        assert "ret_5d" in feats.columns

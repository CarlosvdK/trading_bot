"""
Tests for feature engineering — no future data leakage.
Skill reference: .claude/skills/feature-engineering/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.ml.features import build_features, winsorize_zscore


def make_symbol_df(n=300) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(42)
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


def make_index_df(n=300) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(123)
    close = 400 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({"close": close}, index=dates)


class TestBuildFeatures:
    def test_returns_dataframe(self):
        sym = make_symbol_df()
        idx = make_index_df()
        feats = build_features(sym, idx, {})
        assert isinstance(feats, pd.DataFrame)
        assert len(feats) == len(sym)

    def test_expected_columns_present(self):
        sym = make_symbol_df()
        idx = make_index_df()
        feats = build_features(sym, idx, {})
        expected = [
            "ret_5d",
            "ret_10d",
            "ret_21d",
            "vol_5d",
            "vol_21d",
            "vol_ratio_5_21",
            "gap_return",
            "volume_surprise",
            "mom_consistency_10d",
            "rsi_14",
            "macd_hist",
            "bband_pctb",
            "dv_momentum",
            "atr_ratio",
            "dist_ma50_pct",
        ]
        for col in expected:
            assert col in feats.columns, f"Missing column: {col}"

    def test_no_future_data_in_features(self):
        """
        Features at time t should only use data up to t.
        Test: modifying future data should not change current features.
        """
        sym = make_symbol_df(300)
        idx = make_index_df(300)

        feats_original = build_features(sym, idx, {})

        # Modify data after index 200
        sym_modified = sym.copy()
        sym_modified.iloc[200:, sym_modified.columns.get_loc("close")] *= 2

        feats_modified = build_features(sym_modified, idx, {})

        # Features at index 199 should be identical
        check_date = sym.index[199]
        pd.testing.assert_series_equal(
            feats_original.loc[check_date].dropna(),
            feats_modified.loc[check_date].dropna(),
            check_names=False,
        )

    def test_features_align_with_input_index(self):
        sym = make_symbol_df()
        idx = make_index_df()
        feats = build_features(sym, idx, {})
        assert feats.index.equals(sym.index)

    def test_custom_windows(self):
        sym = make_symbol_df()
        idx = make_index_df()
        config = {"return_windows": [3, 7], "vol_windows": [10]}
        feats = build_features(sym, idx, config)
        assert "ret_3d" in feats.columns
        assert "ret_7d" in feats.columns
        assert "vol_10d" in feats.columns


class TestWinsorizeZscore:
    def test_clips_extremes(self):
        np.random.seed(42)
        n = 500
        data = pd.DataFrame(
            {"feat": np.random.randn(n)},
            index=pd.bdate_range("2020-01-01", periods=n),
        )
        # Add extreme outlier
        data.iloc[-1, 0] = 100.0

        result = winsorize_zscore(data, window=252, clip_sigma=3.0)
        # The z-score of the extreme should be clipped to 3
        assert result["feat"].dropna().max() <= 3.0
        assert result["feat"].dropna().min() >= -3.0

    def test_returns_same_shape(self):
        n = 300
        data = pd.DataFrame(
            {"a": np.random.randn(n), "b": np.random.randn(n)},
            index=pd.bdate_range("2020-01-01", periods=n),
        )
        result = winsorize_zscore(data)
        assert result.shape == data.shape

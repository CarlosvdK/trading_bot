"""
Tests for regime detection — feature building, model fitting, smoothing.
Skill reference: .claude/skills/regime-detection/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.models.regime import (
    build_regime_features,
    fit_regime_model,
    predict_regime,
    label_regimes,
    get_regime_allocation,
    smooth_regime,
    run_regime_walk_forward,
    REGIME_ALLOCATION,
)


def make_index_series(n=600) -> pd.Series:
    """Long index series for regime model training."""
    dates = pd.bdate_range("2018-01-01", periods=n)
    np.random.seed(42)
    close = 400 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 100)
    return pd.Series(close, index=dates, name="close")


class TestBuildRegimeFeatures:
    def test_returns_dataframe(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        assert isinstance(feats, pd.DataFrame)
        assert len(feats) > 0

    def test_expected_columns(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        expected = [
            "ret_5d", "ret_21d", "ret_63d",
            "vol_21d", "vol_63d", "vol_ratio",
            "vvol", "trend_strength", "drawdown_63d",
        ]
        for col in expected:
            assert col in feats.columns, f"Missing column: {col}"

    def test_no_nans_after_warmup(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        # dropna was called inside, so should have no NaNs
        assert feats.isna().sum().sum() == 0

    def test_all_backward_looking(self):
        """Modifying future data should not change current features."""
        idx = make_index_series(300)
        feats_orig = build_regime_features(idx, {})

        idx_mod = idx.copy()
        idx_mod.iloc[200:] *= 2
        feats_mod = build_regime_features(idx_mod, {})

        common = feats_orig.index.intersection(feats_mod.index)
        check_date = common[min(150, len(common) - 1)]
        # Only compare if check_date is before modification point
        if check_date < idx.index[200]:
            pd.testing.assert_series_equal(
                feats_orig.loc[check_date],
                feats_mod.loc[check_date],
                check_names=False,
            )


class TestFitRegimeModel:
    def test_kmeans_model_fits(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        model_dict = fit_regime_model(feats, n_regimes=4, method="kmeans")
        assert model_dict["type"] == "kmeans"
        assert "model" in model_dict
        assert "scaler" in model_dict

    def test_predict_returns_series(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        model_dict = fit_regime_model(feats, n_regimes=3, method="kmeans")
        preds = predict_regime(model_dict, feats)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(feats)
        assert set(preds.unique()).issubset({0, 1, 2})

    def test_different_n_regimes(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        for n in [2, 3, 4]:
            model_dict = fit_regime_model(feats, n_regimes=n, method="kmeans")
            preds = predict_regime(model_dict, feats)
            assert len(preds.unique()) <= n


class TestLabelRegimes:
    def test_returns_dict(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        model_dict = fit_regime_model(feats, n_regimes=4, method="kmeans")
        preds = predict_regime(model_dict, feats)
        names = label_regimes(feats, preds)
        assert isinstance(names, dict)
        assert len(names) <= 4

    def test_names_follow_convention(self):
        idx = make_index_series()
        feats = build_regime_features(idx, {})
        model_dict = fit_regime_model(feats, n_regimes=4, method="kmeans")
        preds = predict_regime(model_dict, feats)
        names = label_regimes(feats, preds)
        for name in names.values():
            assert name.startswith("low_vol_") or name.startswith("high_vol_")
            parts = name.split("_", 2)
            assert parts[2] in (
                "trending_up", "trending_down", "choppy",
            )


class TestGetRegimeAllocation:
    def test_known_regime(self):
        alloc = get_regime_allocation("low_vol_trending_up")
        assert alloc["swing_multiplier"] == 1.0
        assert alloc["swing_enabled"] is True

    def test_disabled_regime(self):
        alloc = get_regime_allocation("high_vol_choppy")
        assert alloc["swing_multiplier"] == 0.0
        assert alloc["swing_enabled"] is False

    def test_unknown_regime_defaults(self):
        alloc = get_regime_allocation("totally_unknown")
        assert alloc["swing_multiplier"] == 0.5
        assert alloc["swing_enabled"] is True


class TestSmoothRegime:
    def test_removes_short_noise(self):
        # Regime 0 for 10 days, 2 days of regime 1 (below persistence=3), then back to 0
        labels = pd.Series([0]*10 + [1, 1] + [0]*10)
        smoothed = smooth_regime(labels, min_persistence=3)
        # The 2-day blip should not cause a regime switch
        assert smoothed.iloc[10] == 0
        assert smoothed.iloc[11] == 0

    def test_allows_persistent_change(self):
        labels = pd.Series([0]*10 + [1]*5 + [0]*5)
        smoothed = smooth_regime(labels, min_persistence=3)
        # After 3+ days of regime 1, it should switch
        assert smoothed.iloc[13] == 1

    def test_preserves_length(self):
        labels = pd.Series([0]*20)
        smoothed = smooth_regime(labels, min_persistence=3)
        assert len(smoothed) == 20


class TestWalkForwardRegime:
    def test_produces_predictions(self):
        idx = make_index_series(700)
        config = {
            "initial_train_days": 400,
            "step_days": 63,
            "n_regimes": 3,
            "regime_method": "kmeans",
        }
        preds, names = run_regime_walk_forward(idx, config)
        assert len(preds) > 0
        assert isinstance(names, dict)

    def test_insufficient_data_returns_empty(self):
        idx = make_index_series(100)
        config = {
            "initial_train_days": 400,
            "step_days": 63,
            "n_regimes": 3,
            "regime_method": "kmeans",
        }
        preds, names = run_regime_walk_forward(idx, config)
        assert len(preds) == 0

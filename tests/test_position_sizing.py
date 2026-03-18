"""
Tests for position sizing — vol targeting, ML scaling, regime adjustment.
Skill reference: .claude/skills/position-sizing/SKILL.md
"""

import pytest
import numpy as np

from src.signals.sizing import (
    vol_target_size,
    notional_to_shares,
    ml_probability_size_scale,
    regime_adjusted_size,
    compute_swing_position_size,
    compute_barriers,
)


class TestVolTargetSize:
    def test_basic_sizing(self):
        size = vol_target_size(
            sleeve_nav=30_000,
            instrument_vol=0.25,
            holding_days=10,
        )
        assert size > 0
        assert size <= 30_000 * 0.15  # Max position cap

    def test_high_vol_smaller_size(self):
        low_vol = vol_target_size(30_000, 0.15, 10)
        high_vol = vol_target_size(30_000, 0.50, 10)
        assert high_vol < low_vol

    def test_zero_vol_returns_zero(self):
        assert vol_target_size(30_000, 0.0, 10) == 0.0

    def test_nan_vol_returns_zero(self):
        assert vol_target_size(30_000, np.nan, 10) == 0.0

    def test_respects_max_position_cap(self):
        size = vol_target_size(
            sleeve_nav=30_000,
            instrument_vol=0.01,  # Very low vol = very large size
            holding_days=10,
            max_position_pct=0.15,
        )
        assert size <= 30_000 * 0.15

    def test_below_minimum_returns_zero(self):
        size = vol_target_size(
            sleeve_nav=100,
            instrument_vol=0.50,
            holding_days=10,
            min_position_usd=500,
        )
        assert size == 0.0


class TestNotionalToShares:
    def test_rounds_down(self):
        shares = notional_to_shares(1050, 100)
        assert shares == 10.0

    def test_zero_price_returns_zero(self):
        assert notional_to_shares(1000, 0) == 0.0

    def test_negative_price_returns_zero(self):
        assert notional_to_shares(1000, -10) == 0.0


class TestMLProbabilityScale:
    def test_below_threshold_returns_zero(self):
        scale = ml_probability_size_scale(0.50, entry_threshold=0.60)
        assert scale == 0.0

    def test_at_threshold_returns_min_scale(self):
        scale = ml_probability_size_scale(0.60, entry_threshold=0.60)
        assert scale == pytest.approx(0.5)

    def test_at_max_returns_max_scale(self):
        scale = ml_probability_size_scale(1.0, entry_threshold=0.60)
        assert scale == pytest.approx(1.5)

    def test_midpoint(self):
        scale = ml_probability_size_scale(0.80, entry_threshold=0.60)
        assert 0.5 < scale < 1.5


class TestRegimeAdjustedSize:
    def test_trending_up_full_size(self):
        adjusted = regime_adjusted_size(10_000, "low_vol_trending_up")
        assert adjusted == 10_000

    def test_choppy_halved(self):
        adjusted = regime_adjusted_size(10_000, "low_vol_choppy")
        assert adjusted == 5_000

    def test_high_vol_choppy_zero(self):
        adjusted = regime_adjusted_size(10_000, "high_vol_choppy")
        assert adjusted == 0

    def test_unknown_regime_conservative(self):
        adjusted = regime_adjusted_size(10_000, "unknown")
        assert adjusted == 5_000

    def test_high_vvol_halves(self):
        adjusted = regime_adjusted_size(
            10_000, "low_vol_trending_up", vvol_pct=0.90
        )
        assert adjusted == 5_000  # 10k * 1.0 * 0.5


class TestComputeSwingPositionSize:
    def test_full_pipeline(self):
        result = compute_swing_position_size(
            symbol="AAPL",
            sleeve_nav=30_000,
            instrument_vol=0.25,
            ml_prob=0.75,
            current_regime="low_vol_trending_up",
            vvol_percentile=0.5,
            price=150.0,
            config={},
        )
        assert result["shares"] > 0
        assert result["reason"] == "OK"

    def test_below_ml_threshold(self):
        result = compute_swing_position_size(
            symbol="AAPL",
            sleeve_nav=30_000,
            instrument_vol=0.25,
            ml_prob=0.50,
            current_regime="low_vol_trending_up",
            vvol_percentile=0.5,
            price=150.0,
            config={},
        )
        assert result["shares"] == 0
        assert result["reason"] == "below_ml_threshold"

    def test_disabled_regime(self):
        result = compute_swing_position_size(
            symbol="AAPL",
            sleeve_nav=30_000,
            instrument_vol=0.25,
            ml_prob=0.75,
            current_regime="high_vol_choppy",
            vvol_percentile=0.5,
            price=150.0,
            config={},
        )
        assert result["shares"] == 0
        assert "regime_disabled" in result["reason"]


class TestComputeBarriers:
    def test_tp_above_entry(self):
        b = compute_barriers(100, 0.25, 10)
        assert b["tp_price"] > 100

    def test_sl_below_entry(self):
        b = compute_barriers(100, 0.25, 10)
        assert b["sl_price"] < 100

    def test_wider_barriers_with_higher_vol(self):
        b_low = compute_barriers(100, 0.15, 10)
        b_high = compute_barriers(100, 0.40, 10)
        assert b_high["tp_pct"] > b_low["tp_pct"]
        assert b_high["sl_pct"] > b_low["sl_pct"]

    def test_k1_k2_asymmetry(self):
        b = compute_barriers(100, 0.25, 10, k1=2.0, k2=1.0)
        assert b["tp_pct"] > b["sl_pct"]

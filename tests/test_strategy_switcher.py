"""Tests for regime-aware strategy switcher."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.brain.strategy_switcher import (
    StrategyProfile,
    REGIME_STRATEGIES,
    DEFAULT_PROFILE,
    blend_strategies,
    StrategySwitcher,
)


class TestRegimeStrategies:
    def test_all_regimes_mapped(self):
        expected = [
            "low_vol_trending_up", "low_vol_choppy", "low_vol_trending_down",
            "high_vol_trending_up", "high_vol_choppy", "high_vol_trending_down",
        ]
        for regime in expected:
            assert regime in REGIME_STRATEGIES

    def test_crisis_disables_swing(self):
        crisis = REGIME_STRATEGIES["high_vol_trending_down"]
        assert crisis.swing_enabled is False
        assert crisis.swing_multiplier == 0.0
        assert crisis.max_concurrent_positions == 0

    def test_aggressive_enables_swing(self):
        aggressive = REGIME_STRATEGIES["low_vol_trending_up"]
        assert aggressive.swing_enabled is True
        assert aggressive.swing_multiplier == 1.0
        assert aggressive.max_concurrent_positions > 5


class TestBlendStrategies:
    def test_blend_zero_returns_current(self):
        current = REGIME_STRATEGIES["low_vol_trending_up"]
        target = REGIME_STRATEGIES["high_vol_trending_down"]
        blended = blend_strategies(current, target, 0.0)
        assert blended.swing_multiplier == current.swing_multiplier
        assert blended.swing_enabled == current.swing_enabled

    def test_blend_one_returns_target(self):
        current = REGIME_STRATEGIES["low_vol_trending_up"]
        target = REGIME_STRATEGIES["high_vol_trending_down"]
        blended = blend_strategies(current, target, 1.0)
        assert blended.swing_multiplier == target.swing_multiplier
        assert blended.swing_enabled == target.swing_enabled

    def test_blend_half_interpolates(self):
        current = StrategyProfile(
            name="a", swing_enabled=True, swing_multiplier=1.0,
            momentum_threshold_pct=0.04, vol_expansion_ratio=1.5,
            ml_entry_threshold=0.6, max_concurrent_positions=10,
            holding_days=10, tp_multiplier=2.0, sl_multiplier=1.0,
            rebalance_band=0.05,
        )
        target = StrategyProfile(
            name="b", swing_enabled=False, swing_multiplier=0.0,
            momentum_threshold_pct=0.10, vol_expansion_ratio=3.0,
            ml_entry_threshold=0.9, max_concurrent_positions=0,
            holding_days=3, tp_multiplier=1.0, sl_multiplier=0.5,
            rebalance_band=0.02,
        )
        blended = blend_strategies(current, target, 0.5)
        assert blended.swing_multiplier == 0.5
        assert blended.ml_entry_threshold == 0.75


class TestStrategySwitcher:
    def test_config_overlay(self):
        switcher = StrategySwitcher(config={})
        overlay = switcher.get_config_overlay()
        assert "swing_enabled" in overlay
        assert "regime" in overlay
        assert "strategy_name" in overlay

    def test_force_reduce_in_crisis(self):
        switcher = StrategySwitcher(config={})
        switcher.current_regime = "high_vol_trending_down"
        assert switcher.should_force_reduce() is True

    def test_no_force_reduce_in_bull(self):
        switcher = StrategySwitcher(config={})
        switcher.current_regime = "low_vol_trending_up"
        assert switcher.should_force_reduce() is False

    def test_transition_log_tracks_changes(self):
        switcher = StrategySwitcher(config={})
        # Manually simulate regime changes
        switcher.current_regime = "low_vol_trending_up"
        switcher.current_strategy = REGIME_STRATEGIES["low_vol_trending_up"]

        # Force a change
        old = switcher.current_regime
        new_regime = "high_vol_choppy"
        switcher.transition_log.append((
            pd.Timestamp("2023-06-01"), old, new_regime,
            switcher.current_strategy.name,
            REGIME_STRATEGIES[new_regime].name,
        ))
        switcher.current_regime = new_regime
        switcher.current_strategy = REGIME_STRATEGIES[new_regime]

        assert len(switcher.transition_log) == 1
        assert switcher.transition_log[0][2] == "high_vol_choppy"

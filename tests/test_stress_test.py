"""Tests for adversarial stress tester."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_feeds.synthetic import generate_gbm_ohlcv
from src.backtesting.stress_test import (
    inject_flash_crash,
    inject_gap_down,
    inject_liquidity_crisis,
    inject_correlation_spike,
    _deep_copy_prices,
)


@pytest.fixture
def sample_prices():
    df = generate_gbm_ohlcv("AAPL", "2023-01-01", "2023-12-31", seed=42)
    return {"AAPL": df}


class TestFlashCrash:
    def test_crash_reduces_price(self, sample_prices):
        date = sample_prices["AAPL"].index[100]
        original_close = sample_prices["AAPL"].loc[date, "close"]
        modified = inject_flash_crash(sample_prices, "AAPL", date, drop_pct=0.10)
        new_close = modified["AAPL"].loc[date, "close"]
        assert new_close < original_close
        assert abs(new_close - original_close * 0.90) < 1.0

    def test_volume_spikes(self, sample_prices):
        date = sample_prices["AAPL"].index[100]
        original_vol = sample_prices["AAPL"].loc[date, "volume"]
        modified = inject_flash_crash(sample_prices, "AAPL", date)
        assert modified["AAPL"].loc[date, "volume"] > original_vol

    def test_does_not_modify_original(self, sample_prices):
        date = sample_prices["AAPL"].index[100]
        original_close = sample_prices["AAPL"].loc[date, "close"]
        inject_flash_crash(sample_prices, "AAPL", date)
        assert sample_prices["AAPL"].loc[date, "close"] == original_close

    def test_nonexistent_symbol(self, sample_prices):
        date = sample_prices["AAPL"].index[100]
        result = inject_flash_crash(sample_prices, "FAKE", date)
        assert "AAPL" in result


class TestGapDown:
    def test_gap_creates_lower_open(self, sample_prices):
        date = sample_prices["AAPL"].index[100]
        prev_close = sample_prices["AAPL"].iloc[99]["close"]
        modified = inject_gap_down(sample_prices, "AAPL", date, gap_pct=0.05)
        new_open = modified["AAPL"].loc[date, "open"]
        assert new_open < prev_close * 0.96


class TestLiquidityCrisis:
    def test_reduces_volume(self, sample_prices):
        date = sample_prices["AAPL"].index[100]
        modified = inject_liquidity_crisis(
            sample_prices, ["AAPL"], date, duration_days=5, volume_factor=0.1
        )
        # Check volume reduced in affected period
        affected_dates = modified["AAPL"].index[
            (modified["AAPL"].index >= date) &
            (modified["AAPL"].index < date + pd.offsets.BDay(5))
        ]
        for d in affected_dates:
            assert modified["AAPL"].loc[d, "volume"] < sample_prices["AAPL"].loc[d, "volume"]


class TestCorrelationSpike:
    def test_modifies_prices(self, sample_prices):
        # Add a second symbol
        prices = {**sample_prices, "MSFT": generate_gbm_ohlcv("MSFT", "2023-01-01", "2023-12-31", seed=99)}
        date = prices["AAPL"].index[100]
        modified = inject_correlation_spike(prices, ["AAPL", "MSFT"], date, duration_days=10)
        # Prices should be modified in the affected period
        affected = modified["AAPL"].index[
            (modified["AAPL"].index >= date) &
            (modified["AAPL"].index < date + pd.offsets.BDay(10))
        ]
        changes = 0
        for d in affected:
            if modified["AAPL"].loc[d, "close"] != prices["AAPL"].loc[d, "close"]:
                changes += 1
        assert changes > 0


class TestDeepCopy:
    def test_copy_is_independent(self, sample_prices):
        copied = _deep_copy_prices(sample_prices)
        copied["AAPL"].iloc[0, 0] = -999
        assert sample_prices["AAPL"].iloc[0, 0] != -999

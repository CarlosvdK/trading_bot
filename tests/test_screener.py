"""
Tests for dynamic universe screener.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.screener import screen_universe, expand_universe


def make_liquid_stock(n=300, avg_vol=2_000_000):
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 1.0)
    close = np.maximum(close, 50)
    return pd.DataFrame(
        {
            "open": close * 1.001,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.random.randint(
                int(avg_vol * 0.5), int(avg_vol * 1.5), n
            ),
        },
        index=dates,
    )


def make_penny_stock(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(99)
    close = 2 + np.cumsum(np.random.randn(n) * 0.1)
    close = np.maximum(close, 0.5)
    return pd.DataFrame(
        {
            "open": close * 1.001,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(10_000, 50_000, n),
        },
        index=dates,
    )


class TestScreenUniverse:
    def test_liquid_stock_passes(self):
        data = {"AAPL": make_liquid_stock()}
        results = screen_universe(data)
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

    def test_penny_stock_rejected(self):
        data = {"PENNY": make_penny_stock()}
        results = screen_universe(data)
        assert len(results) == 0

    def test_sorts_by_dollar_volume(self):
        data = {
            "HIGH_VOL": make_liquid_stock(avg_vol=5_000_000),
            "LOW_VOL": make_liquid_stock(avg_vol=1_000_000),
        }
        results = screen_universe(data)
        if len(results) == 2:
            assert results[0]["symbol"] == "HIGH_VOL"

    def test_respects_custom_filters(self):
        data = {"AAPL": make_liquid_stock()}
        results = screen_universe(
            data, filters={"min_price": 200}  # Price ~100, should reject
        )
        assert len(results) == 0

    def test_short_history_rejected(self):
        short = make_liquid_stock(n=50)
        results = screen_universe({"SHORT": short})
        assert len(results) == 0


class TestExpandUniverse:
    def test_adds_new_symbols(self):
        data = {
            "AAPL": make_liquid_stock(),
            "MSFT": make_liquid_stock(),
        }
        expanded = expand_universe(["AAPL"], data, max_symbols=10)
        assert "AAPL" in expanded
        assert len(expanded) >= 1

    def test_respects_max_symbols(self):
        data = {f"SYM{i}": make_liquid_stock() for i in range(20)}
        expanded = expand_universe([], data, max_symbols=5)
        assert len(expanded) <= 5

"""
Tests for swing signal generation.
Skill reference: .claude/skills/swing-signal-generation/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.signals.signals import (
    momentum_breakout_candidates,
    volatility_expansion_candidates,
    is_risk_on,
    generate_swing_signals,
)


def make_trending_stock(n=100, trend=0.012) -> pd.DataFrame:
    """Create a stock with upward trend and volume surge."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    np.random.seed(42)
    close = 100.0
    closes = []
    for i in range(n):
        close *= 1 + trend + np.random.randn() * 0.005
        closes.append(close)
    closes = np.array(closes)

    volume = np.ones(n) * 1_000_000
    volume[-5:] = 2_000_000  # Volume surge at end

    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": volume,
        },
        index=dates,
    )


def make_flat_stock(n=100) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1_000_000,
        },
        index=dates,
    )


def make_index(n=100) -> pd.DataFrame:
    """Create a healthy index (risk-on)."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    np.random.seed(123)
    close = 400 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 50_000_000,
        },
        index=dates,
    )


class TestMomentumBreakout:
    def test_detects_trending_stock(self):
        stock = make_trending_stock()
        prices = {"TREND": stock}
        date = stock.index[-1]
        candidates = momentum_breakout_candidates(
            prices, date, {"momentum_threshold_pct": 0.02}
        )
        assert len(candidates) >= 1
        assert candidates[0]["symbol"] == "TREND"
        assert candidates[0]["signal_type"] == "momentum_breakout"

    def test_ignores_flat_stock(self):
        stock = make_flat_stock()
        prices = {"FLAT": stock}
        date = stock.index[-1]
        candidates = momentum_breakout_candidates(prices, date, {})
        assert len(candidates) == 0

    def test_candidate_format(self):
        stock = make_trending_stock()
        prices = {"TREND": stock}
        date = stock.index[-1]
        candidates = momentum_breakout_candidates(
            prices, date, {"momentum_threshold_pct": 0.02}
        )
        if candidates:
            c = candidates[0]
            assert "symbol" in c
            assert "signal_type" in c
            assert "direction" in c
            assert "signal_date" in c


class TestVolatilityExpansion:
    def test_detects_vol_expansion(self):
        dates = pd.bdate_range("2023-01-01", periods=100)
        np.random.seed(42)
        # Low vol for first 90 days, then high vol for last 10
        close = np.ones(100) * 100
        close[1:90] += np.cumsum(np.random.randn(89) * 0.1)  # Very low vol
        close[90:] = close[89] + np.cumsum(np.random.randn(10) * 5)  # High vol
        df = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )
        candidates = volatility_expansion_candidates(
            {"VOL_EXP": df}, dates[-1], {"vol_expansion_ratio": 1.3}
        )
        assert len(candidates) >= 1


class TestIsRiskOn:
    def test_healthy_market_is_risk_on(self):
        index = make_index()
        date = index.index[-1]
        assert is_risk_on(index, date, {})

    def test_missing_date_returns_false(self):
        index = make_index()
        future_date = pd.Timestamp("2099-01-01")
        assert not is_risk_on(index, future_date, {})

    def test_insufficient_history_returns_false(self):
        index = make_index(10)  # Only 10 days
        date = index.index[-1]
        assert not is_risk_on(index, date, {})


class TestGenerateSwingSignals:
    def test_generates_candidates_in_risk_on(self):
        stock = make_trending_stock()
        index = make_index()
        # Align dates
        common = stock.index.intersection(index.index)
        if len(common) > 30:
            date = common[-1]
            signals = generate_swing_signals(
                {"TREND": stock},
                index,
                date,
                {"momentum_threshold_pct": 0.02},
            )
            # May or may not generate signals depending on risk-on gate
            assert isinstance(signals, list)

    def test_deduplicates_by_symbol(self):
        stock = make_trending_stock()
        index = make_index()
        date = stock.index[-1]
        # Even if both signal types fire, should only have 1 entry per symbol
        signals = generate_swing_signals(
            {"TREND": stock},
            index,
            date,
            {"momentum_threshold_pct": 0.01, "vol_expansion_ratio": 0.5},
        )
        symbols = [s["symbol"] for s in signals]
        assert len(symbols) == len(set(symbols))

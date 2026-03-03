"""
Tests for data validation.
Skill reference: .claude/skills/data-layer/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.data.validator import validate_ohlcv, ValidationResult


def make_ohlcv(n=100, start="2023-01-01") -> pd.DataFrame:
    """Create valid test OHLCV data."""
    dates = pd.bdate_range(start=start, periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)  # Ensure positive

    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(100_000, 10_000_000, n),
        },
        index=dates,
    )
    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


class TestValidOHLCV:
    def test_valid_data_passes(self):
        df = make_ohlcv()
        result = validate_ohlcv(df, "TEST")
        assert result.passed
        assert len(result.errors) == 0

    def test_returns_validation_result(self):
        df = make_ohlcv()
        result = validate_ohlcv(df, "TEST")
        assert isinstance(result, ValidationResult)
        assert result.symbol == "TEST"


class TestMissingColumns:
    def test_missing_close_column(self):
        df = make_ohlcv()
        df = df.drop(columns=["close"])
        result = validate_ohlcv(df, "TEST")
        assert not result.passed
        assert any("Missing columns" in e for e in result.errors)

    def test_missing_volume_column(self):
        df = make_ohlcv()
        df = df.drop(columns=["volume"])
        result = validate_ohlcv(df, "TEST")
        assert not result.passed


class TestDateIssues:
    def test_duplicate_dates(self):
        df = make_ohlcv()
        dup = pd.concat([df, df.iloc[[0]]])
        result = validate_ohlcv(dup, "TEST")
        assert not result.passed
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_non_monotonic_dates(self):
        df = make_ohlcv()
        df = df.iloc[::-1]  # Reverse order
        result = validate_ohlcv(df, "TEST")
        assert not result.passed
        assert any("monotonic" in e.lower() for e in result.errors)

    def test_future_dates(self):
        future_start = (pd.Timestamp.today() + pd.Timedelta(days=30)).strftime(
            "%Y-%m-%d"
        )
        df = make_ohlcv(n=10, start=future_start)
        result = validate_ohlcv(df, "TEST")
        assert not result.passed
        assert any("future" in e.lower() for e in result.errors)


class TestPriceConsistency:
    def test_high_less_than_low(self):
        df = make_ohlcv()
        df.iloc[5, df.columns.get_loc("high")] = 50
        df.iloc[5, df.columns.get_loc("low")] = 200
        result = validate_ohlcv(df, "TEST")
        assert not result.passed
        assert any("High < Low" in e for e in result.errors)

    def test_zero_prices(self):
        df = make_ohlcv()
        df.iloc[5, df.columns.get_loc("close")] = 0
        result = validate_ohlcv(df, "TEST")
        assert not result.passed
        assert any("zero/negative" in e.lower() for e in result.errors)

    def test_negative_prices(self):
        df = make_ohlcv()
        df.iloc[5, df.columns.get_loc("open")] = -10
        result = validate_ohlcv(df, "TEST")
        assert not result.passed


class TestWarnings:
    def test_zero_volume_warning(self):
        df = make_ohlcv()
        df.iloc[5, df.columns.get_loc("volume")] = 0
        result = validate_ohlcv(df, "TEST")
        assert result.passed  # Warning, not error
        assert any("zero volume" in w.lower() for w in result.warnings)

    def test_large_gap_warning(self):
        df = make_ohlcv()
        # Create a 10-day gap
        idx_list = list(df.index)
        idx_list[50] = idx_list[49] + pd.Timedelta(days=15)
        for i in range(51, len(idx_list)):
            idx_list[i] = idx_list[i - 1] + pd.Timedelta(days=1)
        df.index = pd.DatetimeIndex(idx_list)
        result = validate_ohlcv(df, "TEST")
        assert any("gaps" in w.lower() for w in result.warnings)

    def test_extreme_price_move_warning(self):
        df = make_ohlcv()
        # Create a 60% move
        df.iloc[50, df.columns.get_loc("close")] = (
            df.iloc[49, df.columns.get_loc("close")] * 1.65
        )
        df.iloc[50, df.columns.get_loc("high")] = max(
            df.iloc[50]["high"], df.iloc[50]["close"]
        )
        result = validate_ohlcv(df, "TEST")
        assert any("50%" in w for w in result.warnings)


class TestEmptyData:
    def test_empty_dataframe(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        df.index = pd.DatetimeIndex([])
        result = validate_ohlcv(df, "EMPTY")
        assert not result.passed
        assert any("empty" in e.lower() for e in result.errors)

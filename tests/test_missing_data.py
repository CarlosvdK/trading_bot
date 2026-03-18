"""
Tests for missing data handling.
Skill reference: .claude/skills/data-layer/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.data_feeds.missing import handle_missing_data


def make_ohlcv_with_gaps(n=100, n_missing=3) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)

    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )
    # Introduce NaN gaps
    for i in range(n_missing):
        idx = 10 + i * 5
        df.iloc[idx, df.columns.get_loc("close")] = np.nan
    return df


class TestHandleMissingData:
    def test_fills_small_gaps(self):
        df = make_ohlcv_with_gaps(n_missing=2)
        result = handle_missing_data(df, max_ffill_days=1)
        assert result is not None
        assert result["close"].isna().sum() == 0

    def test_rejects_too_much_missing(self):
        dates = pd.bdate_range("2023-01-01", periods=20)
        close = [100.0] * 20
        df = pd.DataFrame(
            {
                "open": close,
                "high": [c * 1.01 for c in close],
                "low": [c * 0.99 for c in close],
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )
        # Set 5 out of 20 to NaN = 25% missing
        for i in [2, 5, 8, 11, 14]:
            df.iloc[i, df.columns.get_loc("close")] = np.nan
        result = handle_missing_data(df, max_missing_pct=0.05)
        assert result is None

    def test_respects_ffill_limit(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": [100, np.nan, np.nan, np.nan, 105, 106, 107, 108, 109, 110],
                "volume": 1_000_000,
            },
            index=dates,
        )
        result = handle_missing_data(df, max_ffill_days=1, max_missing_pct=0.5)
        assert result is not None
        # Only 1 day of forward fill — the other 2 NaNs remain
        assert result["close"].isna().sum() == 2

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = handle_missing_data(df)
        assert result is None

    def test_clean_data_passes_through(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": range(100, 110),
                "volume": 1_000_000,
            },
            index=dates,
        )
        result = handle_missing_data(df)
        assert result is not None
        assert len(result) == 10

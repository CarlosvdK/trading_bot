"""
Tests for corporate actions handling.
Skill reference: .claude/skills/data-layer/SKILL.md
"""

import pytest
import pandas as pd
import numpy as np

from src.data.corporate_actions import apply_corporate_actions


def make_ohlcv_simple() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=50)
    return pd.DataFrame(
        {
            "open": 400.0,
            "high": 410.0,
            "low": 390.0,
            "close": 400.0,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


class TestCorporateActions:
    def test_split_adjusts_pre_split_prices(self):
        df = make_ohlcv_simple()
        actions = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "date": "2020-02-01",
                    "action_type": "split",
                    "adjustment_factor": 0.25,
                }
            ]
        )

        adjusted = apply_corporate_actions(df, "AAPL", actions)
        split_date = pd.Timestamp("2020-02-01")

        pre_split = adjusted[adjusted.index < split_date]
        post_split = adjusted[adjusted.index >= split_date]

        assert pre_split["close"].iloc[0] == pytest.approx(100.0)  # 400 * 0.25
        assert post_split["close"].iloc[0] == pytest.approx(400.0)

    def test_split_adjusts_volume_inversely(self):
        df = make_ohlcv_simple()
        actions = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "date": "2020-02-01",
                    "action_type": "split",
                    "adjustment_factor": 0.25,
                }
            ]
        )

        adjusted = apply_corporate_actions(df, "AAPL", actions)
        split_date = pd.Timestamp("2020-02-01")
        pre_split = adjusted[adjusted.index < split_date]

        assert pre_split["volume"].iloc[0] == pytest.approx(4_000_000.0)

    def test_no_action_for_wrong_symbol(self):
        df = make_ohlcv_simple()
        actions = pd.DataFrame(
            [
                {
                    "symbol": "TSLA",
                    "date": "2020-02-01",
                    "action_type": "split",
                    "adjustment_factor": 0.2,
                }
            ]
        )

        adjusted = apply_corporate_actions(df, "AAPL", actions)
        assert adjusted["close"].iloc[0] == pytest.approx(400.0)

    def test_multiple_splits(self):
        df = make_ohlcv_simple()
        actions = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "date": "2020-01-15",
                    "action_type": "split",
                    "adjustment_factor": 0.5,
                },
                {
                    "symbol": "AAPL",
                    "date": "2020-02-15",
                    "action_type": "split",
                    "adjustment_factor": 0.5,
                },
            ]
        )

        adjusted = apply_corporate_actions(df, "AAPL", actions)
        # Before first split: adjusted by both factors (0.5 * 0.5 = 0.25)
        first_date = adjusted.index[0]
        assert adjusted.loc[first_date, "close"] == pytest.approx(100.0)

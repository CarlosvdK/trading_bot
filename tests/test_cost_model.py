"""
Tests for cost model — fills, partial fills, fees.
Skill reference: .claude/skills/backtesting-engine/SKILL.md
"""

import pytest
import numpy as np

from src.backtesting.cost_model import CostModel


class TestFillPrice:
    def test_buy_fill_never_exceeds_bar_high(self):
        cm = CostModel(market_impact_factor=10.0)  # Extreme impact
        fill = cm.fill_price(
            bar_open=100,
            side="BUY",
            order_notional=1_000_000,
            adv_notional=100_000,
            bar_high=105,
            bar_low=95,
        )
        assert fill <= 105

    def test_sell_fill_never_below_bar_low(self):
        cm = CostModel(market_impact_factor=10.0)
        fill = cm.fill_price(
            bar_open=100,
            side="SELL",
            order_notional=1_000_000,
            adv_notional=100_000,
            bar_high=105,
            bar_low=95,
        )
        assert fill >= 95

    def test_buy_slippage_is_positive(self):
        cm = CostModel()
        fill = cm.fill_price(
            bar_open=100,
            side="BUY",
            order_notional=10_000,
            adv_notional=1_000_000,
            bar_high=105,
            bar_low=95,
        )
        assert fill > 100  # Slippage pushes BUY price up

    def test_sell_slippage_is_negative(self):
        cm = CostModel()
        fill = cm.fill_price(
            bar_open=100,
            side="SELL",
            order_notional=10_000,
            adv_notional=1_000_000,
            bar_high=105,
            bar_low=95,
        )
        assert fill < 100  # Slippage pushes SELL price down

    def test_small_order_minimal_impact(self):
        cm = CostModel()
        fill = cm.fill_price(
            bar_open=100,
            side="BUY",
            order_notional=100,
            adv_notional=10_000_000,
            bar_high=105,
            bar_low=95,
        )
        assert abs(fill - 100) < 0.5  # Very small slippage


class TestPartialFill:
    def test_small_order_fills_fully(self):
        cm = CostModel()
        filled = cm.partial_fill_qty(
            order_qty=100,
            adv_notional=10_000_000,
            fill_price=100,
            participation_rate=0.05,
        )
        assert filled == 100

    def test_large_order_partially_fills(self):
        cm = CostModel()
        filled = cm.partial_fill_qty(
            order_qty=100_000,
            adv_notional=1_000_000,
            fill_price=100,
            participation_rate=0.05,
        )
        # Max fillable = 0.05 * 1M / 100 = 500 shares
        assert filled == 500
        assert filled < 100_000

    def test_participation_rate_respected(self):
        cm = CostModel()
        filled = cm.partial_fill_qty(
            order_qty=1000,
            adv_notional=100_000,
            fill_price=50,
            participation_rate=0.05,
        )
        max_notional = 0.05 * 100_000
        max_qty = max_notional / 50
        assert filled <= max_qty


class TestRoundtripCost:
    def test_roundtrip_positive(self):
        cm = CostModel()
        cost = cm.total_roundtrip_bps(
            order_notional=10_000, adv_notional=1_000_000
        )
        assert cost > 0

    def test_short_has_borrow_cost(self):
        cm = CostModel()
        long_cost = cm.total_roundtrip_bps(
            order_notional=10_000,
            adv_notional=1_000_000,
            is_short=False,
        )
        short_cost = cm.total_roundtrip_bps(
            order_notional=10_000,
            adv_notional=1_000_000,
            is_short=True,
            holding_days=10,
        )
        assert short_cost > long_cost

    def test_larger_order_higher_cost(self):
        cm = CostModel()
        small = cm.total_roundtrip_bps(
            order_notional=1_000, adv_notional=1_000_000
        )
        large = cm.total_roundtrip_bps(
            order_notional=100_000, adv_notional=1_000_000
        )
        assert large > small

    def test_settlement_lag_default(self):
        cm = CostModel()
        assert cm.settlement_lag_days == 2

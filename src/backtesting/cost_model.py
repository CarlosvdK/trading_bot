"""
Realistic transaction cost model.
Skill reference: .claude/skills/backtesting-engine/SKILL.md
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CostModel:
    """
    Realistic transaction cost model for equity/ETF trading.
    All costs in basis points (bps) unless noted.
    """

    commission_bps: float = 0.5
    regulatory_fee_bps: float = 0.3
    spread_cost_bps: float = 1.0
    market_impact_factor: float = 0.1
    borrow_cost_bps_annual: float = 100
    settlement_lag_days: int = 2

    def total_roundtrip_bps(
        self,
        order_notional: float,
        adv_notional: float,
        holding_days: int = 10,
        is_short: bool = False,
    ) -> float:
        """Total round-trip cost estimate in bps."""
        participation = order_notional / max(adv_notional, 1)
        impact = self.market_impact_factor * np.sqrt(participation) * 10_000

        entry_cost = self.commission_bps + self.spread_cost_bps + impact
        exit_cost = (
            self.commission_bps
            + self.regulatory_fee_bps
            + self.spread_cost_bps
            + impact
        )
        borrow_cost = (
            self.borrow_cost_bps_annual * holding_days / 252
            if is_short
            else 0.0
        )

        return entry_cost + exit_cost + borrow_cost

    def fill_price(
        self,
        bar_open: float,
        side: str,
        order_notional: float,
        adv_notional: float,
        bar_high: float,
        bar_low: float,
    ) -> float:
        """
        Simulated fill price with slippage.
        Uses open price + market impact + half-spread.
        """
        participation = order_notional / max(adv_notional, 1)
        impact_pct = self.market_impact_factor * np.sqrt(participation)
        spread_pct = self.spread_cost_bps / 10_000

        if side == "BUY":
            fill = bar_open * (1 + impact_pct + spread_pct)
            return min(fill, bar_high)
        else:
            fill = bar_open * (1 - impact_pct - spread_pct)
            return max(fill, bar_low)

    def partial_fill_qty(
        self,
        order_qty: float,
        adv_notional: float,
        fill_price: float,
        participation_rate: float = 0.05,
    ) -> float:
        """
        Simulate partial fills. Large orders vs ADV fill partially.
        """
        max_fillable_notional = participation_rate * adv_notional
        max_fillable_qty = max_fillable_notional / max(fill_price, 0.01)
        return min(order_qty, max_fillable_qty)

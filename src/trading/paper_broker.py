"""
Paper Broker — simulates broker execution with realistic fills.
Skill reference: .claude/skills/paper-broker-order-manager/SKILL.md
"""

import logging
from typing import Optional

import pandas as pd

from src.trading.order_types import (
    Order,
    Fill,
    OrderType,
    OrderSide,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class PaperBroker:
    """
    Simulates broker execution with realistic fills, slippage, and partial fills.
    Uses same interface as LiveBrokerStub.
    """

    mode = "paper"

    def __init__(self, price_data: dict, cost_model, config: dict):
        self.price_data = price_data
        self.cost_model = cost_model
        self.config = config

    def execute(
        self, order: Order, current_date: pd.Timestamp
    ) -> Optional[Fill]:
        """
        Simulate order execution. Fills at next bar's open (after signal bar).
        For MOC orders, fills at bar's close.
        """
        df = self.price_data.get(order.symbol)
        if df is None:
            logger.warning(f"No data for {order.symbol}")
            return None

        if current_date not in df.index:
            return None

        idx = df.index.get_loc(current_date)
        if idx + 1 >= len(df):
            logger.warning(
                f"No next bar for {order.symbol} after {current_date}"
            )
            return None

        fill_bar = df.iloc[idx + 1]
        adv = self._get_adv(order.symbol, current_date)

        # MOC fills at close
        if order.order_type == OrderType.MOC:
            raw_fill_price = fill_bar["close"]
        else:
            raw_fill_price = fill_bar["open"]

        # Limit order logic
        if order.order_type == OrderType.LIMIT:
            if (
                order.side == OrderSide.BUY
                and fill_bar["low"] > order.limit_price
            ):
                return None
            if (
                order.side == OrderSide.SELL
                and fill_bar["high"] < order.limit_price
            ):
                return None
            raw_fill_price = order.limit_price

        # Stop order logic
        if order.order_type == OrderType.STOP:
            if (
                order.side == OrderSide.SELL
                and fill_bar["low"] > order.stop_price
            ):
                return None
            raw_fill_price = min(order.stop_price, fill_bar["open"])

        # Apply slippage via cost model
        notional = order.qty * raw_fill_price
        fill_price = self.cost_model.fill_price(
            raw_fill_price,
            order.side.value,
            notional,
            adv,
            fill_bar["high"],
            fill_bar["low"],
        )
        slippage = abs(fill_price - raw_fill_price) / raw_fill_price

        # Partial fill
        filled_qty = self.cost_model.partial_fill_qty(
            order.qty, adv, fill_price
        )
        if filled_qty < order.qty * 0.01:
            logger.warning(
                f"Near-zero fill for {order.symbol}. ADV too low."
            )
            return None

        # Fees
        fees = (
            filled_qty
            * fill_price
            * (self.cost_model.commission_bps / 10_000)
        )

        status = (
            OrderStatus.FILLED
            if filled_qty >= order.qty * 0.99
            else OrderStatus.PARTIAL
        )

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=filled_qty,
            fill_price=fill_price,
            fees=fees,
            slippage=slippage,
            filled_at=df.index[idx + 1],
            sleeve=order.sleeve,
            status=status,
        )

    def _get_adv(
        self, symbol: str, date: pd.Timestamp, window: int = 30
    ) -> float:
        """Average daily notional volume over trailing window."""
        df = self.price_data.get(symbol)
        if df is None:
            return 1_000_000
        if date not in df.index:
            return 1_000_000
        idx = df.index.get_loc(date)
        if idx < window:
            return 1_000_000
        recent = df.iloc[max(0, idx - window) : idx]
        return float((recent["close"] * recent["volume"]).mean())


class LiveBrokerStub:
    """
    Stub for live broker integration.
    Raises NotImplementedError — forces explicit implementation before going live.
    """

    mode = "live"

    def execute(
        self, order: Order, current_date: pd.Timestamp
    ) -> Optional[Fill]:
        raise NotImplementedError(
            "Live broker not implemented. "
            "Integrate your broker's API here (e.g., IBKR, Alpaca). "
            "Never deploy this stub to live trading."
        )

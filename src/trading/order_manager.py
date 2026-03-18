"""
Order Manager — single gateway for all orders. Routes through Risk Governor.
Skill reference: .claude/skills/paper-broker-order-manager/SKILL.md
"""

import logging
import signal
import sys
from typing import List, Optional

import pandas as pd

from src.trading.order_types import Order, Fill

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Routes orders through risk checks then to broker.
    Maintains order log and position state.
    """

    def __init__(
        self,
        broker,
        risk_governor,
        portfolio_state,
        config: dict,
    ):
        self.broker = broker
        self.risk_governor = risk_governor
        self.state = portfolio_state
        self.config = config
        self.order_log: List[dict] = []
        self._orders_this_minute: int = 0
        self._notional_this_minute: float = 0.0
        self._last_minute: Optional[pd.Timestamp] = None

        self.max_orders_per_minute = config.get("max_orders_per_minute", 20)
        self.max_notional_per_minute = config.get(
            "max_notional_per_minute", 500_000
        )

    def submit(
        self, order: Order, current_date: pd.Timestamp
    ) -> Optional[Fill]:
        """
        Submit an order. Returns Fill if executed, None if rejected.

        Flow: rate limit → Risk Governor → broker → log.
        """
        # --- Rate Limit ---
        if not self._check_rate_limit(order, current_date):
            self._log_order(
                order,
                status="REJECTED",
                reason="rate_limit",
                date=current_date,
            )
            return None

        # --- Risk Governor ---
        notional = order.qty * self._get_price_estimate(order.symbol)
        allowed, reason = self.risk_governor.pre_trade_check(
            symbol=order.symbol,
            side=order.side,
            notional=notional,
            sleeve=order.sleeve,
            state=self.state,
            sector=order.sector,
            current_date=(
                current_date.date()
                if hasattr(current_date, "date")
                else current_date
            ),
        )

        if not allowed:
            logger.warning(
                f"Order rejected by Risk Governor: {order.symbol} — {reason}"
            )
            self._log_order(
                order,
                status="REJECTED",
                reason=reason,
                date=current_date,
            )
            return None

        # --- Execute via Broker ---
        try:
            fill = self.broker.execute(order, current_date)
        except Exception as e:
            logger.error(
                f"Broker execution error for {order.symbol}: {e}"
            )
            self._log_order(
                order,
                status="REJECTED",
                reason=str(e),
                date=current_date,
            )
            return None

        if fill:
            self._log_order(
                order,
                status=fill.status.value,
                fill=fill,
                date=current_date,
            )
            logger.info(
                f"Fill: {order.side.value} {fill.filled_qty:.0f} "
                f"{order.symbol} @ {fill.fill_price:.4f} | "
                f"fees={fill.fees:.2f} | sleeve={order.sleeve}"
            )

        return fill

    def _check_rate_limit(
        self, order: Order, current_date: pd.Timestamp
    ) -> bool:
        """Reset counters each minute."""
        minute = (
            current_date.floor("min")
            if hasattr(current_date, "floor")
            else current_date
        )
        if minute != self._last_minute:
            self._orders_this_minute = 0
            self._notional_this_minute = 0.0
            self._last_minute = minute

        if self._orders_this_minute >= self.max_orders_per_minute:
            logger.warning("Order rate limit hit")
            return False

        est_notional = order.qty * (order.limit_price or 100)
        if (
            self._notional_this_minute + est_notional
            > self.max_notional_per_minute
        ):
            logger.warning("Notional rate limit hit")
            return False

        self._orders_this_minute += 1
        self._notional_this_minute += est_notional
        return True

    def _get_price_estimate(self, symbol: str) -> float:
        """Best available price estimate for pre-trade sizing."""
        pos = self.state.positions.get(symbol, {})
        if isinstance(pos, dict):
            return pos.get("last_price", pos.get("avg_cost", 100.0))
        return getattr(pos, "avg_cost", 100.0)

    def _log_order(
        self,
        order: Order,
        status: str,
        date: pd.Timestamp,
        reason: str = "",
        fill: Optional[Fill] = None,
    ):
        record = {
            "timestamp": date,
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "qty": order.qty,
            "sleeve": order.sleeve,
            "limit_price": order.limit_price,
            "status": status,
            "reason": reason,
            "ml_prob": order.ml_prob,
        }
        if fill:
            record.update(
                {
                    "fill_price": fill.fill_price,
                    "filled_qty": fill.filled_qty,
                    "fees": fill.fees,
                    "slippage": fill.slippage,
                }
            )
        self.order_log.append(record)


class GracefulShutdown:
    """
    Handles SIGTERM/SIGINT for clean exit.
    Never interrupts mid-order-submission.
    """

    def __init__(self, order_manager: OrderManager, portfolio_state):
        self.order_manager = order_manager
        self.state = portfolio_state
        self._shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        logger.info(
            f"[SHUTDOWN] Signal {signum} received. Finishing current cycle..."
        )
        self._shutdown_requested = True

    @property
    def should_shutdown(self):
        return self._shutdown_requested

    def save_state(self, path: str):
        """Persist state to disk before exit."""
        import json

        state_data = {
            "order_log_count": len(self.order_manager.order_log),
            "shutdown_clean": True,
        }
        with open(path, "w") as f:
            json.dump(state_data, f, indent=2, default=str)
        logger.info(f"[SHUTDOWN] State saved to {path}. Exiting cleanly.")
        sys.exit(0)

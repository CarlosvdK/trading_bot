"""
Interactive Brokers Adapter — connects to TWS/Gateway via ib_insync.

Requirements:
  pip install ib_insync
  TWS or IB Gateway must be running with API enabled

Setup:
  1. Open TWS → Edit → Global Configuration → API → Settings
  2. Enable "Enable ActiveX and Socket Clients"
  3. Set Socket Port (4002 for paper TWS, 4001 for live TWS,
     7497 for paper Gateway, 7496 for live Gateway)
  4. Add 127.0.0.1 to trusted IPs
  5. Uncheck "Read-Only API" to allow order placement

Environment variables:
  IBKR_HOST (default: 127.0.0.1)
  IBKR_PORT (default: 4002 for paper TWS)
  IBKR_CLIENT_ID (default: 1)
"""

import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from src.trading.order_types import (
    Order,
    Fill,
    OrderType,
    OrderSide,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class IBKRBroker:
    """
    Live broker implementation for Interactive Brokers.
    Uses ib_insync for a clean async-compatible interface.
    """

    mode = "live"

    def __init__(self, config: dict):
        self.config = config
        self._ib = None
        self._connected = False

        # Connection settings
        self.host = os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("IBKR_PORT", "4002"))  # 4002=paper TWS, 4001=live TWS
        self.client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))

        # Safety: warn on live ports
        if self.port in (4001, 7496):
            logger.warning(
                "WARNING: Connected to LIVE trading port (%d). "
                "Use port 4002 (TWS) or 7497 (Gateway) for paper trading.",
                self.port,
            )

        self._connect()

    def _connect(self):
        """Connect to TWS/IB Gateway."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(
                f"Connected to IBKR at {self.host}:{self.port} "
                f"(client_id={self.client_id})"
            )

            # Log account info
            accounts = self._ib.managedAccounts()
            logger.info(f"IBKR accounts: {accounts}")
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            raise ConnectionError(
                f"Cannot connect to IBKR at {self.host}:{self.port}. "
                f"Make sure TWS or IB Gateway is running with API enabled."
            )

    def execute(
        self, order: Order, current_date: pd.Timestamp
    ) -> Optional[Fill]:
        """Submit order to IBKR and wait for fill."""
        if not self._connected:
            logger.error("Not connected to IBKR")
            return None

        from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder

        # Create contract
        contract = Stock(order.symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        # Map order type
        action = "BUY" if order.side in (OrderSide.BUY, OrderSide.COVER) else "SELL"

        if order.order_type == OrderType.MARKET or order.order_type == OrderType.MOC:
            ib_order = MarketOrder(action, order.qty)
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                logger.error(f"Limit order for {order.symbol} missing limit_price")
                return None
            ib_order = LimitOrder(action, order.qty, order.limit_price)
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                logger.error(f"Stop order for {order.symbol} missing stop_price")
                return None
            ib_order = StopOrder(action, order.qty, order.stop_price)
        else:
            logger.error(f"Unsupported order type: {order.order_type}")
            return None

        # Submit
        logger.info(
            f"Submitting to IBKR: {action} {order.qty} {order.symbol} "
            f"({order.order_type.value})"
        )
        trade = self._ib.placeOrder(contract, ib_order)

        # Wait for fill (with timeout)
        timeout = self.config.get("execution", {}).get("fill_timeout_seconds", 30)
        start = time.time()

        while time.time() - start < timeout:
            self._ib.sleep(0.5)
            if trade.isDone():
                break

        # Check fill status
        if not trade.fills:
            status = trade.orderStatus.status
            if status in ("Cancelled", "Inactive"):
                logger.warning(f"IBKR order cancelled/inactive: {order.symbol}")
                return None
            elif status == "Submitted" or status == "PreSubmitted":
                logger.warning(
                    f"IBKR order still pending after {timeout}s: {order.symbol}. "
                    f"Cancelling..."
                )
                self._ib.cancelOrder(ib_order)
                return None
            else:
                logger.warning(f"IBKR order status: {status} for {order.symbol}")
                return None

        # Process fill
        fill_data = trade.fills[0]
        execution = fill_data.execution

        total_qty = sum(f.execution.shares for f in trade.fills)
        avg_price = sum(
            f.execution.shares * f.execution.avgPrice for f in trade.fills
        ) / total_qty if total_qty > 0 else 0

        total_commission = sum(f.commissionReport.commission for f in trade.fills
                               if f.commissionReport.commission is not None)

        fill_status = (
            OrderStatus.FILLED
            if total_qty >= order.qty * 0.99
            else OrderStatus.PARTIAL
        )

        logger.info(
            f"IBKR fill: {action} {total_qty} {order.symbol} @ ${avg_price:.2f} "
            f"(commission=${total_commission:.2f})"
        )

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=total_qty,
            fill_price=avg_price,
            fees=total_commission,
            slippage=0.0,  # Real fill — no simulated slippage
            filled_at=pd.Timestamp.now(),
            sleeve=order.sleeve,
            status=fill_status,
        )

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions from IBKR."""
        if not self._connected:
            return {}

        positions = {}
        for pos in self._ib.positions():
            sym = pos.contract.symbol
            positions[sym] = {
                "qty": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost,
            }
        return positions

    def get_account_summary(self) -> dict:
        """Get account value summary from IBKR."""
        if not self._connected:
            return {}

        summary = {}
        for item in self._ib.accountSummary():
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                           "GrossPositionValue", "UnrealizedPnL", "RealizedPnL"):
                summary[item.tag] = float(item.value)
        return summary

    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        if not self._connected:
            return None

        from ib_insync import Stock
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(2)

        price = ticker.marketPrice()
        self._ib.cancelMktData(contract)

        return float(price) if price and price > 0 else None

    def disconnect(self):
        """Clean disconnect from IBKR."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def __del__(self):
        self.disconnect()


def reconcile_positions(
    ibkr_broker: IBKRBroker,
    orchestrator_positions: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Reconcile internal position state with IBKR actual positions.
    Call this on startup to handle restarts.

    Returns dict of discrepancies:
        symbol -> {"internal": qty, "broker": qty, "action": "sync"|"close"|"unknown"}
    """
    broker_positions = ibkr_broker.get_positions()
    discrepancies = {}

    # Check all internal positions exist at broker
    for sym, pos in orchestrator_positions.items():
        broker_qty = broker_positions.get(sym, {}).get("qty", 0)
        internal_qty = pos.get("qty", 0)

        if abs(broker_qty - internal_qty) > 0.01:
            discrepancies[sym] = {
                "internal": internal_qty,
                "broker": broker_qty,
                "action": "sync" if broker_qty > 0 else "close",
            }

    # Check broker positions not in internal state
    for sym, bpos in broker_positions.items():
        if sym not in orchestrator_positions and bpos.get("qty", 0) != 0:
            discrepancies[sym] = {
                "internal": 0,
                "broker": bpos["qty"],
                "action": "unknown",
            }

    if discrepancies:
        logger.warning(f"Position discrepancies found: {discrepancies}")
    else:
        logger.info("Position reconciliation OK — all positions match")

    return discrepancies

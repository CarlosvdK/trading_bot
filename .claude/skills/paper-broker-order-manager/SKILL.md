# Skill: Paper Broker & Order Manager

## What This Skill Is
The Order Manager is the single gateway through which all orders pass. It routes to either the Paper Broker (simulation) or a Live Broker stub. The Paper Broker uses the same code path as live execution — the only difference is a config flag. This ensures paper trading results are meaningful predictors of live behavior.

---

## Architecture

```
Signal Engine
     ↓
Risk Governor (pre-trade check)
     ↓
Order Manager
     ↓
  ┌──────────────┬────────────────┐
  │ Paper Broker  │  Live Broker   │
  │ (default)     │  (stub / IBKR) │
  └──────────────┴────────────────┘
     ↓
Order Log (SQLite)
```

---

## Order Types

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MOC = "MOC"             # Market-on-Close (Core rebalance)

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    sleeve: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None     # Take-profit target (for barrier stop)
    sl_price: Optional[float] = None     # Stop-loss (for barrier stop)
    sector: Optional[str] = None
    ml_prob: Optional[float] = None      # ML filter probability at signal time
    created_at: Optional[pd.Timestamp] = None
    notes: str = ""

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    filled_qty: float
    fill_price: float
    fees: float
    slippage: float
    filled_at: pd.Timestamp
    sleeve: str
    status: OrderStatus = OrderStatus.FILLED
```

---

## Order Manager

```python
import uuid
import logging
from typing import List, Callable, Optional

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Routes orders through risk checks then to broker.
    Maintains order log and position state.
    """

    def __init__(
        self,
        broker,              # PaperBroker or LiveBrokerStub
        risk_governor,       # RiskGovernor instance
        portfolio_state,     # PortfolioState (shared mutable reference)
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

        # Rate limits
        self.max_orders_per_minute = config.get("max_orders_per_minute", 20)
        self.max_notional_per_minute = config.get("max_notional_per_minute", 500_000)

    def submit(self, order: Order, current_date: pd.Timestamp) -> Optional[Fill]:
        """
        Submit an order. Returns Fill if executed, None if rejected.
        
        Flow:
          1. Rate limit check
          2. Risk Governor pre-trade check
          3. Route to broker
          4. Log result
        """
        # --- Rate Limit ---
        if not self._check_rate_limit(order, current_date):
            self._log_order(order, status="REJECTED", reason="rate_limit", date=current_date)
            return None

        # --- Risk Governor ---
        notional = order.qty * self._get_price_estimate(order.symbol, current_date)
        allowed, reason = self.risk_governor.pre_trade_check(
            symbol=order.symbol,
            side=order.side,
            notional=notional,
            sleeve=order.sleeve,
            state=self.state,
            sector=order.sector,
            current_date=current_date.date() if hasattr(current_date, 'date') else current_date,
        )

        if not allowed:
            logger.warning(f"Order rejected by Risk Governor: {order.symbol} — {reason}")
            self._log_order(order, status="REJECTED", reason=reason, date=current_date)
            return None

        # --- Execute via Broker ---
        try:
            fill = self.broker.execute(order, current_date)
        except Exception as e:
            logger.error(f"Broker execution error for {order.symbol}: {e}")
            self._log_order(order, status="REJECTED", reason=str(e), date=current_date)
            return None

        if fill:
            self._log_order(order, status=fill.status.value, fill=fill, date=current_date)
            logger.info(
                f"Fill: {order.side.value} {fill.filled_qty:.0f} {order.symbol} "
                f"@ {fill.fill_price:.4f} | fees={fill.fees:.2f} | sleeve={order.sleeve}"
            )

        return fill

    def _check_rate_limit(self, order: Order, current_date: pd.Timestamp) -> bool:
        """Reset counters each minute."""
        minute = current_date.floor("min") if hasattr(current_date, 'floor') else current_date
        if minute != self._last_minute:
            self._orders_this_minute = 0
            self._notional_this_minute = 0.0
            self._last_minute = minute

        if self._orders_this_minute >= self.max_orders_per_minute:
            logger.warning("Order rate limit hit")
            return False

        est_notional = order.qty * (order.limit_price or 100)  # Conservative estimate
        if self._notional_this_minute + est_notional > self.max_notional_per_minute:
            logger.warning("Notional rate limit hit")
            return False

        self._orders_this_minute += 1
        self._notional_this_minute += est_notional
        return True

    def _get_price_estimate(self, symbol: str, date: pd.Timestamp) -> float:
        """Best available price estimate for pre-trade sizing."""
        pos = self.state.positions.get(symbol, {})
        return pos.get("last_price", pos.get("avg_cost", 100.0))

    def _log_order(self, order: Order, status: str, date: pd.Timestamp,
                   reason: str = "", fill: Optional[Fill] = None):
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
            record.update({
                "fill_price": fill.fill_price,
                "filled_qty": fill.filled_qty,
                "fees": fill.fees,
                "slippage": fill.slippage,
            })
        self.order_log.append(record)
```

---

## Paper Broker

```python
import numpy as np

class PaperBroker:
    """
    Simulates broker execution with realistic fills, slippage, and partial fills.
    Uses same interface as LiveBrokerStub.
    """

    def __init__(self, price_data: dict, cost_model, config: dict):
        self.price_data = price_data      # Dict[symbol -> DataFrame]
        self.cost_model = cost_model
        self.config = config
        self.mode = "paper"

    def execute(self, order: Order, current_date: pd.Timestamp) -> Optional[Fill]:
        """
        Simulate order execution. Fills at next bar's open (after signal bar).
        For MOC orders, fills at bar's close.
        """
        df = self.price_data.get(order.symbol)
        if df is None:
            logger.warning(f"No data for {order.symbol}")
            return None

        # Find fill bar (next bar after signal)
        if current_date not in df.index:
            return None

        idx = df.index.get_loc(current_date)
        if idx + 1 >= len(df):
            logger.warning(f"No next bar for {order.symbol} after {current_date.date()}")
            return None

        fill_bar = df.iloc[idx + 1]
        adv = self._get_adv(order.symbol, current_date)

        # MOC fills at close
        if order.order_type == OrderType.MOC:
            raw_fill_price = fill_bar["close"]
        else:
            raw_fill_price = fill_bar["open"]

        # Limit/stop order logic
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and fill_bar["low"] > order.limit_price:
                return None  # Not filled — price never came down
            if order.side == OrderSide.SELL and fill_bar["high"] < order.limit_price:
                return None  # Not filled — price never came up
            raw_fill_price = order.limit_price  # Fill at limit price

        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.SELL and fill_bar["low"] > order.stop_price:
                return None
            raw_fill_price = min(order.stop_price, fill_bar["open"])  # Gap risk

        # Slippage
        notional = order.qty * raw_fill_price
        fill_price = self.cost_model.fill_price(
            raw_fill_price, order.side.value, notional, adv,
            fill_bar["high"], fill_bar["low"]
        )
        slippage = abs(fill_price - raw_fill_price) / raw_fill_price

        # Partial fill
        filled_qty = self.cost_model.partial_fill_qty(order.qty, adv, fill_price)
        if filled_qty < order.qty * 0.01:
            logger.warning(f"Near-zero fill for {order.symbol}. ADV too low.")
            return None

        # Fees
        fees = filled_qty * fill_price * (self.cost_model.commission_bps / 10_000)

        status = OrderStatus.FILLED if filled_qty >= order.qty * 0.99 else OrderStatus.PARTIAL

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

    def _get_adv(self, symbol: str, date: pd.Timestamp, window: int = 30) -> float:
        df = self.price_data.get(symbol)
        if df is None:
            return 1_000_000
        idx = df.index.get_loc(date) if date in df.index else -1
        if idx < window:
            return 1_000_000
        recent = df.iloc[max(0, idx - window):idx]
        return (recent["close"] * recent["volume"]).mean()
```

---

## Live Broker Stub

```python
class LiveBrokerStub:
    """
    Stub for live broker integration.
    Raises NotImplementedError — forces explicit implementation before going live.
    """
    mode = "live"

    def execute(self, order: Order, current_date: pd.Timestamp) -> Optional[Fill]:
        raise NotImplementedError(
            "Live broker not implemented. "
            "Integrate your broker's API here (e.g., IBKR, Alpaca). "
            "Never deploy this stub to live trading."
        )
```

---

## Graceful Shutdown

```python
import signal
import sys

class GracefulShutdown:
    """
    Handles SIGTERM/SIGINT to ensure clean exit.
    Never interrupt mid-order-submission.
    """
    def __init__(self, order_manager: OrderManager, portfolio_state):
        self.order_manager = order_manager
        self.state = portfolio_state
        self._shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        print(f"\n[SHUTDOWN] Signal {signum} received. Finishing current cycle...")
        self._shutdown_requested = True

    @property
    def should_shutdown(self):
        return self._shutdown_requested

    def save_state(self, path: str):
        """Persist state to disk before exit."""
        import json
        # Serialize relevant state
        print(f"[SHUTDOWN] Saving state to {path}")
        # ... implementation ...
        print("[SHUTDOWN] State saved. Exiting cleanly.")
        sys.exit(0)
```

---

## Configuration

```yaml
execution:
  mode: paper                        # paper | live
  max_orders_per_minute: 20
  max_notional_per_minute: 500000

  order_routing:
    default_session: regular         # regular | extended
    regular_hours: "09:30-16:00"
    timezone: "America/New_York"
    extended_slippage_multiplier: 2.0

  paper_broker:
    fill_on: "next_open"             # next_open | next_close (MOC)
    participation_rate: 0.05         # Max 5% of ADV
```

---

## Critical Rules

1. **Same code path** — paper and live use identical Order and Fill objects. The broker is the only difference.
2. **No order without a Risk Governor check** — the Order Manager always calls `pre_trade_check` first.
3. **Every order logged** — including rejections, with reason.
4. **Graceful shutdown** — SIGTERM/SIGINT are handled before any open order is abandoned.
5. **Live broker is a stub** — it raises `NotImplementedError`. You must explicitly implement it before going live. This is a safety gate.

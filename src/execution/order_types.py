"""
Order and Fill data types shared by Order Manager, Paper Broker, and Live Broker.
Skill reference: .claude/skills/paper-broker-order-manager/SKILL.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MOC = "MOC"  # Market-on-Close (Core rebalance)


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
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    sector: Optional[str] = None
    ml_prob: Optional[float] = None
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

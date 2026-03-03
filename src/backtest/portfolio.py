"""
Portfolio accounting — positions and sleeve accounts.
Skill reference: .claude/skills/backtesting-engine/SKILL.md
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    sleeve: str
    qty: float
    avg_cost: float
    entry_date: pd.Timestamp
    sector: Optional[str] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.avg_cost

    def unrealized_pnl(self, current_price: float) -> float:
        return self.qty * (current_price - self.avg_cost)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost * (
            1 if self.qty > 0 else -1
        )


class SleeveAccount:
    """Per-sleeve portfolio accounting."""

    def __init__(self, sleeve_name: str, initial_cash: float):
        self.sleeve_name = sleeve_name
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.trades_log: list = []
        self.peak_value: float = initial_cash

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        """Current NAV including unrealized P&L."""
        mtm = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.avg_cost)
            mtm += pos.qty * price
        return mtm

    def open_position(
        self,
        symbol: str,
        qty: float,
        fill_price: float,
        fees: float,
        date: pd.Timestamp,
        sector: Optional[str] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
    ):
        notional = abs(qty) * fill_price
        cost = notional + fees
        if cost > self.cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
            )

        self.cash -= cost
        self.total_fees += fees
        self.positions[symbol] = Position(
            symbol=symbol,
            sleeve=self.sleeve_name,
            qty=qty,
            avg_cost=fill_price,
            entry_date=date,
            sector=sector,
            stop_price=stop_price,
            target_price=target_price,
        )
        self._log_trade("OPEN", symbol, qty, fill_price, fees, date)

    def close_position(
        self,
        symbol: str,
        fill_price: float,
        fees: float,
        date: pd.Timestamp,
    ) -> float:
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")

        pos = self.positions.pop(symbol)
        pnl = pos.qty * (fill_price - pos.avg_cost) - fees
        proceeds = abs(pos.qty) * fill_price - fees

        self.cash += proceeds
        self.realized_pnl += pnl
        self.total_fees += fees
        self._log_trade(
            "CLOSE", symbol, -pos.qty, fill_price, fees, date, pnl=pnl
        )
        return pnl

    def _log_trade(
        self, action, symbol, qty, price, fees, date, pnl=None
    ):
        self.trades_log.append(
            {
                "date": date,
                "sleeve": self.sleeve_name,
                "action": action,
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "fees": fees,
                "pnl": pnl,
            }
        )

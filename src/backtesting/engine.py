"""
Multi-sleeve backtesting engine.
Skill reference: .claude/skills/backtesting-engine/SKILL.md
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.backtesting.cost_model import CostModel
from src.backtesting.portfolio import SleeveAccount

logger = logging.getLogger(__name__)


class Backtester:
    """
    Multi-sleeve portfolio backtester with realistic cost model.
    Supports walk-forward ML retraining.
    """

    def __init__(self, config: dict, cost_model: CostModel):
        self.config = config
        self.cost_model = cost_model
        self.benchmark_symbol = config.get("benchmark_symbol", "SPY")

        initial_nav = config["initial_nav"]
        allocs = config.get(
            "sleeve_allocations", {"core": 0.6, "swing": 0.3, "cash_buffer": 0.1}
        )

        self.sleeves = {
            "core": SleeveAccount(
                "core", initial_nav * allocs.get("core", 0.6)
            ),
            "swing": SleeveAccount(
                "swing", initial_nav * allocs.get("swing", 0.3)
            ),
            "cash": SleeveAccount(
                "cash", initial_nav * allocs.get("cash_buffer", 0.1)
            ),
        }

        self.date_range: pd.DatetimeIndex = None
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.settlement_queue: list = []
        self.nav_history: list = []

    def load_data(self, price_data: Dict[str, pd.DataFrame]):
        """Load OHLCV DataFrames keyed by symbol."""
        self.price_data = price_data
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)
        self.date_range = pd.DatetimeIndex(sorted(all_dates))

    def get_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Get close prices for all symbols on a given date."""
        prices = {}
        for sym, df in self.price_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "close"]
        return prices

    def get_adv(
        self, symbol: str, date: pd.Timestamp, window: int = 30
    ) -> float:
        """30-day average daily volume in notional terms."""
        df = self.price_data.get(symbol)
        if df is None:
            return 1e6
        if date not in df.index:
            return 1e6
        idx = df.index.get_loc(date)
        if idx < window:
            return 1e6
        recent = df.iloc[max(0, idx - window) : idx]
        return (recent["close"] * recent["volume"]).mean()

    def _process_settlements(self, current_date: pd.Timestamp):
        """Apply settled proceeds to available cash."""
        ready = [
            (d, sl, amt)
            for d, sl, amt in self.settlement_queue
            if d <= current_date
        ]
        for settle_date, sleeve_name, amount in ready:
            self.sleeves[sleeve_name].cash += amount
        self.settlement_queue = [
            (d, sl, amt)
            for d, sl, amt in self.settlement_queue
            if d > current_date
        ]

    def total_nav(self, prices: Dict[str, float]) -> float:
        return sum(
            acc.mark_to_market(prices) for acc in self.sleeves.values()
        )

    def run(
        self,
        signal_func,
        ml_retrain_func=None,
        wf_boundaries: list = None,
    ) -> dict:
        """Main simulation loop."""
        wf_set = set(wf_boundaries or [])
        peak_nav = None

        for i, date in enumerate(self.date_range):
            prices = self.get_prices(date)
            if not prices:
                continue

            self._process_settlements(date)

            if ml_retrain_func and date in wf_set:
                logger.info(f"Retraining ML models at {date.date()}")
                ml_retrain_func(train_end_date=date)

            try:
                orders = signal_func(date, prices, self.sleeves)
            except Exception as e:
                logger.error(f"Signal error on {date.date()}: {e}")
                orders = []

            for order in orders:
                self._execute_order(order, prices, date)

            self._check_barriers(prices, date)

            nav = self.total_nav(prices)
            if peak_nav is None:
                peak_nav = nav
            peak_nav = max(peak_nav, nav)
            drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0

            self.nav_history.append(
                {
                    "date": date,
                    "nav": nav,
                    "drawdown": drawdown,
                    "core_nav": self.sleeves["core"].mark_to_market(prices),
                    "swing_nav": self.sleeves["swing"].mark_to_market(
                        prices
                    ),
                }
            )

        return self._compute_results()

    def _execute_order(
        self, order: dict, prices: Dict[str, float], date: pd.Timestamp
    ):
        symbol = order["symbol"]
        sleeve_name = order["sleeve"]
        side = order["side"]
        qty = order["qty"]

        if symbol not in self.price_data:
            return
        df = self.price_data[symbol]
        if date not in df.index:
            return
        bar = df.loc[date]
        adv = self.get_adv(symbol, date)
        notional = qty * bar["open"]

        filled_qty = self.cost_model.partial_fill_qty(
            qty, adv, bar["open"]
        )

        fill_price = self.cost_model.fill_price(
            bar["open"],
            side,
            filled_qty * bar["open"],
            adv,
            bar["high"],
            bar["low"],
        )
        fees_notional = (
            filled_qty * fill_price * (self.cost_model.commission_bps / 10_000)
        )

        sleeve = self.sleeves[sleeve_name]
        try:
            if side == "BUY":
                sleeve.open_position(
                    symbol,
                    filled_qty,
                    fill_price,
                    fees_notional,
                    date,
                    sector=order.get("sector"),
                    stop_price=order.get("stop_price"),
                    target_price=order.get("target_price"),
                )
            elif side == "SELL":
                pnl = sleeve.close_position(
                    symbol, fill_price, fees_notional, date
                )
                proceeds = filled_qty * fill_price - fees_notional
                settle_date = date + pd.offsets.BDay(
                    self.cost_model.settlement_lag_days
                )
                self.settlement_queue.append(
                    (settle_date, sleeve_name, proceeds)
                )
                sleeve.cash -= proceeds
        except ValueError as e:
            logger.warning(f"Order rejected: {e}")

    def _check_barriers(
        self, prices: Dict[str, float], date: pd.Timestamp
    ):
        """Auto-close positions that hit stop or target."""
        for sleeve in self.sleeves.values():
            for symbol, pos in list(sleeve.positions.items()):
                price = prices.get(symbol)
                if price is None:
                    continue
                if pos.stop_price and price <= pos.stop_price:
                    fees = price * abs(pos.qty) * 0.0005
                    sleeve.close_position(symbol, price, fees, date)
                elif pos.target_price and price >= pos.target_price:
                    fees = price * abs(pos.qty) * 0.0005
                    sleeve.close_position(symbol, price, fees, date)

    def _compute_results(self) -> dict:
        """Compute standard performance metrics."""
        if not self.nav_history:
            return {"total_return": 0, "nav_history": pd.DataFrame()}

        nav_df = pd.DataFrame(self.nav_history).set_index("date")
        daily_ret = nav_df["nav"].pct_change().dropna()

        sharpe = (
            daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            if daily_ret.std() > 0
            else 0
        )
        total_return = (
            (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) - 1
        )
        max_dd = nav_df["drawdown"].min()
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = (daily_ret > 0).mean()

        all_trades = []
        for sleeve in self.sleeves.values():
            all_trades.extend(sleeve.trades_log)
        trades_df = pd.DataFrame(all_trades)

        return {
            "total_return": total_return,
            "annualized_sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate_daily": win_rate,
            "total_trades": len(trades_df),
            "total_fees": sum(
                s.total_fees for s in self.sleeves.values()
            ),
            "nav_history": nav_df,
            "trades": trades_df,
        }

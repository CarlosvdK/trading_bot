---
name: backtesting-engine
description: Realistic multi-sleeve backtesting engine + cost model (slippage, partial fills, T+2 settlement, conservative accounting). Use whenever implementing simulation, fills, fees, NAV, or performance metrics.
---

# Skill: Backtesting Engine & Cost Model

## When to use this skill
Use this skill whenever you are:
- implementing or modifying the backtest/simulation loop
- modeling fills, slippage, spread, commissions, regulatory fees, market impact
- handling partial fills, settlement lag, cash availability
- doing portfolio accounting (positions, avg cost, realized/unrealized P&L, sleeve attribution)
- producing performance metrics and trade logs

## Non-negotiable invariants (must hold)
- Fills occur at **next bar open** after signal. Never fill on the signal bar close.
- **T+2 settlement**: sale proceeds are not deployable until settlement date.
- **Partial fills** for large orders vs ADV (do not assume full liquidity).
- Costs must bias toward **harder-to-beat backtests** (conservative slippage/fees).
- No future prices may be referenced (no look-ahead).

## Required outputs (what you must implement)
- `CostModel` with `fill_price`, `partial_fill_qty`, and round-trip cost estimator
- Portfolio accounting: `Position`, `SleeveAccount` with trade log + MTM NAV
- `Backtester` that runs daily loop, executes orders, processes settlement queue, checks barriers
- Results: NAV history dataframe, trades dataframe, core/swing attribution, basic metrics

## Required tests (minimum)
Create or update tests to cover:
- Fill price never exceeds bar high on BUY / never below bar low on SELL
- Partial fill caps notional to participation_rate * ADV
- T+2 settlement: proceeds unavailable until settlement date (cash decreases then later increases)
- No fill at signal bar close: verify next bar open usage
- Fees applied and recorded; realized pnl matches (entry/exit/fees)

# Skill: Backtesting Engine & Cost Model

## What This Skill Is
A realistic backtesting engine that simulates portfolio accounting, order fills with costs, partial fills, settlement lag, and per-sleeve P&L attribution. The key principle: **every design choice should make the backtest harder to beat, not easier.**

---

## Cost Model

### Why Cost Modeling Is Critical
A 10-day swing trade needs >15 bps of gross alpha just to break even. Getting this wrong is the most common reason a backtest looks great but live trading loses money.

### Cost Components

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class CostModel:
    """
    Realistic transaction cost model for equity/ETF trading.
    All costs in basis points (bps) unless noted.
    """
    commission_bps: float = 0.5          # Per side. Use 0.5 even for "free" brokers
    regulatory_fee_bps: float = 0.3      # SEC/FINRA fees on sells only
    spread_cost_bps: float = 1.0         # Half-spread for aggressive orders
    market_impact_factor: float = 0.1    # Coefficient in sqrt(order/ADV) model
    borrow_cost_bps_annual: float = 100  # Short selling borrow (annualized)
    settlement_lag_days: int = 2         # T+2 for equities

    def total_roundtrip_bps(
        self,
        order_notional: float,
        adv_notional: float,
        holding_days: int = 10,
        is_short: bool = False,
    ) -> float:
        """
        Total round-trip cost estimate in bps.
        """
        participation = order_notional / max(adv_notional, 1)
        impact = self.market_impact_factor * np.sqrt(participation) * 10_000  # to bps

        entry_cost = (
            self.commission_bps +
            self.spread_cost_bps +
            impact
        )
        exit_cost = (
            self.commission_bps +
            self.regulatory_fee_bps +    # Regulatory on sell
            self.spread_cost_bps +
            impact
        )
        borrow_cost = (
            self.borrow_cost_bps_annual * holding_days / 252
            if is_short else 0.0
        )

        return entry_cost + exit_cost + borrow_cost

    def fill_price(
        self,
        bar_open: float,
        side: str,         # "BUY" or "SELL"
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
            return min(fill, bar_high)   # Can't fill above bar high
        else:
            fill = bar_open * (1 - impact_pct - spread_pct)
            return max(fill, bar_low)    # Can't fill below bar low

    def partial_fill_qty(
        self,
        order_qty: float,
        adv_notional: float,
        fill_price: float,
        participation_rate: float = 0.05,
    ) -> float:
        """
        Simulate partial fills. Large orders vs ADV fill partially.
        participation_rate: max % of ADV we'll consume per order.
        """
        max_fillable_notional = participation_rate * adv_notional
        max_fillable_qty = max_fillable_notional / fill_price
        return min(order_qty, max_fillable_qty)
```

---

## Portfolio Accounting

```python
from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    sleeve: str
    qty: float              # Negative = short
    avg_cost: float         # Average fill price
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
        return (current_price - self.avg_cost) / self.avg_cost * (1 if self.qty > 0 else -1)


class SleeveAccount:
    """Per-sleeve portfolio accounting."""

    def __init__(self, sleeve_name: str, initial_cash: float):
        self.sleeve_name = sleeve_name
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.trades_log: list = []

        # For drawdown tracking
        self.peak_value: float = initial_cash

    @property
    def market_value(self) -> float:
        return sum(p.qty * p.avg_cost for p in self.positions.values())  # Use cost; mark-to-market in NAV calc

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
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")

        self.cash -= cost
        self.total_fees += fees
        self.positions[symbol] = Position(
            symbol=symbol, sleeve=self.sleeve_name, qty=qty,
            avg_cost=fill_price, entry_date=date,
            sector=sector, stop_price=stop_price, target_price=target_price,
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
        self._log_trade("CLOSE", symbol, -pos.qty, fill_price, fees, date, pnl=pnl)
        return pnl

    def _log_trade(self, action, symbol, qty, price, fees, date, pnl=None):
        self.trades_log.append({
            "date": date, "sleeve": self.sleeve_name, "action": action,
            "symbol": symbol, "qty": qty, "price": price, "fees": fees, "pnl": pnl,
        })
```

---

## Backtesting Engine

```python
class Backtester:
    """
    Multi-sleeve portfolio backtester with realistic cost model.
    Supports walk-forward ML retraining.
    """

    def __init__(self, config: dict, cost_model: CostModel):
        self.config = config
        self.cost_model = cost_model
        self.benchmark_symbol = config.get("benchmark_symbol", "SPY")

        # Initialize sleeves
        initial_nav = config["initial_nav"]
        allocs = config["sleeve_allocations"]

        self.sleeves = {
            "core":  SleeveAccount("core",  initial_nav * allocs.get("core", 0.6)),
            "swing": SleeveAccount("swing", initial_nav * allocs.get("swing", 0.3)),
            "cash":  SleeveAccount("cash",  initial_nav * allocs.get("cash_buffer", 0.1)),
        }

        self.date_range: pd.DatetimeIndex = None
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.settlement_queue: list = []   # (settlement_date, sleeve, cash_amount)
        self.nav_history: list = []
        self.benchmark_history: list = []

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

    def get_adv(self, symbol: str, date: pd.Timestamp, window: int = 30) -> float:
        """30-day average daily volume in notional terms."""
        df = self.price_data.get(symbol)
        if df is None:
            return 1e6  # Default $1M ADV
        idx = df.index.get_loc(date) if date in df.index else -1
        if idx < window:
            return 1e6
        recent = df.iloc[max(0, idx - window):idx]
        return (recent["close"] * recent["volume"]).mean()

    def _process_settlements(self, current_date: pd.Timestamp):
        """Apply settled proceeds to available cash."""
        ready = [(d, sl, amt) for d, sl, amt in self.settlement_queue if d <= current_date]
        for settle_date, sleeve_name, amount in ready:
            self.sleeves[sleeve_name].cash += amount
        self.settlement_queue = [
            (d, sl, amt) for d, sl, amt in self.settlement_queue if d > current_date
        ]

    def total_nav(self, prices: Dict[str, float]) -> float:
        return sum(acc.mark_to_market(prices) for acc in self.sleeves.values())

    def run(
        self,
        signal_func,        # Callable(date, prices, state) -> list of order dicts
        ml_retrain_func=None,  # Optional: called at walk-forward boundaries
        wf_boundaries: list = None,  # Walk-forward retrain dates
    ) -> dict:
        """
        Main simulation loop.
        """
        prices_history = {}
        wf_set = set(wf_boundaries or [])
        peak_nav = None

        for i, date in enumerate(self.date_range):
            prices = self.get_prices(date)
            if not prices:
                continue

            # Settlement
            self._process_settlements(date)

            # Walk-forward ML retrain
            if ml_retrain_func and date in wf_set:
                logger.info(f"Retraining ML models at {date.date()}")
                ml_retrain_func(train_end_date=date)

            # Generate signals
            try:
                orders = signal_func(date, prices, self.sleeves)
            except Exception as e:
                logger.error(f"Signal error on {date.date()}: {e}")
                orders = []

            # Execute orders
            for order in orders:
                self._execute_order(order, prices, date)

            # Check stop/target barriers
            self._check_barriers(prices, date)

            # Mark to market
            nav = self.total_nav(prices)
            if peak_nav is None:
                peak_nav = nav
            peak_nav = max(peak_nav, nav)
            drawdown = (nav - peak_nav) / peak_nav

            self.nav_history.append({
                "date": date,
                "nav": nav,
                "drawdown": drawdown,
                "core_nav": self.sleeves["core"].mark_to_market(prices),
                "swing_nav": self.sleeves["swing"].mark_to_market(prices),
            })

        return self._compute_results()

    def _execute_order(self, order: dict, prices: Dict[str, float], date: pd.Timestamp):
        symbol = order["symbol"]
        sleeve_name = order["sleeve"]
        side = order["side"]        # BUY, SELL
        qty = order["qty"]

        if symbol not in prices:
            logger.warning(f"No price for {symbol} on {date.date()}, skipping")
            return

        df = self.price_data[symbol]
        if date not in df.index:
            return
        bar = df.loc[date]
        adv = self.get_adv(symbol, date)
        notional = qty * bar["open"]

        # Partial fill
        filled_qty = self.cost_model.partial_fill_qty(qty, adv, bar["open"])
        if filled_qty < qty * 0.5:
            logger.warning(f"Partial fill: {symbol} {filled_qty:.0f}/{qty:.0f} on {date.date()}")

        fill_price = self.cost_model.fill_price(
            bar["open"], side, filled_qty * bar["open"], adv, bar["high"], bar["low"]
        )
        fees_notional = filled_qty * fill_price * (self.cost_model.commission_bps / 10_000)

        sleeve = self.sleeves[sleeve_name]
        try:
            if side == "BUY":
                sleeve.open_position(
                    symbol, filled_qty, fill_price, fees_notional, date,
                    sector=order.get("sector"),
                    stop_price=order.get("stop_price"),
                    target_price=order.get("target_price"),
                )
            elif side == "SELL":
                pnl = sleeve.close_position(symbol, fill_price, fees_notional, date)
                # Queue settlement (T+2)
                proceeds = filled_qty * fill_price - fees_notional
                settle_date = date + pd.offsets.BDay(self.cost_model.settlement_lag_days)
                self.settlement_queue.append((settle_date, sleeve_name, proceeds))
                sleeve.cash -= proceeds  # Remove from cash until settled
        except ValueError as e:
            logger.warning(f"Order rejected: {e}")

    def _check_barriers(self, prices: Dict[str, float], date: pd.Timestamp):
        """Auto-close positions that hit stop or target."""
        for sleeve in self.sleeves.values():
            for symbol, pos in list(sleeve.positions.items()):
                price = prices.get(symbol)
                if price is None:
                    continue
                if pos.stop_price and price <= pos.stop_price:
                    logger.info(f"Stop hit: {symbol} @ {price:.2f} (stop={pos.stop_price:.2f})")
                    sleeve.close_position(symbol, price, price * 0.0005, date)
                elif pos.target_price and price >= pos.target_price:
                    logger.info(f"Target hit: {symbol} @ {price:.2f}")
                    sleeve.close_position(symbol, price, price * 0.0005, date)

    def _compute_results(self) -> dict:
        """Compute standard performance metrics."""
        nav_df = pd.DataFrame(self.nav_history).set_index("date")
        daily_ret = nav_df["nav"].pct_change().dropna()

        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        total_return = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) - 1
        max_dd = nav_df["drawdown"].min()
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = (daily_ret > 0).mean()

        all_trades = []
        for sleeve in self.sleeves.values():
            all_trades.extend(sleeve.trades_log)
        trades_df = pd.DataFrame(all_trades)

        results = {
            "total_return": total_return,
            "annualized_sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate_daily": win_rate,
            "total_trades": len(trades_df),
            "total_fees": sum(s.total_fees for s in self.sleeves.values()),
            "nav_history": nav_df,
            "trades": trades_df,
        }

        print(f"\n{'='*50}")
        print(f"Backtest Results")
        print(f"{'='*50}")
        print(f"Total Return:      {total_return:.2%}")
        print(f"Annualized Sharpe: {sharpe:.3f}")
        print(f"Max Drawdown:      {max_dd:.2%}")
        print(f"Calmar Ratio:      {calmar:.3f}")
        print(f"Total Trades:      {len(trades_df)}")
        print(f"Total Fees:        ${results['total_fees']:,.2f}")

        return results
```

---

## Configuration

```yaml
backtest:
  initial_nav: 100000
  benchmark_symbol: "SPY"
  sleeve_allocations:
    core: 0.60
    swing: 0.30
    cash_buffer: 0.10

cost_model:
  commission_bps: 0.5
  regulatory_fee_bps: 0.3
  spread_cost_bps: 1.0
  market_impact_factor: 0.1
  borrow_cost_bps_annual: 100
  participation_rate: 0.05        # Max 5% of ADV per order
  settlement_lag_days: 2
```

---

## Backtest Integrity Rules

1. **Never use adjusted prices for fills** — fills happen at raw prices. Use adjusted prices for signal generation and return calculation.
2. **T+2 settlement** — don't reinvest sale proceeds until 2 business days after sale.
3. **Partial fills** — for any order > 1% of ADV, apply partial fill model.
4. **No future bar prices** — fill at next bar's open after signal, not at signal bar's close.
5. **Walk-forward ML** — retrain models at each fold boundary using only data up to that date.

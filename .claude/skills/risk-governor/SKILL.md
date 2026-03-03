---
name: risk-governor
description: Risk Governor — highest-priority component. Enforces drawdown kill-switch, daily loss halt, position limits, PDT rules, exposure caps. Use whenever implementing or modifying risk checks, kill-switches, position limits, exposure controls, or PDT enforcement.
triggers:
  - risk limits
  - drawdown
  - kill-switch
  - PDT rule
  - exposure cap
  - position limit
  - max loss
  - risk governor
  - pre-trade check
priority: P0
---

# Skill: Risk Governor

## What This Skill Is
The Risk Governor is the highest-priority component of the entire trading system. It sits above all sleeves and all execution logic. No order reaches the broker without passing through the Risk Governor. It enforces hard limits on drawdown, exposure, position sizing, and per-sleeve losses — and triggers kill-switches when breaches occur.

**Implement and test this first. Everything else is subordinate to it.**

---

## Architecture Overview

```
Signal Engine (Core / Swing)
         ↓
   Risk Governor  ←── Portfolio State
         ↓
   Order Manager
         ↓
   Paper / Live Broker
```

The Risk Governor has two modes:
1. **Pre-trade check** — Called before any order is submitted. Returns `(allowed: bool, reason: str)`.
2. **Post-trade / periodic check** — Called after fills and on a schedule. May trigger kill-switches.

---

## Risk Limits Hierarchy

```
Level 0: Portfolio Kill-Switch
  max_portfolio_drawdown: -15% peak-to-trough → halt ALL orders, 5-day cooldown

Level 1: Portfolio Daily
  max_daily_loss_pct: -3% of NAV → halt new orders for remainder of day

Level 2: Swing Sleeve
  max_swing_weekly_loss: -5% of swing sleeve allocation → halt swing for week
  max_concurrent_positions: 10 → reject new signals when at cap

Level 3: Per-Position
  max_position_pct_swing: 15% of swing sleeve → reject oversized entry
  max_sector_pct_swing: 30% of swing sleeve → reject sector-concentrated entry
  per_trade_hard_stop: 2x SL barrier → place stop order at entry

Level 4: Exposure
  max_gross_exposure_pct: 100% of NAV → reject if would exceed
  max_cross_sleeve_corr: 0.70 → halve swing allocation if breached

Level 5: Regulatory
  pdt_day_trade_limit: 3 round-trips per 5 business days (if account < $25,000)
```

---

## Implementation

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from enum import Enum
from datetime import date, timedelta


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class KillSwitchReason(str, Enum):
    PORTFOLIO_DRAWDOWN = "portfolio_max_drawdown"
    DAILY_LOSS = "daily_max_loss"
    SWING_WEEKLY_LOSS = "swing_weekly_max_loss"
    MANUAL = "manual"


@dataclass
class RiskConfig:
    # Portfolio-level
    max_portfolio_drawdown: float = 0.15       # 15% peak-to-trough
    max_daily_loss_pct: float = 0.03           # 3% of NAV
    manual_restart_required: bool = True
    kill_switch_cooldown_days: int = 5

    # Swing sleeve
    max_swing_weekly_loss: float = 0.05        # 5% of swing sleeve NAV
    max_concurrent_positions: int = 10
    max_position_pct_swing: float = 0.15       # 15% of swing sleeve
    max_sector_pct_swing: float = 0.30         # 30% of swing sleeve
    per_trade_stop_multiplier: float = 2.0     # 2x SL barrier

    # Exposure
    max_gross_exposure_pct: float = 1.0        # 100% of NAV
    max_cross_sleeve_corr: float = 0.70

    # PDT (US equities)
    enforce_pdt_rule: bool = True
    pdt_account_threshold: float = 25_000.0    # USD
    pdt_max_day_trades: int = 3

    # Core sleeve
    core_rebalance_band: float = 0.05          # 5% drift triggers rebalance
    core_min_adv_usd: float = 1_000_000        # $1M ADV minimum for Core


@dataclass
class PortfolioState:
    nav: float                                  # Current NAV
    peak_nav: float                             # Highest NAV ever recorded
    cash: float
    sleeve_values: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, Dict] = field(default_factory=dict)
    day_start_nav: float = 0.0
    week_start_swing_nav: float = 0.0
    account_value: float = 0.0                  # For PDT check
    day_trade_dates: list = field(default_factory=list)  # Dates of round-trips

    @property
    def current_drawdown(self) -> float:
        if self.peak_nav == 0:
            return 0.0
        return (self.nav - self.peak_nav) / self.peak_nav

    @property
    def daily_pnl_pct(self) -> float:
        if self.day_start_nav == 0:
            return 0.0
        return (self.nav - self.day_start_nav) / self.day_start_nav


class RiskGovernor:
    """
    Central risk enforcement layer.
    All methods are synchronous and deterministic.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.kill_switch_active: bool = False
        self.kill_switch_reason: Optional[KillSwitchReason] = None
        self.kill_switch_date: Optional[date] = None
        self.swing_halted_until: Optional[date] = None
        self.daily_halt_until: Optional[date] = None

    # ------------------------------------------------------------------ #
    #  PRE-TRADE CHECK                                                     #
    # ------------------------------------------------------------------ #

    def pre_trade_check(
        self,
        symbol: str,
        side: OrderSide,
        notional: float,
        sleeve: str,
        state: PortfolioState,
        sector: Optional[str] = None,
        sl_barrier_pct: Optional[float] = None,
        current_date: date = None,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Must return (False, reason) for ANY breach.
        """
        if current_date is None:
            current_date = date.today()

        # --- Kill switch ---
        if self.kill_switch_active:
            if self.config.manual_restart_required:
                return False, f"Kill switch active ({self.kill_switch_reason}). Manual restart required."
            cooldown_end = self.kill_switch_date + timedelta(days=self.config.kill_switch_cooldown_days)
            if current_date < cooldown_end:
                return False, f"Kill switch cooldown until {cooldown_end}"

        # --- Daily halt ---
        if self.daily_halt_until and current_date <= self.daily_halt_until:
            return False, f"Daily loss halt active until {self.daily_halt_until}"

        # --- Swing halt ---
        if sleeve == "swing" and self.swing_halted_until and current_date <= self.swing_halted_until:
            return False, f"Swing sleeve halted until {self.swing_halted_until}"

        # --- Portfolio drawdown ---
        dd = state.current_drawdown
        if dd <= -self.config.max_portfolio_drawdown:
            self._trigger_kill_switch(KillSwitchReason.PORTFOLIO_DRAWDOWN, current_date)
            return False, f"Portfolio drawdown {dd:.1%} exceeds limit {-self.config.max_portfolio_drawdown:.1%}"

        # --- Daily loss ---
        daily_loss = state.daily_pnl_pct
        if daily_loss <= -self.config.max_daily_loss_pct:
            self.daily_halt_until = current_date
            return False, f"Daily loss {daily_loss:.1%} exceeds limit {-self.config.max_daily_loss_pct:.1%}"

        # --- Swing weekly loss ---
        if sleeve == "swing":
            swing_nav = state.sleeve_values.get("swing", 0)
            swing_week_start = state.week_start_swing_nav
            if swing_week_start > 0:
                swing_weekly_loss = (swing_nav - swing_week_start) / swing_week_start
                if swing_weekly_loss <= -self.config.max_swing_weekly_loss:
                    # Halt until end of week
                    days_to_friday = (4 - current_date.weekday()) % 7
                    self.swing_halted_until = current_date + timedelta(days=days_to_friday)
                    return False, f"Swing weekly loss {swing_weekly_loss:.1%} exceeds limit"

        # --- Concurrent positions (Swing) ---
        if sleeve == "swing" and side in (OrderSide.BUY, OrderSide.SHORT):
            swing_positions = [p for p in state.positions.values() if p.get('sleeve') == 'swing']
            if len(swing_positions) >= self.config.max_concurrent_positions:
                return False, f"Max concurrent swing positions ({self.config.max_concurrent_positions}) reached"

        # --- Position size (Swing) ---
        if sleeve == "swing":
            swing_alloc = state.sleeve_values.get("swing", state.nav * 0.3)
            if notional / swing_alloc > self.config.max_position_pct_swing:
                return False, (
                    f"Position size {notional/swing_alloc:.1%} exceeds "
                    f"max {self.config.max_position_pct_swing:.1%} of swing sleeve"
                )

        # --- Sector concentration (Swing) ---
        if sleeve == "swing" and sector:
            swing_alloc = state.sleeve_values.get("swing", state.nav * 0.3)
            sector_exposure = sum(
                p['notional'] for p in state.positions.values()
                if p.get('sleeve') == 'swing' and p.get('sector') == sector
            )
            if (sector_exposure + notional) / swing_alloc > self.config.max_sector_pct_swing:
                return False, f"Sector {sector} concentration would exceed {self.config.max_sector_pct_swing:.1%}"

        # --- Gross exposure ---
        current_gross = sum(p['notional'] for p in state.positions.values())
        if (current_gross + notional) / state.nav > self.config.max_gross_exposure_pct:
            return False, f"Gross exposure limit {self.config.max_gross_exposure_pct:.1%} would be breached"

        # --- PDT rule ---
        if self.config.enforce_pdt_rule and state.account_value < self.config.pdt_account_threshold:
            if side in (OrderSide.SELL, OrderSide.COVER):
                # Check if this would be a day trade (same-day open + close)
                pos = state.positions.get(symbol, {})
                if pos.get('entry_date') == current_date:
                    recent_day_trades = [
                        d for d in state.day_trade_dates
                        if d >= current_date - timedelta(days=5)
                    ]
                    if len(recent_day_trades) >= self.config.pdt_max_day_trades:
                        return False, (
                            f"PDT limit: {len(recent_day_trades)} day trades in rolling 5 days. "
                            f"Account value ${state.account_value:,.0f} < ${self.config.pdt_account_threshold:,.0f}"
                        )

        return True, "OK"

    # ------------------------------------------------------------------ #
    #  PERIODIC CHECK (call every 15 min for Swing, EOD for Core)         #
    # ------------------------------------------------------------------ #

    def periodic_check(self, state: PortfolioState, current_date: date = None) -> list:
        """
        Run all portfolio-level checks. Returns list of triggered alerts.
        """
        if current_date is None:
            current_date = date.today()

        alerts = []

        # Update peak NAV
        if state.nav > state.peak_nav:
            state.peak_nav = state.nav

        # Drawdown check
        dd = state.current_drawdown
        if dd <= -self.config.max_portfolio_drawdown:
            self._trigger_kill_switch(KillSwitchReason.PORTFOLIO_DRAWDOWN, current_date)
            alerts.append(f"KILL SWITCH: Portfolio drawdown {dd:.1%}")

        # Daily loss check
        daily = state.daily_pnl_pct
        if daily <= -self.config.max_daily_loss_pct:
            self.daily_halt_until = current_date
            alerts.append(f"DAILY HALT: Loss {daily:.1%}")

        # Warn at 50% of limit
        if dd <= -self.config.max_portfolio_drawdown * 0.5:
            alerts.append(f"WARNING: Drawdown {dd:.1%} at 50% of kill-switch level")

        return alerts

    # ------------------------------------------------------------------ #
    #  KILL SWITCH                                                        #
    # ------------------------------------------------------------------ #

    def _trigger_kill_switch(self, reason: KillSwitchReason, current_date: date):
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_switch_reason = reason
            self.kill_switch_date = current_date
            print(f"[KILL SWITCH TRIGGERED] Reason: {reason} | Date: {current_date}")

    def manual_reset_kill_switch(self, operator_id: str):
        """Requires manual confirmation — logs the reset event."""
        print(f"[KILL SWITCH RESET] By operator: {operator_id} at {date.today()}")
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_date = None

    # ------------------------------------------------------------------ #
    #  POSITION SIZING HELPER                                             #
    # ------------------------------------------------------------------ #

    def vol_target_size(
        self,
        instrument_vol: float,       # Annualized vol of instrument
        holding_days: int,
        sleeve_vol_budget: float,    # Max vol contribution from this position
        max_notional: float,         # Hard cap
    ) -> float:
        """
        Vol-targeting position size.
        target_size = sleeve_vol_budget / (instrument_vol * sqrt(holding_days / 252))
        """
        if instrument_vol <= 0:
            return 0.0
        period_vol = instrument_vol * np.sqrt(holding_days / 252)
        raw_size = sleeve_vol_budget / period_vol
        return min(raw_size, max_notional)
```

---

## Unit Tests

```python
# tests/test_risk_governor.py
import pytest
from datetime import date
from src.risk.risk_governor import RiskGovernor, RiskConfig, PortfolioState, OrderSide

def make_state(**kwargs) -> PortfolioState:
    defaults = dict(
        nav=100_000, peak_nav=100_000, cash=30_000,
        sleeve_values={"core": 60_000, "swing": 30_000, "cash": 10_000},
        positions={}, day_start_nav=100_000, week_start_swing_nav=30_000,
        account_value=50_000,
    )
    defaults.update(kwargs)
    return PortfolioState(**defaults)

def test_normal_order_passes():
    gov = RiskGovernor(RiskConfig())
    state = make_state()
    ok, reason = gov.pre_trade_check("AAPL", OrderSide.BUY, 4_000, "swing", state)
    assert ok, reason

def test_kill_switch_on_drawdown():
    gov = RiskGovernor(RiskConfig(max_portfolio_drawdown=0.15))
    state = make_state(nav=84_000, peak_nav=100_000)  # -16% drawdown
    ok, reason = gov.pre_trade_check("AAPL", OrderSide.BUY, 1_000, "swing", state)
    assert not ok
    assert "drawdown" in reason.lower()
    assert gov.kill_switch_active

def test_daily_loss_halt():
    gov = RiskGovernor(RiskConfig(max_daily_loss_pct=0.03))
    state = make_state(nav=96_500, day_start_nav=100_000)  # -3.5% daily
    ok, reason = gov.pre_trade_check("AAPL", OrderSide.BUY, 1_000, "swing", state)
    assert not ok
    assert "daily" in reason.lower()

def test_max_concurrent_positions():
    positions = {f"SYM{i}": {"sleeve": "swing", "notional": 3_000} for i in range(10)}
    gov = RiskGovernor(RiskConfig(max_concurrent_positions=10))
    state = make_state(positions=positions)
    ok, reason = gov.pre_trade_check("NEW", OrderSide.BUY, 3_000, "swing", state)
    assert not ok
    assert "concurrent" in reason.lower()

def test_position_size_limit():
    gov = RiskGovernor(RiskConfig(max_position_pct_swing=0.15))
    state = make_state()  # swing = 30k, 15% = 4500
    ok, reason = gov.pre_trade_check("AAPL", OrderSide.BUY, 5_000, "swing", state)
    assert not ok
    assert "size" in reason.lower()

def test_pdt_rule_blocks_4th_day_trade():
    gov = RiskGovernor(RiskConfig(enforce_pdt_rule=True, pdt_account_threshold=25_000))
    today = date(2024, 1, 15)
    state = make_state(
        account_value=20_000,
        positions={"AAPL": {"sleeve": "swing", "notional": 3000, "entry_date": today}},
        day_trade_dates=[today - pd.Timedelta(days=d) for d in [1, 2, 3]],  # 3 trades already
    )
    ok, reason = gov.pre_trade_check("AAPL", OrderSide.SELL, 3_000, "swing", state, current_date=today)
    assert not ok
    assert "PDT" in reason
```

---

## Configuration

```yaml
risk:
  max_portfolio_drawdown: 0.15
  max_daily_loss_pct: 0.03
  kill_switch_cooldown_days: 5
  manual_restart_required: true

  swing:
    max_weekly_loss: 0.05
    max_concurrent_positions: 10
    max_position_pct: 0.15
    max_sector_pct: 0.30
    per_trade_stop_multiplier: 2.0

  exposure:
    max_gross_exposure_pct: 1.0
    max_cross_sleeve_corr: 0.70

  pdt:
    enforce: true
    account_threshold: 25000
    max_day_trades_rolling_5d: 3

  core:
    rebalance_band: 0.05
    min_adv_usd: 1000000
```

---

## Operational Rules

1. **Never bypass** the Risk Governor, even for "obvious" trades.
2. **Kill switch requires manual reset** in production (config: `manual_restart_required: true`).
3. **Log every check** — both passed and failed — with timestamp, symbol, reason.
4. **Test suite must pass** before any deployment. Zero exceptions.
5. **Risk checks are synchronous** — they block the order submission thread.

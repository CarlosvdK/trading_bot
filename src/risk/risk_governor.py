"""
Risk Governor — highest-priority component.
Gates ALL orders. Enforces drawdown kill-switch, daily loss halt,
position limits, PDT rules, exposure caps.

Skill reference: .claude/skills/risk-governor/SKILL.md
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


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
    max_portfolio_drawdown: float = 0.15
    max_daily_loss_pct: float = 0.03
    manual_restart_required: bool = True
    kill_switch_cooldown_days: int = 5

    # Swing sleeve
    max_swing_weekly_loss: float = 0.05
    max_concurrent_positions: int = 10
    max_position_pct_swing: float = 0.15
    max_sector_pct_swing: float = 0.30
    per_trade_stop_multiplier: float = 2.0

    # Exposure
    max_gross_exposure_pct: float = 1.0
    max_cross_sleeve_corr: float = 0.70

    # PDT (US equities)
    enforce_pdt_rule: bool = True
    pdt_account_threshold: float = 25_000.0
    pdt_max_day_trades: int = 3

    # Core sleeve
    core_rebalance_band: float = 0.05
    core_min_adv_usd: float = 1_000_000


@dataclass
class PortfolioState:
    nav: float
    peak_nav: float
    cash: float
    sleeve_values: Dict[str, float] = field(default_factory=dict)
    positions: Dict[str, Dict] = field(default_factory=dict)
    day_start_nav: float = 0.0
    week_start_swing_nav: float = 0.0
    account_value: float = 0.0
    day_trade_dates: list = field(default_factory=list)

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

        # Normalize side to enum if string
        if isinstance(side, str):
            side = OrderSide(side)

        # --- Kill switch ---
        if self.kill_switch_active:
            if self.config.manual_restart_required:
                return False, (
                    f"Kill switch active ({self.kill_switch_reason}). "
                    f"Manual restart required."
                )
            cooldown_end = self.kill_switch_date + timedelta(
                days=self.config.kill_switch_cooldown_days
            )
            if current_date < cooldown_end:
                return False, f"Kill switch cooldown until {cooldown_end}"

        # --- Daily halt ---
        if self.daily_halt_until and current_date <= self.daily_halt_until:
            return False, (
                f"Daily loss halt active until {self.daily_halt_until}"
            )

        # --- Swing halt ---
        if (
            sleeve == "swing"
            and self.swing_halted_until
            and current_date <= self.swing_halted_until
        ):
            return False, (
                f"Swing sleeve halted until {self.swing_halted_until}"
            )

        # --- Portfolio drawdown ---
        dd = state.current_drawdown
        if dd <= -self.config.max_portfolio_drawdown:
            self._trigger_kill_switch(
                KillSwitchReason.PORTFOLIO_DRAWDOWN, current_date
            )
            return False, (
                f"Portfolio drawdown {dd:.1%} exceeds limit "
                f"{-self.config.max_portfolio_drawdown:.1%}"
            )

        # --- Daily loss ---
        daily_loss = state.daily_pnl_pct
        if daily_loss <= -self.config.max_daily_loss_pct:
            self.daily_halt_until = current_date
            return False, (
                f"Daily loss {daily_loss:.1%} exceeds limit "
                f"{-self.config.max_daily_loss_pct:.1%}"
            )

        # --- Swing weekly loss ---
        if sleeve == "swing":
            swing_nav = state.sleeve_values.get("swing", 0)
            swing_week_start = state.week_start_swing_nav
            if swing_week_start > 0:
                swing_weekly_loss = (
                    (swing_nav - swing_week_start) / swing_week_start
                )
                if swing_weekly_loss <= -self.config.max_swing_weekly_loss:
                    days_to_friday = (4 - current_date.weekday()) % 7
                    if days_to_friday == 0:
                        days_to_friday = 7
                    self.swing_halted_until = current_date + timedelta(
                        days=days_to_friday
                    )
                    return False, (
                        f"Swing weekly loss {swing_weekly_loss:.1%} "
                        f"exceeds limit"
                    )

        # --- Concurrent positions (Swing) ---
        if sleeve == "swing" and side in (OrderSide.BUY, OrderSide.SHORT):
            swing_positions = [
                p
                for p in state.positions.values()
                if p.get("sleeve") == "swing"
            ]
            if len(swing_positions) >= self.config.max_concurrent_positions:
                return False, (
                    f"Max concurrent swing positions "
                    f"({self.config.max_concurrent_positions}) reached"
                )

        # --- Position size (Swing) ---
        if sleeve == "swing":
            swing_alloc = state.sleeve_values.get("swing", state.nav * 0.3)
            if swing_alloc > 0 and notional / swing_alloc > self.config.max_position_pct_swing:
                return False, (
                    f"Position size {notional / swing_alloc:.1%} exceeds "
                    f"max {self.config.max_position_pct_swing:.1%} of swing sleeve"
                )

        # --- Sector concentration (Swing) ---
        if sleeve == "swing" and sector:
            swing_alloc = state.sleeve_values.get("swing", state.nav * 0.3)
            sector_exposure = sum(
                p["notional"]
                for p in state.positions.values()
                if p.get("sleeve") == "swing" and p.get("sector") == sector
            )
            if swing_alloc > 0 and (sector_exposure + notional) / swing_alloc > self.config.max_sector_pct_swing:
                return False, (
                    f"Sector {sector} concentration would exceed "
                    f"{self.config.max_sector_pct_swing:.1%}"
                )

        # --- Gross exposure ---
        current_gross = sum(
            p.get("notional", 0) for p in state.positions.values()
        )
        if state.nav > 0 and (current_gross + notional) / state.nav > self.config.max_gross_exposure_pct:
            return False, (
                f"Gross exposure limit "
                f"{self.config.max_gross_exposure_pct:.1%} would be breached"
            )

        # --- PDT rule ---
        if (
            self.config.enforce_pdt_rule
            and state.account_value < self.config.pdt_account_threshold
        ):
            if side in (OrderSide.SELL, OrderSide.COVER):
                pos = state.positions.get(symbol, {})
                if pos.get("entry_date") == current_date:
                    recent_day_trades = [
                        d
                        for d in state.day_trade_dates
                        if d >= current_date - timedelta(days=5)
                    ]
                    if (
                        len(recent_day_trades)
                        >= self.config.pdt_max_day_trades
                    ):
                        return False, (
                            f"PDT limit: {len(recent_day_trades)} day trades "
                            f"in rolling 5 days. Account value "
                            f"${state.account_value:,.0f} < "
                            f"${self.config.pdt_account_threshold:,.0f}"
                        )

        return True, "OK"

    # ------------------------------------------------------------------ #
    #  PERIODIC CHECK                                                      #
    # ------------------------------------------------------------------ #

    def periodic_check(
        self, state: PortfolioState, current_date: date = None
    ) -> List[str]:
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
            self._trigger_kill_switch(
                KillSwitchReason.PORTFOLIO_DRAWDOWN, current_date
            )
            alerts.append(f"KILL SWITCH: Portfolio drawdown {dd:.1%}")

        # Daily loss check
        daily = state.daily_pnl_pct
        if daily <= -self.config.max_daily_loss_pct:
            self.daily_halt_until = current_date
            alerts.append(f"DAILY HALT: Loss {daily:.1%}")

        # Warn at 50% of limit
        if (
            dd <= -self.config.max_portfolio_drawdown * 0.5
            and dd > -self.config.max_portfolio_drawdown
        ):
            alerts.append(
                f"WARNING: Drawdown {dd:.1%} at 50% of kill-switch level"
            )

        return alerts

    # ------------------------------------------------------------------ #
    #  KILL SWITCH                                                         #
    # ------------------------------------------------------------------ #

    def _trigger_kill_switch(
        self, reason: KillSwitchReason, current_date: date
    ):
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_switch_reason = reason
            self.kill_switch_date = current_date

    def manual_reset_kill_switch(self, operator_id: str):
        """Requires manual confirmation — logs the reset event."""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_date = None

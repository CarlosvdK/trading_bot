"""
Tests for Risk Governor — all risk limits enforced correctly.
Skill reference: .claude/skills/risk-governor/SKILL.md
"""

import pytest
from datetime import date, timedelta

from src.risk_management.risk_governor import (
    RiskGovernor,
    RiskConfig,
    PortfolioState,
    OrderSide,
    KillSwitchReason,
)


def make_state(**kwargs) -> PortfolioState:
    defaults = dict(
        nav=100_000,
        peak_nav=100_000,
        cash=30_000,
        sleeve_values={"core": 60_000, "swing": 30_000, "cash": 10_000},
        positions={},
        day_start_nav=100_000,
        week_start_swing_nav=30_000,
        account_value=50_000,
    )
    defaults.update(kwargs)
    return PortfolioState(**defaults)


class TestPreTradeCheck:
    def test_normal_order_passes(self):
        gov = RiskGovernor(RiskConfig())
        state = make_state()
        ok, reason = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 4_000, "swing", state
        )
        assert ok, reason

    def test_core_order_passes(self):
        gov = RiskGovernor(RiskConfig())
        state = make_state()
        ok, reason = gov.pre_trade_check(
            "SPY", OrderSide.BUY, 10_000, "core", state
        )
        assert ok, reason

    def test_string_side_accepted(self):
        gov = RiskGovernor(RiskConfig())
        state = make_state()
        ok, reason = gov.pre_trade_check(
            "AAPL", "BUY", 4_000, "swing", state
        )
        assert ok, reason


class TestKillSwitch:
    def test_kill_switch_on_drawdown(self):
        gov = RiskGovernor(RiskConfig(max_portfolio_drawdown=0.15))
        state = make_state(nav=84_000, peak_nav=100_000)  # -16%
        ok, reason = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 1_000, "swing", state
        )
        assert not ok
        assert "drawdown" in reason.lower()
        assert gov.kill_switch_active

    def test_kill_switch_blocks_subsequent_orders(self):
        gov = RiskGovernor(RiskConfig(max_portfolio_drawdown=0.15))
        state = make_state(nav=84_000, peak_nav=100_000)

        # Trigger kill switch
        gov.pre_trade_check("AAPL", OrderSide.BUY, 1_000, "swing", state)
        assert gov.kill_switch_active

        # Subsequent order should also fail
        state2 = make_state()  # even healthy state
        ok, reason = gov.pre_trade_check(
            "MSFT", OrderSide.BUY, 1_000, "swing", state2
        )
        assert not ok
        assert "kill switch" in reason.lower()

    def test_kill_switch_requires_manual_restart(self):
        gov = RiskGovernor(
            RiskConfig(
                max_portfolio_drawdown=0.15, manual_restart_required=True
            )
        )
        state = make_state(nav=84_000, peak_nav=100_000)
        gov.pre_trade_check("AAPL", OrderSide.BUY, 1_000, "swing", state)

        # Even after cooldown, manual restart is needed
        ok, reason = gov.pre_trade_check(
            "AAPL",
            OrderSide.BUY,
            1_000,
            "swing",
            make_state(),
            current_date=date.today() + timedelta(days=30),
        )
        assert not ok
        assert "manual restart" in reason.lower()

    def test_manual_reset_clears_kill_switch(self):
        gov = RiskGovernor(RiskConfig(max_portfolio_drawdown=0.15))
        state = make_state(nav=84_000, peak_nav=100_000)
        gov.pre_trade_check("AAPL", OrderSide.BUY, 1_000, "swing", state)
        assert gov.kill_switch_active

        gov.manual_reset_kill_switch("operator_1")
        assert not gov.kill_switch_active

        ok, _ = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 1_000, "swing", make_state()
        )
        assert ok

    def test_kill_switch_cooldown_when_auto_restart(self):
        today = date(2024, 6, 1)
        gov = RiskGovernor(
            RiskConfig(
                max_portfolio_drawdown=0.15,
                manual_restart_required=False,
                kill_switch_cooldown_days=5,
            )
        )
        state = make_state(nav=84_000, peak_nav=100_000)
        gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 1_000, "swing", state, current_date=today
        )

        # During cooldown
        ok, reason = gov.pre_trade_check(
            "AAPL",
            OrderSide.BUY,
            1_000,
            "swing",
            make_state(),
            current_date=today + timedelta(days=3),
        )
        assert not ok
        assert "cooldown" in reason.lower()

        # After cooldown
        ok, reason = gov.pre_trade_check(
            "AAPL",
            OrderSide.BUY,
            1_000,
            "swing",
            make_state(),
            current_date=today + timedelta(days=6),
        )
        assert ok


class TestDailyLoss:
    def test_daily_loss_halt(self):
        gov = RiskGovernor(RiskConfig(max_daily_loss_pct=0.03))
        state = make_state(nav=96_500, day_start_nav=100_000)  # -3.5%
        ok, reason = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 1_000, "swing", state
        )
        assert not ok
        assert "daily" in reason.lower()

    def test_daily_loss_at_boundary_passes(self):
        gov = RiskGovernor(RiskConfig(max_daily_loss_pct=0.03))
        state = make_state(nav=97_100, day_start_nav=100_000)  # -2.9%
        ok, _ = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 1_000, "swing", state
        )
        assert ok


class TestSwingLimits:
    def test_max_concurrent_positions(self):
        positions = {
            f"SYM{i}": {"sleeve": "swing", "notional": 3_000}
            for i in range(10)
        }
        gov = RiskGovernor(RiskConfig(max_concurrent_positions=10))
        state = make_state(positions=positions)
        ok, reason = gov.pre_trade_check(
            "NEW", OrderSide.BUY, 3_000, "swing", state
        )
        assert not ok
        assert "concurrent" in reason.lower()

    def test_under_concurrent_limit_passes(self):
        positions = {
            f"SYM{i}": {"sleeve": "swing", "notional": 3_000}
            for i in range(5)
        }
        gov = RiskGovernor(RiskConfig(max_concurrent_positions=10))
        state = make_state(positions=positions)
        ok, _ = gov.pre_trade_check(
            "NEW", OrderSide.BUY, 3_000, "swing", state
        )
        assert ok

    def test_sell_allowed_at_position_limit(self):
        """Closing positions should still work at limit."""
        positions = {
            f"SYM{i}": {"sleeve": "swing", "notional": 3_000}
            for i in range(10)
        }
        gov = RiskGovernor(RiskConfig(max_concurrent_positions=10))
        state = make_state(positions=positions)
        ok, _ = gov.pre_trade_check(
            "SYM0", OrderSide.SELL, 3_000, "swing", state
        )
        assert ok

    def test_position_size_limit(self):
        gov = RiskGovernor(RiskConfig(max_position_pct_swing=0.15))
        state = make_state()  # swing = 30k, 15% = 4500
        ok, reason = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 5_000, "swing", state
        )
        assert not ok
        assert "size" in reason.lower()

    def test_position_under_size_limit_passes(self):
        gov = RiskGovernor(RiskConfig(max_position_pct_swing=0.15))
        state = make_state()
        ok, _ = gov.pre_trade_check(
            "AAPL", OrderSide.BUY, 4_000, "swing", state
        )
        assert ok

    def test_sector_concentration_limit(self):
        positions = {
            "AAPL": {
                "sleeve": "swing",
                "notional": 5_000,
                "sector": "tech",
            },
            "MSFT": {
                "sleeve": "swing",
                "notional": 4_000,
                "sector": "tech",
            },
        }
        gov = RiskGovernor(RiskConfig(max_sector_pct_swing=0.30))
        state = make_state(positions=positions)
        # tech already at 9k / 30k = 30%, adding more should fail
        ok, reason = gov.pre_trade_check(
            "GOOGL", OrderSide.BUY, 1_000, "swing", state, sector="tech"
        )
        assert not ok
        assert "sector" in reason.lower()

    def test_swing_weekly_loss_halt(self):
        today = date(2024, 6, 5)  # Wednesday
        gov = RiskGovernor(RiskConfig(max_swing_weekly_loss=0.05))
        state = make_state(
            sleeve_values={"swing": 28_000},
            week_start_swing_nav=30_000,  # -6.7%
        )
        ok, reason = gov.pre_trade_check(
            "AAPL",
            OrderSide.BUY,
            1_000,
            "swing",
            state,
            current_date=today,
        )
        assert not ok
        assert "weekly" in reason.lower()


class TestExposure:
    def test_gross_exposure_limit(self):
        positions = {
            f"SYM{i}": {"sleeve": "core", "notional": 15_000}
            for i in range(6)
        }
        gov = RiskGovernor(
            RiskConfig(
                max_gross_exposure_pct=1.0,
                max_position_pct_swing=1.0,  # Disable position size check
            )
        )
        state = make_state(positions=positions)  # 90k gross, nav=100k
        ok, reason = gov.pre_trade_check(
            "NEW", OrderSide.BUY, 15_000, "swing", state
        )
        assert not ok
        assert "exposure" in reason.lower()


class TestPDT:
    def test_pdt_rule_blocks_4th_day_trade(self):
        today = date(2024, 1, 15)
        gov = RiskGovernor(
            RiskConfig(enforce_pdt_rule=True, pdt_account_threshold=25_000)
        )
        state = make_state(
            account_value=20_000,
            positions={
                "AAPL": {
                    "sleeve": "swing",
                    "notional": 3000,
                    "entry_date": today,
                }
            },
            day_trade_dates=[today - timedelta(days=d) for d in [1, 2, 3]],
        )
        ok, reason = gov.pre_trade_check(
            "AAPL",
            OrderSide.SELL,
            3_000,
            "swing",
            state,
            current_date=today,
        )
        assert not ok
        assert "PDT" in reason

    def test_pdt_not_enforced_above_threshold(self):
        today = date(2024, 1, 15)
        gov = RiskGovernor(
            RiskConfig(enforce_pdt_rule=True, pdt_account_threshold=25_000)
        )
        state = make_state(
            account_value=30_000,
            positions={
                "AAPL": {
                    "sleeve": "swing",
                    "notional": 3000,
                    "entry_date": today,
                }
            },
            day_trade_dates=[today - timedelta(days=d) for d in [1, 2, 3]],
        )
        ok, _ = gov.pre_trade_check(
            "AAPL",
            OrderSide.SELL,
            3_000,
            "swing",
            state,
            current_date=today,
        )
        assert ok

    def test_pdt_allows_non_day_trade(self):
        today = date(2024, 1, 15)
        gov = RiskGovernor(
            RiskConfig(enforce_pdt_rule=True, pdt_account_threshold=25_000)
        )
        state = make_state(
            account_value=20_000,
            positions={
                "AAPL": {
                    "sleeve": "swing",
                    "notional": 3000,
                    "entry_date": today - timedelta(days=3),  # Not same day
                }
            },
            day_trade_dates=[today - timedelta(days=d) for d in [1, 2, 3]],
        )
        ok, _ = gov.pre_trade_check(
            "AAPL",
            OrderSide.SELL,
            3_000,
            "swing",
            state,
            current_date=today,
        )
        assert ok


class TestPeriodicCheck:
    def test_updates_peak_nav(self):
        gov = RiskGovernor(RiskConfig())
        state = make_state(nav=110_000, peak_nav=100_000)
        gov.periodic_check(state)
        assert state.peak_nav == 110_000

    def test_drawdown_warning_at_50pct(self):
        gov = RiskGovernor(RiskConfig(max_portfolio_drawdown=0.15))
        state = make_state(nav=92_500, peak_nav=100_000)  # -7.5%
        alerts = gov.periodic_check(state)
        assert any("WARNING" in a for a in alerts)

    def test_drawdown_triggers_kill_switch(self):
        gov = RiskGovernor(RiskConfig(max_portfolio_drawdown=0.15))
        state = make_state(nav=84_000, peak_nav=100_000)
        alerts = gov.periodic_check(state)
        assert any("KILL SWITCH" in a for a in alerts)
        assert gov.kill_switch_active

    def test_daily_loss_triggers_halt(self):
        gov = RiskGovernor(RiskConfig(max_daily_loss_pct=0.03))
        state = make_state(nav=96_000, day_start_nav=100_000)
        alerts = gov.periodic_check(state)
        assert any("DAILY HALT" in a for a in alerts)


class TestPortfolioState:
    def test_current_drawdown(self):
        state = make_state(nav=85_000, peak_nav=100_000)
        assert state.current_drawdown == pytest.approx(-0.15)

    def test_daily_pnl_pct(self):
        state = make_state(nav=97_000, day_start_nav=100_000)
        assert state.daily_pnl_pct == pytest.approx(-0.03)

    def test_zero_peak_drawdown(self):
        state = make_state(nav=0, peak_nav=0)
        assert state.current_drawdown == 0.0

    def test_zero_day_start_pnl(self):
        state = make_state(nav=100_000, day_start_nav=0)
        assert state.daily_pnl_pct == 0.0

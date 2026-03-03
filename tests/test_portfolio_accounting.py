"""
Tests for portfolio accounting — P&L calculation exact.
Skill reference: .claude/skills/backtesting-engine/SKILL.md
"""

import pytest
import pandas as pd

from src.backtest.portfolio import Position, SleeveAccount


class TestPosition:
    def test_notional(self):
        pos = Position("AAPL", "swing", 100, 150.0, pd.Timestamp("2024-01-01"))
        assert pos.notional == 15_000

    def test_unrealized_pnl_profit(self):
        pos = Position("AAPL", "swing", 100, 150.0, pd.Timestamp("2024-01-01"))
        assert pos.unrealized_pnl(160.0) == 1_000

    def test_unrealized_pnl_loss(self):
        pos = Position("AAPL", "swing", 100, 150.0, pd.Timestamp("2024-01-01"))
        assert pos.unrealized_pnl(140.0) == -1_000

    def test_unrealized_pnl_pct(self):
        pos = Position("AAPL", "swing", 100, 100.0, pd.Timestamp("2024-01-01"))
        assert pos.unrealized_pnl_pct(110.0) == pytest.approx(0.10)

    def test_short_position_pnl(self):
        pos = Position("AAPL", "swing", -100, 150.0, pd.Timestamp("2024-01-01"))
        # Short: profit when price drops
        assert pos.unrealized_pnl(140.0) == 1_000

    def test_zero_avg_cost(self):
        pos = Position("AAPL", "swing", 100, 0.0, pd.Timestamp("2024-01-01"))
        assert pos.unrealized_pnl_pct(100.0) == 0.0


class TestSleeveAccount:
    def test_initial_state(self):
        acc = SleeveAccount("swing", 30_000)
        assert acc.cash == 30_000
        assert len(acc.positions) == 0
        assert acc.realized_pnl == 0.0

    def test_open_position_reduces_cash(self):
        acc = SleeveAccount("swing", 30_000)
        acc.open_position("AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01"))
        # Cost = 100 * 150 + 5 = 15_005
        assert acc.cash == pytest.approx(30_000 - 15_005)
        assert "AAPL" in acc.positions

    def test_close_position_adds_cash(self):
        acc = SleeveAccount("swing", 30_000)
        acc.open_position("AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01"))
        pnl = acc.close_position("AAPL", 160.0, 5.0, pd.Timestamp("2024-01-15"))
        # PnL = 100 * (160 - 150) - 5 = 995
        assert pnl == pytest.approx(995.0)
        assert "AAPL" not in acc.positions

    def test_realized_pnl_tracked(self):
        acc = SleeveAccount("swing", 30_000)
        acc.open_position("AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01"))
        acc.close_position("AAPL", 160.0, 5.0, pd.Timestamp("2024-01-15"))
        assert acc.realized_pnl == pytest.approx(995.0)

    def test_total_fees_tracked(self):
        acc = SleeveAccount("swing", 30_000)
        acc.open_position("AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01"))
        acc.close_position("AAPL", 160.0, 3.0, pd.Timestamp("2024-01-15"))
        assert acc.total_fees == pytest.approx(8.0)  # 5 + 3

    def test_insufficient_cash_raises(self):
        acc = SleeveAccount("swing", 1_000)
        with pytest.raises(ValueError, match="Insufficient cash"):
            acc.open_position(
                "AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01")
            )

    def test_close_nonexistent_raises(self):
        acc = SleeveAccount("swing", 30_000)
        with pytest.raises(ValueError, match="No position"):
            acc.close_position("AAPL", 150.0, 5.0, pd.Timestamp("2024-01-01"))

    def test_mark_to_market(self):
        acc = SleeveAccount("swing", 30_000)
        acc.open_position("AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01"))
        mtm = acc.mark_to_market({"AAPL": 160.0})
        # Cash = 30000 - 15005 = 14995, Position = 100 * 160 = 16000
        assert mtm == pytest.approx(14_995 + 16_000)

    def test_trades_log_records_open_and_close(self):
        acc = SleeveAccount("swing", 30_000)
        acc.open_position("AAPL", 100, 150.0, 5.0, pd.Timestamp("2024-01-01"))
        acc.close_position("AAPL", 160.0, 5.0, pd.Timestamp("2024-01-15"))
        assert len(acc.trades_log) == 2
        assert acc.trades_log[0]["action"] == "OPEN"
        assert acc.trades_log[1]["action"] == "CLOSE"
        assert acc.trades_log[1]["pnl"] == pytest.approx(995.0)

    def test_multiple_positions(self):
        acc = SleeveAccount("swing", 50_000)
        acc.open_position("AAPL", 50, 150.0, 3.0, pd.Timestamp("2024-01-01"))
        acc.open_position("MSFT", 30, 300.0, 3.0, pd.Timestamp("2024-01-01"))
        assert len(acc.positions) == 2
        mtm = acc.mark_to_market({"AAPL": 155.0, "MSFT": 310.0})
        cash_remaining = 50_000 - (50 * 150 + 3) - (30 * 300 + 3)
        expected = cash_remaining + 50 * 155 + 30 * 310
        assert mtm == pytest.approx(expected)

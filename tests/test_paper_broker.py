"""
Tests for Paper Broker and Order Manager.
Skill reference: .claude/skills/paper-broker-order-manager/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.execution.order_types import (
    Order,
    Fill,
    OrderType,
    OrderSide,
    OrderStatus,
)
from src.execution.paper_broker import PaperBroker, LiveBrokerStub
from src.execution.order_manager import OrderManager
from src.backtest.cost_model import CostModel
from src.risk.risk_governor import RiskGovernor, RiskConfig, PortfolioState


def make_price_data(n=100) -> dict:
    """Create price data for testing."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 50)
    volume = np.random.randint(500_000, 5_000_000, n).astype(float)
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.002),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return {"AAPL": df}


def make_order(
    symbol="AAPL",
    side=OrderSide.BUY,
    qty=100,
    order_type=OrderType.MARKET,
    **kwargs,
) -> Order:
    return Order(
        order_id="test-001",
        symbol=symbol,
        side=side,
        order_type=order_type,
        qty=qty,
        sleeve="swing",
        **kwargs,
    )


class TestPaperBroker:
    def test_market_order_fills(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order()
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill is not None
        assert fill.filled_qty > 0
        assert fill.fill_price > 0

    def test_fills_at_next_bar(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order()
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill.filled_at == prices["AAPL"].index[51]

    def test_moc_fills_at_close(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order(order_type=OrderType.MOC)
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill is not None
        # Fill price should be near next bar's close
        next_close = prices["AAPL"].iloc[51]["close"]
        assert abs(fill.fill_price - next_close) / next_close < 0.05

    def test_no_fill_at_last_bar(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order()
        last_date = prices["AAPL"].index[-1]
        fill = broker.execute(order, last_date)
        assert fill is None

    def test_unknown_symbol_returns_none(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order(symbol="UNKNOWN")
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill is None

    def test_limit_buy_not_filled_if_price_too_high(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order(
            order_type=OrderType.LIMIT, limit_price=10.0  # Way below market
        )
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill is None

    def test_buy_slippage_positive(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order(qty=100)
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill.slippage >= 0

    def test_fill_has_fees(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        order = make_order()
        date = prices["AAPL"].index[50]
        fill = broker.execute(order, date)
        assert fill.fees > 0


class TestLiveBrokerStub:
    def test_raises_not_implemented(self):
        broker = LiveBrokerStub()
        order = make_order()
        with pytest.raises(NotImplementedError, match="Live broker not implemented"):
            broker.execute(order, pd.Timestamp("2024-01-01"))


class TestOrderManager:
    def _make_manager(self):
        prices = make_price_data()
        cm = CostModel()
        broker = PaperBroker(prices, cm, {})
        risk_config = RiskConfig(
            max_position_pct_swing=1.0,
            max_gross_exposure_pct=5.0,
        )
        governor = RiskGovernor(risk_config)
        state = PortfolioState(
            nav=100_000,
            peak_nav=100_000,
            cash=40_000,
            day_start_nav=100_000,
            week_start_swing_nav=30_000,
            positions={},
            sleeve_values={"swing": 30_000, "core": 60_000},
        )
        manager = OrderManager(broker, governor, state, {})
        return manager, prices

    def test_submit_fills_order(self):
        manager, prices = self._make_manager()
        order = make_order()
        date = prices["AAPL"].index[50]
        fill = manager.submit(order, date)
        assert fill is not None
        assert len(manager.order_log) == 1
        assert manager.order_log[0]["status"] == "FILLED"

    def test_rejected_order_logged(self):
        manager, prices = self._make_manager()
        # Trigger kill switch to force rejection
        manager.risk_governor.kill_switch_active = True
        order = make_order()
        date = prices["AAPL"].index[50]
        fill = manager.submit(order, date)
        assert fill is None
        assert len(manager.order_log) == 1
        assert manager.order_log[0]["status"] == "REJECTED"

    def test_rate_limit_blocks_excess_orders(self):
        manager, prices = self._make_manager()
        manager.max_orders_per_minute = 2
        date = prices["AAPL"].index[50]

        for i in range(3):
            order = Order(
                order_id=f"test-{i:03d}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=10,
                sleeve="swing",
            )
            manager.submit(order, date)

        # 3rd order should be rate-limited
        assert len(manager.order_log) == 3
        assert manager.order_log[2]["status"] == "REJECTED"
        assert manager.order_log[2]["reason"] == "rate_limit"

    def test_order_log_format(self):
        manager, prices = self._make_manager()
        order = make_order()
        date = prices["AAPL"].index[50]
        manager.submit(order, date)

        log = manager.order_log[0]
        assert "timestamp" in log
        assert "order_id" in log
        assert "symbol" in log
        assert "side" in log
        assert "status" in log

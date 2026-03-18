"""Tests for portfolio analytics."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.market_intel.portfolio_analytics import (
    PerformanceMetrics,
    TradeAnalytics,
    RiskDecomposition,
    PortfolioReport,
    RollingAnalysis,
)


@pytest.fixture
def sample_nav():
    """Simple NAV series with known properties."""
    dates = pd.bdate_range("2023-01-01", periods=252)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0004, 0.01, 252)
    nav = 100000 * np.cumprod(1 + returns)
    return pd.Series(nav, index=dates)


@pytest.fixture
def constant_nav():
    """Flat NAV for edge case testing."""
    dates = pd.bdate_range("2023-01-01", periods=100)
    return pd.Series(100000.0, index=dates)


@pytest.fixture
def sample_trades():
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AAPL", "TSLA"],
        "pnl": [500, -200, 300, -100, 800],
        "signal_type": ["momentum", "momentum", "mean_reversion", "momentum", "gap"],
        "close_date": pd.bdate_range("2023-06-01", periods=5),
    })


class TestPerformanceMetrics:
    def test_sharpe_manual(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav, risk_free_rate=0.04)
        # Sharpe should be a reasonable number
        assert -5 < metrics.sharpe_ratio < 5

    def test_max_drawdown_negative(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav)
        assert metrics.max_drawdown <= 0

    def test_constant_nav_zero_return(self, constant_nav):
        metrics = PerformanceMetrics.from_nav_series(constant_nav)
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_total_return(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav)
        expected = sample_nav.iloc[-1] / sample_nav.iloc[0] - 1
        assert abs(metrics.total_return - expected) < 1e-10

    def test_to_dict(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav)
        d = metrics.to_dict()
        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d

    def test_to_dataframe(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav)
        df = metrics.to_dataframe()
        assert len(df) == 1
        assert "sharpe_ratio" in df.columns

    def test_var_cvar(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav)
        assert metrics.var_95 < 0  # Should be negative (loss)
        assert metrics.cvar_95 <= metrics.var_95  # CVaR worse than VaR

    def test_win_rate(self, sample_nav):
        metrics = PerformanceMetrics.from_nav_series(sample_nav)
        assert 0 <= metrics.win_rate_daily <= 1


class TestTradeAnalytics:
    def test_win_rate(self, sample_trades):
        ta = TradeAnalytics.from_trades(sample_trades)
        assert ta.win_rate == 0.6  # 3 wins out of 5

    def test_total_trades(self, sample_trades):
        ta = TradeAnalytics.from_trades(sample_trades)
        assert ta.total_trades == 5

    def test_profit_factor(self, sample_trades):
        ta = TradeAnalytics.from_trades(sample_trades)
        assert ta.profit_factor > 0  # Profitable overall

    def test_by_signal_type(self, sample_trades):
        ta = TradeAnalytics.from_trades(sample_trades)
        by_type = ta.by_signal_type()
        assert "momentum" in by_type
        assert by_type["momentum"]["count"] == 3

    def test_empty_trades(self):
        ta = TradeAnalytics.from_trades(pd.DataFrame())
        assert ta.total_trades == 0
        assert ta.win_rate == 0.0

    def test_sqn(self, sample_trades):
        ta = TradeAnalytics.from_trades(sample_trades)
        # SQN should be a number
        assert isinstance(ta.sqn, float)


class TestRiskDecomposition:
    def test_sleeve_attribution(self, sample_nav):
        nav_df = pd.DataFrame({
            "nav": sample_nav.values,
            "core_nav": sample_nav.values * 0.7,
            "swing_nav": sample_nav.values * 0.3,
        }, index=sample_nav.index)

        rd = RiskDecomposition(nav_df)
        attr = rd.sleeve_attribution()
        assert "core" in attr
        assert "swing" in attr

    def test_rolling_sharpe(self, sample_nav):
        nav_df = pd.DataFrame({"nav": sample_nav.values}, index=sample_nav.index)
        rd = RiskDecomposition(nav_df)
        rs = rd.rolling_sharpe(window=63)
        assert len(rs) > 0

    def test_drawdown_analysis(self, sample_nav):
        nav_df = pd.DataFrame({"nav": sample_nav.values}, index=sample_nav.index)
        rd = RiskDecomposition(nav_df)
        dds = rd.drawdown_analysis()
        assert isinstance(dds, list)
        for dd in dds:
            assert "depth" in dd
            assert dd["depth"] < 0


class TestPortfolioReport:
    def test_text_report(self, sample_nav):
        nav_df = pd.DataFrame({"nav": sample_nav.values}, index=sample_nav.index)
        report = PortfolioReport({"nav_history": nav_df, "total_trades": 50, "total_fees": 100})
        text = report.generate_text_report()
        assert "PORTFOLIO PERFORMANCE REPORT" in text

    def test_compare_strategies(self, sample_nav):
        nav_df = pd.DataFrame({"nav": sample_nav.values}, index=sample_nav.index)
        results = [{"nav_history": nav_df}, {"nav_history": nav_df}]
        comparison = PortfolioReport.compare_strategies(results, ["Strategy A", "Strategy B"])
        assert len(comparison) == 2
        assert "strategy" in comparison.columns

    def test_monthly_returns(self, sample_nav):
        nav_df = pd.DataFrame({"nav": sample_nav.values}, index=sample_nav.index)
        report = PortfolioReport({"nav_history": nav_df})
        table = report.monthly_returns_table()
        assert isinstance(table, str)
        assert len(table) > 0

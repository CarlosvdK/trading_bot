"""
Tests for the autonomous orchestrator.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.brain.orchestrator import TradingOrchestrator


def make_test_data(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {}
    for sym, seed in [("AAPL", 42), ("MSFT", 123)]:
        np.random.seed(seed)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 50)
        data[sym] = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.003),
                "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
                "close": close,
                "volume": np.random.randint(500_000, 5_000_000, n),
            },
            index=dates,
        )
    np.random.seed(789)
    spy = 400 + np.cumsum(np.random.randn(n) * 0.3)
    data["SPY"] = pd.DataFrame(
        {"open": spy * 0.999, "high": spy * 1.005, "low": spy * 0.995,
         "close": spy, "volume": 50_000_000},
        index=dates,
    )
    return data


@pytest.fixture
def config():
    return {
        "portfolio": {"initial_nav": 100_000},
        "risk": {},
        "ml": {
            "n_estimators": 10,
            "max_depth": 2,
            "min_oos_auc_to_deploy": 0.40,
            "entry_threshold": 0.50,
        },
        "swing_signals": {"momentum_threshold_pct": 0.02},
        "sizing": {"holding_days": 10},
        "features": {"index_symbol": "SPY"},
        "labeling": {"k1": 2.0, "k2": 1.0, "horizon_days": 10},
        "walk_forward": {"initial_train_days": 150, "test_days": 30, "step_days": 30},
    }


@pytest.fixture
def orchestrator(config, tmp_path):
    return TradingOrchestrator(config, str(tmp_path))


class TestOrchestratorInit:
    def test_initializes(self, orchestrator):
        assert orchestrator.portfolio_state.nav > 0
        assert len(orchestrator.open_positions) == 0

    def test_state_file_dir_created(self, orchestrator):
        assert orchestrator.state_file.parent.exists()


class TestWarmUp:
    def test_warm_up_trains_regime(self, orchestrator):
        data = make_test_data()
        index_df = data["SPY"]
        orchestrator.warm_up(data, index_df)
        assert orchestrator.regime_model is not None
        assert orchestrator.regime_names  # Non-empty dict

    def test_warm_up_trains_ml(self, orchestrator):
        data = make_test_data()
        index_df = data["SPY"]
        orchestrator.warm_up(data, index_df)
        # Model should be trained (may or may not deploy depending on AUC)
        assert orchestrator.last_retrain_date is not None
        assert orchestrator._warmup_done is True

    def test_warm_up_idempotent(self, orchestrator):
        data = make_test_data()
        index_df = data["SPY"]
        orchestrator.warm_up(data, index_df)
        v1 = orchestrator.ml_trainer.model_version
        orchestrator.warm_up(data, index_df)  # Should be no-op
        assert orchestrator.ml_trainer.model_version == v1


class TestRunDaily:
    def test_runs_without_error(self, orchestrator):
        data = make_test_data()
        index_df = data["SPY"]
        date = data["AAPL"].index[200]
        actions = orchestrator.run_daily(date, data, index_df)
        assert "date" in actions
        assert "regime" in actions

    def test_nav_tracked(self, orchestrator):
        data = make_test_data()
        index_df = data["SPY"]
        for date in data["AAPL"].index[100:110]:
            orchestrator.run_daily(date, data, index_df)
        assert len(orchestrator.daily_nav) >= 5

    def test_ml_model_active_flag(self, orchestrator):
        data = make_test_data()
        index_df = data["SPY"]
        date = data["AAPL"].index[200]
        actions = orchestrator.run_daily(date, data, index_df)
        assert "ml_model_active" in actions


class TestExitManagement:
    def test_tp_exit(self, orchestrator):
        orchestrator.open_positions["TEST"] = {
            "qty": 100,
            "entry_price": 100.0,
            "entry_date": pd.Timestamp("2020-01-01"),
            "tp_price": 105.0,
            "sl_price": 95.0,
            "ml_prob": 0.7,
            "regime": "low_vol_trending_up",
            "notional": 10000,
            "inst_vol": 0.25,
        }
        orchestrator.portfolio_state.positions["TEST"] = {"notional": 10000}
        exits = orchestrator._check_exits(
            {"TEST": 106.0}, pd.Timestamp("2020-02-01")
        )
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "TP_HIT"
        assert exits[0]["pnl"] > 0

    def test_sl_exit(self, orchestrator):
        orchestrator.open_positions["TEST"] = {
            "qty": 100,
            "entry_price": 100.0,
            "entry_date": pd.Timestamp("2020-01-01"),
            "tp_price": 105.0,
            "sl_price": 95.0,
            "ml_prob": 0.7,
            "regime": "low_vol_trending_up",
            "notional": 10000,
            "inst_vol": 0.25,
        }
        orchestrator.portfolio_state.positions["TEST"] = {"notional": 10000}
        exits = orchestrator._check_exits(
            {"TEST": 94.0}, pd.Timestamp("2020-02-01")
        )
        assert len(exits) == 1
        assert exits[0]["exit_reason"] == "SL_HIT"
        assert exits[0]["pnl"] < 0

    def test_feedback_loop_updates_prediction_log(self, orchestrator):
        """When a trade closes, the prediction log gets the actual label."""
        orchestrator.prediction_log.append({
            "date": pd.Timestamp("2020-01-01"),
            "symbol": "TEST",
            "ml_prob": 0.65,
            "actual_label": None,
        })
        orchestrator.open_positions["TEST"] = {
            "qty": 100,
            "entry_price": 100.0,
            "entry_date": pd.Timestamp("2020-01-01"),
            "tp_price": 105.0,
            "sl_price": 95.0,
            "ml_prob": 0.65,
            "regime": "unknown",
            "notional": 10000,
            "inst_vol": 0.25,
        }
        orchestrator.portfolio_state.positions["TEST"] = {"notional": 10000}
        orchestrator._check_exits(
            {"TEST": 106.0}, pd.Timestamp("2020-02-01")
        )
        # Prediction log should now have actual label = 1 (profitable trade)
        assert orchestrator.prediction_log[0]["actual_label"] == 1


class TestPerformanceSummary:
    def test_empty_log(self, orchestrator):
        summary = orchestrator.get_performance_summary()
        assert summary["n_trades"] == 0

    def test_with_trades(self, orchestrator):
        orchestrator.trade_log = [
            {"symbol": "AAPL", "pnl": 100, "pnl_pct": 0.05,
             "exit_reason": "TP_HIT", "days_held": 5, "ml_prob": 0.7, "regime": "low_vol_trending_up"},
            {"symbol": "MSFT", "pnl": -50, "pnl_pct": -0.02,
             "exit_reason": "SL_HIT", "days_held": 3, "ml_prob": 0.6, "regime": "low_vol_trending_up"},
        ]
        orchestrator.performance_by_regime = {"low_vol_trending_up": [0.05, -0.02]}
        summary = orchestrator.get_performance_summary()
        assert summary["n_trades"] == 2
        assert summary["win_rate"] == 0.5
        assert summary["total_pnl"] == 50


class TestAdaptiveRecommendations:
    def test_insufficient_trades(self, orchestrator):
        recs = orchestrator.get_adaptive_recommendations()
        assert recs.get("status") == "insufficient_trades"


class TestStatePersistence:
    def test_save_and_load(self, orchestrator):
        orchestrator.last_retrain_date = pd.Timestamp("2024-01-01")
        orchestrator.portfolio_state.nav = 105_000
        orchestrator._save_state()

        # Create new orchestrator pointing to same dir
        new_orch = TradingOrchestrator(
            orchestrator.config, str(orchestrator.root_dir)
        )
        assert new_orch.last_retrain_date == pd.Timestamp("2024-01-01")


class TestRegimeDetection:
    def test_detects_regime_with_data(self, orchestrator):
        data = make_test_data(n=300)
        index_df = data["SPY"]
        orchestrator._train_regime(index_df)
        assert orchestrator.regime_model is not None

        date = index_df.index[250]
        regime = orchestrator._detect_regime(index_df, date)
        assert regime != ""  # Should return something

    def test_returns_unknown_for_short_data(self, orchestrator):
        dates = pd.bdate_range("2020-01-01", periods=50)
        short_df = pd.DataFrame({"close": 100 + np.arange(50)}, index=dates)
        regime = orchestrator._detect_regime(short_df, dates[-1])
        assert regime == "unknown"

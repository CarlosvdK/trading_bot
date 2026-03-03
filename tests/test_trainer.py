"""
Tests for ML training pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from src.ml.trainer import MLTrainer


def make_price_data(n=500):
    """Create synthetic price data for multiple symbols."""
    dates = pd.bdate_range("2018-01-01", periods=n)
    data = {}
    for sym, seed in [("AAPL", 42), ("MSFT", 123), ("GOOGL", 456)]:
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
    # SPY as index
    np.random.seed(789)
    spy_close = 400 + np.cumsum(np.random.randn(n) * 0.3)
    data["SPY"] = pd.DataFrame(
        {
            "open": spy_close * 0.999,
            "high": spy_close * 1.005,
            "low": spy_close * 0.995,
            "close": spy_close,
            "volume": 50_000_000,
        },
        index=dates,
    )
    return data


@pytest.fixture
def price_data():
    return make_price_data()


@pytest.fixture
def index_df(price_data):
    return price_data["SPY"]


@pytest.fixture
def trainer():
    config = {
        "ml": {
            "n_estimators": 30,
            "max_depth": 3,
            "min_oos_auc_to_deploy": 0.45,  # Low bar for synthetic data
        },
        "labeling": {"k1": 2.0, "k2": 1.0, "horizon_days": 10},
        "walk_forward": {
            "initial_train_days": 200,
            "test_days": 60,
            "step_days": 60,
            "embargo_days": 12,
        },
        "features": {"index_symbol": "SPY"},
    }
    return MLTrainer(config, models_dir="/tmp/test_models")


class TestBuildTrainingData:
    def test_builds_X_and_y(self, trainer, price_data, index_df):
        dates = price_data["AAPL"].index
        X, y = trainer.build_training_data(
            price_data, index_df,
            signal_dates=dates,
            train_end=dates[400],
        )
        assert len(X) > 0
        assert len(y) == len(X)
        assert set(y.unique()).issubset({0, 1})

    def test_no_future_data(self, trainer, price_data, index_df):
        dates = price_data["AAPL"].index
        train_end = dates[300]
        X, y = trainer.build_training_data(
            price_data, index_df,
            signal_dates=dates,
            train_end=train_end,
        )
        # All feature dates should be <= train_end
        assert X.index.max() <= train_end


class TestTrainModel:
    def test_trains_and_returns_metrics(self, trainer, price_data, index_df):
        dates = price_data["AAPL"].index
        X, y = trainer.build_training_data(
            price_data, index_df,
            signal_dates=dates,
            train_end=dates[400],
        )
        model, metrics = trainer.train_model(X, y)
        assert model is not None
        assert "oos_roc_auc" in metrics
        assert "oos_f1" in metrics
        assert "ece" in metrics
        assert "top_features" in metrics

    def test_predictions_in_range(self, trainer, price_data, index_df):
        dates = price_data["AAPL"].index
        X, y = trainer.build_training_data(
            price_data, index_df,
            signal_dates=dates,
            train_end=dates[400],
        )
        model, _ = trainer.train_model(X, y)
        trainer.current_model = model
        probs = trainer.predict(X)
        assert probs.min() >= 0
        assert probs.max() <= 1


class TestWalkForwardTrain:
    def test_walk_forward_produces_model(self, trainer, price_data, index_df):
        model, report = trainer.walk_forward_train(price_data, index_df)
        assert report["n_folds"] >= 1
        assert "avg_oos_auc" in report

    def test_model_saved_if_deployed(self, trainer, price_data, index_df, tmp_path):
        trainer.models_dir = str(tmp_path)
        model, report = trainer.walk_forward_train(price_data, index_df)
        if report["status"] == "deployed":
            assert trainer.current_model is not None
            assert trainer.model_version >= 1


class TestPredictSingle:
    def test_returns_float(self, trainer, price_data, index_df):
        # Without model → default 0.5
        prob = trainer.predict_single(
            price_data["AAPL"], index_df, price_data["AAPL"].index[200]
        )
        assert isinstance(prob, float)
        assert 0 <= prob <= 1

"""
Tests for model calibration, PSI drift detection, retrain triggers.
Skill reference: .claude/skills/model-calibration-drift/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.models.calibration import calibrate_model, reliability_diagram
from src.models.drift import (
    compute_psi,
    monitor_feature_drift,
    compute_live_metrics,
    should_retrain,
)


class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        np.random.seed(42)
        data = np.random.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution_high_psi(self):
        np.random.seed(42)
        expected = np.random.randn(1000)
        actual = np.random.randn(1000) + 3  # Shifted by 3 std
        psi = compute_psi(expected, actual)
        assert psi > 0.20

    def test_psi_always_non_negative(self):
        np.random.seed(42)
        expected = np.random.randn(500)
        actual = np.random.randn(500) * 2
        psi = compute_psi(expected, actual)
        assert psi >= 0

    def test_small_shift_moderate_psi(self):
        np.random.seed(42)
        expected = np.random.randn(1000)
        actual = np.random.randn(1000) + 0.5
        psi = compute_psi(expected, actual)
        assert 0.01 < psi < 0.50


class TestMonitorFeatureDrift:
    def test_stable_features_no_alerts(self):
        np.random.seed(42)
        n = 500
        train = pd.DataFrame({
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
        })
        current = pd.DataFrame({
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
        })
        report = monitor_feature_drift(train, current)
        assert report["requires_retrain"] is False
        assert len(report["alerts"]) == 0

    def test_drifted_feature_triggers_alert(self):
        np.random.seed(42)
        n = 500
        train = pd.DataFrame({"feat_a": np.random.randn(n)})
        current = pd.DataFrame({"feat_a": np.random.randn(100) + 5})
        report = monitor_feature_drift(train, current)
        assert report["requires_retrain"] is True
        assert report["n_features_drifted"] >= 1

    def test_missing_column_skipped(self):
        train = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        current = pd.DataFrame({"a": np.random.randn(50)})
        report = monitor_feature_drift(train, current)
        assert "b" not in report["feature_report"]

    def test_insufficient_data_skipped(self):
        train = pd.DataFrame({"a": np.random.randn(10)})  # < 30
        current = pd.DataFrame({"a": np.random.randn(50)})
        report = monitor_feature_drift(train, current)
        assert len(report["feature_report"]) == 0


class TestComputeLiveMetrics:
    def test_good_model_ok_status(self):
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        # Simulate a decent model: prob > 0.6 when label=1
        labels = np.random.binomial(1, 0.6, n)
        probs = labels * 0.3 + np.random.rand(n) * 0.4 + 0.2
        probs = np.clip(probs, 0, 1)
        preds = pd.DataFrame({
            "date": dates,
            "ml_prob": probs,
            "actual_label": labels,
        })
        metrics = compute_live_metrics(preds, lookback_days=365)
        assert metrics["status"] == "OK"
        assert "roc_auc" in metrics

    def test_insufficient_data(self):
        preds = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=10),
            "ml_prob": np.random.rand(10),
            "actual_label": np.random.binomial(1, 0.5, 10),
        })
        metrics = compute_live_metrics(preds)
        assert metrics["status"] == "insufficient_data"

    def test_random_model_degraded(self):
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        preds = pd.DataFrame({
            "date": dates,
            "ml_prob": np.random.rand(n),
            "actual_label": np.random.binomial(1, 0.5, n),
        })
        metrics = compute_live_metrics(preds, lookback_days=365)
        # Random model AUC ~ 0.50, should trigger degradation
        assert metrics["roc_auc"] < 0.60


class TestShouldRetrain:
    def test_scheduled_retrain(self):
        retrain, reason = should_retrain(
            drift_report={"requires_retrain": False},
            performance_metrics={"status": "OK"},
            days_since_last_train=100,
            max_days_between_trains=90,
        )
        assert retrain is True
        assert "Scheduled" in reason

    def test_drift_retrain(self):
        retrain, reason = should_retrain(
            drift_report={"requires_retrain": True, "n_features_drifted": 3},
            performance_metrics={"status": "OK"},
            days_since_last_train=10,
        )
        assert retrain is True
        assert "drift" in reason.lower()

    def test_performance_retrain(self):
        retrain, reason = should_retrain(
            drift_report={"requires_retrain": False},
            performance_metrics={"status": "RETRAIN_URGENT", "roc_auc": 0.51},
            days_since_last_train=10,
        )
        assert retrain is True
        assert "Performance" in reason

    def test_no_retrain_needed(self):
        retrain, reason = should_retrain(
            drift_report={"requires_retrain": False},
            performance_metrics={"status": "OK"},
            days_since_last_train=30,
        )
        assert retrain is False
        assert "No retrain" in reason


class TestReliabilityDiagram:
    def test_returns_ece(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 500)
        y_prob = np.clip(y_true + np.random.randn(500) * 0.2, 0, 1)
        result = reliability_diagram(y_true, y_prob)
        assert "ece" in result
        assert isinstance(result["ece"], float)

    def test_perfect_calibration_low_ece(self):
        np.random.seed(42)
        n = 1000
        y_prob = np.random.rand(n)
        y_true = np.array([
            np.random.binomial(1, p) for p in y_prob
        ])
        result = reliability_diagram(y_true, y_prob, n_bins=5)
        # Perfectly generated, ECE should be reasonable
        assert result["ece"] < 0.15


class TestCalibrateModel:
    def test_calibration_works(self):
        from sklearn.ensemble import GradientBoostingClassifier

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Train/cal split
        X_train, X_cal = X[:400], X[400:]
        y_train, y_cal = y[:400], y[400:]

        base = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        base.fit(X_train, y_train)

        cal_model = calibrate_model(
            base,
            pd.DataFrame(X_cal),
            pd.Series(y_cal),
            method="sigmoid",
        )
        probs = cal_model.predict_proba(X_cal)[:, 1]
        assert len(probs) == len(X_cal)
        assert probs.min() >= 0
        assert probs.max() <= 1

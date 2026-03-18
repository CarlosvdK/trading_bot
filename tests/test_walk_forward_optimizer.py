"""Tests for walk-forward optimizer."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.walk_forward_optimizer import (
    ParameterSpace,
    WalkForwardOptimizer,
    CombinatorialPurgedCV,
    OverfitDetector,
)


class TestParameterSpace:
    def test_combinations(self):
        ps = ParameterSpace("test", {"a": [1, 2], "b": [3, 4, 5]})
        combos = ps.combinations()
        assert len(combos) == 6
        assert ps.n_combinations() == 6

    def test_single_param(self):
        ps = ParameterSpace("test", {"a": [1, 2, 3]})
        assert ps.n_combinations() == 3


class TestFoldGeneration:
    def test_no_overlap(self):
        dates = pd.bdate_range("2020-01-01", "2023-12-31")
        opt = WalkForwardOptimizer(config={}, objective="sharpe")
        folds = opt.generate_folds(dates, n_folds=5, embargo_days=5)

        for fold in folds:
            # Test should start after train end + embargo
            assert fold["test_start"] > fold["train_end"]

    def test_embargo_gap(self):
        dates = pd.bdate_range("2020-01-01", "2023-12-31")
        opt = WalkForwardOptimizer(config={}, objective="sharpe")
        folds = opt.generate_folds(dates, n_folds=3, embargo_days=10, purge_days=5)

        for fold in folds:
            gap = (fold["test_start"] - fold["train_end"]).days
            assert gap >= 10  # At least embargo_days

    def test_expanding_window(self):
        dates = pd.bdate_range("2020-01-01", "2023-12-31")
        opt = WalkForwardOptimizer(config={}, objective="sharpe")
        folds = opt.generate_folds(dates, n_folds=3, expanding=True)

        # Each fold's training set should start at same date (expanding)
        if len(folds) > 1:
            assert folds[0]["train_start"] == folds[1]["train_start"]


class TestStabilityScore:
    def test_perfect_consistency(self):
        opt = WalkForwardOptimizer(config={}, objective="sharpe")
        score = opt.compute_stability_score([1.5, 1.5, 1.5, 1.5])
        assert score == 1.0

    def test_mixed_results(self):
        opt = WalkForwardOptimizer(config={}, objective="sharpe")
        score = opt.compute_stability_score([2.0, -1.0, 1.5, -0.5])
        assert 0 <= score <= 1
        assert score < 0.8  # Mixed results should have lower stability

    def test_single_fold(self):
        opt = WalkForwardOptimizer(config={}, objective="sharpe")
        score = opt.compute_stability_score([1.5])
        assert score == 1.0


class TestCombinatorialPurgedCV:
    def test_n_paths(self):
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        assert cv.n_paths() == 15  # C(6,2) = 15

    def test_generates_paths(self):
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2)
        dates = pd.bdate_range("2020-01-01", "2023-12-31")
        paths = cv.generate_paths(dates)
        assert len(paths) == 6  # C(4,2) = 6

        for path in paths:
            assert len(path["train_indices"]) > 0
            assert len(path["test_indices"]) > 0
            # No overlap
            assert len(set(path["train_indices"]) & set(path["test_indices"])) == 0


class TestOverfitDetector:
    def test_flags_overfit(self):
        # IS performance great, OOS terrible
        result = OverfitDetector.detect_overfit(
            in_sample_results=[3.0, 2.5, 2.0, 1.5, 1.0],
            out_of_sample_results=[-0.5, -1.0, 0.1, -0.3, 0.2],
        )
        assert "pbo" in result
        assert "deflated_sharpe" in result
        assert "is_overfit" in result

    def test_consistent_results_not_overfit(self):
        # IS and OOS agree
        result = OverfitDetector.detect_overfit(
            in_sample_results=[1.0, 1.5, 2.0, 2.5, 3.0],
            out_of_sample_results=[0.8, 1.2, 1.8, 2.3, 2.8],
        )
        assert result["pbo"] < 0.5

    def test_empty_inputs(self):
        result = OverfitDetector.detect_overfit([], [])
        assert result["is_overfit"] is True


class TestObjectives:
    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError):
            WalkForwardOptimizer(config={}, objective="invalid_metric")

    def test_valid_objectives(self):
        for obj in ["sharpe", "calmar", "return", "sortino", "profit_factor"]:
            opt = WalkForwardOptimizer(config={}, objective=obj)
            assert opt.objective == obj

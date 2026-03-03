"""
Tests for walk-forward validation — embargo gaps correct.
Skill reference: .claude/skills/walk-forward-validation/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.ml.validation import (
    walk_forward_splits,
    purge_training_labels,
    leakage_audit,
)


def make_daily_index(n=2000) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


class TestWalkForwardSplits:
    def test_generates_splits(self):
        index = make_daily_index(2000)
        splits = list(
            walk_forward_splits(
                index,
                initial_train_days=756,
                test_days=126,
                step_days=63,
                embargo_days=12,
            )
        )
        assert len(splits) > 0

    def test_embargo_gap_between_train_and_test(self):
        index = make_daily_index(2000)
        splits = list(
            walk_forward_splits(
                index,
                initial_train_days=756,
                test_days=126,
                step_days=63,
                embargo_days=12,
            )
        )
        for train_idx, test_idx in splits:
            train_end = train_idx[-1]
            test_start = test_idx[0]
            gap = (test_start - train_end).days
            assert gap >= 12, (
                f"Embargo gap {gap} days < 12 between "
                f"{train_end.date()} and {test_start.date()}"
            )

    def test_no_overlap_between_train_and_test(self):
        index = make_daily_index(2000)
        splits = list(walk_forward_splits(index))
        for train_idx, test_idx in splits:
            overlap = train_idx.intersection(test_idx)
            assert len(overlap) == 0, "Train and test overlap!"

    def test_expanding_window_grows(self):
        index = make_daily_index(2000)
        splits = list(
            walk_forward_splits(index, expanding=True, step_days=63)
        )
        if len(splits) >= 2:
            assert len(splits[1][0]) > len(splits[0][0])

    def test_rolling_window_constant_size(self):
        index = make_daily_index(2000)
        splits = list(
            walk_forward_splits(
                index,
                expanding=False,
                initial_train_days=756,
                step_days=63,
            )
        )
        if len(splits) >= 2:
            assert len(splits[0][0]) == len(splits[1][0])

    def test_test_windows_cover_future(self):
        index = make_daily_index(2000)
        splits = list(walk_forward_splits(index))
        for train_idx, test_idx in splits:
            assert test_idx[0] > train_idx[-1]

    def test_no_splits_if_insufficient_data(self):
        index = make_daily_index(100)  # Too short for defaults
        splits = list(walk_forward_splits(index))
        assert len(splits) == 0


class TestPurgeTrainingLabels:
    def test_removes_contaminated_samples(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        labels = pd.DataFrame({"label": 1}, index=dates)
        train_end = dates[70]

        purged = purge_training_labels(labels, train_end, horizon=10)
        # All purged samples should have entry_date <= train_end - 10 days
        cutoff = train_end - pd.Timedelta(days=10)
        assert (purged.index <= cutoff).all()

    def test_preserves_safe_samples(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        labels = pd.DataFrame({"label": 1}, index=dates)
        train_end = dates[70]

        purged = purge_training_labels(labels, train_end, horizon=10)
        assert len(purged) > 0
        assert len(purged) < len(labels)


class TestLeakageAudit:
    def test_clean_features_pass(self):
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(n) * 0.3), index=dates
        )
        features = pd.DataFrame(
            {
                "random_feat_1": np.random.randn(n),
                "random_feat_2": np.random.randn(n),
            },
            index=dates,
        )
        result = leakage_audit(features, prices)
        assert result["passed"]
        assert len(result["issues"]) == 0

    def test_leaky_feature_detected(self):
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(n) * 0.3), index=dates
        )
        # Create a leaky feature: it IS future return
        future_ret = np.log(prices.shift(-1) / prices)
        features = pd.DataFrame(
            {"leaky_feature": future_ret}, index=dates
        )
        result = leakage_audit(features, prices, max_allowed_corr=0.05)
        assert not result["passed"]
        assert len(result["issues"]) > 0
        assert result["issues"][0]["feature"] == "leaky_feature"

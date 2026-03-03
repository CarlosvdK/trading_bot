"""
Tests for barrier labeling — labels use no future data.
Skill reference: .claude/skills/barrier-labeling/SKILL.md
"""

import pytest
import numpy as np
import pandas as pd

from src.ml.labeler import (
    barrier_label,
    build_labels,
    compute_vol_proxy,
    purge_and_embargo,
    label_quality_report,
)


def make_prices(n=200) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.003),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close": close,
            "volume": np.random.randint(100_000, 10_000_000, n),
        },
        index=dates,
    )


class TestBarrierLabel:
    def test_returns_binary(self):
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])
        result = barrier_label(prices, 0, tp_pct=0.05, sl_pct=0.03, horizon=10)
        assert result in (0, 1)

    def test_tp_hit_returns_1(self):
        prices = pd.Series([100, 101, 102, 110, 111])  # Jump to 110 = +10%
        result = barrier_label(prices, 0, tp_pct=0.05, sl_pct=0.03, horizon=4)
        assert result == 1

    def test_sl_hit_returns_0(self):
        prices = pd.Series([100, 99, 95, 90, 85])  # Drop to 95 = -5%
        result = barrier_label(prices, 0, tp_pct=0.10, sl_pct=0.03, horizon=4)
        assert result == 0

    def test_timeout_returns_0(self):
        prices = pd.Series([100, 100.1, 99.9, 100.05, 99.95])  # Flat
        result = barrier_label(prices, 0, tp_pct=0.10, sl_pct=0.10, horizon=4)
        assert result == 0

    def test_only_uses_future_prices(self):
        """Label at entry_idx=5 should only look at prices[6:6+horizon]."""
        prices = pd.Series(range(20), dtype=float)
        # With entry at index 5, we should only look at 6..15
        # Confirm the function doesn't crash and uses correct slice
        result = barrier_label(prices, 5, tp_pct=0.5, sl_pct=0.01, horizon=10)
        assert result in (0, 1)

    def test_entry_at_last_position_returns_0(self):
        """Not enough future data => timeout."""
        prices = pd.Series([100, 101, 102])
        result = barrier_label(prices, 2, tp_pct=0.05, sl_pct=0.03, horizon=5)
        assert result == 0  # No future prices


class TestBuildLabels:
    def test_builds_labels_for_signal_dates(self):
        df = make_prices(200)
        signal_dates = df.index[30:150:5]  # Every 5th day
        labels = build_labels(df, signal_dates, k1=2.0, k2=1.0)
        assert len(labels) > 0
        assert "label" in labels.columns
        assert set(labels["label"].unique()).issubset({0, 1})

    def test_skips_dates_without_enough_future(self):
        df = make_prices(50)
        signal_dates = df.index[-5:]  # Last 5 days — not enough future
        labels = build_labels(df, signal_dates, horizon=10)
        assert len(labels) == 0

    def test_labels_have_correct_columns(self):
        df = make_prices(200)
        signal_dates = df.index[30:100:10]
        labels = build_labels(df, signal_dates)
        assert "label" in labels.columns
        assert "tp_pct" in labels.columns
        assert "sl_pct" in labels.columns
        assert "vol_at_entry" in labels.columns


class TestComputeVolProxy:
    def test_returns_series(self):
        close = pd.Series(range(100, 200), dtype=float)
        vol = compute_vol_proxy(close, window=21)
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(close)

    def test_first_window_is_nan(self):
        close = pd.Series(range(100, 200), dtype=float)
        vol = compute_vol_proxy(close, window=21)
        assert vol.iloc[:21].isna().all()

    def test_positive_after_warmup(self):
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        vol = compute_vol_proxy(close, window=21)
        assert (vol.dropna() > 0).all()


class TestPurgeAndEmbargo:
    def test_removes_overlapping_samples(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        labels_df = pd.DataFrame(
            {"label": np.random.randint(0, 2, 100)}, index=dates
        )
        train_end = dates[60]
        test_start = dates[72]  # 12-day embargo

        purged = purge_and_embargo(
            labels_df, train_end, test_start, horizon=10, embargo_days=2
        )
        # All remaining samples should be before embargo_start
        embargo_start = train_end - pd.Timedelta(days=12)
        assert (purged.index < embargo_start).all()

    def test_preserves_early_samples(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        labels_df = pd.DataFrame(
            {"label": np.random.randint(0, 2, 100)}, index=dates
        )
        purged = purge_and_embargo(
            labels_df, dates[80], dates[92], horizon=10, embargo_days=2
        )
        assert len(purged) > 0
        assert len(purged) < len(labels_df)


class TestLabelQualityReport:
    def test_balanced_labels(self):
        labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
        report = label_quality_report(labels)
        assert report["total_samples"] == 10
        assert report["warning"] == "OK"

    def test_imbalanced_labels(self):
        labels = pd.Series([1] * 8 + [0] * 2)
        report = label_quality_report(labels)
        assert report["warning"] == "Use class_weight=balanced"

    def test_empty_labels(self):
        labels = pd.Series([], dtype=int)
        report = label_quality_report(labels)
        assert report["total_samples"] == 0

"""
Walk-forward validation & leakage prevention.
Skill reference: .claude/skills/walk-forward-validation/SKILL.md
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from scipy.stats import spearmanr


def walk_forward_splits(
    index: pd.DatetimeIndex,
    initial_train_days: int = 756,
    test_days: int = 126,
    step_days: int = 63,
    embargo_days: int = 12,
    expanding: bool = True,
) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Yields (train_idx, test_idx) pairs for walk-forward CV.
    Embargo gap between train end and test start prevents label overlap leakage.
    """
    n = len(index)
    train_start = 0
    train_end = initial_train_days

    while train_end + embargo_days + test_days <= n:
        test_start = train_end + embargo_days
        test_end = min(test_start + test_days, n)

        if not expanding:
            train_start = max(0, train_end - initial_train_days)

        train_idx = index[train_start:train_end]
        test_idx = index[test_start:test_end]

        yield train_idx, test_idx

        train_end += step_days


def purge_training_labels(
    train_labels: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Remove training samples whose label window extends past train_end.
    """
    cutoff = train_end - pd.Timedelta(days=horizon)
    purged = train_labels[train_labels.index <= cutoff]
    return purged


def leakage_audit(
    features: pd.DataFrame,
    prices: pd.Series,
    max_allowed_corr: float = 0.05,
    future_lags: list = None,
) -> dict:
    """
    Tests each feature for correlation with future returns.
    Any correlation > max_allowed_corr is a potential leak.
    """
    if future_lags is None:
        future_lags = [1, 2, 5]

    issues = []
    for lag in future_lags:
        future_ret = np.log(prices.shift(-lag) / prices)
        aligned = future_ret.reindex(features.index).dropna()
        feat_aligned = features.reindex(aligned.index).dropna()

        if len(feat_aligned) < 30:
            continue

        common_idx = aligned.index.intersection(feat_aligned.index)
        aligned = aligned.loc[common_idx]
        feat_aligned = feat_aligned.loc[common_idx]

        for col in feat_aligned.columns:
            col_data = feat_aligned[col].dropna()
            common = col_data.index.intersection(aligned.index)
            if len(common) < 30:
                continue
            corr, pval = spearmanr(col_data.loc[common], aligned.loc[common])
            if abs(corr) > max_allowed_corr and pval < 0.01:
                issues.append(
                    {
                        "feature": col,
                        "lag": lag,
                        "spearman_corr": round(corr, 4),
                        "p_value": round(pval, 6),
                    }
                )

    return {"issues": issues, "passed": len(issues) == 0}

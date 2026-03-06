---
name: walk-forward-validation
description: Walk-forward cross-validation for financial ML — expanding/rolling windows, embargo gaps, purging, leakage audit. Use whenever splitting train/test data, running CV, or auditing for look-ahead bias.
triggers:
  - walk-forward
  - train test split
  - cross validation
  - embargo
  - purging
  - leakage audit
  - look-ahead bias
  - OOS evaluation
priority: P1
---

# Skill: Walk-Forward Validation & Leakage Prevention

## What This Skill Is
Walk-forward validation is the only correct way to evaluate ML models on financial time series. Standard cross-validation is wrong for finance because it allows future data to train models that predict the past — a form of look-ahead bias. This skill defines how to split, purge, and evaluate models in a time-ordered, leak-free way.

---

## Why Standard Cross-Validation Fails in Finance

```
Standard K-Fold:
  Fold 1 test: [Jan, Feb, Mar]  — trained on [Apr...Dec] ← FUTURE DATA IN TRAINING
  Fold 2 test: [Apr, May, Jun]  — trained on [Jan-Mar, Jul-Dec] ← FUTURE DATA IN TRAINING

Walk-Forward:
  Fold 1: Train [Jan 2020 - Dec 2021] → Test [Jan 2022 - Jun 2022]
  Fold 2: Train [Jan 2020 - Jun 2022] → Test [Jul 2022 - Dec 2022]
  (expanding window) or
  Fold 2: Train [Jul 2020 - Jun 2022] → Test [Jul 2022 - Dec 2022]
  (rolling window)
```

Use **expanding window** (all history) unless you have strong reason to believe regime non-stationarity requires rolling.

---

## Walk-Forward Implementation

```python
import pandas as pd
import numpy as np
from typing import Iterator, Tuple

def walk_forward_splits(
    index: pd.DatetimeIndex,
    initial_train_days: int = 756,    # ~3 years
    test_days: int = 126,             # ~6 months
    step_days: int = 63,              # ~3 months
    embargo_days: int = 12,           # label_horizon + buffer
    expanding: bool = True,           # True=expanding, False=rolling
) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Yields (train_idx, test_idx) pairs for walk-forward CV.
    
    Embargo gap is applied between train end and test start to
    prevent label overlap leakage.
    """
    n = len(index)
    train_start = 0
    train_end = initial_train_days

    while train_end + embargo_days + test_days <= n:
        test_start = train_end + embargo_days
        test_end = min(test_start + test_days, n)

        if not expanding:
            train_start = train_end - initial_train_days

        train_idx = index[train_start:train_end]
        test_idx = index[test_start:test_end]

        yield train_idx, test_idx

        train_end += step_days

    
# Usage
splits = list(walk_forward_splits(df.index))
print(f"Total folds: {len(splits)}")
for i, (tr, te) in enumerate(splits):
    print(f"Fold {i+1}: Train {tr[0].date()}→{tr[-1].date()} | Test {te[0].date()}→{te[-1].date()}")
```

---

## The Purging Step

Even with embargo, some training labels may have their forward-looking window overlap with the test period. Purging removes these contaminated samples.

```python
def purge_training_labels(
    train_labels: pd.DataFrame,     # index = entry_date
    train_end: pd.Timestamp,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Remove training samples whose label window extends past train_end.
    A sample at date t has a label window of [t, t+horizon].
    If t + horizon > train_end, the label uses future (test-period) data.
    """
    cutoff = train_end - pd.Timedelta(days=horizon)
    purged = train_labels[train_labels.index <= cutoff]
    n_removed = len(train_labels) - len(purged)
    if n_removed > 0:
        print(f"Purged {n_removed} samples with forward-looking labels")
    return purged
```

---

## Full Walk-Forward Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score
import joblib

def run_walk_forward(
    features: pd.DataFrame,    # Full feature matrix, indexed by date
    labels: pd.Series,         # Full label series, indexed by date
    config: dict,
) -> dict:
    """
    Run complete walk-forward validation.
    Returns per-fold metrics and all OOS predictions.
    """
    all_preds = []
    fold_metrics = []

    splits = list(walk_forward_splits(
        features.index,
        initial_train_days=config['initial_train_days'],
        test_days=config['test_days'],
        step_days=config['step_days'],
        embargo_days=config['embargo_days'],
    ))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # --- TRAIN SET ---
        X_train_raw = features.loc[train_idx]
        y_train_raw = labels.loc[train_idx]

        # Purge: remove samples whose label window bleeds into test
        train_end = train_idx[-1]
        valid_mask = X_train_raw.index <= (train_end - pd.Timedelta(days=config['horizon_days']))
        X_train = X_train_raw[valid_mask]
        y_train = y_train_raw[valid_mask]

        # Drop rows where labels are NaN (unlabeled)
        valid = y_train.notna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 100:
            print(f"Fold {fold_idx+1}: insufficient training data, skipping")
            continue

        # --- TEST SET ---
        X_test = features.loc[test_idx]
        y_test = labels.loc[test_idx].dropna()
        X_test = X_test.loc[y_test.index]

        # --- MODEL PIPELINE ---
        # CRITICAL: scaler fitted ONLY on train data
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', HistGradientBoostingClassifier(
                class_weight='balanced',
                max_iter=200,
                random_state=42,
            ))
        ])

        # Calibration on last 20% of training data
        cal_split = int(len(X_train) * 0.8)
        model.fit(X_train.iloc[:cal_split], y_train.iloc[:cal_split])
        
        calibrated = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        calibrated.fit(X_train.iloc[cal_split:], y_train.iloc[cal_split:])

        # --- PREDICT ---
        probs = calibrated.predict_proba(X_test)[:, 1]
        preds = (probs >= config.get('entry_threshold', 0.6)).astype(int)

        # --- METRICS ---
        auc = roc_auc_score(y_test, probs) if len(y_test.unique()) > 1 else 0.5
        f1 = f1_score(y_test, preds, zero_division=0)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'train_start': train_idx[0],
            'train_end': train_idx[-1],
            'test_start': test_idx[0],
            'test_end': test_idx[-1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'roc_auc': auc,
            'f1': f1,
            'positive_rate': probs.mean(),
        })

        all_preds.append(pd.Series(probs, index=X_test.index, name='prob'))

        # Save model for last fold (most recent = production candidate)
        if fold_idx == len(splits) - 1:
            joblib.dump(calibrated, config.get('model_path', 'model.pkl'))
            print(f"Saved model trained through {train_idx[-1].date()}")

        print(f"Fold {fold_idx+1}: AUC={auc:.3f}  F1={f1:.3f}  n_train={len(X_train)}")

    metrics_df = pd.DataFrame(fold_metrics)
    preds_series = pd.concat(all_preds).sort_index()

    print(f"\nMean OOS AUC: {metrics_df['roc_auc'].mean():.3f}")
    print(f"Mean OOS F1:  {metrics_df['f1'].mean():.3f}")

    return {
        'fold_metrics': metrics_df,
        'oos_predictions': preds_series,
        'deployment_ready': metrics_df['roc_auc'].mean() >= 0.55,
    }
```

---

## Leakage Audit Function

Run this before every training run. It should find no significant correlations.

```python
from scipy.stats import spearmanr

def leakage_audit(
    features: pd.DataFrame,
    prices: pd.Series,          # Close prices
    max_allowed_corr: float = 0.05,
    future_lags: list = [1, 2, 5],
) -> dict:
    """
    Tests each feature for correlation with future returns.
    Any correlation > max_allowed_corr is a potential leak.
    """
    issues = []
    for lag in future_lags:
        future_ret = np.log(prices / prices.shift(-lag)).shift(-1)  # Forward return
        aligned = future_ret.reindex(features.index).dropna()
        feat_aligned = features.reindex(aligned.index)

        for col in features.columns:
            corr, pval = spearmanr(feat_aligned[col].fillna(0), aligned)
            if abs(corr) > max_allowed_corr and pval < 0.01:
                issues.append({
                    'feature': col,
                    'lag': lag,
                    'spearman_corr': round(corr, 4),
                    'p_value': round(pval, 6),
                })

    if issues:
        print(f"WARNING: {len(issues)} potential leakage issues found:")
        for iss in issues:
            print(f"  {iss['feature']} @ lag {iss['lag']}: corr={iss['spearman_corr']}")
    else:
        print("Leakage audit PASSED — no significant future correlations found.")

    return {'issues': issues, 'passed': len(issues) == 0}
```

---

## Quick-Reference: Leakage Checklist

Before every training run, confirm:

- [ ] All features at time `t` use only data from `t` or earlier
- [ ] `RobustScaler` / `StandardScaler` fit only on training fold
- [ ] Embargo gap >= `label_horizon + 2` days
- [ ] Purging applied: training samples with `entry_date + horizon > train_end` removed
- [ ] Cross-sectional ranks computed within each date's training universe only
- [ ] Regime labels trained causally (no future vol used to classify past regimes)
- [ ] `leakage_audit()` run and passed

---

## Configuration

```yaml
walk_forward:
  initial_train_days: 756    # ~3 years
  test_days: 126             # ~6 months per fold
  step_days: 63              # ~3 months step
  embargo_days: 12           # horizon + 2 buffer
  expanding: true            # expanding window (recommended)
  min_train_samples: 100     # Skip fold if less
  horizon_days: 10           # Must match labeling config

deployment_gates:
  min_oos_auc: 0.55
  min_oos_f1: 0.45
  disable_if_rolling_sharpe_below: 0.5   # Over trailing 90 days live
```

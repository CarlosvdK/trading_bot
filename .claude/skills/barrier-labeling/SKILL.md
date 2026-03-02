# Skill: Barrier Labeling & ML Label Design

## What This Skill Is
Barrier labeling is the correct way to create supervised learning targets for trade-filtering models. Instead of predicting raw future returns (which are noisy and symmetric), you define two price barriers — a Take-Profit (TP) and a Stop-Loss (SL) — and label each trade based on which barrier is hit first within a maximum holding horizon H.

---

## Core Concept: The Triple-Barrier Method

For each candidate entry at time `t`:

```
TP barrier  = entry_price * (1 + k1 * vol_proxy)
SL barrier  = entry_price * (1 - k2 * vol_proxy)
Horizon     = t + H trading days

Label = +1  if TP hit before SL and before t+H
Label = -1  if SL hit before TP and before t+H
Label =  0  if neither hit before t+H (timeout)
```

### Recommended Simplified Version (Binary)
Start with binary labels — it reduces class imbalance complexity:

```
Label = 1   if TP hit first (favorable outcome)
Label = 0   if SL hit first OR timeout (unfavorable)
```

Use three-class only after binary is working and when you need directional asymmetry.

---

## Vol Proxy: How to Compute It (No TA Libraries)

**Never use ATR from a TA library.** Compute it directly from returns:

```python
import numpy as np
import pandas as pd

def compute_vol_proxy(close: pd.Series, window: int = 21) -> pd.Series:
    """
    Annualized rolling volatility from log returns.
    Computed strictly from past data — no future leakage.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)

# Usage
vol = compute_vol_proxy(df['close'], window=21)
tp_distance = k1 * vol  # e.g. k1 = 0.02 for 2% of annualized vol
sl_distance = k2 * vol  # e.g. k2 = 0.01
```

**Key rule:** `vol_proxy` at time `t` uses only prices up to and including `t`. Never look forward.

---

## Labeling Function (Leak-Proof Implementation)

```python
import pandas as pd
import numpy as np

def barrier_label(
    prices: pd.Series,       # Close prices, indexed by date
    entry_idx: int,          # Integer location of entry bar
    tp_pct: float,           # e.g. 0.02 (2%)
    sl_pct: float,           # e.g. 0.01 (1%)
    horizon: int = 10,       # Max holding days
) -> int:
    """
    Returns: 1 (TP hit), 0 (SL hit or timeout)
    
    CRITICAL: Only uses prices[entry_idx : entry_idx + horizon + 1]
    Never references any price before entry for the label itself.
    """
    entry_price = prices.iloc[entry_idx]
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    future = prices.iloc[entry_idx + 1 : entry_idx + 1 + horizon]

    for price in future:
        if price >= tp_price:
            return 1
        if price <= sl_price:
            return 0

    return 0  # Timeout = unfavorable


def build_labels(
    df: pd.DataFrame,           # Must have 'close' column
    signal_dates: pd.DatetimeIndex,  # Dates where signals fire
    k1: float = 2.0,            # TP multiplier
    k2: float = 1.0,            # SL multiplier
    vol_window: int = 21,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Build labeled dataset for all signal dates.
    Returns DataFrame with entry_date, label, tp_pct, sl_pct.
    """
    log_ret = np.log(df['close'] / df['close'].shift(1))
    vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    results = []
    for date in signal_dates:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        # Need at least horizon bars ahead
        if idx + horizon >= len(df):
            continue
        # Vol at entry time (backward-looking only)
        v = vol.iloc[idx]
        if pd.isna(v):
            continue
        # Daily vol proxy (de-annualize for daily barriers)
        daily_vol = v / np.sqrt(252)
        tp_pct = k1 * daily_vol * np.sqrt(horizon)
        sl_pct = k2 * daily_vol * np.sqrt(horizon)

        label = barrier_label(df['close'], idx, tp_pct, sl_pct, horizon)
        results.append({
            'entry_date': date,
            'label': label,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'vol_at_entry': v,
        })

    return pd.DataFrame(results).set_index('entry_date')
```

---

## Embargo & Purging (Critical for Walk-Forward)

Labels span from `t` to `t + horizon`. This creates an overlap problem:

```
Train period ends at:  t_train_end
Label horizon:         H = 10 days
Embargo zone:          t_train_end to t_train_end + H + 2 days
Test period starts at: t_train_end + H + 2 days (minimum)

Purge from training:   Remove all samples where
                       entry_date + horizon >= t_train_end
```

```python
def purge_and_embargo(
    labeled_df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    horizon: int = 10,
    embargo_days: int = 2,
) -> pd.DataFrame:
    """
    Remove training samples whose labels overlap the test period.
    """
    embargo_start = train_end - pd.Timedelta(days=horizon + embargo_days)
    clean = labeled_df[labeled_df.index < embargo_start]
    return clean
```

---

## Label Quality Checks

Run these after building labels — before training:

```python
def label_quality_report(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    total = len(labels)
    return {
        'total_samples': total,
        'class_1_pct': counts.get(1, 0) / total,
        'class_0_pct': counts.get(0, 0) / total,
        'imbalance_ratio': counts.get(1, 0) / max(counts.get(0, 1), 1),
        'warning': 'Use class_weight=balanced' if abs(counts.get(1,0)/total - 0.5) > 0.15 else 'OK'
    }
```

**Healthy label distribution:** 40-60% class 1. If heavily skewed (< 30% or > 70% class 1), adjust k1/k2 multipliers or horizon.

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong | Fix |
|---|---|---|
| Computing vol_proxy on full dataset then slicing | Future vol leaks into past features | Compute vol inside each walk-forward fold |
| Using adjusted close for labels but unadjusted for features | Price level mismatch | Use same price series throughout |
| Label timeout as class 0 with no consideration | Timeouts may be informative | Log timeout rate separately; if > 50%, shorten horizon or widen barriers |
| Using the same vol_window for features AND barriers | Overfits the labeling to features | Use separate, documented windows |

---

## Configuration Parameters

```yaml
labeling:
  k1: 2.0              # TP multiplier (in vol units)
  k2: 1.0              # SL multiplier (in vol units)  
  horizon_days: 10     # Max holding period for label
  vol_window: 21       # Rolling window for vol proxy
  embargo_days: 12     # label_horizon + 2 buffer
  binary_labels: true  # true=binary, false=three-class
```

---

## References
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapters 3 & 4.
- The triple-barrier method is the standard for supervised trade labeling in institutional quant research.

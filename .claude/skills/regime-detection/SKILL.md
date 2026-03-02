# Skill: Regime Detection

## What This Skill Is
Regime detection classifies the current market environment into discrete states (e.g., trending/choppy, high-vol/low-vol, risk-on/risk-off). The regime label is used by the Risk Governor and sleeve engines to adjust position sizing, disable the Swing sleeve during unfavorable regimes, and widen/narrow barriers.

---

## Why Regime Detection Matters

| Regime | Swing Behavior | Action |
|---|---|---|
| Trending + Low Vol | Momentum works well | Run swing at full allocation |
| Choppy + Low Vol | Mean-reversion preferred, momentum fails | Reduce swing allocation 50% |
| High Vol + Risk-Off | Whipsaws, fills worse, correlation spikes | Disable swing or paper-trade only |
| Vol Spike (VVol high) | Tail risk elevated, sizing models unstable | Halve all position sizes |

---

## Approach: Feature-Based Regime Clustering

Use an unsupervised approach (HMM or KMeans) to detect latent regimes. Then label states post-hoc by inspecting their feature profiles. This avoids the leakage problem of defining regimes by future returns.

---

## Step 1: Regime Features (Backward-Looking)

```python
import numpy as np
import pandas as pd

def build_regime_features(
    index_close: pd.Series,    # Reference index (e.g. SPY)
    config: dict,
) -> pd.DataFrame:
    """
    Features for regime classification.
    All backward-looking. Computed on reference index.
    """
    log_ret = np.log(index_close / index_close.shift(1))
    feats = pd.DataFrame(index=index_close.index)

    # Trend
    feats['ret_5d'] = np.log(index_close / index_close.shift(5))
    feats['ret_21d'] = np.log(index_close / index_close.shift(21))
    feats['ret_63d'] = np.log(index_close / index_close.shift(63))

    # Volatility level
    feats['vol_21d'] = log_ret.rolling(21).std() * np.sqrt(252)
    feats['vol_63d'] = log_ret.rolling(63).std() * np.sqrt(252)

    # Vol regime change
    feats['vol_ratio'] = feats['vol_21d'] / feats['vol_63d']   # >1 = expanding vol

    # Vol-of-vol (instability)
    feats['vvol'] = feats['vol_21d'].rolling(21).std()

    # Trend consistency (up-day fraction)
    feats['trend_strength'] = (log_ret > 0).rolling(21).mean()

    # Drawdown from rolling peak
    rolling_peak = index_close.rolling(63).max()
    feats['drawdown_63d'] = (index_close - rolling_peak) / rolling_peak

    return feats.dropna()
```

---

## Step 2: HMM Regime Model

```python
# Requires: pip install hmmlearn
# Falls back to KMeans if not available

import warnings

def fit_regime_model(
    features: pd.DataFrame,
    n_regimes: int = 4,
    method: str = "hmm",     # "hmm" or "kmeans"
) -> object:
    """
    Fit a regime detection model on training features.
    Returns fitted model.
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X = scaler.fit_transform(features.fillna(0))

    if method == "hmm":
        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(X)
            return {"type": "hmm", "model": model, "scaler": scaler}
        except ImportError:
            warnings.warn("hmmlearn not installed, falling back to KMeans")

    # KMeans fallback
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    model.fit(X)
    return {"type": "kmeans", "model": model, "scaler": scaler}


def predict_regime(
    model_dict: dict,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Predict regime labels for given features.
    Returns integer labels indexed by date.
    """
    scaler = model_dict["scaler"]
    model = model_dict["model"]
    X = scaler.transform(features.fillna(0))

    if model_dict["type"] == "hmm":
        labels = model.predict(X)
    else:
        labels = model.predict(X)

    return pd.Series(labels, index=features.index, name="regime")
```

---

## Step 3: Label Regimes Post-Hoc

After fitting, inspect each cluster's feature profile to assign meaningful names:

```python
def label_regimes(
    features: pd.DataFrame,
    regime_series: pd.Series,
) -> dict:
    """
    Returns dict mapping regime_id -> regime_name based on feature profiles.
    Human-readable labels assigned after inspection.
    """
    profiles = features.copy()
    profiles["regime"] = regime_series

    summary = profiles.groupby("regime").mean()

    regime_names = {}
    for regime_id, row in summary.iterrows():
        vol = row.get("vol_21d", 0)
        trend = row.get("ret_21d", 0)
        ratio = row.get("vol_ratio", 1)

        if vol > summary["vol_21d"].median() * 1.3:
            vol_label = "high_vol"
        else:
            vol_label = "low_vol"

        if abs(trend) > summary["ret_21d"].abs().median():
            trend_label = "trending_up" if trend > 0 else "trending_down"
        else:
            trend_label = "choppy"

        regime_names[regime_id] = f"{vol_label}_{trend_label}"

    print("Regime profiles:")
    print(summary.round(4))
    print("\nAssigned names:", regime_names)
    return regime_names
```

---

## Step 4: Regime → Allocation Mapping

```python
REGIME_ALLOCATION = {
    "low_vol_trending_up":    {"swing_multiplier": 1.0, "swing_enabled": True},
    "low_vol_choppy":         {"swing_multiplier": 0.5, "swing_enabled": True},
    "low_vol_trending_down":  {"swing_multiplier": 0.5, "swing_enabled": True},
    "high_vol_trending_up":   {"swing_multiplier": 0.3, "swing_enabled": True},
    "high_vol_choppy":        {"swing_multiplier": 0.0, "swing_enabled": False},
    "high_vol_trending_down": {"swing_multiplier": 0.0, "swing_enabled": False},
}

def get_regime_allocation(regime_name: str) -> dict:
    return REGIME_ALLOCATION.get(regime_name, {"swing_multiplier": 0.5, "swing_enabled": True})
```

---

## Step 5: Walk-Forward Regime Model

Regime model must be retrained causally — only on data available up to the train fold's end date.

```python
def run_regime_walk_forward(
    index_close: pd.Series,
    config: dict,
) -> pd.Series:
    """
    Walk-forward regime predictions. At each fold, model trained on
    data up to fold boundary, predicts next fold only.
    Returns full out-of-sample regime series.
    """
    feat_df = build_regime_features(index_close, config)
    all_preds = []

    initial = config.get("initial_train_days", 504)   # 2 years
    step = config.get("step_days", 63)
    n_regimes = config.get("n_regimes", 4)
    method = config.get("regime_method", "hmm")

    for train_end_i in range(initial, len(feat_df), step):
        train_feats = feat_df.iloc[:train_end_i]
        test_feats = feat_df.iloc[train_end_i : train_end_i + step]

        if len(test_feats) == 0:
            break

        model_dict = fit_regime_model(train_feats, n_regimes, method)
        preds = predict_regime(model_dict, test_feats)
        all_preds.append(preds)

    return pd.concat(all_preds).sort_index()
```

---

## Integration with Risk Governor

```python
# In your main trading loop:
current_regime = regime_series.loc[current_date]
regime_name = regime_names.get(current_regime, "unknown")
allocation = get_regime_allocation(regime_name)

if not allocation["swing_enabled"]:
    risk_governor.swing_halted_until = next_regime_check_date
    logger.warning(f"Swing disabled: regime={regime_name}")

# Adjust swing position sizes
swing_size_multiplier = allocation["swing_multiplier"]
# Pass to position sizer: actual_size = vol_target_size * swing_size_multiplier
```

---

## Regime Persistence Check

Before trusting a regime shift, require it to persist for N consecutive days:

```python
def smooth_regime(
    regime_series: pd.Series,
    min_persistence: int = 3,   # Require N consecutive days in new regime
) -> pd.Series:
    """
    Prevents whipsawing between regimes on noisy days.
    Only switches regime after N consecutive days in new state.
    """
    smoothed = regime_series.copy()
    current = regime_series.iloc[0]
    streak = 1

    for i in range(1, len(regime_series)):
        if regime_series.iloc[i] == current:
            streak += 1
        else:
            if streak >= min_persistence:
                current = regime_series.iloc[i]
                streak = 1
            # else: stay in current regime, reset streak counter

        smoothed.iloc[i] = current

    return smoothed
```

---

## Configuration

```yaml
regime_detection:
  method: "hmm"              # "hmm" or "kmeans"
  n_regimes: 4
  initial_train_days: 504    # ~2 years
  step_days: 63              # Retrain every quarter
  min_persistence_days: 3    # Days before regime switch is accepted
  index_symbol: "SPY"        # Reference index

  allocation_by_regime:
    low_vol_trending_up:    { swing_multiplier: 1.0, swing_enabled: true }
    low_vol_choppy:         { swing_multiplier: 0.5, swing_enabled: true }
    high_vol_choppy:        { swing_multiplier: 0.0, swing_enabled: false }
    high_vol_trending_down: { swing_multiplier: 0.0, swing_enabled: false }
```

---

## Key Warnings

- **Never use future vol to define training regime labels** — this is the most common regime-model leakage mistake.
- Regime labels are only interpretable post-hoc. They have no inherent meaning (regime "2" is not always "bull market").
- Regime models work best with 3-5 years of training data. Fewer than 2 years = unreliable.
- Monitor regime transition frequency. If the model switches regime every day, the `min_persistence` filter needs increasing.

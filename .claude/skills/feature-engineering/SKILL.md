# Skill: Feature Engineering for Trade Filtering

## What This Skill Is
How to build a clean, leak-free feature matrix for the Trade Filter and Regime Detection ML models. All features are derived from price/volume data only, using rolling backward-looking windows. No TA library required.

---

## The Golden Rule
> **Every feature at time `t` must use only data available at or before time `t`.**

This means:
- Rolling windows end at `t` (never include `t+1` or later)
- No `shift(-1)` in features (only `shift(1)` or greater)
- Normalize/standardize using rolling stats, not full-dataset stats

---

## Feature Registry

### 1. Rolling Returns (Momentum Signal)

```python
def rolling_returns(close: pd.Series, windows: list = [5, 10, 21]) -> pd.DataFrame:
    """Log returns over multiple lookback windows."""
    log_close = np.log(close)
    return pd.DataFrame({
        f'ret_{w}d': log_close - log_close.shift(w)
        for w in windows
    })
```

### 2. Rolling Volatility (Risk & Regime Signal)

```python
def rolling_vol(close: pd.Series, windows: list = [5, 21, 63]) -> pd.DataFrame:
    """Annualized realized volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return pd.DataFrame({
        f'vol_{w}d': log_ret.rolling(w).std() * np.sqrt(252)
        for w in windows
    })
```

### 3. Gap Return (Overnight Gap)

```python
def gap_return(open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Today's open vs. yesterday's close."""
    return np.log(open_ / prev_close.shift(1)).rename('gap_return')
```

### 4. Volume Surprise

```python
def volume_surprise(volume: pd.Series, window: int = 21) -> pd.Series:
    """Today's volume relative to recent average."""
    avg_vol = volume.rolling(window).mean()
    return ((volume / avg_vol) - 1).rename('volume_surprise')
```

### 5. Volatility Ratio (Vol Regime Change Detector)

```python
def vol_ratio(close: pd.Series, short: int = 5, long: int = 21) -> pd.Series:
    """Short-term vol / long-term vol. >1 = expanding vol."""
    log_ret = np.log(close / close.shift(1))
    short_vol = log_ret.rolling(short).std()
    long_vol = log_ret.rolling(long).std()
    return (short_vol / long_vol).rename('vol_ratio')
```

### 6. Vol-of-Vol (VVol Proxy — Tail Risk Signal)

```python
def vol_of_vol(close: pd.Series, vol_window: int = 21, vvol_window: int = 21) -> pd.Series:
    """Rolling std of daily vol changes. High = unstable vol regime."""
    log_ret = np.log(close / close.shift(1))
    daily_vol = log_ret.rolling(vol_window).std()
    return daily_vol.rolling(vvol_window).std().rename('vvol_proxy')
```

### 7. Index Context Features (Cross-Asset)

```python
def index_features(index_close: pd.Series, windows: list = [5, 21]) -> pd.DataFrame:
    """Market context from a reference index (e.g., SPY)."""
    log_ret = np.log(index_close / index_close.shift(1))
    feats = {}
    for w in windows:
        feats[f'index_ret_{w}d'] = np.log(index_close / index_close.shift(w))
        feats[f'index_vol_{w}d'] = log_ret.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(feats)
```

### 8. Cross-Sectional Rank Features

```python
def cross_sectional_ranks(
    returns_matrix: pd.DataFrame,   # rows=dates, cols=symbols
    col: str = 'ret_5d',
) -> pd.DataFrame:
    """
    Percentile rank of each symbol's feature within the daily universe.
    CRITICAL: rank computed within each row (date) only — no future info.
    """
    return returns_matrix.rank(axis=1, pct=True).add_suffix('_cs_rank')
```

### 9. Momentum Consistency (Trend Quality)

```python
def momentum_consistency(close: pd.Series, window: int = 10) -> pd.Series:
    """
    Fraction of up days in the last N days.
    1.0 = strong uptrend, 0.0 = strong downtrend.
    """
    daily_ret = close.pct_change()
    return (daily_ret > 0).rolling(window).mean().rename('mom_consistency')
```

---

## Full Feature Builder

```python
import pandas as pd
import numpy as np

def build_features(
    symbol_df: pd.DataFrame,    # columns: open, high, low, close, volume
    index_df: pd.DataFrame,     # columns: close (reference index)
    config: dict,
) -> pd.DataFrame:
    """
    Builds complete feature matrix for one symbol.
    All features are strictly backward-looking.
    
    Args:
        symbol_df: OHLCV DataFrame indexed by date
        index_df:  Index OHLCV DataFrame (e.g., SPY), indexed by date
        config:    Feature config dict
    
    Returns:
        DataFrame of features, indexed by date. NaN rows = insufficient history.
    """
    df = symbol_df.copy()
    idx_close = index_df['close'].reindex(df.index).ffill()

    feats = pd.DataFrame(index=df.index)

    # --- Return features ---
    for w in config.get('return_windows', [5, 10, 21]):
        log_close = np.log(df['close'])
        feats[f'ret_{w}d'] = log_close - log_close.shift(w)

    # --- Volatility features ---
    log_ret = np.log(df['close'] / df['close'].shift(1))
    for w in config.get('vol_windows', [5, 21]):
        feats[f'vol_{w}d'] = log_ret.rolling(w).std() * np.sqrt(252)

    # --- Vol ratio ---
    feats['vol_ratio_5_21'] = feats['vol_5d'] / feats['vol_21d']

    # --- VVol proxy ---
    feats['vvol_proxy'] = feats['vol_21d'].rolling(21).std()

    # --- Gap ---
    feats['gap_return'] = np.log(df['open'] / df['close'].shift(1))

    # --- Volume surprise ---
    feats['volume_surprise'] = (df['volume'] / df['volume'].rolling(21).mean()) - 1

    # --- Momentum consistency ---
    feats['mom_consistency_10d'] = (df['close'].pct_change() > 0).rolling(10).mean()

    # --- Index context ---
    for w in [5, 21]:
        log_idx = np.log(idx_close)
        feats[f'index_ret_{w}d'] = log_idx - log_idx.shift(w)
        feats[f'index_vol_{w}d'] = np.log(idx_close / idx_close.shift(1)).rolling(w).std() * np.sqrt(252)

    # --- Relative return (symbol vs index) ---
    feats['rel_ret_5d'] = feats['ret_5d'] - feats['index_ret_5d']

    # --- Winsorize all features at 3 sigma (based on rolling window) ---
    for col in feats.columns:
        roll_mean = feats[col].rolling(252, min_periods=60).mean()
        roll_std = feats[col].rolling(252, min_periods=60).std()
        feats[col] = feats[col].clip(
            lower=roll_mean - 3 * roll_std,
            upper=roll_mean + 3 * roll_std,
        )

    return feats


def winsorize_zscore(
    feats: pd.DataFrame,
    window: int = 252,
    clip_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Z-score features using rolling mean/std (backward-looking).
    Clips at clip_sigma to remove outliers.
    This is the correct normalization for time-series features.
    
    NEVER use StandardScaler on the full dataset before splitting.
    Use this inside the walk-forward train fold, or use RobustScaler in pipeline.
    """
    result = pd.DataFrame(index=feats.index)
    for col in feats.columns:
        roll_mean = feats[col].rolling(window, min_periods=60).mean()
        roll_std = feats[col].rolling(window, min_periods=60).std().replace(0, np.nan)
        z = (feats[col] - roll_mean) / roll_std
        result[col] = z.clip(-clip_sigma, clip_sigma)
    return result
```

---

## Feature Importance & Collinearity Check

```python
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

def check_feature_collinearity(X: pd.DataFrame, vif_threshold: float = 10.0) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.
    Features with VIF > threshold are candidates for removal.
    """
    from numpy.linalg import inv
    X_arr = X.dropna().values
    corr = np.corrcoef(X_arr, rowvar=False)
    try:
        inv_corr = inv(corr)
        vif = pd.Series(np.diag(inv_corr), index=X.columns, name='VIF')
        flagged = vif[vif > vif_threshold]
        if len(flagged) > 0:
            print(f"High collinearity features (VIF > {vif_threshold}):")
            print(flagged.sort_values(ascending=False))
        return vif
    except np.linalg.LinAlgError:
        print("Singular matrix — features are perfectly collinear")
        return pd.Series()
```

---

## Feature Config

```yaml
features:
  return_windows: [5, 10, 21]
  vol_windows: [5, 21]
  volume_surprise_window: 21
  momentum_consistency_window: 10
  vvol_window: 21
  index_symbol: "SPY"           # Reference index for context features
  winsorize_sigma: 3.0
  rolling_norm_window: 252
  min_history_days: 63          # Drop rows with less history than this
```

---

## Anti-Patterns

| Don't Do This | Do This Instead |
|---|---|
| `StandardScaler().fit_transform(full_dataset)` | Fit scaler inside walk-forward train fold only |
| `talib.RSI(close, 14)` | Use `rolling_returns` and `rolling_vol` |
| `df['future_ret'] = df['close'].pct_change(-1)` | Only use `pct_change(+1)` (past) for features |
| Cross-sectional ranks using test-period symbols | Lock universe at train-set composition |
| Computing features after label creation | Features and labels are independent; compute features first |

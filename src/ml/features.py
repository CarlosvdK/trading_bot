"""
Leak-free feature engineering for trade filtering.
Skill reference: .claude/skills/feature-engineering/SKILL.md
"""

import numpy as np
import pandas as pd


def build_features(
    symbol_df: pd.DataFrame,
    index_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Builds complete feature matrix for one symbol.
    All features are strictly backward-looking.
    """
    df = symbol_df.copy()
    idx_close = index_df["close"].reindex(df.index).ffill()

    feats = pd.DataFrame(index=df.index)

    # --- Return features ---
    log_close = np.log(df["close"])
    for w in config.get("return_windows", [5, 10, 21]):
        feats[f"ret_{w}d"] = log_close - log_close.shift(w)

    # --- Volatility features ---
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for w in config.get("vol_windows", [5, 21]):
        feats[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # --- Vol ratio ---
    if "vol_5d" in feats.columns and "vol_21d" in feats.columns:
        feats["vol_ratio_5_21"] = feats["vol_5d"] / feats["vol_21d"]

    # --- VVol proxy ---
    if "vol_21d" in feats.columns:
        feats["vvol_proxy"] = feats["vol_21d"].rolling(21).std()

    # --- Gap ---
    feats["gap_return"] = np.log(df["open"] / df["close"].shift(1))

    # --- Volume surprise ---
    feats["volume_surprise"] = (
        df["volume"] / df["volume"].rolling(21).mean()
    ) - 1

    # --- Momentum consistency ---
    feats["mom_consistency_10d"] = (
        (df["close"].pct_change() > 0).rolling(10).mean()
    )

    # --- Index context ---
    if idx_close is not None and not idx_close.isna().all():
        log_idx = np.log(idx_close)
        idx_log_ret = np.log(idx_close / idx_close.shift(1))
        for w in [5, 21]:
            feats[f"index_ret_{w}d"] = log_idx - log_idx.shift(w)
            feats[f"index_vol_{w}d"] = (
                idx_log_ret.rolling(w).std() * np.sqrt(252)
            )

    # --- Relative return (symbol vs index) ---
    if "ret_5d" in feats.columns and "index_ret_5d" in feats.columns:
        feats["rel_ret_5d"] = feats["ret_5d"] - feats["index_ret_5d"]

    # --- Winsorize all features at 3 sigma (rolling) ---
    clip_sigma = config.get("winsorize_sigma", 3.0)
    norm_window = config.get("rolling_norm_window", 252)
    for col in feats.columns:
        roll_mean = feats[col].rolling(norm_window, min_periods=60).mean()
        roll_std = feats[col].rolling(norm_window, min_periods=60).std()
        feats[col] = feats[col].clip(
            lower=roll_mean - clip_sigma * roll_std,
            upper=roll_mean + clip_sigma * roll_std,
        )

    return feats


def build_single(
    symbol_df: pd.DataFrame,
    index_df: pd.DataFrame,
    date: pd.Timestamp,
    config: dict,
) -> pd.Series:
    """Build feature vector for a single date."""
    feats = build_features(symbol_df, index_df, config)
    if date not in feats.index:
        return None
    row = feats.loc[date]
    if row.isnull().any():
        return None
    return row


def winsorize_zscore(
    feats: pd.DataFrame,
    window: int = 252,
    clip_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Z-score features using rolling mean/std (backward-looking).
    NEVER use StandardScaler on the full dataset before splitting.
    """
    result = pd.DataFrame(index=feats.index)
    for col in feats.columns:
        roll_mean = feats[col].rolling(window, min_periods=60).mean()
        roll_std = (
            feats[col].rolling(window, min_periods=60).std().replace(0, np.nan)
        )
        z = (feats[col] - roll_mean) / roll_std
        result[col] = z.clip(-clip_sigma, clip_sigma)
    return result


def check_feature_collinearity(
    X: pd.DataFrame, vif_threshold: float = 10.0
) -> pd.Series:
    """
    Compute Variance Inflation Factor for each feature.
    Features with VIF > threshold are candidates for removal.
    """
    from numpy.linalg import inv

    X_clean = X.dropna()
    if len(X_clean) < 10 or len(X_clean.columns) < 2:
        return pd.Series(dtype=float)

    corr = np.corrcoef(X_clean.values, rowvar=False)
    try:
        inv_corr = inv(corr)
        vif = pd.Series(np.diag(inv_corr), index=X.columns, name="VIF")
        return vif
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float)

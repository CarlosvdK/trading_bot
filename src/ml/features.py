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
    sector_etf_df: pd.DataFrame = None,
    universe_closes: pd.DataFrame = None,
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
    if "ret_21d" in feats.columns and "index_ret_21d" in feats.columns:
        feats["rel_ret_21d"] = feats["ret_21d"] - feats["index_ret_21d"]

    # --- RSI (14-day, backward-looking) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD signal (12/26/9 EMA, backward-looking) ---
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_hist"] = (macd_line - signal_line) / df["close"]  # Normalize

    # --- Bollinger %B (20-day, backward-looking) ---
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    feats["bband_pctb"] = (df["close"] - (bb_mid - 2 * bb_std)) / (
        4 * bb_std.replace(0, np.nan)
    )

    # --- Dollar volume momentum (liquidity trend) ---
    dv = df["close"] * df["volume"]
    dv_5 = dv.rolling(5).mean()
    dv_21 = dv.rolling(21).mean()
    feats["dv_momentum"] = dv_5 / dv_21.replace(0, np.nan) - 1

    # --- Average true range ratio (volatility regime) ---
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_5 = tr.rolling(5).mean()
    atr_21 = tr.rolling(21).mean()
    feats["atr_ratio"] = atr_5 / atr_21.replace(0, np.nan)

    # --- Mean reversion signal (distance from 50-day MA) ---
    ma50 = df["close"].rolling(50).mean()
    feats["dist_ma50_pct"] = (df["close"] - ma50) / ma50.replace(0, np.nan)

    # --- Sector features (if sector ETF data provided) ---
    if sector_etf_df is not None and not sector_etf_df.empty:
        sector_close = sector_etf_df["close"].reindex(df.index).ffill()
        log_sector = np.log(sector_close.replace(0, np.nan))

        feats["sector_ret_21d"] = log_sector - log_sector.shift(21)

        if "ret_21d" in feats.columns:
            feats["sector_rel_ret"] = feats["ret_21d"] - feats["sector_ret_21d"]

    # --- Market breadth features (if universe close prices provided) ---
    if universe_closes is not None and not universe_closes.empty:
        uc = universe_closes.reindex(df.index).ffill()

        # % of universe above their 50-day MA
        uc_ma50 = uc.rolling(50, min_periods=30).mean()
        pct_above = (uc > uc_ma50).astype(float).mean(axis=1)
        feats["breadth_pct_above_ma50"] = pct_above

        # Rolling 10-day advance/decline ratio
        daily_ret = uc.pct_change()
        advancing = (daily_ret > 0).sum(axis=1)
        declining = (daily_ret < 0).sum(axis=1).replace(0, np.nan)
        ad_ratio = advancing / declining
        feats["breadth_advance_decline"] = ad_ratio.rolling(10).mean()

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

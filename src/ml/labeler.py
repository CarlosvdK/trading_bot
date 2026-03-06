"""
Barrier labeling for ML training — triple-barrier method.
Skill reference: .claude/skills/barrier-labeling/SKILL.md
"""

import numpy as np
import pandas as pd


def compute_vol_proxy(close: pd.Series, window: int = 21) -> pd.Series:
    """
    Annualized rolling volatility from log returns.
    Computed strictly from past data — no future leakage.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)


def barrier_label(
    prices: pd.Series,
    entry_idx: int,
    tp_pct: float,
    sl_pct: float,
    horizon: int = 10,
) -> int:
    """
    Returns: 1 (TP hit), 0 (SL hit or timeout).

    CRITICAL: Only uses prices[entry_idx : entry_idx + horizon + 1].
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
    df: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    k1: float = 2.0,
    k2: float = 1.0,
    vol_window: int = 21,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Build labeled dataset for all signal dates.
    Returns DataFrame with entry_date, label, tp_pct, sl_pct.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    results = []
    for date in signal_dates:
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx + horizon >= len(df):
            continue
        v = vol.iloc[idx]
        if pd.isna(v) or v <= 0:
            continue
        daily_vol = v / np.sqrt(252)
        tp_pct = k1 * daily_vol * np.sqrt(horizon)
        sl_pct = k2 * daily_vol * np.sqrt(horizon)

        label = barrier_label(df["close"], idx, tp_pct, sl_pct, horizon)
        results.append(
            {
                "entry_date": date,
                "label": label,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "vol_at_entry": v,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=["label", "tp_pct", "sl_pct", "vol_at_entry"]
        )

    return pd.DataFrame(results).set_index("entry_date")


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


def label_quality_report(labels: pd.Series) -> dict:
    """Check label balance."""
    counts = labels.value_counts()
    total = len(labels)
    if total == 0:
        return {"total_samples": 0, "warning": "No samples"}
    return {
        "total_samples": total,
        "class_1_pct": counts.get(1, 0) / total,
        "class_0_pct": counts.get(0, 0) / total,
        "imbalance_ratio": counts.get(1, 0) / max(counts.get(0, 1), 1),
        "warning": (
            "Use class_weight=balanced"
            if abs(counts.get(1, 0) / total - 0.5) > 0.15
            else "OK"
        ),
    }

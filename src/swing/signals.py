"""
Swing signal generation (non-TA).
Skill reference: .claude/skills/swing-signal-generation/SKILL.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def momentum_breakout_candidates(
    prices: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Identify stocks showing multi-day momentum with recent acceleration.
    Trigger: 5-day return > threshold, today up, volume above average.
    """
    lookback = config.get("momentum_lookback_days", 5)
    threshold = config.get("momentum_threshold_pct", 0.04)
    vol_surge_min = config.get("volume_surge_min", 1.2)

    candidates = []

    for symbol, df in prices.items():
        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)
        if idx < max(lookback, 21):
            continue

        close = df["close"]
        ret_n = (close.iloc[idx] / close.iloc[idx - lookback]) - 1
        ret_1 = (close.iloc[idx] / close.iloc[idx - 1]) - 1

        avg_vol = df["volume"].iloc[idx - 21 : idx].mean()
        vol_ratio = df["volume"].iloc[idx] / avg_vol if avg_vol > 0 else 0

        if ret_n > threshold and ret_1 > 0 and vol_ratio >= vol_surge_min:
            candidates.append(
                {
                    "symbol": symbol,
                    "signal_type": "momentum_breakout",
                    "direction": "LONG",
                    "signal_date": current_date,
                    "ret_n": ret_n,
                    "ret_1": ret_1,
                    "vol_ratio": vol_ratio,
                }
            )

    return candidates


def volatility_expansion_candidates(
    prices: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Identify stocks where short-term vol has expanded vs. recent baseline.
    Trigger: 5-day vol / 21-day vol > expansion_ratio.
    """
    expansion_ratio = config.get("vol_expansion_ratio", 1.5)
    short_window = config.get("vol_short_window", 5)
    long_window = config.get("vol_long_window", 21)

    candidates = []

    for symbol, df in prices.items():
        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)
        if idx < long_window + 5:
            continue

        log_ret = np.log(df["close"] / df["close"].shift(1))
        short_vol = log_ret.iloc[idx - short_window : idx].std()
        long_vol = log_ret.iloc[idx - long_window : idx].std()

        if long_vol == 0:
            continue

        vol_ratio_val = short_vol / long_vol
        recent_ret = (
            df["close"].iloc[idx] / df["close"].iloc[idx - short_window]
        ) - 1

        if vol_ratio_val > expansion_ratio:
            candidates.append(
                {
                    "symbol": symbol,
                    "signal_type": "vol_expansion",
                    "direction": "LONG" if recent_ret > 0 else "SHORT",
                    "signal_date": current_date,
                    "vol_ratio": vol_ratio_val,
                    "recent_ret": recent_ret,
                }
            )

    return candidates


def is_risk_on(
    index_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
) -> bool:
    """
    Simple risk-on/off classification from index behavior.
    Risk-ON if index 5d return > -1% and vol ratio < 1.5.
    """
    if current_date not in index_df.index:
        return False

    idx = index_df.index.get_loc(current_date)
    if idx < 21:
        return False

    log_ret = np.log(
        index_df["close"] / index_df["close"].shift(1)
    )

    index_ret_5d = (
        index_df["close"].iloc[idx] / index_df["close"].iloc[idx - 5]
    ) - 1
    vol_5d = log_ret.iloc[idx - 5 : idx].std()
    vol_21d = log_ret.iloc[idx - 21 : idx].std()
    vol_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.5

    risk_on_config = config.get("risk_on_gate", {})
    ret_threshold = risk_on_config.get("min_index_ret_5d", -0.01)
    vol_threshold = risk_on_config.get("max_vol_ratio", 1.5)

    return index_ret_5d > ret_threshold and vol_ratio < vol_threshold


def generate_swing_signals(
    prices: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Complete swing signal pipeline (without ML filter — that's added in the
    full pipeline when the model is available).
    1. Risk-on gate
    2. Generate candidates
    3. Deduplicate by symbol
    """
    if not is_risk_on(index_df, current_date, config):
        return []

    all_candidates = []

    if config.get("momentum_signal_enabled", True):
        all_candidates += momentum_breakout_candidates(
            prices, current_date, config
        )

    if config.get("vol_expansion_signal_enabled", True):
        all_candidates += volatility_expansion_candidates(
            prices, current_date, config
        )

    # Deduplicate by symbol (keep highest signal strength)
    seen = {}
    for cand in all_candidates:
        sym = cand["symbol"]
        if sym not in seen or cand.get("ret_n", 0) > seen[sym].get(
            "ret_n", 0
        ):
            seen[sym] = cand
    candidates = list(seen.values())

    return candidates

"""
Ensemble signal framework — combines multiple uncorrelated alpha signals
with confluence scoring and configurable weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.signals.signals import (
    momentum_breakout_candidates,
    volatility_expansion_candidates,
    is_risk_on,
)


# ------------------------------------------------------------------ #
#  New Signal Generators                                               #
# ------------------------------------------------------------------ #

def mean_reversion_candidates(
    prices: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Mean reversion: stocks that dropped significantly but show recovery signs.
    Trigger: N-day return < -threshold, RSI turning up, volume spike on recovery.
    """
    lookback = config.get("mr_lookback_days", 10)
    drop_threshold = config.get("mr_drop_threshold", -0.08)
    vol_surge = config.get("mr_volume_surge", 1.5)

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

        # RSI recovery check (simplified)
        delta = close.diff()
        gain = delta.clip(lower=0).iloc[idx - 14:idx].mean()
        loss = (-delta.clip(upper=0)).iloc[idx - 14:idx].mean()
        rsi = 100 - (100 / (1 + gain / loss)) if loss > 0 else 50

        avg_vol = df["volume"].iloc[idx - 21:idx].mean()
        vol_ratio = df["volume"].iloc[idx] / avg_vol if avg_vol > 0 else 0

        if ret_n < drop_threshold and ret_1 > 0 and rsi < 40 and vol_ratio >= vol_surge:
            candidates.append({
                "symbol": symbol,
                "signal_type": "mean_reversion",
                "direction": "LONG",
                "signal_date": current_date,
                "ret_n": ret_n,
                "ret_1": ret_1,
                "rsi": rsi,
                "vol_ratio": vol_ratio,
            })

    return candidates


def volume_anomaly_candidates(
    prices: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Volume anomaly: unusual volume with price consolidation.
    Potential accumulation/distribution before a move.
    """
    vol_threshold = config.get("va_volume_threshold", 2.0)
    price_range_max = config.get("va_price_range_pct", 0.02)

    candidates = []
    for symbol, df in prices.items():
        if current_date not in df.index:
            continue
        idx = df.index.get_loc(current_date)
        if idx < 21:
            continue

        avg_vol = df["volume"].iloc[idx - 20:idx].mean()
        vol_ratio = df["volume"].iloc[idx] / avg_vol if avg_vol > 0 else 0

        # Price consolidation: small range over last 5 days
        recent_close = df["close"].iloc[idx - 5:idx + 1]
        price_range = (recent_close.max() - recent_close.min()) / recent_close.mean()

        if vol_ratio >= vol_threshold and price_range <= price_range_max:
            today_ret = (df["close"].iloc[idx] - df["open"].iloc[idx]) / df["open"].iloc[idx]
            direction = "LONG" if today_ret >= 0 else "SHORT"

            candidates.append({
                "symbol": symbol,
                "signal_type": "volume_anomaly",
                "direction": direction,
                "signal_date": current_date,
                "vol_ratio": vol_ratio,
                "price_range": price_range,
            })

    return candidates


def gap_and_go_candidates(
    prices: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Gap-and-go: gap up on high volume that holds through the day.
    """
    min_gap = config.get("gg_min_gap_pct", 0.015)
    vol_surge = config.get("gg_volume_surge", 1.5)

    candidates = []
    for symbol, df in prices.items():
        if current_date not in df.index:
            continue
        idx = df.index.get_loc(current_date)
        if idx < 21:
            continue

        prev_close = df["close"].iloc[idx - 1]
        today_open = df["open"].iloc[idx]
        today_close = df["close"].iloc[idx]
        gap_pct = (today_open - prev_close) / prev_close

        avg_vol = df["volume"].iloc[idx - 20:idx].mean()
        vol_ratio = df["volume"].iloc[idx] / avg_vol if avg_vol > 0 else 0

        if gap_pct >= min_gap and today_close >= today_open and vol_ratio >= vol_surge:
            candidates.append({
                "symbol": symbol,
                "signal_type": "gap_and_go",
                "direction": "LONG",
                "signal_date": current_date,
                "gap_pct": gap_pct,
                "vol_ratio": vol_ratio,
            })

    return candidates


def relative_strength_candidates(
    prices: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Relative strength: stocks outperforming index with accelerating strength.
    """
    rs_window = config.get("rs_window", 21)
    rs_accel_window = config.get("rs_accel_window", 5)
    min_rs = config.get("rs_min_outperformance", 0.05)

    candidates = []

    if current_date not in index_df.index:
        return candidates
    idx_loc = index_df.index.get_loc(current_date)
    if idx_loc < rs_window:
        return candidates

    index_ret = (
        index_df["close"].iloc[idx_loc] / index_df["close"].iloc[idx_loc - rs_window]
    ) - 1
    index_ret_short = (
        index_df["close"].iloc[idx_loc] / index_df["close"].iloc[idx_loc - rs_accel_window]
    ) - 1

    for symbol, df in prices.items():
        if current_date not in df.index:
            continue
        idx = df.index.get_loc(current_date)
        if idx < rs_window:
            continue

        stock_ret = (df["close"].iloc[idx] / df["close"].iloc[idx - rs_window]) - 1
        stock_ret_short = (df["close"].iloc[idx] / df["close"].iloc[idx - rs_accel_window]) - 1

        rel_strength = stock_ret - index_ret
        rel_accel = (stock_ret_short - index_ret_short) - (rel_strength * rs_accel_window / rs_window)

        if rel_strength >= min_rs and rel_accel > 0:
            candidates.append({
                "symbol": symbol,
                "signal_type": "relative_strength",
                "direction": "LONG",
                "signal_date": current_date,
                "rel_strength": rel_strength,
                "rel_accel": rel_accel,
            })

    return candidates


# ------------------------------------------------------------------ #
#  Ensemble Signal Generator                                           #
# ------------------------------------------------------------------ #

DEFAULT_SIGNAL_WEIGHTS = {
    "momentum_breakout": 1.0,
    "vol_expansion": 0.8,
    "mean_reversion": 0.9,
    "volume_anomaly": 0.6,
    "gap_and_go": 0.7,
    "relative_strength": 0.85,
}


class EnsembleSignalGenerator:
    """
    Runs all signal generators, scores by confluence, and returns
    ranked candidates with ensemble scores.
    """

    def __init__(
        self,
        config: dict,
        signal_weights: Optional[Dict[str, float]] = None,
        index_df: Optional[pd.DataFrame] = None,
    ):
        self.config = config
        self.signal_weights = signal_weights or DEFAULT_SIGNAL_WEIGHTS
        self.index_df = index_df

    def generate(
        self,
        prices: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> List[dict]:
        """
        Generate ensemble signals from all sources.

        Returns:
            Sorted list of candidates with ensemble_score field, deduplicated by symbol.
        """
        all_signals: List[dict] = []

        generators = [
            ("momentum_breakout", lambda: momentum_breakout_candidates(prices, current_date, self.config)),
            ("vol_expansion", lambda: volatility_expansion_candidates(prices, current_date, self.config)),
            ("mean_reversion", lambda: mean_reversion_candidates(prices, current_date, self.config)),
            ("volume_anomaly", lambda: volume_anomaly_candidates(prices, current_date, self.config)),
            ("gap_and_go", lambda: gap_and_go_candidates(prices, current_date, self.config)),
        ]

        if self.index_df is not None:
            generators.append(
                ("relative_strength", lambda: relative_strength_candidates(
                    prices, self.index_df, current_date, self.config
                ))
            )

        for name, gen_func in generators:
            try:
                signals = gen_func()
                for s in signals:
                    s["source_signal"] = name
                all_signals.extend(signals)
            except Exception:
                continue

        return self._score_and_deduplicate(all_signals)

    def _score_and_deduplicate(self, signals: List[dict]) -> List[dict]:
        """Score candidates by signal confluence and deduplicate by symbol."""
        if not signals:
            return []

        by_symbol: Dict[str, List[dict]] = {}
        for s in signals:
            sym = s["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append(s)

        scored = []
        for symbol, sym_signals in by_symbol.items():
            signal_types = set()
            total_weight = 0.0
            for s in sym_signals:
                st = s.get("source_signal", s.get("signal_type", "unknown"))
                if st not in signal_types:
                    signal_types.add(st)
                    total_weight += self.signal_weights.get(st, 0.5)

            n_signals = len(signal_types)
            max_possible_weight = sum(self.signal_weights.values())
            ensemble_score = total_weight / max_possible_weight if max_possible_weight > 0 else 0

            best = max(sym_signals, key=lambda s: s.get("ret_n", s.get("rel_strength", 0)))
            best["ensemble_score"] = round(ensemble_score, 4)
            best["n_confirming_signals"] = n_signals
            best["confirming_types"] = list(signal_types)
            scored.append(best)

        scored.sort(key=lambda s: s["ensemble_score"], reverse=True)
        return scored

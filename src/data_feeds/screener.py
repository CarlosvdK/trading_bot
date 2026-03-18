"""
Dynamic Universe Screener — finds tradeable symbols based on
liquidity, price, and data quality filters.

Replaces the static universe.csv for a self-sufficient system.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default minimum filters for swing trading
DEFAULT_FILTERS = {
    "min_price": 10.0,          # Avoid penny stocks
    "max_price": 5000.0,        # Avoid BRK.A-type outliers
    "min_avg_volume": 500_000,  # Need liquidity for fills
    "min_avg_dollar_vol": 5_000_000,  # $5M daily turnover minimum
    "min_history_days": 252,    # 1 year minimum history
    "max_missing_pct": 0.03,    # Max 3% missing bars
    "min_vol": 0.10,            # Skip ultra-low-vol stocks (dead money)
    "max_vol": 1.50,            # Skip hyper-volatile names
    "max_spread_est_pct": 0.5,  # Estimated spread > 0.5% = illiquid
}


def screen_universe(
    price_data: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp = None,
    filters: dict = None,
) -> List[dict]:
    """
    Screen all available symbols and return those meeting quality/liquidity bars.

    Returns list of dicts with symbol + screening metrics.
    Sorted by dollar volume (most liquid first).
    """
    filters = {**DEFAULT_FILTERS, **(filters or {})}
    if as_of_date is None:
        as_of_date = pd.Timestamp.today()

    candidates = []

    for symbol, df in price_data.items():
        result = _evaluate_symbol(symbol, df, as_of_date, filters)
        if result["passed"]:
            candidates.append(result)

    # Sort by dollar volume (best liquidity first)
    candidates.sort(key=lambda x: -x["avg_dollar_vol"])

    logger.info(
        f"Screener: {len(candidates)}/{len(price_data)} symbols passed "
        f"(as of {as_of_date.date()})"
    )
    return candidates


def _evaluate_symbol(
    symbol: str,
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    filters: dict,
) -> dict:
    """Evaluate a single symbol against screening filters."""
    result = {
        "symbol": symbol,
        "passed": False,
        "rejection_reason": None,
    }

    # Trim to data up to as_of_date
    df = df[df.index <= as_of_date]

    # History length
    if len(df) < filters["min_history_days"]:
        result["rejection_reason"] = f"history={len(df)}d < {filters['min_history_days']}d"
        return result

    recent = df.tail(63)  # Last quarter for screening metrics

    # Price filter
    last_price = recent["close"].iloc[-1]
    result["last_price"] = last_price
    if last_price < filters["min_price"]:
        result["rejection_reason"] = f"price={last_price:.2f} < {filters['min_price']}"
        return result
    if last_price > filters["max_price"]:
        result["rejection_reason"] = f"price={last_price:.2f} > {filters['max_price']}"
        return result

    # Volume filter
    avg_volume = recent["volume"].mean()
    avg_dollar_vol = (recent["close"] * recent["volume"]).mean()
    result["avg_volume"] = avg_volume
    result["avg_dollar_vol"] = avg_dollar_vol

    if avg_volume < filters["min_avg_volume"]:
        result["rejection_reason"] = f"volume={avg_volume:.0f} < {filters['min_avg_volume']}"
        return result
    if avg_dollar_vol < filters["min_avg_dollar_vol"]:
        result["rejection_reason"] = f"dollar_vol=${avg_dollar_vol/1e6:.1f}M < ${filters['min_avg_dollar_vol']/1e6:.1f}M"
        return result

    # Missing data
    total_expected = len(pd.bdate_range(df.index[0], df.index[-1]))
    missing_pct = 1 - len(df) / max(total_expected, 1)
    result["missing_pct"] = missing_pct
    if missing_pct > filters["max_missing_pct"]:
        result["rejection_reason"] = f"missing={missing_pct:.1%} > {filters['max_missing_pct']:.1%}"
        return result

    # Volatility filter
    log_ret = np.log(recent["close"] / recent["close"].shift(1)).dropna()
    ann_vol = log_ret.std() * np.sqrt(252) if len(log_ret) > 5 else 0
    result["ann_vol"] = ann_vol

    if ann_vol < filters["min_vol"]:
        result["rejection_reason"] = f"vol={ann_vol:.2f} < {filters['min_vol']}"
        return result
    if ann_vol > filters["max_vol"]:
        result["rejection_reason"] = f"vol={ann_vol:.2f} > {filters['max_vol']}"
        return result

    # Estimated spread (proxy: high-low range relative to close)
    spread_est = ((recent["high"] - recent["low"]) / recent["close"]).mean()
    result["spread_est_pct"] = spread_est * 100
    if spread_est * 100 > filters["max_spread_est_pct"]:
        result["rejection_reason"] = f"spread_est={spread_est*100:.2f}% > {filters['max_spread_est_pct']}%"
        return result

    # All filters passed
    result["passed"] = True
    result["history_days"] = len(df)
    return result


def expand_universe(
    current_symbols: List[str],
    all_price_data: Dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp = None,
    max_symbols: int = 50,
    filters: dict = None,
) -> List[str]:
    """
    Dynamically expand the trading universe by screening new candidates.
    Keeps current symbols + adds the best new ones up to max_symbols.
    """
    all_candidates = screen_universe(all_price_data, as_of_date, filters)
    candidate_symbols = [c["symbol"] for c in all_candidates]

    # Keep current symbols that still pass
    kept = [s for s in current_symbols if s in candidate_symbols]
    # Add new ones
    new = [s for s in candidate_symbols if s not in current_symbols]

    expanded = kept + new[:max_symbols - len(kept)]
    added = [s for s in expanded if s not in current_symbols]
    removed = [s for s in current_symbols if s not in expanded]

    if added or removed:
        logger.info(
            f"Universe update: {len(current_symbols)} → {len(expanded)} symbols | "
            f"+{len(added)} added, -{len(removed)} removed"
        )

    return expanded

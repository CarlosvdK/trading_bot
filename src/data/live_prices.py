"""
Live Price Fetcher — real-time and recent quotes for trading.

Uses yfinance for:
  1. Current quotes (last price, bid/ask, volume)
  2. Intraday bars (1m, 5m, 15m)
  3. Recent daily bars (last N days — gap-fill for today)

This bridges the gap between historical CSVs and live trading.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_live_quotes(symbols: List[str]) -> Dict[str, dict]:
    """
    Get current quotes for a list of symbols.

    Returns dict of symbol -> {
        price, bid, ask, volume, market_cap, change_pct, timestamp
    }
    """
    import yfinance as yf

    quotes = {}
    # Batch via Tickers object
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                info = tickers.tickers[sym].fast_info
                quotes[sym] = {
                    "price": float(info.get("lastPrice", 0) or info.get("previousClose", 0)),
                    "previous_close": float(info.get("previousClose", 0)),
                    "market_cap": float(info.get("marketCap", 0)),
                    "timestamp": datetime.now(),
                }
            except Exception as e:
                logger.debug(f"Quote failed for {sym}: {e}")
    except Exception as e:
        logger.warning(f"Batch quote fetch failed: {e}")

    return quotes


def get_latest_bars(
    symbols: List[str],
    period: str = "5d",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Get recent OHLCV bars for symbols. Use this to update historical
    data with the latest bars before running signals.

    Args:
        symbols: List of ticker symbols
        period: How far back ("1d", "5d", "1mo")
        interval: Bar size ("1m", "5m", "15m", "1d")

    Returns:
        Dict of symbol -> DataFrame with columns: open, high, low, close, volume
    """
    import yfinance as yf

    result = {}
    batch_size = 50

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )

            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = data.copy()
                        if isinstance(sym_df.columns, pd.MultiIndex):
                            sym_df = sym_df.droplevel("Ticker", axis=1)
                    else:
                        sym_df = data[sym]

                    sym_df = sym_df.dropna(how="all")
                    if not sym_df.empty:
                        # Normalize column names to lowercase
                        sym_df.columns = [c.lower() for c in sym_df.columns]
                        # Strip timezone if present
                        if sym_df.index.tz is not None:
                            sym_df.index = sym_df.index.tz_localize(None)
                        result[sym] = sym_df
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Batch download failed: {e}")

    logger.info(f"Fetched latest bars for {len(result)}/{len(symbols)} symbols")
    return result


def update_price_data(
    price_data: Dict[str, pd.DataFrame],
    symbols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Update existing price_data dict with the latest bars from yfinance.
    Appends new rows that don't already exist (by date).

    This is what you call before running daily signals to make sure
    you have today's data.
    """
    if symbols is None:
        symbols = list(price_data.keys())

    latest = get_latest_bars(symbols, period="5d", interval="1d")

    updated_count = 0
    for sym, new_df in latest.items():
        if sym in price_data:
            existing = price_data[sym]
            # Only append rows with dates not already in existing
            new_dates = new_df.index.difference(existing.index)
            if len(new_dates) > 0:
                new_rows = new_df.loc[new_dates]
                price_data[sym] = pd.concat([existing, new_rows]).sort_index()
                updated_count += 1
        else:
            price_data[sym] = new_df
            updated_count += 1

    if updated_count:
        logger.info(f"Updated {updated_count} symbols with latest data")

    return price_data


def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    try:
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        # ET is UTC-5 (EST) or UTC-4 (EDT)
        # Approximate: use -4 during Mar-Nov, -5 during Nov-Mar
        month = now_utc.month
        et_offset = -4 if 3 <= month <= 11 else -5
        now_et = now_utc + timedelta(hours=et_offset)

        # Weekday check (0=Mon, 6=Sun)
        if now_et.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now_et.replace(hour=9, minute=30, second=0)
        market_close = now_et.replace(hour=16, minute=0, second=0)

        return market_open <= now_et <= market_close
    except Exception:
        return False


def time_until_market_open() -> Optional[timedelta]:
    """Returns timedelta until next market open, or None if market is open."""
    if is_market_open():
        return None

    from datetime import timezone
    now_utc = datetime.now(timezone.utc)
    month = now_utc.month
    et_offset = -4 if 3 <= month <= 11 else -5
    now_et = now_utc + timedelta(hours=et_offset)

    # Find next weekday 9:30 AM ET
    target = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    if now_et >= target:
        # Already past 9:30 today, go to next day
        target += timedelta(days=1)

    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)

    return target - now_et

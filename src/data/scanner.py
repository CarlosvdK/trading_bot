"""
Dynamic Stock Scanner — discovers new trading opportunities.

Uses yfinance to scan for stocks showing unusual activity (volume surges,
big moves, sector momentum) and adds them to the trading universe.

This is how the bot finds opportunities beyond the fixed S&P 500 base.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Scan pools — groups of tickers to check for opportunities
# These are well-known liquid stocks/ETFs outside the core S&P 500 base
SCAN_POOLS = {
    "mid_cap_growth": [
        "DKNG", "COIN", "MARA", "RIOT", "SOFI", "HOOD", "AFRM", "UPST",
        "SHOP", "SQ", "SNAP", "PINS", "U", "ROKU", "ZM", "DOCU",
        "BILL", "HUBS", "DDOG", "NET", "ZS", "OKTA", "MDB", "SNOW",
        "PATH", "CFLT", "ESTC", "TEAM", "GTLB", "MNDY",
    ],
    "biotech": [
        "MRNA", "BNTX", "SGEN", "EXAS", "RARE", "IONS", "SRPT",
        "NBIX", "PCVX", "ALNY", "BMRN", "INCY", "HALO", "JAZZ",
    ],
    "china_adr": [
        "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "TME",
        "BILI", "IQ", "ZTO", "VNET",
    ],
    "crypto_adjacent": [
        "MSTR", "COIN", "MARA", "RIOT", "HUT", "BITF", "CLSK",
    ],
    "sector_etfs_extended": [
        "QQQ", "IWM", "DIA", "VTI", "ARKK", "ARKG", "ARKF",
        "SMH", "XBI", "KRE", "XHB", "XRT", "XOP", "GDX", "GDXJ",
        "TLT", "HYG", "LQD", "EEM", "EFA", "FXI", "EWJ", "EWZ",
        "GLD", "SLV", "USO", "UNG",
    ],
    "recent_ipos": [
        "ARM", "BIRK", "CART", "CAVA", "KVYO", "VRT", "ONON",
        "DUOL", "IOT", "TOST",
    ],
}


def get_scan_pool(pools: List[str] = None) -> List[str]:
    """Get deduplicated list of tickers to scan."""
    if pools is None:
        pools = list(SCAN_POOLS.keys())

    tickers = set()
    for pool_name in pools:
        if pool_name in SCAN_POOLS:
            tickers.update(SCAN_POOLS[pool_name])

    return sorted(tickers)


def scan_for_opportunities(
    scan_tickers: List[str],
    existing_universe: List[str],
    lookback_days: int = 30,
    min_avg_volume: int = 1_000_000,
    min_avg_dollar_vol: float = 10_000_000,
    min_price: float = 5.0,
    max_price: float = 5000.0,
    volume_surge_threshold: float = 2.0,
    momentum_threshold: float = 0.05,
    max_new_symbols: int = 20,
) -> List[dict]:
    """
    Scan a pool of tickers for ones showing unusual activity.

    Returns list of dicts with:
        symbol, reason, avg_volume, avg_dollar_vol, momentum_5d, volume_surge

    Reasons a stock gets flagged:
        - "volume_surge": recent volume >> average volume
        - "momentum": strong recent price move
        - "breakout": new 20-day high with volume
    """
    import yfinance as yf

    # Filter out stocks already in universe
    existing = set(existing_universe)
    to_scan = [t for t in scan_tickers if t not in existing]

    if not to_scan:
        logger.info("Scanner: all scan pool tickers already in universe")
        return []

    logger.info(f"Scanner: checking {len(to_scan)} tickers for opportunities...")

    # Bulk download recent data
    try:
        data = yf.download(
            to_scan,
            period=f"{lookback_days + 10}d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        logger.error(f"Scanner download failed: {e}")
        return []

    opportunities = []

    for sym in to_scan:
        try:
            if len(to_scan) == 1:
                sym_df = data.copy()
                if isinstance(sym_df.columns, pd.MultiIndex):
                    sym_df = sym_df.droplevel("Ticker", axis=1)
            else:
                sym_df = data[sym]

            sym_df = sym_df.dropna(how="all")
            if sym_df.empty or len(sym_df) < 10:
                continue

            close = sym_df["Close"]
            volume = sym_df["Volume"]
            high = sym_df["High"]

            last_price = close.iloc[-1]
            if last_price < min_price or last_price > max_price:
                continue

            avg_vol = volume.tail(20).mean()
            if avg_vol < min_avg_volume:
                continue

            avg_dollar_vol = (close * volume).tail(20).mean()
            if avg_dollar_vol < min_avg_dollar_vol:
                continue

            # Volume surge: last 5 days vs 20-day average
            recent_vol = volume.tail(5).mean()
            vol_surge = recent_vol / max(avg_vol, 1)

            # Momentum: 5-day return
            mom_5d = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0

            # Breakout: near 20-day high with above-average volume
            high_20d = high.tail(20).max()
            near_high = close.iloc[-1] >= high_20d * 0.98

            # Determine if this is an opportunity
            reasons = []
            if vol_surge >= volume_surge_threshold:
                reasons.append("volume_surge")
            if abs(mom_5d) >= momentum_threshold:
                reasons.append("momentum")
            if near_high and vol_surge >= 1.3:
                reasons.append("breakout")

            if reasons:
                opportunities.append({
                    "symbol": sym,
                    "reason": "+".join(reasons),
                    "last_price": round(float(last_price), 2),
                    "avg_volume": int(avg_vol),
                    "avg_dollar_vol": round(float(avg_dollar_vol), 0),
                    "momentum_5d": round(float(mom_5d), 4),
                    "volume_surge": round(float(vol_surge), 2),
                    "near_20d_high": near_high,
                })

        except Exception as e:
            logger.debug(f"Scanner: {sym} failed: {e}")
            continue

    # Sort by volume surge (strongest signal first), limit results
    opportunities.sort(key=lambda x: x["volume_surge"], reverse=True)
    opportunities = opportunities[:max_new_symbols]

    if opportunities:
        logger.info(
            f"Scanner: found {len(opportunities)} opportunities: "
            f"{[o['symbol'] for o in opportunities]}"
        )
    else:
        logger.info("Scanner: no new opportunities found")

    return opportunities


def add_to_universe(
    opportunities: List[dict],
    universe_path: str,
    data_dir: str,
    start_date: str = "2015-01-01",
) -> List[str]:
    """
    Download data for new opportunities and add them to universe.csv.
    Returns list of symbols successfully added.
    """
    from src.data.api_providers import YFinanceDownloader

    if not opportunities:
        return []

    universe_path = Path(universe_path)
    if universe_path.exists():
        universe_df = pd.read_csv(universe_path)
        existing = set(universe_df["symbol"].tolist())
    else:
        universe_df = pd.DataFrame(
            columns=["symbol", "active_from", "active_to", "notes", "sector_etf"]
        )
        existing = set()

    downloader = YFinanceDownloader(data_dir)
    added = []

    for opp in opportunities:
        sym = opp["symbol"]
        if sym in existing:
            continue

        # Download historical data
        path = downloader.download_symbol(sym, start_date)
        if path is None:
            continue

        # Add to universe
        new_row = pd.DataFrame([{
            "symbol": sym,
            "active_from": "2010-01-01",
            "active_to": "",
            "notes": f"Scanner: {opp['reason']}",
            "sector_etf": "",
        }])
        universe_df = pd.concat([universe_df, new_row], ignore_index=True)
        added.append(sym)
        logger.info(f"Scanner: added {sym} to universe ({opp['reason']})")

    if added:
        universe_df.to_csv(universe_path, index=False)
        logger.info(f"Scanner: {len(added)} new symbols added to universe")

    return added


def run_full_scan(
    universe_path: str,
    data_dir: str,
    pools: List[str] = None,
    max_new: int = 20,
) -> List[str]:
    """
    Complete scan pipeline: scan → filter → download → add to universe.
    Returns list of newly added symbols.
    """
    # Load current universe
    if Path(universe_path).exists():
        uni = pd.read_csv(universe_path)
        existing = uni["symbol"].tolist()
    else:
        existing = []

    # Scan
    scan_tickers = get_scan_pool(pools)
    opportunities = scan_for_opportunities(
        scan_tickers, existing, max_new_symbols=max_new
    )

    # Add winners
    added = add_to_universe(opportunities, universe_path, data_dir)
    return added

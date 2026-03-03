#!/usr/bin/env python3
"""
Generate a full trading universe from S&P 500 constituents + sector ETFs.

Fetches the current S&P 500 list from Wikipedia, adds sector ETFs for
sector-relative features, and writes data/universe.csv.

Usage:
    python scripts/generate_universe.py
    python scripts/generate_universe.py --max-symbols 200
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# SPDR Select Sector ETFs
SECTOR_ETFS = {
    "XLK": ("Information Technology", "1998-12-22"),
    "XLF": ("Financials", "1998-12-22"),
    "XLV": ("Health Care", "1998-12-22"),
    "XLE": ("Energy", "1998-12-22"),
    "XLI": ("Industrials", "1998-12-22"),
    "XLY": ("Consumer Discretionary", "1998-12-22"),
    "XLP": ("Consumer Staples", "1998-12-22"),
    "XLU": ("Utilities", "1998-12-22"),
    "XLRE": ("Real Estate", "2015-10-08"),
    "XLC": ("Communication Services", "2018-06-18"),
    "XLB": ("Materials", "1998-12-22"),
}

# GICS sector name → sector ETF symbol
GICS_TO_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Materials": "XLB",
}


def fetch_sp500_tickers() -> pd.DataFrame:
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    df = df.rename(columns={"Symbol": "symbol", "GICS Sector": "sector"})
    # Fix tickers with dots (BRK.B → BRK-B for yfinance compatibility)
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    return df[["symbol", "sector"]]


def generate_universe(max_symbols: int = 300) -> pd.DataFrame:
    """Build full universe DataFrame in universe.csv format."""
    logger.info("Fetching S&P 500 constituents from Wikipedia...")
    sp500 = fetch_sp500_tickers()
    logger.info(f"Found {len(sp500)} S&P 500 constituents")

    # Exclude known problematic tickers
    exclude = {"BRK-B", "BF-B"}
    sp500 = sp500[~sp500["symbol"].isin(exclude)]

    rows = []

    # SPY as reference index
    rows.append({
        "symbol": "SPY",
        "active_from": "1993-01-29",
        "active_to": "",
        "notes": "Reference index ETF",
        "sector_etf": "",
    })

    # Sector ETFs
    for etf, (sector_name, inception) in SECTOR_ETFS.items():
        rows.append({
            "symbol": etf,
            "active_from": inception,
            "active_to": "",
            "notes": f"Sector ETF: {sector_name}",
            "sector_etf": "",
        })

    # S&P 500 stocks (up to budget)
    budget = max_symbols - len(rows)
    for _, row in sp500.head(budget).iterrows():
        sector_etf = GICS_TO_ETF.get(row["sector"], "")
        rows.append({
            "symbol": row["symbol"],
            "active_from": "2010-01-01",
            "active_to": "",
            "notes": f"S&P 500 - {row['sector']}",
            "sector_etf": sector_etf,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate trading universe from S&P 500"
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=300,
        help="Maximum number of symbols (default: 300)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "universe.csv"),
        help="Output path for universe.csv",
    )
    args = parser.parse_args()

    df = generate_universe(args.max_symbols)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_etfs = sum(1 for n in df["notes"] if "Sector ETF" in str(n))
    n_stocks = sum(1 for n in df["notes"] if "S&P 500" in str(n))

    logger.info(f"Wrote {len(df)} symbols to {args.output}")
    logger.info(f"  Index: 1 (SPY)")
    logger.info(f"  Sector ETFs: {n_etfs}")
    logger.info(f"  Stocks: {n_stocks}")


if __name__ == "__main__":
    main()

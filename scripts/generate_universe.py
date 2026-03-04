#!/usr/bin/env python3
"""
Generate a full trading universe — hardcoded S&P 500 + sector ETFs.

No Wikipedia scraping. No external dependencies. Just a reliable,
curated list of the most liquid US stocks across all sectors.

Usage:
    python scripts/generate_universe.py
    python scripts/generate_universe.py --max-symbols 200
    python scripts/generate_universe.py --output data/universe.csv
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

# ---------------------------------------------------------------------------
# Hardcoded S&P 500 universe — curated, no web scraping needed.
# Format: (symbol, GICS sector)
# This covers ~300 of the most liquid S&P 500 names across all 11 sectors.
# ---------------------------------------------------------------------------

SP500_STOCKS = [
    # Information Technology (75 names)
    ("AAPL", "Information Technology"),
    ("MSFT", "Information Technology"),
    ("NVDA", "Information Technology"),
    ("AVGO", "Information Technology"),
    ("ORCL", "Information Technology"),
    ("CRM", "Information Technology"),
    ("CSCO", "Information Technology"),
    ("AMD", "Information Technology"),
    ("ACN", "Information Technology"),
    ("ADBE", "Information Technology"),
    ("IBM", "Information Technology"),
    ("TXN", "Information Technology"),
    ("QCOM", "Information Technology"),
    ("INTU", "Information Technology"),
    ("AMAT", "Information Technology"),
    ("NOW", "Information Technology"),
    ("PANW", "Information Technology"),
    ("MU", "Information Technology"),
    ("LRCX", "Information Technology"),
    ("ADI", "Information Technology"),
    ("KLAC", "Information Technology"),
    ("SNPS", "Information Technology"),
    ("CDNS", "Information Technology"),
    ("CRWD", "Information Technology"),
    ("MSI", "Information Technology"),
    ("FTNT", "Information Technology"),
    ("MCHP", "Information Technology"),
    ("APH", "Information Technology"),
    ("ROP", "Information Technology"),
    ("NXPI", "Information Technology"),
    ("ADSK", "Information Technology"),
    ("TEL", "Information Technology"),
    ("ON", "Information Technology"),
    ("CTSH", "Information Technology"),
    ("IT", "Information Technology"),
    ("MPWR", "Information Technology"),
    ("PLTR", "Information Technology"),
    ("CDW", "Information Technology"),
    ("FSLR", "Information Technology"),
    ("HPQ", "Information Technology"),
    ("GLW", "Information Technology"),
    ("NTAP", "Information Technology"),
    ("TYL", "Information Technology"),
    ("ZBRA", "Information Technology"),
    ("SWKS", "Information Technology"),
    ("EPAM", "Information Technology"),
    ("TRMB", "Information Technology"),
    ("PTC", "Information Technology"),
    ("VRSN", "Information Technology"),
    ("WDAY", "Information Technology"),
    # Financials (40 names)
    ("JPM", "Financials"),
    ("V", "Financials"),
    ("MA", "Financials"),
    ("BAC", "Financials"),
    ("WFC", "Financials"),
    ("GS", "Financials"),
    ("MS", "Financials"),
    ("SPGI", "Financials"),
    ("BLK", "Financials"),
    ("C", "Financials"),
    ("AXP", "Financials"),
    ("SCHW", "Financials"),
    ("MMC", "Financials"),
    ("CB", "Financials"),
    ("PGR", "Financials"),
    ("ICE", "Financials"),
    ("AON", "Financials"),
    ("CME", "Financials"),
    ("MCO", "Financials"),
    ("USB", "Financials"),
    ("PNC", "Financials"),
    ("TFC", "Financials"),
    ("AJG", "Financials"),
    ("AFL", "Financials"),
    ("MET", "Financials"),
    ("TRV", "Financials"),
    ("AIG", "Financials"),
    ("ALL", "Financials"),
    ("BK", "Financials"),
    ("STT", "Financials"),
    ("FITB", "Financials"),
    ("COF", "Financials"),
    ("SYF", "Financials"),
    ("HBAN", "Financials"),
    ("RF", "Financials"),
    ("CFG", "Financials"),
    ("KEY", "Financials"),
    ("CINF", "Financials"),
    ("L", "Financials"),
    ("NTRS", "Financials"),
    # Health Care (35 names)
    ("UNH", "Health Care"),
    ("JNJ", "Health Care"),
    ("LLY", "Health Care"),
    ("ABBV", "Health Care"),
    ("MRK", "Health Care"),
    ("TMO", "Health Care"),
    ("ABT", "Health Care"),
    ("PFE", "Health Care"),
    ("DHR", "Health Care"),
    ("AMGN", "Health Care"),
    ("BMY", "Health Care"),
    ("ELV", "Health Care"),
    ("MDT", "Health Care"),
    ("ISRG", "Health Care"),
    ("GILD", "Health Care"),
    ("SYK", "Health Care"),
    ("CI", "Health Care"),
    ("VRTX", "Health Care"),
    ("REGN", "Health Care"),
    ("BSX", "Health Care"),
    ("ZTS", "Health Care"),
    ("BDX", "Health Care"),
    ("HUM", "Health Care"),
    ("EW", "Health Care"),
    ("HCA", "Health Care"),
    ("IQV", "Health Care"),
    ("IDXX", "Health Care"),
    ("A", "Health Care"),
    ("DXCM", "Health Care"),
    ("MTD", "Health Care"),
    ("BAX", "Health Care"),
    ("BIIB", "Health Care"),
    ("HOLX", "Health Care"),
    ("ALGN", "Health Care"),
    ("ZBH", "Health Care"),
    # Consumer Discretionary (25 names)
    ("AMZN", "Consumer Discretionary"),
    ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"),
    ("LOW", "Consumer Discretionary"),
    ("NKE", "Consumer Discretionary"),
    ("SBUX", "Consumer Discretionary"),
    ("TJX", "Consumer Discretionary"),
    ("BKNG", "Consumer Discretionary"),
    ("CMG", "Consumer Discretionary"),
    ("ORLY", "Consumer Discretionary"),
    ("AZO", "Consumer Discretionary"),
    ("MAR", "Consumer Discretionary"),
    ("ROST", "Consumer Discretionary"),
    ("DHI", "Consumer Discretionary"),
    ("LEN", "Consumer Discretionary"),
    ("GM", "Consumer Discretionary"),
    ("F", "Consumer Discretionary"),
    ("YUM", "Consumer Discretionary"),
    ("DRI", "Consumer Discretionary"),
    ("EBAY", "Consumer Discretionary"),
    ("POOL", "Consumer Discretionary"),
    ("BBY", "Consumer Discretionary"),
    ("APTV", "Consumer Discretionary"),
    ("GRMN", "Consumer Discretionary"),
    # Industrials (30 names)
    ("CAT", "Industrials"),
    ("GE", "Industrials"),
    ("RTX", "Industrials"),
    ("HON", "Industrials"),
    ("UNP", "Industrials"),
    ("BA", "Industrials"),
    ("DE", "Industrials"),
    ("UPS", "Industrials"),
    ("LMT", "Industrials"),
    ("ADP", "Industrials"),
    ("MMM", "Industrials"),
    ("GD", "Industrials"),
    ("NOC", "Industrials"),
    ("CSX", "Industrials"),
    ("NSC", "Industrials"),
    ("ITW", "Industrials"),
    ("EMR", "Industrials"),
    ("WM", "Industrials"),
    ("ETN", "Industrials"),
    ("FDX", "Industrials"),
    ("TT", "Industrials"),
    ("PAYX", "Industrials"),
    ("FAST", "Industrials"),
    ("VRSK", "Industrials"),
    ("CTAS", "Industrials"),
    ("RSG", "Industrials"),
    ("PWR", "Industrials"),
    ("SWK", "Industrials"),
    ("ROK", "Industrials"),
    ("DAL", "Industrials"),
    # Consumer Staples (20 names)
    ("PG", "Consumer Staples"),
    ("KO", "Consumer Staples"),
    ("PEP", "Consumer Staples"),
    ("COST", "Consumer Staples"),
    ("WMT", "Consumer Staples"),
    ("PM", "Consumer Staples"),
    ("MO", "Consumer Staples"),
    ("MDLZ", "Consumer Staples"),
    ("CL", "Consumer Staples"),
    ("ADM", "Consumer Staples"),
    ("GIS", "Consumer Staples"),
    ("KMB", "Consumer Staples"),
    ("SYY", "Consumer Staples"),
    ("STZ", "Consumer Staples"),
    ("KHC", "Consumer Staples"),
    ("HSY", "Consumer Staples"),
    ("MKC", "Consumer Staples"),
    ("KDP", "Consumer Staples"),
    ("TSN", "Consumer Staples"),
    ("CAG", "Consumer Staples"),
    # Energy (15 names)
    ("XOM", "Energy"),
    ("CVX", "Energy"),
    ("COP", "Energy"),
    ("EOG", "Energy"),
    ("SLB", "Energy"),
    ("MPC", "Energy"),
    ("PSX", "Energy"),
    ("VLO", "Energy"),
    ("OXY", "Energy"),
    ("WMB", "Energy"),
    ("KMI", "Energy"),
    ("HAL", "Energy"),
    ("DVN", "Energy"),
    ("OKE", "Energy"),
    ("FANG", "Energy"),
    # Communication Services (15 names)
    ("GOOGL", "Communication Services"),
    ("META", "Communication Services"),
    ("NFLX", "Communication Services"),
    ("DIS", "Communication Services"),
    ("CMCSA", "Communication Services"),
    ("T", "Communication Services"),
    ("VZ", "Communication Services"),
    ("TMUS", "Communication Services"),
    ("CHTR", "Communication Services"),
    ("RBLX", "Communication Services"),
    ("EA", "Communication Services"),
    ("TTWO", "Communication Services"),
    ("WBD", "Communication Services"),
    ("OMC", "Communication Services"),
    ("LYV", "Communication Services"),
    # Utilities (12 names)
    ("NEE", "Utilities"),
    ("DUK", "Utilities"),
    ("SO", "Utilities"),
    ("D", "Utilities"),
    ("AEP", "Utilities"),
    ("SRE", "Utilities"),
    ("EXC", "Utilities"),
    ("XEL", "Utilities"),
    ("ED", "Utilities"),
    ("WEC", "Utilities"),
    ("ES", "Utilities"),
    ("AWK", "Utilities"),
    # Real Estate (12 names)
    ("PLD", "Real Estate"),
    ("AMT", "Real Estate"),
    ("CCI", "Real Estate"),
    ("EQIX", "Real Estate"),
    ("PSA", "Real Estate"),
    ("O", "Real Estate"),
    ("DLR", "Real Estate"),
    ("WELL", "Real Estate"),
    ("SPG", "Real Estate"),
    ("AVB", "Real Estate"),
    ("EQR", "Real Estate"),
    ("ARE", "Real Estate"),
    # Materials (12 names)
    ("LIN", "Materials"),
    ("APD", "Materials"),
    ("SHW", "Materials"),
    ("FCX", "Materials"),
    ("ECL", "Materials"),
    ("NEM", "Materials"),
    ("DOW", "Materials"),
    ("NUE", "Materials"),
    ("VMC", "Materials"),
    ("MLM", "Materials"),
    ("PPG", "Materials"),
    ("DD", "Materials"),
]

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


def generate_universe(max_symbols: int = 300) -> pd.DataFrame:
    """Build full universe DataFrame in universe.csv format."""
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
    for symbol, sector in SP500_STOCKS[:budget]:
        sector_etf = GICS_TO_ETF.get(sector, "")
        rows.append({
            "symbol": symbol,
            "active_from": "2010-01-01",
            "active_to": "",
            "notes": f"S&P 500 - {sector}",
            "sector_etf": sector_etf,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate trading universe (hardcoded S&P 500, no web scraping)"
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

    # Sector breakdown
    sectors = {}
    for _, row in df.iterrows():
        if "S&P 500" in str(row["notes"]):
            sector = str(row["notes"]).replace("S&P 500 - ", "")
            sectors[sector] = sectors.get(sector, 0) + 1

    logger.info(f"Wrote {len(df)} symbols to {args.output}")
    logger.info(f"  Index: 1 (SPY)")
    logger.info(f"  Sector ETFs: {n_etfs}")
    logger.info(f"  Stocks: {n_stocks}")
    logger.info(f"  Sector breakdown:")
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        logger.info(f"    {sector}: {count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download OHLCV data for the trading universe.
Uses yfinance (primary), Polygon, Stooq, Alpha Vantage (fallbacks).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --symbols AAPL MSFT GOOGL
    python scripts/download_data.py --start 2010-01-01 --source yfinance
    python scripts/download_data.py --retry-failed
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_feeds.api_providers import (
    DataDownloader,
    PolygonDownloader,
    AlphaVantageDownloader,
    StooqDownloader,
    YFinanceDownloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_UNIVERSE = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "data" / "ohlcv")


def main():
    parser = argparse.ArgumentParser(
        description="Download OHLCV data for trading universe"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to download (default: read from data/universe.csv)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (default: today)",
    )
    parser.add_argument(
        "--source",
        choices=["polygon", "alphavantage", "stooq", "yfinance", "auto"],
        default="auto",
        help="Data source (default: auto = yfinance first, then fallback chain)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only download symbols that have no CSV file yet",
    )
    args = parser.parse_args()

    # Resolve symbols
    if args.symbols:
        symbols = args.symbols
    else:
        universe_path = (
            Path(__file__).resolve().parent.parent / "data" / "universe.csv"
        )
        if universe_path.exists():
            import pandas as pd
            uni = pd.read_csv(universe_path)
            symbols = uni["symbol"].tolist()
            logger.info(f"Loaded {len(symbols)} symbols from {universe_path}")
        else:
            symbols = DEFAULT_UNIVERSE
            logger.info(f"No universe file found, using defaults: {symbols}")

    # Filter to only missing symbols if --retry-failed
    if args.retry_failed:
        existing = {p.stem for p in Path(args.output_dir).glob("*.csv")}
        before = len(symbols)
        symbols = [s for s in symbols if s not in existing]
        logger.info(
            f"Retry mode: {len(symbols)} symbols need downloading "
            f"({before - len(symbols)} already exist)"
        )
        if not symbols:
            logger.info("All symbols already downloaded!")
            return

    logger.info(f"Downloading {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
    logger.info(f"Date range: {args.start} -> {args.end or 'today'}")
    logger.info(f"Output: {args.output_dir}")

    # Select downloader
    if args.source == "auto":
        downloader = DataDownloader(args.output_dir)
        results = downloader.download_universe(
            symbols, args.start, args.end
        )
    elif args.source == "yfinance":
        dl = YFinanceDownloader(args.output_dir)
        results = dl.download_universe(symbols, args.start, args.end)
    elif args.source == "polygon":
        dl = PolygonDownloader(args.output_dir)
        results = dl.download_universe(symbols, args.start, args.end)
    elif args.source == "stooq":
        dl = StooqDownloader(args.output_dir)
        results = dl.download_universe(symbols, args.start, args.end)
    elif args.source == "alphavantage":
        dl = AlphaVantageDownloader(args.output_dir)
        results = dl.download_universe(symbols)

    # Summary
    succeeded = sum(1 for v in results.values() if v)
    failed = [s for s, v in results.items() if v is None]

    print(f"\n{'='*50}")
    print(f"Download complete: {succeeded}/{len(symbols)} symbols")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:20])}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    print(f"Data saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

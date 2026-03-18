#!/usr/bin/env python3
"""
Scan the market for new trading opportunities.

Checks pools of stocks (mid-cap growth, biotech, crypto-adjacent, etc.)
for unusual volume, momentum, or breakouts. Automatically downloads data
and adds winners to the universe.

Usage:
    python scripts/scan_market.py
    python scripts/scan_market.py --pools mid_cap_growth biotech
    python scripts/scan_market.py --max-new 10
    python scripts/scan_market.py --dry-run   # scan without adding
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_feeds.scanner import (
    get_scan_pool,
    scan_for_opportunities,
    add_to_universe,
    SCAN_POOLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Scan market for new trading opportunities"
    )
    parser.add_argument(
        "--pools",
        nargs="+",
        default=None,
        choices=list(SCAN_POOLS.keys()),
        help=f"Scan pools to check (default: all). Options: {list(SCAN_POOLS.keys())}",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=20,
        help="Max new symbols to add (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report but don't add to universe",
    )
    parser.add_argument(
        "--universe",
        default=str(ROOT / "data" / "universe.csv"),
        help="Path to universe.csv",
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "ohlcv"),
        help="Path to OHLCV data directory",
    )
    args = parser.parse_args()

    # Load existing universe
    import pandas as pd
    if Path(args.universe).exists():
        uni = pd.read_csv(args.universe)
        existing = uni["symbol"].tolist()
    else:
        existing = []

    logger.info(f"Current universe: {len(existing)} symbols")

    # Get scan pool
    scan_tickers = get_scan_pool(args.pools)
    logger.info(f"Scanning {len(scan_tickers)} tickers across {args.pools or 'all'} pools")

    # Scan
    opportunities = scan_for_opportunities(
        scan_tickers, existing, max_new_symbols=args.max_new
    )

    # Report
    if opportunities:
        print(f"\n{'='*70}")
        print(f"  SCANNER RESULTS: {len(opportunities)} opportunities found")
        print(f"{'='*70}")
        print(f"  {'Symbol':<8} {'Reason':<25} {'Price':>8} {'Vol Surge':>10} {'Mom 5d':>8}")
        print(f"  {'-'*8} {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
        for opp in opportunities:
            print(
                f"  {opp['symbol']:<8} {opp['reason']:<25} "
                f"${opp['last_price']:>7.2f} {opp['volume_surge']:>9.1f}x "
                f"{opp['momentum_5d']:>+7.1%}"
            )
        print(f"{'='*70}")
    else:
        print("\nNo new opportunities found.")

    # Add to universe (unless dry run)
    if opportunities and not args.dry_run:
        added = add_to_universe(
            opportunities, args.universe, args.data_dir
        )
        if added:
            print(f"\nAdded {len(added)} symbols to universe: {added}")
        else:
            print("\nNo new symbols added (all already in universe or download failed)")
    elif args.dry_run and opportunities:
        print("\n(Dry run — no symbols added)")


if __name__ == "__main__":
    main()

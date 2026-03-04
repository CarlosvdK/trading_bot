#!/usr/bin/env python3
"""
Scan news for trading signals — standalone script.

Fetches news from yfinance + RSS, analyzes sentiment, reasons about
cross-stock impacts, and outputs actionable signals.

Usage:
    python scripts/scan_news.py
    python scripts/scan_news.py --symbols NVDA AMD TSLA
    python scripts/scan_news.py --top 10
    python scripts/scan_news.py --no-rss           # faster, yfinance only
    python scripts/scan_news.py --show-propagation  # show cross-stock reasoning
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.news_fetcher import fetch_all_news
from src.analysis.sentiment import analyze_news_batch, aggregate_symbol_sentiment
from src.analysis.cross_stock import propagate_sentiment
from src.analysis.news_signals import generate_news_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Scan news for trading signals with cross-stock reasoning"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to scan (default: from universe.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top signals to show (default: 20)",
    )
    parser.add_argument(
        "--no-rss",
        action="store_true",
        help="Skip RSS feeds (faster, yfinance only)",
    )
    parser.add_argument(
        "--show-propagation",
        action="store_true",
        help="Show cross-stock reasoning details",
    )
    parser.add_argument(
        "--show-headlines",
        action="store_true",
        help="Show individual headlines with sentiment scores",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.10,
        help="Minimum absolute sentiment score (default: 0.10)",
    )
    parser.add_argument(
        "--universe",
        default=str(ROOT / "data" / "universe.csv"),
        help="Path to universe.csv",
    )
    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    elif Path(args.universe).exists():
        uni = pd.read_csv(args.universe)
        symbols = uni["symbol"].tolist()
    else:
        logger.error("No symbols provided and universe.csv not found")
        sys.exit(1)

    logger.info(f"Scanning news for {len(symbols)} symbols...")

    # Fetch & analyze
    news_items = fetch_all_news(
        symbols,
        include_rss=not args.no_rss,
        max_age_hours=72,
    )

    if not news_items:
        print("\nNo news items found.")
        return

    # Analyze sentiment
    news_items = analyze_news_batch(news_items)

    # Show individual headlines if requested
    if args.show_headlines:
        print(f"\n{'='*80}")
        print(f"  RAW HEADLINES ({len(news_items)} items)")
        print(f"{'='*80}")
        for item in news_items[:50]:
            sym = item.get("symbol", "???")
            score = item.get("final_score", 0)
            indicator = "+" if score > 0.1 else "-" if score < -0.1 else " "
            print(f"  [{indicator}] {sym:>6} | {score:+.3f} | {item['title'][:70]}")
        print()

    # Generate full signals with propagation
    signals = generate_news_signals(
        symbols,
        min_score=args.min_score,
        max_signals=args.top,
        news_items=news_items,
    )

    # Display results
    if signals:
        print(f"\n{'='*80}")
        print(f"  NEWS SIGNALS: {len(signals)} opportunities found")
        print(f"{'='*80}")
        print(
            f"  {'Symbol':<8} {'Dir':<6} {'Score':>7} {'Mag':>6} "
            f"{'Arts':>5} {'Event':<15} {'Reason'}"
        )
        print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*5} {'-'*15} {'-'*30}")

        for sig in signals:
            print(
                f"  {sig['symbol']:<8} {sig['direction']:<6} "
                f"{sig['score']:>+.3f} {sig['magnitude']:>5.3f} "
                f"{sig['n_articles']:>5} {sig['event_type']:<15} "
                f"{sig['reason'][:40]}"
            )

            # Show headlines
            if sig.get("headlines"):
                for h in sig["headlines"][:2]:
                    print(f"           └─ {h[:65]}")

            # Show propagation
            if args.show_propagation and sig.get("propagated_from"):
                for prop in sig["propagated_from"]:
                    print(
                        f"           ↗ from {prop['source']} "
                        f"({prop['relationship']}, impact={prop['impact']:+.3f})"
                    )

        print(f"{'='*80}")

        # Summary
        bullish = [s for s in signals if s["direction"] == "LONG"]
        bearish = [s for s in signals if s["direction"] == "SHORT"]
        propagated = [s for s in signals if s.get("propagated_from")]

        print(f"\n  Summary:")
        print(f"    Bullish signals:   {len(bullish)}")
        print(f"    Bearish signals:   {len(bearish)}")
        print(f"    Cross-stock:       {len(propagated)} (from related stock reasoning)")
        if bullish:
            print(f"    Strongest bull:    {bullish[0]['symbol']} ({bullish[0]['score']:+.3f})")
        if bearish:
            print(f"    Strongest bear:    {bearish[0]['symbol']} ({bearish[0]['score']:+.3f})")
    else:
        print("\nNo strong news signals found.")


if __name__ == "__main__":
    main()

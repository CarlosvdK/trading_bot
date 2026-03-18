"""
Pre-Market Analyzer — builds overnight playbook for market open.

Runs continuously while markets are closed:
  1. Accumulates news with rolling sentiment
  2. Detects overnight catalysts (earnings, FDA, macro)
  3. Tracks futures/pre-market indicators
  4. Ranks opportunities by conviction
  5. Builds a prioritized "morning playbook" of trades

The playbook is what the bot executes at 9:30 AM.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_feeds.news_fetcher import fetch_all_news
from src.market_intel.sentiment import (
    analyze_news_batch,
    aggregate_symbol_sentiment,
    detect_event_type,
)
from src.market_intel.cross_stock import propagate_sentiment
from src.market_intel.news_signals import get_news_boost

logger = logging.getLogger(__name__)


class PreMarketAnalyzer:
    """
    Accumulates overnight intelligence and builds a morning playbook.

    Usage:
        analyzer = PreMarketAnalyzer(symbols)
        # Call repeatedly overnight:
        analyzer.scan()
        analyzer.scan()
        # At market open:
        playbook = analyzer.build_playbook()
    """

    def __init__(self, symbols: List[str], config: dict = None):
        self.symbols = symbols
        self.config = config or {}

        # Accumulated state
        self.news_history: List[dict] = []
        self.sentiment_snapshots: List[dict] = []  # Time series of sentiment
        self.scan_count = 0
        self.last_scan_time: Optional[datetime] = None

        # Overnight tracking
        self.overnight_catalysts: List[dict] = []
        self.sentiment_by_symbol: Dict[str, list] = defaultdict(list)

    def scan(self) -> dict:
        """
        Run one scan cycle. Call this every 30-60 minutes overnight.
        Fetches news, analyzes sentiment, tracks changes.

        Returns summary of what was found this scan.
        """
        self.scan_count += 1
        scan_time = datetime.now()

        # Fetch fresh news
        try:
            news_items = fetch_all_news(
                self.symbols,
                include_rss=True,
                include_yahoo=True,
                max_age_hours=self.config.get("overnight_max_age_hours", 12),
            )
        except Exception as e:
            logger.warning(f"Overnight news fetch failed: {e}")
            return {"status": "fetch_failed", "error": str(e)}

        if not news_items:
            self.last_scan_time = scan_time
            return {"status": "no_news", "scan_number": self.scan_count}

        # Analyze sentiment
        news_items = analyze_news_batch(news_items)

        # Find NEW items (not seen before)
        existing_titles = {n["title"][:80].lower() for n in self.news_history}
        new_items = [
            n for n in news_items
            if n["title"][:80].lower() not in existing_titles
        ]

        # Add to history
        self.news_history.extend(new_items)

        # Detect catalysts in new items
        for item in new_items:
            event_type = detect_event_type(item["title"])
            magnitude = item.get("magnitude", 0)

            if magnitude > 0.3 or event_type in ("earnings", "fda", "merger_acquisition"):
                self.overnight_catalysts.append({
                    "symbol": item.get("symbol", ""),
                    "title": item["title"],
                    "event_type": event_type,
                    "score": item.get("final_score", 0),
                    "magnitude": magnitude,
                    "published": item.get("published", scan_time),
                    "detected_at": scan_time,
                })

        # Take sentiment snapshot
        symbol_sentiment = aggregate_symbol_sentiment(self.news_history)
        full_sentiment = propagate_sentiment(symbol_sentiment, self.symbols)

        # Track sentiment evolution per symbol
        for sym, sent in full_sentiment.items():
            self.sentiment_by_symbol[sym].append({
                "time": scan_time,
                "score": sent["score"],
                "magnitude": sent["magnitude"],
                "n_articles": sent.get("n_articles", 0),
            })

        self.sentiment_snapshots.append({
            "time": scan_time,
            "n_symbols_with_signal": len(full_sentiment),
            "avg_magnitude": np.mean([s["magnitude"] for s in full_sentiment.values()]) if full_sentiment else 0,
        })

        self.last_scan_time = scan_time

        summary = {
            "status": "ok",
            "scan_number": self.scan_count,
            "new_items": len(new_items),
            "total_accumulated": len(self.news_history),
            "catalysts": len(self.overnight_catalysts),
            "symbols_with_signal": len(full_sentiment),
        }

        if new_items:
            logger.info(
                f"Overnight scan #{self.scan_count}: "
                f"{len(new_items)} new items, "
                f"{len(self.overnight_catalysts)} catalysts total"
            )

        return summary

    def build_playbook(
        self,
        max_trades: int = 10,
        min_conviction: float = 0.15,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """
        Build the morning playbook — a prioritized list of trades
        to execute at market open.

        Returns:
            playbook: dict with:
                trades: list of trade recommendations
                catalysts: overnight catalysts summary
                market_mood: overall sentiment direction
                confidence: how confident the playbook is (0-1)
        """
        if not self.news_history:
            return {
                "trades": [],
                "catalysts": [],
                "market_mood": "neutral",
                "confidence": 0.0,
                "scan_count": self.scan_count,
            }

        # Build final sentiment from ALL accumulated news
        symbol_sentiment = aggregate_symbol_sentiment(self.news_history)
        full_sentiment = propagate_sentiment(symbol_sentiment, self.symbols)

        # Score each symbol
        trade_candidates = []
        for sym, sent in full_sentiment.items():
            score = sent["score"]
            magnitude = sent["magnitude"]

            if abs(score) < min_conviction:
                continue

            # Boost score based on:
            # 1. Sentiment trend (improving/worsening over overnight)
            trend_boost = self._compute_trend_boost(sym)

            # 2. Catalyst presence
            sym_catalysts = [c for c in self.overnight_catalysts if c["symbol"] == sym]
            catalyst_boost = min(0.2, len(sym_catalysts) * 0.1)

            # 3. News volume (more articles = higher conviction)
            volume_boost = min(0.15, sent.get("n_articles", 0) * 0.03)

            # Combined conviction score
            conviction = abs(score) + trend_boost + catalyst_boost + volume_boost
            conviction = min(1.0, conviction)

            direction = "LONG" if score > 0 else "AVOID"

            # Build reason
            reasons = []
            if sym_catalysts:
                top_catalyst = max(sym_catalysts, key=lambda c: c["magnitude"])
                reasons.append(f"{top_catalyst['event_type']}: {top_catalyst['title'][:60]}")
            if sent.get("propagated_from"):
                sources = [p["source"] for p in sent["propagated_from"][:2]]
                reasons.append(f"ripple from {', '.join(sources)}")
            if trend_boost > 0.05:
                reasons.append("sentiment improving overnight")
            if sent.get("n_articles", 0) >= 3:
                reasons.append(f"{sent['n_articles']} articles")

            trade_candidates.append({
                "symbol": sym,
                "direction": direction,
                "conviction": round(conviction, 3),
                "sentiment_score": round(score, 4),
                "magnitude": round(magnitude, 4),
                "news_boost": round(get_news_boost(sym, [{
                    "symbol": sym,
                    "score": score,
                    "magnitude": magnitude,
                }]), 4),
                "n_articles": sent.get("n_articles", 0),
                "catalysts": [c["title"][:80] for c in sym_catalysts],
                "reasons": reasons,
                "propagated_from": sent.get("propagated_from", []),
                "event_types": list(set(c["event_type"] for c in sym_catalysts)) if sym_catalysts else [],
            })

        # Sort by conviction (highest first)
        trade_candidates.sort(key=lambda x: x["conviction"], reverse=True)
        trades = trade_candidates[:max_trades]

        # Overall market mood
        all_scores = [sent["score"] for sent in full_sentiment.values() if sent.get("n_articles", 0) > 0]
        if all_scores:
            avg_score = np.mean(all_scores)
            if avg_score > 0.1:
                market_mood = "bullish"
            elif avg_score < -0.1:
                market_mood = "bearish"
            else:
                market_mood = "neutral"
        else:
            market_mood = "quiet"

        # Confidence based on data quality
        confidence = min(1.0, self.scan_count * 0.15 + len(self.news_history) * 0.01)

        playbook = {
            "trades": trades,
            "catalysts": self.overnight_catalysts[-10:],  # Last 10 catalysts
            "market_mood": market_mood,
            "confidence": round(confidence, 2),
            "scan_count": self.scan_count,
            "total_news": len(self.news_history),
            "generated_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Playbook built: {len(trades)} trades, "
            f"mood={market_mood}, confidence={confidence:.0%}"
        )

        return playbook

    def _compute_trend_boost(self, symbol: str) -> float:
        """
        Check if sentiment for a symbol is improving or worsening
        over the overnight scans.
        """
        history = self.sentiment_by_symbol.get(symbol, [])
        if len(history) < 2:
            return 0.0

        # Compare last score vs first score
        first_score = history[0]["score"]
        last_score = history[-1]["score"]
        trend = last_score - first_score

        # Positive trend = sentiment improving (bullish getting more bullish)
        if last_score > 0 and trend > 0:
            return min(0.1, trend)
        elif last_score < 0 and trend < 0:
            return min(0.1, abs(trend))

        return 0.0

    def get_news_signals(self) -> List[dict]:
        """
        Convert accumulated sentiment into news_signals format
        compatible with the orchestrator's get_news_boost().
        """
        if not self.news_history:
            return []

        symbol_sentiment = aggregate_symbol_sentiment(self.news_history)
        full_sentiment = propagate_sentiment(symbol_sentiment, self.symbols)

        signals = []
        for sym, sent in full_sentiment.items():
            if abs(sent["score"]) < 0.1:
                continue
            signals.append({
                "symbol": sym,
                "score": sent["score"],
                "magnitude": sent["magnitude"],
                "direction": "LONG" if sent["score"] > 0 else "SHORT",
                "n_articles": sent.get("n_articles", 0),
                "propagated_from": sent.get("propagated_from", []),
            })

        signals.sort(key=lambda x: x["magnitude"], reverse=True)
        return signals

    def reset(self):
        """Reset for a new overnight cycle."""
        self.news_history.clear()
        self.sentiment_snapshots.clear()
        self.overnight_catalysts.clear()
        self.sentiment_by_symbol.clear()
        self.scan_count = 0
        self.last_scan_time = None


class WeekendAnalyzer:
    """
    Weekend-specific analysis — heavier compute that we don't want
    during trading hours.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def run_weekend_tasks(
        self,
        price_data: Dict[str, pd.DataFrame],
        orchestrator=None,
    ) -> dict:
        """
        Run weekend maintenance tasks:
          1. Retrain ML model on full dataset
          2. Scan for new universe additions
          3. Refresh data (download latest)
          4. Compute sector rotation signals

        Returns summary of what was done.
        """
        results = {}

        # 1. Retrain ML model if we have an orchestrator
        if orchestrator and orchestrator.ml_trainer:
            index_symbol = self.config.get("features", {}).get("index_symbol", "SPY")
            index_df = price_data.get(index_symbol, pd.DataFrame())
            if not index_df.empty:
                logger.info("Weekend: retraining ML model...")
                model, report = orchestrator.ml_trainer.walk_forward_train(
                    price_data, index_df, orchestrator.config
                )
                results["retrain"] = {
                    "status": report.get("status"),
                    "auc": report.get("avg_oos_auc"),
                }

        # 2. Scan for new stocks
        try:
            from src.data_feeds.scanner import run_full_scan
            universe_file = self.config.get("data", {}).get("universe_file", "data/universe.csv")
            data_dir = self.config.get("data", {}).get("data_dir", "data/ohlcv")
            added = run_full_scan(universe_file, data_dir, max_new=10)
            results["scanner"] = {"added": added}
            if added:
                logger.info(f"Weekend: added {len(added)} new symbols: {added}")
        except Exception as e:
            logger.warning(f"Weekend scan failed: {e}")
            results["scanner"] = {"error": str(e)}

        # 3. Sector rotation analysis
        results["sector_rotation"] = self._analyze_sector_rotation(price_data)

        return results

    def _analyze_sector_rotation(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> dict:
        """
        Check which sectors are gaining/losing momentum.
        Useful for adjusting universe weights.
        """
        sector_etfs = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB"]
        rotation = {}

        for etf in sector_etfs:
            df = price_data.get(etf)
            if df is None or len(df) < 21:
                continue

            close = df["close"]
            ret_5d = (close.iloc[-1] / close.iloc[-6]) - 1 if len(close) >= 6 else 0
            ret_21d = (close.iloc[-1] / close.iloc[-22]) - 1 if len(close) >= 22 else 0

            rotation[etf] = {
                "ret_5d": round(float(ret_5d), 4),
                "ret_21d": round(float(ret_21d), 4),
                "trend": "up" if ret_5d > 0 and ret_21d > 0 else "down" if ret_5d < 0 and ret_21d < 0 else "mixed",
            }

        # Sort by 5d momentum
        sorted_sectors = sorted(rotation.items(), key=lambda x: x[1]["ret_5d"], reverse=True)

        if sorted_sectors:
            logger.info(
                f"Sector rotation: strongest={sorted_sectors[0][0]} "
                f"({sorted_sectors[0][1]['ret_5d']:+.1%}), "
                f"weakest={sorted_sectors[-1][0]} "
                f"({sorted_sectors[-1][1]['ret_5d']:+.1%})"
            )

        return rotation

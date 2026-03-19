"""
News Signal Generator — converts news sentiment into trading signals.

Pipeline:
  1. Fetch news (yfinance + RSS)
  2. Score sentiment (VADER + financial keywords)
  3. Propagate to related stocks (cross-stock reasoning)
  4. Generate buy/avoid signals based on composite scores

These signals feed into the orchestrator alongside the existing
momentum/vol-expansion signals and get filtered by the ML model.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.data_feeds.news_fetcher import fetch_all_news
from src.market_intel.sentiment import (
    analyze_news_batch,
    aggregate_symbol_sentiment,
    detect_event_type,
)
from src.market_intel.cross_stock import propagate_sentiment

logger = logging.getLogger(__name__)


def generate_news_signals(
    symbols: List[str],
    min_score: float = 0.15,
    min_magnitude: float = 0.10,
    max_signals: int = 20,
    include_rss: bool = True,
    max_age_hours: int = 72,
    news_items: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Full news signal pipeline: fetch → analyze → propagate → signal.

    Args:
        symbols: Universe of known symbols
        min_score: Minimum absolute sentiment score to generate signal
        min_magnitude: Minimum signal magnitude
        max_signals: Cap on number of signals returned
        include_rss: Whether to include RSS feeds (slower but broader)
        max_age_hours: Max age of news to consider
        news_items: Pre-fetched news items (skip fetching if provided)

    Returns:
        List of signal dicts sorted by magnitude (strongest first):
            symbol, direction, score, magnitude, reason, headlines,
            n_articles, propagated_from, event_type
    """
    # 1. Fetch news
    if news_items is None:
        news_items = fetch_all_news(
            symbols,
            include_rss=include_rss,
            max_age_hours=max_age_hours,
        )

    if not news_items:
        logger.info("No news items found")
        return []

    # 2. Analyze sentiment
    news_items = analyze_news_batch(news_items)

    # 3. Aggregate per symbol
    symbol_sentiment = aggregate_symbol_sentiment(news_items)

    # 4. Propagate to related stocks
    full_sentiment = propagate_sentiment(symbol_sentiment, symbols)

    # 4b. LLM enhancement — deeper analysis for high-priority symbols
    try:
        from src.market_intel.llm_sentiment import enhance_sentiment_batch
        symbol_headlines = {}
        for item in news_items:
            sym = item.get("symbol", "")
            if sym:
                symbol_headlines.setdefault(sym, []).append(item["title"])
        full_sentiment = enhance_sentiment_batch(symbol_headlines, full_sentiment)
    except Exception as e:
        logger.debug(f"LLM sentiment enhancement skipped: {e}")

    # 5. Generate signals
    signals = []
    for sym, sent in full_sentiment.items():
        score = sent["score"]
        magnitude = sent["magnitude"]

        if abs(score) < min_score or magnitude < min_magnitude:
            continue

        direction = "LONG" if score > 0 else "SHORT"

        # Build reason string
        reasons = []
        if sent.get("n_articles", 0) > 0:
            reasons.append(f"{sent['n_articles']} articles")
        if sent.get("propagated_from"):
            sources = [p["source"] for p in sent["propagated_from"]]
            reasons.append(f"propagated from {','.join(sources[:3])}")
        if sent.get("bullish_count", 0) > sent.get("bearish_count", 0):
            reasons.append("bullish consensus")
        elif sent.get("bearish_count", 0) > sent.get("bullish_count", 0):
            reasons.append("bearish consensus")

        # Detect event type from top headline
        event_type = "general"
        if sent.get("headlines"):
            event_type = detect_event_type(sent["headlines"][0])

        signals.append({
            "symbol": sym,
            "signal_type": "news_sentiment",
            "direction": direction,
            "signal_date": pd.Timestamp.now(),
            "score": round(score, 4),
            "magnitude": round(magnitude, 4),
            "reason": "news:" + "+".join(reasons) if reasons else "news:sentiment",
            "headlines": sent.get("headlines", [])[:3],
            "n_articles": sent.get("n_articles", 0),
            "propagated_from": sent.get("propagated_from", []),
            "event_type": event_type,
            "news_volume_signal": sent.get("news_volume_signal", 1.0),
        })

    # Sort by magnitude (strongest signal first)
    signals.sort(key=lambda x: x["magnitude"], reverse=True)
    signals = signals[:max_signals]

    if signals:
        logger.info(
            f"News signals: {len(signals)} generated | "
            f"top: {signals[0]['symbol']} ({signals[0]['direction']}, "
            f"score={signals[0]['score']:+.3f})"
        )
    else:
        logger.info("News signals: none strong enough")

    return signals


def build_news_features(
    symbol: str,
    news_signals: List[dict],
) -> dict:
    """
    Build features from news signals for a specific symbol.
    These get added to the ML feature set.

    Returns dict of feature_name -> value:
        news_sentiment: composite sentiment score (-1 to +1)
        news_magnitude: strength of news signal
        news_volume: number of articles (normalized)
        news_is_propagated: 1 if signal came from related stock, 0 if direct
        news_event_score: event-specific impact modifier
    """
    # Find this symbol's signal
    matching = [s for s in news_signals if s["symbol"] == symbol]

    if not matching:
        return {
            "news_sentiment": 0.0,
            "news_magnitude": 0.0,
            "news_volume": 0.0,
            "news_is_propagated": 0.0,
            "news_event_score": 0.0,
        }

    sig = matching[0]  # Take strongest signal

    # Event score: some events are more impactful
    event_multipliers = {
        "earnings": 1.5,
        "fda": 1.8,
        "merger_acquisition": 1.6,
        "macro": 1.2,
        "trade_policy": 1.3,
        "tech_sector": 1.1,
        "crypto": 1.2,
        "analyst": 1.0,
        "legal": 1.3,
        "general": 0.8,
    }
    event_mult = event_multipliers.get(sig.get("event_type", "general"), 1.0)

    return {
        "news_sentiment": sig["score"],
        "news_magnitude": sig["magnitude"],
        "news_volume": min(1.0, sig.get("n_articles", 0) / 5.0),
        "news_is_propagated": 1.0 if sig.get("propagated_from") else 0.0,
        "news_event_score": sig["magnitude"] * event_mult,
    }


def get_news_boost(
    symbol: str,
    news_signals: List[dict],
) -> float:
    """
    Get a simple boost/penalty for ML probability based on news.

    Returns value between -0.15 and +0.15 that gets added to the ML
    probability score. This lets strong news override a weak ML signal
    or strengthen a strong one.
    """
    matching = [s for s in news_signals if s["symbol"] == symbol]
    if not matching:
        return 0.0

    sig = matching[0]
    # Scale: magnitude of 0.5 → ±0.10 boost
    boost = sig["score"] * min(sig["magnitude"], 0.5) * 0.3

    # Cap at ±0.15
    return max(-0.15, min(0.15, boost))

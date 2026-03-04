"""
News Sentiment Analyzer — scores headlines and reasons about cross-stock impact.

Two-layer system:
  1. VADER sentiment on each headline → raw sentiment score (-1 to +1)
  2. Financial keyword boosting → adjusts score for finance-specific terms
  3. Cross-stock reasoning → propagates impact to related stocks

The cross-stock reasoning is key: if "NVDA beats earnings by 40%", this module
figures out that AMD, AVGO, SMH, MARA (GPU miners) all benefit too.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Financial keyword boosters — VADER doesn't know finance
BULLISH_KEYWORDS = {
    # Earnings
    "beat": 0.3, "beats": 0.3, "exceeded": 0.3, "surpassed": 0.3,
    "record revenue": 0.4, "record earnings": 0.4, "record profit": 0.4,
    "blowout": 0.4, "crushed estimates": 0.5,
    "raised guidance": 0.4, "raises guidance": 0.4, "upgrades guidance": 0.4,
    # Deals & growth
    "acquisition": 0.2, "acquires": 0.2, "merger": 0.2,
    "partnership": 0.2, "contract win": 0.3, "wins contract": 0.3,
    "fda approval": 0.5, "fda approved": 0.5,
    "breakthrough": 0.3, "innovation": 0.2,
    # Analyst actions
    "upgrade": 0.3, "upgrades": 0.3, "price target raised": 0.3,
    "buy rating": 0.3, "outperform": 0.2, "overweight": 0.2,
    # Market
    "rally": 0.2, "surges": 0.3, "soars": 0.3, "jumps": 0.2,
    "all-time high": 0.3, "new high": 0.2,
    "strong demand": 0.3, "accelerating growth": 0.3,
    "stock split": 0.2, "buyback": 0.2, "dividend increase": 0.2,
}

BEARISH_KEYWORDS = {
    # Earnings
    "miss": -0.3, "misses": -0.3, "missed": -0.3, "disappointed": -0.3,
    "lowered guidance": -0.4, "lowers guidance": -0.4, "cuts guidance": -0.4,
    "warns": -0.3, "warning": -0.3, "profit warning": -0.5,
    "revenue decline": -0.4, "revenue drop": -0.4,
    # Negative events
    "layoffs": -0.3, "lays off": -0.3, "job cuts": -0.3,
    "lawsuit": -0.2, "sued": -0.2, "investigation": -0.3,
    "recall": -0.3, "safety concern": -0.3,
    "data breach": -0.4, "hack": -0.3,
    "bankruptcy": -0.5, "default": -0.4, "debt crisis": -0.4,
    "fda rejection": -0.5, "fda rejects": -0.5, "trial failure": -0.5,
    # Analyst actions
    "downgrade": -0.3, "downgrades": -0.3, "price target cut": -0.3,
    "sell rating": -0.3, "underperform": -0.2, "underweight": -0.2,
    # Market
    "crash": -0.4, "plunges": -0.3, "plummets": -0.3, "tanks": -0.3,
    "selloff": -0.3, "sell-off": -0.3,
    "tariff": -0.2, "tariffs": -0.2, "trade war": -0.3,
    "recession": -0.3, "slowdown": -0.2,
    "short seller": -0.2, "fraud": -0.4, "sec investigation": -0.4,
}


def analyze_sentiment(title: str) -> dict:
    """
    Analyze a single headline for sentiment.

    Returns dict with:
        vader_score: raw VADER compound score (-1 to +1)
        finance_boost: adjustment from financial keywords
        final_score: combined score, clipped to [-1, +1]
        magnitude: abs(final_score) — how strong the signal is
        keywords_found: list of matched keywords
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    vader = analyzer.polarity_scores(title)
    vader_score = vader["compound"]

    # Apply financial keyword boosters
    title_lower = title.lower()
    finance_boost = 0.0
    keywords_found = []

    for keyword, boost in BULLISH_KEYWORDS.items():
        if keyword in title_lower:
            finance_boost += boost
            keywords_found.append(f"+{keyword}")

    for keyword, boost in BEARISH_KEYWORDS.items():
        if keyword in title_lower:
            finance_boost += boost  # boost is already negative
            keywords_found.append(f"{keyword}")

    # Combine: VADER base + financial context
    # Weight financial keywords higher since VADER doesn't understand finance well
    final_score = vader_score * 0.4 + finance_boost * 0.6
    final_score = max(-1.0, min(1.0, final_score))

    return {
        "vader_score": vader_score,
        "finance_boost": finance_boost,
        "final_score": final_score,
        "magnitude": abs(final_score),
        "keywords_found": keywords_found,
    }


def analyze_news_batch(news_items: List[dict]) -> List[dict]:
    """
    Analyze sentiment for a batch of news items.
    Adds sentiment fields to each news dict in-place.
    """
    for item in news_items:
        sentiment = analyze_sentiment(item["title"])
        item.update(sentiment)

    return news_items


def aggregate_symbol_sentiment(
    news_items: List[dict],
    decay_hours: float = 24.0,
) -> Dict[str, dict]:
    """
    Aggregate sentiment scores per symbol with time decay.

    Recent news gets higher weight. Multiple news items for the same
    symbol get combined (not just averaged — volume of news matters too).

    Returns dict of symbol -> {
        score: weighted average sentiment (-1 to +1),
        magnitude: strength of signal,
        n_articles: number of articles,
        bullish_count: number of bullish articles,
        bearish_count: number of bearish articles,
        headlines: top headlines,
        news_volume_signal: how unusual the news volume is,
    }
    """
    from datetime import datetime

    now = datetime.now()
    by_symbol = defaultdict(list)

    for item in news_items:
        sym = item.get("symbol", "")
        if not sym:
            continue

        # Time decay: recent news weighted more
        age_hours = max(0, (now - item["published"]).total_seconds() / 3600)
        time_weight = np.exp(-age_hours / decay_hours)

        by_symbol[sym].append({
            "score": item.get("final_score", 0),
            "magnitude": item.get("magnitude", 0),
            "time_weight": time_weight,
            "title": item["title"],
            "published": item["published"],
        })

    results = {}
    for sym, articles in by_symbol.items():
        if not articles:
            continue

        weights = [a["time_weight"] for a in articles]
        scores = [a["score"] for a in articles]
        total_weight = sum(weights)

        if total_weight == 0:
            continue

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        bullish = sum(1 for s in scores if s > 0.1)
        bearish = sum(1 for s in scores if s < -0.1)

        # News volume signal: more articles = stronger signal
        # 1 article = 1.0x, 3+ articles = up to 1.5x
        volume_multiplier = min(1.5, 1.0 + 0.15 * (len(articles) - 1))

        results[sym] = {
            "score": round(weighted_score, 4),
            "magnitude": round(abs(weighted_score) * volume_multiplier, 4),
            "n_articles": len(articles),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "news_volume_signal": round(volume_multiplier, 2),
            "headlines": [a["title"] for a in sorted(
                articles, key=lambda x: x["time_weight"], reverse=True
            )[:5]],
        }

    return results


def detect_event_type(title: str) -> str:
    """Classify a headline into an event type for cross-stock reasoning."""
    title_lower = title.lower()

    patterns = {
        "earnings": r"earnings|revenue|profit|quarterly|q[1-4]|eps|guidance|beat|miss",
        "fda": r"fda|drug approval|clinical trial|phase \d|pipeline",
        "merger_acquisition": r"acqui|merger|buyout|takeover|deal|bid",
        "macro": r"fed|interest rate|inflation|gdp|employment|jobs|payroll|cpi|fomc",
        "trade_policy": r"tariff|trade war|sanctions|ban|embargo|export control",
        "tech_sector": r"ai\b|artificial intelligence|chip|semiconductor|cloud|data center",
        "crypto": r"bitcoin|crypto|blockchain|ethereum|btc|mining",
        "energy": r"oil|opec|natural gas|crude|energy price|drilling",
        "analyst": r"upgrade|downgrade|price target|rating|coverage|analyst",
        "legal": r"lawsuit|sued|investigation|sec |doj|antitrust|fine|penalty",
    }

    for event_type, pattern in patterns.items():
        if re.search(pattern, title_lower):
            return event_type

    return "general"

"""
News sentiment NLP pipeline.
Full preprocessing and scoring — ready to plug in when API keys arrive.
Uses VADER with financial domain augmentations.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


FINANCIAL_LEXICON = {
    "bullish": 3.0, "upgrade": 2.5, "outperform": 2.0, "buyback": 2.0,
    "beat": 1.5, "exceeded": 1.5, "dividend": 1.0, "growth": 1.0,
    "rally": 2.0, "breakout": 1.5, "surge": 2.0, "soar": 2.5,
    "upside": 1.5, "catalyst": 1.0, "momentum": 0.8, "accumulation": 1.0,
    "overweight": 1.5, "reiterate": 0.3, "initiated": 0.5, "buy": 1.5,
    "bearish": -3.0, "downgrade": -2.5, "underperform": -2.0, "dilution": -2.0,
    "miss": -1.5, "missed": -1.5, "restructuring": -1.0, "layoffs": -1.5,
    "recall": -1.5, "lawsuit": -1.5, "fraud": -3.0, "bankruptcy": -3.5,
    "default": -2.5, "selloff": -2.0, "plunge": -2.5, "crash": -3.0,
    "warning": -1.5, "underweight": -1.5, "subpoena": -2.0,
    "investigation": -1.5, "shortfall": -1.5, "decline": -1.0,
    "weakness": -1.0, "headwind": -1.0, "sell": -1.5,
    "guidance": 0.5, "acquisition": 0.5, "merger": 0.3, "spinoff": 0.3,
    "ipo": 0.5, "volatility": -0.5,
}


class FinancialSentimentScorer:
    """VADER-based scorer augmented with financial domain lexicon."""

    def __init__(self, custom_lexicon: Optional[Dict[str, float]] = None):
        self.analyzer = SentimentIntensityAnalyzer()
        lexicon = {**FINANCIAL_LEXICON, **(custom_lexicon or {})}
        self.analyzer.lexicon.update(lexicon)

    def score_text(self, text: str) -> Dict[str, float]:
        """Score arbitrary text, returning full sentiment breakdown."""
        scores = self.analyzer.polarity_scores(text)
        return {
            "compound": scores["compound"], "pos": scores["pos"],
            "neg": scores["neg"], "neu": scores["neu"],
            "financial_score": scores["compound"],
        }

    def score_headline(self, headline: str) -> float:
        """Quick compound score for a headline."""
        return self.analyzer.polarity_scores(headline)["compound"]

    def score_article(self, title: str, body: str) -> Dict[str, float]:
        """Weighted title (0.6) + body (0.4) score."""
        t = self.score_text(title)
        b = self.score_text(body)
        return {k: t[k] * 0.6 + b[k] * 0.4 for k in t}


DEFAULT_SOURCE_CREDIBILITY = {
    "reuters": 1.0, "bloomberg": 1.0, "wsj": 0.95, "cnbc": 0.8,
    "marketwatch": 0.75, "seekingalpha": 0.6, "benzinga": 0.65,
    "yahoo": 0.7, "unknown": 0.5,
}


class NewsAggregator:
    """Aggregates sentiment across multiple news items with time decay."""

    def __init__(self, scorer: Optional[FinancialSentimentScorer] = None,
                 source_credibility: Optional[Dict[str, float]] = None):
        self.scorer = scorer or FinancialSentimentScorer()
        self.source_credibility = source_credibility or DEFAULT_SOURCE_CREDIBILITY

    def aggregate_sentiment(
        self, news_items: List[Dict],
        current_time: Optional[datetime] = None,
        decay_hours: float = 24.0,
    ) -> Dict[str, Dict]:
        """Aggregate sentiment by symbol with time-decay and source weighting."""
        if current_time is None:
            current_time = datetime.now()

        by_symbol: Dict[str, List[Dict]] = {}
        for item in news_items:
            sym = item.get("symbol", "UNKNOWN")
            score = self.scorer.score_headline(item.get("title", ""))
            pub = item.get("published_at", current_time)
            if isinstance(pub, str):
                try:
                    pub = pd.Timestamp(pub).to_pydatetime()
                except Exception:
                    pub = current_time

            hours_ago = max((current_time - pub).total_seconds() / 3600, 0)
            time_weight = np.exp(-0.693 * hours_ago / decay_hours)
            source = item.get("source", "unknown").lower()
            source_weight = self.source_credibility.get(source, 0.5)

            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append({
                "score": score, "time_weight": time_weight,
                "source_weight": source_weight,
                "combined_weight": time_weight * source_weight,
            })

        results = {}
        for sym, items in by_symbol.items():
            scores = [i["score"] for i in items]
            weights = [i["combined_weight"] for i in items]
            total_w = sum(weights)
            avg_score = np.mean(scores) if scores else 0.0
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
            n_bull = sum(1 for s in scores if s > 0.05)
            n_bear = sum(1 for s in scores if s < -0.05)

            results[sym] = {
                "avg_score": float(avg_score), "weighted_score": float(weighted_score),
                "n_articles": len(items),
                "sentiment_trend": float(weighted_score - avg_score),
                "bull_bear_ratio": n_bull / n_bear if n_bear > 0 else float(n_bull) if n_bull > 0 else 0.0,
            }
        return results


class SentimentFeatureBuilder:
    """Builds backward-looking sentiment features for ML pipeline."""

    def build_features(
        self, sentiment_history: Dict[str, List[Tuple[pd.Timestamp, float]]],
        current_date: pd.Timestamp,
    ) -> Dict[str, Dict[str, float]]:
        """Build features from historical scores. All backward-looking."""
        results = {}
        for symbol, history in sentiment_history.items():
            past = [(d, s) for d, s in history if d <= current_date]
            if not past:
                continue
            dates, scores = zip(*past)
            series = pd.Series(scores, index=pd.DatetimeIndex(dates)).sort_index()

            ma_3 = series.iloc[-3:].mean() if len(series) >= 3 else series.mean()
            ma_7 = series.iloc[-7:].mean() if len(series) >= 7 else series.mean()

            results[symbol] = {
                "sentiment_ma_3d": float(ma_3),
                "sentiment_ma_7d": float(ma_7),
                "sentiment_momentum": float(ma_3 - ma_7),
                "sentiment_dispersion": float(series.iloc[-7:].std()) if len(series) >= 2 else 0.0,
                "extreme_sentiment_flag": 1.0 if abs(ma_3) > 0.5 else 0.0,
                "news_volume_zscore": 0.0,
            }
        return results


COMPANY_TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "tesla": "TSLA", "jpmorgan": "JPM", "johnson & johnson": "JNJ",
    "visa": "V", "netflix": "NFLX", "amd": "AMD",
}

EVENT_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "quarterly", "annual report", "beat", "miss", "guidance"],
    "merger": ["merger", "acquisition", "acquire", "buyout", "takeover", "deal"],
    "regulatory": ["sec", "fda", "antitrust", "regulation", "compliance", "fine", "penalty", "subpoena"],
    "macro": ["fed", "interest rate", "inflation", "gdp", "unemployment", "fomc", "treasury"],
    "technical": ["breakout", "support", "resistance", "moving average", "rsi", "macd"],
}


class KeywordExtractor:
    """Extract tickers, entities, and classify event types from text."""

    def __init__(self, company_map: Optional[Dict[str, str]] = None,
                 event_keywords: Optional[Dict[str, List[str]]] = None):
        self.company_map = company_map or COMPANY_TICKER_MAP
        self.event_keywords = event_keywords or EVENT_KEYWORDS

    def extract_tickers(self, text: str) -> List[str]:
        """Find stock tickers in text ($AAPL, AAPL, etc.)."""
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        word_tickers = re.findall(r'\b([A-Z]{2,5})\b', text)
        stop_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                      "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "CEO", "CFO",
                      "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET",
                      "HAS", "HIM", "HIS", "HOW", "ITS", "MAY", "SAY", "SHE"}
        word_tickers = [t for t in word_tickers if t not in stop_words]
        return sorted(set(dollar_tickers + word_tickers))

    def extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction using company name mapping."""
        text_lower = text.lower()
        return sorted({ticker for name, ticker in self.company_map.items() if name in text_lower})

    def classify_event_type(self, text: str) -> str:
        """Classify text into event categories using keyword matching."""
        text_lower = text.lower()
        scores = {et: sum(1 for kw in kws if kw in text_lower)
                  for et, kws in self.event_keywords.items()}
        scores = {k: v for k, v in scores.items() if v > 0}
        return max(scores, key=scores.get) if scores else "other"


class SentimentSignalGenerator:
    """Generate trading signals from sentiment data."""

    def __init__(self, scorer: Optional[FinancialSentimentScorer] = None, config: Optional[dict] = None):
        self.scorer = scorer or FinancialSentimentScorer()
        self.config = config or {}

    def generate_signals(
        self, sentiment_data: Dict[str, Dict],
        prices: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        config: Optional[dict] = None,
    ) -> List[dict]:
        """Generate signals when sentiment diverges from or confirms price."""
        cfg = config or self.config
        contrarian_threshold = cfg.get("sentiment_contrarian_threshold", 0.3)
        confirmation_threshold = cfg.get("sentiment_confirmation_threshold", 0.2)

        signals = []
        for symbol, sent in sentiment_data.items():
            if symbol not in prices or current_date not in prices[symbol].index:
                continue
            df = prices[symbol]
            idx = df.index.get_loc(current_date)
            if idx < 5:
                continue

            price_ret_5d = (df["close"].iloc[idx] / df["close"].iloc[idx - 5]) - 1
            sentiment_score = sent.get("weighted_score", 0)

            if sentiment_score > contrarian_threshold and price_ret_5d < -0.03:
                signals.append({"symbol": symbol, "signal_type": "sentiment_contrarian",
                                "direction": "LONG", "signal_date": current_date,
                                "sentiment_score": sentiment_score, "price_ret_5d": price_ret_5d})
            elif sentiment_score < -contrarian_threshold and price_ret_5d > 0.03:
                signals.append({"symbol": symbol, "signal_type": "sentiment_contrarian",
                                "direction": "SHORT", "signal_date": current_date,
                                "sentiment_score": sentiment_score, "price_ret_5d": price_ret_5d})

            if sentiment_score > confirmation_threshold and price_ret_5d > 0.02:
                signals.append({"symbol": symbol, "signal_type": "sentiment_confirmation",
                                "direction": "LONG", "signal_date": current_date,
                                "sentiment_score": sentiment_score, "price_ret_5d": price_ret_5d})
        return signals

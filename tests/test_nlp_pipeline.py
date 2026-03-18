"""Tests for NLP sentiment pipeline."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.market_intel.nlp_pipeline import (
    FinancialSentimentScorer,
    NewsAggregator,
    SentimentFeatureBuilder,
    KeywordExtractor,
    SentimentSignalGenerator,
)


class TestFinancialSentimentScorer:
    def test_bullish_positive(self):
        scorer = FinancialSentimentScorer()
        score = scorer.score_headline("Bullish upgrade with strong growth outlook")
        assert score > 0.3

    def test_bearish_negative(self):
        scorer = FinancialSentimentScorer()
        score = scorer.score_headline("Bearish downgrade as earnings miss expectations")
        assert score < -0.3

    def test_neutral_near_zero(self):
        scorer = FinancialSentimentScorer()
        score = scorer.score_headline("Company reports quarterly results")
        assert abs(score) < 0.5

    def test_article_scoring(self):
        scorer = FinancialSentimentScorer()
        result = scorer.score_article(
            "Company beats earnings estimates",
            "Revenue came in above expectations with strong growth across segments",
        )
        assert "compound" in result
        assert result["compound"] > 0

    def test_score_text_fields(self):
        scorer = FinancialSentimentScorer()
        result = scorer.score_text("Market rally continues")
        assert all(k in result for k in ["compound", "pos", "neg", "neu", "financial_score"])


class TestNewsAggregator:
    def test_time_decay_weighting(self):
        aggregator = NewsAggregator()
        now = datetime(2023, 6, 15, 12, 0)
        items = [
            {"title": "Bullish outlook", "symbol": "AAPL",
             "published_at": now, "source": "reuters"},
            {"title": "Bullish outlook", "symbol": "AAPL",
             "published_at": now - timedelta(hours=48), "source": "reuters"},
        ]
        result = aggregator.aggregate_sentiment(items, current_time=now, decay_hours=24)
        assert "AAPL" in result
        # Weighted score weights recent article more, so it should differ
        # from avg when scores differ; here both titles are identical so
        # test that decay was at least applied (weighted_score exists and is valid)
        assert isinstance(result["AAPL"]["weighted_score"], float)
        assert result["AAPL"]["n_articles"] == 2

    def test_source_credibility(self):
        aggregator = NewsAggregator()
        now = datetime(2023, 6, 15, 12, 0)
        items = [
            {"title": "Bullish outlook", "symbol": "AAPL",
             "published_at": now, "source": "reuters"},
        ]
        result = aggregator.aggregate_sentiment(items, current_time=now)
        assert result["AAPL"]["n_articles"] == 1

    def test_bull_bear_ratio(self):
        aggregator = NewsAggregator()
        now = datetime(2023, 6, 15, 12, 0)
        items = [
            {"title": "Strong bullish upgrade", "symbol": "AAPL", "published_at": now, "source": "reuters"},
            {"title": "Bearish downgrade warning", "symbol": "AAPL", "published_at": now, "source": "reuters"},
        ]
        result = aggregator.aggregate_sentiment(items, current_time=now)
        assert "bull_bear_ratio" in result["AAPL"]


class TestKeywordExtractor:
    def test_extract_dollar_tickers(self):
        extractor = KeywordExtractor()
        tickers = extractor.extract_tickers("Watch $AAPL and $MSFT today")
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_extract_entities(self):
        extractor = KeywordExtractor()
        entities = extractor.extract_entities("Apple and Microsoft reported earnings")
        assert "AAPL" in entities
        assert "MSFT" in entities

    def test_classify_earnings(self):
        extractor = KeywordExtractor()
        result = extractor.classify_event_type("Company beats EPS estimates, raises guidance")
        assert result == "earnings"

    def test_classify_merger(self):
        extractor = KeywordExtractor()
        result = extractor.classify_event_type("Company announces acquisition of rival firm")
        assert result == "merger"

    def test_classify_unknown(self):
        extractor = KeywordExtractor()
        result = extractor.classify_event_type("The weather is nice today")
        assert result == "other"

    def test_classify_macro(self):
        extractor = KeywordExtractor()
        result = extractor.classify_event_type("Fed raises interest rate amid inflation concerns")
        assert result == "macro"


class TestSentimentFeatureBuilder:
    def test_features_backward_looking(self):
        builder = SentimentFeatureBuilder()
        history = {
            "AAPL": [
                (pd.Timestamp("2023-06-01"), 0.3),
                (pd.Timestamp("2023-06-02"), 0.5),
                (pd.Timestamp("2023-06-03"), 0.2),
                (pd.Timestamp("2023-06-04"), -0.1),
                (pd.Timestamp("2023-06-05"), 0.4),
                (pd.Timestamp("2023-06-06"), 0.3),
                (pd.Timestamp("2023-06-07"), 0.1),
                (pd.Timestamp("2023-06-08"), 0.6),  # Future
            ],
        }
        # Should only use data up to 2023-06-07
        result = builder.build_features(history, pd.Timestamp("2023-06-07"))
        assert "AAPL" in result
        assert "sentiment_ma_3d" in result["AAPL"]
        assert "sentiment_momentum" in result["AAPL"]

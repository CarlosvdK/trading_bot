"""Tests for news sentiment analysis and cross-stock reasoning."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.market_intel.sentiment import (
    analyze_sentiment,
    analyze_news_batch,
    aggregate_symbol_sentiment,
    detect_event_type,
)
from src.market_intel.cross_stock import (
    get_related_stocks,
    propagate_sentiment,
    STOCK_RELATIONSHIPS,
)
from src.market_intel.news_signals import (
    generate_news_signals,
    build_news_features,
    get_news_boost,
)
from src.data_feeds.news_fetcher import match_news_to_symbols


class TestSentimentAnalysis:
    def test_bullish_headline(self):
        result = analyze_sentiment("NVDA beats earnings expectations with record revenue")
        assert result["final_score"] > 0
        assert result["magnitude"] > 0
        assert len(result["keywords_found"]) > 0

    def test_bearish_headline(self):
        result = analyze_sentiment("Company warns of revenue decline and layoffs ahead")
        assert result["final_score"] < 0
        assert result["magnitude"] > 0

    def test_neutral_headline(self):
        result = analyze_sentiment("Stock market closes for the day")
        assert abs(result["final_score"]) < 0.3

    def test_fda_approval_strong_bullish(self):
        result = analyze_sentiment("FDA approval for breakthrough cancer drug")
        assert result["final_score"] > 0.2

    def test_downgrade_bearish(self):
        result = analyze_sentiment("Analyst downgrades stock with price target cut")
        assert result["final_score"] < 0

    def test_returns_all_required_fields(self):
        result = analyze_sentiment("Test headline")
        assert "vader_score" in result
        assert "finance_boost" in result
        assert "final_score" in result
        assert "magnitude" in result
        assert "keywords_found" in result

    def test_score_bounded(self):
        # Even extreme headlines should be bounded -1 to +1
        result = analyze_sentiment(
            "Record revenue beats expectations, raised guidance, "
            "FDA approval, upgrade, all-time high"
        )
        assert -1.0 <= result["final_score"] <= 1.0


class TestAnalyzeNewsBatch:
    def test_adds_sentiment_to_items(self):
        items = [
            {"title": "NVDA crushes earnings", "symbol": "NVDA"},
            {"title": "Market crash fears grow", "symbol": ""},
        ]
        result = analyze_news_batch(items)
        assert len(result) == 2
        assert "final_score" in result[0]
        assert "final_score" in result[1]
        assert result[0]["final_score"] > result[1]["final_score"]


class TestAggregateSymbolSentiment:
    def test_aggregates_multiple_articles(self):
        now = datetime.now()
        items = [
            {"symbol": "AAPL", "title": "Apple beats", "final_score": 0.5,
             "magnitude": 0.5, "published": now},
            {"symbol": "AAPL", "title": "Apple strong demand", "final_score": 0.3,
             "magnitude": 0.3, "published": now - timedelta(hours=6)},
            {"symbol": "MSFT", "title": "Microsoft warns", "final_score": -0.4,
             "magnitude": 0.4, "published": now},
        ]
        result = aggregate_symbol_sentiment(items)
        assert "AAPL" in result
        assert "MSFT" in result
        assert result["AAPL"]["score"] > 0
        assert result["MSFT"]["score"] < 0
        assert result["AAPL"]["n_articles"] == 2
        assert result["AAPL"]["bullish_count"] == 2

    def test_time_decay(self):
        now = datetime.now()
        items_recent = [
            {"symbol": "TEST", "title": "Good", "final_score": 0.5,
             "magnitude": 0.5, "published": now},
        ]
        items_old = [
            {"symbol": "TEST", "title": "Good", "final_score": 0.5,
             "magnitude": 0.5, "published": now - timedelta(hours=48)},
        ]
        recent = aggregate_symbol_sentiment(items_recent)
        old = aggregate_symbol_sentiment(items_old)
        # Recent news should have higher magnitude due to time weighting
        assert recent["TEST"]["magnitude"] >= old["TEST"]["magnitude"]

    def test_empty_items(self):
        result = aggregate_symbol_sentiment([])
        assert result == {}

    def test_skips_no_symbol(self):
        items = [
            {"symbol": "", "title": "Something", "final_score": 0.5,
             "magnitude": 0.5, "published": datetime.now()},
        ]
        result = aggregate_symbol_sentiment(items)
        assert len(result) == 0


class TestDetectEventType:
    def test_earnings(self):
        assert detect_event_type("NVDA beats Q3 earnings") == "earnings"

    def test_fda(self):
        assert detect_event_type("FDA approves new drug candidate") == "fda"

    def test_merger(self):
        assert detect_event_type("Company announces acquisition deal") == "merger_acquisition"

    def test_macro(self):
        assert detect_event_type("Fed raises interest rate by 25 basis points") == "macro"

    def test_crypto(self):
        assert detect_event_type("Bitcoin surges past $100k") == "crypto"

    def test_general(self):
        assert detect_event_type("Company holds annual meeting") == "general"


class TestCrossStockRelationships:
    def test_nvda_has_related_stocks(self):
        related = get_related_stocks("NVDA")
        symbols = [r["symbol"] for r in related]
        assert "AMD" in symbols
        assert "SMH" in symbols

    def test_reverse_relationships(self):
        # AMD should be related to NVDA even though NVDA's entry lists AMD
        related = get_related_stocks("AMD")
        symbols = [r["symbol"] for r in related]
        assert "NVDA" in symbols

    def test_unknown_symbol(self):
        related = get_related_stocks("ZZZZZ")
        assert related == []

    def test_direction_field_present(self):
        related = get_related_stocks("NVDA")
        for r in related:
            assert r["impact_direction"] in ("same", "inverse")
            assert 0 < r["strength"] <= 1.0

    def test_event_based_impacts(self):
        related = get_related_stocks(
            "NVDA", event_type="tech_sector",
            title="AI spending boom drives massive capex"
        )
        symbols = [r["symbol"] for r in related]
        # Should include AI-related stocks from event mapping
        assert len(related) > 3


class TestPropagateSentiment:
    def test_propagates_to_related(self):
        sentiment = {
            "NVDA": {
                "score": 0.8,
                "magnitude": 0.8,
                "n_articles": 3,
                "bullish_count": 3,
                "bearish_count": 0,
                "headlines": ["NVDA beats earnings by 40%"],
                "news_volume_signal": 1.3,
            }
        }
        known = ["NVDA", "AMD", "AVGO", "SMH", "MU", "AAPL"]
        result = propagate_sentiment(sentiment, known)

        # AMD should have gotten propagated sentiment
        assert "AMD" in result
        assert result["AMD"]["score"] > 0
        assert result["AMD"].get("propagated_from")

    def test_does_not_propagate_weak(self):
        sentiment = {
            "NVDA": {
                "score": 0.05,  # Too weak to propagate
                "magnitude": 0.05,
                "n_articles": 1,
                "bullish_count": 0,
                "bearish_count": 0,
                "headlines": ["NVDA stable"],
                "news_volume_signal": 1.0,
            }
        }
        known = ["NVDA", "AMD"]
        result = propagate_sentiment(sentiment, known)
        # AMD should NOT appear since source score is too weak
        assert "AMD" not in result

    def test_preserves_direct_sentiment(self):
        sentiment = {
            "NVDA": {
                "score": 0.6,
                "magnitude": 0.6,
                "n_articles": 2,
                "bullish_count": 2,
                "bearish_count": 0,
                "headlines": ["NVDA good"],
                "news_volume_signal": 1.0,
            },
            "AMD": {
                "score": -0.3,
                "magnitude": 0.3,
                "n_articles": 1,
                "bullish_count": 0,
                "bearish_count": 1,
                "headlines": ["AMD bad"],
                "news_volume_signal": 1.0,
            },
        }
        known = ["NVDA", "AMD"]
        result = propagate_sentiment(sentiment, known)
        # AMD had direct bearish + propagated bullish — should blend
        assert "AMD" in result

    def test_only_propagates_to_known(self):
        sentiment = {
            "NVDA": {
                "score": 0.8,
                "magnitude": 0.8,
                "n_articles": 3,
                "bullish_count": 3,
                "bearish_count": 0,
                "headlines": ["NVDA good"],
                "news_volume_signal": 1.3,
            }
        }
        # Only NVDA known — shouldn't propagate to AMD
        result = propagate_sentiment(sentiment, ["NVDA"])
        assert "AMD" not in result


class TestNewsSignals:
    def test_generate_from_pre_fetched(self):
        """Test signal generation with pre-fetched news items."""
        items = [
            {
                "symbol": "NVDA",
                "title": "NVDA beats earnings with record revenue, raised guidance",
                "published": datetime.now(),
                "source": "test",
            },
            {
                "symbol": "AAPL",
                "title": "Apple announces massive stock buyback program",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        signals = generate_news_signals(
            ["NVDA", "AAPL", "AMD", "SMH"],
            min_score=0.05,
            news_items=items,
        )
        assert len(signals) > 0
        # Should have signal for NVDA at minimum
        nvda_sigs = [s for s in signals if s["symbol"] == "NVDA"]
        assert len(nvda_sigs) > 0
        assert nvda_sigs[0]["direction"] == "LONG"

    def test_empty_news(self):
        signals = generate_news_signals(
            ["AAPL"], news_items=[]
        )
        assert signals == []

    def test_max_signals_respected(self):
        items = [
            {
                "symbol": f"SYM{i}",
                "title": "Stock beats earnings with record revenue",
                "published": datetime.now(),
                "source": "test",
            }
            for i in range(30)
        ]
        signals = generate_news_signals(
            [f"SYM{i}" for i in range(30)],
            max_signals=5,
            news_items=items,
        )
        assert len(signals) <= 5


class TestBuildNewsFeatures:
    def test_with_signal(self):
        signals = [{
            "symbol": "NVDA",
            "direction": "LONG",
            "score": 0.6,
            "magnitude": 0.5,
            "n_articles": 3,
            "propagated_from": [],
            "event_type": "earnings",
        }]
        feats = build_news_features("NVDA", signals)
        assert feats["news_sentiment"] == 0.6
        assert feats["news_magnitude"] == 0.5
        assert feats["news_is_propagated"] == 0.0
        assert feats["news_event_score"] > 0

    def test_without_signal(self):
        feats = build_news_features("AAPL", [])
        assert feats["news_sentiment"] == 0.0
        assert feats["news_magnitude"] == 0.0


class TestGetNewsBoost:
    def test_bullish_boost(self):
        signals = [{
            "symbol": "NVDA",
            "score": 0.6,
            "magnitude": 0.5,
        }]
        boost = get_news_boost("NVDA", signals)
        assert boost > 0
        assert boost <= 0.15

    def test_bearish_penalty(self):
        signals = [{
            "symbol": "NVDA",
            "score": -0.6,
            "magnitude": 0.5,
        }]
        boost = get_news_boost("NVDA", signals)
        assert boost < 0
        assert boost >= -0.15

    def test_no_signal(self):
        boost = get_news_boost("AAPL", [])
        assert boost == 0.0

    def test_bounded(self):
        signals = [{
            "symbol": "NVDA",
            "score": 1.0,
            "magnitude": 1.0,
        }]
        boost = get_news_boost("NVDA", signals)
        assert -0.15 <= boost <= 0.15


class TestMatchNewsToSymbols:
    def test_matches_ticker_in_title(self):
        items = [{"title": "NVDA surges on AI demand", "symbol": ""}]
        matched = match_news_to_symbols(items, ["NVDA", "AMD"])
        assert any(m["symbol"] == "NVDA" for m in matched)

    def test_matches_company_name(self):
        items = [{"title": "Apple launches new iPhone", "symbol": ""}]
        matched = match_news_to_symbols(items, ["AAPL"])
        assert any(m["symbol"] == "AAPL" for m in matched)

    def test_preserves_existing_symbol(self):
        items = [{"title": "Something", "symbol": "TSLA"}]
        matched = match_news_to_symbols(items, ["TSLA"])
        assert matched[0]["symbol"] == "TSLA"

    def test_keeps_unmatched(self):
        items = [{"title": "Weather forecast for tomorrow", "symbol": ""}]
        matched = match_news_to_symbols(items, ["AAPL"])
        # Unmatched items are kept with empty symbol
        assert len(matched) == 1

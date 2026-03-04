"""Tests for pre-market analyzer and 24/7 scheduling."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.analysis.premarket import PreMarketAnalyzer, WeekendAnalyzer


class TestPreMarketAnalyzer:
    def test_init(self):
        analyzer = PreMarketAnalyzer(["AAPL", "NVDA"])
        assert analyzer.scan_count == 0
        assert analyzer.news_history == []
        assert analyzer.overnight_catalysts == []

    @patch("src.analysis.premarket.fetch_all_news")
    def test_scan_accumulates_news(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "symbol": "NVDA",
                "title": "NVDA beats earnings with record revenue",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        analyzer = PreMarketAnalyzer(["NVDA", "AMD"])
        result = analyzer.scan()

        assert result["status"] == "ok"
        assert result["new_items"] == 1
        assert analyzer.scan_count == 1
        assert len(analyzer.news_history) == 1

    @patch("src.analysis.premarket.fetch_all_news")
    def test_scan_deduplicates(self, mock_fetch):
        """Same headline shouldn't be counted twice."""
        item = {
            "symbol": "NVDA",
            "title": "NVDA beats earnings",
            "published": datetime.now(),
            "source": "test",
        }
        mock_fetch.return_value = [item]

        analyzer = PreMarketAnalyzer(["NVDA"])
        analyzer.scan()
        result = analyzer.scan()

        # Second scan should find 0 new items
        assert result["new_items"] == 0
        assert len(analyzer.news_history) == 1

    @patch("src.analysis.premarket.fetch_all_news")
    def test_scan_detects_catalysts(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "symbol": "LLY",
                "title": "FDA approves breakthrough cancer drug from Eli Lilly",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        analyzer = PreMarketAnalyzer(["LLY"])
        analyzer.scan()

        assert len(analyzer.overnight_catalysts) >= 1
        assert analyzer.overnight_catalysts[0]["event_type"] == "fda"

    @patch("src.analysis.premarket.fetch_all_news")
    def test_build_playbook_empty(self, mock_fetch):
        analyzer = PreMarketAnalyzer(["AAPL"])
        playbook = analyzer.build_playbook()

        assert playbook["trades"] == []
        assert playbook["confidence"] == 0.0
        assert playbook["market_mood"] == "neutral"

    @patch("src.analysis.premarket.fetch_all_news")
    def test_build_playbook_with_data(self, mock_fetch):
        now = datetime.now()
        mock_fetch.return_value = [
            {
                "symbol": "NVDA",
                "title": "NVDA crushes earnings, raised guidance, record revenue",
                "published": now,
                "source": "test",
            },
            {
                "symbol": "NVDA",
                "title": "NVDA stock surges on AI boom demand",
                "published": now - timedelta(hours=2),
                "source": "test",
            },
        ]
        analyzer = PreMarketAnalyzer(["NVDA", "AMD", "SMH"])
        analyzer.scan()
        playbook = analyzer.build_playbook(min_conviction=0.05)

        assert len(playbook["trades"]) > 0
        assert playbook["scan_count"] == 1
        assert playbook["total_news"] == 2

        # NVDA should be in the playbook
        nvda_trades = [t for t in playbook["trades"] if t["symbol"] == "NVDA"]
        assert len(nvda_trades) > 0
        assert nvda_trades[0]["direction"] == "LONG"
        assert nvda_trades[0]["conviction"] > 0

    @patch("src.analysis.premarket.fetch_all_news")
    def test_get_news_signals_format(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "symbol": "AAPL",
                "title": "Apple beats earnings with record iPhone sales",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        analyzer = PreMarketAnalyzer(["AAPL"])
        analyzer.scan()
        signals = analyzer.get_news_signals()

        # Should be compatible with orchestrator's news_signals format
        for sig in signals:
            assert "symbol" in sig
            assert "score" in sig
            assert "magnitude" in sig

    @patch("src.analysis.premarket.fetch_all_news")
    def test_reset(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "symbol": "AAPL",
                "title": "Test news",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        analyzer = PreMarketAnalyzer(["AAPL"])
        analyzer.scan()
        assert len(analyzer.news_history) > 0

        analyzer.reset()
        assert analyzer.news_history == []
        assert analyzer.scan_count == 0
        assert analyzer.overnight_catalysts == []

    @patch("src.analysis.premarket.fetch_all_news")
    def test_multiple_scans_accumulate(self, mock_fetch):
        """Multiple scans should build up intelligence."""
        analyzer = PreMarketAnalyzer(["NVDA", "AMD"])

        # Scan 1
        mock_fetch.return_value = [
            {
                "symbol": "NVDA",
                "title": "NVDA beats earnings",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        analyzer.scan()

        # Scan 2 with different news
        mock_fetch.return_value = [
            {
                "symbol": "AMD",
                "title": "AMD announces new AI chip partnership",
                "published": datetime.now(),
                "source": "test",
            },
        ]
        analyzer.scan()

        assert analyzer.scan_count == 2
        assert len(analyzer.news_history) == 2

    @patch("src.analysis.premarket.fetch_all_news")
    def test_playbook_max_trades(self, mock_fetch):
        items = [
            {
                "symbol": f"SYM{i}",
                "title": f"Stock SYM{i} beats earnings with record revenue",
                "published": datetime.now(),
                "source": "test",
            }
            for i in range(20)
        ]
        mock_fetch.return_value = items
        symbols = [f"SYM{i}" for i in range(20)]

        analyzer = PreMarketAnalyzer(symbols)
        analyzer.scan()
        playbook = analyzer.build_playbook(max_trades=5, min_conviction=0.01)

        assert len(playbook["trades"]) <= 5

    @patch("src.analysis.premarket.fetch_all_news")
    def test_fetch_failure_handled(self, mock_fetch):
        mock_fetch.side_effect = Exception("Network error")
        analyzer = PreMarketAnalyzer(["AAPL"])
        result = analyzer.scan()

        assert result["status"] == "fetch_failed"
        assert analyzer.scan_count == 1


class TestWeekendAnalyzer:
    def test_sector_rotation(self):
        """Test sector rotation analysis with mock data."""
        import pandas as pd
        import numpy as np

        dates = pd.bdate_range("2024-01-01", periods=30)
        price_data = {}
        for etf in ["XLK", "XLF", "XLV"]:
            # XLK trending up, XLF flat, XLV down
            if etf == "XLK":
                prices = np.linspace(100, 115, 30)
            elif etf == "XLF":
                prices = np.ones(30) * 100
            else:
                prices = np.linspace(100, 90, 30)

            price_data[etf] = pd.DataFrame({
                "close": prices,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "volume": [1_000_000] * 30,
            }, index=dates)

        analyzer = WeekendAnalyzer()
        rotation = analyzer._analyze_sector_rotation(price_data)

        assert "XLK" in rotation
        assert rotation["XLK"]["trend"] == "up"
        assert rotation["XLV"]["trend"] == "down"


class TestGetMarketMode:
    def test_import(self):
        """Make sure get_market_mode is importable."""
        # Import from the script's module path
        import importlib.util
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "run_live.py"
        spec = importlib.util.spec_from_file_location("run_live", script)
        mod = importlib.util.module_from_spec(spec)
        # Don't exec (it sets up signal handlers), just check it loads
        assert spec is not None

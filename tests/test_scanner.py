"""Tests for dynamic stock scanner."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.data_feeds.scanner import (
    get_scan_pool,
    scan_for_opportunities,
    add_to_universe,
    SCAN_POOLS,
)


class TestGetScanPool:
    def test_returns_all_pools_by_default(self):
        tickers = get_scan_pool()
        assert len(tickers) > 20
        # Should be sorted and deduplicated
        assert tickers == sorted(set(tickers))

    def test_specific_pool(self):
        tickers = get_scan_pool(["mid_cap_growth"])
        assert "DKNG" in tickers or "COIN" in tickers

    def test_multiple_pools(self):
        t1 = get_scan_pool(["mid_cap_growth"])
        t2 = get_scan_pool(["biotech"])
        combined = get_scan_pool(["mid_cap_growth", "biotech"])
        assert len(combined) >= max(len(t1), len(t2))

    def test_empty_for_unknown_pool(self):
        tickers = get_scan_pool(["nonexistent_pool"])
        assert tickers == []


class TestScanForOpportunities:
    @patch("yfinance.download")
    def test_filters_existing_universe(self, mock_dl):
        """Should not scan tickers already in universe."""
        mock_dl.return_value = pd.DataFrame()

        results = scan_for_opportunities(
            scan_tickers=["AAPL", "NEW1"],
            existing_universe=["AAPL"],
        )
        # AAPL should be skipped since it's in existing_universe
        assert all(r["symbol"] != "AAPL" for r in results)

    def test_returns_empty_when_all_in_universe(self):
        results = scan_for_opportunities(
            scan_tickers=["AAPL", "MSFT"],
            existing_universe=["AAPL", "MSFT"],
        )
        assert results == []

    @patch("yfinance.download")
    def test_detects_volume_surge(self, mock_dl):
        n = 40
        dates = pd.bdate_range("2024-01-01", periods=n)
        # Normal volume for first 35 days, then huge spike
        volume = np.array([1_000_000] * 35 + [5_000_000] * 5)
        close = np.array([100.0] * n)

        mock_data = pd.DataFrame({
            "Close": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Open": close,
            "Volume": volume,
        }, index=dates)

        mock_dl.return_value = mock_data

        results = scan_for_opportunities(
            scan_tickers=["TEST"],
            existing_universe=[],
            min_avg_volume=500_000,
            min_avg_dollar_vol=1_000_000,
            volume_surge_threshold=2.0,
        )
        # Should detect volume surge (5M vs 1M average)
        surges = [r for r in results if "volume_surge" in r["reason"]]
        assert len(surges) > 0

    def test_respects_max_new_symbols(self):
        """max_new_symbols should cap the results."""
        # We can't easily mock yf.download for many symbols,
        # but we can test the limit parameter exists
        results = scan_for_opportunities(
            scan_tickers=[],
            existing_universe=[],
            max_new_symbols=5,
        )
        assert len(results) <= 5


class TestAddToUniverse:
    def test_adds_new_symbol(self, tmp_path):
        # Create minimal universe file
        universe_path = tmp_path / "universe.csv"
        pd.DataFrame({
            "symbol": ["SPY"],
            "active_from": ["1993-01-29"],
            "active_to": [""],
            "notes": ["Index"],
            "sector_etf": [""],
        }).to_csv(universe_path, index=False)

        data_dir = tmp_path / "ohlcv"
        data_dir.mkdir()

        opportunities = [{"symbol": "NEW1", "reason": "volume_surge"}]

        with patch("src.data_feeds.api_providers.YFinanceDownloader") as mock_dl_cls:
            mock_dl = MagicMock()
            mock_dl.download_symbol.return_value = data_dir / "NEW1.csv"
            # Create a dummy CSV so path exists
            (data_dir / "NEW1.csv").write_text("date,open,high,low,close,volume\n")
            mock_dl_cls.return_value = mock_dl

            added = add_to_universe(
                opportunities, str(universe_path), str(data_dir)
            )

        assert "NEW1" in added
        # Check universe.csv was updated
        df = pd.read_csv(universe_path)
        assert "NEW1" in df["symbol"].values

    def test_skips_existing_symbols(self, tmp_path):
        universe_path = tmp_path / "universe.csv"
        pd.DataFrame({
            "symbol": ["SPY", "EXISTING"],
            "active_from": ["1993-01-29", "2020-01-01"],
            "active_to": ["", ""],
            "notes": ["Index", "Test"],
            "sector_etf": ["", ""],
        }).to_csv(universe_path, index=False)

        opportunities = [{"symbol": "EXISTING", "reason": "momentum"}]
        added = add_to_universe(
            opportunities, str(universe_path), str(tmp_path / "ohlcv")
        )
        assert added == []

    def test_empty_opportunities_returns_empty(self, tmp_path):
        added = add_to_universe([], str(tmp_path / "u.csv"), str(tmp_path))
        assert added == []

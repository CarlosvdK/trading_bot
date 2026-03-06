"""Tests for YFinanceDownloader — unit tests with mocked yfinance."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.api_providers import YFinanceDownloader


class TestYFinanceDownloader:
    def test_init_creates_output_dir(self, tmp_path):
        dl = YFinanceDownloader(str(tmp_path / "new_dir"))
        assert dl.output_dir.exists()

    @patch("yfinance.Ticker")
    def test_download_symbol_saves_csv(self, mock_ticker_cls, tmp_path):
        dates = pd.bdate_range("2024-01-02", periods=5)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [98, 99, 100, 101, 102],
                "Close": [103, 104, 105, 106, 107],
                "Volume": [1e6, 1.1e6, 0.9e6, 1.2e6, 1e6],
            },
            index=dates,
        )
        mock_ticker_cls.return_value = mock_ticker

        dl = YFinanceDownloader(str(tmp_path))
        path = dl.download_symbol("AAPL", "2024-01-01", "2024-01-10")

        assert path is not None
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) == 5

    @patch("yfinance.Ticker")
    def test_download_symbol_returns_none_for_empty(self, mock_ticker_cls, tmp_path):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        dl = YFinanceDownloader(str(tmp_path))
        path = dl.download_symbol("FAKE", "2024-01-01")
        assert path is None

    @patch("yfinance.download")
    def test_download_bulk(self, mock_download, tmp_path):
        dates = pd.bdate_range("2024-01-02", periods=5)
        # Simulate multi-ticker download with MultiIndex columns
        data = {}
        for sym in ["AAPL", "MSFT"]:
            data[(sym, "Open")] = [100 + i for i in range(5)]
            data[(sym, "High")] = [105 + i for i in range(5)]
            data[(sym, "Low")] = [98 + i for i in range(5)]
            data[(sym, "Close")] = [103 + i for i in range(5)]
            data[(sym, "Volume")] = [1e6] * 5

        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        mock_download.return_value = df

        dl = YFinanceDownloader(str(tmp_path))
        results = dl.download_bulk(["AAPL", "MSFT"], "2024-01-01")

        assert results["AAPL"] is not None
        assert results["MSFT"] is not None
        assert (tmp_path / "AAPL.csv").exists()
        assert (tmp_path / "MSFT.csv").exists()

    def test_download_universe_delegates_to_bulk(self, tmp_path):
        dl = YFinanceDownloader(str(tmp_path))
        # Mock download_bulk
        dl.download_bulk = MagicMock(return_value={"AAPL": tmp_path / "AAPL.csv"})
        results = dl.download_universe(["AAPL"], "2024-01-01")
        dl.download_bulk.assert_called_once()
        assert "AAPL" in results

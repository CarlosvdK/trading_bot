"""
Tests for API data providers — unit tests (no real API calls).
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.api_providers import (
    PolygonDownloader,
    AlphaVantageDownloader,
    StooqDownloader,
    DataDownloader,
)


class TestPolygonDownloader:
    def test_init_with_key(self, tmp_path):
        dl = PolygonDownloader(str(tmp_path), api_key="test_key")
        assert dl.api_key == "test_key"
        assert dl.output_dir.exists()

    @patch("src.data.api_providers.requests.get")
    def test_download_symbol_parses_response(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "resultsCount": 3,
            "results": [
                {"t": 1704067200000, "o": 100, "h": 105, "l": 98, "c": 103, "v": 1000000},
                {"t": 1704153600000, "o": 103, "h": 107, "l": 101, "c": 106, "v": 1200000},
                {"t": 1704240000000, "o": 106, "h": 108, "l": 104, "c": 107, "v": 900000},
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        dl = PolygonDownloader(str(tmp_path), api_key="test")
        path = dl.download_symbol("AAPL", "2024-01-01", "2024-01-03")
        assert path is not None
        assert path.exists()

        df = pd.read_csv(path)
        assert len(df) == 3
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    @patch("src.data.api_providers.requests.get")
    def test_empty_response_returns_none(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"resultsCount": 0, "results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        dl = PolygonDownloader(str(tmp_path), api_key="test")
        path = dl.download_symbol("FAKE")
        assert path is None


class TestAlphaVantageDownloader:
    def test_init(self, tmp_path):
        dl = AlphaVantageDownloader(str(tmp_path), api_key="test_av")
        assert dl.api_key == "test_av"

    @patch("src.data.api_providers.requests.get")
    def test_rate_limit_raises(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Note": "Thank you for using..."}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        dl = AlphaVantageDownloader(str(tmp_path), api_key="test")
        with pytest.raises(RuntimeError, match="rate limit"):
            dl.download_symbol("AAPL")

    @patch("src.data.api_providers.requests.get")
    def test_error_message_raises(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Error Message": "Invalid symbol"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        dl = AlphaVantageDownloader(str(tmp_path), api_key="test")
        with pytest.raises(ValueError, match="Invalid symbol"):
            dl.download_symbol("XXXX")


class TestStooqDownloader:
    def test_init_no_key_needed(self, tmp_path):
        dl = StooqDownloader(str(tmp_path))
        assert dl.output_dir.exists()

    @patch("src.data.api_providers.requests.get")
    def test_download_parses_csv(self, mock_get, tmp_path):
        csv_content = (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,100,105,98,103,1000000\n"
            "2024-01-03,103,107,101,106,1200000\n"
        )
        mock_resp = MagicMock()
        mock_resp.text = csv_content
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        dl = StooqDownloader(str(tmp_path))
        path = dl.download_symbol("AAPL", "2024-01-01", "2024-01-05")
        assert path is not None

        df = pd.read_csv(path)
        assert len(df) == 2
        assert "close" in df.columns


class TestDataDownloader:
    @patch("src.data.api_providers.get_secret")
    def test_builds_source_chain(self, mock_secret, tmp_path):
        mock_secret.side_effect = lambda k, **kw: (
            "fake_polygon_key"
            if k == "POLYGON_API_KEY"
            else "fake_av_key"
            if k == "ALPHA_VANTAGE_API_KEY"
            else None
        )
        dl = DataDownloader(str(tmp_path))
        source_names = [name for name, _ in dl.sources]
        assert "polygon" in source_names
        assert "stooq" in source_names
        assert "alphavantage" in source_names

    @patch("src.data.api_providers.get_secret")
    def test_stooq_always_present(self, mock_secret, tmp_path):
        mock_secret.return_value = None
        dl = DataDownloader(str(tmp_path))
        source_names = [name for name, _ in dl.sources]
        assert "stooq" in source_names

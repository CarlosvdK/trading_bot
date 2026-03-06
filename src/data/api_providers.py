"""
API-based data providers: yfinance, Polygon, Alpha Vantage, Stooq.
All providers download OHLCV → save to CSV → load via CSVDataProvider.
This ensures backtests always run from local files (reproducibility).
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from src.utils.secrets import get_secret

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polygon.io provider (full history, requires paid plan for >2y)
# ---------------------------------------------------------------------------

class PolygonDownloader:
    """
    Downloads daily OHLCV from Polygon.io REST API.
    Saves to CSV files compatible with CSVDataProvider.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, output_dir: str, api_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or get_secret("POLYGON_API_KEY")

    def download_symbol(
        self,
        symbol: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> Path:
        """
        Download daily adjusted OHLCV for a symbol.
        Returns path to saved CSV file.
        """
        if end_date is None:
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start_date}/{end_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        all_results = []
        while url:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("resultsCount", 0) == 0:
                break

            all_results.extend(data.get("results", []))

            # Pagination
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": self.api_key}
            else:
                url = None

            # Rate limit: 5 calls/min on free tier
            time.sleep(0.25)

        if not all_results:
            logger.warning(f"No data returned for {symbol} from Polygon")
            return None

        df = pd.DataFrame(all_results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").drop_duplicates(subset="date")

        out_path = self.output_dir / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        logger.info(
            f"Polygon: saved {len(df)} bars for {symbol} → {out_path}"
        )
        return out_path

    def download_universe(
        self,
        symbols: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        delay_between: float = 0.5,
    ) -> dict:
        """Download all symbols in universe. Returns {symbol: path_or_None}."""
        results = {}
        for i, sym in enumerate(symbols):
            logger.info(f"Downloading {sym} ({i+1}/{len(symbols)})...")
            try:
                path = self.download_symbol(sym, start_date, end_date)
                results[sym] = path
            except Exception as e:
                logger.error(f"Failed to download {sym}: {e}")
                results[sym] = None
            time.sleep(delay_between)
        return results


# ---------------------------------------------------------------------------
# Alpha Vantage provider (free tier: 25 calls/day)
# ---------------------------------------------------------------------------

class AlphaVantageDownloader:
    """
    Downloads daily OHLCV from Alpha Vantage.
    Free tier is severely rate-limited (25/day). Use for supplementary data.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, output_dir: str, api_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or get_secret("ALPHA_VANTAGE_API_KEY")

    def download_symbol(
        self,
        symbol: str,
        outputsize: str = "full",
    ) -> Path:
        """
        Download daily adjusted OHLCV.
        outputsize: "compact" (100 days) or "full" (20+ years).
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "datatype": "json",
        }

        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(
                f"Alpha Vantage rate limit hit: {data['Note']}"
            )

        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            logger.warning(f"No time series data for {symbol}")
            return None

        records = []
        for date_str, bar in data[ts_key].items():
            records.append({
                "date": date_str,
                "open": float(bar["1. open"]),
                "high": float(bar["2. high"]),
                "low": float(bar["3. low"]),
                "close": float(bar["5. adjusted close"]),
                "volume": int(bar["6. volume"]),
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        out_path = self.output_dir / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        logger.info(
            f"AlphaVantage: saved {len(df)} bars for {symbol} → {out_path}"
        )
        return out_path

    def download_universe(
        self,
        symbols: List[str],
        delay_between: float = 15.0,  # AV free tier: ~5 calls/min
    ) -> dict:
        results = {}
        for i, sym in enumerate(symbols):
            logger.info(f"Downloading {sym} ({i+1}/{len(symbols)})...")
            try:
                path = self.download_symbol(sym)
                results[sym] = path
            except Exception as e:
                logger.error(f"Failed to download {sym}: {e}")
                results[sym] = None
            if i < len(symbols) - 1:
                time.sleep(delay_between)
        return results


# ---------------------------------------------------------------------------
# Stooq.com provider (free CSV endpoint, no API key needed)
# ---------------------------------------------------------------------------

class StooqDownloader:
    """
    Downloads daily OHLCV from Stooq.com CSV endpoints.
    Free, no API key required. Good for US equities and indices.
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_symbol(
        self,
        symbol: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> Path:
        """
        Download daily OHLCV from Stooq.
        Stooq uses .us suffix for US stocks (e.g., AAPL.US).
        """
        stooq_symbol = f"{symbol}.us"
        sd = pd.Timestamp(start_date)
        ed = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()

        params = {
            "s": stooq_symbol,
            "d1": sd.strftime("%Y%m%d"),
            "d2": ed.strftime("%Y%m%d"),
            "i": "d",  # daily
        }

        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        # Stooq returns CSV directly
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        if df.empty or "Close" not in df.columns:
            # Try without .us suffix (indices like SPY)
            params["s"] = symbol
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))

        if df.empty or len(df) < 2:
            logger.warning(f"No Stooq data for {symbol}")
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])

        # Ensure standard column names
        col_map = {}
        for target in ["open", "high", "low", "close", "volume"]:
            for col in df.columns:
                if col == target:
                    col_map[col] = target
                    break
        df = df.rename(columns=col_map)

        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").drop_duplicates(subset="date")

        out_path = self.output_dir / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        logger.info(
            f"Stooq: saved {len(df)} bars for {symbol} → {out_path}"
        )
        return out_path

    def download_universe(
        self,
        symbols: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        delay_between: float = 1.0,
    ) -> dict:
        results = {}
        for i, sym in enumerate(symbols):
            logger.info(f"Downloading {sym} ({i+1}/{len(symbols)})...")
            try:
                path = self.download_symbol(sym, start_date, end_date)
                results[sym] = path
            except Exception as e:
                logger.error(f"Failed to download {sym}: {e}")
                results[sym] = None
            if i < len(symbols) - 1:
                time.sleep(delay_between)
        return results


# ---------------------------------------------------------------------------
# Yahoo Finance provider (free, no API key, bulk-capable)
# ---------------------------------------------------------------------------

class YFinanceDownloader:
    """
    Downloads daily OHLCV from Yahoo Finance via yfinance.
    Free, no API key required. Supports efficient bulk downloads.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_symbol(
        self,
        symbol: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> Optional[Path]:
        """Download daily adjusted OHLCV for a single symbol."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date or None,
            auto_adjust=True,
        )

        if df.empty or len(df) < 2:
            logger.warning(f"yfinance: no data for {symbol}")
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df = df.sort_index().reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        out_path = self.output_dir / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"yfinance: saved {len(df)} bars for {symbol}")
        return out_path

    def download_bulk(
        self,
        symbols: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        batch_size: int = 50,
    ) -> dict:
        """
        Bulk download using yf.download() for efficiency.
        Downloads in batches to avoid timeouts on large universes.
        """
        import yfinance as yf

        results = {}
        total = len(symbols)

        for i in range(0, total, batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(
                f"yfinance batch {batch_num}/{total_batches}: "
                f"symbols {i + 1}-{min(i + batch_size, total)}/{total}"
            )

            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )

                for sym in batch:
                    try:
                        if len(batch) == 1:
                            sym_df = data.copy()
                            # yfinance may return MultiIndex columns for single ticker
                            if isinstance(sym_df.columns, pd.MultiIndex):
                                sym_df = sym_df.droplevel("Ticker", axis=1)
                        else:
                            sym_df = data[sym]

                        sym_df = sym_df.dropna(how="all")
                        if sym_df.empty or len(sym_df) < 2:
                            results[sym] = None
                            continue

                        sym_df.columns = [
                            c.lower() if isinstance(c, str) else str(c).lower()
                            for c in sym_df.columns
                        ]
                        sym_df = sym_df[["open", "high", "low", "close", "volume"]]
                        sym_df.index.name = "date"
                        sym_df = sym_df.reset_index()
                        sym_df["date"] = pd.to_datetime(
                            sym_df["date"]
                        ).dt.tz_localize(None)

                        out_path = self.output_dir / f"{sym}.csv"
                        sym_df.to_csv(out_path, index=False)
                        results[sym] = out_path
                    except Exception as e:
                        logger.warning(f"yfinance: failed to extract {sym}: {e}")
                        results[sym] = None
            except Exception as e:
                logger.error(f"yfinance bulk batch failed: {e}")
                for sym in batch:
                    results[sym] = None

            time.sleep(1)

        succeeded = sum(1 for v in results.values() if v)
        logger.info(f"yfinance bulk: {succeeded}/{total} symbols downloaded")
        return results

    def download_universe(
        self,
        symbols: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Use bulk download for efficiency."""
        return self.download_bulk(symbols, start_date, end_date)


# ---------------------------------------------------------------------------
# Multi-source downloader with fallback chain
# ---------------------------------------------------------------------------

class DataDownloader:
    """
    Downloads OHLCV data using a priority chain of sources.
    Default chain: yfinance → Polygon → Stooq → Alpha Vantage.
    Saves all data as CSV to a single output directory.
    """

    def __init__(self, output_dir: str, config: dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = config or {}

        self.sources = []

        # yfinance (free, no key, bulk-capable — try first)
        try:
            import yfinance as _yf  # noqa: F401
            self.sources.append(
                ("yfinance", YFinanceDownloader(output_dir))
            )
        except ImportError:
            logger.info("yfinance not installed — skipping as data source")

        # Polygon (best quality if key available)
        polygon_key = get_secret("POLYGON_API_KEY", required=False)
        if polygon_key:
            self.sources.append(
                ("polygon", PolygonDownloader(output_dir, polygon_key))
            )

        # Stooq (free, no key needed)
        self.sources.append(
            ("stooq", StooqDownloader(output_dir))
        )

        # Alpha Vantage (rate-limited fallback)
        av_key = get_secret("ALPHA_VANTAGE_API_KEY", required=False)
        if av_key:
            self.sources.append(
                ("alphavantage", AlphaVantageDownloader(output_dir, av_key))
            )

    def download_symbol(
        self,
        symbol: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> Optional[Path]:
        """Try each source in priority order until one succeeds."""
        path = self._download_with_fallback(symbol, start_date, end_date)
        if path is None:
            logger.error(f"{symbol}: all sources failed")
        return path

    def download_universe(
        self,
        symbols: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> dict:
        """Download all symbols. Uses bulk yfinance first, then fallback for failures."""
        results = {}

        # If yfinance available, try bulk download first (much faster)
        if self.sources and self.sources[0][0] == "yfinance":
            yf_dl = self.sources[0][1]
            results = yf_dl.download_bulk(symbols, start_date, end_date)

            # Fall through to other sources for failures
            failed = [s for s, p in results.items() if p is None]
            if failed:
                logger.info(
                    f"yfinance missed {len(failed)} symbols, trying fallbacks..."
                )
                for sym in failed:
                    path = self._download_with_fallback(
                        sym, start_date, end_date, skip_sources={"yfinance"}
                    )
                    results[sym] = path
        else:
            # Sequential fallback for each symbol
            for i, sym in enumerate(symbols):
                logger.info(f"[{i + 1}/{len(symbols)}] {sym}...")
                results[sym] = self.download_symbol(sym, start_date, end_date)
                time.sleep(0.3)

        succeeded = sum(1 for v in results.values() if v)
        logger.info(
            f"Download complete: {succeeded}/{len(symbols)} symbols succeeded"
        )
        return results

    def _download_with_fallback(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str],
        skip_sources: set = None,
    ) -> Optional[Path]:
        """Try each source except skipped ones."""
        skip_sources = skip_sources or set()
        for name, downloader in self.sources:
            if name in skip_sources:
                continue
            try:
                if isinstance(downloader, AlphaVantageDownloader):
                    path = downloader.download_symbol(symbol)
                else:
                    path = downloader.download_symbol(
                        symbol, start_date, end_date
                    )
                if path and path.exists():
                    logger.info(f"{symbol}: downloaded from {name}")
                    return path
            except Exception as e:
                logger.warning(f"{symbol}: {name} failed — {e}")
        return None

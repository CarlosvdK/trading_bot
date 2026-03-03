"""
Data Provider interface and CSV implementation.
Skill reference: .claude/skills/data-layer/SKILL.md
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import warnings

import pandas as pd


class DataProvider(ABC):
    """
    Abstract interface for all data sources.
    Implement this to swap between CSV, database, API providers.
    """

    @abstractmethod
    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns: open, high, low, close, volume
        Indexed by date (DatetimeIndex, no timezone).
        Prices must be split/dividend adjusted.
        """
        pass

    @abstractmethod
    def available_symbols(self) -> List[str]:
        """Return list of all available symbols."""
        pass

    @abstractmethod
    def get_universe(self, date: str) -> List[str]:
        """
        Return point-in-time universe of active symbols on a given date.
        This is the ONLY correct way to avoid survivorship bias.
        """
        pass


class CSVDataProvider(DataProvider):
    """Default data provider — loads OHLCV from local CSV files."""

    def __init__(
        self,
        data_dir: str,
        universe_file: Optional[str] = None,
        validate_on_load: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.universe_file = universe_file
        self.validate_on_load = validate_on_load
        self._cache: Dict[str, pd.DataFrame] = {}
        self._hash_registry_path = self.data_dir / ".data_hashes.json"
        self._hash_registry = self._load_hash_registry()
        self._universe: Optional[pd.DataFrame] = None

        if universe_file and Path(universe_file).exists():
            self._universe = pd.read_csv(
                universe_file, parse_dates=["active_from", "active_to"]
            )

    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if symbol in self._cache:
            df = self._cache[symbol]
        else:
            path = self.data_dir / f"{symbol}.csv"
            if not path.exists():
                raise FileNotFoundError(f"No data file for {symbol}: {path}")

            self._check_hash(symbol, path)

            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            df.index = pd.DatetimeIndex(df.index)
            df.columns = [c.lower() for c in df.columns]

            # Use adj_close as close if available
            if "adj_close" in df.columns:
                df["close"] = df["adj_close"]

            df = df.sort_index()

            if self.validate_on_load:
                from src.data.validator import validate_ohlcv

                result = validate_ohlcv(df, symbol)
                if not result.passed:
                    raise ValueError(
                        f"Data validation failed for {symbol}: "
                        + "; ".join(result.errors)
                    )

            self._cache[symbol] = df

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df.copy()

    def available_symbols(self) -> List[str]:
        return [
            f.stem
            for f in self.data_dir.glob("*.csv")
            if not f.stem.startswith(".")
        ]

    def get_universe(self, date: str) -> List[str]:
        """Point-in-time universe. Falls back to all available if no universe file."""
        if self._universe is None:
            warnings.warn(
                "No universe file — survivorship bias not controlled. "
                "Backtest returns may be overstated by 20-40%.",
                UserWarning,
            )
            return self.available_symbols()

        ts = pd.Timestamp(date)
        active = self._universe[
            (self._universe["active_from"] <= ts)
            & (
                self._universe["active_to"].isna()
                | (self._universe["active_to"] >= ts)
            )
        ]
        return active["symbol"].tolist()

    def _check_hash(self, symbol: str, path: Path):
        current_hash = self._compute_hash(path)
        stored = self._hash_registry.get(symbol)
        if stored is None:
            self._hash_registry[symbol] = current_hash
            self._save_hash_registry()
        elif stored != current_hash:
            warnings.warn(
                f"Data file for {symbol} has changed since last load. "
                f"Verify no corruption or unexpected update.",
                UserWarning,
            )
            self._hash_registry[symbol] = current_hash
            self._save_hash_registry()

    def _compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _load_hash_registry(self) -> dict:
        if self._hash_registry_path.exists():
            return json.loads(self._hash_registry_path.read_text())
        return {}

    def _save_hash_registry(self):
        self._hash_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._hash_registry_path.write_text(
            json.dumps(self._hash_registry, indent=2)
        )

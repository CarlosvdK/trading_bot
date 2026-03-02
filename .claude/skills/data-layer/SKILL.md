# Skill: Data Layer, Validation & Corporate Actions

## What This Skill Is
The data layer is the foundation of everything. Incorrect, unadjusted, or leaky data produces phantom signals and wrong backtest results. This skill covers: CSV data loading, validation, corporate action handling, survivorship bias, and the abstract data provider interface.

---

## Data Provider Interface

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional

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
        Returns DataFrame with columns: open, high, low, close, volume, adj_close
        Indexed by date (DatetimeIndex, no timezone or UTC-normalized).
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
```

---

## CSV Data Provider

### Expected File Format

```
# data/AAPL.csv
date,open,high,low,close,volume,adj_close
2020-01-02,296.24,300.60,295.19,300.35,33870100,298.83
2020-01-03,297.15,300.58,296.50,297.43,36580700,295.94
...
```

**Rules:**
- Dates in ISO format: `YYYY-MM-DD`
- All prices are adjusted for splits and dividends via `adj_close`
- If only one price series, use adjusted prices and name column `close`
- Volume in shares (not notional)
- No timezone suffix on dates

```python
import pandas as pd
import numpy as np
import hashlib
import json
import os
from pathlib import Path

class CSVDataProvider(DataProvider):
    """Default data provider — loads OHLCV from local CSV files."""

    def __init__(self, data_dir: str, universe_file: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.universe_file = universe_file
        self._cache: Dict[str, pd.DataFrame] = {}
        self._hash_registry_path = self.data_dir / ".data_hashes.json"
        self._hash_registry = self._load_hash_registry()
        self._universe: Optional[pd.DataFrame] = None
        if universe_file:
            self._universe = pd.read_csv(universe_file, parse_dates=["active_from", "active_to"])

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

            # Hash check — warn if file changed unexpectedly
            self._check_hash(symbol, path)

            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            df.index = pd.DatetimeIndex(df.index)
            df.columns = [c.lower() for c in df.columns]

            # Use adj_close as close if available
            if "adj_close" in df.columns:
                df["close"] = df["adj_close"]

            # Sort ascending
            df = df.sort_index()
            self._cache[symbol] = df

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df.copy()

    def available_symbols(self) -> List[str]:
        return [f.stem for f in self.data_dir.glob("*.csv") if not f.stem.startswith(".")]

    def get_universe(self, date: str) -> List[str]:
        """Point-in-time universe. Falls back to all available if no universe file."""
        if self._universe is None:
            # WARNING: survivorship bias! Log it.
            return self.available_symbols()

        ts = pd.Timestamp(date)
        active = self._universe[
            (self._universe["active_from"] <= ts) &
            (self._universe["active_to"].isna() | (self._universe["active_to"] >= ts))
        ]
        return active["symbol"].tolist()

    def _check_hash(self, symbol: str, path: Path):
        current_hash = self._compute_hash(path)
        stored = self._hash_registry.get(symbol)
        if stored is None:
            self._hash_registry[symbol] = current_hash
            self._save_hash_registry()
        elif stored != current_hash:
            import warnings
            warnings.warn(
                f"Data file for {symbol} has changed since last load. "
                f"Verify no corruption or unexpected update.",
                UserWarning
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
        self._hash_registry_path.write_text(json.dumps(self._hash_registry, indent=2))
```

---

## Data Validator

Run on every data load. Fail loudly — silent data issues cause silent P&L errors.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ValidationResult:
    symbol: str
    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame, symbol: str, max_gap_days: int = 5) -> ValidationResult:
    """
    Validate a single symbol's OHLCV DataFrame.
    Returns ValidationResult with all issues found.
    """
    result = ValidationResult(symbol=symbol, passed=True)

    # 1. Required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        result.errors.append(f"Missing columns: {missing}")
        result.passed = False

    if not result.passed:
        return result  # Can't continue without required columns

    # 2. Monotonic index (no duplicate/out-of-order dates)
    if not df.index.is_monotonic_increasing:
        result.errors.append("Dates are not monotonically increasing")
        result.passed = False

    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        result.errors.append(f"{n_dups} duplicate date(s) found")
        result.passed = False

    # 3. No future dates
    today = pd.Timestamp.today().normalize()
    future = df[df.index > today]
    if len(future) > 0:
        result.errors.append(f"{len(future)} rows have future dates")
        result.passed = False

    # 4. OHLCV internal consistency
    if (df["high"] < df["low"]).any():
        n = (df["high"] < df["low"]).sum()
        result.errors.append(f"{n} rows where High < Low")
        result.passed = False

    if (df["high"] < df["close"]).any():
        n = (df["high"] < df["close"]).sum()
        result.warnings.append(f"{n} rows where High < Close (check adjustment)")

    if (df["low"] > df["close"]).any():
        n = (df["low"] > df["close"]).sum()
        result.warnings.append(f"{n} rows where Low > Close (check adjustment)")

    # 5. Zero/negative prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            n = (df[col] <= 0).sum()
            result.errors.append(f"{n} zero/negative values in {col}")
            result.passed = False

    # 6. Zero volume
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        result.warnings.append(f"{zero_vol} rows with zero volume (holidays/halts?)")

    # 7. Large gaps (missing trading days)
    date_diffs = df.index.to_series().diff().dt.days.dropna()
    large_gaps = date_diffs[date_diffs > max_gap_days]
    if len(large_gaps) > 0:
        result.warnings.append(
            f"{len(large_gaps)} gaps > {max_gap_days} days. "
            f"Largest: {large_gaps.max()} days at {large_gaps.idxmax().date()}"
        )

    # 8. Extreme price changes (possible split not adjusted)
    daily_ret = df["close"].pct_change().abs()
    extreme = daily_ret[daily_ret > 0.50]
    if len(extreme) > 0:
        result.warnings.append(
            f"{len(extreme)} days with >50% price move. "
            f"Verify split/dividend adjustments. Dates: {extreme.index.tolist()[:5]}"
        )

    # 9. Data freshness
    latest = df.index[-1]
    staleness = (today - latest).days
    if staleness > 3:
        result.warnings.append(f"Data is {staleness} days stale (latest: {latest.date()})")

    return result


def validate_all_symbols(
    provider: DataProvider,
    symbols: List[str],
) -> pd.DataFrame:
    """Validate all symbols, return summary DataFrame."""
    results = []
    for sym in symbols:
        try:
            df = provider.load_symbol(sym)
            r = validate_ohlcv(df, sym)
        except Exception as e:
            r = ValidationResult(sym, False, errors=[str(e)])
        results.append({
            "symbol": r.symbol,
            "passed": r.passed,
            "n_warnings": len(r.warnings),
            "n_errors": len(r.errors),
            "warnings": "; ".join(r.warnings),
            "errors": "; ".join(r.errors),
        })

    df = pd.DataFrame(results)
    failed = df[~df["passed"]]
    if len(failed) > 0:
        print(f"\nWARNING: {len(failed)} symbols failed validation:")
        print(failed[["symbol", "errors"]].to_string())
    else:
        print(f"All {len(symbols)} symbols passed validation.")

    return df
```

---

## Corporate Actions Handling

### Corporate Action Types & Required Handling

| Action | Effect on Raw Data | Required Fix |
|---|---|---|
| Stock split (2:1) | Price halves, volume doubles | Multiply all pre-split prices by 0.5 |
| Reverse split (1:10) | Price x10 | Multiply all pre-split prices by 10 |
| Cash dividend | Ex-div price drop | Subtract dividend from all pre-ex prices |
| Spin-off | Parent price drops | Adjust parent; create child symbol |
| Merger / acquisition | Symbol delisted | Mark symbol as inactive in universe file |
| Symbol change | Ticker change | Map old to new in universe file |

### Corporate Actions File Format

```
# data/corporate_actions.csv
symbol,date,action_type,adjustment_factor,notes
AAPL,2020-08-31,split,0.25,4:1 split (multiply pre-split prices by 0.25)
TSLA,2020-08-31,split,0.2,5:1 split
GME,2022-07-22,split,0.25,4:1 split
```

```python
def apply_corporate_actions(
    df: pd.DataFrame,
    symbol: str,
    actions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply corporate action adjustments to raw OHLCV data.
    Modifies prices for all dates before each action.
    """
    symbol_actions = actions_df[
        (actions_df["symbol"] == symbol) &
        (actions_df["action_type"].isin(["split", "reverse_split"]))
    ].sort_values("date")

    df = df.copy()

    for _, action in symbol_actions.iterrows():
        action_date = pd.Timestamp(action["date"])
        factor = float(action["adjustment_factor"])

        # Adjust all prices BEFORE the action date
        mask = df.index < action_date
        price_cols = ["open", "high", "low", "close"]
        df.loc[mask, price_cols] *= factor
        # Adjust volume inversely
        df.loc[mask, "volume"] /= factor

    return df
```

---

## Survivorship Bias — Point-in-Time Universe

```
# data/universe_template.csv
# This file defines which symbols were ACTIVE at each point in time.
# Without this, your backtest only includes survivors — companies that
# didn't go bankrupt, get acquired, or get delisted.
# Backtest returns are inflated by 20-40% without this file.

symbol,active_from,active_to,notes
AAPL,1990-01-01,,           # Still active (blank = still active)
LEHM,1994-01-01,2008-09-15, # Lehman Brothers — delisted 2008
ENRN,1996-01-01,2001-12-02, # Enron — delisted 2001
```

**If you don't have a universe file**, add this warning to all backtest outputs:

```python
SURVIVORSHIP_BIAS_WARNING = """
WARNING: Survivorship Bias Not Controlled
This backtest uses only currently-available symbols.
Companies that were delisted, acquired, or went bankrupt
are not included. Backtest returns may be overstated by
20-40% compared to a live trading scenario.
"""
```

---

## Missing Data Policy

```python
def handle_missing_data(
    df: pd.DataFrame,
    max_ffill_days: int = 1,
    max_missing_pct: float = 0.05,
) -> Optional[pd.DataFrame]:
    """
    Forward-fill missing bars up to max_ffill_days.
    Returns None if symbol has too much missing data.
    """
    total_bars = len(df)
    missing = df["close"].isna().sum()
    missing_pct = missing / total_bars

    if missing_pct > max_missing_pct:
        print(f"Symbol dropped: {missing_pct:.1%} missing data (>{max_missing_pct:.1%} threshold)")
        return None

    # Forward-fill only (never interpolate OHLCV)
    df = df.fillna(method="ffill", limit=max_ffill_days)
    return df
```

---

## Timezone Convention

```python
# All internal timestamps: date-only (no time component) for daily data
# Display: configurable via config['display_timezone']

import pytz

def normalize_timestamps(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Normalize index to date-only for daily data."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(tz).normalize()
    else:
        df.index = pd.DatetimeIndex(df.index.date)
    return df
```

---

## Configuration

```yaml
data:
  data_dir: "data/ohlcv"
  universe_file: "data/universe.csv"           # Optional but strongly recommended
  corporate_actions_file: "data/corporate_actions.csv"
  max_ffill_days: 1
  max_missing_pct: 0.05
  max_staleness_days: 3
  validate_on_load: true
  timezone: "America/New_York"
```

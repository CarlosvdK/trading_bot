"""
OHLCV data validation.
Skill reference: .claude/skills/data-layer/SKILL.md
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class ValidationResult:
    symbol: str
    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def validate_ohlcv(
    df: pd.DataFrame, symbol: str, max_gap_days: int = 5
) -> ValidationResult:
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
        return result

    # 2. Empty DataFrame
    if len(df) == 0:
        result.errors.append("DataFrame is empty")
        result.passed = False
        return result

    # 3. Monotonic index (no duplicate/out-of-order dates)
    if not df.index.is_monotonic_increasing:
        result.errors.append("Dates are not monotonically increasing")
        result.passed = False

    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        result.errors.append(f"{n_dups} duplicate date(s) found")
        result.passed = False

    # 4. No future dates
    today = pd.Timestamp.today().normalize()
    future = df[df.index > today]
    if len(future) > 0:
        result.errors.append(f"{len(future)} rows have future dates")
        result.passed = False

    # 5. OHLCV internal consistency
    if (df["high"] < df["low"]).any():
        n = (df["high"] < df["low"]).sum()
        result.errors.append(f"{n} rows where High < Low")
        result.passed = False

    if (df["high"] < df["close"]).any():
        n = (df["high"] < df["close"]).sum()
        result.warnings.append(
            f"{n} rows where High < Close (check adjustment)"
        )

    if (df["low"] > df["close"]).any():
        n = (df["low"] > df["close"]).sum()
        result.warnings.append(
            f"{n} rows where Low > Close (check adjustment)"
        )

    # 6. Zero/negative prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            n = (df[col] <= 0).sum()
            result.errors.append(f"{n} zero/negative values in {col}")
            result.passed = False

    # 7. Zero volume
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        result.warnings.append(
            f"{zero_vol} rows with zero volume (holidays/halts?)"
        )

    # 8. Large gaps (missing trading days)
    if len(df) > 1:
        date_diffs = df.index.to_series().diff().dt.days.dropna()
        large_gaps = date_diffs[date_diffs > max_gap_days]
        if len(large_gaps) > 0:
            result.warnings.append(
                f"{len(large_gaps)} gaps > {max_gap_days} days. "
                f"Largest: {large_gaps.max()} days at {large_gaps.idxmax().date()}"
            )

    # 9. Extreme price changes (possible split not adjusted)
    if len(df) > 1:
        daily_ret = df["close"].pct_change().abs()
        extreme = daily_ret[daily_ret > 0.50]
        if len(extreme) > 0:
            dates_str = [str(d.date()) for d in extreme.index[:5]]
            result.warnings.append(
                f"{len(extreme)} days with >50% price move. "
                f"Verify split/dividend adjustments. Dates: {dates_str}"
            )

    # 10. Data freshness
    latest = df.index[-1]
    staleness = (today - latest).days
    if staleness > 3:
        result.warnings.append(
            f"Data is {staleness} days stale (latest: {latest.date()})"
        )

    return result

"""
Missing data handling.
Skill reference: .claude/skills/data-layer/SKILL.md
"""

from typing import Optional

import pandas as pd


def handle_missing_data(
    df: pd.DataFrame,
    max_ffill_days: int = 1,
    max_missing_pct: float = 0.05,
) -> Optional[pd.DataFrame]:
    """
    Forward-fill missing bars up to max_ffill_days.
    Returns None if symbol has too much missing data.
    Never interpolates — only forward-fill.
    """
    total_bars = len(df)
    if total_bars == 0:
        return None

    missing = df["close"].isna().sum()
    missing_pct = missing / total_bars

    if missing_pct > max_missing_pct:
        return None

    df = df.ffill(limit=max_ffill_days)
    return df

"""
Corporate actions handling (splits, reverse splits).
Skill reference: .claude/skills/data-layer/SKILL.md
"""

import pandas as pd


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
        (actions_df["symbol"] == symbol)
        & (actions_df["action_type"].isin(["split", "reverse_split"]))
    ].sort_values("date")

    df = df.copy()

    for _, action in symbol_actions.iterrows():
        action_date = pd.Timestamp(action["date"])
        factor = float(action["adjustment_factor"])

        mask = df.index < action_date
        price_cols = ["open", "high", "low", "close"]
        existing_price_cols = [c for c in price_cols if c in df.columns]
        df.loc[mask, existing_price_cols] *= factor
        if "volume" in df.columns:
            df.loc[mask, "volume"] /= factor

    return df

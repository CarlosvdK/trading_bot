"""
Volatility-targeting position sizing.
Skill reference: .claude/skills/position-sizing/SKILL.md
"""

import numpy as np


REGIME_SIZE_MULTIPLIERS = {
    "low_vol_trending_up": 1.0,
    "low_vol_choppy": 0.5,
    "low_vol_trending_down": 0.6,
    "high_vol_trending_up": 0.4,
    "high_vol_choppy": 0.0,
    "high_vol_trending_down": 0.0,
    "unknown": 0.5,
}


def vol_target_size(
    sleeve_nav: float,
    instrument_vol: float,
    holding_days: int,
    target_position_vol_pct: float = 0.005,
    max_position_pct: float = 0.15,
    min_position_usd: float = 100.0,
) -> float:
    """
    Volatility-targeted position size in notional dollars.
    Each position contributes approximately target_position_vol_pct
    of the sleeve's NAV as daily volatility.
    """
    if instrument_vol <= 0 or np.isnan(instrument_vol):
        return 0.0

    period_vol = instrument_vol * np.sqrt(holding_days / 252)
    vol_budget = sleeve_nav * target_position_vol_pct
    target_notional = vol_budget / period_vol

    max_notional = sleeve_nav * max_position_pct
    target_notional = min(target_notional, max_notional)

    if target_notional < min_position_usd:
        return 0.0

    return target_notional


def notional_to_shares(notional: float, price: float) -> float:
    """Convert notional size to share quantity. Always round down."""
    if price <= 0:
        return 0.0
    return float(np.floor(notional / price))


def ml_probability_size_scale(
    ml_prob: float,
    entry_threshold: float = 0.6,
    max_scale: float = 1.5,
    min_scale: float = 0.5,
) -> float:
    """
    Scale position size by ML model confidence.
    prob=threshold -> min_scale, prob=1.0 -> max_scale.
    """
    if ml_prob < entry_threshold:
        return 0.0

    confidence_range = 1.0 - entry_threshold
    if confidence_range <= 0:
        return min_scale

    relative_confidence = (ml_prob - entry_threshold) / confidence_range
    scale = min_scale + (max_scale - min_scale) * relative_confidence
    return scale


def regime_adjusted_size(
    base_notional: float,
    current_regime: str,
    vvol_pct: float = 0.5,
) -> float:
    """Apply regime and vol-of-vol adjustments to base position size."""
    regime_mult = REGIME_SIZE_MULTIPLIERS.get(current_regime, 0.5)
    vvol_mult = 1.0 if vvol_pct < 0.80 else 0.5
    return base_notional * regime_mult * vvol_mult


def compute_swing_position_size(
    symbol: str,
    sleeve_nav: float,
    instrument_vol: float,
    ml_prob: float,
    current_regime: str,
    vvol_percentile: float,
    price: float,
    config: dict,
) -> dict:
    """Full position sizing pipeline for a Swing trade."""
    base_notional = vol_target_size(
        sleeve_nav=sleeve_nav,
        instrument_vol=instrument_vol,
        holding_days=config.get("holding_days", 10),
        target_position_vol_pct=config.get("target_position_vol_pct", 0.005),
        max_position_pct=config.get("max_position_pct_swing", 0.15),
    )

    if base_notional == 0:
        return {"notional": 0, "shares": 0, "reason": "vol_too_high_or_missing"}

    ml_scale = ml_probability_size_scale(
        ml_prob,
        entry_threshold=config.get("ml_entry_threshold", 0.6),
        max_scale=config.get("ml_max_size_scale", 1.5),
        min_scale=config.get("ml_min_size_scale", 0.5),
    )

    if ml_scale == 0:
        return {"notional": 0, "shares": 0, "reason": "below_ml_threshold"}

    regime_notional = regime_adjusted_size(
        base_notional * ml_scale,
        current_regime,
        vvol_percentile,
    )

    if regime_notional == 0:
        return {
            "notional": 0,
            "shares": 0,
            "reason": f"regime_disabled: {current_regime}",
        }

    shares = notional_to_shares(regime_notional, price)
    final_notional = shares * price

    return {
        "notional": final_notional,
        "shares": shares,
        "base_notional": base_notional,
        "ml_scale": ml_scale,
        "regime_mult": REGIME_SIZE_MULTIPLIERS.get(current_regime, 0.5),
        "final_notional": final_notional,
        "reason": "OK",
    }


def compute_barriers(
    entry_price: float,
    instrument_vol: float,
    holding_days: int,
    k1: float = 2.0,
    k2: float = 1.0,
) -> dict:
    """Compute TP and SL prices from vol-based barriers."""
    period_vol = instrument_vol * np.sqrt(holding_days / 252)
    tp_pct = k1 * period_vol
    sl_pct = k2 * period_vol

    return {
        "tp_price": entry_price * (1 + tp_pct),
        "sl_price": entry_price * (1 - sl_pct),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "period_vol": period_vol,
    }

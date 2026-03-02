# Skill: Position Sizing & Volatility Targeting

## What This Skill Is
How to size positions correctly for the Swing and Core sleeves. The goal is to allocate risk, not capital — every position should contribute approximately the same amount of volatility to the portfolio. This naturally reduces size in high-vol regimes and increases it in low-vol periods.

---

## Why Volatility Targeting?

**Problem with fixed-percent sizing:**
- Buy 10% of sleeve in AAPL (low vol): contributes 2% portfolio vol
- Buy 10% of sleeve in GME (high vol): contributes 30% portfolio vol
- Result: GME dominates your risk budget despite equal capital allocation

**Vol-targeting solution:**
- Every position contributes the same target vol to the portfolio
- High-vol instruments get smaller positions
- Low-vol instruments get larger positions
- Risk is evenly distributed

---

## Core Formula

```
target_notional = (sleeve_vol_budget_per_position) / (instrument_vol * sqrt(holding_days / 252))

Where:
  sleeve_vol_budget_per_position = sleeve_nav * target_position_vol_pct
  instrument_vol = annualized volatility (rolling 21-day)
  holding_days   = expected holding period (e.g., 10 days for Swing)
```

```python
import numpy as np
import pandas as pd
from typing import Optional

def vol_target_size(
    sleeve_nav: float,
    instrument_vol: float,        # Annualized, e.g., 0.25 for 25%
    holding_days: int,            # Expected holding period
    target_position_vol_pct: float = 0.005,  # 0.5% of sleeve per position
    max_position_pct: float = 0.15,          # Hard cap: 15% of sleeve
    min_position_usd: float = 100.0,         # Minimum position size
) -> float:
    """
    Volatility-targeted position size in notional dollars.
    
    Each position contributes approximately target_position_vol_pct
    of the sleeve's NAV as daily volatility.
    
    Args:
        sleeve_nav:              Current sleeve NAV
        instrument_vol:          Annualized vol (e.g., 0.25)
        holding_days:            Expected hold (1-60)
        target_position_vol_pct: Target vol contribution per position (default: 0.5%)
        max_position_pct:        Hard cap as fraction of sleeve NAV
        min_position_usd:        Reject positions below this threshold
    
    Returns:
        Target notional size in USD
    """
    if instrument_vol <= 0 or np.isnan(instrument_vol):
        return 0.0

    # Scale vol to holding period
    period_vol = instrument_vol * np.sqrt(holding_days / 252)

    # Compute target notional
    vol_budget = sleeve_nav * target_position_vol_pct
    target_notional = vol_budget / period_vol

    # Apply caps
    max_notional = sleeve_nav * max_position_pct
    target_notional = min(target_notional, max_notional)

    # Minimum size check
    if target_notional < min_position_usd:
        return 0.0

    return target_notional


def notional_to_shares(notional: float, price: float) -> float:
    """Convert notional size to share quantity. Always round down."""
    if price <= 0:
        return 0.0
    return np.floor(notional / price)
```

---

## ML Probability Scaling

When the Trade Filter Model's confidence is high, you can increase size. When confidence is near the threshold, reduce size:

```python
def ml_probability_size_scale(
    ml_prob: float,
    entry_threshold: float = 0.6,
    max_scale: float = 1.5,
    min_scale: float = 0.5,
) -> float:
    """
    Scale position size by ML model confidence.
    
    prob=0.6 (threshold) → scale=min_scale (smallest allowed size)
    prob=1.0 → scale=max_scale (largest allowed size)
    
    This is "meta-labeling" sizing: bet more when the model is more confident.
    """
    if ml_prob < entry_threshold:
        return 0.0  # Below threshold: no trade

    # Linear interpolation from threshold to 1.0
    confidence_range = 1.0 - entry_threshold
    relative_confidence = (ml_prob - entry_threshold) / confidence_range

    scale = min_scale + (max_scale - min_scale) * relative_confidence
    return scale
```

---

## Regime-Adjusted Sizing

```python
REGIME_SIZE_MULTIPLIERS = {
    "low_vol_trending_up":    1.0,
    "low_vol_choppy":         0.5,
    "low_vol_trending_down":  0.6,
    "high_vol_trending_up":   0.4,
    "high_vol_choppy":        0.0,    # No swing trades
    "high_vol_trending_down": 0.0,
    "unknown":                0.5,    # Conservative default
}


def regime_adjusted_size(
    base_notional: float,
    current_regime: str,
    vvol_pct: float = 0.5,    # VVol percentile (0-1). High = unstable vol
) -> float:
    """
    Apply regime and vol-of-vol adjustments to base position size.
    """
    regime_mult = REGIME_SIZE_MULTIPLIERS.get(current_regime, 0.5)

    # Extra halving if vol-of-vol is high (unstable regime)
    vvol_mult = 1.0 if vvol_pct < 0.80 else 0.5

    return base_notional * regime_mult * vvol_mult
```

---

## Full Sizing Pipeline

```python
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
    """
    Full position sizing pipeline for a Swing trade.
    Returns dict with notional, shares, and sizing breakdown.
    """
    # Step 1: Base vol-targeted size
    base_notional = vol_target_size(
        sleeve_nav=sleeve_nav,
        instrument_vol=instrument_vol,
        holding_days=config.get("holding_days", 10),
        target_position_vol_pct=config.get("target_position_vol_pct", 0.005),
        max_position_pct=config.get("max_position_pct_swing", 0.15),
    )

    if base_notional == 0:
        return {"notional": 0, "shares": 0, "reason": "vol_too_high_or_missing"}

    # Step 2: ML confidence scaling
    ml_scale = ml_probability_size_scale(
        ml_prob,
        entry_threshold=config.get("ml_entry_threshold", 0.6),
        max_scale=config.get("ml_max_size_scale", 1.5),
        min_scale=config.get("ml_min_size_scale", 0.5),
    )

    if ml_scale == 0:
        return {"notional": 0, "shares": 0, "reason": "below_ml_threshold"}

    # Step 3: Regime adjustment
    regime_notional = regime_adjusted_size(
        base_notional * ml_scale,
        current_regime,
        vvol_percentile,
    )

    if regime_notional == 0:
        return {"notional": 0, "shares": 0, "reason": f"regime_disabled: {current_regime}"}

    # Step 4: Convert to shares
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
```

---

## Barrier Width (TP/SL) from Vol

Position sizing and barrier width should use the same vol proxy for consistency:

```python
def compute_barriers(
    entry_price: float,
    instrument_vol: float,    # Annualized
    holding_days: int,
    k1: float = 2.0,          # TP multiplier
    k2: float = 1.0,          # SL multiplier
) -> dict:
    """
    Compute TP and SL prices from vol-based barriers.
    Barriers scale with the holding period (longer hold = wider barriers).
    """
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
```

---

## Core Sleeve Sizing

Core sleeve uses target weights, not vol-targeting. Rebalance when weights drift:

```python
def core_rebalance_orders(
    current_weights: dict,     # symbol -> actual weight
    target_weights: dict,      # symbol -> target weight
    sleeve_nav: float,
    prices: dict,              # symbol -> price
    rebalance_band: float = 0.05,
) -> list:
    """
    Generate rebalance orders for Core sleeve.
    Only trades positions that drifted beyond the band.
    Returns list of (symbol, side, notional) tuples.
    """
    orders = []

    for symbol, target_wt in target_weights.items():
        actual_wt = current_weights.get(symbol, 0.0)
        drift = actual_wt - target_wt

        if abs(drift) > rebalance_band:
            # Need to sell if overweight, buy if underweight
            trade_notional = abs(drift) * sleeve_nav
            side = "SELL" if drift > 0 else "BUY"
            orders.append({
                "symbol": symbol,
                "side": side,
                "notional": trade_notional,
                "order_type": "MOC",   # Rebalance at close
                "reason": f"drift={drift:.2%}",
            })

    return orders
```

---

## Configuration

```yaml
sizing:
  swing:
    holding_days: 10
    target_position_vol_pct: 0.005    # 0.5% of sleeve per position
    max_position_pct_swing: 0.15      # Hard cap: 15% of sleeve
    min_position_usd: 500
    ml_entry_threshold: 0.60
    ml_max_size_scale: 1.5
    ml_min_size_scale: 0.5
    vol_proxy_window: 21

  core:
    rebalance_band: 0.05              # +/-5% drift triggers rebalance
    order_type: "MOC"                 # Always rebalance at close

  barriers:
    k1: 2.0                           # TP multiplier
    k2: 1.0                           # SL multiplier
```

---

## Common Sizing Mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| Fixed % of portfolio regardless of vol | High-vol names dominate risk | Use vol-targeting |
| Using 21-day vol window for both features AND sizing | Overfits sizing to signal | Document and justify vol window choice |
| Not capping position size | Single name can be 50%+ of sleeve | Always apply `max_position_pct` cap |
| Sizing before ML filter probability | Sizes trades the model will reject | Apply ML probability scale after vol size |
| Not scaling down in high-vol regime | Full size into worst conditions | Apply `REGIME_SIZE_MULTIPLIERS` |

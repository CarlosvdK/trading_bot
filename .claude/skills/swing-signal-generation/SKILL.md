# Skill: Swing Signal Generation (Non-TA)

## What This Skill Is
How to generate Swing trade candidates using simple, robust, non-TA triggers. The role of these signals is NOT to be the edge — the ML filter is the edge. Signals just produce a candidate list. Simple rules outperform complex TA because they generate enough candidates for the ML filter to work with.

---

## Design Principle

```
Swing Signal = Candidate Generator + ML Filter

Candidate Generator: "Something interesting happened" (broad, high recall)
ML Filter:           "Is this interesting thing actually tradeable?" (precise, high precision)

Don't try to encode edge in the signal rules.
The rules just produce raw candidates. The ML model decides.
```

---

## Signal 1: Multi-Day Momentum Breakout

A stock that has moved strongly in one direction over N days, with an acceleration on the most recent day.

```python
import pandas as pd
import numpy as np
from typing import List, Dict

def momentum_breakout_candidates(
    prices: Dict[str, pd.DataFrame],  # symbol -> OHLCV DataFrame
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Identify stocks showing multi-day momentum with recent acceleration.
    
    Trigger: 
      - 5-day return > threshold_pct
      - Today's return > 0 (momentum continuing)
      - Volume above average (conviction)
    
    NO TA indicators. Pure return and volume.
    """
    lookback = config.get("momentum_lookback_days", 5)
    threshold = config.get("momentum_threshold_pct", 0.04)  # 4%
    vol_surge_min = config.get("volume_surge_min", 1.2)     # 20% above avg
    
    candidates = []
    
    for symbol, df in prices.items():
        if current_date not in df.index:
            continue
        
        idx = df.index.get_loc(current_date)
        if idx < max(lookback, 21):
            continue  # Insufficient history
        
        # Features (all backward-looking)
        close = df["close"]
        ret_n = (close.iloc[idx] / close.iloc[idx - lookback]) - 1
        ret_1 = (close.iloc[idx] / close.iloc[idx - 1]) - 1
        
        avg_vol = df["volume"].iloc[idx - 21:idx].mean()
        vol_ratio = df["volume"].iloc[idx] / avg_vol if avg_vol > 0 else 0
        
        # Trigger conditions
        if (
            ret_n > threshold and
            ret_1 > 0 and
            vol_ratio >= vol_surge_min
        ):
            candidates.append({
                "symbol": symbol,
                "signal_type": "momentum_breakout",
                "direction": "LONG",
                "signal_date": current_date,
                "ret_n": ret_n,
                "ret_1": ret_1,
                "vol_ratio": vol_ratio,
            })
    
    return candidates
```

---

## Signal 2: Volatility Expansion (Breakout from Compression)

A stock that has been quiet (low vol) recently, then suddenly expands. This often precedes a directional move.

```python
def volatility_expansion_candidates(
    prices: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Identify stocks where short-term vol has expanded vs. recent baseline.
    
    Trigger:
      - 5-day vol / 21-day vol > expansion_ratio
      - Indicates regime change from compression to expansion
    
    Direction agnostic — filter with ML.
    """
    expansion_ratio = config.get("vol_expansion_ratio", 1.5)
    short_window = config.get("vol_short_window", 5)
    long_window = config.get("vol_long_window", 21)
    
    candidates = []
    
    for symbol, df in prices.items():
        if current_date not in df.index:
            continue
        
        idx = df.index.get_loc(current_date)
        if idx < long_window + 5:
            continue
        
        log_ret = np.log(df["close"] / df["close"].shift(1))
        
        short_vol = log_ret.iloc[idx - short_window:idx].std()
        long_vol = log_ret.iloc[idx - long_window:idx].std()
        
        if long_vol == 0:
            continue
        
        vol_ratio = short_vol / long_vol
        recent_ret = (df["close"].iloc[idx] / df["close"].iloc[idx - short_window]) - 1
        
        if vol_ratio > expansion_ratio:
            candidates.append({
                "symbol": symbol,
                "signal_type": "vol_expansion",
                "direction": "LONG" if recent_ret > 0 else "SHORT",
                "signal_date": current_date,
                "vol_ratio": vol_ratio,
                "recent_ret": recent_ret,
            })
    
    return candidates
```

---

## Signal 3: Cross-Asset Risk-On Proxy

When the broader market shows risk-on behavior (index up + vol down), momentum in individual names is more likely to persist. Use this as a "regime gate" for other signals.

```python
def is_risk_on(
    index_df: pd.DataFrame,    # Reference index (e.g., SPY)
    current_date: pd.Timestamp,
    config: dict,
) -> bool:
    """
    Simple risk-on/off classification from index behavior.
    Used to gate Swing signal generation.
    
    Risk-ON if:
      - Index 5-day return > -1% (not in sharp drawdown)
      - Index 5-day vol < 1.5x its 21-day vol (not in vol spike)
    """
    if current_date not in index_df.index:
        return False
    
    idx = index_df.index.get_loc(current_date)
    if idx < 21:
        return False
    
    log_ret = np.log(index_df["close"] / index_df["close"].shift(1))
    
    index_ret_5d = (index_df["close"].iloc[idx] / index_df["close"].iloc[idx - 5]) - 1
    vol_5d = log_ret.iloc[idx - 5:idx].std()
    vol_21d = log_ret.iloc[idx - 21:idx].std()
    vol_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.5
    
    ret_threshold = config.get("risk_on_min_index_ret", -0.01)
    vol_threshold = config.get("risk_on_max_vol_ratio", 1.5)
    
    return index_ret_5d > ret_threshold and vol_ratio < vol_threshold
```

---

## ML Filter Integration

After generating candidates, pass them through the ML filter:

```python
def apply_ml_filter(
    candidates: List[dict],
    feature_builder,          # FeatureBuilder instance
    ml_model,                 # Calibrated trade filter model
    prices: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
) -> List[dict]:
    """
    Filter candidates using ML trade filter model.
    Returns only candidates where P(favorable) >= threshold.
    """
    entry_threshold = config.get("ml_entry_threshold", 0.60)
    accepted = []
    
    for cand in candidates:
        symbol = cand["symbol"]
        sym_df = prices.get(symbol)
        
        if sym_df is None or current_date not in sym_df.index:
            continue
        
        # Build feature vector for this candidate at signal date
        try:
            features = feature_builder.build_single(
                symbol_df=sym_df,
                index_df=index_df,
                date=current_date,
            )
        except Exception as e:
            continue
        
        if features is None or features.isnull().any():
            continue
        
        # ML prediction
        prob = ml_model.predict_proba(features.values.reshape(1, -1))[0, 1]
        cand["ml_prob"] = round(prob, 4)
        
        if prob >= entry_threshold:
            accepted.append(cand)
    
    print(f"ML filter: {len(accepted)}/{len(candidates)} candidates accepted "
          f"(threshold={entry_threshold})")
    
    return accepted
```

---

## Full Swing Signal Pipeline

```python
def generate_swing_signals(
    prices: Dict[str, pd.DataFrame],
    index_df: pd.DataFrame,
    current_date: pd.Timestamp,
    ml_model,
    feature_builder,
    risk_governor,
    portfolio_state,
    config: dict,
) -> List[dict]:
    """
    Complete Swing signal pipeline:
    1. Check regime / risk-on gate
    2. Generate candidates
    3. ML filter
    4. Size positions
    5. Return order-ready signals
    """
    # Step 1: Risk-on gate
    if not is_risk_on(index_df, current_date, config):
        print(f"{current_date.date()}: Risk-off — no Swing signals generated")
        return []
    
    # Step 2: Generate candidates
    all_candidates = []
    
    if config.get("momentum_signal_enabled", True):
        all_candidates += momentum_breakout_candidates(prices, current_date, config)
    
    if config.get("vol_expansion_signal_enabled", True):
        all_candidates += volatility_expansion_candidates(prices, current_date, config)
    
    # Deduplicate by symbol (keep highest signal strength)
    seen = {}
    for cand in all_candidates:
        sym = cand["symbol"]
        if sym not in seen or cand.get("ret_n", 0) > seen[sym].get("ret_n", 0):
            seen[sym] = cand
    candidates = list(seen.values())
    
    if not candidates:
        return []
    
    # Step 3: ML filter
    filtered = apply_ml_filter(
        candidates, feature_builder, ml_model, prices, index_df, current_date, config
    )
    
    # Step 4: Convert to order-ready format
    orders = []
    for cand in filtered:
        symbol = cand["symbol"]
        sym_df = prices[symbol]
        
        if current_date not in sym_df.index:
            continue
        
        price = sym_df.loc[current_date, "close"]
        vol = np.log(sym_df["close"] / sym_df["close"].shift(1)).rolling(21).std().loc[current_date]
        vol_ann = vol * np.sqrt(252)
        
        # Barriers
        k1 = config.get("k1", 2.0)
        k2 = config.get("k2", 1.0)
        horizon = config.get("holding_days", 10)
        period_vol = vol_ann * np.sqrt(horizon / 252)
        tp_price = price * (1 + k1 * period_vol)
        sl_price = price * (1 - k2 * period_vol)
        
        # Size
        from skills.position_sizing import compute_swing_position_size
        sizing = compute_swing_position_size(
            symbol=symbol,
            sleeve_nav=portfolio_state.sleeve_values.get("swing", 30_000),
            instrument_vol=vol_ann,
            ml_prob=cand["ml_prob"],
            current_regime=config.get("current_regime", "unknown"),
            vvol_percentile=0.5,
            price=price,
            config=config,
        )
        
        if sizing["shares"] == 0:
            continue
        
        orders.append({
            "symbol": symbol,
            "side": "BUY" if cand["direction"] == "LONG" else "SHORT",
            "qty": sizing["shares"],
            "order_type": "MARKET",
            "sleeve": "swing",
            "tp_price": round(tp_price, 4),
            "sl_price": round(sl_price, 4),
            "ml_prob": cand["ml_prob"],
            "signal_type": cand["signal_type"],
        })
    
    return orders
```

---

## Configuration

```yaml
swing_signals:
  momentum_signal_enabled: true
  momentum_lookback_days: 5
  momentum_threshold_pct: 0.04      # 4% over 5 days
  volume_surge_min: 1.2             # 20% above 21-day avg

  vol_expansion_signal_enabled: true
  vol_expansion_ratio: 1.5          # Short vol / long vol
  vol_short_window: 5
  vol_long_window: 21

  risk_on_gate:
    enabled: true
    min_index_ret_5d: -0.01         # Index can't be down more than 1%
    max_vol_ratio: 1.5              # Short vol can't be 1.5x long vol

  ml_entry_threshold: 0.60
  holding_days: 10
  k1: 2.0
  k2: 1.0
```

---

## What Makes a Good Signal vs. Bad Signal

| Good Signal | Bad Signal |
|---|---|
| Simple, few conditions (2-3) | Complex with 10+ conditions |
| High recall (many candidates) | Highly selective (few candidates) |
| Based on price/volume math only | Based on TA library indicators |
| Stateless (no memory between days) | Stateful without explicit tracking |
| ML does the filtering | Signal tries to do its own filtering |
| Documented trigger conditions | "It felt right" logic |

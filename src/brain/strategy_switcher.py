"""
Regime-aware strategy switching.
Adjusts strategy parameters based on detected market regime.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.models.regime import (
    build_regime_features,
    fit_regime_model,
    predict_regime,
    label_regimes,
    smooth_regime,
)


@dataclass
class StrategyProfile:
    """Complete strategy parameter set for a given market regime."""
    name: str
    swing_enabled: bool
    swing_multiplier: float
    momentum_threshold_pct: float
    vol_expansion_ratio: float
    ml_entry_threshold: float
    max_concurrent_positions: int
    holding_days: int
    tp_multiplier: float
    sl_multiplier: float
    rebalance_band: float


REGIME_STRATEGIES: Dict[str, StrategyProfile] = {
    "low_vol_trending_up": StrategyProfile(
        name="aggressive", swing_enabled=True, swing_multiplier=1.0,
        momentum_threshold_pct=0.03, vol_expansion_ratio=1.3,
        ml_entry_threshold=0.55, max_concurrent_positions=10,
        holding_days=12, tp_multiplier=2.5, sl_multiplier=1.0, rebalance_band=0.05,
    ),
    "low_vol_choppy": StrategyProfile(
        name="selective", swing_enabled=True, swing_multiplier=0.5,
        momentum_threshold_pct=0.05, vol_expansion_ratio=1.6,
        ml_entry_threshold=0.65, max_concurrent_positions=5,
        holding_days=8, tp_multiplier=2.0, sl_multiplier=1.0, rebalance_band=0.04,
    ),
    "low_vol_trending_down": StrategyProfile(
        name="defensive", swing_enabled=True, swing_multiplier=0.5,
        momentum_threshold_pct=0.06, vol_expansion_ratio=1.8,
        ml_entry_threshold=0.70, max_concurrent_positions=4,
        holding_days=6, tp_multiplier=1.5, sl_multiplier=0.8, rebalance_band=0.03,
    ),
    "high_vol_trending_up": StrategyProfile(
        name="opportunistic", swing_enabled=True, swing_multiplier=0.3,
        momentum_threshold_pct=0.06, vol_expansion_ratio=2.0,
        ml_entry_threshold=0.70, max_concurrent_positions=4,
        holding_days=5, tp_multiplier=2.0, sl_multiplier=0.7, rebalance_band=0.04,
    ),
    "high_vol_choppy": StrategyProfile(
        name="cash", swing_enabled=False, swing_multiplier=0.0,
        momentum_threshold_pct=0.10, vol_expansion_ratio=3.0,
        ml_entry_threshold=0.90, max_concurrent_positions=0,
        holding_days=3, tp_multiplier=1.5, sl_multiplier=0.5, rebalance_band=0.02,
    ),
    "high_vol_trending_down": StrategyProfile(
        name="crisis", swing_enabled=False, swing_multiplier=0.0,
        momentum_threshold_pct=0.10, vol_expansion_ratio=3.0,
        ml_entry_threshold=0.95, max_concurrent_positions=0,
        holding_days=3, tp_multiplier=1.0, sl_multiplier=0.5, rebalance_band=0.02,
    ),
}

DEFAULT_PROFILE = StrategyProfile(
    name="default", swing_enabled=True, swing_multiplier=0.5,
    momentum_threshold_pct=0.04, vol_expansion_ratio=1.5,
    ml_entry_threshold=0.60, max_concurrent_positions=6,
    holding_days=10, tp_multiplier=2.0, sl_multiplier=1.0, rebalance_band=0.05,
)


def blend_strategies(
    current: StrategyProfile,
    target: StrategyProfile,
    blend_factor: float,
) -> StrategyProfile:
    """
    Smooth transition between strategies.
    blend_factor: 0.0 = current, 1.0 = target.
    """
    bf = np.clip(blend_factor, 0.0, 1.0)

    def interp(a: float, b: float) -> float:
        return a * (1 - bf) + b * bf

    return StrategyProfile(
        name=target.name if bf >= 0.5 else current.name,
        swing_enabled=target.swing_enabled if bf >= 0.5 else current.swing_enabled,
        swing_multiplier=round(interp(current.swing_multiplier, target.swing_multiplier), 3),
        momentum_threshold_pct=round(interp(current.momentum_threshold_pct, target.momentum_threshold_pct), 4),
        vol_expansion_ratio=round(interp(current.vol_expansion_ratio, target.vol_expansion_ratio), 2),
        ml_entry_threshold=round(interp(current.ml_entry_threshold, target.ml_entry_threshold), 3),
        max_concurrent_positions=int(round(interp(current.max_concurrent_positions, target.max_concurrent_positions))),
        holding_days=int(round(interp(current.holding_days, target.holding_days))),
        tp_multiplier=round(interp(current.tp_multiplier, target.tp_multiplier), 2),
        sl_multiplier=round(interp(current.sl_multiplier, target.sl_multiplier), 2),
        rebalance_band=round(interp(current.rebalance_band, target.rebalance_band), 3),
    )


class StrategySwitcher:
    """Integrates regime detection with strategy parameter switching."""

    def __init__(self, config: dict, regime_model_dict: Optional[dict] = None):
        self.config = config
        self.regime_model_dict = regime_model_dict
        self.regime_names: Dict[int, str] = {}
        self.current_regime: str = "unknown"
        self.current_strategy: StrategyProfile = DEFAULT_PROFILE
        self.transition_log: List[Tuple] = []
        self._days_in_regime: int = 0

    def update_regime(self, index_close: pd.Series, current_date: pd.Timestamp) -> str:
        """Predict current regime and update strategy."""
        features = build_regime_features(index_close.loc[:current_date], self.config)
        if features.empty:
            return self.current_regime

        if self.regime_model_dict is None:
            n_regimes = self.config.get("n_regimes", 4)
            method = self.config.get("regime_method", "kmeans")
            self.regime_model_dict = fit_regime_model(features, n_regimes, method)
            train_preds = predict_regime(self.regime_model_dict, features)
            self.regime_names = label_regimes(features, train_preds)

        recent = features.iloc[[-1]]
        pred = predict_regime(self.regime_model_dict, recent)
        regime_id = pred.iloc[0]
        regime_name = self.regime_names.get(regime_id, "unknown")

        if regime_name != self.current_regime:
            old_regime = self.current_regime
            old_strategy = self.current_strategy.name
            self.current_regime = regime_name
            target = REGIME_STRATEGIES.get(regime_name, DEFAULT_PROFILE)
            self.current_strategy = target
            self._days_in_regime = 0
            self.transition_log.append((
                current_date, old_regime, regime_name, old_strategy, target.name,
            ))
        else:
            self._days_in_regime += 1

        return self.current_regime

    def get_strategy(self) -> StrategyProfile:
        """Return current active strategy profile."""
        return self.current_strategy

    def get_config_overlay(self) -> dict:
        """Return config overrides for current regime."""
        s = self.current_strategy
        return {
            "swing_enabled": s.swing_enabled,
            "swing_multiplier": s.swing_multiplier,
            "momentum_threshold_pct": s.momentum_threshold_pct,
            "vol_expansion_ratio": s.vol_expansion_ratio,
            "ml_entry_threshold": s.ml_entry_threshold,
            "max_concurrent_positions": s.max_concurrent_positions,
            "holding_days": s.holding_days,
            "tp_multiplier": s.tp_multiplier,
            "sl_multiplier": s.sl_multiplier,
            "core_rebalance_band": s.rebalance_band,
            "regime": self.current_regime,
            "strategy_name": s.name,
        }

    def should_force_reduce(self) -> bool:
        """True in crisis regimes — triggers position reduction."""
        return self.current_regime in ("high_vol_choppy", "high_vol_trending_down")


def backtest_regime_switching(index_close: pd.Series, config: dict) -> pd.DataFrame:
    """Walk-forward regime prediction + strategy assignment for each date."""
    features = build_regime_features(index_close, config)
    initial = config.get("initial_train_days", 504)
    step = config.get("step_days", 63)
    n_regimes = config.get("n_regimes", 4)
    method = config.get("regime_method", "kmeans")

    rows = []
    for train_end in range(initial, len(features), step):
        train = features.iloc[:train_end]
        test = features.iloc[train_end:train_end + step]
        if test.empty:
            break

        model = fit_regime_model(train, n_regimes, method)
        preds = predict_regime(model, test)
        train_preds = predict_regime(model, train)
        names = label_regimes(train, train_preds)

        for date, regime_id in preds.items():
            regime_name = names.get(regime_id, "unknown")
            strategy = REGIME_STRATEGIES.get(regime_name, DEFAULT_PROFILE)
            rows.append({
                "date": date, "regime": regime_name,
                "strategy_name": strategy.name, "swing_enabled": strategy.swing_enabled,
                "swing_multiplier": strategy.swing_multiplier,
                "ml_entry_threshold": strategy.ml_entry_threshold,
                "max_concurrent_positions": strategy.max_concurrent_positions,
                "holding_days": strategy.holding_days,
            })

    return pd.DataFrame(rows)

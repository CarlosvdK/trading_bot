"""Regime Adapter — adapts thresholds and preferences by market regime."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


REGIME_LABELS = [
    "low_vol_trending_up",
    "low_vol_choppy",
    "low_vol_trending_down",
    "high_vol_trending_up",
    "high_vol_choppy",
    "high_vol_trending_down",
    "macro_driven",
    "earnings_event",
    "risk_on",
    "risk_off",
    "liquidity_stressed",
    "sector_dispersed",
]


@dataclass
class RegimeThresholds:
    """Adaptive thresholds for a given regime."""
    approval_threshold: float = 0.6  # min weighted approval for global vote
    min_voters: int = 5
    min_confidence: float = 0.6
    specialist_pass_threshold: float = 0.5
    position_size_multiplier: float = 1.0
    preferred_vehicles: List[str] = field(default_factory=list)
    max_new_positions: int = 5
    require_stop_loss: bool = False
    reduce_concentration: bool = False


# Regime -> threshold mapping
REGIME_THRESHOLD_MAP: Dict[str, RegimeThresholds] = {
    "low_vol_trending_up": RegimeThresholds(
        approval_threshold=0.5,
        min_voters=4,
        min_confidence=0.55,
        specialist_pass_threshold=0.45,
        position_size_multiplier=1.2,
        preferred_vehicles=["shares_spot", "call_option", "leaps"],
        max_new_positions=7,
    ),
    "low_vol_choppy": RegimeThresholds(
        approval_threshold=0.65,
        min_voters=5,
        min_confidence=0.65,
        specialist_pass_threshold=0.55,
        position_size_multiplier=0.7,
        preferred_vehicles=["shares_spot", "covered_call", "pairs_trade"],
        max_new_positions=3,
    ),
    "low_vol_trending_down": RegimeThresholds(
        approval_threshold=0.7,
        min_voters=5,
        min_confidence=0.7,
        specialist_pass_threshold=0.6,
        position_size_multiplier=0.5,
        preferred_vehicles=["put_option", "put_spread", "pairs_trade"],
        max_new_positions=3,
        require_stop_loss=True,
    ),
    "high_vol_trending_up": RegimeThresholds(
        approval_threshold=0.6,
        min_voters=5,
        min_confidence=0.65,
        specialist_pass_threshold=0.5,
        position_size_multiplier=0.8,
        preferred_vehicles=["shares_spot", "call_spread"],
        max_new_positions=4,
        require_stop_loss=True,
    ),
    "high_vol_choppy": RegimeThresholds(
        approval_threshold=0.8,
        min_voters=6,
        min_confidence=0.75,
        specialist_pass_threshold=0.65,
        position_size_multiplier=0.4,
        preferred_vehicles=["no_trade", "pairs_trade", "put_spread"],
        max_new_positions=2,
        require_stop_loss=True,
        reduce_concentration=True,
    ),
    "high_vol_trending_down": RegimeThresholds(
        approval_threshold=0.85,
        min_voters=7,
        min_confidence=0.8,
        specialist_pass_threshold=0.7,
        position_size_multiplier=0.3,
        preferred_vehicles=["no_trade", "put_option", "protective_put"],
        max_new_positions=1,
        require_stop_loss=True,
        reduce_concentration=True,
    ),
    "risk_off": RegimeThresholds(
        approval_threshold=0.8,
        min_voters=6,
        min_confidence=0.75,
        specialist_pass_threshold=0.65,
        position_size_multiplier=0.4,
        preferred_vehicles=["no_trade", "protective_put", "pairs_trade"],
        max_new_positions=2,
        require_stop_loss=True,
    ),
    "risk_on": RegimeThresholds(
        approval_threshold=0.5,
        min_voters=4,
        min_confidence=0.55,
        specialist_pass_threshold=0.45,
        position_size_multiplier=1.1,
        preferred_vehicles=["shares_spot", "call_option"],
        max_new_positions=6,
    ),
    "liquidity_stressed": RegimeThresholds(
        approval_threshold=0.9,
        min_voters=7,
        min_confidence=0.85,
        specialist_pass_threshold=0.75,
        position_size_multiplier=0.2,
        preferred_vehicles=["no_trade"],
        max_new_positions=0,
        require_stop_loss=True,
        reduce_concentration=True,
    ),
}

# Default thresholds for unknown regimes
DEFAULT_THRESHOLDS = RegimeThresholds(
    approval_threshold=0.65,
    min_voters=5,
    min_confidence=0.6,
    specialist_pass_threshold=0.5,
    position_size_multiplier=0.8,
    preferred_vehicles=["shares_spot"],
    max_new_positions=4,
)


class RegimeAdapter:
    """
    Detects market regime and provides adaptive thresholds.

    Thresholds are NOT fixed — they adapt to regime, signal quality,
    and portfolio context. Benign regimes lower barriers to encourage
    participation. Stressed regimes raise barriers dramatically.
    """

    def __init__(self):
        self._current_regime: str = ""
        self._regime_history: List[Tuple[pd.Timestamp, str]] = []

    def detect_regime(
        self,
        index_df: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
    ) -> str:
        """
        Classify current market regime from index data.

        Uses volatility, trend, and drawdown to classify into one of
        the REGIME_LABELS categories.
        """
        if index_df.empty or "close" not in index_df.columns:
            return "low_vol_choppy"

        close = index_df["close"]
        if current_date is not None and current_date in close.index:
            idx = close.index.get_loc(current_date)
        else:
            idx = len(close) - 1

        if idx < 63:
            return "low_vol_choppy"

        log_ret = np.log(close / close.shift(1)).dropna()

        # Volatility classification
        vol_21 = log_ret.iloc[max(0, idx - 21):idx + 1].std() * np.sqrt(252)
        vol_63 = log_ret.iloc[max(0, idx - 63):idx + 1].std() * np.sqrt(252)
        high_vol = vol_21 > 0.20  # annualized > 20%

        # Trend classification
        ret_21 = (close.iloc[idx] / close.iloc[idx - 21]) - 1
        ret_63 = (close.iloc[idx] / close.iloc[idx - 63]) - 1

        # Drawdown
        peak = close.iloc[max(0, idx - 63):idx + 1].max()
        drawdown = (close.iloc[idx] - peak) / peak

        # Liquidity check (vol-of-vol)
        if len(log_ret) > 21:
            vvol = log_ret.iloc[max(0, idx - 21):idx + 1].rolling(5).std().std()
        else:
            vvol = 0

        # Classification
        if vvol > 0.02 or drawdown < -0.10:
            regime = "liquidity_stressed"
        elif drawdown < -0.05 and high_vol:
            regime = "risk_off"
        elif ret_21 > 0.03 and not high_vol:
            regime = "risk_on"
        elif high_vol:
            if ret_21 > 0.01:
                regime = "high_vol_trending_up"
            elif ret_21 < -0.01:
                regime = "high_vol_trending_down"
            else:
                regime = "high_vol_choppy"
        else:
            if ret_21 > 0.01:
                regime = "low_vol_trending_up"
            elif ret_21 < -0.01:
                regime = "low_vol_trending_down"
            else:
                regime = "low_vol_choppy"

        self._current_regime = regime
        if current_date:
            self._regime_history.append((current_date, regime))

        return regime

    def get_thresholds(self, regime: Optional[str] = None) -> RegimeThresholds:
        """Get adaptive thresholds for the given regime."""
        r = regime or self._current_regime
        return REGIME_THRESHOLD_MAP.get(r, DEFAULT_THRESHOLDS)

    def adjust_for_signal_quality(
        self,
        thresholds: RegimeThresholds,
        avg_confidence: float,
        n_proposals: int,
    ) -> RegimeThresholds:
        """
        Further adapt thresholds based on current signal quality.

        When signal quality is high (many high-confidence proposals),
        slightly lower thresholds. When quality is low, raise them.
        """
        import copy
        adjusted = copy.copy(thresholds)

        if avg_confidence > 0.8 and n_proposals > 5:
            adjusted.approval_threshold *= 0.9
            adjusted.min_confidence *= 0.95
        elif avg_confidence < 0.5 or n_proposals < 2:
            adjusted.approval_threshold = min(0.95, adjusted.approval_threshold * 1.15)
            adjusted.min_confidence = min(0.9, adjusted.min_confidence * 1.1)
            adjusted.position_size_multiplier *= 0.7

        return adjusted

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def regime_history(self) -> List[Tuple[pd.Timestamp, str]]:
        return list(self._regime_history)

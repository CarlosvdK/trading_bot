"""Trading Agent — core agent class that generates trade picks based on DNA personality."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable

from src.agents.agent_dna import AgentDNA
from src.agents.sector_mapping import get_sector, get_sub_industry, get_agent_universe


@dataclass
class TradePick:
    """A single trade idea from an agent."""

    symbol: str
    direction: str           # "long" or "short"
    confidence: float        # 0.0 to 1.0
    agent_id: str
    strategy_used: str
    reasoning: str           # brief text explanation
    suggested_hold_days: int
    timestamp: datetime = field(default_factory=datetime.now)
    sector: str = ""
    sub_industry: str = ""
    raw_score: float = 0.0   # pre-adjustment score
    peer_approved: bool = False  # whether peer pre-vote approved this
    peer_approval_pct: float = 0.0  # % of peers that approved


class TradingAgent:
    """
    A trading agent that scans the universe and generates trade picks
    based on its DNA personality configuration.
    """

    def __init__(
        self,
        dna: AgentDNA,
        feature_config: Optional[dict] = None,
        regime_detector: Optional[dict] = None,
        regime_names: Optional[dict] = None,
        sentiment_scores: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            dna: Agent personality configuration.
            feature_config: Config dict for feature engineering.
            regime_detector: Fitted regime model dict (from regime.fit_regime_model).
            regime_names: Mapping of regime_id to human-readable names.
            sentiment_scores: Optional {symbol: sentiment_score} from NLP pipeline.
        """
        self.dna = dna
        self.feature_config = feature_config or {
            "return_windows": [5, 10, 21],
            "vol_windows": [5, 21],
        }
        self.regime_detector = regime_detector
        self.regime_names = regime_names or {}
        self.sentiment_scores = sentiment_scores or {}
        self._current_regime: Optional[str] = None

        # Strategy dispatch
        self._strategy_map: Dict[str, Callable] = {
            "momentum": self._apply_momentum,
            "mean_reversion": self._apply_mean_reversion,
            "value": self._apply_value,
            "growth": self._apply_growth,
            "event_driven": self._apply_event_driven,
            "volatility": self._apply_volatility,
            "sentiment": self._apply_sentiment,
            "breakout": self._apply_breakout,
        }

    def scan(
        self,
        universe_data: Dict[str, pd.DataFrame],
        index_df: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
    ) -> List[TradePick]:
        """
        Scan the universe and return trade picks sorted by confidence.

        Args:
            universe_data: {symbol: OHLCV DataFrame} for the full universe.
            index_df: Index OHLCV DataFrame for relative calculations.
            current_date: Date to scan at. If None, uses last available date.

        Returns:
            List of TradePick sorted by confidence descending.
        """
        # Detect regime if we have a detector
        if self.regime_detector and index_df is not None and current_date is not None:
            self._detect_regime(index_df, current_date)

        # Filter universe to agent's known sectors
        agent_universe = self._filter_universe(universe_data)
        if not agent_universe:
            return []

        picks: List[TradePick] = []

        for symbol, df in agent_universe.items():
            if df.empty or len(df) < self.dna.lookback_days:
                continue

            if current_date is not None and current_date not in df.index:
                continue

            try:
                score, direction, reasoning = self._score_symbol(
                    symbol, df, index_df, current_date
                )
            except Exception:
                continue

            # Apply regime adjustment
            if self.dna.regime_sensitivity > 0 and self._current_regime:
                score = self._adjust_for_regime(score, self._current_regime)

            # Apply contrarian inversion
            if self.dna.contrarian_factor > 0:
                score, direction = self._apply_contrarian(score, direction)

            # Convert score to confidence (sigmoid mapping)
            confidence = self._score_to_confidence(score)

            if confidence < self.dna.min_confidence:
                continue

            sector = get_sector(symbol) or ""
            sub = get_sub_industry(symbol) or ""
            min_days, max_days = self.dna.holding_days_range
            hold_days = int((min_days + max_days) / 2)

            picks.append(TradePick(
                symbol=symbol,
                direction=direction,
                confidence=round(confidence, 4),
                agent_id=self.dna.agent_id,
                strategy_used=self.dna.primary_strategy,
                reasoning=reasoning,
                suggested_hold_days=hold_days,
                sector=sector,
                sub_industry=sub,
                raw_score=round(score, 4),
            ))

        # Sort by confidence and limit to max picks
        picks.sort(key=lambda p: p.confidence, reverse=True)
        return picks[:self.dna.max_picks_per_scan]

    def _filter_universe(
        self, universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Filter universe to only symbols in agent's known sectors."""
        if not self.dna.primary_sectors and not self.dna.secondary_sectors:
            return universe_data  # "all sectors" agents see everything

        # Check if agent has "all" as a sector
        all_sectors = self.dna.primary_sectors + self.dna.secondary_sectors
        if "all" in all_sectors:
            return universe_data

        agent_symbols = get_agent_universe(
            self.dna.primary_sectors, self.dna.secondary_sectors
        )
        return {
            sym: df for sym, df in universe_data.items()
            if sym in agent_symbols
        }

    def _score_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame],
        current_date: Optional[pd.Timestamp],
    ) -> tuple:
        """
        Score a symbol using primary (and optionally secondary) strategy.

        Returns:
            (score, direction, reasoning) tuple.
        """
        primary_fn = self._strategy_map.get(self.dna.primary_strategy)
        if not primary_fn:
            return 0.0, "long", "unknown strategy"

        score, direction, reasoning = primary_fn(df, current_date, index_df)

        # Blend with secondary strategy if configured
        if self.dna.secondary_strategy:
            secondary_fn = self._strategy_map.get(self.dna.secondary_strategy)
            if secondary_fn:
                s2, d2, r2 = secondary_fn(df, current_date, index_df)
                # Primary gets 70% weight, secondary 30%
                score = score * 0.7 + s2 * 0.3
                if abs(s2) > abs(score):
                    direction = d2
                reasoning = f"{reasoning}; {r2}"

        return score, direction, reasoning

    # ------------------------------------------------------------------ #
    #  Strategy Implementations (all backward-looking)                     #
    # ------------------------------------------------------------------ #

    def _apply_momentum(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Momentum: ROC, RSI, price vs moving averages, volume trend."""
        idx = self._get_idx(df, current_date)
        lb = min(self.dna.lookback_days, idx)

        close = df["close"]
        roc = (close.iloc[idx] / close.iloc[idx - lb]) - 1

        # RSI
        rsi = self._compute_rsi(close, idx)

        # Price vs SMA
        sma = close.iloc[idx - lb:idx + 1].mean()
        price_vs_sma = (close.iloc[idx] - sma) / sma

        # Volume trend
        vol_recent = df["volume"].iloc[max(0, idx - 5):idx + 1].mean()
        vol_baseline = df["volume"].iloc[max(0, idx - lb):idx + 1].mean()
        vol_trend = (vol_recent / vol_baseline - 1) if vol_baseline > 0 else 0

        # Composite score
        score = (roc * 0.35) + (price_vs_sma * 0.25) + ((rsi - 50) / 100 * 0.2) + (vol_trend * 0.2)
        direction = "long" if score > 0 else "short"
        reasoning = f"ROC={roc:.3f} RSI={rsi:.0f} PvSMA={price_vs_sma:.3f} VolTrend={vol_trend:.3f}"
        return score, direction, reasoning

    def _apply_mean_reversion(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Mean reversion: z-score of price vs SMA, bollinger position, RSI extremes."""
        idx = self._get_idx(df, current_date)
        lb = min(self.dna.lookback_days, idx)
        close = df["close"]

        # Z-score vs SMA
        sma = close.iloc[idx - lb:idx + 1].mean()
        std = close.iloc[idx - lb:idx + 1].std()
        z_score = (close.iloc[idx] - sma) / std if std > 0 else 0

        # Bollinger band position
        bb_mid = close.iloc[max(0, idx - 20):idx + 1].mean()
        bb_std = close.iloc[max(0, idx - 20):idx + 1].std()
        if bb_std > 0:
            bb_pct = (close.iloc[idx] - (bb_mid - 2 * bb_std)) / (4 * bb_std)
        else:
            bb_pct = 0.5

        # RSI extremes (favor oversold for long, overbought for short)
        rsi = self._compute_rsi(close, idx)
        rsi_signal = 0
        if rsi < 30:
            rsi_signal = (30 - rsi) / 30  # positive = oversold = long signal
        elif rsi > 70:
            rsi_signal = (70 - rsi) / 30  # negative = overbought = short signal

        # Mean reversion score (negative z = buy, positive z = sell)
        score = (-z_score * 0.4) + ((0.5 - bb_pct) * 0.3) + (rsi_signal * 0.3)
        direction = "long" if score > 0 else "short"
        reasoning = f"Zscore={z_score:.2f} BB%={bb_pct:.2f} RSI={rsi:.0f}"
        return score, direction, reasoning

    def _apply_value(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Value: price vs 52w range, relative volume, support levels."""
        idx = self._get_idx(df, current_date)
        close = df["close"]

        # Price position in 52-week range
        lb_252 = min(252, idx)
        high_52w = close.iloc[max(0, idx - lb_252):idx + 1].max()
        low_52w = close.iloc[max(0, idx - lb_252):idx + 1].min()
        range_52w = high_52w - low_52w
        if range_52w > 0:
            pct_of_range = (close.iloc[idx] - low_52w) / range_52w
        else:
            pct_of_range = 0.5

        # Discount from high (value = bigger discount)
        discount = 1 - (close.iloc[idx] / high_52w) if high_52w > 0 else 0

        # Volume relative to average (accumulation signal)
        vol_avg = df["volume"].iloc[max(0, idx - 21):idx].mean()
        vol_ratio = df["volume"].iloc[idx] / vol_avg if vol_avg > 0 else 1

        # Support level (price near recent lows)
        low_20 = close.iloc[max(0, idx - 20):idx + 1].min()
        near_support = 1 - (close.iloc[idx] - low_20) / (close.iloc[idx] * 0.1 + 1e-8)
        near_support = max(0, min(1, near_support))

        # Value score: higher discount = more attractive
        score = (discount * 0.4) + ((1 - pct_of_range) * 0.3) + (near_support * 0.15) + ((vol_ratio - 1) * 0.15)
        direction = "long"  # value agents are naturally long-biased
        reasoning = f"Discount={discount:.2f} Range%={pct_of_range:.2f} VolRatio={vol_ratio:.2f}"
        return score, direction, reasoning

    def _apply_growth(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Growth: consecutive higher highs, expanding range, volume acceleration."""
        idx = self._get_idx(df, current_date)
        close = df["close"]
        high = df["high"]

        # Consecutive higher highs
        higher_highs = 0
        for i in range(1, min(11, idx)):
            if high.iloc[idx - i + 1] > high.iloc[idx - i]:
                higher_highs += 1
            else:
                break
        hh_score = higher_highs / 10

        # Range expansion (ATR trending up)
        tr_recent = (df["high"].iloc[max(0, idx - 5):idx + 1] - df["low"].iloc[max(0, idx - 5):idx + 1]).mean()
        tr_baseline = (df["high"].iloc[max(0, idx - 20):idx + 1] - df["low"].iloc[max(0, idx - 20):idx + 1]).mean()
        range_expansion = (tr_recent / tr_baseline - 1) if tr_baseline > 0 else 0

        # Volume acceleration
        vol_5 = df["volume"].iloc[max(0, idx - 5):idx + 1].mean()
        vol_20 = df["volume"].iloc[max(0, idx - 20):idx + 1].mean()
        vol_accel = (vol_5 / vol_20 - 1) if vol_20 > 0 else 0

        # Price slope (upward trend)
        if idx >= 10:
            x = np.arange(10)
            y = close.iloc[idx - 9:idx + 1].values
            slope = np.polyfit(x, y, 1)[0] / close.iloc[idx] if len(y) == 10 else 0
        else:
            slope = 0

        score = (hh_score * 0.3) + (range_expansion * 0.2) + (vol_accel * 0.2) + (slope * 0.3)
        direction = "long"
        reasoning = f"HH={higher_highs} RangeExp={range_expansion:.3f} VolAccel={vol_accel:.3f}"
        return score, direction, reasoning

    def _apply_event_driven(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Event-driven: volume spikes, gap detection, unusual range expansion."""
        idx = self._get_idx(df, current_date)
        close = df["close"]

        # Volume spike
        vol_avg = df["volume"].iloc[max(0, idx - 20):idx].mean()
        vol_ratio = df["volume"].iloc[idx] / vol_avg if vol_avg > 0 else 1
        vol_spike = max(0, vol_ratio - 1.5)  # only count if >1.5x

        # Gap detection
        prev_close = close.iloc[idx - 1] if idx > 0 else close.iloc[idx]
        gap = (df["open"].iloc[idx] - prev_close) / prev_close if prev_close > 0 else 0

        # Range expansion
        today_range = (df["high"].iloc[idx] - df["low"].iloc[idx]) / close.iloc[idx]
        avg_range = ((df["high"].iloc[max(0, idx - 20):idx] - df["low"].iloc[max(0, idx - 20):idx]) / close.iloc[max(0, idx - 20):idx]).mean()
        range_expansion = (today_range / avg_range - 1) if avg_range > 0 else 0

        # Day return direction
        day_ret = (close.iloc[idx] / df["open"].iloc[idx]) - 1 if df["open"].iloc[idx] > 0 else 0

        score = (vol_spike * 0.4) + (abs(gap) * 0.3) + (range_expansion * 0.3)
        direction = "long" if (gap > 0 or day_ret > 0) else "short"
        reasoning = f"VolSpike={vol_ratio:.2f} Gap={gap:.3f} RangeExp={range_expansion:.3f}"
        return score, direction, reasoning

    def _apply_volatility(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Volatility: ATR expansion/contraction, regime detection."""
        idx = self._get_idx(df, current_date)
        close = df["close"]
        log_ret = np.log(close / close.shift(1))

        # Short vs long vol
        vol_5 = log_ret.iloc[max(0, idx - 5):idx + 1].std() * np.sqrt(252)
        vol_21 = log_ret.iloc[max(0, idx - 21):idx + 1].std() * np.sqrt(252)
        vol_ratio = vol_5 / vol_21 if vol_21 > 0 else 1

        # ATR trend
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_5 = tr.iloc[max(0, idx - 5):idx + 1].mean()
        atr_21 = tr.iloc[max(0, idx - 21):idx + 1].mean()
        atr_ratio = atr_5 / atr_21 if atr_21 > 0 else 1

        # Vol contraction = potential breakout signal
        if vol_ratio < 0.7:
            score = 0.5  # low vol = compression = potential breakout
            direction = "long"
            reasoning = f"VolCompression: ratio={vol_ratio:.2f}"
        elif vol_ratio > 1.5:
            # High vol expansion: trade in direction of move
            ret_5 = (close.iloc[idx] / close.iloc[max(0, idx - 5)]) - 1
            score = abs(ret_5) * vol_ratio * 0.5
            direction = "long" if ret_5 > 0 else "short"
            reasoning = f"VolExpansion: ratio={vol_ratio:.2f} ret5d={ret_5:.3f}"
        else:
            score = 0.1
            direction = "long"
            reasoning = f"NeutralVol: ratio={vol_ratio:.2f}"

        return score, direction, reasoning

    def _apply_sentiment(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Sentiment: uses NLP pipeline scores if available."""
        idx = self._get_idx(df, current_date)
        symbol = df.name if hasattr(df, 'name') else ""
        close = df["close"]

        # Get sentiment score
        sent_score = self.sentiment_scores.get(symbol, 0.0)

        # Price momentum for confirmation/divergence
        ret_5 = (close.iloc[idx] / close.iloc[max(0, idx - 5)]) - 1

        if abs(sent_score) < 0.05:
            # No meaningful sentiment
            score = 0.0
            direction = "long"
            reasoning = f"NoSentiment: score={sent_score:.3f}"
        elif sent_score > 0 and ret_5 > 0:
            # Bullish confirmation
            score = sent_score * 0.6 + ret_5 * 0.4
            direction = "long"
            reasoning = f"BullConfirm: sent={sent_score:.3f} ret5d={ret_5:.3f}"
        elif sent_score > 0 and ret_5 < -0.03:
            # Bullish divergence (sentiment up, price down)
            score = sent_score * 0.5
            direction = "long"
            reasoning = f"BullDivergence: sent={sent_score:.3f} ret5d={ret_5:.3f}"
        elif sent_score < 0 and ret_5 < 0:
            # Bearish confirmation
            score = abs(sent_score) * 0.6 + abs(ret_5) * 0.4
            direction = "short"
            reasoning = f"BearConfirm: sent={sent_score:.3f} ret5d={ret_5:.3f}"
        else:
            score = abs(sent_score) * 0.3
            direction = "long" if sent_score > 0 else "short"
            reasoning = f"SentimentMixed: sent={sent_score:.3f} ret5d={ret_5:.3f}"

        return score, direction, reasoning

    def _apply_breakout(
        self, df: pd.DataFrame, current_date: Optional[pd.Timestamp],
        index_df: Optional[pd.DataFrame] = None,
    ) -> tuple:
        """Breakout: resistance level breaks, volume confirmation, range expansion."""
        idx = self._get_idx(df, current_date)
        close = df["close"]
        high = df["high"]

        # Resistance: highest close in lookback
        lb = min(self.dna.lookback_days, idx - 1)
        resistance = close.iloc[max(0, idx - lb):idx].max()
        breakout_pct = (close.iloc[idx] - resistance) / resistance if resistance > 0 else 0

        # Volume confirmation
        vol_avg = df["volume"].iloc[max(0, idx - 20):idx].mean()
        vol_ratio = df["volume"].iloc[idx] / vol_avg if vol_avg > 0 else 1

        # Range expansion
        today_range = (high.iloc[idx] - df["low"].iloc[idx])
        avg_range = (high.iloc[max(0, idx - 20):idx] - df["low"].iloc[max(0, idx - 20):idx]).mean()
        range_exp = today_range / avg_range if avg_range > 0 else 1

        if breakout_pct > 0 and vol_ratio > 1.2:
            score = (breakout_pct * 0.4) + ((vol_ratio - 1) * 0.3) + ((range_exp - 1) * 0.3)
            direction = "long"
            reasoning = f"Breakout={breakout_pct:.3f} Vol={vol_ratio:.2f} Range={range_exp:.2f}"
        elif breakout_pct < -0.03 and vol_ratio > 1.2:
            # Breakdown
            score = abs(breakout_pct) * 0.5 + (vol_ratio - 1) * 0.3
            direction = "short"
            reasoning = f"Breakdown={breakout_pct:.3f} Vol={vol_ratio:.2f}"
        else:
            score = 0.05
            direction = "long"
            reasoning = f"NoBreakout: pct={breakout_pct:.3f} vol={vol_ratio:.2f}"

        return score, direction, reasoning

    # ------------------------------------------------------------------ #
    #  Adjustments                                                         #
    # ------------------------------------------------------------------ #

    def _adjust_for_regime(self, raw_score: float, regime: str) -> float:
        """Adjust score based on current market regime and agent's sensitivity."""
        sensitivity = self.dna.regime_sensitivity

        # Regime multipliers
        regime_multipliers = {
            "low_vol_trending_up": 1.2,
            "low_vol_choppy": 0.8,
            "low_vol_trending_down": 0.6,
            "high_vol_trending_up": 0.9,
            "high_vol_choppy": 0.4,
            "high_vol_trending_down": 0.3,
        }
        multiplier = regime_multipliers.get(regime, 1.0)

        # Blend between no adjustment (1.0) and full adjustment based on sensitivity
        adjusted_multiplier = 1.0 + sensitivity * (multiplier - 1.0)
        return raw_score * adjusted_multiplier

    def _apply_contrarian(self, score: float, direction: str) -> tuple:
        """Apply contrarian factor -- invert signal when factor > 0.5."""
        cf = self.dna.contrarian_factor
        if cf > 0.5:
            # Invert: strong signals become contrarian trades
            inversion_strength = (cf - 0.5) * 2  # 0 to 1
            score = score * (1 - 2 * inversion_strength)
            if score < 0:
                score = abs(score)
                direction = "short" if direction == "long" else "long"
        return score, direction

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_idx(self, df: pd.DataFrame, current_date: Optional[pd.Timestamp]) -> int:
        """Get integer index for current_date, or last available."""
        if current_date is not None and current_date in df.index:
            return df.index.get_loc(current_date)
        return len(df) - 1

    def _compute_rsi(self, close: pd.Series, idx: int, period: int = 14) -> float:
        """Compute RSI at a given index (backward-looking only)."""
        if idx < period:
            return 50.0
        delta = close.diff()
        gains = delta.clip(lower=0).iloc[max(0, idx - period):idx + 1]
        losses = (-delta.clip(upper=0)).iloc[max(0, idx - period):idx + 1]
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _score_to_confidence(self, score: float) -> float:
        """Map raw score to 0-1 confidence using sigmoid."""
        # Scale factor adjusts sensitivity
        scale = 5.0
        sigmoid = 1 / (1 + np.exp(-scale * score))
        # Map from (0.5, 1.0) for positive scores
        return max(0.0, min(1.0, sigmoid))

    def _detect_regime(
        self, index_df: pd.DataFrame, current_date: pd.Timestamp
    ) -> None:
        """Detect current market regime from index data."""
        try:
            from src.models.regime import build_regime_features, predict_regime, label_regimes

            if "close" not in index_df.columns:
                return
            features = build_regime_features(index_df["close"], self.feature_config)
            if features.empty:
                return
            if self.regime_detector:
                preds = predict_regime(self.regime_detector, features)
                if not preds.empty:
                    names = self.regime_names or label_regimes(features, preds)
                    last_regime_id = preds.iloc[-1]
                    self._current_regime = names.get(last_regime_id, "unknown")
        except Exception:
            pass

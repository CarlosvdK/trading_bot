"""
Autonomous Orchestrator — the daily brain loop.

Handles the complete lifecycle:
  1. Warm up: train ML + regime models if none exist
  2. Check drift / model health → retrain if needed
  3. Detect regime
  4. Generate signals → ML filter → size → risk check → execute
  5. Monitor open positions (barrier exits)
  6. Log everything & feed outcomes back for drift monitoring
  7. Adapt parameters based on feedback

This is what makes the bot self-sufficient.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.risk_management.risk_governor import RiskGovernor, RiskConfig, PortfolioState
from src.models.trainer import MLTrainer
from src.models.features import build_features
from src.models.labeler import compute_vol_proxy
from src.models.regime import (
    build_regime_features,
    fit_regime_model,
    predict_regime,
    label_regimes,
    get_regime_allocation,
    smooth_regime,
)
from src.models.drift import monitor_feature_drift, compute_live_metrics
from src.signals.signals import generate_swing_signals
from src.signals.sizing import compute_swing_position_size, compute_barriers
from src.utilities.audit import AuditLogger
from src.market_intel.news_signals import generate_news_signals, get_news_boost
from src.trading.order_types import Order, OrderType, OrderSide

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Self-sufficient trading orchestrator.
    Manages the full lifecycle: data → learn → predict → trade → adapt.
    """

    def __init__(self, config: dict, root_dir: str, order_manager=None):
        self.config = config
        self.root_dir = Path(root_dir)
        self.state_file = self.root_dir / "state" / "orchestrator_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load sector mapping from universe.csv
        self._sector_mapping = self._load_sector_mapping()
        self._universe_closes = None  # Built lazily on first use

        # Inject sector mapping into config for trainer
        config_with_sectors = dict(config)
        config_with_sectors["_sector_mapping"] = self._sector_mapping

        # Core components
        self.risk_config = RiskConfig(**{
            k: v for k, v in config.get("risk", {}).items()
            if k in RiskConfig.__dataclass_fields__
        })
        self.risk_governor = RiskGovernor(self.risk_config)
        self.ml_trainer = MLTrainer(
            config_with_sectors, models_dir=str(self.root_dir / "models")
        )
        self.audit = AuditLogger(str(self.root_dir / "logs" / "audit.jsonl"))

        # Execution pipeline (optional — if None, uses direct position tracking)
        self.order_manager = order_manager
        self._order_counter = 0

        # State
        self.portfolio_state = self._init_portfolio_state()
        self.open_positions: Dict[str, dict] = {}
        self.trade_log: List[dict] = []
        self.prediction_log: List[dict] = []
        self.daily_nav: List[dict] = []

        # Regime state
        self.regime_model = None
        self.regime_names = {}
        self.regime_series = None
        self._regime_last_trained = None

        # News state
        self.news_signals: List[dict] = []
        self.last_news_scan = None
        self._news_enabled = config.get("news", {}).get("enabled", True)
        self._news_scan_interval_days = config.get("news", {}).get("scan_interval_days", 1)

        # Adaptive state
        self.performance_by_regime: Dict[str, list] = {}
        self.last_retrain_date = None
        self.last_drift_check = None
        self._warmup_done = False

        # Load persisted state
        self._load_state()

        # Try to load latest ML model
        self.ml_trainer.load_latest_model()

    def _init_portfolio_state(self) -> PortfolioState:
        initial_nav = self.config.get("portfolio", {}).get("initial_nav", 100_000)
        allocs = self.config.get("portfolio", {}).get(
            "sleeve_allocations",
            {"core": 0.60, "swing": 0.30, "cash_buffer": 0.10},
        )
        swing_alloc = allocs.get("swing", 0.30)
        core_alloc = allocs.get("core", 0.60)
        cash_alloc = allocs.get("cash_buffer", 0.10)
        return PortfolioState(
            nav=initial_nav,
            peak_nav=initial_nav,
            # Cash = swing cash (available for trading) + cash buffer
            cash=initial_nav * (swing_alloc + cash_alloc),
            sleeve_values={
                "swing": initial_nav * swing_alloc,
                "core": initial_nav * core_alloc,
            },
            positions={},
            day_start_nav=initial_nav,
        )

    def _load_sector_mapping(self) -> dict:
        """Load symbol → sector_etf mapping from universe.csv."""
        universe_file = self.config.get("data", {}).get(
            "universe_file",
            str(self.root_dir / "data" / "universe.csv"),
        )
        try:
            if Path(universe_file).exists():
                df = pd.read_csv(universe_file)
                if "sector_etf" in df.columns:
                    mapping = dict(zip(df["symbol"], df["sector_etf"].fillna("")))
                    logger.info(
                        f"Loaded sector mapping for {sum(1 for v in mapping.values() if v)} symbols"
                    )
                    return mapping
        except Exception as e:
            logger.warning(f"Could not load sector mapping: {e}")
        return {}

    def _get_universe_closes(
        self, price_data: Dict[str, pd.DataFrame], as_of: pd.Timestamp = None
    ) -> pd.DataFrame:
        """Build/cache universe closes DataFrame for breadth features."""
        if self._universe_closes is None:
            self._universe_closes = self.ml_trainer._build_universe_closes(price_data)
        if as_of is not None and not self._universe_closes.empty:
            return self._universe_closes[self._universe_closes.index <= as_of]
        return self._universe_closes

    # ------------------------------------------------------------------
    # Warm-up: train everything from scratch if no models exist
    # ------------------------------------------------------------------

    def warm_up(
        self,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        as_of_date: pd.Timestamp = None,
    ):
        """
        One-time setup: train ML model and regime model if none exist.
        Call this before the first run_daily() with all available data.
        """
        if self._warmup_done:
            return

        # Build universe closes cache for breadth features
        self._universe_closes = self.ml_trainer._build_universe_closes(price_data)

        # --- Train regime model ---
        if index_df is not None and not index_df.empty and len(index_df) > 200:
            logger.info("Warm-up: training regime model...")
            self._train_regime(index_df)

        # --- Train ML model if none loaded ---
        if self.ml_trainer.current_model is None:
            logger.info("Warm-up: no ML model found, training from scratch...")
            self._retrain(price_data, index_df, as_of_date or pd.Timestamp.today())
        else:
            logger.info(
                f"Warm-up: loaded ML model v{self.ml_trainer.model_version} "
                f"(AUC={self.ml_trainer.current_meta.get('oos_roc_auc', 'N/A')})"
            )

        self._warmup_done = True
        self.audit.log("WARMUP_COMPLETE", {
            "ml_model_version": self.ml_trainer.model_version,
            "ml_model_loaded": self.ml_trainer.current_model is not None,
            "regime_model_loaded": self.regime_model is not None,
        })

    # ------------------------------------------------------------------
    # Main daily loop
    # ------------------------------------------------------------------

    def run_daily(
        self,
        current_date: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
    ) -> dict:
        """
        Execute one day of the trading loop.
        Returns dict with actions taken and current state.
        """
        actions = {
            "date": str(current_date.date()),
            "signals": [],
            "trades": [],
            "exits": [],
            "retrained": False,
            "regime": "unknown",
            "ml_model_active": self.ml_trainer.current_model is not None,
        }

        prices = {}
        for sym, df in price_data.items():
            if current_date in df.index:
                prices[sym] = df.loc[current_date, "close"]
        if not prices:
            return actions

        # 1. Update NAV
        self._update_nav(prices, current_date)

        # 2. Risk check
        alerts = self.risk_governor.periodic_check(self.portfolio_state)
        if alerts:
            for alert in alerts:
                logger.warning(f"Risk alert: {alert}")
                self.audit.log("RISK_ALERT", {"alert": alert, "date": str(current_date.date())})
        if self.risk_governor.kill_switch_active:
            actions["kill_switch"] = True
            self.daily_nav.append({
                "date": current_date,
                "nav": self.portfolio_state.nav,
                "regime": "kill_switch",
                "n_positions": len(self.open_positions),
            })
            self._save_state()
            return actions

        # 3. Regime detection (use cached model, refresh periodically)
        prev_regime = actions.get("regime", "unknown")
        regime_name = self._detect_regime(index_df, current_date)
        actions["regime"] = regime_name

        # 3b. LLM regime narration on regime change
        if regime_name != prev_regime and prev_regime != "unknown":
            self._narrate_regime_change(prev_regime, regime_name, current_date, index_df)

        # 4. Check if retrain needed (time-based + drift-based + perf-based)
        if self._should_retrain(current_date, price_data, index_df):
            self._retrain(price_data, index_df, current_date)
            actions["retrained"] = True

        # 5. Refresh news signals (periodically)
        self._refresh_news(current_date, list(price_data.keys()))
        actions["news_signals"] = len(self.news_signals)

        # 6. Check barrier exits on open positions
        exits = self._check_exits(prices, current_date)
        actions["exits"] = exits

        # 7. Generate new signals (if regime allows swing trading)
        alloc = get_regime_allocation(regime_name)
        if alloc["swing_enabled"]:
            signals = self._generate_signals(
                price_data, index_df, prices, current_date, regime_name
            )
            actions["signals"] = [s["symbol"] for s in signals]
            actions["trades"] = [s["symbol"] for s in signals if s.get("executed")]

        # 8. Run drift monitoring (weekly)
        self._check_drift(current_date, price_data, index_df)

        # 9. Record daily state
        self.daily_nav.append({
            "date": current_date,
            "nav": self.portfolio_state.nav,
            "regime": regime_name,
            "n_positions": len(self.open_positions),
        })

        # 10. Save state
        self._save_state()

        return actions

    # ------------------------------------------------------------------
    # News scanning
    # ------------------------------------------------------------------

    def _refresh_news(self, current_date: pd.Timestamp, symbols: List[str]):
        """Refresh news signals periodically."""
        if not self._news_enabled:
            return

        needs_refresh = (
            self.last_news_scan is None
            or (current_date - self.last_news_scan).days >= self._news_scan_interval_days
        )
        if not needs_refresh:
            return

        try:
            self.news_signals = generate_news_signals(
                symbols,
                min_score=self.config.get("news", {}).get("min_score", 0.15),
                max_signals=self.config.get("news", {}).get("max_signals", 30),
                include_rss=self.config.get("news", {}).get("include_rss", True),
                max_age_hours=self.config.get("news", {}).get("max_age_hours", 72),
            )
            self.last_news_scan = current_date

            if self.news_signals:
                logger.info(
                    f"News scan: {len(self.news_signals)} signals | "
                    f"top: {self.news_signals[0]['symbol']} "
                    f"({self.news_signals[0]['direction']}, "
                    f"score={self.news_signals[0]['score']:+.3f})"
                )
                self.audit.log("NEWS_SCAN", {
                    "date": str(current_date.date()),
                    "n_signals": len(self.news_signals),
                    "top_signals": [
                        {"symbol": s["symbol"], "score": s["score"], "direction": s["direction"]}
                        for s in self.news_signals[:5]
                    ],
                })

                # Generate LLM theses for top news-driven candidates
                self._generate_theses(self.news_signals[:5], current_date)
        except Exception as e:
            logger.warning(f"News scan failed: {e}")
            self.news_signals = []

    # ------------------------------------------------------------------
    # LLM Thesis Generation
    # ------------------------------------------------------------------

    def _generate_theses(self, top_signals: List[dict], current_date: pd.Timestamp):
        """Generate investment theses for top news-driven signals."""
        try:
            from src.market_intel.thesis_generator import generate_thesis

            for sig in top_signals:
                thesis = generate_thesis(
                    symbol=sig["symbol"],
                    price_data={
                        "lookback": 21,
                        "price_vs_sma": 0,
                        "rsi": 50,
                        "vol_trend": 0,
                        "ret_5d": sig.get("score", 0),
                        "ret_21d": 0,
                    },
                    agent_signal={
                        "direction": sig.get("direction", "long").lower(),
                        "strategy": "news_driven",
                        "confidence": min(1.0, abs(sig.get("score", 0)) * 2),
                        "reasoning": sig.get("reason", ""),
                    },
                    sentiment_summary="; ".join(sig.get("headlines", [])[:3]),
                    regime=sig.get("regime", "unknown"),
                    sector=sig.get("sector", "unknown"),
                )
                if thesis:
                    logger.info(
                        f"THESIS {sig['symbol']}: "
                        f"Bull({thesis.bull_upside_pct:+.0f}%) "
                        f"Bear({thesis.bear_downside_pct:+.0f}%) "
                        f"Net={thesis.net_conviction:+.2f} "
                        f"Size={thesis.sizing_suggestion}"
                    )
                    self.audit.log("THESIS_GENERATED", {
                        "date": str(current_date.date()),
                        "symbol": sig["symbol"],
                        "bull_summary": thesis.bull_summary,
                        "bear_summary": thesis.bear_summary,
                        "net_conviction": thesis.net_conviction,
                        "sizing": thesis.sizing_suggestion,
                        "catalysts": thesis.bull_catalysts,
                        "risks": thesis.bear_risks,
                    })
        except Exception as e:
            logger.debug(f"Thesis generation skipped: {e}")

    # ------------------------------------------------------------------
    # Signal generation with ML filter
    # ------------------------------------------------------------------

    def _generate_signals(
        self,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        prices: Dict[str, float],
        current_date: pd.Timestamp,
        regime_name: str,
    ) -> list:
        """Generate, filter, size, and execute swing signals."""
        swing_config = self.config.get("swing_signals", {})
        sizing_config = self.config.get("sizing", {})
        ml_threshold = self.config.get("ml", {}).get("entry_threshold", 0.55)

        candidates = generate_swing_signals(
            price_data, index_df, current_date, swing_config
        )

        executed = []
        for cand in candidates:
            sym = cand["symbol"]
            if sym in self.open_positions or sym not in prices:
                continue

            vol_df = price_data.get(sym)
            if vol_df is None:
                continue

            # Get vol
            vol_proxy = compute_vol_proxy(vol_df["close"])
            if current_date not in vol_proxy.index:
                continue
            inst_vol = vol_proxy.loc[current_date]
            if np.isnan(inst_vol) or inst_vol <= 0:
                continue

            # ML filter — this is where the model earns its keep
            sector_etf_sym = self._sector_mapping.get(sym)
            sector_etf_df = price_data.get(sector_etf_sym) if sector_etf_sym else None
            uc = self._get_universe_closes(price_data, current_date)

            ml_prob = self.ml_trainer.predict_single(
                vol_df, index_df, current_date,
                sector_etf_df=sector_etf_df,
                universe_closes=uc,
            )

            # Apply news boost to ML probability
            news_boost = get_news_boost(sym, self.news_signals)
            adjusted_prob = max(0.0, min(1.0, ml_prob + news_boost))

            # Log prediction for future drift monitoring
            self.prediction_log.append({
                "date": current_date,
                "symbol": sym,
                "ml_prob": ml_prob,
                "news_boost": news_boost,
                "adjusted_prob": adjusted_prob,
                "actual_label": None,  # Filled later when position closes
            })

            # Skip if adjusted probability is below threshold
            # (Only enforce if we have a real model, not the 0.5 default)
            if self.ml_trainer.current_model is not None and adjusted_prob < ml_threshold:
                logger.debug(
                    f"ML filter: {sym} prob={ml_prob:.3f} news={news_boost:+.3f} "
                    f"adj={adjusted_prob:.3f} < {ml_threshold}"
                )
                continue

            # Size position (use adjusted prob for sizing — news conviction matters)
            swing_nav = self.portfolio_state.sleeve_values.get("swing", 30_000)
            result = compute_swing_position_size(
                symbol=sym,
                sleeve_nav=swing_nav,
                instrument_vol=inst_vol,
                ml_prob=adjusted_prob,
                current_regime=regime_name,
                vvol_percentile=0.5,
                price=prices[sym],
                config=sizing_config,
            )

            if result["shares"] <= 0:
                continue

            # Compute barriers
            notional = result["shares"] * prices[sym]
            barriers = compute_barriers(
                prices[sym], inst_vol,
                sizing_config.get("holding_days", 10),
            )

            # Execute via OrderManager if available, else direct
            fill_price = prices[sym]
            fill_qty = result["shares"]

            if self.order_manager is not None:
                self._order_counter += 1
                order = Order(
                    order_id=f"orch-{self._order_counter:06d}",
                    symbol=sym,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    qty=result["shares"],
                    sleeve="swing",
                    tp_price=barriers["tp_price"],
                    sl_price=barriers["sl_price"],
                    sector=cand.get("sector"),
                    ml_prob=adjusted_prob,
                    created_at=current_date,
                )
                fill = self.order_manager.submit(order, current_date)
                if fill is None:
                    continue  # Rejected by risk governor or broker
                fill_price = fill.fill_price
                fill_qty = fill.filled_qty
                notional = fill_qty * fill_price
            else:
                # Direct mode (backtest) — do risk check ourselves
                allowed, reason = self.risk_governor.pre_trade_check(
                    symbol=sym,
                    side="BUY",
                    notional=notional,
                    sleeve="swing",
                    state=self.portfolio_state,
                    sector=cand.get("sector"),
                    current_date=current_date.date(),
                )
                if not allowed:
                    logger.info(f"Signal {sym} blocked by risk: {reason}")
                    continue

            # Record position
            self.open_positions[sym] = {
                "qty": fill_qty,
                "entry_price": fill_price,
                "entry_date": current_date,
                "tp_price": barriers["tp_price"],
                "sl_price": barriers["sl_price"],
                "ml_prob": ml_prob,
                "news_boost": news_boost,
                "adjusted_prob": adjusted_prob,
                "regime": regime_name,
                "notional": notional,
                "inst_vol": inst_vol,
            }
            self.portfolio_state.positions[sym] = {"notional": notional}

            news_str = f" news={news_boost:+.2f}" if news_boost != 0 else ""
            logger.info(
                f"OPEN {sym}: {fill_qty} shares @ {fill_price:.2f} | "
                f"ML={ml_prob:.2f}{news_str} adj={adjusted_prob:.2f} | "
                f"TP={barriers['tp_price']:.2f} SL={barriers['sl_price']:.2f}"
            )
            self.audit.log("TRADE_OPEN", {
                "symbol": sym,
                "shares": fill_qty,
                "price": fill_price,
                "ml_prob": ml_prob,
                "news_boost": news_boost,
                "adjusted_prob": adjusted_prob,
                "regime": regime_name,
            })

            cand["executed"] = True
            executed.append(cand)

        return executed

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        prices: Dict[str, float],
        current_date: pd.Timestamp,
    ) -> list:
        """Check TP/SL/timeout for all open positions."""
        exits = []
        holding_days = self.config.get("sizing", {}).get("holding_days", 10)

        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            price = prices.get(sym)
            if price is None:
                continue

            days_held = (current_date - pos["entry_date"]).days
            exit_reason = None

            if price >= pos["tp_price"]:
                exit_reason = "TP_HIT"
            elif price <= pos["sl_price"]:
                exit_reason = "SL_HIT"
            elif days_held > holding_days * 1.5:
                exit_reason = "TIMEOUT"

            if exit_reason:
                # Route exit through OrderManager if available
                exit_price = price
                exit_qty = pos["qty"]

                if self.order_manager is not None:
                    self._order_counter += 1
                    sell_order = Order(
                        order_id=f"orch-{self._order_counter:06d}",
                        symbol=sym,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        qty=pos["qty"],
                        sleeve="swing",
                        notes=exit_reason,
                        created_at=current_date,
                    )
                    fill = self.order_manager.submit(sell_order, current_date)
                    if fill is None:
                        continue  # Broker rejected — keep position
                    exit_price = fill.fill_price
                    exit_qty = fill.filled_qty

                pnl = (exit_price - pos["entry_price"]) * exit_qty
                pnl_pct = (exit_price / pos["entry_price"]) - 1

                logger.info(
                    f"CLOSE {sym}: {exit_reason} | PnL=${pnl:+,.2f} ({pnl_pct:+.2%}) | "
                    f"held {days_held}d"
                )

                # Record trade
                trade = {
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "qty": exit_qty,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "ml_prob": pos["ml_prob"],
                    "regime": pos["regime"],
                    "days_held": days_held,
                }
                self.trade_log.append(trade)
                exits.append(trade)

                # Update prediction log with actual label (feedback loop)
                label = 1 if pnl > 0 else 0
                for pred in reversed(self.prediction_log):
                    if pred["symbol"] == sym and pred["actual_label"] is None:
                        pred["actual_label"] = label
                        break

                # Track performance by regime for adaptation
                regime = pos["regime"]
                if regime not in self.performance_by_regime:
                    self.performance_by_regime[regime] = []
                self.performance_by_regime[regime].append(pnl_pct)

                audit_trade = {
                    k: str(v) if isinstance(v, pd.Timestamp) else v
                    for k, v in trade.items()
                }
                self.audit.log("TRADE_CLOSE", audit_trade)

                del self.open_positions[sym]
                if sym in self.portfolio_state.positions:
                    del self.portfolio_state.positions[sym]

                # Update cash
                self.portfolio_state.cash += pos["notional"] + pnl

        return exits

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _train_regime(self, index_df: pd.DataFrame):
        """Train regime model from index data."""
        if index_df.empty or len(index_df) < 100:
            return

        close = index_df["close"] if "close" in index_df.columns else index_df
        feats = build_regime_features(close, self.config)
        if feats.empty or len(feats) < 50:
            return

        self.regime_model = fit_regime_model(feats, n_regimes=4, method="kmeans")
        raw = predict_regime(self.regime_model, feats)
        self.regime_names = label_regimes(feats, raw)
        self.regime_series = smooth_regime(raw, min_persistence=3)
        self._regime_last_trained = feats.index[-1]

        logger.info(f"Regime model trained: {self.regime_names}")

    def _detect_regime(
        self,
        index_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> str:
        if index_df.empty or len(index_df) < 100:
            return "unknown"

        idx_up_to = index_df[index_df.index <= current_date]
        if len(idx_up_to) < 100:
            return "unknown"

        # Refresh regime model every ~63 trading days (quarterly)
        needs_refresh = (
            self.regime_model is None
            or self._regime_last_trained is None
            or (current_date - self._regime_last_trained).days > 63
        )

        if needs_refresh:
            self._train_regime(idx_up_to)

        if self.regime_series is not None and current_date in self.regime_series.index:
            regime_id = self.regime_series.loc[current_date]
            return self.regime_names.get(regime_id, "unknown")

        # Predict for just today using latest features
        try:
            feats = build_regime_features(idx_up_to["close"], self.config)
            if feats.empty:
                return "unknown"
            today_feats = feats.iloc[[-1]]
            pred = predict_regime(self.regime_model, today_feats)
            return self.regime_names.get(pred.iloc[0], "unknown")
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # LLM Regime Narration
    # ------------------------------------------------------------------

    def _narrate_regime_change(
        self,
        prev_regime: str,
        new_regime: str,
        current_date: pd.Timestamp,
        index_df: pd.DataFrame,
    ):
        """Generate LLM narrative when regime changes."""
        try:
            from src.market_intel.regime_narrator import narrate_regime_change

            # Build feature context for narrator
            feats = build_regime_features(
                index_df[index_df.index <= current_date]["close"], self.config
            )
            if feats.empty:
                return

            last_feats = feats.iloc[-1].to_dict()
            alloc = get_regime_allocation(new_regime)

            narrative = narrate_regime_change(
                prev_regime=prev_regime,
                new_regime=new_regime,
                date=str(current_date.date()),
                regime_features=last_feats,
                swing_mult=alloc.get("swing_multiplier", 1.0),
                swing_enabled=alloc.get("swing_enabled", True),
            )

            logger.info(
                f"REGIME CHANGE: {prev_regime} -> {new_regime}\n"
                f"  Narrative: {narrative.narrative}\n"
                f"  Actions: {', '.join(narrative.action_items)}"
            )

            self.audit.log("REGIME_CHANGE_NARRATED", {
                "date": str(current_date.date()),
                "prev_regime": prev_regime,
                "new_regime": new_regime,
                "narrative": narrative.narrative,
                "drivers": narrative.likely_drivers,
                "actions": narrative.action_items,
                "risk_watchlist": narrative.risk_watchlist,
                "opportunity_sectors": narrative.opportunity_sectors,
                "avoid_sectors": narrative.avoid_sectors,
            })
        except Exception as e:
            logger.debug(f"Regime narration skipped: {e}")

    # ------------------------------------------------------------------
    # Drift monitoring (runs weekly as part of daily loop)
    # ------------------------------------------------------------------

    def _check_drift(
        self,
        current_date: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
    ):
        """Weekly drift check — monitors prediction quality and feature drift."""
        check_interval = self.config.get("ml", {}).get(
            "retraining", {}
        ).get("drift_check_interval_days", 7)

        if (
            self.last_drift_check is not None
            and (current_date - self.last_drift_check).days < check_interval
        ):
            return  # Not time yet

        self.last_drift_check = current_date

        # --- Prediction quality monitoring ---
        preds_with_labels = [
            p for p in self.prediction_log if p["actual_label"] is not None
        ]
        if len(preds_with_labels) >= 20:
            pred_df = pd.DataFrame(preds_with_labels)
            metrics = compute_live_metrics(pred_df, lookback_days=90)
            self.audit.log("DRIFT_CHECK", {
                "date": str(current_date.date()),
                "n_predictions": len(preds_with_labels),
                "status": metrics.get("status", "unknown"),
                "live_auc": metrics.get("live_auc"),
                "live_brier": metrics.get("live_brier"),
            })
            if metrics.get("status") in ("RETRAIN_URGENT", "DISABLE_STRATEGY"):
                logger.warning(f"Performance degraded: {metrics}")

        # --- Feature drift via PSI ---
        if (
            self.ml_trainer.current_meta
            and index_df is not None
            and not index_df.empty
        ):
            for sym, df in list(price_data.items())[:3]:  # Check top 3 symbols
                try:
                    recent = df.loc[:current_date].tail(60)
                    historical = df.loc[:current_date].tail(300)
                    if len(recent) < 30 or len(historical) < 100:
                        continue

                    idx_recent = index_df.loc[:current_date].tail(60)
                    idx_hist = index_df.loc[:current_date].tail(300)

                    current_feats = build_features(recent, idx_recent, self.config)
                    train_feats = build_features(historical, idx_hist, self.config)

                    common_cols = [
                        c for c in train_feats.columns if c in current_feats.columns
                    ]
                    if common_cols and len(current_feats) >= 10:
                        drift = monitor_feature_drift(
                            train_feats[common_cols],
                            current_feats[common_cols],
                        )
                        if drift.get("alerts"):
                            logger.info(
                                f"Drift check ({sym}): "
                                f"{len(drift['alerts'])} features drifted"
                            )
                        if drift.get("requires_retrain"):
                            logger.warning(
                                f"Feature drift detected in {sym}: {drift['alerts']}"
                            )
                            self.audit.log("FEATURE_DRIFT", {
                                "date": str(current_date.date()),
                                "symbol": sym,
                                "drifted_features": drift["alerts"],
                            })
                            break  # One symbol drifting is enough signal
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Retrain logic
    # ------------------------------------------------------------------

    def _should_retrain(
        self,
        current_date: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
    ) -> bool:
        """Decide if ML model needs retraining."""
        retrain_config = self.config.get("ml", {}).get("retraining", {})
        max_days = retrain_config.get("max_days_between_trains", 90)

        # Never trained and no model loaded
        if self.ml_trainer.current_model is None and self.last_retrain_date is None:
            return True

        # Never retrained in this session
        if self.last_retrain_date is None:
            # We have a loaded model — check if it's stale
            if self.ml_trainer.current_meta:
                age = self.ml_trainer.current_meta.get("age_days", 999)
                if age > max_days:
                    return True
            return False

        days_since = (current_date - self.last_retrain_date).days

        # Scheduled retrain
        if days_since >= max_days:
            return True

        # Performance-based retrain: check recent predictions
        preds_with_labels = [
            p for p in self.prediction_log if p["actual_label"] is not None
        ]
        if len(preds_with_labels) >= 30:
            pred_df = pd.DataFrame(preds_with_labels)
            metrics = compute_live_metrics(pred_df, lookback_days=60)
            if metrics.get("status") in ("RETRAIN_URGENT", "DISABLE_STRATEGY"):
                logger.warning(f"Performance degradation triggers retrain: {metrics}")
                return True

        return False

    def _retrain(
        self,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ):
        """Execute model retraining."""
        logger.info(f"Retraining ML model (triggered at {current_date.date()})...")

        model, report = self.ml_trainer.walk_forward_train(
            price_data, index_df, self.config
        )

        self.last_retrain_date = current_date
        self.audit.log("MODEL_RETRAIN", {
            "date": str(current_date.date()),
            "status": report.get("status"),
            "avg_oos_auc": report.get("avg_oos_auc"),
            "n_folds": report.get("n_folds"),
        })

        if model is not None:
            logger.info(f"New model deployed: AUC={report.get('avg_oos_auc')}")
        else:
            logger.warning(f"Retrain failed or below threshold: {report.get('status')}")

    # ------------------------------------------------------------------
    # NAV and state management
    # ------------------------------------------------------------------

    def _update_nav(self, prices: Dict[str, float], current_date: pd.Timestamp):
        """Mark all positions to market and update portfolio state."""
        position_value = sum(
            pos["qty"] * prices.get(sym, pos["entry_price"])
            for sym, pos in self.open_positions.items()
        )
        total_nav = self.portfolio_state.cash + position_value + \
            self.portfolio_state.sleeve_values.get("core", 0)

        self.portfolio_state.nav = total_nav
        self.portfolio_state.peak_nav = max(self.portfolio_state.peak_nav, total_nav)
        self.portfolio_state.day_start_nav = total_nav
        self.portfolio_state.sleeve_values["swing"] = (
            self.portfolio_state.cash + position_value
        )

    def _save_state(self):
        """Persist orchestrator state to disk."""
        state = {
            "last_retrain_date": str(self.last_retrain_date) if self.last_retrain_date else None,
            "last_drift_check": str(self.last_drift_check) if self.last_drift_check else None,
            "model_version": self.ml_trainer.model_version,
            "open_positions": {
                sym: {
                    k: str(v) if isinstance(v, pd.Timestamp) else v
                    for k, v in pos.items()
                }
                for sym, pos in self.open_positions.items()
            },
            "n_trades": len(self.trade_log),
            "portfolio_nav": self.portfolio_state.nav,
            "performance_by_regime": {
                k: {"n_trades": len(v), "avg_return": float(np.mean(v)) if v else 0}
                for k, v in self.performance_by_regime.items()
            },
        }
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def _load_state(self):
        """Load persisted state from disk."""
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text())
            if state.get("last_retrain_date"):
                self.last_retrain_date = pd.Timestamp(state["last_retrain_date"])
            if state.get("last_drift_check"):
                self.last_drift_check = pd.Timestamp(state["last_drift_check"])
            logger.info(
                f"Loaded state: model v{state.get('model_version', 0)}, "
                f"{state.get('n_trades', 0)} historical trades"
            )
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    # ------------------------------------------------------------------
    # Analytics & feedback
    # ------------------------------------------------------------------

    def get_performance_summary(self) -> dict:
        """Get comprehensive performance analytics."""
        if not self.trade_log:
            return {"n_trades": 0}

        trades = pd.DataFrame(self.trade_log)
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]

        summary = {
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades) if len(trades) > 0 else 0,
            "avg_win": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
            "avg_loss": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
            "total_pnl": trades["pnl"].sum(),
            "avg_days_held": trades["days_held"].mean(),
            "by_exit_reason": trades.groupby("exit_reason")["pnl"].agg(
                ["count", "sum", "mean"]
            ).to_dict() if len(trades) > 0 else {},
            "by_regime": {},
        }

        # Per-regime breakdown
        for regime, returns in self.performance_by_regime.items():
            if returns:
                summary["by_regime"][regime] = {
                    "n_trades": len(returns),
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                    "avg_return": float(np.mean(returns)),
                    "total_return": sum(returns),
                }

        return summary

    def get_adaptive_recommendations(self) -> dict:
        """
        Analyze trade history to suggest parameter adjustments.
        This is how the bot learns from its own performance.
        """
        recs = {}

        if len(self.trade_log) < 20:
            return {"status": "insufficient_trades"}

        trades = pd.DataFrame(self.trade_log)

        # 1. ML threshold tuning
        if "ml_prob" in trades.columns:
            high_conf = trades[trades["ml_prob"] >= 0.70]
            low_conf = trades[(trades["ml_prob"] >= 0.55) & (trades["ml_prob"] < 0.65)]
            if len(high_conf) > 5 and len(low_conf) > 5:
                high_wr = (high_conf["pnl"] > 0).mean()
                low_wr = (low_conf["pnl"] > 0).mean()
                if high_wr > low_wr + 0.10:
                    recs["raise_threshold"] = {
                        "reason": f"High-conf WR={high_wr:.0%} vs low-conf WR={low_wr:.0%}",
                        "suggested": 0.65,
                    }

        # 2. Regime-specific sizing
        for regime, returns in self.performance_by_regime.items():
            if len(returns) >= 10:
                wr = sum(1 for r in returns if r > 0) / len(returns)
                if wr < 0.35:
                    recs[f"reduce_{regime}"] = {
                        "reason": f"Win rate {wr:.0%} in {regime}",
                        "suggested": "Reduce or disable swing in this regime",
                    }

        # 3. Holding period analysis
        for reason in ["TP_HIT", "SL_HIT", "TIMEOUT"]:
            subset = trades[trades["exit_reason"] == reason]
            if len(subset) > 5:
                avg_pnl = subset["pnl_pct"].mean()
                if reason == "TIMEOUT" and avg_pnl < -0.01:
                    recs["shorten_holding"] = {
                        "reason": f"Timeouts avg {avg_pnl:.1%} return",
                        "suggested": "Reduce holding_days",
                    }

        return recs

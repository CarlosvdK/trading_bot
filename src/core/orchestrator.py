"""
Autonomous Orchestrator — the daily brain loop.

Handles the complete lifecycle:
  1. Fetch new data
  2. Check drift / model health → retrain if needed
  3. Detect regime
  4. Generate signals → ML filter → size → risk check → execute
  5. Monitor open positions (barrier exits)
  6. Log everything
  7. Adapt parameters based on feedback

This is what makes the bot self-sufficient.
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.provider import CSVDataProvider
from src.data.missing import handle_missing_data
from src.utils.config_loader import load_config
from src.risk.risk_governor import RiskGovernor, RiskConfig, PortfolioState
from src.ml.trainer import MLTrainer
from src.ml.features import build_features
from src.ml.labeler import compute_vol_proxy
from src.ml.regime import (
    build_regime_features,
    fit_regime_model,
    predict_regime,
    label_regimes,
    get_regime_allocation,
    smooth_regime,
)
from src.ml.drift import monitor_feature_drift, compute_live_metrics, should_retrain
from src.swing.signals import generate_swing_signals
from src.swing.sizing import compute_swing_position_size, compute_barriers
from src.utils.audit import AuditLogger

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Self-sufficient trading orchestrator.
    Manages the full lifecycle: data → learn → predict → trade → adapt.
    """

    def __init__(self, config: dict, root_dir: str):
        self.config = config
        self.root_dir = Path(root_dir)
        self.state_file = self.root_dir / "state" / "orchestrator_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Core components
        self.risk_config = RiskConfig(**{
            k: v for k, v in config.get("risk", {}).items()
            if k in RiskConfig.__dataclass_fields__
        })
        self.risk_governor = RiskGovernor(self.risk_config)
        self.ml_trainer = MLTrainer(
            config, models_dir=str(self.root_dir / "models")
        )
        self.audit = AuditLogger(str(self.root_dir / "logs" / "audit.jsonl"))

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

        # Adaptive state
        self.performance_by_regime: Dict[str, list] = {}
        self.last_retrain_date = None
        self.last_drift_check = None

        # Load persisted state
        self._load_state()

        # Try to load latest ML model
        self.ml_trainer.load_latest_model()

    def _init_portfolio_state(self) -> PortfolioState:
        initial_nav = self.config.get("portfolio", {}).get("initial_nav", 100_000)
        return PortfolioState(
            nav=initial_nav,
            peak_nav=initial_nav,
            cash=initial_nav * 0.10,
            sleeve_values={
                "swing": initial_nav * 0.30,
                "core": initial_nav * 0.60,
            },
            positions={},
            day_start_nav=initial_nav,
        )

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

        # 3. Regime detection
        regime_name = self._detect_regime(index_df, current_date)
        actions["regime"] = regime_name

        # 4. Check if retrain needed
        if self._should_retrain(current_date, price_data, index_df):
            self._retrain(price_data, index_df, current_date)
            actions["retrained"] = True

        # 5. Check barrier exits on open positions
        exits = self._check_exits(prices, current_date)
        actions["exits"] = exits

        # 6. Generate new signals (if regime allows)
        alloc = get_regime_allocation(regime_name)
        if alloc["swing_enabled"]:
            signals = self._generate_signals(
                price_data, index_df, prices, current_date, regime_name
            )
            actions["signals"] = [s["symbol"] for s in signals]
            actions["trades"] = [s["symbol"] for s in signals if s.get("executed")]

        # 7. Record daily state
        self.daily_nav.append({
            "date": current_date,
            "nav": self.portfolio_state.nav,
            "regime": regime_name,
            "n_positions": len(self.open_positions),
        })

        # 8. Save state
        self._save_state()

        return actions

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
            ml_prob = self.ml_trainer.predict_single(
                vol_df, index_df, current_date
            )

            # Log prediction for future drift monitoring
            self.prediction_log.append({
                "date": current_date,
                "symbol": sym,
                "ml_prob": ml_prob,
                "actual_label": None,  # Filled later when position closes
            })

            # Size position
            swing_nav = self.portfolio_state.sleeve_values.get("swing", 30_000)
            result = compute_swing_position_size(
                symbol=sym,
                sleeve_nav=swing_nav,
                instrument_vol=inst_vol,
                ml_prob=ml_prob,
                current_regime=regime_name,
                vvol_percentile=0.5,
                price=prices[sym],
                config=sizing_config,
            )

            if result["shares"] <= 0:
                continue

            # Risk check
            notional = result["shares"] * prices[sym]
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

            # Compute barriers
            barriers = compute_barriers(
                prices[sym], inst_vol,
                sizing_config.get("holding_days", 10),
            )

            # Execute (in paper/backtest this opens position)
            self.open_positions[sym] = {
                "qty": result["shares"],
                "entry_price": prices[sym],
                "entry_date": current_date,
                "tp_price": barriers["tp_price"],
                "sl_price": barriers["sl_price"],
                "ml_prob": ml_prob,
                "regime": regime_name,
                "notional": notional,
                "inst_vol": inst_vol,
            }
            self.portfolio_state.positions[sym] = {"notional": notional}

            logger.info(
                f"OPEN {sym}: {result['shares']} shares @ {prices[sym]:.2f} | "
                f"ML={ml_prob:.2f} | TP={barriers['tp_price']:.2f} SL={barriers['sl_price']:.2f}"
            )
            self.audit.log("TRADE_OPEN", {
                "symbol": sym,
                "shares": result["shares"],
                "price": prices[sym],
                "ml_prob": ml_prob,
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
                pnl = (price - pos["entry_price"]) * pos["qty"]
                pnl_pct = (price / pos["entry_price"]) - 1

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
                    "exit_price": price,
                    "qty": pos["qty"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "ml_prob": pos["ml_prob"],
                    "regime": pos["regime"],
                    "days_held": days_held,
                }
                self.trade_log.append(trade)
                exits.append(trade)

                # Update prediction log with actual label
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

        feats = build_regime_features(idx_up_to["close"], self.config)
        if feats.empty:
            return "unknown"

        # Retrain regime model periodically
        if self.regime_model is None or len(feats) % 63 == 0:
            self.regime_model = fit_regime_model(feats, n_regimes=4, method="kmeans")
            raw = predict_regime(self.regime_model, feats)
            self.regime_names = label_regimes(feats, raw)
            self.regime_series = smooth_regime(raw, min_persistence=3)

        if self.regime_series is not None and current_date in self.regime_series.index:
            regime_id = self.regime_series.loc[current_date]
            return self.regime_names.get(regime_id, "unknown")

        # Predict for just today
        try:
            today_feats = feats.iloc[[-1]]
            pred = predict_regime(self.regime_model, today_feats)
            return self.regime_names.get(pred.iloc[0], "unknown")
        except Exception:
            return "unknown"

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
        check_interval = retrain_config.get("drift_check_interval_days", 7)

        # Never retrained
        if self.last_retrain_date is None:
            return True

        days_since = (current_date - self.last_retrain_date).days

        # Scheduled retrain
        if days_since >= max_days:
            return True

        # Weekly drift check
        if (
            self.last_drift_check is None
            or (current_date - self.last_drift_check).days >= check_interval
        ):
            self.last_drift_check = current_date

            # Check prediction drift
            preds_with_labels = [
                p for p in self.prediction_log
                if p["actual_label"] is not None
            ]
            if len(preds_with_labels) >= 30:
                pred_df = pd.DataFrame(preds_with_labels)
                metrics = compute_live_metrics(pred_df, lookback_days=90)
                if metrics.get("status") in ("RETRAIN_URGENT", "DISABLE_STRATEGY"):
                    logger.warning(f"Performance degraded: {metrics}")
                    return True

            # Check feature drift
            if self.ml_trainer.current_meta:
                train_feats_list = self.ml_trainer.current_meta.get("feature_list", [])
                if train_feats_list and index_df is not None and not index_df.empty:
                    # Build current features for drift check
                    for sym, df in list(price_data.items())[:1]:
                        try:
                            current_feats = build_features(
                                df.loc[:current_date].tail(60),
                                index_df.loc[:current_date].tail(60),
                                self.config,
                            )
                            if len(current_feats) >= 10:
                                train_feats = build_features(
                                    df.tail(300),
                                    index_df.tail(300),
                                    self.config,
                                )
                                common_cols = [
                                    c for c in train_feats.columns
                                    if c in current_feats.columns
                                ]
                                if common_cols:
                                    drift = monitor_feature_drift(
                                        train_feats[common_cols],
                                        current_feats[common_cols],
                                    )
                                    if drift.get("requires_retrain"):
                                        logger.warning(f"Feature drift detected: {drift['alerts']}")
                                        return True
                        except Exception:
                            pass

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
                k: {"n_trades": len(v), "avg_return": np.mean(v) if v else 0}
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
                    "avg_return": np.mean(returns),
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

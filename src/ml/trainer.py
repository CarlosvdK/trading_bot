"""
ML Training Pipeline — ties features, labels, walk-forward, calibration
into a single train → evaluate → deploy workflow.

This is the brain: it learns from historical data and continuously
retrains itself as new data arrives.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

from src.ml.features import build_features, winsorize_zscore
from src.ml.labeler import build_labels, label_quality_report, compute_vol_proxy
from src.ml.validation import walk_forward_splits, purge_training_labels, leakage_audit
from src.ml.calibration import calibrate_model, reliability_diagram
from src.ml.persistence import save_model, load_model_with_meta

logger = logging.getLogger(__name__)


class MLTrainer:
    """
    End-to-end ML training pipeline with walk-forward validation,
    calibration, and auto-deployment.

    The trainer is designed to be called repeatedly as new data arrives.
    It tracks its own state (last train date, model version) and decides
    when retraining is needed.
    """

    def __init__(self, config: dict, models_dir: str = "models"):
        self.config = config
        self.models_dir = models_dir
        self.current_model = None
        self.current_meta = None
        self.model_version = 0
        self.train_history = []

        # ML hyperparameters — can be overridden via config
        ml_config = config.get("ml", {})
        self.n_estimators = ml_config.get("n_estimators", 200)
        self.max_depth = ml_config.get("max_depth", 4)
        self.learning_rate = ml_config.get("learning_rate", 0.05)
        self.min_samples_leaf = ml_config.get("min_samples_leaf", 20)
        self.entry_threshold = ml_config.get("entry_threshold", 0.60)
        self.min_oos_auc = ml_config.get("min_oos_auc_to_deploy", 0.55)
        self.cal_method = ml_config.get("calibration_method", "isotonic")
        self.cal_fraction = ml_config.get("calibration_fraction", 0.20)

        # Labeling params
        label_config = config.get("labeling", {})
        self.k1 = label_config.get("k1", 2.0)
        self.k2 = label_config.get("k2", 1.0)
        self.horizon = label_config.get("horizon_days", 10)
        self.vol_window = label_config.get("vol_window", 21)

    def build_training_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        signal_dates: pd.DatetimeIndex,
        train_end: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix (X) and labels (y) for training.
        Only uses data up to train_end (no future leakage).
        """
        all_X = []
        all_y = []

        for symbol, df in price_data.items():
            if symbol == self.config.get("features", {}).get("index_symbol", "SPY"):
                continue  # Don't trade the index itself

            df_train = df[df.index <= train_end]
            idx_train = index_df[index_df.index <= train_end]

            if len(df_train) < 100:
                continue

            # Build features
            feats = build_features(df_train, idx_train, self.config)
            if feats.empty:
                continue

            # Build labels — use all available signal dates within training period
            sym_signals = signal_dates[signal_dates.isin(df_train.index)]
            sym_signals = sym_signals[sym_signals <= train_end - pd.Timedelta(days=self.horizon + 5)]

            if len(sym_signals) < 20:
                # If no explicit signals, generate labels at regular intervals
                valid_idx = df_train.index[63:-self.horizon-5]  # Skip warmup and tail
                step = max(1, len(valid_idx) // 200)  # ~200 samples per symbol
                sym_signals = valid_idx[::step]

            labels = build_labels(
                df_train, sym_signals,
                k1=self.k1, k2=self.k2,
                vol_window=self.vol_window, horizon=self.horizon,
            )
            if labels.empty:
                continue

            # Align features with labels
            common = feats.index.intersection(labels.index)
            if len(common) < 10:
                continue

            X_sym = feats.loc[common]
            y_sym = labels.loc[common, "label"]

            # Tag with symbol for tracking
            X_sym = X_sym.copy()
            X_sym["_symbol"] = symbol

            all_X.append(X_sym)
            all_y.append(y_sym)

        if not all_X:
            return pd.DataFrame(), pd.Series(dtype=float)

        X = pd.concat(all_X)
        y = pd.concat(all_y)

        logger.info(
            f"Training data: {len(X)} samples, "
            f"{y.sum():.0f} positive ({y.mean():.1%}), "
            f"{len(X) - y.sum():.0f} negative"
        )
        return X, y

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[object, dict]:
        """
        Train a GBT model with calibration on held-out set.
        Returns (calibrated_model, metrics_dict).
        """
        # Remove tracking columns
        feature_cols = [c for c in X.columns if not c.startswith("_")]
        X_clean = X[feature_cols].fillna(0)

        # Quality check
        quality = label_quality_report(y)
        logger.info(f"Label quality: {quality}")

        # Train/calibration split (time-ordered, not random)
        n = len(X_clean)
        cal_size = max(int(n * self.cal_fraction), 50)
        train_size = n - cal_size

        X_train = X_clean.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_cal = X_clean.iloc[train_size:]
        y_cal = y.iloc[train_size:]

        # Determine class weight
        use_balanced = quality.get("warning", "") != "OK"

        # Train GBT
        gbt = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            subsample=0.8,
            random_state=42,
        )
        # GBT doesn't support class_weight directly — use sample_weight
        if use_balanced:
            pos_weight = len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1
            neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))
            weights = y_train.map({1: pos_weight, 0: neg_weight})
            gbt.fit(X_train, y_train, sample_weight=weights)
        else:
            gbt.fit(X_train, y_train)

        # Calibrate
        cal_model = calibrate_model(gbt, X_cal, y_cal, method=self.cal_method)

        # Evaluate on calibration set
        y_prob = cal_model.predict_proba(X_cal)[:, 1]
        auc = roc_auc_score(y_cal, y_prob) if len(y_cal.unique()) > 1 else 0.5
        y_pred = (y_prob >= self.entry_threshold).astype(int)
        f1 = f1_score(y_cal, y_pred, zero_division=0)

        # Reliability check
        rel = reliability_diagram(y_cal.values, y_prob)

        # Feature importance
        importance = dict(zip(feature_cols, gbt.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

        metrics = {
            "oos_roc_auc": round(auc, 4),
            "oos_f1": round(f1, 4),
            "ece": rel["ece"],
            "well_calibrated": rel["well_calibrated"],
            "n_train_samples": train_size,
            "n_cal_samples": cal_size,
            "label_balance": quality,
            "top_features": top_features,
            "feature_list": feature_cols,
        }

        logger.info(
            f"Model trained: AUC={auc:.4f}, F1={f1:.4f}, "
            f"ECE={rel['ece']:.4f}, "
            f"top feature={top_features[0][0] if top_features else 'N/A'}"
        )

        return cal_model, metrics

    def walk_forward_train(
        self,
        price_data: Dict[str, pd.DataFrame],
        index_df: pd.DataFrame,
        config: dict = None,
    ) -> Tuple[Optional[object], dict]:
        """
        Full walk-forward training pipeline.
        Trains on expanding windows, evaluates OOS on each fold,
        deploys the final model if it meets quality gates.
        """
        config = config or self.config
        wf_config = config.get("walk_forward", {})

        # Use index dates as the master timeline
        if index_df.empty:
            all_dates = set()
            for df in price_data.values():
                all_dates.update(df.index)
            master_dates = pd.DatetimeIndex(sorted(all_dates))
        else:
            master_dates = index_df.index

        splits = list(walk_forward_splits(
            master_dates,
            initial_train_days=wf_config.get("initial_train_days", 756),
            test_days=wf_config.get("test_days", 126),
            step_days=wf_config.get("step_days", 63),
            embargo_days=wf_config.get("embargo_days", 12),
            expanding=wf_config.get("expanding", True),
        ))

        if not splits:
            logger.warning("No walk-forward splits possible — insufficient data")
            return None, {"status": "insufficient_data"}

        logger.info(f"Walk-forward: {len(splits)} folds")

        fold_metrics = []
        final_model = None
        final_metrics = None

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            train_end = train_idx[-1]
            test_start = test_idx[0]

            logger.info(
                f"Fold {fold_i+1}/{len(splits)}: "
                f"train up to {train_end.date()}, "
                f"test {test_start.date()} → {test_idx[-1].date()}"
            )

            # Build training data
            X_train, y_train = self.build_training_data(
                price_data, index_df,
                signal_dates=train_idx,
                train_end=train_end,
            )

            if len(X_train) < 50:
                logger.warning(f"Fold {fold_i+1}: insufficient training data ({len(X_train)} samples)")
                continue

            # Purge labels near test boundary
            safe_idx = X_train.index[X_train.index <= train_end - pd.Timedelta(days=self.horizon + 2)]
            X_train = X_train.loc[safe_idx]
            y_train = y_train.loc[safe_idx]

            if len(X_train) < 50:
                continue

            # Train
            model, metrics = self.train_model(X_train, y_train)

            # OOS evaluation on test period
            X_test, y_test = self.build_training_data(
                price_data, index_df,
                signal_dates=test_idx,
                train_end=test_idx[-1],
            )

            if len(X_test) > 10:
                feature_cols = [c for c in X_test.columns if not c.startswith("_")]
                X_test_clean = X_test[feature_cols].fillna(0)
                try:
                    y_prob_test = model.predict_proba(X_test_clean)[:, 1]
                    oos_auc = roc_auc_score(y_test, y_prob_test) if len(y_test.unique()) > 1 else 0.5
                    metrics["oos_test_auc"] = round(oos_auc, 4)
                    logger.info(f"  OOS AUC: {oos_auc:.4f}")
                except Exception as e:
                    logger.warning(f"  OOS eval failed: {e}")
                    metrics["oos_test_auc"] = None

            fold_metrics.append(metrics)
            final_model = model
            final_metrics = metrics

        if final_model is None:
            return None, {"status": "training_failed", "folds": fold_metrics}

        # Deployment gate
        avg_auc = np.mean([
            m.get("oos_test_auc", m.get("oos_roc_auc", 0))
            for m in fold_metrics if m.get("oos_test_auc") or m.get("oos_roc_auc")
        ])

        if avg_auc >= self.min_oos_auc:
            self.model_version += 1
            self.current_model = final_model
            self.current_meta = final_metrics

            # Save
            prefix = f"{self.models_dir}/trade_filter_v{self.model_version:03d}"
            save_model(final_model, {
                **final_metrics,
                "train_end": str(splits[-1][0][-1].date()),
                "avg_oos_auc": round(avg_auc, 4),
                "n_folds": len(fold_metrics),
            }, prefix)

            logger.info(
                f"Model v{self.model_version} deployed: "
                f"avg OOS AUC={avg_auc:.4f} >= {self.min_oos_auc}"
            )
        else:
            logger.warning(
                f"Model NOT deployed: avg OOS AUC={avg_auc:.4f} < {self.min_oos_auc}"
            )

        self.train_history.append({
            "version": self.model_version,
            "avg_oos_auc": round(avg_auc, 4),
            "n_folds": len(fold_metrics),
            "deployed": avg_auc >= self.min_oos_auc,
            "fold_metrics": fold_metrics,
        })

        return final_model if avg_auc >= self.min_oos_auc else None, {
            "status": "deployed" if avg_auc >= self.min_oos_auc else "below_threshold",
            "avg_oos_auc": round(avg_auc, 4),
            "n_folds": len(fold_metrics),
            "fold_metrics": fold_metrics,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get calibrated probabilities for new data.
        Returns array of probabilities, or default 0.5 if no model loaded.
        """
        if self.current_model is None:
            return np.full(len(X), 0.5)

        feature_cols = [c for c in X.columns if not c.startswith("_")]
        X_clean = X[feature_cols].fillna(0)

        try:
            return self.current_model.predict_proba(X_clean)[:, 1]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.full(len(X), 0.5)

    def predict_single(
        self,
        symbol_df: pd.DataFrame,
        index_df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> float:
        """
        Get ML probability for a single symbol on a single date.
        Used in live/paper signal pipeline.
        """
        if self.current_model is None:
            return 0.5

        try:
            feats = build_features(
                symbol_df.loc[:date], index_df.loc[:date], self.config
            )
            if feats.empty or date not in feats.index:
                return 0.5

            X = feats.loc[[date]].fillna(0)
            feature_cols = [c for c in X.columns if not c.startswith("_")]
            prob = self.current_model.predict_proba(X[feature_cols])[:, 1][0]
            return float(prob)
        except Exception as e:
            logger.warning(f"predict_single failed for {date}: {e}")
            return 0.5

    def load_latest_model(self) -> bool:
        """Try to load the most recent saved model."""
        from pathlib import Path
        import glob

        pattern = f"{self.models_dir}/trade_filter_v*.pkl"
        files = sorted(glob.glob(pattern))
        if not files:
            logger.info("No saved models found")
            return False

        latest = files[-1].replace(".pkl", "")
        try:
            model, meta = load_model_with_meta(latest)
            self.current_model = model
            self.current_meta = meta
            version_str = Path(latest).stem.split("_v")[-1]
            self.model_version = int(version_str)
            logger.info(
                f"Loaded model v{self.model_version}: "
                f"AUC={meta.get('oos_roc_auc', 'N/A')}, "
                f"age={meta.get('age_days', '?')} days"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

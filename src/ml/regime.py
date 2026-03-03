"""
Market regime detection via HMM/KMeans clustering.
Skill reference: .claude/skills/regime-detection/SKILL.md
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def build_regime_features(
    index_close: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """
    Features for regime classification.
    All backward-looking. Computed on reference index.
    """
    log_ret = np.log(index_close / index_close.shift(1))
    feats = pd.DataFrame(index=index_close.index)

    feats["ret_5d"] = np.log(index_close / index_close.shift(5))
    feats["ret_21d"] = np.log(index_close / index_close.shift(21))
    feats["ret_63d"] = np.log(index_close / index_close.shift(63))

    feats["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
    feats["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)

    feats["vol_ratio"] = feats["vol_21d"] / feats["vol_63d"]
    feats["vvol"] = feats["vol_21d"].rolling(21).std()
    feats["trend_strength"] = (log_ret > 0).rolling(21).mean()

    rolling_peak = index_close.rolling(63).max()
    feats["drawdown_63d"] = (index_close - rolling_peak) / rolling_peak

    return feats.dropna()


def fit_regime_model(
    features: pd.DataFrame,
    n_regimes: int = 4,
    method: str = "hmm",
) -> dict:
    """
    Fit a regime detection model on training features.
    Returns dict with model, scaler, and method type.
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X = scaler.fit_transform(features.fillna(0))

    if method == "hmm":
        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(X)
            return {"type": "hmm", "model": model, "scaler": scaler}
        except ImportError:
            warnings.warn("hmmlearn not installed, falling back to KMeans")

    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    model.fit(X)
    return {"type": "kmeans", "model": model, "scaler": scaler}


def predict_regime(
    model_dict: dict,
    features: pd.DataFrame,
) -> pd.Series:
    """
    Predict regime labels for given features.
    Returns integer labels indexed by date.
    """
    scaler = model_dict["scaler"]
    model = model_dict["model"]
    X = scaler.transform(features.fillna(0))
    labels = model.predict(X)
    return pd.Series(labels, index=features.index, name="regime")


def label_regimes(
    features: pd.DataFrame,
    regime_series: pd.Series,
) -> dict:
    """
    Map regime_id -> human-readable regime_name based on feature profiles.
    """
    profiles = features.copy()
    profiles["regime"] = regime_series

    summary = profiles.groupby("regime").mean()

    regime_names = {}
    for regime_id, row in summary.iterrows():
        vol = row.get("vol_21d", 0)
        trend = row.get("ret_21d", 0)

        if vol > summary["vol_21d"].median() * 1.3:
            vol_label = "high_vol"
        else:
            vol_label = "low_vol"

        if abs(trend) > summary["ret_21d"].abs().median():
            trend_label = "trending_up" if trend > 0 else "trending_down"
        else:
            trend_label = "choppy"

        regime_names[regime_id] = f"{vol_label}_{trend_label}"

    return regime_names


# Regime -> allocation mapping
REGIME_ALLOCATION = {
    "low_vol_trending_up": {"swing_multiplier": 1.0, "swing_enabled": True},
    "low_vol_choppy": {"swing_multiplier": 0.5, "swing_enabled": True},
    "low_vol_trending_down": {"swing_multiplier": 0.5, "swing_enabled": True},
    "high_vol_trending_up": {"swing_multiplier": 0.3, "swing_enabled": True},
    "high_vol_choppy": {"swing_multiplier": 0.0, "swing_enabled": False},
    "high_vol_trending_down": {"swing_multiplier": 0.0, "swing_enabled": False},
}


def get_regime_allocation(regime_name: str) -> dict:
    """Get allocation parameters for a given regime name."""
    return REGIME_ALLOCATION.get(
        regime_name, {"swing_multiplier": 0.5, "swing_enabled": True}
    )


def smooth_regime(
    regime_series: pd.Series,
    min_persistence: int = 3,
) -> pd.Series:
    """
    Prevents whipsawing between regimes on noisy days.
    Only switches regime after N consecutive days in new state.
    """
    smoothed = regime_series.copy()
    current = regime_series.iloc[0]
    pending = None
    pending_streak = 0

    for i in range(1, len(regime_series)):
        raw = regime_series.iloc[i]
        if raw == current:
            # Back to current regime — reset any pending switch
            pending = None
            pending_streak = 0
        elif raw == pending:
            # Continuation of candidate new regime
            pending_streak += 1
            if pending_streak >= min_persistence:
                current = pending
                pending = None
                pending_streak = 0
        else:
            # Different new regime — start tracking it
            pending = raw
            pending_streak = 1

        smoothed.iloc[i] = current

    return smoothed


def run_regime_walk_forward(
    index_close: pd.Series,
    config: dict,
) -> Tuple[pd.Series, dict]:
    """
    Walk-forward regime predictions. At each fold, model trained on
    data up to fold boundary, predicts next fold only.
    Returns (out-of-sample regime series, regime_names from last fold).
    """
    feat_df = build_regime_features(index_close, config)
    all_preds = []

    initial = config.get("initial_train_days", 504)
    step = config.get("step_days", 63)
    n_regimes = config.get("n_regimes", 4)
    method = config.get("regime_method", "kmeans")

    regime_names = {}

    for train_end_i in range(initial, len(feat_df), step):
        train_feats = feat_df.iloc[:train_end_i]
        test_feats = feat_df.iloc[train_end_i : train_end_i + step]

        if len(test_feats) == 0:
            break

        model_dict = fit_regime_model(train_feats, n_regimes, method)
        preds = predict_regime(model_dict, test_feats)
        all_preds.append(preds)

        # Label regimes from training data
        train_preds = predict_regime(model_dict, train_feats)
        regime_names = label_regimes(train_feats, train_preds)

    if not all_preds:
        return pd.Series(dtype=int), {}

    combined = pd.concat(all_preds).sort_index()
    return combined, regime_names

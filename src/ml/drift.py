"""
Feature drift detection via PSI, performance monitoring, retrain triggers.
Skill reference: .claude/skills/model-calibration-drift/SKILL.md
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Population Stability Index.

    PSI < 0.10:  No significant shift
    PSI 0.10-0.20: Moderate shift — monitor
    PSI > 0.20:  Significant shift — retrain recommended
    PSI > 0.25:  Severe shift — retrain immediately
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    expected_pct = expected_counts / (len(expected) + epsilon)
    actual_pct = actual_counts / (len(actual) + epsilon)

    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def monitor_feature_drift(
    train_features: pd.DataFrame,
    current_features: pd.DataFrame,
    alert_threshold: float = 0.20,
    retrain_threshold: float = 0.25,
) -> dict:
    """
    Compute PSI for all features. Return drift report.
    """
    report = {}
    alerts = []

    for col in train_features.columns:
        if col not in current_features.columns:
            continue
        exp = train_features[col].dropna().values
        act = current_features[col].dropna().values
        if len(exp) < 30 or len(act) < 10:
            continue

        psi = compute_psi(exp, act)
        status = "STABLE"
        if psi > retrain_threshold:
            status = "RETRAIN_NOW"
            alerts.append(f"CRITICAL: {col} PSI={psi:.3f} > {retrain_threshold}")
        elif psi > alert_threshold:
            status = "MONITOR"
            alerts.append(f"WARNING: {col} PSI={psi:.3f} > {alert_threshold}")

        report[col] = {"psi": round(psi, 4), "status": status}

    return {
        "feature_report": report,
        "alerts": alerts,
        "requires_retrain": any(
            r["status"] == "RETRAIN_NOW" for r in report.values()
        ),
        "n_features_drifted": sum(
            1 for r in report.values() if r["status"] != "STABLE"
        ),
    }


def compute_live_metrics(
    predictions: pd.DataFrame,
    lookback_days: int = 90,
) -> dict:
    """
    Compute model performance metrics on recent live predictions.
    predictions must have columns: date, ml_prob, actual_label
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

    cutoff = predictions["date"].max() - pd.Timedelta(days=lookback_days)
    recent = predictions[predictions["date"] >= cutoff].dropna()

    if len(recent) < 30:
        return {"status": "insufficient_data", "n_samples": len(recent)}

    y_true = recent["actual_label"].values
    y_prob = recent["ml_prob"].values
    y_pred = (y_prob >= 0.6).astype(int)

    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else 0.5
    )
    brier = brier_score_loss(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "n_samples": len(recent),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "f1": round(f1, 4),
        "win_rate": round(float(y_true.mean()), 4),
        "status": "OK",
    }

    if auc < 0.50:
        metrics["status"] = "DISABLE_STRATEGY"
        metrics["alert"] = f"AUC {auc:.3f} below random. Disable Swing immediately."
    elif auc < 0.53:
        metrics["status"] = "RETRAIN_URGENT"
        metrics["alert"] = f"AUC {auc:.3f} degraded. Retrain model."

    return metrics


def should_retrain(
    drift_report: dict,
    performance_metrics: dict,
    days_since_last_train: int,
    max_days_between_trains: int = 90,
) -> Tuple[bool, str]:
    """
    Returns (should_retrain, reason).
    """
    reasons = []

    if days_since_last_train >= max_days_between_trains:
        reasons.append(
            f"Scheduled: {days_since_last_train} days since last train"
        )

    if drift_report.get("requires_retrain"):
        reasons.append(
            f"Feature drift: {drift_report['n_features_drifted']} features drifted"
        )

    status = performance_metrics.get("status", "OK")
    if status in ("RETRAIN_URGENT", "DISABLE_STRATEGY"):
        reasons.append(
            f"Performance degradation: AUC={performance_metrics.get('roc_auc')}"
        )

    return bool(reasons), "; ".join(reasons) if reasons else "No retrain needed"

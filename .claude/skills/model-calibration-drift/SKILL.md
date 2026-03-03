---
name: model-calibration-drift
description: ML model calibration (isotonic/Platt), PSI drift detection, performance monitoring, retrain triggers. Use whenever calibrating probabilities, monitoring model drift, computing PSI, or deciding when to retrain.
triggers:
  - calibration
  - PSI
  - drift
  - Brier score
  - retrain
  - reliability diagram
  - ECE
  - model monitoring
  - population stability
priority: P2
---

# Skill: ML Model Calibration & Drift Monitoring

## What This Skill Is
A trade filter model that outputs a "probability" of 0.72 means nothing unless those probabilities are calibrated — i.e., 72% of trades with a score of 0.72 should actually be profitable. This skill covers: calibration, reliability diagrams, population stability index (PSI) for drift detection, and model retraining triggers.

---

## Why Calibration Matters for Position Sizing

If you use raw GBT probabilities for sizing (bigger bet = higher probability), you need those probabilities to be accurate. Gradient Boosted Trees are systematically miscalibrated — they cluster near 0 and 1. Without calibration, your sizing will be wrong.

```
Raw GBT output:  0.88 → actual win rate: 62%
Calibrated:      0.88 → actual win rate: 87%

Result of miscalibration: you size as if you have 87% edge when you only have 62%.
```

---

## Calibration Implementation

```python
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

def calibrate_model(
    base_model,              # Fitted sklearn estimator
    X_cal: pd.DataFrame,     # Calibration features (held-out from training)
    y_cal: pd.Series,        # Calibration labels
    method: str = "isotonic",  # "isotonic" or "sigmoid" (Platt)
):
    """
    Post-hoc calibration of a fitted classifier.
    
    Use isotonic regression (non-parametric) unless you have fewer than
    1000 calibration samples, in which case use sigmoid (Platt scaling).
    
    Args:
        base_model:  Pre-fitted classifier (e.g., from walk-forward fold)
        X_cal:       Calibration set features (20% of training window, held out)
        y_cal:       Calibration set labels
        method:      "isotonic" or "sigmoid"
    
    Returns:
        Calibrated classifier that outputs true probabilities
    """
    n_cal = len(X_cal)
    if n_cal < 300:
        print(f"WARNING: Only {n_cal} calibration samples — use 'sigmoid' method")
        method = "sigmoid"

    calibrated = CalibratedClassifierCV(base_model, cv="prefit", method=method)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str = None,
) -> dict:
    """
    Plot reliability (calibration) diagram.
    A perfectly calibrated model lies on the diagonal.
    
    Returns dict with fraction_of_positives and mean_predicted_value per bin.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(len(frac_pos)):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        bin_weight = bin_mask.sum() / len(y_prob)
        if bin_weight > 0:
            ece += bin_weight * abs(frac_pos[i] - mean_pred[i])

    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"(ECE < 0.05 = well calibrated, > 0.10 = poorly calibrated)")

    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "s-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title)
        ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception:
        pass  # Skip plotting in headless environments

    return {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
        "ece": ece,
        "well_calibrated": ece < 0.05,
    }
```

---

## Population Stability Index (PSI) — Drift Detection

PSI measures how much a feature's distribution has shifted between training and production. A large shift means the model was trained on a different "world" than it's now predicting on.

```python
def compute_psi(
    expected: np.ndarray,     # Training distribution
    actual: np.ndarray,       # Current production distribution
    n_bins: int = 10,
    epsilon: float = 1e-6,    # Avoid log(0)
) -> float:
    """
    Population Stability Index.
    
    PSI < 0.10:  No significant shift — model stable
    PSI 0.10-0.20: Moderate shift — monitor closely
    PSI > 0.20:  Significant shift — retrain recommended
    PSI > 0.25:  Severe shift — retrain immediately
    """
    # Build bins from expected distribution
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    expected_pct = expected_counts / (len(expected) + epsilon)
    actual_pct = actual_counts / (len(actual) + epsilon)

    # Avoid division by zero or log(0)
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def monitor_feature_drift(
    train_features: pd.DataFrame,    # Features from last training period
    current_features: pd.DataFrame,  # Features from current production window
    alert_threshold: float = 0.20,
    retrain_threshold: float = 0.25,
) -> dict:
    """
    Compute PSI for all features. Return drift report.
    Run this weekly on live system.
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

    if alerts:
        print("\n".join(alerts))
    else:
        print(f"Drift monitor: all {len(report)} features STABLE")

    return {
        "feature_report": report,
        "alerts": alerts,
        "requires_retrain": any("RETRAIN" in r["status"] for r in report.values()),
        "n_features_drifted": sum(1 for r in report.values() if r["status"] != "STABLE"),
    }
```

---

## Prediction Performance Monitoring

Track these metrics weekly on live predictions:

```python
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

def compute_live_metrics(
    predictions: pd.DataFrame,    # columns: date, ml_prob, actual_label
    lookback_days: int = 90,
) -> dict:
    """
    Compute model performance metrics on recent live predictions.
    Used to trigger retrain or strategy disable.
    """
    cutoff = predictions["date"].max() - pd.Timedelta(days=lookback_days)
    recent = predictions[predictions["date"] >= cutoff].dropna()

    if len(recent) < 30:
        return {"status": "insufficient_data", "n_samples": len(recent)}

    y_true = recent["actual_label"].values
    y_prob = recent["ml_prob"].values
    y_pred = (y_prob >= 0.6).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    brier = brier_score_loss(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    win_rate = y_true.mean()

    metrics = {
        "n_samples": len(recent),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "f1": round(f1, 4),
        "win_rate": round(win_rate, 4),
        "status": "OK",
    }

    # Performance gates
    if auc < 0.50:
        metrics["status"] = "DISABLE_STRATEGY"
        metrics["alert"] = f"AUC {auc:.3f} below random. Disable Swing immediately."
    elif auc < 0.53:
        metrics["status"] = "RETRAIN_URGENT"
        metrics["alert"] = f"AUC {auc:.3f} degraded. Retrain model."

    return metrics
```

---

## Retrain Trigger Logic

```python
def should_retrain(
    drift_report: dict,
    performance_metrics: dict,
    days_since_last_train: int,
    max_days_between_trains: int = 90,
) -> tuple:
    """
    Returns (should_retrain: bool, reason: str).
    """
    reasons = []

    # Time-based retrain
    if days_since_last_train >= max_days_between_trains:
        reasons.append(f"Scheduled: {days_since_last_train} days since last train")

    # Drift-based retrain
    if drift_report.get("requires_retrain"):
        reasons.append(f"Feature drift: {drift_report['n_features_drifted']} features drifted")

    # Performance-based retrain
    status = performance_metrics.get("status", "OK")
    if status in ("RETRAIN_URGENT", "DISABLE_STRATEGY"):
        reasons.append(f"Performance degradation: AUC={performance_metrics.get('roc_auc')}")

    return bool(reasons), "; ".join(reasons) if reasons else "No retrain needed"
```

---

## Model Persistence with Metadata

Always save models with a JSON sidecar containing training metadata:

```python
import joblib
import json
from datetime import datetime

def save_model(model, metadata: dict, path_prefix: str):
    """
    Save model + metadata sidecar.
    Example: save_model(model, meta, "models/trade_filter_20240101")
    Creates:
      models/trade_filter_20240101.pkl
      models/trade_filter_20240101.json
    """
    model_path = f"{path_prefix}.pkl"
    meta_path = f"{path_prefix}.json"

    joblib.dump(model, model_path)

    # Ensure all metadata is JSON-serializable
    safe_meta = {
        "saved_at": datetime.utcnow().isoformat(),
        "train_start": str(metadata.get("train_start")),
        "train_end": str(metadata.get("train_end")),
        "feature_list": metadata.get("feature_list", []),
        "oos_roc_auc": metadata.get("oos_roc_auc"),
        "oos_f1": metadata.get("oos_f1"),
        "ece": metadata.get("ece"),
        "n_train_samples": metadata.get("n_train_samples"),
        "model_type": type(model).__name__,
    }
    with open(meta_path, "w") as f:
        json.dump(safe_meta, f, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Metadata: {meta_path}")


def load_model_with_meta(path_prefix: str):
    """Load model and validate it's not expired."""
    model = joblib.load(f"{path_prefix}.pkl")
    with open(f"{path_prefix}.json") as f:
        meta = json.load(f)

    # Check age
    saved_at = pd.Timestamp(meta["saved_at"])
    age_days = (pd.Timestamp.utcnow() - saved_at).days
    if age_days > 90:
        print(f"WARNING: Model is {age_days} days old. Consider retraining.")

    return model, meta
```

---

## Weekly Monitoring Checklist

Run every Monday before the trading week:

```python
def weekly_model_health_check(
    model,
    train_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    recent_predictions: pd.DataFrame,
    model_meta: dict,
) -> dict:
    """Run full weekly health check."""
    from datetime import date

    drift = monitor_feature_drift(train_features, recent_features)
    perf = compute_live_metrics(recent_predictions)
    days_since = (pd.Timestamp.today() - pd.Timestamp(model_meta["train_end"])).days
    retrain, reason = should_retrain(drift, perf, days_since)

    report = {
        "check_date": str(date.today()),
        "drift_report": drift,
        "performance": perf,
        "days_since_train": days_since,
        "retrain_recommended": retrain,
        "retrain_reason": reason,
    }

    print(f"\n{'='*50}")
    print(f"Weekly Model Health Check — {date.today()}")
    print(f"{'='*50}")
    print(f"Model age:        {days_since} days")
    print(f"Live AUC (90d):   {perf.get('roc_auc', 'N/A')}")
    print(f"Features drifted: {drift.get('n_features_drifted', 0)}")
    print(f"Retrain needed:   {retrain}")
    if retrain:
        print(f"Reason:           {reason}")

    return report
```

---

## Configuration

```yaml
ml:
  calibration:
    method: "isotonic"           # "isotonic" or "sigmoid"
    min_cal_samples: 300         # Use sigmoid if fewer samples

  drift_monitoring:
    run_every_days: 7            # Weekly
    alert_threshold: 0.20        # PSI alert level
    retrain_threshold: 0.25      # PSI retrain level
    lookback_days: 30            # Production window for PSI

  performance_monitoring:
    lookback_days: 90
    min_auc_to_trade: 0.53       # Disable swing if below
    min_samples_for_eval: 30

  retraining:
    max_days_between_trains: 90
    min_oos_auc_to_deploy: 0.55
```

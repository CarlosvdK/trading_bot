"""
ML model calibration (isotonic/Platt scaling).
Skill reference: .claude/skills/model-calibration-drift/SKILL.md
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def calibrate_model(
    base_model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "isotonic",
):
    """
    Post-hoc calibration of a fitted classifier.

    Use isotonic regression unless < 300 calibration samples,
    in which case use sigmoid (Platt scaling).
    """
    n_cal = len(X_cal)
    if n_cal < 300:
        method = "sigmoid"

    calibrated = CalibratedClassifierCV(base_model, cv="prefit", method=method)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: str = None,
) -> dict:
    """
    Compute reliability/calibration metrics.
    Returns dict with ECE and per-bin statistics.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(len(frac_pos)):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_weight = bin_mask.sum() / len(y_prob)
        if bin_weight > 0:
            ece += bin_weight * abs(frac_pos[i] - mean_pred[i])

    if save_path:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.plot(mean_pred, frac_pos, "s-", label="Model")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Reliability Diagram")
            ax.legend()
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    return {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
        "ece": float(ece),
        "well_calibrated": ece < 0.05,
    }

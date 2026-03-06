"""
Model persistence with metadata sidecar.
Skill reference: .claude/skills/model-calibration-drift/SKILL.md
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def save_model(model, metadata: dict, path_prefix: str) -> None:
    """
    Save model + metadata sidecar JSON.
    Creates: {path_prefix}.pkl and {path_prefix}.json
    """
    model_path = f"{path_prefix}.pkl"
    meta_path = f"{path_prefix}.json"

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    safe_meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "train_start": str(metadata.get("train_start", "")),
        "train_end": str(metadata.get("train_end", "")),
        "feature_list": metadata.get("feature_list", []),
        "oos_roc_auc": metadata.get("oos_roc_auc"),
        "oos_f1": metadata.get("oos_f1"),
        "ece": metadata.get("ece"),
        "n_train_samples": metadata.get("n_train_samples"),
        "model_type": type(model).__name__,
    }
    with open(meta_path, "w") as f:
        json.dump(safe_meta, f, indent=2)

    logger.info(f"Model saved: {model_path}")


def load_model_with_meta(
    path_prefix: str,
    warn_age_days: int = 90,
) -> tuple:
    """
    Load model and metadata. Warns if model is older than warn_age_days.
    Returns (model, metadata_dict).
    """
    model = joblib.load(f"{path_prefix}.pkl")
    with open(f"{path_prefix}.json") as f:
        meta = json.load(f)

    saved_at = pd.Timestamp(meta["saved_at"])
    age_days = (pd.Timestamp.now(tz="UTC") - saved_at).days
    if age_days > warn_age_days:
        logger.warning(f"Model is {age_days} days old. Consider retraining.")
    meta["age_days"] = age_days

    return model, meta

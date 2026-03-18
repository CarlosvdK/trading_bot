"""
Tests for model persistence — save/load with metadata sidecar.
Skill reference: .claude/skills/model-calibration-drift/SKILL.md
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.models.persistence import save_model, load_model_with_meta


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def tmp_prefix(tmp_path):
    return str(tmp_path / "test_model")


class TestSaveModel:
    def test_creates_pkl_and_json(self, trained_model, tmp_prefix):
        meta = {"train_start": "2024-01-01", "train_end": "2024-06-01"}
        save_model(trained_model, meta, tmp_prefix)
        assert Path(f"{tmp_prefix}.pkl").exists()
        assert Path(f"{tmp_prefix}.json").exists()

    def test_metadata_contains_required_fields(self, trained_model, tmp_prefix):
        meta = {
            "train_start": "2024-01-01",
            "train_end": "2024-06-01",
            "feature_list": ["a", "b", "c"],
            "n_train_samples": 200,
        }
        save_model(trained_model, meta, tmp_prefix)
        with open(f"{tmp_prefix}.json") as f:
            saved_meta = json.load(f)
        assert "saved_at" in saved_meta
        assert saved_meta["train_start"] == "2024-01-01"
        assert saved_meta["model_type"] == "GradientBoostingClassifier"

    def test_creates_parent_dirs(self, trained_model, tmp_path):
        prefix = str(tmp_path / "subdir" / "nested" / "model")
        save_model(trained_model, {}, prefix)
        assert Path(f"{prefix}.pkl").exists()


class TestLoadModel:
    def test_roundtrip(self, trained_model, tmp_prefix):
        meta = {
            "train_start": "2024-01-01",
            "train_end": "2024-06-01",
            "oos_roc_auc": 0.65,
        }
        save_model(trained_model, meta, tmp_prefix)
        loaded, loaded_meta = load_model_with_meta(tmp_prefix)

        # Model should produce same predictions
        X_test = np.random.randn(10, 3)
        np.testing.assert_array_equal(
            trained_model.predict(X_test),
            loaded.predict(X_test),
        )
        assert loaded_meta["oos_roc_auc"] == 0.65

    def test_age_days_computed(self, trained_model, tmp_prefix):
        save_model(trained_model, {}, tmp_prefix)
        _, meta = load_model_with_meta(tmp_prefix)
        assert "age_days" in meta
        assert meta["age_days"] >= 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_with_meta(str(tmp_path / "nonexistent"))

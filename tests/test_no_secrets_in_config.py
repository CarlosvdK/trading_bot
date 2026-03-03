"""
Tests for secrets management and config security.
Skill reference: .claude/skills/secrets-security/SKILL.md
"""

import os
import glob
import tempfile
import pytest

import yaml

from src.utils.config_loader import (
    load_config,
    _validate_no_secrets_in_config,
    _validate_risk_params,
)
from src.utils.secrets import get_secret


class TestNoSecretsInConfig:
    def test_no_secrets_in_yaml_configs(self):
        """Fail if any YAML config file contains secret-looking values."""
        secret_patterns = [
            "api_key",
            "api_secret",
            "password",
            "token",
            "webhook_secret",
        ]
        config_files = glob.glob("config/*.yaml") + glob.glob("config/*.yml")
        violations = []

        for filepath in config_files:
            with open(filepath) as f:
                content = f.read().lower()
            for pattern in secret_patterns:
                if pattern in content:
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern in line and ":" in line:
                            parts = line.split(":", 1)
                            if len(parts) > 1 and parts[1].strip() not in (
                                "",
                                "null",
                                "~",
                                "none",
                            ):
                                violations.append(
                                    f"{filepath}:{i} — '{pattern}' has a value"
                                )

        assert not violations, (
            f"Potential secrets found in config files:\n"
            + "\n".join(violations)
            + "\nMove these to environment variables."
        )

    def test_env_file_not_committed(self):
        """Fail if .env file exists AND is tracked by git."""
        if not os.path.exists(".env"):
            return

        try:
            import subprocess

            result = subprocess.run(
                ["git", "ls-files", ".env"], capture_output=True, text=True
            )
            if ".env" in result.stdout:
                pytest.fail(
                    ".env file is tracked by git! Remove it: git rm --cached .env"
                )
        except FileNotFoundError:
            pass


class TestConfigValidation:
    def test_valid_config_loads(self):
        config = load_config("config/example.yaml")
        assert isinstance(config, dict)
        assert "system" in config
        assert "risk" in config

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("config/nonexistent.yaml")

    def test_unsafe_drawdown_limit_rejected(self):
        config = {"risk": {"max_portfolio_drawdown": 0.50}}  # 50% — too high
        with pytest.raises(ValueError, match="unsafe parameter"):
            _validate_risk_params(config)

    def test_unsafe_daily_loss_rejected(self):
        config = {"risk": {"max_daily_loss_pct": 0.50}}
        with pytest.raises(ValueError, match="unsafe parameter"):
            _validate_risk_params(config)

    def test_safe_params_pass(self):
        config = {
            "risk": {
                "max_portfolio_drawdown": 0.15,
                "max_daily_loss_pct": 0.03,
            }
        }
        _validate_risk_params(config)  # Should not raise

    def test_secret_in_config_detected(self):
        config = {"broker_api_key": "sk-12345678901234567890"}
        with pytest.raises(ValueError, match="SECURITY ERROR"):
            _validate_no_secrets_in_config(config)

    def test_nested_secret_detected(self):
        config = {"broker": {"api_secret": "super_secret_value_here"}}
        with pytest.raises(ValueError, match="SECURITY ERROR"):
            _validate_no_secrets_in_config(config)

    def test_empty_secret_key_allowed(self):
        config = {"api_key": ""}
        _validate_no_secrets_in_config(config)  # Empty string is OK

    def test_null_secret_key_allowed(self):
        config = {"api_key": "null"}
        _validate_no_secrets_in_config(config)  # "null" is OK

    def test_short_secret_value_allowed(self):
        config = {"api_key": "abc"}
        _validate_no_secrets_in_config(config)  # Too short to flag

    def test_config_with_safe_risk_params(self):
        """Test that example.yaml passes all validation."""
        config = load_config("config/example.yaml")
        assert config["risk"]["max_portfolio_drawdown"] == 0.15
        assert config["portfolio"]["initial_nav"] == 100000

    def test_empty_config_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as f:
            f.write("")
            f.flush()
            with pytest.raises(ValueError, match="empty"):
                load_config(f.name)
        os.unlink(f.name)

    def test_labeling_k1_too_high(self):
        config = {"labeling": {"k1": 15.0}}
        with pytest.raises(ValueError, match="unsafe"):
            _validate_risk_params(config)

    def test_labeling_k1_in_range(self):
        config = {"labeling": {"k1": 2.0, "k2": 1.0, "horizon_days": 10}}
        _validate_risk_params(config)  # Should not raise


class TestGetSecret:
    def test_get_secret_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET", "my_value")
        assert get_secret("TEST_SECRET") == "my_value"

    def test_get_required_secret_missing_raises(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        with pytest.raises(ValueError, match="not found"):
            get_secret("NONEXISTENT_KEY", required=True)

    def test_get_optional_secret_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        result = get_secret("NONEXISTENT_KEY", required=False)
        assert result is None

"""
Config loading and validation.
Skill reference: .claude/skills/secrets-security/SKILL.md
"""

from pathlib import Path
from typing import Any

import yaml


# Valid ranges for risk parameters — anything outside these is dangerous
RISK_PARAM_RANGES = {
    "risk.max_portfolio_drawdown": (0.01, 0.30),
    "risk.max_daily_loss_pct": (0.005, 0.10),
    "risk.swing.max_weekly_loss": (0.01, 0.20),
    "risk.swing.max_concurrent_positions": (1, 50),
    "risk.swing.max_position_pct": (0.01, 0.50),
    "risk.swing.max_sector_pct": (0.05, 1.0),
    "risk.exposure.max_gross_exposure_pct": (0.1, 2.0),
    "portfolio.initial_nav": (1_000, 100_000_000),
    "labeling.k1": (0.5, 10.0),
    "labeling.k2": (0.1, 5.0),
    "labeling.horizon_days": (1, 60),
}

# Known secret patterns — fail loudly if found in config
SECRET_PATTERNS = [
    "api_key",
    "api_secret",
    "password",
    "token",
    "secret",
    "private_key",
    "access_key",
    "auth_key",
    "credential",
]


def load_config(path: str) -> dict:
    """
    Load YAML config and validate all parameters.
    Exits with error message if any parameter is out of safe range.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    _validate_no_secrets_in_config(config)
    _validate_risk_params(config)

    return config


def _get_nested(d: dict, key_path: str) -> Any:
    """Get nested dict value by dot-separated key path."""
    keys = key_path.split(".")
    val = d
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return None
        val = val[k]
    return val


def _validate_risk_params(config: dict):
    """Validate all risk parameters are within safe ranges."""
    errors = []

    for param_path, (min_val, max_val) in RISK_PARAM_RANGES.items():
        val = _get_nested(config, param_path)
        if val is None:
            continue
        if not (min_val <= val <= max_val):
            errors.append(
                f"  {param_path}: value={val}, allowed=[{min_val}, {max_val}]"
            )

    if errors:
        raise ValueError(
            "Config validation FAILED — unsafe parameter values:\n"
            + "\n".join(errors)
            + "\n\nFix these values before running the system."
        )


def _validate_no_secrets_in_config(config: dict, path: str = ""):
    """
    Recursively scan config for values that look like secrets.
    Raises ValueError if any secret-looking key has a non-empty value.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            full_key = f"{path}.{k}" if path else k
            if any(pattern in k.lower() for pattern in SECRET_PATTERNS):
                if (
                    isinstance(v, str)
                    and len(v) > 5
                    and v not in ("", "null", "None")
                ):
                    raise ValueError(
                        f"SECURITY ERROR: Config key '{full_key}' appears "
                        f"to contain a secret value.\n"
                        f"Move this to an environment variable: "
                        f"export {k.upper()}='your_value'\n"
                        f"Then reference it in code with: "
                        f"get_secret('{k.upper()}')"
                    )
            _validate_no_secrets_in_config(v, full_key)

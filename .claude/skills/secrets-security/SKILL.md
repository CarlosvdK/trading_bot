---
name: secrets-security
description: Secrets management — env var loading, config validation, .gitignore, HMAC webhooks, audit logging. Use whenever handling API keys, credentials, config validation, or security hardening.
triggers:
  - API key
  - secret
  - credential
  - .env
  - config validation
  - HMAC
  - webhook signing
  - audit log
  - security
  - gitignore
priority: P0
---

# Skill: Secrets Management & Security

## What This Skill Is
How to handle API keys, broker credentials, and webhook secrets safely. A single leaked API key can expose your brokerage account to unauthorized trades. This skill covers: environment variable loading, config validation, .gitignore setup, and audit logging.

---

## The Golden Rule
> **Never write credentials, API keys, or secrets to any file that could be committed to version control.**

This means:
- No keys in `config/example.yaml`
- No keys in `config/live.yaml`  
- No keys in any `.py` source file
- No keys in Jupyter notebooks
- No keys in README examples

---

## Environment Variable Loading

```python
# src/utils/secrets.py
import os
from typing import Optional
from pathlib import Path

# Load .env file if present (development only)
try:
    from dotenv import load_dotenv
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from .env")
except ImportError:
    pass  # python-dotenv not required in production


def get_secret(key: str, required: bool = True) -> Optional[str]:
    """
    Load a secret from environment variables.
    Never reads from config files.
    
    Args:
        key: Environment variable name (e.g., "BROKER_API_KEY")
        required: If True and key not found, raises ValueError
    
    Returns:
        Secret value, or None if not required and not found
    """
    value = os.environ.get(key)
    if value is None and required:
        raise ValueError(
            f"Required secret '{key}' not found in environment variables.\n"
            f"Set it with: export {key}='your_value'\n"
            f"Or add to .env file (never commit .env to git)."
        )
    return value


# Convenience accessors for known secrets
def get_broker_api_key() -> str:
    return get_secret("BROKER_API_KEY")

def get_broker_api_secret() -> str:
    return get_secret("BROKER_API_SECRET")

def get_webhook_secret() -> Optional[str]:
    return get_secret("WEBHOOK_SECRET", required=False)

def get_db_password() -> Optional[str]:
    return get_secret("DB_PASSWORD", required=False)
```

---

## .env File Template (Development Only)

```bash
# .env — NEVER COMMIT THIS FILE
# Add .env to .gitignore immediately

BROKER_API_KEY=your_api_key_here
BROKER_API_SECRET=your_api_secret_here
WEBHOOK_SECRET=your_webhook_secret_here
DB_PASSWORD=optional_db_password
```

---

## .gitignore Setup

```gitignore
# .gitignore — add these immediately before first commit

# Secrets
.env
*.key
*.pem
secrets/
credentials/

# Data (may contain sensitive position info)
data/ohlcv/
*.sqlite
*.db

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Jupyter (may contain key outputs)
*.ipynb_checkpoints/

# Logs (may contain position/order details)
logs/
*.log
```

---

## Config Validator

Every config parameter must be validated on startup. A misconfigured risk limit (e.g., `max_portfolio_drawdown: 1.0`) could silently disable a safety control.

```python
# src/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Any, Dict

# Valid ranges for risk parameters — anything outside these is dangerous
RISK_PARAM_RANGES = {
    "risk.max_portfolio_drawdown":      (0.01, 0.30),   # 1%-30%
    "risk.max_daily_loss_pct":          (0.005, 0.10),  # 0.5%-10%
    "risk.swing.max_weekly_loss":       (0.01, 0.20),   # 1%-20%
    "risk.swing.max_concurrent_positions": (1, 50),
    "risk.swing.max_position_pct":      (0.01, 0.50),   # 1%-50%
    "risk.swing.max_sector_pct":        (0.05, 1.0),
    "risk.exposure.max_gross_exposure_pct": (0.1, 2.0),
    "backtest.initial_nav":             (1_000, 100_000_000),
    "labeling.k1":                      (0.5, 10.0),
    "labeling.k2":                      (0.1, 5.0),
    "labeling.horizon_days":            (1, 60),
}


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
            continue  # Optional params are skipped
        if not (min_val <= val <= max_val):
            errors.append(
                f"  {param_path}: value={val}, allowed=[{min_val}, {max_val}]"
            )

    if errors:
        raise ValueError(
            "Config validation FAILED — unsafe parameter values:\n" +
            "\n".join(errors) +
            "\n\nFix these values before running the system."
        )

    print("Config validation PASSED.")


# Known secret patterns — fail loudly if found in config
SECRET_PATTERNS = [
    "api_key", "api_secret", "password", "token", "secret",
    "private_key", "access_key", "auth_key", "credential",
]

def _validate_no_secrets_in_config(config: dict, path: str = ""):
    """
    Recursively scan config for values that look like secrets.
    Raises ValueError if any secret-looking key has a non-empty value.
    """
    if isinstance(config, dict):
        for k, v in config.items():
            full_key = f"{path}.{k}" if path else k
            # Check if key name suggests a secret
            if any(pattern in k.lower() for pattern in SECRET_PATTERNS):
                if isinstance(v, str) and len(v) > 5 and v not in ("", "null", "None"):
                    raise ValueError(
                        f"SECURITY ERROR: Config key '{full_key}' appears to contain a secret value.\n"
                        f"Move this to an environment variable: export {k.upper()}='your_value'\n"
                        f"Then reference it in code with: get_secret('{k.upper()}')"
                    )
            _validate_no_secrets_in_config(v, full_key)
```

---

## Test: No Secrets in Config

```python
# tests/test_no_secrets_in_config.py
import os
import glob
import pytest
import yaml

SECRET_PATTERNS = ["api_key", "api_secret", "password", "token", "webhook_secret"]

def test_no_secrets_in_yaml_configs():
    """Fail if any YAML config file contains secret-looking values."""
    config_files = glob.glob("config/*.yaml") + glob.glob("config/*.yml")
    violations = []

    for filepath in config_files:
        with open(filepath) as f:
            content = f.read().lower()
        for pattern in SECRET_PATTERNS:
            if pattern in content:
                # Find the actual line
                for i, line in enumerate(content.splitlines(), 1):
                    if pattern in line and ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) > 1 and parts[1].strip() not in ("", "null", "~", "none"):
                            violations.append(f"{filepath}:{i} — '{pattern}' has a value")

    assert not violations, (
        f"Potential secrets found in config files:\n" +
        "\n".join(violations) +
        "\n\nMove these to environment variables."
    )


def test_env_file_not_committed():
    """Fail if .env file exists AND is tracked by git."""
    if not os.path.exists(".env"):
        return  # No .env file, fine

    try:
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", ".env"],
            capture_output=True, text=True
        )
        if ".env" in result.stdout:
            pytest.fail(".env file is tracked by git! Remove it: git rm --cached .env")
    except FileNotFoundError:
        pass  # Git not available in this environment
```

---

## HMAC Webhook Signing

If your system sends alerts to a webhook (Slack, Discord, custom), sign the payload to prevent spoofing:

```python
import hmac
import hashlib
import json
import time

def sign_webhook_payload(payload: dict, secret: str) -> dict:
    """
    Add HMAC signature to webhook payload.
    Receiver must verify the signature before trusting the payload.
    """
    body = json.dumps(payload, sort_keys=True)
    timestamp = str(int(time.time()))
    message = f"{timestamp}.{body}"

    sig = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return {
        "payload": payload,
        "timestamp": timestamp,
        "signature": f"sha256={sig}",
    }


def verify_webhook_signature(signed: dict, secret: str, max_age_seconds: int = 300) -> bool:
    """Verify HMAC signature on received webhook."""
    timestamp = signed.get("timestamp", "0")
    if abs(time.time() - int(timestamp)) > max_age_seconds:
        return False  # Replay attack protection

    body = json.dumps(signed["payload"], sort_keys=True)
    message = f"{timestamp}.{body}"
    expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    received = signed.get("signature", "").replace("sha256=", "")

    return hmac.compare_digest(expected, received)
```

---

## Audit Log with Hash Chaining

```python
import hashlib
import json
from datetime import datetime

class AuditLogger:
    """
    Append-only audit log with hash chaining.
    Each entry includes a hash of the previous entry, making tampering detectable.
    """

    def __init__(self, path: str):
        self.path = path
        self._last_hash = self._compute_file_hash()

    def log(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
            "prev_hash": self._last_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["entry_hash"] = entry_hash

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._last_hash = entry_hash

    def _compute_file_hash(self) -> str:
        try:
            with open(self.path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return "genesis"
```

---

## Quick-Reference Security Checklist

Before any deployment:

- [ ] `.env` in `.gitignore` — run `git check-ignore .env`
- [ ] `test_no_secrets_in_config.py` passes
- [ ] SQLite file permissions: `chmod 600 portfolio.db`
- [ ] Webhook URLs treated as secrets (not in README examples)
- [ ] `get_secret()` used everywhere credentials are needed
- [ ] `BROKER_API_KEY` scoped to minimum permissions (read + trade, NOT withdraw)
- [ ] Kill switch requires manual restart (`manual_restart_required: true`)
- [ ] Log retention >= 7 years for compliance

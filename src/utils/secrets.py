"""
Secrets management — environment variable loading only.
Skill reference: .claude/skills/secrets-security/SKILL.md
"""

import os
from pathlib import Path
from typing import Optional

# Load .env file if present (development only)
try:
    from dotenv import load_dotenv

    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not required in production


def get_secret(key: str, required: bool = True) -> Optional[str]:
    """
    Load a secret from environment variables.
    Never reads from config files.
    """
    value = os.environ.get(key)
    if value is None and required:
        raise ValueError(
            f"Required secret '{key}' not found in environment variables.\n"
            f"Set it with: export {key}='your_value'\n"
            f"Or add to .env file (never commit .env to git)."
        )
    return value


def get_broker_api_key() -> str:
    return get_secret("BROKER_API_KEY")


def get_broker_api_secret() -> str:
    return get_secret("BROKER_API_SECRET")


def get_webhook_secret() -> Optional[str]:
    return get_secret("WEBHOOK_SECRET", required=False)


def get_db_password() -> Optional[str]:
    return get_secret("DB_PASSWORD", required=False)

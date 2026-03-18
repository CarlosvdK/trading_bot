"""Supabase client singleton — loaded from .env."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


def get_supabase_client():
    """Get or create the Supabase client singleton."""
    global _client
    if _client is not None:
        return _client

    from dotenv import load_dotenv
    load_dotenv()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        logger.warning(
            "SUPABASE_URL or SUPABASE_SERVICE_KEY not set. "
            "Storage will be unavailable."
        )
        return None

    from supabase import create_client
    _client = create_client(url, key)
    logger.info(f"Supabase connected: {url}")
    return _client

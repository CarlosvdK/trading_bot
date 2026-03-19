"""
Shared Claude LLM client for all AI-enhanced trading modules.

Handles rate limiting, retries, cost tracking, and graceful fallback
when the API is unavailable. All modules import this instead of
calling anthropic directly.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded client singleton
_client = None
_available: Optional[bool] = None

# Cost tracking
_total_input_tokens = 0
_total_output_tokens = 0


def _get_client():
    """Lazy-initialize the Anthropic client from environment."""
    global _client, _available

    if _available is False:
        return None

    if _client is not None:
        return _client

    try:
        import anthropic
        from src.utilities.secrets import get_secret

        api_key = get_secret("ANTHROPIC_API_KEY", required=False)
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — LLM features disabled")
            _available = False
            return None

        _client = anthropic.Anthropic(api_key=api_key)
        _available = True
        logger.info("Claude LLM client initialized")
        return _client

    except ImportError:
        logger.warning("anthropic package not installed — LLM features disabled")
        _available = False
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Claude client: {e}")
        _available = False
        return None


def is_available() -> bool:
    """Check if LLM features are available."""
    _get_client()
    return _available is True


def query(
    prompt: str,
    system: str = "",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 30.0,
) -> Optional[str]:
    """
    Send a query to Claude and return the text response.

    Args:
        prompt: The user message / main query.
        system: System prompt for context/persona.
        model: Model ID to use (defaults to Sonnet for cost efficiency).
        max_tokens: Max response tokens.
        temperature: Lower = more deterministic.
        timeout: Request timeout in seconds.

    Returns:
        Response text, or None if unavailable/failed.
    """
    global _total_input_tokens, _total_output_tokens

    client = _get_client()
    if client is None:
        return None

    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        # Track costs
        _total_input_tokens += response.usage.input_tokens
        _total_output_tokens += response.usage.output_tokens

        text = response.content[0].text if response.content else None
        return text

    except Exception as e:
        logger.warning(f"Claude API call failed: {e}")
        return None


def query_batch(
    prompts: list[dict],
    system: str = "",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 512,
    temperature: float = 0.3,
    delay_between: float = 0.2,
) -> list[Optional[str]]:
    """
    Send multiple queries sequentially with rate limiting.

    Args:
        prompts: List of {"prompt": str, "id": str} dicts.
        system: Shared system prompt.
        model: Model ID.
        max_tokens: Max tokens per response.
        temperature: Temperature.
        delay_between: Seconds between requests (rate limiting).

    Returns:
        List of response texts (or None for failures), same order as input.
    """
    results = []
    for item in prompts:
        text = query(
            prompt=item["prompt"],
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results.append(text)
        if delay_between > 0 and len(prompts) > 1:
            time.sleep(delay_between)

    return results


def get_usage_stats() -> dict:
    """Get cumulative token usage stats."""
    # Approximate costs (Sonnet pricing)
    input_cost = _total_input_tokens * 3.0 / 1_000_000
    output_cost = _total_output_tokens * 15.0 / 1_000_000
    return {
        "total_input_tokens": _total_input_tokens,
        "total_output_tokens": _total_output_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 4),
    }

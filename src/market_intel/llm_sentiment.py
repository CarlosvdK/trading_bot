"""
LLM-Enhanced Sentiment Analysis — uses Claude to deeply analyze financial news.

Replaces simple VADER + keyword matching with nuanced LLM understanding of:
- Earnings context (beat by how much? guidance direction?)
- Regulatory impact (FDA phase? severity of investigation?)
- Supply chain ripple effects (who benefits, who loses?)
- Market positioning (is this priced in already?)
- Sector-wide implications vs company-specific

Falls back to VADER if LLM is unavailable.
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SENTIMENT_SYSTEM_PROMPT = """You are a senior equity research analyst at a top hedge fund.
Your job is to analyze financial news headlines and provide precise trading-relevant sentiment scores.

Rules:
- Score from -1.0 (extremely bearish) to +1.0 (extremely bullish)
- Consider whether the news is already priced in
- Differentiate between company-specific and sector-wide impact
- Consider second-order effects (supply chain, competitors)
- Be skeptical of vague or promotional headlines
- Weight earnings/guidance/FDA events much higher than analyst opinions
- Return ONLY valid JSON, no commentary"""

SENTIMENT_PROMPT_TEMPLATE = """Analyze these financial news headlines for {symbol} and return a JSON object.

Headlines:
{headlines}

Return this exact JSON structure:
{{
  "sentiment_score": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0, how confident you are in the score>,
  "magnitude": <float 0.0 to 1.0, how impactful this news is>,
  "event_type": "<earnings|fda|merger|macro|analyst|legal|sector|general>",
  "priced_in_estimate": <float 0.0 to 1.0, how much is likely priced in>,
  "time_horizon": "<intraday|days|weeks|months>",
  "affected_symbols": ["<list of other tickers likely affected>"],
  "bull_case": "<one sentence bull interpretation>",
  "bear_case": "<one sentence bear interpretation>",
  "key_insight": "<the most important non-obvious takeaway>"
}}"""


def analyze_with_llm(
    symbol: str,
    headlines: List[str],
    max_headlines: int = 8,
) -> Optional[dict]:
    """
    Use Claude to deeply analyze news headlines for a symbol.

    Args:
        symbol: Stock ticker.
        headlines: List of news headline strings.
        max_headlines: Max headlines to send (cost control).

    Returns:
        Dict with LLM analysis, or None if unavailable.
    """
    from src.market_intel.llm_client import query, is_available

    if not is_available() or not headlines:
        return None

    # Limit headlines for cost
    trimmed = headlines[:max_headlines]
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(trimmed))

    prompt = SENTIMENT_PROMPT_TEMPLATE.format(
        symbol=symbol,
        headlines=numbered,
    )

    response = query(
        prompt=prompt,
        system=SENTIMENT_SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.2,
    )

    if not response:
        return None

    try:
        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)

        # Validate required fields
        required = ["sentiment_score", "confidence", "magnitude"]
        if not all(k in result for k in required):
            logger.warning(f"LLM sentiment missing fields for {symbol}")
            return None

        # Clamp values
        result["sentiment_score"] = max(-1.0, min(1.0, float(result["sentiment_score"])))
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        result["magnitude"] = max(0.0, min(1.0, float(result["magnitude"])))
        result["symbol"] = symbol
        result["source"] = "llm"

        return result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse LLM sentiment for {symbol}: {e}")
        return None


def enhance_sentiment_batch(
    symbol_news: Dict[str, List[str]],
    existing_sentiment: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Enhance existing VADER sentiment with LLM analysis for high-priority symbols.

    Only calls LLM for symbols where:
    - There are 2+ headlines (enough context for LLM)
    - VADER magnitude > 0.15 (worth investigating)
    - OR VADER is uncertain (magnitude < 0.05 with 3+ headlines)

    Args:
        symbol_news: {symbol: [headline_strings]}.
        existing_sentiment: {symbol: VADER sentiment dict}.

    Returns:
        Updated sentiment dict with LLM enrichment.
    """
    from src.market_intel.llm_client import is_available

    if not is_available():
        return existing_sentiment

    enhanced = dict(existing_sentiment)

    # Prioritize symbols for LLM analysis
    priority_symbols = []
    for sym, headlines in symbol_news.items():
        if len(headlines) < 2:
            continue

        vader = existing_sentiment.get(sym, {})
        vader_mag = abs(vader.get("score", 0))
        n_articles = vader.get("n_articles", len(headlines))

        # High-impact news worth deeper analysis
        if vader_mag > 0.15:
            priority_symbols.append((sym, vader_mag, headlines))
        # Uncertain signal with lots of coverage — LLM can disambiguate
        elif vader_mag < 0.05 and n_articles >= 3:
            priority_symbols.append((sym, 0.5, headlines))

    # Sort by priority (highest magnitude first), limit to 15 per cycle
    priority_symbols.sort(key=lambda x: x[1], reverse=True)
    priority_symbols = priority_symbols[:15]

    for sym, _, headlines in priority_symbols:
        llm_result = analyze_with_llm(sym, headlines)
        if llm_result is None:
            continue

        # Blend LLM with VADER (LLM gets 70% weight when available)
        vader_score = enhanced.get(sym, {}).get("score", 0)
        llm_score = llm_result["sentiment_score"]
        blended_score = vader_score * 0.3 + llm_score * 0.7

        # Update the sentiment dict
        if sym in enhanced:
            enhanced[sym]["score"] = round(blended_score, 4)
            enhanced[sym]["magnitude"] = round(
                max(enhanced[sym].get("magnitude", 0), llm_result["magnitude"]), 4
            )
            enhanced[sym]["llm_analysis"] = llm_result
        else:
            enhanced[sym] = {
                "score": round(blended_score, 4),
                "magnitude": round(llm_result["magnitude"], 4),
                "n_articles": len(headlines),
                "bullish_count": 1 if blended_score > 0.1 else 0,
                "bearish_count": 1 if blended_score < -0.1 else 0,
                "headlines": headlines[:5],
                "llm_analysis": llm_result,
            }

        # Cross-stock propagation from LLM insights
        affected = llm_result.get("affected_symbols", [])
        for affected_sym in affected:
            if affected_sym not in enhanced:
                # Propagate at reduced strength
                enhanced[affected_sym] = {
                    "score": round(blended_score * 0.3, 4),
                    "magnitude": round(llm_result["magnitude"] * 0.3, 4),
                    "n_articles": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "headlines": [],
                    "propagated_from": [{
                        "source": sym,
                        "relationship": "llm_identified",
                        "impact": round(blended_score * 0.3, 4),
                    }],
                }

    return enhanced

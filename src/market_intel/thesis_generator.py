"""
Thesis Generator — uses Claude to create structured bull/bear investment theses.

For each stock that reaches Stage B of the pipeline, generates:
- A concise bull case with catalysts and price targets
- A concise bear case with risks and downside scenarios
- Key metrics to watch for thesis confirmation/invalidation
- Recommended position sizing based on thesis conviction

This gives each agent richer context when voting, and provides
the human operator with readable justifications for every trade.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

THESIS_SYSTEM_PROMPT = """You are a senior portfolio manager at a multi-strategy hedge fund.
Generate concise, actionable investment theses. Be specific with numbers and catalysts.
No fluff — every sentence must contain actionable information.
Return ONLY valid JSON, no commentary."""

THESIS_PROMPT_TEMPLATE = """Generate a bull/bear thesis for {symbol} based on this context:

Price Action (last {lookback}d):
- Current price vs SMA: {price_vs_sma}
- RSI: {rsi}
- Volume trend: {vol_trend}
- 5d return: {ret_5d}
- 21d return: {ret_21d}

Agent Signal:
- Direction: {direction}
- Strategy: {strategy}
- Confidence: {confidence}
- Agent reasoning: {reasoning}

News Sentiment: {sentiment_summary}
Market Regime: {regime}
Sector: {sector}

Return this exact JSON:
{{
  "bull_thesis": {{
    "summary": "<2-3 sentence bull case>",
    "catalysts": ["<specific near-term catalyst 1>", "<catalyst 2>"],
    "target_upside_pct": <float, estimated upside %>,
    "timeframe": "<days|weeks|months>",
    "conviction": <float 0-1>
  }},
  "bear_thesis": {{
    "summary": "<2-3 sentence bear case>",
    "risks": ["<specific risk 1>", "<risk 2>"],
    "target_downside_pct": <float, estimated downside %>,
    "timeframe": "<days|weeks|months>",
    "conviction": <float 0-1>
  }},
  "key_levels": {{
    "support": "<price level or % description>",
    "resistance": "<price level or % description>"
  }},
  "watch_metrics": ["<metric to monitor for thesis validation>", "<metric 2>"],
  "net_conviction": <float -1 to 1, negative=bearish, positive=bullish>,
  "sizing_suggestion": "<undersize|normal|oversize based on conviction>"
}}"""


@dataclass
class InvestmentThesis:
    """Structured investment thesis for a trade candidate."""
    symbol: str
    bull_summary: str = ""
    bull_catalysts: List[str] = field(default_factory=list)
    bull_upside_pct: float = 0.0
    bear_summary: str = ""
    bear_risks: List[str] = field(default_factory=list)
    bear_downside_pct: float = 0.0
    net_conviction: float = 0.0
    sizing_suggestion: str = "normal"
    watch_metrics: List[str] = field(default_factory=list)
    support_level: str = ""
    resistance_level: str = ""
    source: str = "llm"


def generate_thesis(
    symbol: str,
    price_data: dict,
    agent_signal: dict,
    sentiment_summary: str = "No recent news",
    regime: str = "unknown",
    sector: str = "unknown",
) -> Optional[InvestmentThesis]:
    """
    Generate a bull/bear thesis for a trade candidate.

    Args:
        symbol: Stock ticker.
        price_data: Dict with price_vs_sma, rsi, vol_trend, ret_5d, ret_21d, lookback.
        agent_signal: Dict with direction, strategy, confidence, reasoning.
        sentiment_summary: Brief text summary of news sentiment.
        regime: Current market regime string.
        sector: Stock's sector.

    Returns:
        InvestmentThesis dataclass, or None if LLM unavailable.
    """
    from src.market_intel.llm_client import query, is_available

    if not is_available():
        return None

    prompt = THESIS_PROMPT_TEMPLATE.format(
        symbol=symbol,
        lookback=price_data.get("lookback", 21),
        price_vs_sma=f"{price_data.get('price_vs_sma', 0):.3f}",
        rsi=f"{price_data.get('rsi', 50):.0f}",
        vol_trend=f"{price_data.get('vol_trend', 0):.3f}",
        ret_5d=f"{price_data.get('ret_5d', 0):.3f}",
        ret_21d=f"{price_data.get('ret_21d', 0):.3f}",
        direction=agent_signal.get("direction", "long"),
        strategy=agent_signal.get("strategy", "unknown"),
        confidence=f"{agent_signal.get('confidence', 0.5):.2f}",
        reasoning=agent_signal.get("reasoning", ""),
        sentiment_summary=sentiment_summary,
        regime=regime,
        sector=sector,
    )

    response = query(
        prompt=prompt,
        system=THESIS_SYSTEM_PROMPT,
        max_tokens=768,
        temperature=0.3,
    )

    if not response:
        return None

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        bull = data.get("bull_thesis", {})
        bear = data.get("bear_thesis", {})

        return InvestmentThesis(
            symbol=symbol,
            bull_summary=bull.get("summary", ""),
            bull_catalysts=bull.get("catalysts", []),
            bull_upside_pct=float(bull.get("target_upside_pct", 0)),
            bear_summary=bear.get("summary", ""),
            bear_risks=bear.get("risks", []),
            bear_downside_pct=float(bear.get("target_downside_pct", 0)),
            net_conviction=max(-1.0, min(1.0, float(data.get("net_conviction", 0)))),
            sizing_suggestion=data.get("sizing_suggestion", "normal"),
            watch_metrics=data.get("watch_metrics", []),
            support_level=data.get("key_levels", {}).get("support", ""),
            resistance_level=data.get("key_levels", {}).get("resistance", ""),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse thesis for {symbol}: {e}")
        return None


def generate_theses_batch(
    candidates: List[dict],
    max_theses: int = 10,
) -> Dict[str, InvestmentThesis]:
    """
    Generate theses for a batch of trade candidates.

    Only generates for the top candidates by confidence (cost control).

    Args:
        candidates: List of dicts with symbol, price_data, agent_signal, etc.
        max_theses: Max number of theses to generate per cycle.

    Returns:
        {symbol: InvestmentThesis} dict.
    """
    from src.market_intel.llm_client import is_available

    if not is_available():
        return {}

    # Sort by confidence, take top N
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("agent_signal", {}).get("confidence", 0),
        reverse=True,
    )[:max_theses]

    results = {}
    for cand in sorted_candidates:
        thesis = generate_thesis(
            symbol=cand["symbol"],
            price_data=cand.get("price_data", {}),
            agent_signal=cand.get("agent_signal", {}),
            sentiment_summary=cand.get("sentiment_summary", "No recent news"),
            regime=cand.get("regime", "unknown"),
            sector=cand.get("sector", "unknown"),
        )
        if thesis:
            results[cand["symbol"]] = thesis

    return results

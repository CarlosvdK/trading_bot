"""
Agent Reasoning Auditor — uses Claude to review agent trade decisions.

After agents vote on a stock, this module:
1. Collects all agent reasonings (for and against)
2. Asks Claude to find logical inconsistencies, missing risks, or groupthink
3. Generates a quality score and flags for human review
4. Identifies when agents are "right for wrong reasons" (dangerous)

This acts as an independent check layer between Stage C (global vote)
and Stage D (portfolio/risk), catching errors that rule-based checks miss.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

AUDITOR_SYSTEM_PROMPT = """You are a risk-focused investment committee reviewer at a hedge fund.
Your job is to critically evaluate trade proposals and agent reasoning for logical flaws.

You are skeptical by default. Look for:
- Circular reasoning ("price went up, so it will go up more")
- Contradictory signals being ignored
- Groupthink (all agents using same logic)
- Missing risk factors
- Over-reliance on a single indicator
- Confirmation bias (ignoring bearish signals for a bullish trade)
- Regime-inappropriate trades

Be concise and specific. Return ONLY valid JSON."""

AUDIT_PROMPT_TEMPLATE = """Review this multi-agent trade decision:

Symbol: {symbol}
Direction: {direction}
Approval Rate: {approval_pct}%
Weighted Confidence: {confidence}

Supporting Agent Reasonings ({n_supporters} agents):
{supporting_reasons}

Dissenting/Absent Context:
- {n_dissent} eligible agents did NOT propose this trade
- Strategies that passed: {strategies}

Market Context:
- Regime: {regime}
- Sector: {sector}
- Recent sentiment: {sentiment}

Return this exact JSON:
{{
  "quality_score": <float 0-1, overall reasoning quality>,
  "groupthink_risk": <float 0-1, how much agents are echoing each other>,
  "logical_flaws": ["<specific flaw 1>", "<flaw 2 if any>"],
  "missing_risks": ["<risk not considered>", "<risk 2 if any>"],
  "contradictions": ["<contradiction between agents if any>"],
  "regime_appropriate": <bool, is this trade appropriate for current regime?>,
  "recommendation": "<approve|flag_for_review|reject>",
  "confidence_adjustment": <float -0.3 to +0.1, how much to adjust confidence>,
  "review_note": "<1-2 sentence summary for human reviewer>"
}}"""


@dataclass
class AuditResult:
    """Result of auditing a trade decision."""
    symbol: str
    quality_score: float = 0.5
    groupthink_risk: float = 0.0
    logical_flaws: List[str] = field(default_factory=list)
    missing_risks: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    regime_appropriate: bool = True
    recommendation: str = "approve"
    confidence_adjustment: float = 0.0
    review_note: str = ""


def audit_decision(
    symbol: str,
    direction: str,
    approval_pct: float,
    confidence: float,
    supporting_reasons: List[str],
    n_dissent: int,
    strategies: List[str],
    regime: str = "unknown",
    sector: str = "unknown",
    sentiment: str = "neutral",
) -> AuditResult:
    """
    Audit a multi-agent trade decision for reasoning quality.

    Args:
        symbol: Stock ticker.
        direction: "long" or "short".
        approval_pct: % of eligible agents that approved.
        confidence: Weighted confidence score.
        supporting_reasons: List of reasoning strings from supporting agents.
        n_dissent: Number of eligible agents that didn't propose this.
        strategies: List of strategy names used.
        regime: Current market regime.
        sector: Stock sector.
        sentiment: Recent sentiment summary.

    Returns:
        AuditResult with quality assessment.
    """
    from src.market_intel.llm_client import query, is_available

    # Fallback: basic heuristic audit if LLM unavailable
    if not is_available():
        return _heuristic_audit(
            symbol, direction, approval_pct, confidence,
            supporting_reasons, strategies, regime
        )

    # Format supporting reasons (limit to 10 for cost)
    reasons_text = "\n".join(
        f"  Agent {i+1}: {r}" for i, r in enumerate(supporting_reasons[:10])
    )

    prompt = AUDIT_PROMPT_TEMPLATE.format(
        symbol=symbol,
        direction=direction,
        approval_pct=f"{approval_pct * 100:.1f}",
        confidence=f"{confidence:.3f}",
        n_supporters=len(supporting_reasons),
        supporting_reasons=reasons_text,
        n_dissent=n_dissent,
        strategies=", ".join(strategies),
        regime=regime,
        sector=sector,
        sentiment=sentiment,
    )

    response = query(
        prompt=prompt,
        system=AUDITOR_SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.2,
    )

    if not response:
        return _heuristic_audit(
            symbol, direction, approval_pct, confidence,
            supporting_reasons, strategies, regime
        )

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        return AuditResult(
            symbol=symbol,
            quality_score=max(0.0, min(1.0, float(data.get("quality_score", 0.5)))),
            groupthink_risk=max(0.0, min(1.0, float(data.get("groupthink_risk", 0.0)))),
            logical_flaws=data.get("logical_flaws", []),
            missing_risks=data.get("missing_risks", []),
            contradictions=data.get("contradictions", []),
            regime_appropriate=bool(data.get("regime_appropriate", True)),
            recommendation=data.get("recommendation", "approve"),
            confidence_adjustment=max(-0.3, min(0.1, float(data.get("confidence_adjustment", 0)))),
            review_note=data.get("review_note", ""),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse audit for {symbol}: {e}")
        return _heuristic_audit(
            symbol, direction, approval_pct, confidence,
            supporting_reasons, strategies, regime
        )


def _heuristic_audit(
    symbol: str,
    direction: str,
    approval_pct: float,
    confidence: float,
    supporting_reasons: List[str],
    strategies: List[str],
    regime: str,
) -> AuditResult:
    """Fallback rule-based audit when LLM is unavailable."""
    flaws = []
    missing = []
    groupthink = 0.0

    # Check for strategy monoculture
    if len(set(strategies)) == 1 and len(supporting_reasons) > 3:
        groupthink = 0.7
        flaws.append(f"All supporters use {strategies[0]} strategy — groupthink risk")

    # Check reasoning diversity
    unique_keywords = set()
    for r in supporting_reasons:
        for word in r.lower().split():
            if len(word) > 4:
                unique_keywords.add(word)
    if len(unique_keywords) < len(supporting_reasons) * 2:
        groupthink = max(groupthink, 0.5)
        flaws.append("Low reasoning diversity among supporters")

    # Regime check
    regime_appropriate = True
    if "high_vol" in regime and direction == "long" and confidence < 0.7:
        regime_appropriate = False
        missing.append("High volatility regime — long trades need higher conviction")

    if "trending_down" in regime and direction == "long":
        missing.append("Counter-trend trade in downtrending regime")

    # Confidence vs approval mismatch
    if confidence > 0.8 and approval_pct < 0.3:
        flaws.append("High confidence but low approval — potential overfit to one agent")

    quality = 1.0 - (len(flaws) * 0.15 + groupthink * 0.2)
    quality = max(0.2, min(1.0, quality))

    adjustment = -0.05 * len(flaws) - 0.1 * groupthink
    recommendation = "approve" if quality > 0.6 else "flag_for_review"

    return AuditResult(
        symbol=symbol,
        quality_score=round(quality, 3),
        groupthink_risk=round(groupthink, 3),
        logical_flaws=flaws,
        missing_risks=missing,
        regime_appropriate=regime_appropriate,
        recommendation=recommendation,
        confidence_adjustment=round(adjustment, 3),
        review_note=f"Heuristic audit: {len(flaws)} flaws, groupthink={groupthink:.0%}",
    )


def audit_batch(
    decisions: List[dict],
    max_audits: int = 8,
) -> Dict[str, AuditResult]:
    """
    Audit a batch of trade decisions, prioritizing highest-confidence ones.

    Args:
        decisions: List of dicts with keys matching audit_decision params.
        max_audits: Max LLM audits per cycle (cost control).

    Returns:
        {symbol: AuditResult} dict.
    """
    # Sort by confidence (audit the most confident first — they're most impactful)
    sorted_decisions = sorted(
        decisions,
        key=lambda d: d.get("confidence", 0),
        reverse=True,
    )[:max_audits]

    results = {}
    for d in sorted_decisions:
        result = audit_decision(
            symbol=d["symbol"],
            direction=d.get("direction", "long"),
            approval_pct=d.get("approval_pct", 0.5),
            confidence=d.get("confidence", 0.5),
            supporting_reasons=d.get("supporting_reasons", []),
            n_dissent=d.get("n_dissent", 0),
            strategies=d.get("strategies", []),
            regime=d.get("regime", "unknown"),
            sector=d.get("sector", "unknown"),
            sentiment=d.get("sentiment", "neutral"),
        )
        results[d["symbol"]] = result

    return results

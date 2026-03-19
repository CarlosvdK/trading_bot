"""
Regime Narrator — uses Claude to explain market regime shifts in plain English.

When the HMM/KMeans regime detector fires a regime change, this module:
1. Explains what likely caused the shift (macro events, vol spike, etc.)
2. Recommends allocation adjustments in plain language
3. Lists historical analogues for the current regime
4. Flags specific risks to watch

This turns opaque regime labels like "high_vol_trending_down" into
actionable intelligence that both agents and humans can use.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NARRATOR_SYSTEM_PROMPT = """You are a macro strategist at a hedge fund.
Your job is to explain market regime changes in clear, actionable terms.
Connect quantitative regime labels to real-world drivers.
Be specific about what to do differently in each regime.
Return ONLY valid JSON."""

REGIME_PROMPT_TEMPLATE = """The trading system's regime detector just identified a regime change.

Previous regime: {prev_regime}
New regime: {new_regime}
Transition date: {date}

Quantitative context:
- 5d index return: {ret_5d}
- 21d index return: {ret_21d}
- 21d realized vol: {vol_21d} (annualized)
- 63d realized vol: {vol_63d} (annualized)
- Vol ratio (short/long): {vol_ratio}
- Trend strength (% up days, 21d): {trend_strength}
- Drawdown from 63d peak: {drawdown}

Current regime allocation rules:
- Swing trading multiplier: {swing_mult}
- Swing trading enabled: {swing_enabled}

Return this JSON:
{{
  "narrative": "<3-4 sentence plain-English explanation of what's happening and why>",
  "likely_drivers": ["<macro driver 1>", "<driver 2>"],
  "action_items": ["<specific action for swing trading>", "<action for risk management>"],
  "historical_analogue": "<brief reference to a similar historical period if applicable>",
  "risk_watchlist": ["<specific risk to monitor>", "<risk 2>"],
  "expected_duration": "<days|weeks|months estimate for this regime>",
  "opportunity_sectors": ["<sectors that tend to outperform in this regime>"],
  "avoid_sectors": ["<sectors that tend to underperform>"],
  "confidence": <float 0-1, how confident in this regime classification>
}}"""


@dataclass
class RegimeNarrative:
    """Human-readable regime change explanation."""
    prev_regime: str
    new_regime: str
    date: str
    narrative: str = ""
    likely_drivers: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    historical_analogue: str = ""
    risk_watchlist: List[str] = field(default_factory=list)
    expected_duration: str = ""
    opportunity_sectors: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)
    confidence: float = 0.5


def narrate_regime_change(
    prev_regime: str,
    new_regime: str,
    date: str,
    regime_features: dict,
    swing_mult: float = 1.0,
    swing_enabled: bool = True,
) -> RegimeNarrative:
    """
    Generate a narrative explanation of a regime change.

    Args:
        prev_regime: Previous regime label (e.g., "low_vol_trending_up").
        new_regime: New regime label.
        date: Date string of the transition.
        regime_features: Dict of quantitative features at transition.
        swing_mult: Current swing trading multiplier.
        swing_enabled: Whether swing trading is enabled.

    Returns:
        RegimeNarrative with explanation and recommendations.
    """
    from src.market_intel.llm_client import query, is_available

    if not is_available():
        return _heuristic_narrative(prev_regime, new_regime, date, regime_features)

    prompt = REGIME_PROMPT_TEMPLATE.format(
        prev_regime=prev_regime,
        new_regime=new_regime,
        date=date,
        ret_5d=f"{regime_features.get('ret_5d', 0):.3f}",
        ret_21d=f"{regime_features.get('ret_21d', 0):.3f}",
        vol_21d=f"{regime_features.get('vol_21d', 0):.1%}",
        vol_63d=f"{regime_features.get('vol_63d', 0):.1%}",
        vol_ratio=f"{regime_features.get('vol_ratio', 1):.2f}",
        trend_strength=f"{regime_features.get('trend_strength', 0.5):.1%}",
        drawdown=f"{regime_features.get('drawdown_63d', 0):.2%}",
        swing_mult=swing_mult,
        swing_enabled=swing_enabled,
    )

    response = query(
        prompt=prompt,
        system=NARRATOR_SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.3,
    )

    if not response:
        return _heuristic_narrative(prev_regime, new_regime, date, regime_features)

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        return RegimeNarrative(
            prev_regime=prev_regime,
            new_regime=new_regime,
            date=date,
            narrative=data.get("narrative", ""),
            likely_drivers=data.get("likely_drivers", []),
            action_items=data.get("action_items", []),
            historical_analogue=data.get("historical_analogue", ""),
            risk_watchlist=data.get("risk_watchlist", []),
            expected_duration=data.get("expected_duration", ""),
            opportunity_sectors=data.get("opportunity_sectors", []),
            avoid_sectors=data.get("avoid_sectors", []),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse regime narrative: {e}")
        return _heuristic_narrative(prev_regime, new_regime, date, regime_features)


def _heuristic_narrative(
    prev_regime: str,
    new_regime: str,
    date: str,
    features: dict,
) -> RegimeNarrative:
    """Fallback rule-based narrative when LLM is unavailable."""
    narratives = {
        "high_vol_trending_down": (
            "Market shifted to high-volatility downtrend. Elevated fear and selling "
            "pressure suggest risk-off positioning. Reduce exposure and tighten stops."
        ),
        "high_vol_trending_up": (
            "Volatile but trending higher — likely a relief rally or rotation. "
            "Participate cautiously with smaller positions and wider stops."
        ),
        "high_vol_choppy": (
            "Whipsaw conditions with no clear trend. This is the worst regime for "
            "swing trading. Stand aside or reduce to minimum exposure."
        ),
        "low_vol_trending_up": (
            "Ideal conditions: calm markets trending higher. Maximum conviction on "
            "quality setups. This is where the bulk of swing profits are made."
        ),
        "low_vol_choppy": (
            "Calm but directionless. Narrow range-bound action. Mean reversion "
            "strategies work best here; momentum strategies will struggle."
        ),
        "low_vol_trending_down": (
            "Quiet selloff — often the most dangerous regime as complacency builds. "
            "Be selective, favor defensive sectors and short-biased setups."
        ),
    }

    narrative = narratives.get(
        new_regime,
        f"Regime changed from {prev_regime} to {new_regime}. Review allocation."
    )

    actions = []
    if "high_vol" in new_regime:
        actions.append("Reduce position sizes by 50%")
        actions.append("Tighten stop losses")
    if "trending_down" in new_regime:
        actions.append("Favor short-biased or defensive setups")
    if "trending_up" in new_regime:
        actions.append("Lean into momentum and breakout strategies")
    if "choppy" in new_regime:
        actions.append("Favor mean reversion over momentum")

    return RegimeNarrative(
        prev_regime=prev_regime,
        new_regime=new_regime,
        date=date,
        narrative=narrative,
        likely_drivers=["Quantitative regime shift detected"],
        action_items=actions,
        confidence=0.4,
    )

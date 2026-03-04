"""
Cross-Stock Reasoning Engine — propagates news impact to related stocks.

When news hits Stock A, this module figures out which other stocks are
affected and in what direction. Uses three types of relationships:

1. Supply chain: AAPL news → suppliers (AVGO, QCOM, TSM) benefit/suffer
2. Competitors: NVDA good news → AMD may also benefit (sector lift) or lose (market share)
3. Sector/thematic: "AI spending boom" → all AI stocks benefit

The key insight: news about ONE stock often moves MANY stocks, and the
bot that figures this out first has an edge.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Stock Relationship Graph
# -----------------------------------------------------------------------
# Relationships: (related_symbol, relationship_type, impact_direction, strength)
#   impact_direction: "same" = moves same way, "inverse" = moves opposite
#   strength: 0.0-1.0 how strongly related

STOCK_RELATIONSHIPS: Dict[str, List[Tuple[str, str, str, float]]] = {
    # === MEGA-CAP TECH ===
    "AAPL": [
        ("AVGO", "supplier", "same", 0.6),
        ("QCOM", "supplier", "same", 0.5),
        ("SWKS", "supplier", "same", 0.5),
        ("XLK", "sector_etf", "same", 0.4),
        ("MSFT", "peer", "same", 0.3),
    ],
    "MSFT": [
        ("GOOGL", "competitor", "same", 0.3),  # Cloud competitor but sector lift
        ("CRM", "competitor", "same", 0.3),
        ("NOW", "peer", "same", 0.3),
        ("XLK", "sector_etf", "same", 0.4),
    ],
    "GOOGL": [
        ("META", "competitor", "same", 0.4),
        ("SNAP", "competitor", "inverse", 0.3),  # Ad competition
        ("PINS", "competitor", "inverse", 0.2),
        ("XLC", "sector_etf", "same", 0.4),
    ],
    "AMZN": [
        ("SHOP", "competitor", "inverse", 0.3),
        ("EBAY", "competitor", "inverse", 0.2),
        ("XLY", "sector_etf", "same", 0.4),
        ("WMT", "competitor", "inverse", 0.2),
    ],
    "META": [
        ("GOOGL", "competitor", "same", 0.3),
        ("SNAP", "competitor", "same", 0.4),  # Social/ad sector lift
        ("PINS", "competitor", "same", 0.3),
        ("XLC", "sector_etf", "same", 0.4),
    ],

    # === SEMICONDUCTORS ===
    "NVDA": [
        ("AMD", "competitor", "same", 0.6),  # GPU/AI sector lift
        ("AVGO", "peer", "same", 0.5),
        ("MRVL", "peer", "same", 0.4),
        ("SMH", "sector_etf", "same", 0.7),
        ("MU", "supplier", "same", 0.5),
        ("MARA", "customer", "same", 0.3),  # GPU miners
        ("RIOT", "customer", "same", 0.3),
        ("LRCX", "supplier", "same", 0.4),  # Equipment
        ("AMAT", "supplier", "same", 0.4),
        ("KLAC", "supplier", "same", 0.3),
    ],
    "AMD": [
        ("NVDA", "competitor", "same", 0.6),
        ("INTC", "competitor", "inverse", 0.4),
        ("SMH", "sector_etf", "same", 0.6),
        ("MU", "peer", "same", 0.4),
    ],
    "AVGO": [
        ("NVDA", "peer", "same", 0.5),
        ("QCOM", "peer", "same", 0.4),
        ("SMH", "sector_etf", "same", 0.5),
    ],

    # === CRYPTO ===
    "COIN": [
        ("MARA", "peer", "same", 0.7),
        ("RIOT", "peer", "same", 0.7),
        ("MSTR", "peer", "same", 0.6),
        ("HUT", "peer", "same", 0.5),
        ("BITF", "peer", "same", 0.5),
        ("CLSK", "peer", "same", 0.5),
    ],
    "MSTR": [
        ("COIN", "peer", "same", 0.6),
        ("MARA", "peer", "same", 0.6),
        ("RIOT", "peer", "same", 0.6),
    ],

    # === BIOTECH/PHARMA ===
    "MRNA": [
        ("BNTX", "competitor", "same", 0.6),  # mRNA sector lift
        ("PFE", "partner", "same", 0.3),
        ("XBI", "sector_etf", "same", 0.5),
    ],
    "LLY": [
        ("NVO", "competitor", "same", 0.5),  # GLP-1 sector
        ("AMGN", "peer", "same", 0.3),
        ("XLV", "sector_etf", "same", 0.4),
    ],
    "PFE": [
        ("MRNA", "competitor", "same", 0.4),
        ("JNJ", "peer", "same", 0.3),
        ("XLV", "sector_etf", "same", 0.4),
    ],

    # === BANKS/FINANCIALS ===
    "JPM": [
        ("BAC", "peer", "same", 0.6),
        ("GS", "peer", "same", 0.5),
        ("MS", "peer", "same", 0.5),
        ("WFC", "peer", "same", 0.5),
        ("C", "peer", "same", 0.5),
        ("XLF", "sector_etf", "same", 0.6),
    ],
    "GS": [
        ("MS", "peer", "same", 0.7),
        ("JPM", "peer", "same", 0.5),
        ("XLF", "sector_etf", "same", 0.5),
    ],

    # === ENERGY ===
    "XOM": [
        ("CVX", "peer", "same", 0.7),
        ("COP", "peer", "same", 0.5),
        ("SLB", "service", "same", 0.4),
        ("XLE", "sector_etf", "same", 0.6),
        ("OXY", "peer", "same", 0.4),
    ],
    "CVX": [
        ("XOM", "peer", "same", 0.7),
        ("COP", "peer", "same", 0.5),
        ("XLE", "sector_etf", "same", 0.6),
    ],

    # === EV / AUTO ===
    "TSLA": [
        ("NIO", "competitor", "same", 0.4),
        ("XPEV", "competitor", "same", 0.3),
        ("LI", "competitor", "same", 0.3),
        ("F", "competitor", "same", 0.2),
        ("GM", "competitor", "same", 0.2),
        ("XLY", "sector_etf", "same", 0.3),
    ],

    # === RETAIL ===
    "WMT": [
        ("COST", "peer", "same", 0.5),
        ("TGT", "peer", "same", 0.5),
        ("XLP", "sector_etf", "same", 0.4),
    ],

    # === CLOUD/SAAS ===
    "CRM": [
        ("NOW", "peer", "same", 0.5),
        ("HUBS", "peer", "same", 0.4),
        ("WDAY", "peer", "same", 0.4),
        ("DDOG", "peer", "same", 0.3),
    ],
    "SNOW": [
        ("DDOG", "peer", "same", 0.5),
        ("MDB", "peer", "same", 0.5),
        ("NET", "peer", "same", 0.3),
        ("CFLT", "peer", "same", 0.4),
    ],
}

# Event-type sector mappings: which stocks get affected by macro events
EVENT_SECTOR_IMPACT = {
    "macro": {
        "rate_hike": [
            # Banks benefit from higher rates
            (["JPM", "BAC", "GS", "WFC", "C", "XLF"], "same", 0.4),
            # Growth/tech hurt by higher rates
            (["XLK", "ARKK", "QQQ"], "inverse", 0.3),
            # Real estate hurt
            (["XLRE", "O", "AMT"], "inverse", 0.4),
        ],
        "rate_cut": [
            (["JPM", "BAC", "GS", "XLF"], "inverse", 0.3),
            (["XLK", "QQQ", "ARKK"], "same", 0.4),
            (["XLRE", "O", "AMT"], "same", 0.4),
        ],
        "recession_fear": [
            (["XLP", "XLU", "GLD"], "same", 0.3),  # Defensives benefit
            (["XLY", "XLI"], "inverse", 0.4),  # Cyclicals hurt
        ],
    },
    "trade_policy": {
        "china_tariff": [
            (["BABA", "JD", "PDD", "NIO", "XPEV"], "inverse", 0.5),
            (["FXI"], "inverse", 0.6),
        ],
        "chip_export": [
            (["NVDA", "AMD", "AVGO", "LRCX", "AMAT"], "inverse", 0.4),
            (["SMH"], "inverse", 0.5),
        ],
    },
    "crypto": {
        "bitcoin_rally": [
            (["COIN", "MSTR", "MARA", "RIOT", "HUT", "BITF", "CLSK"], "same", 0.6),
        ],
        "crypto_regulation": [
            (["COIN", "MSTR", "MARA", "RIOT"], "inverse", 0.5),
        ],
    },
    "tech_sector": {
        "ai_spending": [
            (["NVDA", "AMD", "AVGO", "MSFT", "GOOGL", "META", "SNOW", "PLTR"], "same", 0.5),
            (["SMH"], "same", 0.5),
        ],
    },
}


def get_related_stocks(
    symbol: str,
    event_type: str = "general",
    title: str = "",
) -> List[dict]:
    """
    Get stocks related to a given symbol, considering the event type.

    Returns list of dicts:
        symbol, relationship, impact_direction, strength
    """
    related = []

    # 1. Direct stock relationships
    if symbol in STOCK_RELATIONSHIPS:
        for rel_sym, rel_type, direction, strength in STOCK_RELATIONSHIPS[symbol]:
            related.append({
                "symbol": rel_sym,
                "relationship": rel_type,
                "impact_direction": direction,
                "strength": strength,
                "source_symbol": symbol,
            })

    # 2. Reverse relationships (if AMD is related to NVDA, NVDA is related to AMD)
    for src_sym, relationships in STOCK_RELATIONSHIPS.items():
        if src_sym == symbol:
            continue
        for rel_sym, rel_type, direction, strength in relationships:
            if rel_sym == symbol:
                related.append({
                    "symbol": src_sym,
                    "relationship": rel_type,
                    "impact_direction": direction,
                    "strength": strength * 0.8,  # Slightly weaker reverse
                    "source_symbol": symbol,
                })

    # 3. Event-based sector impacts
    title_lower = title.lower()
    if event_type in EVENT_SECTOR_IMPACT:
        for sub_event, impacts in EVENT_SECTOR_IMPACT[event_type].items():
            # Check if title matches the sub-event
            if _title_matches_subevent(title_lower, sub_event):
                for syms, direction, strength in impacts:
                    for sym in syms:
                        if sym != symbol:
                            related.append({
                                "symbol": sym,
                                "relationship": f"event:{sub_event}",
                                "impact_direction": direction,
                                "strength": strength,
                                "source_symbol": symbol,
                            })

    # Deduplicate (keep strongest relationship per symbol)
    best = {}
    for r in related:
        sym = r["symbol"]
        if sym not in best or r["strength"] > best[sym]["strength"]:
            best[sym] = r

    return list(best.values())


def propagate_sentiment(
    symbol_sentiment: Dict[str, dict],
    known_symbols: List[str],
) -> Dict[str, dict]:
    """
    Take direct sentiment scores and propagate to related stocks.

    This is the core reasoning engine:
    - If NVDA has sentiment +0.8 and AMD is related with strength 0.6,
      AMD gets an additional propagated sentiment of +0.8 * 0.6 * 0.5 = +0.24

    The propagated score is added to any direct sentiment the stock already has.

    Returns updated sentiment dict with propagated scores added.
    """
    propagated = dict(symbol_sentiment)  # Start with direct scores

    for sym, sent in symbol_sentiment.items():
        score = sent.get("score", 0)
        if abs(score) < 0.1:
            continue  # Only propagate meaningful signals

        # Get this headline's dominant event type
        headlines = sent.get("headlines", [])
        event_type = "general"
        if headlines:
            from src.analysis.sentiment import detect_event_type
            event_type = detect_event_type(headlines[0])

        related = get_related_stocks(sym, event_type, headlines[0] if headlines else "")

        for rel in related:
            rel_sym = rel["symbol"]
            if rel_sym not in known_symbols:
                continue

            # Calculate propagated impact
            direction_mult = 1.0 if rel["impact_direction"] == "same" else -1.0
            propagated_score = score * rel["strength"] * direction_mult * 0.5

            if abs(propagated_score) < 0.02:
                continue

            if rel_sym in propagated:
                # Blend with existing sentiment
                existing = propagated[rel_sym]
                existing["score"] = existing["score"] + propagated_score
                existing["score"] = max(-1.0, min(1.0, existing["score"]))
                existing["magnitude"] = abs(existing["score"])
                if "propagated_from" not in existing:
                    existing["propagated_from"] = []
                existing["propagated_from"].append({
                    "source": sym,
                    "relationship": rel["relationship"],
                    "impact": round(propagated_score, 4),
                })
            else:
                # New entry — entirely from propagation
                propagated[rel_sym] = {
                    "score": round(propagated_score, 4),
                    "magnitude": round(abs(propagated_score), 4),
                    "n_articles": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "news_volume_signal": 0,
                    "headlines": [],
                    "propagated_from": [{
                        "source": sym,
                        "relationship": rel["relationship"],
                        "impact": round(propagated_score, 4),
                    }],
                }

    return propagated


def _title_matches_subevent(title_lower: str, sub_event: str) -> bool:
    """Check if a headline matches a sub-event pattern."""
    patterns = {
        "rate_hike": r"rate hike|rate increase|raises rate|hawkish",
        "rate_cut": r"rate cut|rate decrease|lowers rate|dovish",
        "recession_fear": r"recession|economic slowdown|contraction|downturn",
        "china_tariff": r"china.*tariff|tariff.*china|trade war|china.*ban",
        "chip_export": r"chip.*export|export.*control|semiconductor.*ban",
        "bitcoin_rally": r"bitcoin.*ris|bitcoin.*rally|btc.*surge|crypto.*surge",
        "crypto_regulation": r"crypto.*regulat|sec.*crypto|crypto.*ban|crypto.*crack",
        "ai_spending": r"ai.*spend|artificial intelligence.*invest|ai.*boom|ai.*capex",
    }

    import re
    pattern = patterns.get(sub_event, sub_event.replace("_", ".*"))
    return bool(re.search(pattern, title_lower))

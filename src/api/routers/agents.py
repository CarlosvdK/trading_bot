"""Agent endpoints — profiles, scores, network graph."""

from fastapi import APIRouter, Depends
from src.api.dependencies import get_repo
from src.agents.agent_definitions import ALL_AGENTS, get_agent_by_id, get_agents_by_peer_group

router = APIRouter(prefix="/api/agents", tags=["Agents"])

# Sector color and shape mapping
SECTOR_COLORS = {
    "technology": "#6366F1", "healthcare": "#EC4899", "financials": "#10B981",
    "energy": "#F59E0B", "consumer_discretionary": "#8B5CF6", "consumer_staples": "#06B6D4",
    "industrials": "#F97316", "materials": "#84CC16", "real_estate": "#14B8A6",
    "utilities": "#64748B", "communication_services": "#3B82F6", "all": "#6366F1",
    # Sub-industries inherit from parent
    "semiconductors": "#6366F1", "cloud_saas": "#6366F1", "biotech": "#EC4899",
    "pharma": "#EC4899", "banks": "#10B981", "insurance": "#10B981",
    "oil_gas": "#F59E0B", "renewables": "#F59E0B", "defense": "#F97316",
    "aerospace": "#F97316", "mining": "#84CC16", "reits": "#14B8A6",
    "retail": "#8B5CF6", "food_beverage": "#06B6D4", "media": "#3B82F6",
    "telecom": "#3B82F6", "gaming": "#3B82F6", "auto": "#8B5CF6",
    "construction": "#F97316", "cyber_security": "#6366F1",
}

SECTOR_SHAPES = {
    "technology": "circle", "healthcare": "diamond", "financials": "square",
    "energy": "hexagon", "consumer_discretionary": "triangle", "consumer_staples": "pentagon",
    "industrials": "octagon", "materials": "star", "real_estate": "diamond",
    "utilities": "square", "communication_services": "hexagon", "all": "circle",
    "semiconductors": "circle", "cloud_saas": "circle", "biotech": "diamond",
    "pharma": "diamond", "banks": "square", "insurance": "square",
    "oil_gas": "hexagon", "renewables": "hexagon", "defense": "octagon",
    "aerospace": "octagon", "mining": "star", "reits": "diamond",
    "retail": "triangle", "food_beverage": "pentagon", "media": "hexagon",
    "telecom": "hexagon", "gaming": "hexagon", "auto": "triangle",
    "construction": "octagon", "cyber_security": "circle",
}


def _dna_to_dict(dna):
    """Convert AgentDNA to serializable dict."""
    return {
        "agentId": dna.agent_id,
        "displayName": dna.display_name,
        "primarySectors": dna.primary_sectors,
        "secondarySectors": dna.secondary_sectors,
        "primaryStrategy": dna.primary_strategy,
        "secondaryStrategy": dna.secondary_strategy,
        "riskAppetite": dna.risk_appetite,
        "contrarianFactor": dna.contrarian_factor,
        "convictionStyle": dna.conviction_style,
        "regimeSensitivity": dna.regime_sensitivity,
        "lookbackDays": dna.lookback_days,
        "holdingPeriod": dna.holding_period,
        "minConfidence": dna.min_confidence,
        "maxPicksPerScan": dna.max_picks_per_scan,
        "peerGroup": dna.peer_group,
    }


def _get_score_for_agent(agent_id, scores_map):
    """Get score dict for an agent from Supabase scores map."""
    score = scores_map.get(agent_id, {})
    return {
        "compositeWeight": score.get("composite_weight", 1.0),
        "riskAdjustedReturn": score.get("risk_adjusted_return", 0.0),
        "calibrationQuality": score.get("calibration_quality", 0.0),
        "reasoningQuality": score.get("reasoning_quality", 0.0),
        "uniqueness": score.get("uniqueness", 0.0),
        "regimeEffectiveness": score.get("regime_effectiveness", 0.0),
        "drawdownBehavior": score.get("drawdown_behavior", 0.0),
        "stability": score.get("stability", 0.0),
        "falsePositiveRate": score.get("false_positive_rate", 0.0),
        "falseNegativeRate": score.get("false_negative_rate", 0.0),
        "peerVoteAccuracy": score.get("peer_vote_accuracy", 0.0),
        "nOutcomes": score.get("n_outcomes", 0),
        "hitRate": score.get("hit_rate", 0.0),
        "status": score.get("status", "healthy"),
    }


@router.get("/leaderboard")
async def agent_leaderboard(repo=Depends(get_repo)):
    """Agents sorted by composite weight."""
    scores_map = {}
    if repo:
        try:
            rows = repo.get_agent_leaderboard()
            scores_map = {r["agent_id"]: r for r in rows}
        except Exception:
            pass

    results = []
    for dna in ALL_AGENTS:
        profile = _dna_to_dict(dna)
        profile["score"] = _get_score_for_agent(dna.agent_id, scores_map)
        results.append(profile)

    results.sort(key=lambda x: x["score"]["compositeWeight"], reverse=True)
    return results


@router.get("/network")
async def agent_network(repo=Depends(get_repo)):
    """Network graph data: nodes and edges for visualization."""
    scores_map = {}
    if repo:
        try:
            rows = repo.get_latest_agent_scores()
            scores_map = {r["agent_id"]: r for r in rows}
        except Exception:
            pass

    nodes = []
    for dna in ALL_AGENTS:
        primary = dna.primary_sectors[0] if dna.primary_sectors else "all"
        weight = scores_map.get(dna.agent_id, {}).get("composite_weight", 1.0)
        status = scores_map.get(dna.agent_id, {}).get("status", "healthy")
        score_val = scores_map.get(dna.agent_id, {}).get("hit_rate", 0.0)

        nodes.append({
            "id": dna.agent_id,
            "label": dna.display_name,
            "sector": primary,
            "shape": SECTOR_SHAPES.get(primary, "circle"),
            "strategy": dna.primary_strategy,
            "weight": weight,
            "status": status,
            "score": round(score_val, 3),
            "group": dna.peer_group,
            "color": SECTOR_COLORS.get(primary, "#6366F1"),
            "size": max(16, min(50, int(weight * 20))),
        })

    # Edges: agents sharing sector or strategy
    edges = []
    seen = set()
    for i, a in enumerate(ALL_AGENTS):
        for j, b in enumerate(ALL_AGENTS):
            if i >= j:
                continue
            key = f"{a.agent_id}:{b.agent_id}"
            if key in seen:
                continue

            shared = 0
            # Shared primary sectors
            for s in a.primary_sectors:
                if s in b.primary_sectors:
                    shared += 2
                elif s in b.secondary_sectors:
                    shared += 1
            # Shared strategy
            if a.primary_strategy == b.primary_strategy:
                shared += 1
            # Same peer group
            if a.peer_group and a.peer_group == b.peer_group:
                shared += 2

            if shared >= 2:
                seen.add(key)
                edges.append({
                    "source": a.agent_id,
                    "target": b.agent_id,
                    "weight": min(shared, 5),
                })

    return {"nodes": nodes, "edges": edges}


@router.get("")
async def list_agents(repo=Depends(get_repo)):
    """List all 121 agents with latest scores."""
    scores_map = {}
    if repo:
        try:
            rows = repo.get_latest_agent_scores()
            scores_map = {r["agent_id"]: r for r in rows}
        except Exception:
            pass

    results = []
    for dna in ALL_AGENTS:
        profile = _dna_to_dict(dna)
        profile["score"] = _get_score_for_agent(dna.agent_id, scores_map)
        results.append(profile)
    return results


@router.get("/{agent_id}")
async def get_agent(agent_id: str, repo=Depends(get_repo)):
    """Single agent detail with trade history."""
    dna = get_agent_by_id(agent_id)
    if not dna:
        return {"error": "Agent not found"}

    profile = _dna_to_dict(dna)

    # Scores
    scores_map = {}
    if repo:
        try:
            rows = repo.get_agent_score_history(agent_id, limit=1)
            if rows:
                scores_map = {agent_id: rows[0]}
        except Exception:
            pass
    profile["score"] = _get_score_for_agent(agent_id, scores_map)

    # Trades
    profile["trades"] = []
    if repo:
        try:
            profile["trades"] = repo.get_agent_trades(agent_id, limit=20)
        except Exception:
            pass

    # Lessons
    profile["lessons"] = []
    if repo:
        try:
            profile["lessons"] = repo.get_agent_lessons(agent_id, limit=20)
        except Exception:
            pass

    return profile


@router.get("/{agent_id}/trades")
async def get_agent_trades(agent_id: str, repo=Depends(get_repo)):
    if not repo:
        return []
    try:
        return repo.get_agent_trades(agent_id, limit=50)
    except Exception:
        return []


@router.get("/{agent_id}/lessons")
async def get_agent_lessons(agent_id: str, repo=Depends(get_repo)):
    if not repo:
        return []
    try:
        return repo.get_agent_lessons(agent_id, limit=50)
    except Exception:
        return []

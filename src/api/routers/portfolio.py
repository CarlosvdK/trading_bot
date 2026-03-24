"""Portfolio endpoints — live IBKR data with Supabase fallback."""

from fastapi import APIRouter, Depends
from src.api.dependencies import get_repo, get_ibkr
from src.agents.sector_mapping import get_sector

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("/summary")
async def portfolio_summary(ibkr=Depends(get_ibkr), repo=Depends(get_repo)):
    """Live portfolio summary from IBKR, fallback to Supabase snapshot."""
    # Try IBKR first
    if ibkr:
        try:
            acct = ibkr.get_account_summary()
            return {
                "totalValue": acct.get("NetLiquidation", 0),
                "cashBalance": acct.get("TotalCashValue", 0),
                "unrealizedPnl": acct.get("UnrealizedPnL", 0),
                "realizedPnl": acct.get("RealizedPnL", 0),
                "dailyPnl": acct.get("DailyPnL", 0),
                "grossExposure": acct.get("GrossPositionValue", 0),
                "netExposure": acct.get("NetLiquidation", 0) - acct.get("TotalCashValue", 0),
                "buyingPower": acct.get("BuyingPower", 0),
                "source": "ibkr_live",
            }
        except Exception:
            pass

    # Fallback to Supabase
    if repo:
        try:
            history = repo.get_portfolio_history(days=1)
            if history:
                snap = history[-1]
                return {
                    "totalValue": snap.get("total_value", 0),
                    "cashBalance": snap.get("cash_balance", 0),
                    "unrealizedPnl": snap.get("unrealized_pnl", 0),
                    "realizedPnl": snap.get("realized_pnl", 0),
                    "dailyPnl": snap.get("daily_pnl", 0),
                    "grossExposure": snap.get("gross_exposure", 0),
                    "netExposure": snap.get("net_exposure", 0),
                    "openPositions": snap.get("open_positions", 0),
                    "drawdown": snap.get("drawdown", 0),
                    "regime": snap.get("regime", ""),
                    "source": "supabase_snapshot",
                }
        except Exception:
            pass

    return {"totalValue": 0, "cashBalance": 0, "source": "unavailable"}


@router.get("/positions")
async def portfolio_positions(ibkr=Depends(get_ibkr), repo=Depends(get_repo)):
    """Live positions from IBKR cross-referenced with trade history."""
    positions = []

    # Get IBKR positions
    ibkr_positions = {}
    if ibkr:
        try:
            ibkr_positions = ibkr.get_positions()
        except Exception:
            pass

    # Get trade history for attribution
    trade_map = {}
    if repo:
        try:
            open_trades = repo.get_open_trades()
            for t in open_trades:
                trade_map[t.get("symbol", "")] = t
        except Exception:
            pass

    for symbol, pos_data in ibkr_positions.items():
        trade = trade_map.get(symbol, {})
        sector = get_sector(symbol) or ""
        qty = pos_data.get("quantity", pos_data.get("position", 0))
        avg_cost = pos_data.get("avgCost", pos_data.get("avg_cost", 0))
        market_price = pos_data.get("marketPrice", pos_data.get("market_price", avg_cost))
        market_value = pos_data.get("marketValue", qty * market_price)
        unreal_pnl = pos_data.get("unrealizedPNL", (market_price - avg_cost) * qty)

        positions.append({
            "symbol": symbol,
            "direction": "long" if qty > 0 else "short",
            "quantity": abs(qty),
            "avgEntryPrice": avg_cost,
            "currentPrice": market_price,
            "marketValue": abs(market_value),
            "unrealizedPnl": unreal_pnl,
            "unrealizedPnlPct": (unreal_pnl / (abs(avg_cost * qty))) if avg_cost * qty != 0 else 0,
            "sector": sector,
            "strategy": trade.get("strategy", ""),
            "agentId": trade.get("agent_id", ""),
            "thesis": trade.get("reasoning", ""),
            "confidence": trade.get("confidence", 0),
            "tradeId": trade.get("id", ""),
            "openTimestamp": trade.get("created_at", ""),
        })

    return positions


@router.get("/trades")
async def recent_trades(limit: int = 50, repo=Depends(get_repo)):
    """Recent trade proposals with vote results, reasoning, and outcomes."""
    if not repo:
        return []
    try:
        return repo.get_recent_trades(limit=limit)
    except Exception:
        return []


@router.get("/history")
async def portfolio_history(days: int = 90, repo=Depends(get_repo)):
    if not repo:
        return []
    try:
        return repo.get_portfolio_history(days=days)
    except Exception:
        return []


@router.get("/allocation")
async def portfolio_allocation(ibkr=Depends(get_ibkr), repo=Depends(get_repo)):
    """Compute allocation breakdowns from live positions."""
    positions = (await portfolio_positions(ibkr, repo))
    if not positions:
        return {"bySector": [], "byStrategy": [], "byDirection": []}

    total_value = sum(p["marketValue"] for p in positions)
    if total_value == 0:
        return {"bySector": [], "byStrategy": [], "byDirection": []}

    # By sector
    sector_map = {}
    for p in positions:
        s = p["sector"] or "Other"
        sector_map[s] = sector_map.get(s, 0) + p["marketValue"]
    by_sector = [{"label": k, "value": v, "pct": v / total_value} for k, v in sector_map.items()]

    # By strategy
    strat_map = {}
    for p in positions:
        s = p["strategy"] or "unattributed"
        strat_map[s] = strat_map.get(s, 0) + p["marketValue"]
    by_strategy = [{"label": k, "value": v, "pct": v / total_value} for k, v in strat_map.items()]

    # By direction
    long_val = sum(p["marketValue"] for p in positions if p["direction"] == "long")
    short_val = sum(p["marketValue"] for p in positions if p["direction"] == "short")
    by_direction = [
        {"label": "Long", "value": long_val, "pct": long_val / total_value},
        {"label": "Short", "value": short_val, "pct": short_val / total_value},
    ]

    return {"bySector": by_sector, "byStrategy": by_strategy, "byDirection": by_direction}

"""Pipeline endpoints — trigger scans, view results."""

import asyncio
import uuid
import logging
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, BackgroundTasks
from src.api.dependencies import get_repo, get_ibkr, get_risk_governor

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])
logger = logging.getLogger("pipeline_api")

# In-memory scan status tracking
_scan_status = {}


@router.get("/status")
async def pipeline_status(repo=Depends(get_repo)):
    """Latest pipeline run status."""
    if not repo:
        return {"status": "no_database"}
    try:
        runs = repo.get_recent_pipeline_runs(limit=1)
        return runs[0] if runs else {"status": "no_runs"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/recent")
async def pipeline_recent(repo=Depends(get_repo)):
    if not repo:
        return []
    try:
        return repo.get_recent_pipeline_runs(limit=10)
    except Exception:
        return []


@router.get("/funnel")
async def pipeline_funnel(repo=Depends(get_repo)):
    """Latest funnel stats."""
    if not repo:
        return {"scanned": 0, "surfaced": 0, "voted": 0, "approved": 0}
    try:
        runs = repo.get_recent_pipeline_runs(limit=1)
        if runs and "stats" in runs[0]:
            return runs[0]["stats"]
        return {"scanned": 0, "surfaced": 0, "voted": 0, "approved": 0}
    except Exception:
        return {"scanned": 0, "surfaced": 0, "voted": 0, "approved": 0}


@router.get("/scan/{scan_id}")
async def get_scan_status(scan_id: str):
    """Check status of a background scan."""
    return _scan_status.get(scan_id, {"status": "not_found"})


def _run_scan(scan_id: str, repo, ibkr, risk_governor):
    """Background task: run the full agent pipeline."""
    import yfinance as yf

    _scan_status[scan_id] = {"status": "running", "started": datetime.now().isoformat()}

    try:
        # Record pipeline start
        db_scan_id = None
        regime = "unknown"
        if repo:
            try:
                db_scan_id = repo.start_pipeline_run(regime=regime)
            except Exception:
                pass

        # Load universe data via yfinance
        logger.info("Loading universe data...")
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
            "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK",
            "ABBV", "KO", "PEP", "BAC", "COST", "TMO", "MCD", "CRM", "AMD",
            "NFLX", "INTC", "CSCO", "ABT", "DHR", "TXN", "NEE", "UPS",
            "PM", "RTX", "HON", "LOW", "QCOM", "CAT", "GE", "BA", "GS",
            "SLB", "DE", "MMM", "IBM", "LMT", "NKE",
        ]
        universe_data = {}
        index_df = None

        try:
            data = yf.download(symbols + ["SPY"], period="6mo", group_by="ticker", progress=False)
            for sym in symbols:
                try:
                    if sym in data.columns.get_level_values(0):
                        df = data[sym].dropna()
                        df.columns = [c.lower() for c in df.columns]
                        if not df.empty and len(df) > 30:
                            universe_data[sym] = df
                except Exception:
                    continue
            # Index
            try:
                idx = data["SPY"].dropna()
                idx.columns = [c.lower() for c in idx.columns]
                index_df = idx
            except Exception:
                pass
        except Exception as e:
            logger.error(f"yfinance download failed: {e}")

        if not universe_data:
            _scan_status[scan_id] = {"status": "failed", "error": "No market data"}
            return

        logger.info(f"Loaded {len(universe_data)} symbols")

        # Build agents and run pipeline
        from src.agents.agent_definitions import ALL_AGENTS
        from src.agents.trading_agent import TradingAgent
        from src.agents.agent_pool import AgentPool
        from src.agents.voting_engine import VotingEngine
        from src.agents.scorekeeper import AgentScorekeeper

        agents = [TradingAgent(dna) for dna in ALL_AGENTS]
        engine = VotingEngine(approval_threshold=0.3, min_voters=3)
        scorekeeper = AgentScorekeeper()
        pool = AgentPool(agents, engine, scorekeeper)

        logger.info("Running daily scan with 121 agents...")

        # Log scan start to activity feed
        if repo:
            try:
                repo.log_activity("system", "news_scan", f"Manual pipeline scan triggered — scanning {len(universe_data)} symbols with 121 agents", market_mode="market")
            except Exception:
                pass

        current_date = pd.Timestamp.now().normalize()
        approved = pool.daily_scan(universe_data, index_df, current_date)
        stats = pool.get_scan_stats()

        logger.info(f"Scan complete: {len(approved)} trades approved")

        # Record to Supabase
        approved_details = []
        if repo:
            for trade in approved:
                try:
                    trade_id = repo.record_proposal(
                        agent_id=trade.supporting_agents[0] if trade.supporting_agents else "system",
                        symbol=trade.symbol,
                        direction=trade.direction,
                        confidence=trade.weighted_confidence,
                        strategy="multi_agent_vote",
                        reasoning=f"Approved by {trade.num_voters} agents, {trade.approval_pct:.0%} approval",
                        hold_days=trade.consensus_hold_days,
                        sector=trade.sector,
                        regime=regime,
                    )
                    repo.record_vote_result(
                        trade_id=trade_id,
                        approval_pct=trade.approval_pct,
                        num_voters=trade.num_voters,
                        vote_result="approved",
                        supporting=trade.supporting_agents,
                        dissenting=trade.dissenting_agents,
                    )
                    approved_details.append({
                        "tradeId": trade_id,
                        "symbol": trade.symbol,
                        "direction": trade.direction,
                        "confidence": trade.weighted_confidence,
                        "approvalPct": trade.approval_pct,
                        "numVoters": trade.num_voters,
                        "holdDays": trade.consensus_hold_days,
                    })
                except Exception as e:
                    logger.warning(f"Failed to record trade {trade.symbol}: {e}")

            # Complete pipeline run
            if db_scan_id:
                try:
                    repo.complete_pipeline_run(db_scan_id, {
                        "scanned": stats.get("unique_symbols", 0),
                        "surfaced": stats.get("total_picks", 0),
                        "voted": stats.get("unique_symbols", 0),
                        "approved": len(approved),
                        "agents_active": stats.get("unique_agents", 0),
                    })
                except Exception:
                    pass

        # Log completion and approved trades to activity feed
        if repo:
            try:
                repo.log_activity("system", "execution", f"Pipeline scan complete — {len(approved)} trades approved from {len(universe_data)} symbols", market_mode="market")
                for trade in approved[:10]:
                    proposer = trade.supporting_agents[0] if trade.supporting_agents else "swarm"
                    repo.log_activity(
                        proposer, "proposal",
                        f"{trade.direction.upper()} {trade.symbol} — {trade.approval_pct:.0%} approval by {trade.num_voters} agents",
                        symbol=trade.symbol,
                        market_mode="market",
                    )
            except Exception:
                pass

        _scan_status[scan_id] = {
            "status": "completed",
            "completed": datetime.now().isoformat(),
            "approved": len(approved),
            "scanned": len(universe_data),
            "totalPicks": stats.get("total_picks", 0),
            "trades": approved_details,
        }

    except Exception as e:
        logger.error(f"Scan failed: {e}")
        _scan_status[scan_id] = {"status": "failed", "error": str(e)}


@router.post("/scan")
async def trigger_scan(
    background_tasks: BackgroundTasks,
    repo=Depends(get_repo),
    ibkr=Depends(get_ibkr),
    rg=Depends(get_risk_governor),
):
    """Trigger a full pipeline scan in the background."""
    scan_id = str(uuid.uuid4())[:8]
    _scan_status[scan_id] = {"status": "queued"}
    background_tasks.add_task(_run_scan, scan_id, repo, ibkr, rg)
    return {"scanId": scan_id, "status": "queued"}

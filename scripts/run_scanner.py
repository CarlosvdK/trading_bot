"""
Autonomous Scanner — runs the 121-agent pipeline on a schedule.

Market hours (9:30 AM - 4:00 PM ET, Mon-Fri):
  - Scans every 2 hours
  - Agents analyze, propose, vote, and record to Supabase

Off-hours:
  - Sleeps and checks every 15 minutes until market opens

Run standalone:  python scripts/run_scanner.py
Or via:          ./scripts/run_system.sh start
"""

import logging
import os
import sys
import time
from datetime import datetime

import pytz

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("scanner")

ET = pytz.timezone("America/New_York")
SCAN_INTERVAL_SECONDS = 2 * 60 * 60   # 2 hours during market
SLEEP_CHECK_SECONDS = 15 * 60          # 15 min off-hours


def is_market_hours() -> bool:
    """Check if US equity market is open (9:30-16:00 ET, Mon-Fri)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def run_scan():
    """Execute one full pipeline scan."""
    logger.info("=" * 60)
    logger.info("Starting pipeline scan...")
    logger.info("=" * 60)

    try:
        from src.storage.supabase_client import get_supabase_client
        from src.storage.repository import TradingRepository

        sb = get_supabase_client()
        if not sb:
            logger.error("Supabase not available. Skipping scan.")
            return

        repo = TradingRepository(sb)

        # Detect regime
        regime = "unknown"
        try:
            import yfinance as yf
            import numpy as np
            spy = yf.download("SPY", period="3mo", interval="1d", progress=False)
            if len(spy) >= 21:
                returns = spy["Close"].pct_change().dropna()
                vol_21d = float(returns.tail(21).std() * (252 ** 0.5))
                ret_21d = float((spy["Close"].iloc[-1] / spy["Close"].iloc[-22]) - 1)
                vol_label = "high_vol" if vol_21d > 0.20 else "low_vol"
                if ret_21d > 0.01:
                    trend = "trending_up"
                elif ret_21d < -0.01:
                    trend = "trending_down"
                else:
                    trend = "choppy"
                regime = f"{vol_label}_{trend}"
            logger.info(f"Detected regime: {regime}")
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")

        # Start pipeline run
        scan_id = repo.start_pipeline_run(regime=regime)
        logger.info(f"Pipeline run: {scan_id}")

        # Download market data
        import yfinance as yf
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "JPM", "BAC", "GS", "V", "MA",
            "JNJ", "UNH", "PFE", "ABBV", "MRK",
            "XOM", "CVX", "COP", "SLB",
            "HD", "MCD", "NKE", "SBUX", "TGT",
            "DIS", "NFLX", "CMCSA",
            "CAT", "DE", "HON", "GE",
            "CRM", "ADBE", "NOW", "SNOW",
            "AMD", "INTC", "AVGO", "QCOM",
            "LMT", "RTX", "BA",
            "NEE", "DUK", "SO",
            "SPY",
        ]

        logger.info(f"Downloading data for {len(symbols)} symbols...")
        data = yf.download(symbols, period="6mo", interval="1d", group_by="ticker", progress=False)
        logger.info(f"Data downloaded: {data.shape}")

        # Build OHLCV dict
        ohlcv = {}
        for sym in symbols:
            try:
                if sym in data.columns.get_level_values(0):
                    df = data[sym].dropna()
                    if len(df) >= 30:
                        ohlcv[sym] = df
            except Exception:
                pass
        logger.info(f"Valid OHLCV for {len(ohlcv)} symbols")

        # Load agents
        from src.agents.agent_definitions import ALL_AGENTS
        from src.agents.trading_agent import TradingAgent
        from src.agents.agent_pool import AgentPool
        from src.agents.voting_engine import VotingEngine

        agents = [TradingAgent(dna) for dna in ALL_AGENTS]
        voting = VotingEngine(approval_threshold=0.30, min_voters=3)
        pool = AgentPool(agents=agents, voting_engine=voting)

        logger.info(f"Loaded {len(agents)} agents")

        # Run scan
        results = pool.daily_scan(ohlcv, regime=regime)
        logger.info(f"Scan complete: {len(results.get('approved', []))} approved, {len(results.get('rejected', []))} rejected")

        # Record to Supabase
        proposals = results.get("all_proposals", [])
        approved = results.get("approved", [])

        for p in proposals:
            try:
                trade_id = repo.record_proposal(
                    agent_id=p.get("agent_id", ""),
                    symbol=p.get("symbol", ""),
                    direction=p.get("direction", "long"),
                    confidence=p.get("confidence", 0),
                    strategy=p.get("strategy", ""),
                    reasoning=p.get("reasoning", ""),
                    hold_days=p.get("hold_days", 5),
                    sector=p.get("sector", ""),
                    regime=regime,
                )
                # Record vote if available
                if "vote" in p:
                    v = p["vote"]
                    repo.record_vote_result(
                        trade_id=trade_id,
                        approval_pct=v.get("approval_pct", 0),
                        num_voters=v.get("num_voters", 0),
                        vote_result="approved" if p in approved else "rejected",
                        supporting=v.get("supporting", []),
                        dissenting=v.get("dissenting", []),
                    )
            except Exception as e:
                logger.error(f"Failed to record proposal: {e}")

        # Complete pipeline run
        repo.complete_pipeline_run(scan_id, {
            "universe_scanned": len(ohlcv),
            "proposals_generated": len(proposals),
            "voted_on": len(proposals),
            "approved": len(approved),
            "rejected": len(proposals) - len(approved),
        })

        logger.info(f"Pipeline run {scan_id} complete")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)


def main():
    logger.info("Autonomous Scanner started")
    logger.info(f"Scan interval: {SCAN_INTERVAL_SECONDS // 60} min (market hours)")
    logger.info(f"Sleep check: {SLEEP_CHECK_SECONDS // 60} min (off-hours)")

    last_scan = 0

    while True:
        try:
            now = time.time()

            if is_market_hours():
                if now - last_scan >= SCAN_INTERVAL_SECONDS:
                    run_scan()
                    last_scan = now
                    logger.info(f"Next scan in {SCAN_INTERVAL_SECONDS // 60} minutes")
                else:
                    remaining = SCAN_INTERVAL_SECONDS - (now - last_scan)
                    logger.debug(f"Next scan in {remaining // 60:.0f} minutes")
                time.sleep(60)  # Check every minute during market hours
            else:
                et_now = datetime.now(ET)
                logger.info(f"Market closed ({et_now.strftime('%a %H:%M ET')}). Sleeping {SLEEP_CHECK_SECONDS // 60} min...")
                time.sleep(SLEEP_CHECK_SECONDS)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")
            break
        except Exception as e:
            logger.error(f"Scanner loop error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
24/7 Trading Bot — never sleeps, always working.

Three modes that cycle automatically:
  OVERNIGHT (4 PM - 9:25 AM):
    - Scans news every 30 min
    - Accumulates sentiment, tracks catalysts
    - Builds morning playbook of trades to execute at open

  MARKET HOURS (9:30 AM - 4 PM):
    - Executes morning playbook at open
    - Runs full orchestrator pipeline (signals + ML + risk)
    - Monitors positions every 15 min for barrier hits
    - Reacts to breaking news intraday

  WEEKEND (Sat-Sun):
    - Retrains ML model (heavy compute, no market impact)
    - Scans for new universe additions
    - Analyzes sector rotation
    - Scans news every 2 hours

Usage:
    python scripts/run_live.py --broker paper
    python scripts/run_live.py --broker ibkr
    python scripts/run_live.py --once             # one daily cycle and exit
    python scripts/run_live.py --mode overnight    # force overnight mode
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_feeds.provider import CSVDataProvider
from src.data_feeds.missing import handle_missing_data
from src.data_feeds.screener import screen_universe
from src.data_feeds.live_prices import (
    update_price_data,
    get_live_quotes,
)
from src.utilities.config_loader import load_config
from src.risk_management.risk_governor import RiskGovernor, RiskConfig, PortfolioState
from src.backtesting.cost_model import CostModel
from src.trading.paper_broker import PaperBroker
from src.trading.order_manager import OrderManager
from src.brain.orchestrator import TradingOrchestrator
from src.market_intel.premarket import PreMarketAnalyzer, WeekendAnalyzer

from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.info(f"Shutdown signal received ({signum}). Finishing current cycle...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def get_et_now() -> datetime:
    """Get current time in US Eastern."""
    now_utc = datetime.now(timezone.utc)
    month = now_utc.month
    et_offset = -4 if 3 <= month <= 11 else -5
    return (now_utc + timedelta(hours=et_offset)).replace(tzinfo=None)


def get_market_mode() -> str:
    """
    Determine current market mode:
      'weekend'   — Saturday or Sunday
      'premarket' — weekday before 9:25 AM ET
      'market'    — weekday 9:25 AM - 4:05 PM ET (includes buffer)
      'overnight' — weekday after 4:05 PM ET
    """
    et = get_et_now()

    if et.weekday() >= 5:
        return "weekend"

    hour_min = et.hour * 60 + et.minute

    if hour_min < 9 * 60 + 25:
        return "premarket"
    elif hour_min <= 16 * 60 + 5:
        return "market"
    else:
        return "overnight"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def create_broker(broker_type: str, price_data: dict, config: dict):
    """Factory for broker instances."""
    if broker_type == "paper":
        cost_model = CostModel(**{
            k: v for k, v in config.get("cost_model", {}).items()
            if k in CostModel.__dataclass_fields__
        })
        return PaperBroker(price_data, cost_model, config)
    elif broker_type == "ibkr":
        try:
            from src.trading.ibkr_broker import IBKRBroker
            return IBKRBroker(config)
        except ImportError:
            logger.error(
                "IBKR broker not available. Install: pip install ib_insync\n"
                "Then ensure TWS or IB Gateway is running."
            )
            sys.exit(1)
    else:
        logger.error(f"Unknown broker type: {broker_type}")
        sys.exit(1)


def load_historical_data(config: dict, start_date: str = "2018-01-01") -> dict:
    """Load all historical OHLCV data from CSV files."""
    data_dir = config.get("data", {}).get("data_dir", str(ROOT / "data" / "ohlcv"))
    universe_file = config.get("data", {}).get(
        "universe_file", str(ROOT / "data" / "universe.csv")
    )

    provider = CSVDataProvider(
        data_dir=data_dir, universe_file=universe_file, validate_on_load=True
    )
    symbols = provider.available_symbols()
    logger.info(f"Found {len(symbols)} data files")

    price_data = {}
    for sym in symbols:
        try:
            df = provider.load_symbol(sym, start_date)
            cleaned = handle_missing_data(df, max_missing_pct=0.05)
            if cleaned is not None and len(cleaned) > 60:
                price_data[sym] = cleaned
        except Exception as e:
            logger.warning(f"{sym}: {e}")

    logger.info(f"Loaded {len(price_data)} symbols")
    return price_data


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_daily_cycle(
    orchestrator: TradingOrchestrator,
    price_data: dict,
    index_df: pd.DataFrame,
    premarket: PreMarketAnalyzer,
) -> dict:
    """Execute the main daily trading cycle at market open."""
    now = pd.Timestamp.now()

    logger.info(f"\n{'='*60}")
    logger.info(f"  MARKET OPEN — DAILY CYCLE: {now.date()}")
    logger.info(f"  NAV: ${orchestrator.portfolio_state.nav:,.2f}")
    logger.info(f"  Open positions: {len(orchestrator.open_positions)}")
    logger.info(f"{'='*60}")

    # 1. Update data with latest bars
    price_data = update_price_data(price_data)
    index_symbol = orchestrator.config.get("features", {}).get("index_symbol", "SPY")
    index_df = price_data.get(index_symbol, index_df)

    # 2. Apply overnight playbook as news signals
    if premarket.scan_count > 0:
        playbook = premarket.build_playbook(price_data=price_data)

        logger.info(f"\n  MORNING PLAYBOOK:")
        logger.info(f"  Market mood: {playbook['market_mood']}")
        logger.info(f"  Confidence: {playbook['confidence']:.0%}")
        logger.info(f"  Overnight scans: {playbook['scan_count']}")
        logger.info(f"  News accumulated: {playbook['total_news']}")

        if playbook["trades"]:
            logger.info(f"  Top opportunities:")
            for t in playbook["trades"][:5]:
                logger.info(
                    f"    {t['direction']:>5} {t['symbol']:<6} "
                    f"conviction={t['conviction']:.2f} "
                    f"sentiment={t['sentiment_score']:+.3f} "
                    f"| {'; '.join(t['reasons'][:2])}"
                )

        if playbook["catalysts"]:
            logger.info(f"  Overnight catalysts:")
            for c in playbook["catalysts"][-3:]:
                logger.info(f"    [{c['event_type']}] {c['title'][:70]}")

        # Feed overnight intelligence into orchestrator's news signals
        orchestrator.news_signals = premarket.get_news_signals()
        logger.info(f"  Injected {len(orchestrator.news_signals)} news signals from overnight analysis")

        # Save playbook
        playbook_dir = ROOT / "results" / "playbooks"
        playbook_dir.mkdir(parents=True, exist_ok=True)
        playbook_file = playbook_dir / f"playbook_{now.strftime('%Y%m%d')}.json"
        with open(playbook_file, "w") as f:
            json.dump(playbook, f, indent=2, default=str)

    # 3. Run the orchestrator (signals + ML + risk + execute)
    actions = orchestrator.run_daily(now, price_data, index_df)

    # 4. Log results
    if actions.get("trades"):
        logger.info(f"  Trades executed: {actions['trades']}")
    if actions.get("exits"):
        for exit_trade in actions["exits"]:
            logger.info(
                f"  Exit: {exit_trade['symbol']} "
                f"PnL=${exit_trade['pnl']:+,.2f} ({exit_trade['pnl_pct']:+.2%})"
            )
    if actions.get("retrained"):
        logger.info("  ML model retrained")

    logger.info(f"  Regime: {actions.get('regime', 'unknown')}")

    # 5. Reset pre-market analyzer for tonight
    premarket.reset()

    return actions


def run_overnight_scan(premarket: PreMarketAnalyzer) -> dict:
    """Run one overnight news scan cycle."""
    et = get_et_now()
    logger.info(f"Overnight scan #{premarket.scan_count + 1} at {et.strftime('%H:%M ET')}")

    result = premarket.scan()

    if result.get("new_items", 0) > 0:
        logger.info(
            f"  Found {result['new_items']} new items "
            f"({result['total_accumulated']} total, "
            f"{result['catalysts']} catalysts)"
        )

    return result


def run_intraday_monitor(orchestrator: TradingOrchestrator):
    """Monitor positions and react to breaking news during market hours."""
    if not orchestrator.open_positions:
        return

    symbols = list(orchestrator.open_positions.keys())
    quotes = get_live_quotes(symbols)

    for sym, pos in list(orchestrator.open_positions.items()):
        quote = quotes.get(sym)
        if not quote or not quote.get("price"):
            continue

        price = quote["price"]

        if price >= pos["tp_price"]:
            logger.warning(
                f"  TP HIT: {sym} @ ${price:.2f} "
                f"(target: ${pos['tp_price']:.2f}) — will close next cycle"
            )
        elif price <= pos["sl_price"]:
            logger.warning(
                f"  SL HIT: {sym} @ ${price:.2f} "
                f"(stop: ${pos['sl_price']:.2f}) — will close next cycle"
            )
        elif price <= pos["entry_price"] * 0.95:
            logger.warning(
                f"  WARNING: {sym} down 5%+ "
                f"(${pos['entry_price']:.2f} -> ${price:.2f})"
            )


def run_weekend_tasks(
    weekend_analyzer: WeekendAnalyzer,
    price_data: dict,
    orchestrator: TradingOrchestrator,
) -> dict:
    """Run weekend maintenance tasks."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  WEEKEND MAINTENANCE")
    logger.info(f"{'='*60}")

    results = weekend_analyzer.run_weekend_tasks(price_data, orchestrator)

    if results.get("retrain"):
        logger.info(f"  ML retrain: {results['retrain']}")
    if results.get("scanner", {}).get("added"):
        logger.info(f"  New symbols: {results['scanner']['added']}")
    if results.get("sector_rotation"):
        sr = results["sector_rotation"]
        up = [f"{k}({v['ret_5d']:+.1%})" for k, v in sr.items() if v["trend"] == "up"]
        down = [f"{k}({v['ret_5d']:+.1%})" for k, v in sr.items() if v["trend"] == "down"]
        if up:
            logger.info(f"  Sectors trending up: {', '.join(up[:5])}")
        if down:
            logger.info(f"  Sectors trending down: {', '.join(down[:5])}")

    return results


# ---------------------------------------------------------------------------
# Supabase persistence
# ---------------------------------------------------------------------------

def _persist_daily(
    repo,
    orchestrator: TradingOrchestrator,
    actions: dict,
    active_trade_ids: Dict[str, str],
) -> None:
    """Persist daily cycle results to Supabase for dashboard."""
    if repo is None:
        return

    try:
        ps = orchestrator.portfolio_state
        regime = actions.get("regime", "unknown")

        # 1. Portfolio snapshot
        drawdown = 1 - (ps.nav / ps.peak_nav) if ps.peak_nav > 0 else 0
        repo.save_portfolio_snapshot({
            "snapshot_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "total_value": round(ps.nav, 2),
            "cash_balance": round(ps.cash, 2),
            "unrealized_pnl": 0.0,
            "realized_pnl": sum(t.get("pnl", 0) for t in orchestrator.trade_log),
            "daily_pnl": round(ps.nav - ps.day_start_nav, 2),
            "gross_exposure": round(sum(
                abs(p.get("qty", 0) * p.get("entry_price", 0))
                for p in orchestrator.open_positions.values()
            ), 2),
            "net_exposure": round(sum(
                p.get("qty", 0) * p.get("entry_price", 0)
                for p in orchestrator.open_positions.values()
            ), 2),
            "open_positions": len(orchestrator.open_positions),
            "drawdown": round(drawdown, 4),
            "regime": regime,
        })

        # 2. Pipeline run
        n_signals = len(actions.get("signals", []))
        n_trades = len(actions.get("trades", []))
        scan_id = repo.start_pipeline_run(regime=regime)
        repo.complete_pipeline_run(scan_id, {
            "scanned": len(orchestrator.config.get("_active_symbols", [])) or 278,
            "surfaced": n_signals,
            "voted": n_signals,
            "approved": n_trades,
        })

        # 3. Record new trades
        for trade in actions.get("trades", []):
            sym = trade.get("symbol", "")
            if not sym:
                continue
            trade_id = repo.record_proposal(
                agent_id=trade.get("agent_id", "orchestrator"),
                symbol=sym,
                direction=trade.get("direction", "LONG"),
                confidence=trade.get("confidence", 0.5),
                strategy=trade.get("strategy", "swing"),
                reasoning=trade.get("reason", ""),
                hold_days=trade.get("hold_days", 10),
                sector=trade.get("sector", ""),
                regime=regime,
            )
            repo.record_execution(
                trade_id=trade_id,
                entry_price=trade.get("entry_price", 0),
                quantity=trade.get("qty", 0),
                fees=trade.get("fees", 0),
            )
            active_trade_ids[sym] = trade_id

        # 4. Record exits
        for exit_trade in actions.get("exits", []):
            sym = exit_trade.get("symbol", "")
            trade_id = active_trade_ids.pop(sym, "")
            if trade_id:
                repo.record_exit(
                    trade_id=trade_id,
                    exit_price=exit_trade.get("exit_price", 0),
                    actual_return=exit_trade.get("pnl_pct", 0),
                    pnl=exit_trade.get("pnl", 0),
                    actual_hold_days=exit_trade.get("days_held", 0),
                )

        # 5. Risk events
        if actions.get("kill_switch"):
            repo.record_risk_event(
                event_type="kill_switch",
                severity="critical",
                message="Kill switch activated",
                details={"regime": regime, "nav": ps.nav},
            )

        logger.info("Persisted daily results to Supabase")
    except Exception as e:
        logger.warning(f"Supabase persistence failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="24/7 Trading Bot — overnight analysis + live trading"
    )
    parser.add_argument(
        "--broker", choices=["paper", "ibkr"], default="paper",
        help="Broker to use (default: paper)",
    )
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "example.yaml"),
        help="Config YAML path",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one daily cycle and exit",
    )
    parser.add_argument(
        "--mode", choices=["auto", "overnight", "market", "weekend"],
        default="auto",
        help="Force a specific mode (default: auto-detect)",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip ML model warmup",
    )
    parser.add_argument(
        "--overnight-interval", type=int, default=30,
        help="Minutes between overnight news scans (default: 30)",
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=15,
        help="Minutes between intraday position checks (default: 15)",
    )
    args = parser.parse_args()

    # --- Config ---
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        config = {
            "portfolio": {"initial_nav": 100_000},
            "risk": {},
            "ml": {"min_oos_auc_to_deploy": 0.53, "entry_threshold": 0.20},
            "swing_signals": {"momentum_threshold_pct": 0.03},
            "sizing": {"holding_days": 10},
            "news": {"enabled": True},
        }

    # --- Load data ---
    logger.info("Loading historical data...")
    price_data = load_historical_data(config)
    if not price_data:
        logger.error("No data loaded. Run: python scripts/download_data.py")
        sys.exit(1)

    screened = screen_universe(price_data)
    active_symbols = [s["symbol"] for s in screened]
    if active_symbols:
        logger.info(f"Screener: {len(active_symbols)}/{len(price_data)} symbols active")
        price_data = {s: price_data[s] for s in active_symbols if s in price_data}

    all_symbols = list(price_data.keys())
    index_symbol = config.get("features", {}).get("index_symbol", "SPY")
    index_df = price_data.get(index_symbol, pd.DataFrame())

    # --- Broker + execution ---
    broker = create_broker(args.broker, price_data, config)

    risk_config = RiskConfig(**{
        k: v for k, v in config.get("risk", {}).items()
        if k in RiskConfig.__dataclass_fields__
    })
    risk_governor = RiskGovernor(risk_config)

    initial_nav = config.get("portfolio", {}).get("initial_nav", 100_000)
    allocs = config.get("portfolio", {}).get(
        "sleeve_allocations", {"core": 0.60, "swing": 0.30, "cash_buffer": 0.10}
    )
    portfolio_state = PortfolioState(
        nav=initial_nav, peak_nav=initial_nav,
        cash=initial_nav * (allocs.get("swing", 0.30) + allocs.get("cash_buffer", 0.10)),
        sleeve_values={
            "swing": initial_nav * allocs.get("swing", 0.30),
            "core": initial_nav * allocs.get("core", 0.60),
        },
        positions={}, day_start_nav=initial_nav,
    )

    exec_config = config.get("execution", {})
    order_manager = OrderManager(broker, risk_governor, portfolio_state, exec_config)

    orchestrator = TradingOrchestrator(config, str(ROOT), order_manager=order_manager)
    orchestrator.portfolio_state = portfolio_state

    # --- Supabase persistence ---
    repo = None
    try:
        from src.storage.supabase_client import get_supabase_client
        from src.storage.repository import TradingRepository
        sb = get_supabase_client()
        if sb:
            repo = TradingRepository(sb)
            logger.info("Supabase persistence enabled")
    except Exception as e:
        logger.warning(f"Supabase unavailable, running without persistence: {e}")

    # Track trade IDs for Supabase lifecycle (symbol -> trade_id)
    active_trade_ids: Dict[str, str] = {}

    # --- Warmup ---
    if not args.no_warmup:
        logger.info("Warming up models...")
        orchestrator.warm_up(price_data, index_df)

    # --- Analyzers ---
    premarket = PreMarketAnalyzer(all_symbols, config)
    weekend = WeekendAnalyzer(config)

    # --- Activity logger helper ---
    def log_activity(
        activity_type: str,
        summary: str,
        agent_id: str = "system",
        symbol: str = "",
        details: Optional[dict] = None,
    ) -> None:
        """Log an agent activity event to Supabase for the live feed."""
        if repo is None:
            return
        try:
            current_mode = args.mode if args.mode != "auto" else get_market_mode()
            repo.log_activity(
                agent_id=agent_id,
                activity_type=activity_type,
                summary=summary,
                symbol=symbol,
                details=details,
                market_mode=current_mode,
            )
        except Exception:
            pass  # Non-fatal

    # --- Banner ---
    mode = args.mode if args.mode != "auto" else get_market_mode()
    logger.info(f"\n{'='*60}")
    logger.info(f"  24/7 TRADING BOT STARTED")
    logger.info(f"  Broker:      {args.broker}")
    logger.info(f"  Universe:    {len(price_data)} stocks")
    logger.info(f"  NAV:         ${portfolio_state.nav:,.2f}")
    logger.info(f"  ML Model:    v{orchestrator.ml_trainer.model_version}")
    logger.info(f"  Current mode: {mode}")
    logger.info(f"  Overnight scan interval: {args.overnight_interval} min")
    logger.info(f"  Intraday monitor interval: {args.monitor_interval} min")
    logger.info(f"{'='*60}\n")

    log_activity(
        "regime",
        f"24/7 Trading Bot started — {len(price_data)} stocks, NAV ${portfolio_state.nav:,.0f}, mode: {mode}",
        details={"broker": args.broker, "universe_size": len(price_data), "ml_version": orchestrator.ml_trainer.model_version},
    )

    # --- State tracking ---
    last_daily_run = None
    last_weekend_run = None
    last_overnight_scan = None

    # --- Main loop ---
    while not _shutdown:
        if args.mode == "auto":
            mode = get_market_mode()
        else:
            mode = args.mode

        now = pd.Timestamp.now()
        today = now.normalize()
        et = get_et_now()

        # ============================================================
        # MARKET HOURS (9:25 AM - 4:05 PM ET weekdays)
        # ============================================================
        if mode == "market":
            # Run daily cycle once at market open
            if last_daily_run != today:
                log_activity("execution", "Market open — running daily trading cycle", details={"date": str(today)})
                actions = run_daily_cycle(orchestrator, price_data, index_df, premarket)
                _persist_daily(repo, orchestrator, actions, active_trade_ids)

                # Log trades and exits
                for trade in actions.get("trades", []):
                    sym = trade.get("symbol", "")
                    log_activity(
                        "execution",
                        f"Entered {trade.get('direction', 'long')} position in {sym} @ ${trade.get('price', 0):.2f}",
                        agent_id=trade.get("agent_id", "orchestrator"),
                        symbol=sym,
                        details=trade,
                    )
                for exit_t in actions.get("exits", []):
                    sym = exit_t.get("symbol", "")
                    log_activity(
                        "execution",
                        f"Exited {sym} — PnL ${exit_t.get('pnl', 0):+,.2f} ({exit_t.get('pnl_pct', 0):+.2%})",
                        symbol=sym,
                        details=exit_t,
                    )
                regime = actions.get("regime", "unknown")
                log_activity("regime", f"Daily cycle complete — regime: {regime}, {len(actions.get('trades', []))} trades, {len(actions.get('exits', []))} exits")

                last_daily_run = today
                if args.once:
                    break

            # Monitor positions
            if orchestrator.open_positions:
                log_activity("monitoring", f"Monitoring {len(orchestrator.open_positions)} open positions", details={"symbols": list(orchestrator.open_positions.keys())})
            run_intraday_monitor(orchestrator)
            time.sleep(args.monitor_interval * 60)

        # ============================================================
        # OVERNIGHT / PRE-MARKET (4 PM - 9:25 AM ET weekdays)
        # ============================================================
        elif mode in ("overnight", "premarket"):
            should_scan = (
                last_overnight_scan is None
                or (datetime.now() - last_overnight_scan).total_seconds()
                >= args.overnight_interval * 60
            )

            if should_scan:
                log_activity("news_scan", f"Scanning news and catalysts (scan #{premarket.scan_count + 1})")
                result = run_overnight_scan(premarket)
                last_overnight_scan = datetime.now()

                new_items = result.get("new_items", 0)
                if new_items > 0:
                    log_activity(
                        "analysis",
                        f"Found {new_items} new items — {result.get('total_accumulated', 0)} total accumulated, {result.get('catalysts', 0)} catalysts",
                        details=result,
                    )

            # Pre-market: show playbook preview close to open
            if mode == "premarket" and et.hour == 9 and et.minute >= 15:
                playbook = premarket.build_playbook(price_data=price_data)
                if playbook["trades"]:
                    logger.info(
                        f"\n  PRE-MARKET PLAYBOOK PREVIEW ({len(playbook['trades'])} trades ready)"
                    )
                    log_activity(
                        "playbook",
                        f"Morning playbook ready — {len(playbook['trades'])} trades, mood: {playbook.get('market_mood', '?')}, confidence: {playbook.get('confidence', 0):.0%}",
                        details={"trades": [t["symbol"] for t in playbook["trades"][:10]]},
                    )
                    for t in playbook["trades"][:5]:
                        logger.info(
                            f"    {t['direction']:>5} {t['symbol']:<6} "
                            f"conviction={t['conviction']:.2f}"
                        )
                        log_activity(
                            "thesis",
                            f"Pre-market thesis: {t['direction']} {t['symbol']} — conviction {t['conviction']:.2f}, sentiment {t.get('sentiment_score', 0):+.3f}",
                            symbol=t["symbol"],
                            details=t,
                        )

            # Sleep until next scan
            time.sleep(min(args.overnight_interval * 60, 300))

        # ============================================================
        # WEEKEND (Sat-Sun)
        # ============================================================
        elif mode == "weekend":
            this_week = today - timedelta(days=today.weekday())

            # Run weekend tasks once per weekend
            if last_weekend_run is None or last_weekend_run < this_week:
                log_activity("retraining", "Starting weekend maintenance — ML retraining, universe scan, sector rotation analysis")
                results = run_weekend_tasks(weekend, price_data, orchestrator)
                last_weekend_run = today

                if results.get("retrain"):
                    log_activity("retraining", f"ML model retrained: {results['retrain']}", details=results.get("retrain"))
                if results.get("scanner", {}).get("added"):
                    log_activity("analysis", f"New symbols added to universe: {', '.join(results['scanner']['added'])}", details=results["scanner"])
                if results.get("sector_rotation"):
                    sr = results["sector_rotation"]
                    up = [k for k, v in sr.items() if v.get("trend") == "up"]
                    down = [k for k, v in sr.items() if v.get("trend") == "down"]
                    log_activity("analysis", f"Sector rotation: {len(up)} sectors trending up, {len(down)} trending down", details={"up": up[:5], "down": down[:5]})

            # Still scan news on weekends (less frequently)
            should_scan = (
                last_overnight_scan is None
                or (datetime.now() - last_overnight_scan).total_seconds() >= 7200
            )
            if should_scan:
                log_activity("news_scan", "Weekend news scan — monitoring for breaking stories")
                result = run_overnight_scan(premarket)
                last_overnight_scan = datetime.now()
                new_items = result.get("new_items", 0)
                if new_items > 0:
                    log_activity("analysis", f"Weekend scan found {new_items} new items", details=result)

            # Sleep longer on weekends
            time.sleep(1800)  # 30 min

    # --- Shutdown ---
    logger.info("Shutting down...")
    summary = orchestrator.get_performance_summary()

    print(f"\n{'='*60}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Final NAV:      ${orchestrator.portfolio_state.nav:,.2f}")
    print(f"  Total Trades:   {summary.get('n_trades', 0)}")
    print(f"  Win Rate:       {summary.get('win_rate', 0):.1%}")
    print(f"  Open Positions: {len(orchestrator.open_positions)}")
    print(f"  Overnight Scans: {premarket.scan_count}")
    print(f"{'='*60}")

    # Save results
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    if orchestrator.daily_nav:
        pd.DataFrame(orchestrator.daily_nav).to_csv(
            results_dir / "live_nav.csv", index=False
        )
    if orchestrator.trade_log:
        pd.DataFrame(orchestrator.trade_log).to_csv(
            results_dir / "live_trades.csv", index=False
        )
    if order_manager.order_log:
        pd.DataFrame(order_manager.order_log).to_csv(
            results_dir / "live_orders.csv", index=False
        )

    logger.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()

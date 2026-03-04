#!/usr/bin/env python3
"""
Live Trading Runner — real-time daily loop with broker integration.

This is the production entry point. It:
  1. Connects to broker (IBKR or Paper)
  2. Loads historical data + fetches today's bars
  3. Runs the full orchestrator pipeline once per day
  4. Monitors positions intraday for barrier exits
  5. Sleeps until next market open
  6. Repeats forever

Usage:
    python scripts/run_live.py --broker paper
    python scripts/run_live.py --broker ibkr
    python scripts/run_live.py --broker paper --once  # run once and exit
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.provider import CSVDataProvider
from src.data.missing import handle_missing_data
from src.data.screener import screen_universe
from src.data.live_prices import (
    update_price_data,
    get_live_quotes,
    is_market_open,
    time_until_market_open,
)
from src.utils.config_loader import load_config
from src.risk.risk_governor import RiskGovernor, RiskConfig, PortfolioState
from src.backtest.cost_model import CostModel
from src.execution.paper_broker import PaperBroker
from src.execution.order_manager import OrderManager, GracefulShutdown
from src.core.orchestrator import TradingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global shutdown flag
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.info(f"Shutdown signal received ({signum}). Finishing current cycle...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


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
            from src.execution.ibkr_broker import IBKRBroker
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


def run_daily_cycle(
    orchestrator: TradingOrchestrator,
    price_data: dict,
    index_df: pd.DataFrame,
    current_date: pd.Timestamp,
) -> dict:
    """Execute one daily trading cycle."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  DAILY CYCLE: {current_date.date()}")
    logger.info(f"  NAV: ${orchestrator.portfolio_state.nav:,.2f}")
    logger.info(f"  Open positions: {len(orchestrator.open_positions)}")
    logger.info(f"{'='*50}")

    # Update data with latest bars
    price_data = update_price_data(price_data)

    # Update index
    index_symbol = orchestrator.config.get("features", {}).get("index_symbol", "SPY")
    index_df = price_data.get(index_symbol, index_df)

    # Run the orchestrator
    actions = orchestrator.run_daily(current_date, price_data, index_df)

    # Log results
    if actions.get("trades"):
        logger.info(f"  Trades: {actions['trades']}")
    if actions.get("exits"):
        for exit_trade in actions["exits"]:
            logger.info(
                f"  Exit: {exit_trade['symbol']} "
                f"PnL=${exit_trade['pnl']:+,.2f} ({exit_trade['pnl_pct']:+.2%})"
            )
    if actions.get("retrained"):
        logger.info("  ML model retrained")

    logger.info(f"  Regime: {actions.get('regime', 'unknown')}")
    logger.info(f"  News signals: {actions.get('news_signals', 0)}")

    return actions


def monitor_positions(
    orchestrator: TradingOrchestrator,
    check_interval_minutes: int = 15,
):
    """
    Intraday position monitoring — checks barriers between daily runs.
    Only runs during market hours.
    """
    if not orchestrator.open_positions:
        return

    symbols = list(orchestrator.open_positions.keys())
    quotes = get_live_quotes(symbols)

    for sym, pos in list(orchestrator.open_positions.items()):
        quote = quotes.get(sym)
        if not quote or not quote.get("price"):
            continue

        price = quote["price"]
        alert = None

        if price >= pos["tp_price"]:
            alert = f"TP HIT: {sym} @ ${price:.2f} (target: ${pos['tp_price']:.2f})"
        elif price <= pos["sl_price"]:
            alert = f"SL HIT: {sym} @ ${price:.2f} (stop: ${pos['sl_price']:.2f})"
        elif price <= pos["entry_price"] * 0.95:
            alert = f"WARNING: {sym} down 5%+ from entry (${pos['entry_price']:.2f} → ${price:.2f})"

        if alert:
            logger.warning(f"  POSITION ALERT: {alert}")


def main():
    parser = argparse.ArgumentParser(description="Live Trading Runner")
    parser.add_argument(
        "--broker",
        choices=["paper", "ibkr"],
        default="paper",
        help="Broker to use (default: paper)",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "example.yaml"),
        help="Config YAML path",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (don't loop)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip ML model warmup (use existing model)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=15,
        help="Minutes between intraday position checks (default: 15)",
    )
    args = parser.parse_args()

    # Load config
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

    # Load historical data
    logger.info("Loading historical data...")
    price_data = load_historical_data(config)
    if not price_data:
        logger.error("No data loaded. Run: python scripts/download_data.py")
        sys.exit(1)

    # Screen universe
    screened = screen_universe(price_data)
    active_symbols = [s["symbol"] for s in screened]
    if active_symbols:
        logger.info(f"Screener: {len(active_symbols)}/{len(price_data)} symbols active")
        price_data = {s: price_data[s] for s in active_symbols if s in price_data}

    # Index
    index_symbol = config.get("features", {}).get("index_symbol", "SPY")
    index_df = price_data.get(index_symbol, pd.DataFrame())

    # Create broker
    broker = create_broker(args.broker, price_data, config)

    # Setup execution pipeline
    risk_config = RiskConfig(**{
        k: v for k, v in config.get("risk", {}).items()
        if k in RiskConfig.__dataclass_fields__
    })
    risk_governor = RiskGovernor(risk_config)

    # Portfolio state
    initial_nav = config.get("portfolio", {}).get("initial_nav", 100_000)
    allocs = config.get("portfolio", {}).get(
        "sleeve_allocations", {"core": 0.60, "swing": 0.30, "cash_buffer": 0.10}
    )
    portfolio_state = PortfolioState(
        nav=initial_nav,
        peak_nav=initial_nav,
        cash=initial_nav * (allocs.get("swing", 0.30) + allocs.get("cash_buffer", 0.10)),
        sleeve_values={
            "swing": initial_nav * allocs.get("swing", 0.30),
            "core": initial_nav * allocs.get("core", 0.60),
        },
        positions={},
        day_start_nav=initial_nav,
    )

    exec_config = config.get("execution", {})
    order_manager = OrderManager(broker, risk_governor, portfolio_state, exec_config)

    # Create orchestrator WITH order_manager
    orchestrator = TradingOrchestrator(config, str(ROOT), order_manager=order_manager)
    # Share portfolio state
    orchestrator.portfolio_state = portfolio_state

    # Warmup
    if not args.no_warmup:
        logger.info("Warming up models...")
        orchestrator.warm_up(price_data, index_df)

    logger.info(f"\n{'='*60}")
    logger.info(f"  LIVE TRADING BOT STARTED")
    logger.info(f"  Broker: {args.broker}")
    logger.info(f"  Universe: {len(price_data)} stocks")
    logger.info(f"  NAV: ${portfolio_state.nav:,.2f}")
    logger.info(f"  ML Model: v{orchestrator.ml_trainer.model_version}")
    logger.info(f"{'='*60}\n")

    # Main loop
    last_daily_run = None

    while not _shutdown:
        now = pd.Timestamp.now()
        today = now.normalize()

        # Run daily cycle once per day (at market open or on first run)
        if last_daily_run != today:
            if is_market_open() or args.once:
                run_daily_cycle(orchestrator, price_data, index_df, now)
                last_daily_run = today

                if args.once:
                    break

        # Intraday position monitoring
        if is_market_open() and orchestrator.open_positions:
            monitor_positions(orchestrator, args.monitor_interval)

        # Sleep logic
        if not is_market_open():
            wait = time_until_market_open()
            if wait:
                hours = wait.total_seconds() / 3600
                logger.info(f"Market closed. Next open in {hours:.1f} hours. Sleeping...")
                # Sleep in chunks so we can respond to shutdown signals
                sleep_seconds = min(wait.total_seconds(), 3600)  # Max 1 hour chunks
                time.sleep(sleep_seconds)
            else:
                time.sleep(60)
        else:
            # During market hours, check every N minutes
            time.sleep(args.monitor_interval * 60)

    # Clean shutdown
    logger.info("Shutting down...")
    summary = orchestrator.get_performance_summary()

    print(f"\n{'='*60}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Final NAV:      ${orchestrator.portfolio_state.nav:,.2f}")
    print(f"  Total Trades:   {summary.get('n_trades', 0)}")
    print(f"  Win Rate:       {summary.get('win_rate', 0):.1%}")
    print(f"  Open Positions: {len(orchestrator.open_positions)}")
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

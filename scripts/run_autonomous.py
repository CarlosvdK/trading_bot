#!/usr/bin/env python3
"""
Autonomous Trading Bot — self-sufficient daily loop.

This is the "set it and forget it" entry point. It:
  1. Loads/downloads data
  2. Trains ML model if needed (walk-forward, calibrated)
  3. Detects market regime
  4. Generates signals, filters with ML, sizes positions
  5. Manages exits (TP/SL/timeout)
  6. Monitors drift, retrains when needed
  7. Adapts parameters from trade feedback
  8. Logs everything to audit trail

Usage:
    python scripts/run_autonomous.py
    python scripts/run_autonomous.py --config config/example.yaml
    python scripts/run_autonomous.py --start 2018-01-01 --end 2026-01-01
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.provider import CSVDataProvider
from src.data.missing import handle_missing_data
from src.data.screener import screen_universe
from src.utils.config_loader import load_config
from src.core.orchestrator import TradingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_all_data(config: dict, start_date: str, end_date: str) -> dict:
    """Load all available OHLCV data."""
    data_dir = config.get("data", {}).get(
        "data_dir", str(ROOT / "data" / "ohlcv")
    )
    universe_file = config.get("data", {}).get(
        "universe_file", str(ROOT / "data" / "universe.csv")
    )

    provider = CSVDataProvider(
        data_dir=data_dir,
        universe_file=universe_file,
        validate_on_load=True,
    )

    symbols = provider.available_symbols()
    logger.info(f"Found {len(symbols)} data files")

    price_data = {}
    for sym in symbols:
        try:
            df = provider.load_symbol(sym, start_date, end_date)
            cleaned = handle_missing_data(df, max_missing_pct=0.05)
            if cleaned is not None and len(cleaned) > 60:
                price_data[sym] = cleaned
        except Exception as e:
            logger.warning(f"{sym}: {e}")

    logger.info(f"Loaded {len(price_data)} symbols")
    return price_data


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Bot — self-learning daily loop"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "example.yaml"),
        help="Config YAML path",
    )
    parser.add_argument("--start", default="2018-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    # Load config
    config_path = args.config
    if Path(config_path).exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {
            "portfolio": {"initial_nav": 100_000},
            "risk": {},
            "ml": {"min_oos_auc_to_deploy": 0.53, "entry_threshold": 0.20},
            "swing_signals": {"momentum_threshold_pct": 0.03},
            "sizing": {"holding_days": 10},
            "walk_forward": {
                "initial_train_days": 504,
                "test_days": 126,
                "step_days": 63,
            },
            "labeling": {"k1": 1.5, "k2": 1.5, "horizon_days": 10},
        }

    # Load data
    price_data = load_all_data(config, args.start, end_date)
    if not price_data:
        logger.error("No data loaded. Run: python scripts/download_data.py")
        sys.exit(1)

    # Dynamic universe screening
    screened = screen_universe(price_data)
    active_symbols = [s["symbol"] for s in screened]
    if active_symbols:
        logger.info(f"Active universe after screening: {active_symbols}")
        price_data = {s: price_data[s] for s in active_symbols if s in price_data}
    else:
        logger.warning("No symbols passed screening — using all available")

    # Index
    index_symbol = config.get("features", {}).get("index_symbol", "SPY")
    index_df = price_data.get(index_symbol, pd.DataFrame())

    # Initialize orchestrator
    orchestrator = TradingOrchestrator(config, str(ROOT))

    # Warm up: train ML model + regime model from historical data
    logger.info("Warming up: training models from historical data...")
    orchestrator.warm_up(price_data, index_df)

    # Build date range
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    date_range = sorted(all_dates)
    date_range = [d for d in date_range if d >= pd.Timestamp(args.start)]
    if end_date:
        date_range = [d for d in date_range if d <= pd.Timestamp(end_date)]

    logger.info(
        f"\n{'='*60}\n"
        f"  AUTONOMOUS BOT STARTING\n"
        f"  Period: {date_range[0].date()} → {date_range[-1].date()}\n"
        f"  Symbols: {len(price_data)}\n"
        f"  Trading days: {len(date_range)}\n"
        f"{'='*60}"
    )

    # Main loop
    for i, date in enumerate(date_range):
        actions = orchestrator.run_daily(date, price_data, index_df)

        # Progress log every 100 days
        if i > 0 and i % 100 == 0:
            summary = orchestrator.get_performance_summary()
            logger.info(
                f"Day {i}/{len(date_range)} | "
                f"NAV=${orchestrator.portfolio_state.nav:,.0f} | "
                f"Trades={summary.get('n_trades', 0)} | "
                f"WR={summary.get('win_rate', 0):.0%}"
            )

    # Final report
    summary = orchestrator.get_performance_summary()
    recs = orchestrator.get_adaptive_recommendations()

    print(f"\n{'='*60}")
    print(f"  AUTONOMOUS BOT RESULTS")
    print(f"{'='*60}")
    print(f"  Period:         {date_range[0].date()} → {date_range[-1].date()}")
    print(f"  Final NAV:      ${orchestrator.portfolio_state.nav:,.2f}")
    print(f"  Total Trades:   {summary.get('n_trades', 0)}")
    print(f"  Win Rate:       {summary.get('win_rate', 0):.1%}")
    print(f"  Avg Win:        {summary.get('avg_win', 0):+.2%}")
    print(f"  Avg Loss:       {summary.get('avg_loss', 0):+.2%}")
    print(f"  Total PnL:      ${summary.get('total_pnl', 0):+,.2f}")
    print(f"  Avg Hold:       {summary.get('avg_days_held', 0):.1f} days")

    ml_version = orchestrator.ml_trainer.model_version
    print(f"  ML Model:       v{ml_version}")
    if orchestrator.ml_trainer.current_meta:
        print(f"  ML AUC:         {orchestrator.ml_trainer.current_meta.get('oos_roc_auc', 'N/A')}")

    if summary.get("by_regime"):
        print(f"\n  Performance by Regime:")
        for regime, stats in summary["by_regime"].items():
            print(
                f"    {regime:30s} | "
                f"trades={stats['n_trades']:3d} | "
                f"WR={stats['win_rate']:.0%} | "
                f"avg={stats['avg_return']:+.2%}"
            )

    if recs and recs.get("status") != "insufficient_trades":
        print(f"\n  Adaptive Recommendations:")
        for key, rec in recs.items():
            print(f"    {key}: {rec.get('reason', '')} → {rec.get('suggested', '')}")

    print(f"{'='*60}")

    # Save results
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if orchestrator.daily_nav:
        nav_df = pd.DataFrame(orchestrator.daily_nav)
        nav_df.to_csv(results_dir / "autonomous_nav.csv", index=False)

    if orchestrator.trade_log:
        trades_df = pd.DataFrame(orchestrator.trade_log)
        trades_df.to_csv(results_dir / "autonomous_trades.csv", index=False)

    logger.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-Agent Voting System — daily scan entry point.

Runs the full 4-stage pipeline:
  Stage A: 121 agents independently scan for opportunities
  Stage B: Specialist subgroup review
  Stage C: Global weighted vote with adaptive thresholds
  Stage D: Portfolio/risk layer

Then selects optimal investment vehicle for each approved trade.

Usage:
    python scripts/run_agent_scan.py
    python scripts/run_agent_scan.py --data-dir data/ --config config/example.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.agent_definitions import ALL_AGENTS
from src.agents.trading_agent import TradingAgent
from src.agents.pipeline import TradingPipeline
from src.agents.vehicle_engine import VehicleSelectionEngine
from src.agents.regime_adapter import RegimeAdapter
from src.agents.enhanced_scoring import EnhancedScoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_scan")


def load_universe_data(data_dir: Path) -> dict:
    """Load OHLCV CSVs for the universe."""
    universe = {}
    csv_dir = data_dir / "ohlcv"
    if not csv_dir.exists():
        csv_dir = data_dir
    for csv_path in sorted(csv_dir.glob("*.csv")):
        symbol = csv_path.stem.upper()
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            df.columns = [c.lower() for c in df.columns]
            if all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
                universe[symbol] = df
        except Exception as e:
            logger.warning(f"Failed to load {csv_path}: {e}")
    return universe


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent voting scan")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory with OHLCV CSVs")
    parser.add_argument("--index-symbol", type=str, default="SPY",
                        help="Index symbol for regime detection")
    parser.add_argument("--state-path", type=str, default="state/agent_scores.json",
                        help="Path to save/load agent scores")
    parser.add_argument("--options", action="store_true",
                        help="Enable options vehicle selection")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir

    # Load market data
    logger.info("Loading universe data...")
    universe_data = load_universe_data(data_dir)
    if not universe_data:
        logger.error(f"No OHLCV data found in {data_dir}")
        sys.exit(1)
    logger.info(f"Loaded {len(universe_data)} symbols")

    # Index data for regime detection
    index_df = universe_data.get(args.index_symbol)

    # Build all 121 agents
    logger.info(f"Building {len(ALL_AGENTS)} agents...")
    agents = [TradingAgent(dna) for dna in ALL_AGENTS]

    # Build pipeline
    scoring = EnhancedScoring()
    vehicle_engine = VehicleSelectionEngine(options_available=args.options)
    regime_adapter = RegimeAdapter()

    pipeline = TradingPipeline(
        agents=agents,
        scoring=scoring,
        vehicle_engine=vehicle_engine,
        regime_adapter=regime_adapter,
    )

    # Load previous scores if available
    state_path = ROOT / args.state_path
    if state_path.exists():
        try:
            scoring.load(str(state_path)) if hasattr(scoring, 'load') else None
            logger.info("Loaded previous agent scores")
        except Exception:
            pass

    # Run the full pipeline
    logger.info("Running 4-stage pipeline...")
    decisions = pipeline.run_daily(
        universe_data=universe_data,
        index_df=index_df,
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"  MULTI-AGENT SCAN RESULTS — {len(decisions)} APPROVED TRADES")
    print(f"  Regime: {regime_adapter.current_regime}")
    print("=" * 80)

    if not decisions:
        print("\n  No trades approved today.\n")
    else:
        for i, d in enumerate(decisions, 1):
            print(f"\n  [{i}] {d.summary()}")
            print(f"      Thesis: {d.thesis[:100]}")
            print(f"      Vehicle: {d.selected_vehicle} — {d.vehicle_rationale[:80]}")
            print(f"      Entry: {d.entry_logic}")
            print(f"      Stop: {d.stop_logic}")
            print(f"      Target: {d.target_logic}")
            print(f"      Risks: {', '.join(d.key_risks[:3]) if d.key_risks else 'None flagged'}")
            print(f"      Dissent: {d.dissent_summary[:100]}")
            if d.alternative_vehicles:
                alts = [f"{a['type']}({a['score']:.3f})" for a in d.alternative_vehicles[:3]]
                print(f"      Alternatives: {', '.join(alts)}")

    # Print leaderboard
    lb = pipeline.get_leaderboard()
    if not lb.empty:
        print(f"\n{'=' * 80}")
        print("  AGENT LEADERBOARD (top 10)")
        print("=" * 80)
        print(lb.head(10).to_string(index=False))

    print()


if __name__ == "__main__":
    main()

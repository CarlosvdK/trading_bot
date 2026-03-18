#!/usr/bin/env python3
"""
Paper trading runner — daily loop using Order Manager + Paper Broker.
Identical code path to live trading, with PaperBroker instead of LiveBroker.

Usage:
    python scripts/run_paper_trading.py
    python scripts/run_paper_trading.py --config config/example.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_feeds.provider import CSVDataProvider
from src.data_feeds.missing import handle_missing_data
from src.utilities.config_loader import load_config
from src.risk_management.risk_governor import RiskGovernor, RiskConfig, PortfolioState
from src.backtesting.cost_model import CostModel
from src.trading.order_types import Order, OrderType, OrderSide
from src.trading.paper_broker import PaperBroker
from src.trading.order_manager import OrderManager, GracefulShutdown
from src.models.labeler import compute_vol_proxy
from src.models.regime import (
    build_regime_features,
    fit_regime_model,
    predict_regime,
    label_regimes,
    get_regime_allocation,
    smooth_regime,
)
from src.signals.signals import generate_swing_signals
from src.signals.sizing import compute_swing_position_size, compute_barriers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_paper_trading(config_path: str):
    """Run paper trading simulation over available historical data."""

    # --- Load config ---
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = {
            "portfolio": {"initial_nav": 100_000},
            "risk": {},
            "swing_signals": {"momentum_threshold_pct": 0.04},
            "sizing": {"holding_days": 10},
            "execution": {},
        }

    # --- Load data ---
    data_dir = config.get("data", {}).get(
        "data_dir", str(ROOT / "data" / "ohlcv")
    )
    universe_file = config.get("data", {}).get(
        "universe_file", str(ROOT / "data" / "universe.csv")
    )

    provider = CSVDataProvider(data_dir=data_dir, universe_file=universe_file)
    symbols = provider.available_symbols()
    if not symbols:
        logger.error("No data files found. Run scripts/download_data.py first.")
        sys.exit(1)

    price_data = {}
    for sym in symbols:
        try:
            df = provider.load_symbol(sym)
            cleaned = handle_missing_data(df)
            if cleaned is not None and len(cleaned) > 60:
                price_data[sym] = cleaned
        except Exception as e:
            logger.warning(f"{sym}: {e}")

    logger.info(f"Loaded {len(price_data)} symbols")

    index_symbol = config.get("features", {}).get("index_symbol", "SPY")
    index_df = price_data.get(index_symbol, pd.DataFrame())

    # --- Setup components ---
    initial_nav = config.get("portfolio", {}).get("initial_nav", 100_000)
    risk_config = RiskConfig()
    risk_governor = RiskGovernor(risk_config)
    cost_model = CostModel()
    broker = PaperBroker(price_data, cost_model, config)

    state = PortfolioState(
        nav=initial_nav,
        peak_nav=initial_nav,
        cash=initial_nav * 0.10,
        sleeve_values={"swing": initial_nav * 0.30, "core": initial_nav * 0.60},
        positions={},
        day_start_nav=initial_nav,
    )

    exec_config = config.get("execution", {})
    order_manager = OrderManager(broker, risk_governor, state, exec_config)
    shutdown = GracefulShutdown(order_manager, state)

    # --- Regime ---
    regime_series = None
    regime_names = None
    if len(index_df) > 200:
        feats = build_regime_features(index_df["close"], config)
        if len(feats) > 100:
            model_dict = fit_regime_model(feats, n_regimes=4, method="kmeans")
            raw = predict_regime(model_dict, feats)
            regime_series = smooth_regime(raw, min_persistence=3)
            regime_names = label_regimes(feats, raw)

    # --- Build date range ---
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    date_range = sorted(all_dates)

    swing_config = config.get("swing_signals", {})
    sizing_config = config.get("sizing", {})
    open_positions = {}
    order_counter = 0
    nav_history = []

    logger.info(
        f"Starting paper trading: {date_range[0].date()} → "
        f"{date_range[-1].date()} ({len(date_range)} days)"
    )

    # --- Main loop ---
    for date in date_range:
        if shutdown.should_shutdown:
            break

        # Prices for today
        prices = {}
        for sym, df in price_data.items():
            if date in df.index:
                prices[sym] = df.loc[date, "close"]
        if not prices:
            continue

        # Update state NAV
        state.nav = sum(
            pos.get("qty", 0) * prices.get(sym, 0)
            for sym, pos in open_positions.items()
        ) + state.cash + state.sleeve_values.get("core", 0)
        state.peak_nav = max(state.peak_nav, state.nav)

        # Regime check
        swing_enabled = True
        regime_name = "unknown"
        if regime_series is not None and date in regime_series.index:
            regime_id = regime_series.loc[date]
            regime_name = (regime_names or {}).get(regime_id, "unknown")
            alloc = get_regime_allocation(regime_name)
            swing_enabled = alloc["swing_enabled"]

        if not swing_enabled:
            nav_history.append({"date": date, "nav": state.nav, "regime": regime_name})
            continue

        # Generate signals
        candidates = generate_swing_signals(
            price_data, index_df, date, swing_config
        )

        # Process candidates
        for cand in candidates:
            sym = cand["symbol"]
            if sym in open_positions or sym not in prices:
                continue

            vol_df = price_data.get(sym)
            if vol_df is None:
                continue
            vol_proxy = compute_vol_proxy(vol_df["close"])
            if date not in vol_proxy.index:
                continue
            inst_vol = vol_proxy.loc[date]
            if np.isnan(inst_vol) or inst_vol <= 0:
                continue

            swing_nav = state.sleeve_values.get("swing", initial_nav * 0.3)
            result = compute_swing_position_size(
                symbol=sym,
                sleeve_nav=swing_nav,
                instrument_vol=inst_vol,
                ml_prob=0.65,
                current_regime=regime_name,
                vvol_percentile=0.5,
                price=prices[sym],
                config=sizing_config,
            )

            if result["shares"] > 0:
                barriers = compute_barriers(
                    prices[sym], inst_vol,
                    sizing_config.get("holding_days", 10),
                )
                order_counter += 1
                order = Order(
                    order_id=f"paper-{order_counter:05d}",
                    symbol=sym,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    qty=result["shares"],
                    sleeve="swing",
                    tp_price=barriers["tp_price"],
                    sl_price=barriers["sl_price"],
                    ml_prob=0.65,
                )
                fill = order_manager.submit(order, date)
                if fill:
                    open_positions[sym] = {
                        "qty": fill.filled_qty,
                        "entry_price": fill.fill_price,
                        "entry_date": date,
                        "tp": barriers["tp_price"],
                        "sl": barriers["sl_price"],
                    }

        # Check barriers on open positions
        for sym in list(open_positions.keys()):
            if sym not in prices:
                continue
            pos = open_positions[sym]
            price = prices[sym]
            should_close = False
            if price >= pos["tp"]:
                should_close = True
            elif price <= pos["sl"]:
                should_close = True
            elif (date - pos["entry_date"]).days > sizing_config.get(
                "holding_days", 10
            ) * 1.5:
                should_close = True

            if should_close:
                order_counter += 1
                sell_order = Order(
                    order_id=f"paper-{order_counter:05d}",
                    symbol=sym,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    qty=pos["qty"],
                    sleeve="swing",
                )
                fill = order_manager.submit(sell_order, date)
                if fill:
                    pnl = (fill.fill_price - pos["entry_price"]) * pos["qty"]
                    logger.info(
                        f"  Closed {sym}: PnL=${pnl:+,.2f} "
                        f"({fill.fill_price/pos['entry_price']-1:+.2%})"
                    )
                    del open_positions[sym]

        nav_history.append({
            "date": date, "nav": state.nav, "regime": regime_name,
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  PAPER TRADING SUMMARY")
    print(f"{'='*60}")
    print(f"  Orders submitted:  {len(order_manager.order_log)}")
    filled = sum(
        1 for o in order_manager.order_log if o["status"] == "FILLED"
    )
    rejected = sum(
        1 for o in order_manager.order_log if o["status"] == "REJECTED"
    )
    print(f"  Filled:            {filled}")
    print(f"  Rejected:          {rejected}")
    print(f"  Open positions:    {len(open_positions)}")

    if nav_history:
        nav_df = pd.DataFrame(nav_history).set_index("date")
        nav_df.to_csv(ROOT / "results" / "paper_nav_history.csv")
        print(f"  Final NAV:         ${nav_history[-1]['nav']:,.2f}")

    # Save order log
    if order_manager.order_log:
        log_df = pd.DataFrame(order_manager.order_log)
        log_df.to_csv(ROOT / "results" / "paper_order_log.csv", index=False)

    print(f"  Results saved to:  {ROOT / 'results'}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "example.yaml"),
        help="Config YAML path",
    )
    args = parser.parse_args()
    run_paper_trading(args.config)


if __name__ == "__main__":
    main()

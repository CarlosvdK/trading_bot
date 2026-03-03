#!/usr/bin/env python3
"""
End-to-end backtest runner.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config config/example.yaml --start 2018-01-01
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.provider import CSVDataProvider
from src.data.corporate_actions import apply_corporate_actions
from src.data.missing import handle_missing_data
from src.utils.config_loader import load_config
from src.risk.risk_governor import RiskGovernor, RiskConfig, PortfolioState
from src.backtest.cost_model import CostModel
from src.backtest.engine import Backtester
from src.ml.features import build_features
from src.ml.labeler import build_labels, compute_vol_proxy
from src.ml.validation import walk_forward_splits
from src.ml.regime import (
    build_regime_features,
    fit_regime_model,
    predict_regime,
    label_regimes,
    get_regime_allocation,
    smooth_regime,
)
from src.swing.signals import generate_swing_signals
from src.swing.sizing import compute_swing_position_size, compute_barriers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_price_data(config: dict, start_date: str, end_date: str) -> dict:
    """Load and validate all symbol data."""
    data_dir = config.get("data", {}).get("data_dir", str(ROOT / "data" / "ohlcv"))
    universe_file = config.get("data", {}).get(
        "universe_file", str(ROOT / "data" / "universe.csv")
    )
    corp_actions_file = config.get("data", {}).get(
        "corporate_actions_file", str(ROOT / "data" / "corporate_actions.csv")
    )

    provider = CSVDataProvider(
        data_dir=data_dir,
        universe_file=universe_file,
        validate_on_load=True,
    )

    # Load corporate actions
    corp_actions = None
    if Path(corp_actions_file).exists():
        corp_actions = pd.read_csv(corp_actions_file, parse_dates=["date"])

    symbols = provider.get_universe(start_date)
    logger.info(f"Universe: {len(symbols)} symbols — {symbols}")

    price_data = {}
    for sym in symbols:
        try:
            df = provider.load_symbol(sym, start_date, end_date)
            if corp_actions is not None:
                df = apply_corporate_actions(df, sym, corp_actions)
            cleaned = handle_missing_data(df, max_missing_pct=0.05)
            if cleaned is not None and len(cleaned) > 60:
                price_data[sym] = cleaned
                logger.info(f"  {sym}: {len(cleaned)} bars loaded")
            else:
                logger.warning(f"  {sym}: insufficient data, skipped")
        except Exception as e:
            logger.warning(f"  {sym}: failed to load — {e}")

    return price_data


def build_signal_func(
    price_data: dict,
    index_df: pd.DataFrame,
    config: dict,
    risk_governor: RiskGovernor,
    regime_series: pd.Series = None,
    regime_names: dict = None,
    ml_model=None,
):
    """
    Returns a signal function compatible with Backtester.run().
    Signature: signal_func(date, prices, sleeves) -> List[dict]
    """
    swing_config = config.get("swing_signals", {})
    sizing_config = config.get("sizing", {})
    risk_config = config.get("risk", {})

    def signal_func(date, prices, sleeves):
        orders = []

        # --- Regime check ---
        swing_enabled = True
        regime_name = "unknown"
        if regime_series is not None and date in regime_series.index:
            regime_id = regime_series.loc[date]
            regime_name = (regime_names or {}).get(regime_id, "unknown")
            alloc = get_regime_allocation(regime_name)
            swing_enabled = alloc["swing_enabled"]

        if not swing_enabled:
            return orders

        # --- Generate swing signals ---
        candidates = generate_swing_signals(
            price_data, index_df, date, swing_config
        )

        # --- Size and create orders ---
        swing_sleeve = sleeves["swing"]
        swing_nav = swing_sleeve.mark_to_market(prices)

        for cand in candidates:
            sym = cand["symbol"]
            if sym in swing_sleeve.positions:
                continue  # Already holding
            if sym not in prices:
                continue

            price = prices[sym]
            vol_df = price_data.get(sym)
            if vol_df is None:
                continue

            vol_proxy = compute_vol_proxy(vol_df["close"])
            if date not in vol_proxy.index:
                continue
            inst_vol = vol_proxy.loc[date]
            if np.isnan(inst_vol) or inst_vol <= 0:
                continue

            # ML probability (use 0.65 default if no model)
            ml_prob = 0.65
            if ml_model is not None:
                try:
                    feats = build_features(
                        vol_df.loc[:date], index_df.loc[:date], config
                    )
                    if len(feats) > 0 and date in feats.index:
                        X = feats.loc[[date]].fillna(0)
                        ml_prob = float(ml_model.predict_proba(X)[:, 1][0])
                except Exception:
                    ml_prob = 0.65

            result = compute_swing_position_size(
                symbol=sym,
                sleeve_nav=swing_nav,
                instrument_vol=inst_vol,
                ml_prob=ml_prob,
                current_regime=regime_name,
                vvol_percentile=0.5,
                price=price,
                config=sizing_config,
            )

            if result["shares"] > 0:
                barriers = compute_barriers(
                    entry_price=price,
                    instrument_vol=inst_vol,
                    holding_days=sizing_config.get("holding_days", 10),
                )
                # Risk Governor check
                state = PortfolioState(
                    nav=sum(s.mark_to_market(prices) for s in sleeves.values()),
                    peak_nav=sum(
                        s.mark_to_market(prices) for s in sleeves.values()
                    ),
                    cash=swing_sleeve.cash,
                    sleeve_values={
                        k: v.mark_to_market(prices)
                        for k, v in sleeves.items()
                    },
                    positions={
                        s: {"notional": p.notional}
                        for s, p in swing_sleeve.positions.items()
                    },
                    day_start_nav=sum(
                        s.mark_to_market(prices) for s in sleeves.values()
                    ),
                )
                notional = result["shares"] * price
                allowed, reason = risk_governor.pre_trade_check(
                    symbol=sym,
                    side="BUY",
                    notional=notional,
                    sleeve="swing",
                    state=state,
                    sector=cand.get("sector"),
                    current_date=date.date()
                    if hasattr(date, "date")
                    else date,
                )

                if allowed:
                    orders.append(
                        {
                            "symbol": sym,
                            "sleeve": "swing",
                            "side": "BUY",
                            "qty": result["shares"],
                            "stop_price": barriers["sl_price"],
                            "target_price": barriers["tp_price"],
                            "sector": cand.get("sector"),
                            "ml_prob": ml_prob,
                        }
                    )

        return orders

    return signal_func


def run_backtest(config_path: str, start_date: str, end_date: str):
    """Run complete backtest pipeline."""

    # --- Load config ---
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = {
            "portfolio": {
                "initial_nav": 100_000,
                "sleeve_allocations": {
                    "core": 0.60,
                    "swing": 0.30,
                    "cash_buffer": 0.10,
                },
            },
            "cost_model": {},
            "risk": {},
            "swing_signals": {"momentum_threshold_pct": 0.04},
            "sizing": {"holding_days": 10},
        }

    portfolio_config = config.get("portfolio", {})
    cost_config = config.get("cost_model", {})
    risk_params = config.get("risk", {})

    # --- Load data ---
    logger.info("Loading price data...")
    price_data = load_price_data(config, start_date, end_date)
    if not price_data:
        logger.error("No price data loaded. Run scripts/download_data.py first.")
        sys.exit(1)

    # Index for regime / risk-on gate
    index_symbol = config.get("features", {}).get("index_symbol", "SPY")
    if index_symbol not in price_data:
        logger.warning(
            f"Index {index_symbol} not in price data. "
            f"Regime detection and risk-on gate disabled."
        )
        index_df = pd.DataFrame()
    else:
        index_df = price_data[index_symbol]

    # --- Regime detection (walk-forward) ---
    regime_series = None
    regime_names = None
    if len(index_df) > 200:
        logger.info("Running walk-forward regime detection...")
        regime_feats = build_regime_features(index_df["close"], config)
        if len(regime_feats) > 100:
            model_dict = fit_regime_model(
                regime_feats, n_regimes=4, method="kmeans"
            )
            raw_regimes = predict_regime(model_dict, regime_feats)
            regime_series = smooth_regime(raw_regimes, min_persistence=3)
            regime_names = label_regimes(regime_feats, raw_regimes)
            logger.info(f"Regime names: {regime_names}")

    # --- Risk Governor ---
    risk_config = RiskConfig(**{
        k: v
        for k, v in risk_params.items()
        if k in RiskConfig.__dataclass_fields__
    })
    risk_governor = RiskGovernor(risk_config)

    # --- Cost Model ---
    cost_model = CostModel(**{
        k: v
        for k, v in cost_config.items()
        if k in CostModel.__dataclass_fields__
    })

    # --- Backtester ---
    bt_config = {
        "initial_nav": portfolio_config.get("initial_nav", 100_000),
        "sleeve_allocations": portfolio_config.get(
            "sleeve_allocations",
            {"core": 0.60, "swing": 0.30, "cash_buffer": 0.10},
        ),
        "benchmark_symbol": index_symbol,
    }
    bt = Backtester(bt_config, cost_model)
    bt.load_data(price_data)

    # --- Signal function ---
    signal_func = build_signal_func(
        price_data=price_data,
        index_df=index_df,
        config=config,
        risk_governor=risk_governor,
        regime_series=regime_series,
        regime_names=regime_names,
    )

    # --- Run ---
    logger.info(
        f"Running backtest: {bt.date_range[0].date()} → "
        f"{bt.date_range[-1].date()} ({len(bt.date_range)} trading days)"
    )
    results = bt.run(signal_func)

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period:           {bt.date_range[0].date()} → {bt.date_range[-1].date()}")
    print(f"  Total Return:     {results['total_return']:.2%}")
    print(f"  Sharpe Ratio:     {results['annualized_sharpe']:.2f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:     {results['calmar_ratio']:.2f}")
    print(f"  Daily Win Rate:   {results['win_rate_daily']:.2%}")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Total Fees:       ${results['total_fees']:,.2f}")
    print(f"{'='*60}")

    # --- Save results ---
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    nav_df = results["nav_history"]
    nav_df.to_csv(results_dir / "nav_history.csv")

    trades_df = results["trades"]
    if not trades_df.empty:
        trades_df.to_csv(results_dir / "trades.csv", index=False)

    logger.info(f"Results saved to {results_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run trading bot backtest")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "example.yaml"),
        help="Config YAML path",
    )
    parser.add_argument("--start", default="2018-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    run_backtest(args.config, args.start, end_date)


if __name__ == "__main__":
    main()

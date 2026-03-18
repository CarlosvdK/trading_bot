"""
Adversarial backtest stress testing.
Deliberately injects market shocks to find strategy weaknesses.
"""

import copy
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional

from src.backtesting.cost_model import CostModel


def _deep_copy_prices(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Deep copy price data dict."""
    return {s: df.copy() for s, df in price_data.items()}


def inject_flash_crash(
    price_data: Dict[str, pd.DataFrame],
    symbol: str,
    date: pd.Timestamp,
    drop_pct: float = 0.10,
    recovery_days: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Simulate a flash crash: sharp drop then partial recovery."""
    data = _deep_copy_prices(price_data)
    if symbol not in data or date not in data[symbol].index:
        return data
    df = data[symbol]
    idx = df.index.get_loc(date)
    crash_close = df.iloc[idx]["close"] * (1 - drop_pct)

    df.loc[date, "close"] = crash_close
    df.loc[date, "low"] = crash_close * 0.97
    df.loc[date, "high"] = df.iloc[idx]["open"]
    df.loc[date, "volume"] = int(df.iloc[idx]["volume"] * 5)

    for r in range(1, recovery_days + 1):
        ri = idx + r
        if ri >= len(df):
            break
        recovery_pct = (r / recovery_days) * drop_pct * 0.6
        rd = df.index[ri]
        df.loc[rd, "open"] = crash_close * (1 + recovery_pct * 0.5)
        df.loc[rd, "close"] = crash_close * (1 + recovery_pct)
        df.loc[rd, "low"] = min(df.loc[rd, "open"], df.loc[rd, "close"]) * 0.99
        df.loc[rd, "high"] = max(df.loc[rd, "open"], df.loc[rd, "close"]) * 1.01
        df.loc[rd, "volume"] = int(df.iloc[ri]["volume"] * 3)

    data[symbol] = df
    return data


def inject_gap_down(
    price_data: Dict[str, pd.DataFrame],
    symbol: str,
    date: pd.Timestamp,
    gap_pct: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Overnight gap down: next day opens much lower than previous close."""
    data = _deep_copy_prices(price_data)
    if symbol not in data or date not in data[symbol].index:
        return data
    df = data[symbol]
    idx = df.index.get_loc(date)
    if idx == 0:
        return data

    prev_close = df.iloc[idx - 1]["close"]
    gap_open = prev_close * (1 - gap_pct)

    df.loc[date, "open"] = gap_open
    df.loc[date, "high"] = max(gap_open, df.loc[date, "close"]) * 1.005
    df.loc[date, "low"] = gap_open * 0.995
    df.loc[date, "close"] = gap_open * 0.99
    df.loc[date, "volume"] = int(df.iloc[idx]["volume"] * 3)

    data[symbol] = df
    return data


def inject_liquidity_crisis(
    price_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    start_date: pd.Timestamp,
    duration_days: int = 10,
    volume_factor: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """Market-wide liquidity dry-up: volume drops to fraction of normal."""
    data = _deep_copy_prices(price_data)
    for symbol in symbols:
        if symbol not in data:
            continue
        df = data[symbol]
        mask = (df.index >= start_date) & (
            df.index < start_date + pd.offsets.BDay(duration_days)
        )
        df.loc[mask, "volume"] = (df.loc[mask, "volume"] * volume_factor).astype(int).clip(lower=100)
        data[symbol] = df
    return data


def inject_correlation_spike(
    price_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    start_date: pd.Timestamp,
    duration_days: int = 20,
    target_corr: float = 0.95,
    direction: float = -0.02,
) -> Dict[str, pd.DataFrame]:
    """Make all symbols move together (correlation -> 1) during crisis."""
    data = _deep_copy_prices(price_data)
    if not symbols:
        return data

    for symbol in symbols:
        if symbol not in data:
            continue
        df = data[symbol]
        mask = (df.index >= start_date) & (
            df.index < start_date + pd.offsets.BDay(duration_days)
        )
        affected_dates = df.index[mask]

        for d in affected_dates:
            idx = df.index.get_loc(d)
            if idx == 0:
                continue
            orig_ret = df.iloc[idx]["close"] / df.iloc[idx - 1]["close"] - 1
            blended_ret = target_corr * direction + (1 - target_corr) * orig_ret
            new_close = max(df.iloc[idx - 1]["close"] * (1 + blended_ret), 0.01)
            df.loc[d, "close"] = new_close
            df.loc[d, "low"] = min(df.loc[d, "low"], new_close)
            df.loc[d, "high"] = max(df.loc[d, "high"], new_close)
            df.loc[d, "volume"] = int(df.iloc[idx]["volume"] * 2)

        data[symbol] = df
    return data


def inject_slippage_regime(cost_model: CostModel, multiplier: float = 3.0) -> CostModel:
    """Return a modified cost model with amplified slippage."""
    stressed = copy.deepcopy(cost_model)
    stressed.spread_bps = cost_model.spread_bps * multiplier
    stressed.market_impact_bps = cost_model.market_impact_bps * multiplier
    return stressed


class StressTestSuite:
    """Runs a battery of stress scenarios against a strategy."""

    def __init__(
        self,
        backtester_factory: Callable,
        signal_func: Callable,
        base_price_data: Dict[str, pd.DataFrame],
        config: dict,
    ):
        self.backtester_factory = backtester_factory
        self.signal_func = signal_func
        self.base_price_data = base_price_data
        self.config = config
        self.results: Dict[str, dict] = {}

    def run_scenario(self, scenario_name: str, modifier_func: Callable) -> dict:
        """Run backtest with modified data/config."""
        modified_data, modified_config = modifier_func(
            _deep_copy_prices(self.base_price_data), self.config.copy()
        )
        cost_model = CostModel(modified_config.get("cost_model", {}))
        bt = self.backtester_factory(modified_config, cost_model)
        bt.load_data(modified_data)
        result = bt.run(self.signal_func)
        self.results[scenario_name] = result
        return result

    def run_all_scenarios(self) -> pd.DataFrame:
        """Run all built-in stress scenarios."""
        symbols = list(self.base_price_data.keys())
        if not symbols:
            return pd.DataFrame()

        ref_df = self.base_price_data[symbols[0]]
        mid_idx = len(ref_df) // 2
        mid_date = ref_df.index[mid_idx]

        scenarios = {
            "baseline": lambda d, c: (d, c),
            "flash_crash": lambda d, c: (inject_flash_crash(d, symbols[0], mid_date, 0.10), c),
            "gap_down": lambda d, c: (inject_gap_down(d, symbols[0], mid_date, 0.05), c),
            "liquidity_crisis": lambda d, c: (inject_liquidity_crisis(d, symbols, mid_date, 10, 0.1), c),
            "correlation_spike": lambda d, c: (inject_correlation_spike(d, symbols, mid_date, 20), c),
        }

        rows = []
        for name, modifier in scenarios.items():
            try:
                result = self.run_scenario(name, modifier)
                rows.append({
                    "scenario": name,
                    "total_return": result.get("total_return", 0),
                    "sharpe": result.get("annualized_sharpe", 0),
                    "max_drawdown": result.get("max_drawdown", 0),
                    "total_trades": result.get("total_trades", 0),
                })
            except Exception as e:
                rows.append({"scenario": name, "error": str(e)})

        return pd.DataFrame(rows)

    def compare_to_baseline(self, baseline_results: dict, stress_results: dict) -> dict:
        """Compute degradation metrics."""
        return {
            "return_degradation": stress_results.get("total_return", 0) - baseline_results.get("total_return", 0),
            "drawdown_increase": stress_results.get("max_drawdown", 0) - baseline_results.get("max_drawdown", 0),
            "sharpe_degradation": stress_results.get("annualized_sharpe", 0) - baseline_results.get("annualized_sharpe", 0),
            "return_survived": stress_results.get("total_return", 0) > 0,
            "drawdown_acceptable": stress_results.get("max_drawdown", 0) > -0.25,
        }

    def generate_report(self) -> dict:
        """Summary of all scenarios with pass/fail."""
        report = {}
        baseline = self.results.get("baseline", {})
        for name, result in self.results.items():
            if name == "baseline":
                continue
            comparison = self.compare_to_baseline(baseline, result)
            report[name] = {**comparison, "passed": comparison["return_survived"] and comparison["drawdown_acceptable"]}

        n_passed = sum(1 for r in report.values() if r.get("passed", False))
        report["summary"] = {"total_scenarios": len(report), "passed": n_passed, "overall_pass": n_passed == len(report)}
        return report


def monte_carlo_stress(
    price_data: Dict[str, pd.DataFrame],
    n_scenarios: int = 100,
    shock_probability: float = 0.05,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Randomly inject shocks throughout data for MC stress testing."""
    rng = np.random.default_rng(seed)
    symbols = list(price_data.keys())
    if not symbols:
        return []

    ref_dates = price_data[symbols[0]].index
    scenarios = []
    for _ in range(n_scenarios):
        modified = _deep_copy_prices(price_data)
        for d in ref_dates:
            if rng.random() < shock_probability:
                shock_type = rng.choice(["flash_crash", "gap_down", "liquidity"])
                target = rng.choice(symbols)
                if shock_type == "flash_crash":
                    modified = inject_flash_crash(modified, target, d, rng.uniform(0.05, 0.20))
                elif shock_type == "gap_down":
                    modified = inject_gap_down(modified, target, d, rng.uniform(0.03, 0.10))
                else:
                    modified = inject_liquidity_crisis(modified, [target], d, rng.integers(3, 10), 0.1)
        scenarios.append(modified)

    return scenarios

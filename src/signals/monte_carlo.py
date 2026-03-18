"""
Monte Carlo position sizing and portfolio simulation.
Optimizes position size via simulation to maximize risk-adjusted returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def simulate_portfolio_paths(
    trades_history: List[Dict],
    n_simulations: int = 10000,
    n_periods: int = 252,
    initial_nav: float = 100000,
    size_multiplier: float = 1.0,
    seed: Optional[int] = None,
) -> Dict:
    """
    Bootstrap portfolio paths from historical trade returns.

    Args:
        trades_history: List of dicts with at least 'return_pct' key.
        n_simulations: Number of MC paths.
        n_periods: Trading days per path.
        initial_nav: Starting portfolio value.
        size_multiplier: Scale all trade returns by this factor.
        seed: Random seed.

    Returns:
        Dict with paths, terminal_values, max_drawdowns, sharpe_ratios.
    """
    rng = np.random.default_rng(seed)

    returns = np.array([t["return_pct"] for t in trades_history if "return_pct" in t])
    if len(returns) == 0:
        return {
            "paths": np.full((n_simulations, n_periods), initial_nav),
            "terminal_values": np.full(n_simulations, initial_nav),
            "max_drawdowns": np.zeros(n_simulations),
            "sharpe_ratios": np.zeros(n_simulations),
        }

    returns = returns * size_multiplier

    # Bootstrap: sample returns with replacement
    sampled = rng.choice(returns, size=(n_simulations, n_periods), replace=True)

    # Build NAV paths
    paths = initial_nav * np.cumprod(1 + sampled, axis=1)

    # Terminal values
    terminal_values = paths[:, -1]

    # Max drawdowns per path
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    max_drawdowns = drawdowns.min(axis=1)

    # Sharpe ratios per path
    daily_returns = np.diff(paths, axis=1) / paths[:, :-1]
    means = daily_returns.mean(axis=1)
    stds = daily_returns.std(axis=1)
    stds[stds == 0] = np.nan
    sharpe_ratios = np.where(np.isnan(stds), 0, means / stds * np.sqrt(252))

    return {
        "paths": paths,
        "terminal_values": terminal_values,
        "max_drawdowns": max_drawdowns,
        "sharpe_ratios": sharpe_ratios,
    }


def optimal_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    half_kelly: bool = True,
) -> float:
    """
    Compute optimal Kelly fraction for position sizing.

    Args:
        win_rate: Probability of winning trade (0-1).
        avg_win: Average win return (positive).
        avg_loss: Average loss return (positive, absolute value).
        half_kelly: Use half-Kelly for safety.

    Returns:
        Optimal fraction of capital per trade, clamped to [0, 0.25].
    """
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
        return 0.0

    b = avg_win / avg_loss  # Win/loss ratio
    p = win_rate
    q = 1 - p

    kelly = (p * b - q) / b

    if half_kelly:
        kelly *= 0.5

    return float(np.clip(kelly, 0.0, 0.25))


def risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade: float,
    ruin_threshold: float = 0.50,
    n_simulations: int = 10000,
    n_trades: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate probability of drawdown exceeding ruin_threshold via MC.

    Args:
        win_rate: Probability of winning trade.
        avg_win: Average win return.
        avg_loss: Average loss return (positive).
        risk_per_trade: Fraction of capital risked per trade.
        ruin_threshold: Drawdown level considered "ruin".
        n_simulations: Number of simulations.
        n_trades: Number of trades per simulation.
        seed: Random seed.

    Returns:
        Probability of ruin (0-1).
    """
    rng = np.random.default_rng(seed)

    wins = rng.random((n_simulations, n_trades)) < win_rate
    trade_returns = np.where(
        wins,
        risk_per_trade * avg_win,
        -risk_per_trade * avg_loss,
    )

    nav = np.cumprod(1 + trade_returns, axis=1)
    running_max = np.maximum.accumulate(nav, axis=1)
    drawdowns = (nav - running_max) / running_max
    max_dd = drawdowns.min(axis=1)

    ruin_count = np.sum(max_dd <= -ruin_threshold)
    return float(ruin_count / n_simulations)


def optimal_position_size_mc(
    trades_history: List[Dict],
    target_max_dd: float = 0.15,
    confidence: float = 0.95,
    n_sims: int = 5000,
    n_periods: int = 252,
    initial_nav: float = 100000,
    seed: Optional[int] = None,
) -> Dict:
    """
    Binary search for largest position size multiplier where
    P(max_dd > target) < (1 - confidence).

    Returns:
        Dict with optimal_multiplier, expected_return, expected_dd, etc.
    """
    lo, hi = 0.1, 3.0
    best_mult = lo
    best_result = None

    for _ in range(20):  # Binary search iterations
        mid = (lo + hi) / 2
        result = simulate_portfolio_paths(
            trades_history, n_sims, n_periods, initial_nav,
            size_multiplier=mid, seed=seed,
        )
        dd_exceedance = np.mean(result["max_drawdowns"] <= -target_max_dd)

        if dd_exceedance <= (1 - confidence):
            best_mult = mid
            best_result = result
            lo = mid
        else:
            hi = mid

    if best_result is None:
        best_result = simulate_portfolio_paths(
            trades_history, n_sims, n_periods, initial_nav,
            size_multiplier=best_mult, seed=seed,
        )

    expected_return = float(np.median(best_result["terminal_values"]) / initial_nav - 1)
    expected_dd = float(np.median(best_result["max_drawdowns"]))
    ci_low = float(np.percentile(best_result["terminal_values"], 5) / initial_nav - 1)
    ci_high = float(np.percentile(best_result["terminal_values"], 95) / initial_nav - 1)

    return {
        "optimal_multiplier": round(best_mult, 3),
        "expected_return": expected_return,
        "expected_dd": expected_dd,
        "confidence_interval": (ci_low, ci_high),
        "dd_exceedance_rate": float(np.mean(best_result["max_drawdowns"] <= -target_max_dd)),
    }


def drawdown_distribution(paths: np.ndarray) -> Dict:
    """
    Compute max drawdown distribution from MC paths.

    Args:
        paths: ndarray of shape (n_sims, n_periods).

    Returns:
        Dict with mean, median, p95, p99 of max drawdowns.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    max_dd = drawdowns.min(axis=1)

    return {
        "mean": float(np.mean(max_dd)),
        "median": float(np.median(max_dd)),
        "p95": float(np.percentile(max_dd, 5)),   # 5th percentile = worst 5%
        "p99": float(np.percentile(max_dd, 1)),
        "std": float(np.std(max_dd)),
    }


def position_size_sensitivity(
    trades_history: List[Dict],
    multiplier_range: Optional[List[float]] = None,
    n_sims: int = 2000,
    n_periods: int = 252,
    initial_nav: float = 100000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute risk/return profile across position size multipliers.

    Returns:
        DataFrame with columns: multiplier, expected_return, max_dd_median,
        max_dd_p95, sharpe_median, risk_of_ruin.
    """
    if multiplier_range is None:
        multiplier_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    rows = []
    for mult in multiplier_range:
        result = simulate_portfolio_paths(
            trades_history, n_sims, n_periods, initial_nav,
            size_multiplier=mult, seed=seed,
        )
        dd_dist = drawdown_distribution(result["paths"])
        ruin_rate = float(np.mean(result["max_drawdowns"] <= -0.50))

        rows.append({
            "multiplier": mult,
            "expected_return": float(np.median(result["terminal_values"]) / initial_nav - 1),
            "max_dd_median": dd_dist["median"],
            "max_dd_p95": dd_dist["p95"],
            "sharpe_median": float(np.median(result["sharpe_ratios"])),
            "risk_of_ruin": ruin_rate,
        })

    return pd.DataFrame(rows)

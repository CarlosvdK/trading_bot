"""
Synthetic OHLCV data generator with regime shifts.
No API keys needed — generates realistic market data for backtesting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
#  Regime Definitions                                                  #
# ------------------------------------------------------------------ #

REGIME_PARAMS = {
    "bull": {"mu": 0.15 / 252, "sigma": 0.15 / np.sqrt(252), "vol_mean": 1e6, "vol_std": 3e5},
    "bear": {"mu": -0.10 / 252, "sigma": 0.25 / np.sqrt(252), "vol_mean": 1.5e6, "vol_std": 5e5},
    "choppy": {"mu": 0.02 / 252, "sigma": 0.20 / np.sqrt(252), "vol_mean": 8e5, "vol_std": 2e5},
    "crisis": {"mu": -0.30 / 252, "sigma": 0.45 / np.sqrt(252), "vol_mean": 3e6, "vol_std": 1e6},
}


@dataclass
class RegimeSpec:
    """Defines a regime period."""
    name: str
    start_date: str
    end_date: str
    mu: float = None
    sigma: float = None

    def __post_init__(self):
        if self.mu is None:
            params = REGIME_PARAMS.get(self.name, REGIME_PARAMS["choppy"])
            self.mu = params["mu"]
            self.sigma = params["sigma"]


# ------------------------------------------------------------------ #
#  Core GBM Generator                                                  #
# ------------------------------------------------------------------ #

def generate_gbm_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    mu: float = 0.10 / 252,
    sigma: float = 0.20 / np.sqrt(252),
    initial_price: float = 100.0,
    vol_mean: float = 1e6,
    vol_std: float = 3e5,
    vol_autocorr: float = 0.7,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using Geometric Brownian Motion.

    Args:
        symbol: Ticker symbol (for labeling).
        start_date: Start date string.
        end_date: End date string.
        mu: Daily drift.
        sigma: Daily volatility.
        initial_price: Starting price.
        vol_mean: Mean daily volume.
        vol_std: Volume standard deviation.
        vol_autocorr: Volume autocorrelation (clustering).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [open, high, low, close, volume], DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Generate log returns
    log_returns = (mu - 0.5 * sigma**2) + sigma * rng.standard_normal(n)

    # Build close prices
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    close = np.exp(log_prices)

    # Generate intraday range from daily vol
    daily_vol = np.abs(log_returns)
    intraday_range = daily_vol * close

    # High/Low around close
    high_offset = rng.uniform(0.3, 0.8, n) * intraday_range
    low_offset = rng.uniform(0.3, 0.8, n) * intraday_range
    high = close + high_offset
    low = close - low_offset
    low = np.maximum(low, close * 0.9)  # Floor at 90% of close

    # Open: between prev close and current close
    prev_close = np.roll(close, 1)
    prev_close[0] = initial_price
    gap_noise = rng.normal(0, sigma * 0.3, n)
    open_price = prev_close * np.exp(gap_noise)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    low = np.maximum(low, 0.01)  # No negative prices

    # Volume with autocorrelation (clustering)
    vol_innovations = rng.lognormal(
        np.log(vol_mean) - 0.5 * (vol_std / vol_mean) ** 2,
        vol_std / vol_mean,
        n,
    )
    volume = np.zeros(n)
    volume[0] = vol_innovations[0]
    for i in range(1, n):
        volume[i] = vol_autocorr * volume[i - 1] + (1 - vol_autocorr) * vol_innovations[i]
    volume = np.maximum(volume, 1000).astype(int)

    df = pd.DataFrame(
        {
            "open": np.round(open_price, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ------------------------------------------------------------------ #
#  Regime-Aware Generator                                              #
# ------------------------------------------------------------------ #

def _interpolate_params(
    params_a: dict, params_b: dict, blend: float
) -> Tuple[float, float, float, float]:
    """Linearly interpolate between two regime parameter sets."""
    mu = params_a["mu"] * (1 - blend) + params_b["mu"] * blend
    sigma = params_a["sigma"] * (1 - blend) + params_b["sigma"] * blend
    vol_mean = params_a["vol_mean"] * (1 - blend) + params_b["vol_mean"] * blend
    vol_std = params_a["vol_std"] * (1 - blend) + params_b["vol_std"] * blend
    return mu, sigma, vol_mean, vol_std


def generate_regime_aware_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    regimes: List[RegimeSpec],
    initial_price: float = 100.0,
    transition_days: int = 10,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate OHLCV with regime transitions.

    Args:
        symbol: Ticker symbol.
        start_date: Start date.
        end_date: End date.
        regimes: List of RegimeSpec defining regime periods.
        initial_price: Starting price.
        transition_days: Days to blend between regimes.
        seed: Random seed.

    Returns:
        (ohlcv_df, regime_series) where regime_series has regime name per date.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    if n == 0:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return empty, pd.Series(dtype=str)

    # Build per-day parameter arrays
    daily_mu = np.zeros(n)
    daily_sigma = np.zeros(n)
    daily_vol_mean = np.full(n, 1e6)
    daily_vol_std = np.full(n, 3e5)
    regime_labels = ["choppy"] * n

    # Assign base regime for each date
    regime_map = {}
    for spec in regimes:
        rs = pd.Timestamp(spec.start_date)
        re = pd.Timestamp(spec.end_date)
        params = REGIME_PARAMS.get(spec.name, REGIME_PARAMS["choppy"])
        if spec.mu is not None:
            params = {**params, "mu": spec.mu}
        if spec.sigma is not None:
            params = {**params, "sigma": spec.sigma}
        regime_map[(rs, re)] = (spec.name, params)

    for i, d in enumerate(dates):
        assigned = False
        for (rs, re), (name, params) in regime_map.items():
            if rs <= d <= re:
                daily_mu[i] = params["mu"]
                daily_sigma[i] = params["sigma"]
                daily_vol_mean[i] = params["vol_mean"]
                daily_vol_std[i] = params["vol_std"]
                regime_labels[i] = name
                assigned = True
                break
        if not assigned:
            p = REGIME_PARAMS["choppy"]
            daily_mu[i] = p["mu"]
            daily_sigma[i] = p["sigma"]
            daily_vol_mean[i] = p["vol_mean"]
            daily_vol_std[i] = p["vol_std"]

    # Smooth transitions
    if transition_days > 1:
        for col in [daily_mu, daily_sigma, daily_vol_mean, daily_vol_std]:
            smoothed = pd.Series(col).rolling(
                transition_days, min_periods=1, center=True
            ).mean().values
            col[:] = smoothed

    # Generate prices
    log_returns = (daily_mu - 0.5 * daily_sigma**2) + daily_sigma * rng.standard_normal(n)
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    close = np.exp(log_prices)

    # OHLCV construction
    intraday_range = np.abs(log_returns) * close
    high = close + rng.uniform(0.3, 0.8, n) * intraday_range
    low = close - rng.uniform(0.3, 0.8, n) * intraday_range
    low = np.maximum(low, close * 0.9)

    prev_close = np.roll(close, 1)
    prev_close[0] = initial_price
    open_price = prev_close * np.exp(rng.normal(0, daily_sigma * 0.3))

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    low = np.maximum(low, 0.01)

    # Volume
    vol_innovations = np.array([
        rng.lognormal(np.log(vm) - 0.5 * (vs / vm) ** 2, max(vs / vm, 0.01))
        for vm, vs in zip(daily_vol_mean, daily_vol_std)
    ])
    volume = np.zeros(n)
    volume[0] = vol_innovations[0]
    for i in range(1, n):
        volume[i] = 0.7 * volume[i - 1] + 0.3 * vol_innovations[i]
    volume = np.maximum(volume, 1000).astype(int)

    ohlcv = pd.DataFrame(
        {
            "open": np.round(open_price, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": volume,
        },
        index=dates,
    )
    ohlcv.index.name = "date"
    regime_series = pd.Series(regime_labels, index=dates, name="regime")

    return ohlcv, regime_series


# ------------------------------------------------------------------ #
#  Correlated Universe Generator                                       #
# ------------------------------------------------------------------ #

def generate_correlated_universe(
    symbols: List[str],
    start_date: str,
    end_date: str,
    correlation_matrix: Optional[np.ndarray] = None,
    regimes: Optional[List[RegimeSpec]] = None,
    initial_prices: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple correlated stocks using Cholesky decomposition.

    Args:
        symbols: List of ticker symbols.
        start_date: Start date.
        end_date: End date.
        correlation_matrix: NxN correlation matrix. Defaults to moderate correlation.
        regimes: Optional regime specs applied to all stocks.
        initial_prices: Starting price per symbol.
        seed: Random seed.

    Returns:
        Dict of symbol -> OHLCV DataFrame.
    """
    rng = np.random.default_rng(seed)
    n_symbols = len(symbols)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    if correlation_matrix is None:
        # Default: moderate correlation (0.3-0.6)
        correlation_matrix = np.full((n_symbols, n_symbols), 0.4)
        np.fill_diagonal(correlation_matrix, 1.0)

    if initial_prices is None:
        initial_prices = {s: rng.uniform(20, 200) for s in symbols}

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(correlation_matrix)
    uncorrelated = rng.standard_normal((n_days, n_symbols))
    correlated_noise = uncorrelated @ L.T

    universe = {}
    for j, symbol in enumerate(symbols):
        init_p = initial_prices.get(symbol, 100.0)

        if regimes:
            # Build per-day params from regimes
            daily_mu = np.full(n_days, 0.10 / 252)
            daily_sigma = np.full(n_days, 0.20 / np.sqrt(252))
            daily_vol_mean = np.full(n_days, 1e6)

            for spec in regimes:
                rs = pd.Timestamp(spec.start_date)
                re = pd.Timestamp(spec.end_date)
                params = REGIME_PARAMS.get(spec.name, REGIME_PARAMS["choppy"])
                for i, d in enumerate(dates):
                    if rs <= d <= re:
                        daily_mu[i] = params["mu"]
                        daily_sigma[i] = params["sigma"]
                        daily_vol_mean[i] = params["vol_mean"]

            log_returns = (daily_mu - 0.5 * daily_sigma**2) + daily_sigma * correlated_noise[:, j]
        else:
            # Default params with per-stock variation
            stock_mu = rng.uniform(0.05, 0.20) / 252
            stock_sigma = rng.uniform(0.15, 0.35) / np.sqrt(252)
            log_returns = (stock_mu - 0.5 * stock_sigma**2) + stock_sigma * correlated_noise[:, j]
            daily_vol_mean = np.full(n_days, rng.uniform(5e5, 2e6))

        close = np.exp(np.log(init_p) + np.cumsum(log_returns))
        intraday = np.abs(log_returns) * close

        high = close + rng.uniform(0.3, 0.8, n_days) * intraday
        low = close - rng.uniform(0.3, 0.8, n_days) * intraday
        low = np.maximum(low, close * 0.9)

        prev_close = np.roll(close, 1)
        prev_close[0] = init_p
        open_price = prev_close * np.exp(rng.normal(0, 0.005, n_days))

        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))
        low = np.maximum(low, 0.01)

        volume = np.maximum(
            rng.lognormal(np.log(daily_vol_mean.mean()), 0.3, n_days), 1000
        ).astype(int)

        universe[symbol] = pd.DataFrame(
            {
                "open": np.round(open_price, 2),
                "high": np.round(high, 2),
                "low": np.round(low, 2),
                "close": np.round(close, 2),
                "volume": volume,
            },
            index=dates,
        )
        universe[symbol].index.name = "date"

    return universe


# ------------------------------------------------------------------ #
#  Index from Universe                                                 #
# ------------------------------------------------------------------ #

def generate_index_from_universe(
    universe_data: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Create a market-cap-weighted index from universe data.

    Args:
        universe_data: Dict of symbol -> OHLCV DataFrame.
        weights: Optional weights per symbol. Defaults to equal weight.

    Returns:
        OHLCV DataFrame representing the index.
    """
    if not universe_data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    symbols = list(universe_data.keys())
    if weights is None:
        weights = {s: 1.0 / len(symbols) for s in symbols}

    # Normalize weights
    total_w = sum(weights.values())
    weights = {s: w / total_w for s, w in weights.items()}

    all_dates = set()
    for df in universe_data.values():
        all_dates.update(df.index)
    dates = sorted(all_dates)

    index_data = []
    for d in dates:
        o = h = l = c = 0.0
        vol = 0
        for s in symbols:
            df = universe_data[s]
            if d in df.index:
                w = weights.get(s, 0)
                row = df.loc[d]
                o += row["open"] * w
                h += row["high"] * w
                l += row["low"] * w
                c += row["close"] * w
                vol += row["volume"]
        index_data.append({"open": o, "high": h, "low": l, "close": c, "volume": vol})

    idx_df = pd.DataFrame(index_data, index=pd.DatetimeIndex(dates))
    idx_df.index.name = "date"
    return idx_df


# ------------------------------------------------------------------ #
#  SyntheticDataProvider                                               #
# ------------------------------------------------------------------ #

class SyntheticDataProvider:
    """
    Drop-in replacement for DataProvider using synthetic data.
    Generates a complete universe with correlated stocks and an index.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        regimes: Optional[List[RegimeSpec]] = None,
        seed: int = 42,
    ):
        self.symbols = symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "JNJ", "V",
        ]
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        self.regimes = regimes or self._default_regimes()
        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._index: Optional[pd.DataFrame] = None

    def _default_regimes(self) -> List[RegimeSpec]:
        """Default regime schedule covering multi-year period."""
        return [
            RegimeSpec("bull", "2020-01-01", "2020-02-15"),
            RegimeSpec("crisis", "2020-02-16", "2020-04-30"),
            RegimeSpec("bull", "2020-05-01", "2021-03-31"),
            RegimeSpec("choppy", "2021-04-01", "2021-09-30"),
            RegimeSpec("bull", "2021-10-01", "2021-12-31"),
            RegimeSpec("bear", "2022-01-01", "2022-06-30"),
            RegimeSpec("choppy", "2022-07-01", "2022-12-31"),
            RegimeSpec("bull", "2023-01-01", "2023-07-31"),
            RegimeSpec("choppy", "2023-08-01", "2023-10-31"),
            RegimeSpec("bull", "2023-11-01", "2024-03-31"),
            RegimeSpec("choppy", "2024-04-01", "2024-06-30"),
            RegimeSpec("bull", "2024-07-01", "2024-12-31"),
        ]

    def load(self) -> Dict[str, pd.DataFrame]:
        """Generate and cache synthetic universe data."""
        if self._data is None:
            self._data = generate_correlated_universe(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                regimes=self.regimes,
                seed=self.seed,
            )
        return self._data

    def get_prices(self) -> Dict[str, pd.DataFrame]:
        """Return dict of symbol -> OHLCV DataFrame."""
        return self.load()

    def get_index(self) -> pd.DataFrame:
        """Return synthetic market index."""
        if self._index is None:
            self._index = generate_index_from_universe(self.load())
        return self._index

    def get_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get OHLCV for a single symbol."""
        data = self.load()
        return data.get(symbol)

    def get_close_matrix(self) -> pd.DataFrame:
        """Return DataFrame of close prices (symbols as columns)."""
        data = self.load()
        closes = {s: df["close"] for s, df in data.items()}
        return pd.DataFrame(closes)

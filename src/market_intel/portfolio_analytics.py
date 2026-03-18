"""
Comprehensive portfolio analytics — performance metrics, trade analysis,
risk decomposition, and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class PerformanceMetrics:
    """Full suite of portfolio performance metrics from a NAV series."""

    def __init__(self, nav_series: pd.Series, risk_free_rate: float = 0.04):
        self.nav = nav_series.dropna()
        self.rf = risk_free_rate
        self.daily_returns = self.nav.pct_change().dropna()
        self._rf_daily = self.rf / 252

    @classmethod
    def from_nav_series(cls, nav_series: pd.Series, risk_free_rate: float = 0.04) -> "PerformanceMetrics":
        return cls(nav_series, risk_free_rate)

    @property
    def total_return(self) -> float:
        if len(self.nav) < 2:
            return 0.0
        return float(self.nav.iloc[-1] / self.nav.iloc[0] - 1)

    @property
    def cagr(self) -> float:
        if len(self.nav) < 2:
            return 0.0
        years = len(self.daily_returns) / 252
        if years <= 0:
            return 0.0
        return float((self.nav.iloc[-1] / self.nav.iloc[0]) ** (1 / years) - 1)

    @property
    def sharpe_ratio(self) -> float:
        excess = self.daily_returns - self._rf_daily
        std = excess.std()
        if std == 0 or np.isnan(std) or std < 1e-15:
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    @property
    def sortino_ratio(self) -> float:
        excess = self.daily_returns - self._rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        cum = (1 + self.daily_returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min()) if len(dd) > 0 else 0.0

    @property
    def max_drawdown_duration_days(self) -> int:
        cum = (1 + self.daily_returns).cumprod()
        peak = cum.cummax()
        in_dd = cum < peak
        if not in_dd.any():
            return 0
        groups = (~in_dd).cumsum()
        dd_lengths = in_dd.groupby(groups).sum()
        return int(dd_lengths.max())

    @property
    def calmar_ratio(self) -> float:
        dd = self.max_drawdown
        return float(self.cagr / abs(dd)) if dd != 0 else 0.0

    @property
    def avg_drawdown(self) -> float:
        cum = (1 + self.daily_returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        return float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    @property
    def ulcer_index(self) -> float:
        cum = (1 + self.daily_returns).cumprod()
        dd_pct = ((cum - cum.cummax()) / cum.cummax() * 100)
        return float(np.sqrt((dd_pct**2).mean()))

    @property
    def omega_ratio(self) -> float:
        threshold = self._rf_daily
        excess = self.daily_returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess <= 0].sum())
        return float(gains / losses) if losses > 0 else float('inf') if gains > 0 else 0.0

    @property
    def tail_ratio(self) -> float:
        p95 = np.percentile(self.daily_returns, 95) if len(self.daily_returns) > 0 else 0
        p5 = abs(np.percentile(self.daily_returns, 5)) if len(self.daily_returns) > 0 else 1
        return float(p95 / p5) if p5 > 0 else 0.0

    @property
    def var_95(self) -> float:
        if len(self.daily_returns) == 0:
            return 0.0
        return float(np.percentile(self.daily_returns, 5))

    @property
    def cvar_95(self) -> float:
        var = self.var_95
        tail = self.daily_returns[self.daily_returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    @property
    def skewness(self) -> float:
        return float(self.daily_returns.skew()) if len(self.daily_returns) > 2 else 0.0

    @property
    def kurtosis(self) -> float:
        return float(self.daily_returns.kurtosis()) if len(self.daily_returns) > 3 else 0.0

    @property
    def best_day(self) -> float:
        return float(self.daily_returns.max()) if len(self.daily_returns) > 0 else 0.0

    @property
    def worst_day(self) -> float:
        return float(self.daily_returns.min()) if len(self.daily_returns) > 0 else 0.0

    @property
    def win_rate_daily(self) -> float:
        if len(self.daily_returns) == 0:
            return 0.0
        return float((self.daily_returns > 0).mean())

    @property
    def profit_factor(self) -> float:
        gains = self.daily_returns[self.daily_returns > 0].sum()
        losses = abs(self.daily_returns[self.daily_returns < 0].sum())
        return float(gains / losses) if losses > 0 else float(gains) if gains > 0 else 0.0

    @property
    def recovery_factor(self) -> float:
        dd = abs(self.max_drawdown)
        return float(self.total_return / dd) if dd > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return, "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio, "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio, "max_drawdown": self.max_drawdown,
            "max_dd_duration_days": self.max_drawdown_duration_days,
            "avg_drawdown": self.avg_drawdown, "ulcer_index": self.ulcer_index,
            "omega_ratio": self.omega_ratio, "tail_ratio": self.tail_ratio,
            "var_95": self.var_95, "cvar_95": self.cvar_95,
            "skewness": self.skewness, "kurtosis": self.kurtosis,
            "best_day": self.best_day, "worst_day": self.worst_day,
            "win_rate_daily": self.win_rate_daily, "profit_factor": self.profit_factor,
            "recovery_factor": self.recovery_factor,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class TradeAnalytics:
    """Analyze individual trade performance."""

    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df

    @classmethod
    def from_trades(cls, trades_df: pd.DataFrame) -> "TradeAnalytics":
        return cls(trades_df)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0 or "pnl" not in self.trades.columns:
            return 0.0
        return float((self.trades["pnl"] > 0).mean())

    @property
    def avg_win(self) -> float:
        if "pnl" not in self.trades.columns:
            return 0.0
        wins = self.trades[self.trades["pnl"] > 0]["pnl"]
        return float(wins.mean()) if len(wins) > 0 else 0.0

    @property
    def avg_loss(self) -> float:
        if "pnl" not in self.trades.columns:
            return 0.0
        losses = self.trades[self.trades["pnl"] < 0]["pnl"]
        return float(losses.mean()) if len(losses) > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        if "pnl" not in self.trades.columns:
            return 0.0
        gains = self.trades[self.trades["pnl"] > 0]["pnl"].sum()
        losses = abs(self.trades[self.trades["pnl"] < 0]["pnl"].sum())
        return float(gains / losses) if losses > 0 else float(gains) if gains > 0 else 0.0

    @property
    def expectancy(self) -> float:
        if "pnl" not in self.trades.columns or self.total_trades == 0:
            return 0.0
        return float(self.trades["pnl"].mean())

    @property
    def sqn(self) -> float:
        """System Quality Number."""
        if "pnl" not in self.trades.columns or self.total_trades < 2:
            return 0.0
        mean = self.trades["pnl"].mean()
        std = self.trades["pnl"].std()
        if std == 0:
            return 0.0
        return float(mean / std * np.sqrt(min(self.total_trades, 100)))

    @property
    def max_consecutive_wins(self) -> int:
        if "pnl" not in self.trades.columns:
            return 0
        wins = (self.trades["pnl"] > 0).astype(int)
        return int(_max_consecutive(wins, 1))

    @property
    def max_consecutive_losses(self) -> int:
        if "pnl" not in self.trades.columns:
            return 0
        losses = (self.trades["pnl"] <= 0).astype(int)
        return int(_max_consecutive(losses, 1))

    def by_signal_type(self) -> dict:
        if "signal_type" not in self.trades.columns or "pnl" not in self.trades.columns:
            return {}
        result = {}
        for st, group in self.trades.groupby("signal_type"):
            result[st] = {
                "count": len(group), "win_rate": float((group["pnl"] > 0).mean()),
                "avg_pnl": float(group["pnl"].mean()), "total_pnl": float(group["pnl"].sum()),
            }
        return result

    def monthly_returns(self) -> pd.DataFrame:
        if "close_date" not in self.trades.columns or "pnl" not in self.trades.columns:
            return pd.DataFrame()
        df = self.trades.copy()
        df["month"] = pd.to_datetime(df["close_date"]).dt.to_period("M")
        return df.groupby("month")["pnl"].sum().to_frame()


class RiskDecomposition:
    """Decompose portfolio risk and return attribution."""

    def __init__(self, nav_history_df: pd.DataFrame, trades_df: Optional[pd.DataFrame] = None):
        self.nav_df = nav_history_df
        self.trades = trades_df

    def sleeve_attribution(self) -> dict:
        """Return contribution from core vs swing sleeves."""
        result = {}
        for col in ["core_nav", "swing_nav"]:
            if col in self.nav_df.columns:
                sleeve = self.nav_df[col]
                if len(sleeve) >= 2:
                    ret = sleeve.iloc[-1] / sleeve.iloc[0] - 1
                    result[col.replace("_nav", "")] = {"total_return": float(ret)}
        return result

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        if "nav" not in self.nav_df.columns:
            return pd.Series(dtype=float)
        rets = self.nav_df["nav"].pct_change().dropna()
        rolling_mean = rets.rolling(window).mean()
        rolling_std = rets.rolling(window).std()
        return (rolling_mean / rolling_std * np.sqrt(252)).dropna()

    def rolling_volatility(self, window: int = 21) -> pd.Series:
        if "nav" not in self.nav_df.columns:
            return pd.Series(dtype=float)
        rets = self.nav_df["nav"].pct_change().dropna()
        return (rets.rolling(window).std() * np.sqrt(252)).dropna()

    def drawdown_analysis(self) -> List[dict]:
        """Each drawdown period with depth, duration, recovery."""
        if "nav" not in self.nav_df.columns:
            return []
        nav = self.nav_df["nav"]
        cum = nav / nav.iloc[0]
        peak = cum.cummax()
        dd = (cum - peak) / peak

        drawdowns = []
        in_dd = False
        start = trough = trough_val = None

        for i, (date, val) in enumerate(dd.items()):
            if val < 0 and not in_dd:
                in_dd = True
                start = date
                trough = date
                trough_val = val
            elif val < 0 and in_dd:
                if val < trough_val:
                    trough = date
                    trough_val = val
            elif val >= 0 and in_dd:
                in_dd = False
                drawdowns.append({
                    "start": start, "trough": trough, "end": date,
                    "depth": float(trough_val),
                    "duration": (date - start).days,
                    "recovery_time": (date - trough).days,
                })

        return drawdowns

    def correlation_to_benchmark(self, benchmark_returns: pd.Series) -> dict:
        if "nav" not in self.nav_df.columns:
            return {}
        port_ret = self.nav_df["nav"].pct_change().dropna()
        aligned = pd.DataFrame({"port": port_ret, "bench": benchmark_returns}).dropna()
        if len(aligned) < 10:
            return {}

        corr = aligned["port"].corr(aligned["bench"])
        beta = aligned["port"].cov(aligned["bench"]) / aligned["bench"].var()
        alpha = (aligned["port"].mean() - beta * aligned["bench"].mean()) * 252
        r_sq = corr ** 2
        te = (aligned["port"] - aligned["bench"]).std() * np.sqrt(252)
        ir = (aligned["port"].mean() - aligned["bench"].mean()) / (aligned["port"] - aligned["bench"]).std() * np.sqrt(252) if te > 0 else 0

        return {"correlation": float(corr), "beta": float(beta), "alpha": float(alpha),
                "r_squared": float(r_sq), "tracking_error": float(te), "info_ratio": float(ir)}

    def tail_risk_metrics(self) -> dict:
        if "nav" not in self.nav_df.columns:
            return {}
        rets = self.nav_df["nav"].pct_change().dropna()
        var95 = float(np.percentile(rets, 5))
        cvar95 = float(rets[rets <= var95].mean()) if (rets <= var95).any() else var95
        return {
            "var_95": var95, "cvar_95": cvar95,
            "max_loss_1d": float(rets.min()),
            "max_loss_5d": float(rets.rolling(5).sum().min()) if len(rets) >= 5 else float(rets.min()),
            "expected_shortfall": cvar95,
        }


class PortfolioReport:
    """Generate formatted portfolio reports."""

    def __init__(self, backtest_results: dict):
        self.results = backtest_results
        self.nav_df = backtest_results.get("nav_history", pd.DataFrame())

    def generate_summary(self) -> dict:
        if "nav" not in self.nav_df.columns:
            return self.results
        metrics = PerformanceMetrics.from_nav_series(self.nav_df["nav"])
        return {**metrics.to_dict(), "total_trades": self.results.get("total_trades", 0),
                "total_fees": self.results.get("total_fees", 0)}

    def generate_text_report(self) -> str:
        summary = self.generate_summary()
        lines = ["=" * 60, "  PORTFOLIO PERFORMANCE REPORT", "=" * 60, ""]
        for key, val in summary.items():
            if isinstance(val, float):
                if "return" in key or "drawdown" in key or "rate" in key or "alpha" in key:
                    lines.append(f"  {key:30s}: {val:>10.2%}")
                else:
                    lines.append(f"  {key:30s}: {val:>10.4f}")
            else:
                lines.append(f"  {key:30s}: {val}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def monthly_returns_table(self) -> str:
        if "nav" not in self.nav_df.columns:
            return "No NAV data available"
        rets = self.nav_df["nav"].pct_change().dropna()
        rets.index = pd.to_datetime(rets.index)
        monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({"return": monthly})
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot_table(values="return", index="year", columns="month", aggfunc="sum")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]
        return pivot.to_string(float_format=lambda x: f"{x:.2%}")

    @staticmethod
    def compare_strategies(results_list: List[dict], names: List[str]) -> pd.DataFrame:
        rows = []
        for result, name in zip(results_list, names):
            nav_df = result.get("nav_history", pd.DataFrame())
            if "nav" in nav_df.columns:
                metrics = PerformanceMetrics.from_nav_series(nav_df["nav"])
                row = {"strategy": name, **metrics.to_dict()}
            else:
                row = {"strategy": name}
            rows.append(row)
        return pd.DataFrame(rows)


class RollingAnalysis:
    """Rolling window analytics."""

    @staticmethod
    def rolling_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
        aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
        cov = aligned["p"].rolling(window).cov(aligned["b"])
        var = aligned["b"].rolling(window).var()
        return (cov / var).dropna()

    @staticmethod
    def rolling_correlation(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63) -> pd.Series:
        aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
        return aligned["p"].rolling(window).corr(aligned["b"]).dropna()

    @staticmethod
    def regime_performance(nav_series: pd.Series, regime_series: pd.Series) -> dict:
        rets = nav_series.pct_change().dropna()
        aligned = pd.DataFrame({"ret": rets, "regime": regime_series}).dropna()
        result = {}
        for regime, group in aligned.groupby("regime"):
            r = group["ret"]
            sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
            result[regime] = {"mean_return": float(r.mean()), "volatility": float(r.std() * np.sqrt(252)),
                              "sharpe": float(sharpe), "n_days": len(r)}
        return result

    @staticmethod
    def drawdown_underwater_chart_data(nav_series: pd.Series) -> pd.Series:
        cum = nav_series / nav_series.iloc[0]
        return (cum - cum.cummax()) / cum.cummax()


def _max_consecutive(series: pd.Series, value: int) -> int:
    """Find max consecutive occurrences of value in series."""
    groups = (series != value).cumsum()
    filtered = series[series == value]
    if len(filtered) == 0:
        return 0
    return int(filtered.groupby(groups).count().max())

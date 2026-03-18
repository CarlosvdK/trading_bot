"""
Walk-forward optimization harness with overfitting detection.
Automated parameter tuning with proper temporal cross-validation.
"""

import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ParameterSpace:
    """Defines a parameter search space."""
    name: str
    param_grid: Dict[str, List]

    def combinations(self) -> List[Dict]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def n_combinations(self) -> int:
        result = 1
        for v in self.param_grid.values():
            result *= len(v)
        return result


def _calmar(returns: pd.Series) -> float:
    total_ret = (1 + returns).prod() - 1
    cum = (1 + returns).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return float(total_ret / abs(dd)) if dd != 0 else 0.0


def _sortino(returns: pd.Series, target: float = 0) -> float:
    excess = returns - target / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses > 0 else float(gains) if gains > 0 else 0.0


class WalkForwardOptimizer:
    """Walk-forward parameter optimization with temporal cross-validation."""

    OBJECTIVES = {
        "sharpe": lambda rets: rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0,
        "calmar": lambda rets: _calmar(rets),
        "return": lambda rets: float((1 + rets).prod() - 1),
        "sortino": lambda rets: _sortino(rets),
        "profit_factor": lambda rets: _profit_factor(rets),
    }

    def __init__(self, config: dict, objective: str = "sharpe"):
        self.config = config
        if objective not in self.OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}")
        self.objective = objective
        self.objective_func = self.OBJECTIVES[objective]

    def generate_folds(
        self, dates: pd.DatetimeIndex, initial_train_pct: float = 0.5,
        n_folds: int = 5, embargo_days: int = 5, purge_days: int = 3,
        expanding: bool = True,
    ) -> List[Dict]:
        """Generate walk-forward folds with embargo and purge gaps."""
        n = len(dates)
        initial_train_end = int(n * initial_train_pct)
        remaining = n - initial_train_end
        fold_size = max(remaining // n_folds, 1)

        folds = []
        for i in range(n_folds):
            test_start_idx = initial_train_end + i * fold_size
            test_end_idx = min(test_start_idx + fold_size, n)
            if test_start_idx >= n:
                break

            train_end_idx = test_start_idx - embargo_days - purge_days
            if train_end_idx <= 0:
                continue

            train_start_idx = 0 if expanding else max(0, train_end_idx - initial_train_end)
            embargo_end_idx = min(test_start_idx, n)

            folds.append({
                "fold": i,
                "train_start": dates[train_start_idx],
                "train_end": dates[min(train_end_idx, n - 1)],
                "embargo_end": dates[min(embargo_end_idx, n - 1)],
                "test_start": dates[min(test_start_idx, n - 1)],
                "test_end": dates[min(test_end_idx - 1, n - 1)],
                "train_size": train_end_idx - train_start_idx,
                "test_size": test_end_idx - test_start_idx,
            })
        return folds

    def optimize(
        self, param_space: ParameterSpace, evaluate_func: Callable,
        dates: pd.DatetimeIndex, n_folds: int = 5, embargo_days: int = 5,
    ) -> Dict:
        """Grid search over parameters with walk-forward evaluation."""
        folds = self.generate_folds(dates, n_folds=n_folds, embargo_days=embargo_days)
        combos = param_space.combinations()

        all_results = []
        for params in combos:
            fold_scores = []
            for fold in folds:
                try:
                    test_returns = evaluate_func(
                        params, (fold["train_start"], fold["train_end"]),
                        (fold["test_start"], fold["test_end"]),
                    )
                    fold_scores.append(self.objective_func(test_returns))
                except Exception:
                    fold_scores.append(float("-inf"))

            avg = np.mean([s for s in fold_scores if s != float("-inf")])
            all_results.append({
                "params": params,
                "avg_score": float(avg) if not np.isnan(avg) else 0,
                "fold_scores": fold_scores,
                "stability": self.compute_stability_score(fold_scores),
            })

        all_results.sort(key=lambda r: r["avg_score"], reverse=True)
        best = all_results[0] if all_results else {"params": {}, "avg_score": 0, "stability": 0}
        return {
            "best_params": best["params"], "best_score": best["avg_score"],
            "stability_score": best["stability"], "all_results": all_results,
            "folds": folds, "n_combinations": len(combos),
        }

    def compute_stability_score(self, fold_scores: List[float]) -> float:
        """How consistent is performance across folds? Score 0-1."""
        valid = [s for s in fold_scores if s != float("-inf") and not np.isnan(s)]
        if len(valid) <= 1:
            return 1.0

        scores = np.array(valid)
        mean = scores.mean()
        std = scores.std()
        cv_score = max(0, 1 - abs(std / mean)) if abs(mean) > 1e-10 else 0.5
        positive_rate = sum(1 for s in scores if s > 0) / len(scores)
        return round(float(cv_score * 0.5 + positive_rate * 0.5), 4)


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV) from Lopez de Prado."""

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def generate_paths(self, dates: pd.DatetimeIndex) -> List[Dict]:
        """Generate all combinatorial fold paths."""
        n = len(dates)
        group_size = n // self.n_splits
        embargo_size = int(n * self.embargo_pct)

        test_combos = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        paths = []
        for test_groups in test_combos:
            test_idx, train_idx = [], []
            for g in range(self.n_splits):
                start = g * group_size
                end = min((g + 1) * group_size, n)
                group_range = list(range(start, end))

                if g in test_groups:
                    test_idx.extend(group_range)
                else:
                    for tg in test_groups:
                        ts, te = tg * group_size, min((tg + 1) * group_size, n)
                        group_range = [i for i in group_range
                                       if not (ts - embargo_size <= i < ts or te <= i < te + embargo_size)]
                    train_idx.extend(group_range)

            paths.append({
                "train_indices": sorted(train_idx), "test_indices": sorted(test_idx),
                "train_dates": dates[sorted(train_idx)] if train_idx else pd.DatetimeIndex([]),
                "test_dates": dates[sorted(test_idx)] if test_idx else pd.DatetimeIndex([]),
            })
        return paths

    def n_paths(self) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


class OverfitDetector:
    """Detect probability of backtest overfitting."""

    @staticmethod
    def detect_overfit(in_sample_results: List[float], out_of_sample_results: List[float]) -> Dict:
        """Estimate PBO and deflated Sharpe ratio."""
        is_arr = np.array(in_sample_results)
        oos_arr = np.array(out_of_sample_results)

        if len(is_arr) == 0 or len(oos_arr) == 0:
            return {"pbo": 0.5, "deflated_sharpe": 0, "is_overfit": True, "n_trials_adjustment": 0}

        if len(is_arr) == len(oos_arr) and len(is_arr) > 1:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(is_arr, oos_arr)
            pbo = max(0, (1 - corr) / 2)
        else:
            is_best_idx = np.argmax(is_arr)
            pbo = 1.0 if is_best_idx < len(oos_arr) and oos_arr[is_best_idx] <= 0 else 0.0

        n_trials = len(is_arr)
        expected_max = np.sqrt(2 * np.log(max(n_trials, 1)))
        deflated_sharpe = (max(is_arr) if len(is_arr) > 0 else 0) - expected_max * 0.5

        return {
            "pbo": float(np.clip(pbo, 0, 1)),
            "deflated_sharpe": float(deflated_sharpe),
            "is_overfit": pbo > 0.5 or deflated_sharpe < 0,
            "n_trials_adjustment": float(expected_max * 0.5),
        }


def optimize_with_early_stopping(
    param_space: ParameterSpace, evaluate_func: Callable,
    dates: pd.DatetimeIndex, objective: str = "sharpe",
    max_evals: int = 100, patience: int = 20,
    n_folds: int = 5, seed: Optional[int] = None,
) -> Dict:
    """Random search with early stopping."""
    rng = np.random.default_rng(seed)
    optimizer = WalkForwardOptimizer(config={}, objective=objective)
    folds = optimizer.generate_folds(dates, n_folds=n_folds)

    all_combos = param_space.combinations()
    n_eval = min(len(all_combos), max_evals)
    indices = rng.choice(len(all_combos), size=n_eval, replace=False).tolist() if len(all_combos) > max_evals else list(range(len(all_combos)))

    best_score, best_params, no_improve = float("-inf"), {}, 0

    for eval_idx, combo_idx in enumerate(indices):
        params = all_combos[combo_idx]
        fold_scores = []
        for fold in folds:
            try:
                test_returns = evaluate_func(params, (fold["train_start"], fold["train_end"]),
                                             (fold["test_start"], fold["test_end"]))
                fold_scores.append(optimizer.objective_func(test_returns))
            except Exception:
                fold_scores.append(float("-inf"))

        valid = [s for s in fold_scores if s != float("-inf")]
        avg = np.mean(valid) if valid else float("-inf")

        if avg > best_score:
            best_score, best_params, no_improve = avg, params, 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return {
        "best_params": best_params, "best_score": float(best_score),
        "n_evals": eval_idx + 1 if indices else 0,
        "early_stopped": no_improve >= patience,
    }

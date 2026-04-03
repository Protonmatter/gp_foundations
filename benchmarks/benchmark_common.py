from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gp_foundations.gp import GaussianProcessRegressor
from gp_foundations.kernels import MaternKernel
from gp_foundations.multioutput import CoregionalizationMatrix
from gp_foundations.wifi_research import JointStrategySimulator, StrategyObservation


def json_dump(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def summarize(values: list[float]) -> dict[str, float]:
    array = [float(value) for value in values]
    return {
        'mean': float(mean(array)),
        'median': float(median(array)),
        'min': float(min(array)),
        'max': float(max(array)),
    }


def synthetic_gp_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    *,
    true_length_scale: float = 0.18,
    true_variance: float = 1.3,
    noise: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_train = np.sort(rng.uniform(0.0, 1.0, size=n_train))[:, None]
    X_test = np.sort(rng.uniform(0.0, 1.0, size=n_test))[:, None]
    X_all = np.vstack([X_train, X_test])
    kernel = MaternKernel(length_scale=true_length_scale, variance=true_variance)
    covariance = kernel(X_all, X_all) + 1e-8 * np.eye(X_all.shape[0])
    latent = rng.multivariate_normal(np.zeros(X_all.shape[0]), covariance)
    y_train = latent[:n_train] + noise * rng.standard_normal(n_train)
    y_test_latent = latent[n_train:]
    return X_train, y_train, X_test, y_test_latent


def fit_fixed_and_learned_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    maxiter: int,
) -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    fixed = GaussianProcessRegressor(MaternKernel(length_scale=0.75, variance=0.3), noise=0.2).fit(X_train, y_train)
    learned = GaussianProcessRegressor(MaternKernel(length_scale=0.75, variance=0.3), noise=0.2).fit(X_train, y_train)
    learned.optimize_hyperparameters(maxiter=maxiter)
    return fixed, learned


def coverage_and_error(mean_values: np.ndarray, std_values: np.ndarray, truth: np.ndarray, level: float) -> tuple[float, float]:
    z_value = float(norm.ppf(0.5 + level / 2.0))
    covered = np.abs(truth - mean_values) <= z_value * std_values
    coverage = float(np.mean(covered))
    return coverage, float(abs(coverage - level))


def benchmark_timed(callable_obj: Callable[[], object]) -> float:
    start = perf_counter()
    callable_obj()
    return float(perf_counter() - start)


def synthetic_strategy_setup(
    grid_size: int,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Callable[[float], float]], list[StrategyObservation]]:
    grid = np.linspace(0.0, 1.0, grid_size)
    strategy_ids = ('strategy_a', 'strategy_b', 'strategy_c')
    rng = np.random.default_rng(seed)
    center_a = float(np.clip(0.70 + rng.normal(0.0, 0.03), 0.55, 0.85))
    center_b = float(np.clip(center_a + rng.normal(0.0, 0.02), 0.55, 0.85))
    center_c = float(np.clip(0.35 + rng.normal(0.0, 0.04), 0.15, 0.55))

    def reward_a(intensity: float) -> float:
        return 1.0 - 2.4 * (intensity - center_a) ** 2

    def reward_b(intensity: float) -> float:
        return 0.96 - 2.1 * (intensity - center_b) ** 2

    def reward_c(intensity: float) -> float:
        return 0.82 - 2.8 * (intensity - center_c) ** 2

    rewards = {
        'strategy_a': reward_a,
        'strategy_b': reward_b,
        'strategy_c': reward_c,
    }
    observations = [
        StrategyObservation('strategy_a', 0.20, reward_a(0.20)),
        StrategyObservation('strategy_b', 0.20, reward_b(0.20)),
        StrategyObservation('strategy_c', 0.20, reward_c(0.20)),
        StrategyObservation('strategy_a', 0.50, reward_a(0.50)),
        StrategyObservation('strategy_a', 0.70, reward_a(0.70)),
        StrategyObservation('strategy_b', 0.50, reward_b(0.50)),
        StrategyObservation('strategy_c', 0.50, reward_c(0.50)),
        StrategyObservation('strategy_a', 0.90, reward_a(0.90)),
    ]
    return grid, strategy_ids, rewards, observations


def build_fixed_and_learned_simulators(
    grid: np.ndarray,
    strategy_ids: tuple[str, ...],
    observations: list[StrategyObservation],
    *,
    optimization_maxiter: int,
) -> tuple[JointStrategySimulator, JointStrategySimulator]:
    common = {
        'strategy_ids': strategy_ids,
        'intensity_grid': grid,
        'kernel': MaternKernel(length_scale=1.2, variance=0.25),
        'coregionalization': CoregionalizationMatrix.identity(len(strategy_ids)),
        'noise': 0.2,
        'window_size': 50,
    }
    fixed = JointStrategySimulator(**common)
    learned = JointStrategySimulator(
        **common,
        enable_built_in_optimization=True,
        slow_interval=len(observations),
        optimization_maxiter=optimization_maxiter,
    )
    fixed.record_many(observations)
    learned.record_many(observations)
    return fixed, learned


def oracle_reward(grid: np.ndarray, rewards: dict[str, Callable[[float], float]]) -> tuple[str, float, float]:
    best_strategy = ''
    best_intensity = 0.0
    best_reward = -float('inf')
    for strategy_id, reward_fn in rewards.items():
        values = [float(reward_fn(float(x))) for x in grid]
        index = int(np.argmax(values))
        reward = values[index]
        if reward > best_reward:
            best_strategy = strategy_id
            best_intensity = float(grid[index])
            best_reward = float(reward)
    return best_strategy, best_intensity, best_reward

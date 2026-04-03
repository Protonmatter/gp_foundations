#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'benchmarks') not in sys.path:
    sys.path.insert(0, str(ROOT / 'benchmarks'))

if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from benchmark_common import (
    benchmark_timed,
    build_fixed_and_learned_simulators,
    json_dump,
    summarize,
    synthetic_gp_dataset,
    synthetic_strategy_setup,
)
from gp_foundations.gp import GaussianProcessRegressor
from gp_foundations.kernels import MaternKernel
from gp_foundations.multioutput import CoregionalizationMatrix
from gp_foundations.wifi_research import JointStrategySimulator


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark fixed-vs-learned runtime.')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--n-train', type=int, default=16)
    parser.add_argument('--n-test', type=int, default=32)
    parser.add_argument('--grid-size', type=int, default=21)
    parser.add_argument('--optimization-maxiter', type=int, default=35)
    args = parser.parse_args()

    gp_fixed: list[float] = []
    gp_learned: list[float] = []
    sim_fixed: list[float] = []
    sim_learned: list[float] = []

    for seed in range(args.trials):
        X_train, y_train, X_test, _ = synthetic_gp_dataset(seed, args.n_train, args.n_test)
        gp_fixed.append(
            benchmark_timed(
                lambda: GaussianProcessRegressor(
                    MaternKernel(length_scale=0.75, variance=0.3),
                    noise=0.2,
                ).fit(X_train, y_train).posterior(X_test)
            )
        )
        gp_learned.append(
            benchmark_timed(
                lambda: _fit_learned_gp(
                    X_train,
                    y_train,
                    X_test,
                    maxiter=args.optimization_maxiter,
                )
            )
        )

        grid, strategy_ids, _, observations = synthetic_strategy_setup(args.grid_size, seed=seed)
        sim_fixed.append(
            benchmark_timed(
                lambda: _build_fixed_simulator(
                    grid,
                    strategy_ids,
                    observations,
                ).recommend()
            )
        )
        sim_learned.append(
            benchmark_timed(
                lambda: build_fixed_and_learned_simulators(
                    grid,
                    strategy_ids,
                    observations,
                    optimization_maxiter=args.optimization_maxiter,
                )[1].recommend()
            )
        )

    json_dump(
        {
            'metric': 'runtime_seconds',
            'trials': args.trials,
            'gp_fixed': summarize(gp_fixed),
            'gp_learned': summarize(gp_learned),
            'simulator_fixed': summarize(sim_fixed),
            'simulator_learned': summarize(sim_learned),
            'gp_slowdown_factor': float(sum(gp_learned) / sum(gp_fixed)),
            'simulator_slowdown_factor': float(sum(sim_learned) / sum(sim_fixed)),
        }
    )


def _fit_learned_gp(
    X_train,
    y_train,
    X_test,
    *,
    maxiter: int,
):
    model = GaussianProcessRegressor(MaternKernel(length_scale=0.75, variance=0.3), noise=0.2).fit(X_train, y_train)
    model.optimize_hyperparameters(maxiter=maxiter)
    return model.posterior(X_test)


def _build_fixed_simulator(
    grid,
    strategy_ids,
    observations,
):
    simulator = JointStrategySimulator(
        strategy_ids=strategy_ids,
        intensity_grid=grid,
        kernel=MaternKernel(length_scale=1.2, variance=0.25),
        coregionalization=CoregionalizationMatrix.identity(len(strategy_ids)),
        noise=0.2,
        window_size=50,
    )
    simulator.record_many(observations)
    return simulator


if __name__ == '__main__':
    main()

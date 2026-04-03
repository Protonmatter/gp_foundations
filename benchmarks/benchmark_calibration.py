#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'benchmarks') not in sys.path:
    sys.path.insert(0, str(ROOT / 'benchmarks'))

from benchmark_common import coverage_and_error, fit_fixed_and_learned_gp, json_dump, summarize, synthetic_gp_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark fixed-vs-learned posterior calibration.')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--n-train', type=int, default=12)
    parser.add_argument('--n-test', type=int, default=40)
    parser.add_argument('--level', type=float, default=0.9)
    parser.add_argument('--optimization-maxiter', type=int, default=50)
    args = parser.parse_args()

    fixed_coverages: list[float] = []
    fixed_errors: list[float] = []
    learned_coverages: list[float] = []
    learned_errors: list[float] = []

    for seed in range(args.trials):
        X_train, y_train, X_test, y_test_latent = synthetic_gp_dataset(seed, args.n_train, args.n_test)
        fixed, learned = fit_fixed_and_learned_gp(X_train, y_train, maxiter=args.optimization_maxiter)
        fixed_posterior = fixed.posterior(X_test)
        learned_posterior = learned.posterior(X_test)

        fixed_coverage, fixed_error = coverage_and_error(fixed_posterior.mean, fixed_posterior.std, y_test_latent, args.level)
        learned_coverage, learned_error = coverage_and_error(learned_posterior.mean, learned_posterior.std, y_test_latent, args.level)
        fixed_coverages.append(fixed_coverage)
        fixed_errors.append(fixed_error)
        learned_coverages.append(learned_coverage)
        learned_errors.append(learned_error)

    json_dump(
        {
            'metric': 'calibration',
            'trials': args.trials,
            'level': args.level,
            'fixed': {
                'coverage': summarize(fixed_coverages),
                'calibration_error': summarize(fixed_errors),
            },
            'learned': {
                'coverage': summarize(learned_coverages),
                'calibration_error': summarize(learned_errors),
            },
        }
    )


if __name__ == '__main__':
    main()

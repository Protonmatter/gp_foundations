#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'benchmarks') not in sys.path:
    sys.path.insert(0, str(ROOT / 'benchmarks'))

from benchmark_common import build_fixed_and_learned_simulators, json_dump, oracle_reward, summarize, synthetic_strategy_setup


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark fixed-vs-learned strategy regret.')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--grid-size', type=int, default=21)
    parser.add_argument('--optimization-maxiter', type=int, default=35)
    args = parser.parse_args()

    fixed_regrets: list[float] = []
    learned_regrets: list[float] = []
    for trial in range(args.trials):
        grid, strategy_ids, rewards, observations = synthetic_strategy_setup(args.grid_size, seed=trial)
        fixed, learned = build_fixed_and_learned_simulators(
            grid,
            strategy_ids,
            observations,
            optimization_maxiter=args.optimization_maxiter,
        )
        _, _, oracle = oracle_reward(grid, rewards)
        fixed_choice = fixed.recommend()
        learned_choice = learned.recommend()
        fixed_regrets.append(float(oracle - rewards[fixed_choice.strategy_id](fixed_choice.intensity)))
        learned_regrets.append(float(oracle - rewards[learned_choice.strategy_id](learned_choice.intensity)))

    json_dump(
        {
            'metric': 'regret',
            'trials': args.trials,
            'fixed': summarize(fixed_regrets),
            'learned': summarize(learned_regrets),
            'improvement': float(sum(fixed_regrets) - sum(learned_regrets)),
        }
    )


if __name__ == '__main__':
    main()

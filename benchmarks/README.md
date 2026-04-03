# Benchmarks

These scripts compare fixed-parameter and learned-parameter behavior for the current exact dense GP stack.

## Scripts

- `benchmark_regret.py`: offline replay regret for the strategy simulator
- `benchmark_calibration.py`: posterior interval coverage and calibration error for the single-output GP
- `benchmark_runtime.py`: wall-clock runtime comparison for fixed vs learned GP and simulator flows

## Examples

```bash
python benchmarks/benchmark_regret.py --trials 20 --grid-size 21
python benchmarks/benchmark_calibration.py --trials 20 --n-train 12 --n-test 40
python benchmarks/benchmark_runtime.py --trials 10 --n-train 16 --n-test 32 --grid-size 21
```

All scripts print JSON to stdout so the output can be redirected into files or post-processed later.

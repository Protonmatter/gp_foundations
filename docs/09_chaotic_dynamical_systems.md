# Chaotic Dynamical Systems and the Logistic Map

Source PDF: `references/pdfs/09_chaotic_dynamical_systems.pdf`

## Repository mapping

- Package module: `gp_foundations.systems`
- Notebook: `notebooks/05_kalman_and_chaos.ipynb`

## Core ideas

The PDF uses the logistic map as a compact example of deterministic chaos and as a practical sequence generator for timing jitter or pseudo-randomized schedules.

## Implemented surface

- `logistic_map(x, rate=4.0)`
- `chaotic_timing_sequence(seed, length, rate, low, high, discard)`

The implementation prioritizes bounded, reproducible sequences suitable for experiments and lightweight scheduling demos.

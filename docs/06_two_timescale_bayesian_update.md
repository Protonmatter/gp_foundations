# The Two-Timescale Bayesian Update Problem

Source PDF: `references/pdfs/06_two_timescale_bayesian_update.pdf`

## Repository mapping

- Package module: `gp_foundations.updates`
- Notebook: `notebooks/04_nonstationary_and_updates.ipynb`

## Core ideas

The PDF separates fast posterior updates from slower hyperparameter or structure updates. The repository encodes that idea as a general scheduling utility rather than a GP-specific optimizer.

## Implemented surface

- `TwoTimescaleUpdater.update(observation)`: always runs the fast path
- slower updates run on either:
  - a fixed interval schedule
  - a custom callable schedule
  - a logarithmic schedule produced by `TwoTimescaleUpdater.logarithmic_schedule`

This design keeps the scheduler reusable across GP and non-GP experiments.

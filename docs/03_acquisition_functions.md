# Acquisition Functions in Bayesian Optimization

Source PDF: `references/pdfs/03_acquisition_functions.pdf`

## Repository mapping

- Package module: `gp_foundations.acquisition`
- Notebook: `notebooks/02_acquisition_and_bo.ipynb`

## Core ideas

The acquisition layer converts posterior beliefs into search decisions.

- Upper confidence bound balances mean and uncertainty with a single optimism parameter.
- Expected improvement scores candidates by expected gain over the current best value.
- Thompson sampling draws one coherent posterior sample and optimizes that sample.

## Implemented surface

- `upper_confidence_bound(mean, std, beta)`
- `expected_improvement(mean, std, best, xi)`
- `thompson_sample(mean, covariance)`
- `softmax_select(scores, temperature)` for discrete arm selection experiments

# gp_foundations

`gp_foundations` is a research repository built from a curated PDF corpus on Gaussian processes, Bayesian optimization, numerical linear algebra, multi-output inference, non-stationary modelling, two-timescale updates, Kalman filtering, chaos, and concurrency.

The repository is package-first: the Python toolkit under `src/gp_foundations/` is the main executable surface, while `docs/` and `notebooks/` mirror the source papers in a repo-native form.

## Status

This is an early-stage research toolkit. The repository is structured and tested, but the priority is coherent scientific code and documentation rather than release packaging.

## Scope

Phase one implements:

- Gaussian process regression with marginal likelihood evaluation
- Opt-in hyperparameter learning for exact Matérn GP and multi-output GP models
- Cholesky-based numerical helpers
- Bayesian optimization acquisition functions
- Intrinsic coregionalization for multi-output Gaussian processes
- Practical non-stationary kernels and a two-timescale update scheduler
- Offline WiFi replay and simulation utilities built on the shared MOGP stack
- Supporting Kalman, chaos, and concurrency utilities

## Layout

- `src/gp_foundations/`: package code
- `src/gp_foundations/wifi_research/`: offline replay and simulation layer
- `tests/`: unit and smoke tests
- `docs/`: curated Markdown documents derived from the PDFs
- `notebooks/`: notebook walkthroughs that mirror the documents
- `references/pdfs/`: original source PDFs copied into the repository

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,plotting]
python -m pytest
```

If `pytest` is not installed locally, the test suite can still run via the standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Example

```python
import numpy as np
from gp_foundations.gp import GaussianProcessRegressor
from gp_foundations.kernels import MaternKernel

X = np.linspace(0.0, 1.0, 8)[:, None]
y = np.sin(2.0 * np.pi * X[:, 0])

gp = GaussianProcessRegressor(MaternKernel(length_scale=0.2), noise=1e-4)
gp.fit(X, y)
posterior = gp.posterior(np.linspace(0.0, 1.0, 100)[:, None])
```

## License

This repository is released under the MIT License. See `LICENSE`.

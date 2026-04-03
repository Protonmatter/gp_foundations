# Non-Stationary Gaussian Processes and Spectral Methods

Source PDF: `references/pdfs/05_non_stationary_gp_spectral.pdf`

## Repository mapping

- Package module: `gp_foundations.kernels`
- Notebook: `notebooks/04_nonstationary_and_updates.ipynb`

## Core ideas

The PDF distinguishes between stationary kernels, which depend only on input differences, and richer models whose behavior changes across the domain.

Phase one implements two practical entry points:

- `SpectralMixtureKernel` for flexible stationary structure through weighted spectral components
- `TimeDecayKernel` for a simple non-stationary approximation that downweights covariance as inputs move away from a reference region

This keeps the repository useful immediately while leaving room for deeper locally stationary or spatio-temporal constructions later.

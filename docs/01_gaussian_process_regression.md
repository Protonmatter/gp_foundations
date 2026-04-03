# Gaussian Process Regression

Source PDF: `references/pdfs/01_gaussian_process_regression.pdf`

## Repository mapping

- Package modules: `gp_foundations.gp`, `gp_foundations.kernels`
- Notebook: `notebooks/01_gp_regression.ipynb`

## Core ideas

A Gaussian process treats the target function as the random object. The model is fully determined by a mean function and a kernel, and inference reduces to conditioning a joint Gaussian distribution on observed values.

The repository implements the standard regression pipeline:

- prior covariance through `MaternKernel`
- posterior mean and covariance through `GaussianProcessRegressor.posterior`
- marginal likelihood scoring through `GaussianProcessRegressor.log_marginal_likelihood`

## Key equations

For training inputs `X`, targets `y`, and test inputs `X*`:

- `K = k(X, X) + sigma_n^2 I`
- `mu* = K_*^T K^-1 y`
- `Sigma* = K_** - K_*^T K^-1 K_*`

The implementation avoids explicit matrix inversion and routes all solves through the Cholesky helpers described in the next document.

# Cholesky Decomposition and Numerical Linear Algebra

Source PDF: `references/pdfs/02_cholesky_decomposition.pdf`

## Repository mapping

- Package module: `gp_foundations.linalg`
- Notebook: `notebooks/01_gp_regression.ipynb`

## Core ideas

The Cholesky factorization writes a symmetric positive definite matrix as `A = L L^T`. In GP work this gives three immediate benefits:

- stable linear solves without forming `A^-1`
- cheap log determinants from the diagonal of `L`
- a natural place to add jitter when kernels are numerically close to singular

## Implemented surface

- `ensure_spd`: symmetrize and add adaptive jitter until the matrix is usable
- `cholesky_factor`: produce `L`
- `cholesky_solve`: solve `A x = b` via forward and backward substitution
- `stable_logdet`: compute `log det(A)` from `L`

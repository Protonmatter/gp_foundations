# Multi-Output Gaussian Processes and Coregionalization

Source PDF: `references/pdfs/04_multi_output_gp_coregionalization.pdf`

## Repository mapping

- Package module: `gp_foundations.multioutput`
- Notebook: `notebooks/03_multioutput_coregionalization.ipynb`

## Core ideas

Multi-output GPs let related tasks borrow strength from one another instead of learning independently. This repository implements the intrinsic coregionalization model (ICM), where the joint covariance factors into:

- an input kernel over `x`
- a coregionalization matrix `B` over output indices

## Implemented surface

- `CoregionalizationMatrix`: PSD-preserving parameterization via lower-triangular factors
- `IntrinsicCoregionalizedGP.fit`: dense multi-output observations
- `IntrinsicCoregionalizedGP.fit_observations`: sparse task-indexed observations
- `IntrinsicCoregionalizedGP.joint_thompson_sample`: coherent multi-task sampling from the posterior

## Practical note

The implementation is intentionally dense and explicit rather than optimized for very large workloads. The current objective is clarity and correctness for research-scale experiments.

# Mathematical Foundations Index

This repository is organized around a ten-document PDF corpus copied into `references/pdfs/`. The PDFs remain the source references, but the Markdown pages here are the canonical editable docs for the repository.

## Reference map

| Topic | PDF | Markdown | Package surface | Notebook |
| --- | --- | --- | --- | --- |
| Index | `00_index.pdf` | `index.md` | package-wide | n/a |
| Gaussian process regression | `01_gaussian_process_regression.pdf` | [01 Gaussian Process Regression](01_gaussian_process_regression.md) | `gp`, `kernels` | [01 GP Regression](../notebooks/01_gp_regression.ipynb) |
| Cholesky decomposition | `02_cholesky_decomposition.pdf` | [02 Cholesky Decomposition](02_cholesky_decomposition.md) | `linalg` | [01 GP Regression](../notebooks/01_gp_regression.ipynb) |
| Acquisition functions | `03_acquisition_functions.pdf` | [03 Acquisition Functions](03_acquisition_functions.md) | `acquisition` | [02 Acquisition and BO](../notebooks/02_acquisition_and_bo.ipynb) |
| Multi-output GP coregionalization | `04_multi_output_gp_coregionalization.pdf` | [04 Multi-Output GP](04_multi_output_gp_coregionalization.md) | `multioutput` | [03 Multi-Output Coregionalization](../notebooks/03_multioutput_coregionalization.ipynb) |
| Non-stationary GP spectral methods | `05_non_stationary_gp_spectral.pdf` | [05 Non-Stationary GP](05_non_stationary_gp_spectral.md) | `kernels` | [04 Non-Stationary and Updates](../notebooks/04_nonstationary_and_updates.ipynb) |
| Two-timescale Bayesian update | `06_two_timescale_bayesian_update.pdf` | [06 Two-Timescale Updates](06_two_timescale_bayesian_update.md) | `updates` | [04 Non-Stationary and Updates](../notebooks/04_nonstationary_and_updates.ipynb) |
| Concurrency design patterns | `07_concurrency_design_patterns.pdf` | [07 Concurrency Design Patterns](07_concurrency_design_patterns.md) | `runtime` | [06 Concurrency Patterns](../notebooks/06_concurrency_patterns.ipynb) |
| Kalman filter | `08_kalman_filter.pdf` | [08 Kalman Filter](08_kalman_filter.md) | `filters` | [05 Kalman and Chaos](../notebooks/05_kalman_and_chaos.ipynb) |
| Chaotic dynamical systems | `09_chaotic_dynamical_systems.pdf` | [09 Chaotic Dynamical Systems](09_chaotic_dynamical_systems.md) | `systems` | [05 Kalman and Chaos](../notebooks/05_kalman_and_chaos.ipynb) |

## Theme summary

The corpus forms a coherent stack:

- Gaussian processes provide the probabilistic model class.
- Cholesky decomposition provides the stable computational backbone.
- Acquisition functions turn GP posteriors into sequential decision rules.
- Coregionalization extends single-output inference to related tasks.
- Non-stationary kernels and two-timescale updates handle changing structure over time.
- Kalman filtering and chaos utilities give lighter-weight state and sequence tools.
- Concurrency patterns make the scientific runtime composable and testable.

## Research addenda

The repository also includes a safe offline WiFi research layer derived from legacy MOGP source material:

- [10 WiFi Research Source Mapping](10_wifi_research_source_mapping.md)
- [11 Offline WiFi Research Workflow](11_wifi_research_workflow.md)
- [12 Trainable GP Models](12_trainable_gp_models.md)
- [13 Sparse and Approximate GP Scaling](13_sparse_approximate_gp_scaling.md)
- [07 WiFi Research Simulation Notebook](../notebooks/07_wifi_research_simulation.ipynb)
- [08 Trainable Models Notebook](../notebooks/08_trainable_models.ipynb)
- [Benchmark Scripts](../benchmarks/README.md)

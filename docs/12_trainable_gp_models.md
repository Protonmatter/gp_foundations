# Trainable GP and MOGP Models

The repository now supports explicit likelihood-based hyperparameter learning for the exact dense `MaternKernel` GP and multi-output GP paths.

## Supported trainable parameters

Single-output `GaussianProcessRegressor`:
- Matérn length scale
- Matérn variance
- observation noise

Multi-output `IntrinsicCoregionalizedGP`:
- shared Matérn length scale
- shared Matérn variance
- observation noise
- coregionalization factor through an unconstrained lower-triangular parameterization with exponentiated diagonal

## API surface

```python
from gp_foundations.gp import GaussianProcessRegressor
from gp_foundations.kernels import MaternKernel

model = GaussianProcessRegressor(MaternKernel(length_scale=0.6, variance=0.4), noise=0.1)
model.fit(X, y)
initial = model.negative_log_marginal_likelihood()
result = model.optimize_hyperparameters(maxiter=100)
```

Both the single-output and multi-output optimizers return an `OptimizationResult` with:
- `success`
- `objective`
- `iterations`
- `parameters`
- `message`

## Parameterization and bounds

Optimization runs in log-parameter space with `scipy.optimize.minimize(..., method="L-BFGS-B")`.

Default bounds:
- length scale: `log(1e-3)` to `log(10)`
- kernel variance: `log(1e-4)` to `log(10)`
- noise: `log(1e-8)` to `log(1)`
- multi-output coregionalization off-diagonals: `[-5, 5]`
- multi-output coregionalization diagonals: `log(1e-6)` to `log(10)` after reconstruction

The multi-output factor is reconstructed with `CoregionalizationMatrix.from_unconstrained(...)`, which preserves positive semidefiniteness.

## Failure handling

- Optimization is explicit and opt-in; `fit()` behavior is unchanged.
- The model is only updated when the optimizer produces a finite objective that is no worse than the current objective.
- Exact dense Cholesky inference is still used throughout, with jitter-based stabilization already present in the repository.
- Only scalar `MaternKernel` hyperparameters are trainable in this milestone. `SpectralMixtureKernel` and `TimeDecayKernel` remain fixed-parameter.

## Replay integration

`JointStrategySimulator` now supports an opt-in built-in slow-update path:

```python
simulator = JointStrategySimulator(
    strategy_ids,
    intensity_grid,
    enable_built_in_optimization=True,
    slow_interval=4,
    optimization_maxiter=50,
)
```

When enabled, the simulator re-optimizes the shared multi-output model from the current replay window on the slow-update cadence. Live network behavior is still out of scope.

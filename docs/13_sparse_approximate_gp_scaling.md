# Sparse and Approximate GP Scaling Milestone

This document captures the next modeling milestone after the trainable exact dense GP work.

## Goal

Extend `gp_foundations` beyond exact dense Cholesky inference so larger replay windows and larger synthetic studies become practical without changing the current exact APIs by default.

## Proposed scope

- Add an additive sparse GP path for the single-output model using inducing points
- Add an additive sparse multi-output path that preserves the current coregionalization parameterization
- Keep the exact dense implementations as the default reference path for correctness and small research workloads
- Reuse the existing benchmark scripts to compare approximation quality, regret, calibration, and runtime against the exact learned models

## Candidate interfaces

Single-output:
- `SparseGaussianProcessRegressor`
- configurable inducing points or inducing-point count
- exact-like posterior API: `fit`, `posterior`, `predict`, `sample_posterior`

Multi-output:
- `SparseIntrinsicCoregionalizedGP`
- shared inducing grid across outputs in the first milestone
- exact-like posterior and joint-sampling API where feasible

Simulator integration:
- opt-in sparse backend selection on `JointStrategySimulator`
- replay-window size thresholds where sparse learning becomes the recommended mode

## Initial technical direction

- Start with inducing-point variational or FITC-style approximations rather than streaming-only approximations
- Restrict the first pass to scalar-input Matérn kernels, matching the current trainable exact path
- Preserve PSD guarantees for the coregionalization component and keep jitter-based stabilization throughout
- Add benchmark targets before claiming the sparse path is ready:
  - runtime reduction versus exact dense learned models
  - bounded degradation in posterior calibration
  - bounded degradation in simulator regret

## Acceptance targets

- Sparse single-output GP runs materially faster than the exact learned GP on larger synthetic datasets
- Sparse multi-output GP runs materially faster than the exact learned MOGP on larger replay windows
- Calibration and regret remain close enough to the exact learned baselines to be useful for internal research screening
- Existing exact tests and notebooks continue to pass unchanged

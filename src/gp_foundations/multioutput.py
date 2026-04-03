from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .acquisition import thompson_sample
from .gp import OptimizationResult
from .kernels import MaternKernel
from .linalg import DEFAULT_JITTER, cholesky_factor, cholesky_solve, stable_logdet


@dataclass
class MultiOutputPosterior:
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray | None = None


@dataclass
class OutputPosterior:
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray | None = None


@dataclass
class JointGridEvaluation:
    points: np.ndarray
    posterior: MultiOutputPosterior
    sample: np.ndarray


class CoregionalizationMatrix:
    def __init__(self, factor: np.ndarray):
        lower = np.asarray(factor, dtype=float)
        if lower.ndim != 2 or lower.shape[0] != lower.shape[1]:
            raise ValueError("factor must be square")
        self._factor = np.tril(lower)

    @classmethod
    def identity(cls, n_outputs: int, scale: float = 1.0) -> "CoregionalizationMatrix":
        if n_outputs <= 0:
            raise ValueError("n_outputs must be positive")
        return cls(np.sqrt(scale) * np.eye(n_outputs))

    @classmethod
    def from_factor(cls, factor: np.ndarray) -> "CoregionalizationMatrix":
        return cls(factor)

    @classmethod
    def from_unconstrained(cls, raw_factor: np.ndarray, diag_floor: float = 1e-6) -> "CoregionalizationMatrix":
        raw = np.asarray(raw_factor, dtype=float)
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
            raise ValueError("raw_factor must be square")
        lower = np.tril(raw, k=-1)
        diag = np.exp(np.diag(raw)) + diag_floor
        return cls(lower + np.diag(diag))

    @property
    def n_outputs(self) -> int:
        return self._factor.shape[0]

    @property
    def factor(self) -> np.ndarray:
        return self._factor.copy()

    @property
    def matrix(self) -> np.ndarray:
        return self._factor @ self._factor.T


class IntrinsicCoregionalizedGP:
    def __init__(self, kernel: object, coregionalization: CoregionalizationMatrix, noise: float = 1e-6):
        self.kernel = kernel
        self.coregionalization = coregionalization
        self.noise = float(noise)
        self.X_obs: np.ndarray | None = None
        self.y_obs: np.ndarray | None = None
        self.output_indices: np.ndarray | None = None
        self._factor: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    def _require_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.X_obs is None or self.y_obs is None or self.output_indices is None:
            raise RuntimeError("fit the multi-output GP before using optimization utilities")
        return self.X_obs, self.y_obs, self.output_indices

    @staticmethod
    def _as_trainable_matern(kernel: object) -> MaternKernel:
        if not isinstance(kernel, MaternKernel):
            raise TypeError("hyperparameter optimization currently supports MaternKernel only")
        length_scale = np.asarray(kernel.length_scale, dtype=float)
        if length_scale.ndim != 0:
            raise TypeError("hyperparameter optimization currently supports scalar Matern length scales only")
        return kernel

    @staticmethod
    def _factor_to_unconstrained(
        factor: np.ndarray,
        diag_floor: float,
    ) -> np.ndarray:
        lower = np.tril(np.asarray(factor, dtype=float))
        raw = np.zeros_like(lower)
        diag = np.maximum(np.diag(lower) - diag_floor, 1e-12)
        np.fill_diagonal(raw, np.log(diag))
        lower_index = np.tril_indices(lower.shape[0], k=-1)
        raw[lower_index] = lower[lower_index]
        return raw

    @staticmethod
    def _parameter_bounds(
        n_outputs: int,
        optimize_noise: bool,
    ) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = [
            (np.log(1e-3), np.log(10.0)),
            (np.log(1e-4), np.log(10.0)),
        ]
        if optimize_noise:
            bounds.append((np.log(1e-8), np.log(1.0)))
        lower_index = np.tril_indices(n_outputs)
        for row, col in zip(*lower_index):
            bounds.append((np.log(1e-6), np.log(10.0)) if row == col else (-5.0, 5.0))
        return bounds

    @staticmethod
    def _pack_log_parameters(
        kernel: MaternKernel,
        coregionalization: CoregionalizationMatrix,
        noise: float,
        optimize_noise: bool,
        diag_floor: float,
    ) -> np.ndarray:
        values: list[float] = [
            np.log(float(kernel.length_scale)),
            np.log(float(kernel.variance)),
        ]
        if optimize_noise:
            values.append(np.log(float(noise)))
        raw = IntrinsicCoregionalizedGP._factor_to_unconstrained(coregionalization.factor, diag_floor)
        values.extend(raw[np.tril_indices(coregionalization.n_outputs)].tolist())
        return np.asarray(values, dtype=float)

    @staticmethod
    def _unpack_log_parameters(
        params: np.ndarray,
        base_kernel: MaternKernel,
        base_noise: float,
        n_outputs: int,
        optimize_noise: bool,
        diag_floor: float,
    ) -> tuple[MaternKernel, CoregionalizationMatrix, float]:
        values = np.asarray(params, dtype=float).reshape(-1)
        prefix = 3 if optimize_noise else 2
        raw_values = values[prefix:]
        expected_raw = n_outputs * (n_outputs + 1) // 2
        if raw_values.size != expected_raw:
            raise ValueError("unexpected optimization parameter shape")

        kernel = replace(
            base_kernel,
            length_scale=float(np.exp(values[0])),
            variance=float(np.exp(values[1])),
        )
        noise = float(np.exp(values[2])) if optimize_noise else float(base_noise)

        raw_factor = np.zeros((n_outputs, n_outputs), dtype=float)
        raw_factor[np.tril_indices(n_outputs)] = raw_values
        coregionalization = CoregionalizationMatrix.from_unconstrained(raw_factor, diag_floor=diag_floor)
        return kernel, coregionalization, noise

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "IntrinsicCoregionalizedGP":
        X_array = np.asarray(X, dtype=float)
        Y_array = np.asarray(Y, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array[:, None]
        if Y_array.ndim != 2:
            raise ValueError("Y must be a 2D array")
        if X_array.shape[0] != Y_array.shape[0]:
            raise ValueError("X and Y must agree on sample count")
        if Y_array.shape[1] != self.coregionalization.n_outputs:
            raise ValueError("Y must match the number of outputs")

        repeated_X = np.repeat(X_array, Y_array.shape[1], axis=0)
        output_indices = np.tile(np.arange(Y_array.shape[1]), X_array.shape[0])
        y_obs = Y_array.reshape(-1)
        return self.fit_observations(repeated_X, y_obs, output_indices)

    def fit_observations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_indices: np.ndarray,
    ) -> "IntrinsicCoregionalizedGP":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        task_array = np.asarray(output_indices, dtype=int).reshape(-1)
        if X_array.ndim == 1:
            X_array = X_array[:, None]
        if not (X_array.shape[0] == y_array.shape[0] == task_array.shape[0]):
            raise ValueError("X, y, and output_indices must have the same length")
        if np.any(task_array < 0) or np.any(task_array >= self.coregionalization.n_outputs):
            raise ValueError("output_indices out of range")

        self.X_obs = X_array
        self.y_obs = y_array
        self.output_indices = task_array

        train_kernel = self._kernel_between(self.X_obs, self.output_indices, self.X_obs, self.output_indices)
        train_kernel = train_kernel + self.noise * np.eye(self.X_obs.shape[0])
        self._factor = cholesky_factor(train_kernel, jitter=DEFAULT_JITTER)
        self._alpha = cholesky_solve(self._factor, self.y_obs)
        return self

    def negative_log_marginal_likelihood(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        output_indices: np.ndarray | None = None,
        *,
        kernel: object | None = None,
        coregionalization: CoregionalizationMatrix | None = None,
        noise: float | None = None,
    ) -> float:
        if X is None or y is None or output_indices is None:
            X_array, y_array, task_array = self._require_training_data()
        else:
            X_array = np.asarray(X, dtype=float)
            y_array = np.asarray(y, dtype=float).reshape(-1)
            task_array = np.asarray(output_indices, dtype=int).reshape(-1)
            if X_array.ndim == 1:
                X_array = X_array[:, None]
            if not (X_array.shape[0] == y_array.shape[0] == task_array.shape[0]):
                raise ValueError("X, y, and output_indices must have the same length")

        kernel_object = self.kernel if kernel is None else kernel
        coreg = self.coregionalization if coregionalization is None else coregionalization
        noise_value = self.noise if noise is None else float(noise)
        if noise_value < 0.0:
            raise ValueError("noise must be non-negative")

        train_kernel = np.asarray(
            self._kernel_between(X_array, task_array, X_array, task_array),
            dtype=float,
        )
        if kernel is not None or coregionalization is not None:
            input_covariance = np.asarray(kernel_object(X_array, X_array), dtype=float)
            task_covariance = coreg.matrix[np.ix_(task_array, task_array)]
            train_kernel = input_covariance * task_covariance
        train_kernel = train_kernel + noise_value * np.eye(X_array.shape[0])
        factor = cholesky_factor(train_kernel, jitter=DEFAULT_JITTER)
        alpha = cholesky_solve(factor, y_array)
        n = X_array.shape[0]
        return float(
            0.5 * y_array @ alpha
            + 0.5 * stable_logdet(factor=factor)
            + 0.5 * n * np.log(2.0 * np.pi)
        )

    def optimize_hyperparameters(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        output_indices: np.ndarray | None = None,
        *,
        optimize_noise: bool = True,
        maxiter: int = 100,
        bounds: list[tuple[float, float]] | None = None,
        diag_floor: float = 1e-6,
    ) -> OptimizationResult:
        if X is None or y is None or output_indices is None:
            X_array, y_array, task_array = self._require_training_data()
        else:
            X_array = np.asarray(X, dtype=float)
            y_array = np.asarray(y, dtype=float).reshape(-1)
            task_array = np.asarray(output_indices, dtype=int).reshape(-1)
            if X_array.ndim == 1:
                X_array = X_array[:, None]
            if not (X_array.shape[0] == y_array.shape[0] == task_array.shape[0]):
                raise ValueError("X, y, and output_indices must have the same length")

        trainable_kernel = self._as_trainable_matern(self.kernel)
        initial_objective = self.negative_log_marginal_likelihood(
            X_array,
            y_array,
            task_array,
            kernel=trainable_kernel,
            coregionalization=self.coregionalization,
            noise=self.noise,
        )
        x0 = self._pack_log_parameters(
            trainable_kernel,
            self.coregionalization,
            self.noise,
            optimize_noise,
            diag_floor,
        )
        parameter_bounds = self._parameter_bounds(self.coregionalization.n_outputs, optimize_noise) if bounds is None else bounds

        def _objective(params: np.ndarray) -> float:
            candidate_kernel, candidate_coreg, candidate_noise = self._unpack_log_parameters(
                params,
                trainable_kernel,
                self.noise,
                self.coregionalization.n_outputs,
                optimize_noise,
                diag_floor,
            )
            return self.negative_log_marginal_likelihood(
                X_array,
                y_array,
                task_array,
                kernel=candidate_kernel,
                coregionalization=candidate_coreg,
                noise=candidate_noise,
            )

        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=parameter_bounds,
            options={"maxiter": int(maxiter)},
        )
        candidate_kernel, candidate_coreg, candidate_noise = self._unpack_log_parameters(
            result.x,
            trainable_kernel,
            self.noise,
            self.coregionalization.n_outputs,
            optimize_noise,
            diag_floor,
        )
        candidate_objective = self.negative_log_marginal_likelihood(
            X_array,
            y_array,
            task_array,
            kernel=candidate_kernel,
            coregionalization=candidate_coreg,
            noise=candidate_noise,
        )
        if np.isfinite(candidate_objective) and candidate_objective <= initial_objective:
            self.kernel = candidate_kernel
            self.coregionalization = candidate_coreg
            self.noise = candidate_noise
            self.fit_observations(X_array, y_array, task_array)

        return OptimizationResult(
            success=bool(result.success),
            objective=float(candidate_objective),
            iterations=int(getattr(result, "nit", 0)),
            parameters={
                "kernel": self.kernel,
                "coregionalization": self.coregionalization,
                "noise": float(self.noise),
            },
            message=str(result.message),
        )

    def _prediction_grid(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array[:, None]
        n_test = X_array.shape[0]
        tiled_X = np.tile(X_array, (self.coregionalization.n_outputs, 1))
        tiled_tasks = np.repeat(np.arange(self.coregionalization.n_outputs), n_test)
        return tiled_X, tiled_tasks

    def _kernel_between(
        self,
        X_left: np.ndarray,
        task_left: np.ndarray,
        X_right: np.ndarray,
        task_right: np.ndarray,
    ) -> np.ndarray:
        input_covariance = np.asarray(self.kernel(X_left, X_right), dtype=float)
        task_covariance = self.coregionalization.matrix[np.ix_(task_left, task_right)]
        return input_covariance * task_covariance

    def posterior(self, X: np.ndarray, return_covariance: bool = True) -> MultiOutputPosterior:
        X_test = np.asarray(X, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        test_X, test_tasks = self._prediction_grid(X_test)
        prior_covariance = self._kernel_between(test_X, test_tasks, test_X, test_tasks)

        if self.X_obs is None or self.y_obs is None or self.output_indices is None or self._factor is None or self._alpha is None:
            mean = np.zeros((X_test.shape[0], self.coregionalization.n_outputs), dtype=float)
            variance = np.clip(np.diag(prior_covariance), 0.0, None).reshape(self.coregionalization.n_outputs, X_test.shape[0]).T
            covariance = prior_covariance if return_covariance else None
            return MultiOutputPosterior(mean=mean, variance=variance, covariance=covariance)

        cross_covariance = self._kernel_between(self.X_obs, self.output_indices, test_X, test_tasks)
        mean_vector = cross_covariance.T @ self._alpha
        projected = np.linalg.solve(self._factor, cross_covariance)
        covariance = prior_covariance - projected.T @ projected
        covariance = 0.5 * (covariance + covariance.T)
        variance = np.clip(np.diag(covariance), 0.0, None)
        mean = mean_vector.reshape(self.coregionalization.n_outputs, X_test.shape[0]).T
        variance_matrix = variance.reshape(self.coregionalization.n_outputs, X_test.shape[0]).T
        if not return_covariance:
            covariance = None
        return MultiOutputPosterior(mean=mean, variance=variance_matrix, covariance=covariance)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.posterior(X, return_covariance=False).mean

    def posterior_for_output(
        self,
        X: np.ndarray,
        output_index: int,
        return_covariance: bool = False,
    ) -> OutputPosterior:
        if output_index < 0 or output_index >= self.coregionalization.n_outputs:
            raise ValueError("output_index out of range")
        X_test = np.asarray(X, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        posterior = self.posterior(X_test, return_covariance=return_covariance)
        covariance = None
        if return_covariance and posterior.covariance is not None:
            n_test = X_test.shape[0]
            start = output_index * n_test
            end = (output_index + 1) * n_test
            covariance = posterior.covariance[start:end, start:end]
        return OutputPosterior(
            mean=posterior.mean[:, output_index],
            variance=posterior.variance[:, output_index],
            covariance=covariance,
        )

    def joint_thompson_sample(
        self,
        X: np.ndarray,
        rng: np.random.Generator | None = None,
        n_samples: int = 1,
    ) -> np.ndarray:
        X_test = np.asarray(X, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        posterior = self.posterior(X_test, return_covariance=True)
        draws = thompson_sample(
            posterior.mean.T.reshape(-1),
            posterior.covariance,
            rng=rng,
            n_samples=n_samples,
        )
        if n_samples == 1:
            return np.asarray(draws, dtype=float).reshape(self.coregionalization.n_outputs, X_test.shape[0]).T
        stacked = np.asarray(draws, dtype=float)
        return stacked.reshape(n_samples, self.coregionalization.n_outputs, X_test.shape[0]).transpose(0, 2, 1)

    def evaluate_grid(
        self,
        X: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> JointGridEvaluation:
        X_test = np.asarray(X, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test[:, None]
        posterior = self.posterior(X_test, return_covariance=True)
        sample = self.joint_thompson_sample(X_test, rng=rng)
        return JointGridEvaluation(points=X_test, posterior=posterior, sample=sample)

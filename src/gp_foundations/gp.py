from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .kernels import MaternKernel
from .linalg import DEFAULT_JITTER, cholesky_factor, cholesky_solve, stable_logdet


@dataclass
class PosteriorPrediction:
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray | None = None

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))


@dataclass
class OptimizationResult:
    success: bool
    objective: float
    iterations: int
    parameters: dict[str, object]
    message: str = ""


class GaussianProcessRegressor:
    def __init__(self, kernel: object, noise: float = 1e-6):
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        self.kernel = kernel
        self.noise = float(noise)
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._factor: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if X_array.ndim == 1:
            X_array = X_array[:, None]
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must agree on sample count")

        self.X_train = X_array
        self.y_train = y_array
        kernel_matrix = np.asarray(self.kernel(self.X_train, self.X_train), dtype=float)
        kernel_matrix = kernel_matrix + self.noise * np.eye(self.X_train.shape[0])
        self._factor = cholesky_factor(kernel_matrix, jitter=DEFAULT_JITTER)
        self._alpha = cholesky_solve(self._factor, self.y_train)
        return self

    def _prepare_test_inputs(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        return array

    def posterior(self, X: np.ndarray, return_covariance: bool = True) -> PosteriorPrediction:
        X_test = self._prepare_test_inputs(X)
        prior_covariance = np.asarray(self.kernel(X_test, X_test), dtype=float)

        if self.X_train is None or self.y_train is None or self._factor is None or self._alpha is None:
            mean = np.zeros(X_test.shape[0], dtype=float)
            variance = np.clip(np.diag(prior_covariance), 0.0, None)
            covariance = prior_covariance if return_covariance else None
            return PosteriorPrediction(mean=mean, variance=variance, covariance=covariance)

        cross_covariance = np.asarray(self.kernel(self.X_train, X_test), dtype=float)
        mean = cross_covariance.T @ self._alpha
        projected = np.linalg.solve(self._factor, cross_covariance)
        covariance = prior_covariance - projected.T @ projected
        covariance = 0.5 * (covariance + covariance.T)
        variance = np.clip(np.diag(covariance), 0.0, None)
        if not return_covariance:
            covariance = None
        return PosteriorPrediction(mean=mean, variance=variance, covariance=covariance)

    def predict(
        self,
        X: np.ndarray,
        *,
        return_std: bool = False,
        return_covariance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        posterior = self.posterior(X, return_covariance=return_covariance)
        if return_covariance:
            return posterior.mean, posterior.covariance
        if return_std:
            return posterior.mean, posterior.std
        return posterior.mean

    def _require_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("fit the GP before using optimization utilities")
        return self.X_train, self.y_train

    @staticmethod
    def _as_trainable_matern(kernel: object) -> MaternKernel:
        if not isinstance(kernel, MaternKernel):
            raise TypeError("hyperparameter optimization currently supports MaternKernel only")
        length_scale = np.asarray(kernel.length_scale, dtype=float)
        if length_scale.ndim != 0:
            raise TypeError("hyperparameter optimization currently supports scalar Matern length scales only")
        return kernel

    @staticmethod
    def _parameter_bounds(optimize_noise: bool) -> list[tuple[float, float]]:
        bounds = [
            (np.log(1e-3), np.log(10.0)),
            (np.log(1e-4), np.log(10.0)),
        ]
        if optimize_noise:
            bounds.append((np.log(1e-8), np.log(1.0)))
        return bounds

    @staticmethod
    def _pack_log_parameters(kernel: MaternKernel, noise: float, optimize_noise: bool) -> np.ndarray:
        values = [
            np.log(float(kernel.length_scale)),
            np.log(float(kernel.variance)),
        ]
        if optimize_noise:
            values.append(np.log(float(noise)))
        return np.asarray(values, dtype=float)

    @staticmethod
    def _unpack_log_parameters(
        params: np.ndarray,
        base_kernel: MaternKernel,
        base_noise: float,
        optimize_noise: bool,
    ) -> tuple[MaternKernel, float]:
        values = np.asarray(params, dtype=float).reshape(-1)
        expected = 3 if optimize_noise else 2
        if values.size != expected:
            raise ValueError("unexpected optimization parameter shape")
        kernel = replace(
            base_kernel,
            length_scale=float(np.exp(values[0])),
            variance=float(np.exp(values[1])),
        )
        noise = float(np.exp(values[2])) if optimize_noise else float(base_noise)
        return kernel, noise

    def negative_log_marginal_likelihood(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        kernel: object | None = None,
        noise: float | None = None,
    ) -> float:
        if X is None or y is None:
            X_array, y_array = self._require_training_data()
        else:
            X_array = np.asarray(X, dtype=float)
            y_array = np.asarray(y, dtype=float).reshape(-1)
            if X_array.ndim == 1:
                X_array = X_array[:, None]
            if X_array.shape[0] != y_array.shape[0]:
                raise ValueError("X and y must agree on sample count")

        kernel_object = self.kernel if kernel is None else kernel
        noise_value = self.noise if noise is None else float(noise)
        if noise_value < 0.0:
            raise ValueError("noise must be non-negative")

        kernel_matrix = np.asarray(kernel_object(X_array, X_array), dtype=float)
        kernel_matrix = kernel_matrix + noise_value * np.eye(X_array.shape[0])
        factor = cholesky_factor(kernel_matrix, jitter=DEFAULT_JITTER)
        alpha = cholesky_solve(factor, y_array)
        n = X_array.shape[0]
        return float(
            0.5 * y_array @ alpha
            + 0.5 * stable_logdet(factor=factor)
            + 0.5 * n * np.log(2.0 * np.pi)
        )

    def log_marginal_likelihood(self) -> float:
        return -self.negative_log_marginal_likelihood()

    def optimize_hyperparameters(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        optimize_noise: bool = True,
        maxiter: int = 100,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizationResult:
        if X is None or y is None:
            X_array, y_array = self._require_training_data()
        else:
            X_array = np.asarray(X, dtype=float)
            y_array = np.asarray(y, dtype=float).reshape(-1)
            if X_array.ndim == 1:
                X_array = X_array[:, None]
            if X_array.shape[0] != y_array.shape[0]:
                raise ValueError("X and y must agree on sample count")

        trainable_kernel = self._as_trainable_matern(self.kernel)
        initial_objective = self.negative_log_marginal_likelihood(
            X_array,
            y_array,
            kernel=trainable_kernel,
            noise=self.noise,
        )
        x0 = self._pack_log_parameters(trainable_kernel, self.noise, optimize_noise)
        parameter_bounds = self._parameter_bounds(optimize_noise) if bounds is None else bounds

        def _objective(params: np.ndarray) -> float:
            candidate_kernel, candidate_noise = self._unpack_log_parameters(
                params,
                trainable_kernel,
                self.noise,
                optimize_noise,
            )
            return self.negative_log_marginal_likelihood(
                X_array,
                y_array,
                kernel=candidate_kernel,
                noise=candidate_noise,
            )

        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=parameter_bounds,
            options={"maxiter": int(maxiter)},
        )
        candidate_kernel, candidate_noise = self._unpack_log_parameters(
            result.x,
            trainable_kernel,
            self.noise,
            optimize_noise,
        )
        candidate_objective = self.negative_log_marginal_likelihood(
            X_array,
            y_array,
            kernel=candidate_kernel,
            noise=candidate_noise,
        )
        if np.isfinite(candidate_objective) and candidate_objective <= initial_objective:
            self.kernel = candidate_kernel
            self.noise = candidate_noise
            self.fit(X_array, y_array)

        return OptimizationResult(
            success=bool(result.success),
            objective=float(candidate_objective),
            iterations=int(getattr(result, "nit", 0)),
            parameters={
                "kernel": self.kernel,
                "noise": float(self.noise),
            },
            message=str(result.message),
        )

    def sample_posterior(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        posterior = self.posterior(X, return_covariance=True)
        generator = rng or np.random.default_rng()
        covariance = np.asarray(posterior.covariance, dtype=float) + DEFAULT_JITTER * np.eye(posterior.mean.size)
        samples = generator.multivariate_normal(posterior.mean, covariance, size=n_samples)
        return samples[0] if n_samples == 1 else samples

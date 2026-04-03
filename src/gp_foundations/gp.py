from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linalg import DEFAULT_JITTER, cholesky_factor, cholesky_solve, stable_logdet


@dataclass
class PosteriorPrediction:
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray | None = None

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))


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

    def log_marginal_likelihood(self) -> float:
        if self.X_train is None or self.y_train is None or self._factor is None or self._alpha is None:
            raise RuntimeError("fit the GP before computing marginal likelihood")
        n = self.X_train.shape[0]
        data_fit = -0.5 * self.y_train @ self._alpha
        complexity_penalty = -0.5 * stable_logdet(factor=self._factor)
        normalization = -0.5 * n * np.log(2.0 * np.pi)
        return float(data_fit + complexity_penalty + normalization)

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

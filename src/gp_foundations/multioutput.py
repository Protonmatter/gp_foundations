from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import thompson_sample
from .linalg import DEFAULT_JITTER, cholesky_factor, cholesky_solve


@dataclass
class MultiOutputPosterior:
    mean: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray | None = None


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

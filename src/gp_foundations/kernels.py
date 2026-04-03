from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


def _prepare_inputs(x1: ArrayLike, x2: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(x1, dtype=float)
    second = first if x2 is None else np.asarray(x2, dtype=float)
    if first.ndim == 1:
        first = first[:, None]
    if second.ndim == 1:
        second = second[:, None]
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("kernel inputs must be 1D or 2D arrays")
    if first.shape[1] != second.shape[1]:
        raise ValueError("kernel inputs must have the same feature dimension")
    return first, second


def _prepare_length_scale(length_scale: float | ArrayLike, n_features: int) -> np.ndarray:
    scale = np.asarray(length_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(n_features, float(scale))
    if scale.shape != (n_features,):
        raise ValueError("length_scale must be scalar or match feature dimension")
    if np.any(scale <= 0.0):
        raise ValueError("length_scale must be strictly positive")
    return scale


def _scaled_distances(x1: np.ndarray, x2: np.ndarray, length_scale: float | ArrayLike) -> np.ndarray:
    scale = _prepare_length_scale(length_scale, x1.shape[1])
    delta = (x1[:, None, :] - x2[None, :, :]) / scale
    return np.sqrt(np.sum(delta * delta, axis=2))


@dataclass(frozen=True)
class MaternKernel:
    length_scale: float | ArrayLike = 1.0
    variance: float = 1.0
    nu: float = 2.5

    def __call__(self, x1: ArrayLike, x2: ArrayLike | None = None) -> np.ndarray:
        first, second = _prepare_inputs(x1, x2)
        distance = _scaled_distances(first, second, self.length_scale)

        if self.nu == 0.5:
            core = np.exp(-distance)
        elif self.nu == 1.5:
            scaled = np.sqrt(3.0) * distance
            core = (1.0 + scaled) * np.exp(-scaled)
        elif self.nu == 2.5:
            scaled = np.sqrt(5.0) * distance
            core = (1.0 + scaled + (5.0 / 3.0) * distance * distance) * np.exp(-scaled)
        else:
            raise ValueError("supported nu values are 0.5, 1.5, and 2.5")

        return float(self.variance) * core


@dataclass(frozen=True)
class SpectralMixtureKernel:
    weights: ArrayLike
    means: ArrayLike
    scales: ArrayLike

    def __call__(self, x1: ArrayLike, x2: ArrayLike | None = None) -> np.ndarray:
        first, second = _prepare_inputs(x1, x2)
        weights = np.atleast_1d(np.asarray(self.weights, dtype=float))
        means = np.asarray(self.means, dtype=float)
        scales = np.asarray(self.scales, dtype=float)

        if means.ndim == 1:
            means = means[:, None]
        if scales.ndim == 1:
            scales = scales[:, None]
        if not (weights.shape[0] == means.shape[0] == scales.shape[0]):
            raise ValueError("weights, means, and scales must agree on component count")
        if means.shape[1] != first.shape[1] or scales.shape[1] != first.shape[1]:
            raise ValueError("spectral parameters must match input dimension")
        if np.any(scales <= 0.0):
            raise ValueError("spectral scales must be strictly positive")

        tau = first[:, None, :] - second[None, :, :]
        kernel = np.zeros((first.shape[0], second.shape[0]), dtype=float)

        for weight, mean, scale in zip(weights, means, scales):
            exp_term = np.exp(-2.0 * np.pi**2 * np.sum((tau**2) * scale[None, None, :], axis=2))
            cos_term = np.cos(2.0 * np.pi * np.sum(tau * mean[None, None, :], axis=2))
            kernel += float(weight) * exp_term * cos_term

        return kernel


@dataclass(frozen=True)
class TimeDecayKernel:
    base_kernel: object
    decay: float = 0.1
    reference_point: float = 0.0

    def __call__(self, x1: ArrayLike, x2: ArrayLike | None = None) -> np.ndarray:
        first, second = _prepare_inputs(x1, x2)
        if self.decay < 0.0:
            raise ValueError("decay must be non-negative")
        base = np.asarray(self.base_kernel(first, second), dtype=float)
        weight_first = np.exp(-self.decay * np.abs(first[:, 0] - self.reference_point))
        weight_second = np.exp(-self.decay * np.abs(second[:, 0] - self.reference_point))
        return base * weight_first[:, None] * weight_second[None, :]

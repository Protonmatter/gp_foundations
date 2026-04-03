from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .linalg import DEFAULT_JITTER


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    return mean_array + np.sqrt(beta) * std_array


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    improvement = mean_array - float(best) - float(xi)
    safe_std = np.where(std_array > 0.0, std_array, 1.0)
    z = improvement / safe_std
    values = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
    return np.where(std_array > 0.0, values, np.maximum(improvement, 0.0))


def thompson_sample(
    mean: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator | None = None,
    n_samples: int = 1,
) -> np.ndarray:
    mean_array = np.asarray(mean, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    generator = rng or np.random.default_rng()
    stabilized = covariance_array + DEFAULT_JITTER * np.eye(mean_array.size)
    draws = generator.multivariate_normal(mean_array, stabilized, size=n_samples)
    return draws[0] if n_samples == 1 else draws


def softmax_select(
    scores: np.ndarray,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    values = np.asarray(scores, dtype=float)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    centered = values - np.max(values)
    logits = centered / temperature
    weights = np.exp(logits)
    probabilities = weights / np.sum(weights)
    generator = rng or np.random.default_rng()
    return int(generator.choice(np.arange(values.size), p=probabilities))

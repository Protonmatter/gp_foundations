from __future__ import annotations

import numpy as np


def logistic_map(x: float | np.ndarray, rate: float = 4.0) -> float | np.ndarray:
    values = np.asarray(x, dtype=float)
    result = rate * values * (1.0 - values)
    if np.isscalar(x):
        return float(result)
    return result


def chaotic_timing_sequence(
    seed: int | float,
    length: int,
    *,
    rate: float = 4.0,
    low: float = 0.0,
    high: float = 1.0,
    discard: int = 64,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if high <= low:
        raise ValueError("high must be greater than low")

    if isinstance(seed, (int, np.integer)):
        generator = np.random.default_rng(int(seed))
        state = float(generator.uniform(1e-6, 1.0 - 1e-6))
    else:
        state = float(seed)
        if not (0.0 < state < 1.0):
            raise ValueError("float seed must lie in (0, 1)")

    for _ in range(discard):
        state = float(logistic_map(state, rate=rate))

    values = np.empty(length, dtype=float)
    for idx in range(length):
        state = float(logistic_map(state, rate=rate))
        values[idx] = state

    return low + (high - low) * values

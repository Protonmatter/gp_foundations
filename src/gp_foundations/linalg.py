from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


DEFAULT_JITTER = 1e-8


def _as_square_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square 2D array")
    return 0.5 * (array + array.T)


def ensure_spd(
    matrix: np.ndarray,
    jitter: float = DEFAULT_JITTER,
    max_tries: int = 8,
    return_jitter: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    candidate = _as_square_matrix(matrix)
    eye = np.eye(candidate.shape[0], dtype=float)
    added_jitter = 0.0

    for _ in range(max_tries + 1):
        test_matrix = candidate + added_jitter * eye
        try:
            np.linalg.cholesky(test_matrix)
            if return_jitter:
                return test_matrix, added_jitter
            return test_matrix
        except np.linalg.LinAlgError:
            added_jitter = jitter if added_jitter == 0.0 else added_jitter * 10.0

    raise np.linalg.LinAlgError("failed to make matrix positive definite with jitter")


def cholesky_factor(
    matrix: np.ndarray,
    jitter: float = DEFAULT_JITTER,
    max_tries: int = 8,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    spd_matrix, added_jitter = ensure_spd(
        matrix,
        jitter=jitter,
        max_tries=max_tries,
        return_jitter=True,
    )
    factor = np.linalg.cholesky(spd_matrix)
    if return_info:
        return factor, added_jitter
    return factor


def cholesky_solve(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = np.asarray(factor, dtype=float)
    values = np.asarray(rhs, dtype=float)
    y = solve_triangular(lower, values, lower=True)
    return solve_triangular(lower.T, y, lower=False)


def stable_logdet(
    matrix: np.ndarray | None = None,
    *,
    factor: np.ndarray | None = None,
    jitter: float = DEFAULT_JITTER,
    max_tries: int = 8,
) -> float:
    if factor is None:
        if matrix is None:
            raise ValueError("matrix or factor must be provided")
        factor = cholesky_factor(matrix, jitter=jitter, max_tries=max_tries)
    diagonal = np.diag(np.asarray(factor, dtype=float))
    return float(2.0 * np.sum(np.log(diagonal)))

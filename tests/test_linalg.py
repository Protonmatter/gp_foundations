import unittest

import numpy as np

from gp_foundations.linalg import cholesky_factor, cholesky_solve, ensure_spd, stable_logdet


class LinalgTests(unittest.TestCase):
    def test_cholesky_solve_matches_numpy(self) -> None:
        matrix = np.array([[2.0, 0.5], [0.5, 1.5]])
        rhs = np.array([1.0, -1.0])
        factor = cholesky_factor(matrix)
        solution = cholesky_solve(factor, rhs)
        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(solution, expected)

    def test_stable_logdet_matches_numpy(self) -> None:
        matrix = np.array([[1.8, 0.2], [0.2, 1.4]])
        sign, expected = np.linalg.slogdet(matrix)
        self.assertEqual(sign, 1.0)
        self.assertAlmostEqual(stable_logdet(matrix), expected)

    def test_ensure_spd_adds_jitter(self) -> None:
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        spd, added_jitter = ensure_spd(matrix, jitter=1e-6, return_jitter=True)
        self.assertGreater(added_jitter, 0.0)
        np.linalg.cholesky(spd)


if __name__ == '__main__':
    unittest.main()

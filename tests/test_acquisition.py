import unittest

import numpy as np

from gp_foundations.acquisition import expected_improvement, softmax_select, thompson_sample, upper_confidence_bound


class AcquisitionTests(unittest.TestCase):
    def test_ucb_prefers_high_mean_and_uncertainty(self) -> None:
        mean = np.array([0.1, 0.4])
        std = np.array([0.5, 0.1])
        scores = upper_confidence_bound(mean, std, beta=4.0)
        self.assertGreater(scores[0], scores[1])

    def test_expected_improvement_handles_zero_variance(self) -> None:
        values = expected_improvement(np.array([0.7, 0.1]), np.array([0.0, 0.0]), best=0.5)
        np.testing.assert_allclose(values, np.array([0.2, 0.0]))

    def test_thompson_sample_is_reproducible(self) -> None:
        mean = np.array([0.0, 1.0])
        covariance = np.array([[1.0, 0.2], [0.2, 1.0]])
        rng = np.random.default_rng(42)
        first = thompson_sample(mean, covariance, rng=rng)
        rng = np.random.default_rng(42)
        second = thompson_sample(mean, covariance, rng=rng)
        np.testing.assert_allclose(first, second)

    def test_softmax_select_returns_valid_index(self) -> None:
        idx = softmax_select(np.array([1.0, 2.0, 3.0]), rng=np.random.default_rng(0))
        self.assertIn(idx, {0, 1, 2})


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from gp_foundations.systems import chaotic_timing_sequence, logistic_map


class SystemTests(unittest.TestCase):
    def test_logistic_map_scalar(self) -> None:
        self.assertAlmostEqual(logistic_map(0.5, rate=4.0), 1.0)

    def test_chaotic_sequence_is_bounded_and_reproducible(self) -> None:
        first = chaotic_timing_sequence(seed=7, length=16, low=0.2, high=0.8)
        second = chaotic_timing_sequence(seed=7, length=16, low=0.2, high=0.8)
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(first >= 0.2))
        self.assertTrue(np.all(first <= 0.8))


if __name__ == '__main__':
    unittest.main()

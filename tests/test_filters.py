import unittest

import numpy as np

from gp_foundations.filters import KalmanFilter


class KalmanFilterTests(unittest.TestCase):
    def test_filter_tracks_constant_signal(self) -> None:
        kf = KalmanFilter(
            transition_matrix=np.array([[1.0]]),
            observation_matrix=np.array([[1.0]]),
            process_covariance=np.array([[0.01]]),
            observation_covariance=np.array([[0.1]]),
            initial_state=np.array([0.0]),
            initial_covariance=np.array([[1.0]]),
        )
        for measurement in [1.0, 0.9, 1.1, 1.0, 1.0]:
            state = kf.step(np.array([measurement]))
        self.assertAlmostEqual(state[0], 1.0, delta=0.2)


if __name__ == '__main__':
    unittest.main()

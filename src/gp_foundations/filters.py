from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(
        self,
        transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        process_covariance: np.ndarray,
        observation_covariance: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        control_matrix: np.ndarray | None = None,
    ):
        self.F = np.asarray(transition_matrix, dtype=float)
        self.H = np.asarray(observation_matrix, dtype=float)
        self.Q = np.asarray(process_covariance, dtype=float)
        self.R = np.asarray(observation_covariance, dtype=float)
        self.x = np.asarray(initial_state, dtype=float).reshape(-1, 1)
        self.P = np.asarray(initial_covariance, dtype=float)
        self.B = None if control_matrix is None else np.asarray(control_matrix, dtype=float)

    @property
    def state(self) -> np.ndarray:
        return self.x.reshape(-1)

    @property
    def covariance(self) -> np.ndarray:
        return self.P

    def predict(self, control: np.ndarray | None = None) -> np.ndarray:
        if control is not None and self.B is not None:
            control_vec = np.asarray(control, dtype=float).reshape(-1, 1)
            self.x = self.F @ self.x + self.B @ control_vec
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=float).reshape(-1, 1)
        innovation = z - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + kalman_gain @ innovation
        identity = np.eye(self.P.shape[0])
        residual = identity - kalman_gain @ self.H
        self.P = residual @ self.P @ residual.T + kalman_gain @ self.R @ kalman_gain.T
        return self.state

    def step(self, measurement: np.ndarray, control: np.ndarray | None = None) -> np.ndarray:
        self.predict(control=control)
        return self.update(measurement)

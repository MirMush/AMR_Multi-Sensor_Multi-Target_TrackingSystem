from __future__ import annotations

from dataclasses import replace

import numpy as np

from .measurement_models import MeasurementModel
from .types import Detection, Track
from .utils import normalize_angle


class CVEKFHooks:
    """
    Minimal CV-EKF predictor/updater used only to make the T6 sandbox runnable.

    Replace this class later with your Task 3-5 tracker implementation.
    """

    def __init__(self, measurement_model: MeasurementModel, sigma_a_mps2: float = 0.4):
        self._model = measurement_model
        self._sigma_a = float(sigma_a_mps2)

    def predict(self, track: Track, target_time_s: float) -> Track:
        dt = max(0.0, float(target_time_s - track.last_time_s))
        if dt <= 0.0:
            return track

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        q11 = (dt**4) / 4.0
        q13 = (dt**3) / 2.0
        q33 = dt**2
        Q = (self._sigma_a**2) * np.array(
            [
                [q11, 0.0, q13, 0.0],
                [0.0, q11, 0.0, q13],
                [q13, 0.0, q33, 0.0],
                [0.0, q13, 0.0, q33],
            ]
        )

        x_pred = F @ track.x
        P_pred = F @ track.P @ F.T + Q
        return replace(track, x=x_pred, P=P_pred, last_time_s=float(target_time_s))

    def update(self, track: Track, detection: Detection) -> Track:
        z_hat, H, R_model = self._model.predict(track.x, detection.sensor_id)
        R = detection.R if detection.R is not None else R_model

        y = detection.z - z_hat
        if y.shape[0] >= 2:
            y[1] = normalize_angle(float(y[1]))

        S = H @ track.P @ H.T + R
        reg = 1e-9 * np.eye(S.shape[0])
        try:
            S_inv = np.linalg.inv(S + reg)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + reg)

        K = track.P @ H.T @ S_inv
        x_upd = track.x + K @ y

        # Joseph form for covariance stability.
        I = np.eye(track.P.shape[0])
        KH = K @ H
        P_upd = (I - KH) @ track.P @ (I - KH).T + K @ R @ K.T

        return replace(track, x=x_upd, P=P_upd, last_time_s=float(detection.time_s))

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .measurement_models import MeasurementModel
from .types import Detection, Track
from EKF import EKF as TopEKF


class CVEKFHooks:
    """
    Adapter that implements the Task6 hook interface by delegating to
    the top-level `EKF` single-target implementation.

    This centralizes the EKF-backed behavior in a single place so scripts
    and the fusion pipeline can continue to call `predict(track, t)` and
    `update(track, detection)` without changing `run_fusion_cycle`.
    """

    def __init__(self, measurement_model: MeasurementModel, sigma_a_mps2: float = 0.4):
        self._model = measurement_model
        self._sigma_a = float(sigma_a_mps2)

    def predict(self, track: Track, target_time_s: float) -> Track:
        dt = max(0.0, float(target_time_s - track.last_time_s))
        if dt <= 0.0:
            return track

        ekf = TopEKF(cfm=self._model._manager, sigma_a=self._sigma_a)
        ekf.x = track.x.copy()
        ekf.P = track.P.copy()
        ekf.predict(dt)
        return replace(track, x=ekf.state(), P=ekf.covariance(), last_time_s=float(target_time_s))

    def update(self, track: Track, detection: Detection) -> Track:
        ekf = TopEKF(cfm=self._model._manager, sigma_a=self._sigma_a)
        ekf.x = track.x.copy()
        ekf.P = track.P.copy()
        ekf.update(detection.z, detection.sensor_id)
        return replace(track, x=ekf.state(), P=ekf.covariance(), last_time_s=float(detection.time_s))

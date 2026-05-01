from __future__ import annotations

from typing import Protocol

import numpy as np

from coord_frame_manager import CoordFrameManager


class MeasurementModel(Protocol):
    """Interface so later tasks can swap in any measurement model implementation."""

    def set_vessel_position(self, north_m: float, east_m: float) -> None:
        ...

    def predict(self, x: np.ndarray, sensor_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class CoordFrameMeasurementModel:
    """
    Adapter around CoordFrameManager.

    This keeps Task 6 logic independent from the concrete implementation so
    T3/T4/T5 can be plugged in with minimal changes.
    """

    def __init__(self, manager: CoordFrameManager | None = None):
        self._manager = manager if manager is not None else CoordFrameManager()

    def set_vessel_position(self, north_m: float, east_m: float) -> None:
        self._manager.update_vessel_pos(north_m, east_m)

    def predict(self, x: np.ndarray, sensor_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_hat = self._manager.h(x, sensor_id)
        H = self._manager.H(x, sensor_id)
        R = self._manager.R(sensor_id)
        return z_hat, H, R

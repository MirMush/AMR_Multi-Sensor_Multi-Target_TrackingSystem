from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .measurement_models import MeasurementModel
from .types import Detection, GateCandidate, Track
from .utils import normalize_angle


def _mahalanobis_distance_squared(y: np.ndarray, S: np.ndarray) -> float:
    # Small diagonal loading helps when S is close to singular.
    reg = 1e-9 * np.eye(S.shape[0])
    try:
        solved = np.linalg.solve(S + reg, y)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(S + reg) @ y
    return float(y.T @ solved)


def compute_gate_candidates(
    tracks: list[Track],
    detections: list[Detection],
    measurement_model: MeasurementModel,
    gate_probability: float = 0.99,
) -> list[GateCandidate]:
    """
    Compute all (track, detection) pairs that pass Mahalanobis gating.

    d^2 = y^T S^-1 y, with y = z - h(x^-), S = H P^- H^T + R
    """

    gated: list[GateCandidate] = []
    for track_idx, track in enumerate(tracks):
        for det_idx, detection in enumerate(detections):
            z_hat, H, R_model = measurement_model.predict(track.x, detection.sensor_id)
            R = detection.R if detection.R is not None else R_model

            y = detection.z - z_hat
            if y.shape[0] >= 2:
                y[1] = normalize_angle(float(y[1]))

            S = H @ track.P @ H.T + R
            d2 = _mahalanobis_distance_squared(y, S)
            gamma = float(chi2.ppf(gate_probability, df=y.shape[0]))

            if d2 <= gamma:
                gated.append(
                    GateCandidate(
                        track_index=track_idx,
                        detection_index=det_idx,
                        d2=d2,
                        gamma=gamma,
                        innovation=y,
                    )
                )

    return gated

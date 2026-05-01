from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import AssociationResult, GateCandidate


def associate_gnn(
    num_tracks: int,
    num_detections: int,
    gated_candidates: list[GateCandidate],
    invalid_cost: float = 1e6,
) -> AssociationResult:
    """
    Global Nearest Neighbour association using Hungarian assignment.

    Only gated pairs are assignable. All other pairs are hard-constrained via
    a very large cost.
    """

    cost_matrix = np.full((num_tracks, num_detections), invalid_cost, dtype=float)
    for candidate in gated_candidates:
        cost_matrix[candidate.track_index, candidate.detection_index] = candidate.d2

    matches: list[tuple[int, int, float]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    if num_tracks > 0 and num_detections > 0:
        rows, cols = linear_sum_assignment(cost_matrix)
        for row, col in zip(rows, cols):
            if cost_matrix[row, col] >= invalid_cost:
                continue
            matches.append((int(row), int(col), float(cost_matrix[row, col])))
            matched_tracks.add(int(row))
            matched_detections.add(int(col))

    unmatched_track_indices = [idx for idx in range(num_tracks) if idx not in matched_tracks]
    unmatched_detection_indices = [idx for idx in range(num_detections) if idx not in matched_detections]

    return AssociationResult(
        matches=matches,
        unmatched_track_indices=unmatched_track_indices,
        unmatched_detection_indices=unmatched_detection_indices,
        cost_matrix=cost_matrix,
    )

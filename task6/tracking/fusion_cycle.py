from __future__ import annotations

from dataclasses import replace

from .association import associate_gnn
from .baseline_ekf import CVEKFHooks
from .gating import compute_gate_candidates
from .measurement_models import MeasurementModel
from .types import Detection, FusionCycleResult, Track


def run_fusion_cycle(
    time_s: float,
    tracks: list[Track],
    detections: list[Detection],
    sensor_available: dict[str, bool],
    measurement_model: MeasurementModel,
    ekf_hooks: CVEKFHooks,
    gate_probability: float = 0.99,
) -> FusionCycleResult:
    """
    One Task 6 cycle:
    1) Predict all tracks
    2) Apply per-sensor availability filtering
    3) Mahalanobis gating
    4) Global association
    5) Update matched tracks and flag unmatched tracks
    """

    available_detections: list[Detection] = []
    skipped_detections: list[Detection] = []
    for detection in detections:
        if sensor_available.get(detection.sensor_id, True):
            available_detections.append(detection)
        else:
            skipped_detections.append(detection)

    predicted_tracks = [ekf_hooks.predict(track, time_s) for track in tracks]
    gated_candidates = compute_gate_candidates(
        predicted_tracks,
        available_detections,
        measurement_model=measurement_model,
        gate_probability=gate_probability,
    )

    association = associate_gnn(
        num_tracks=len(predicted_tracks),
        num_detections=len(available_detections),
        gated_candidates=gated_candidates,
    )

    match_by_track = {track_idx: det_idx for track_idx, det_idx, _ in association.matches}
    updated_tracks: list[Track] = []
    for track_idx, predicted_track in enumerate(predicted_tracks):
        matched_det_idx = match_by_track.get(track_idx)
        if matched_det_idx is None:
            updated_tracks.append(replace(predicted_track, missed_count=predicted_track.missed_count + 1))
            continue

        detection = available_detections[matched_det_idx]
        updated_track = ekf_hooks.update(predicted_track, detection)
        updated_tracks.append(replace(updated_track, missed_count=0))

    return FusionCycleResult(
        updated_tracks=updated_tracks,
        association=association,
        gated_candidates=gated_candidates,
        available_detections=available_detections,
        skipped_detections=skipped_detections,
        predicted_tracks=predicted_tracks,
    )

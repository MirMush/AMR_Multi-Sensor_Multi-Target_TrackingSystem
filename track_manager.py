"""
T7 — Track Manager

Full track lifecycle on top of the T6 gating + association pipeline:
  - Tentative : spawned from every unmatched detection
  - Confirmed : M-of-N hits within a sliding window of N scans
  - Coasting  : predict-only when missed; gate widens naturally as P grows
  - Deleted   : after K_del consecutive missed detections
  - Merge     : duplicate tentative/confirmed tracks merged by Mahalanobis distance
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# Make task6 tracking modules importable from project root
_TASK6 = Path(__file__).resolve().parent / "task6"
if str(_TASK6) not in sys.path:
    sys.path.insert(0, str(_TASK6))

from task6.tracking.baseline_ekf import CVEKFHooks          # noqa: E402
from task6.tracking.gating import compute_gate_candidates    # noqa: E402
from task6.tracking.association import associate_gnn         # noqa: E402
from task6.tracking.measurement_models import MeasurementModel  # noqa: E402
from task6.tracking.types import Detection, Track            # noqa: E402

# ---------------------------------------------------------------------------
# Extended track status constants
# ---------------------------------------------------------------------------
TENTATIVE = "tentative"
CONFIRMED = "confirmed"
COASTING  = "coasting"
DELETED   = "deleted"


@dataclass
class TrackManagerConfig:
    M: int   = 3      # hits needed for confirmation
    N: int   = 15     # confirmation window (1-Hz ticks). Radar fires at 0.3 Hz → 1 hit per ~3
                      # ticks; N=15 gives ~4-5 radar hits so M=3 is reliably achievable.
    K_del: int = 15   # consecutive missed ticks before deletion. Radar gap = 3 ticks, so
                      # K_del must exceed the inter-scan gap to avoid spurious deletions.
    merge_threshold: float = 9.21   # Mahalanobis² threshold for duplicate merge (chi2(2), 99%)
    gate_probability: float = 0.99
    sigma_a: float = 0.05


@dataclass
class ManagedTrack:
    """Wraps a Track with lifecycle state."""
    track: Track
    status: str = TENTATIVE
    hit_history: list[bool] = field(default_factory=list)  # True=hit, False=miss, last N scans
    consecutive_misses: int = 0
    age: int = 0  # scans since creation

    # For velocity init from first two detections
    first_detection: Detection | None = None
    first_det_time: float | None = None

    @property
    def track_id(self) -> int:
        return self.track.track_id

    @property
    def is_confirmed(self) -> bool:
        return self.status == CONFIRMED

    @property
    def is_active(self) -> bool:
        return self.status in (TENTATIVE, CONFIRMED, COASTING)


class TrackManager:
    """
    Manages the full track lifecycle for multi-target tracking.

    Usage:
        tm = TrackManager(measurement_model, config)
        for t in time_steps:
            tm.update_vessel_pos(pN, pE)
            confirmed = tm.step(t, detections, sensor_available)
    """

    def __init__(
        self,
        measurement_model: MeasurementModel,
        config: TrackManagerConfig | None = None,
    ):
        self._model  = measurement_model
        self._cfg    = config or TrackManagerConfig()
        self._hooks  = CVEKFHooks(measurement_model, sigma_a_mps2=self._cfg.sigma_a)
        self._tracks : list[ManagedTrack] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_vessel_pos(self, north_m: float, east_m: float) -> None:
        self._model.set_vessel_position(north_m, east_m)

    def step(
        self,
        time_s: float,
        detections: list[Detection],
        sensor_available: dict[str, bool] | None = None,
    ) -> list[Track]:
        """
        Run one tracking cycle. Returns only confirmed tracks.
        """
        if sensor_available is None:
            sensor_available = {d.sensor_id: True for d in detections}

        available = [d for d in detections if sensor_available.get(d.sensor_id, True)]

        # 1. Predict all active tracks
        active = [mt for mt in self._tracks if mt.is_active]
        predicted_tracks = [self._hooks.predict(mt.track, time_s) for mt in active]

        # 2. Gate + associate
        gate_candidates = compute_gate_candidates(
            predicted_tracks, available, self._model, self._cfg.gate_probability
        )
        association = associate_gnn(
            num_tracks=len(predicted_tracks),
            num_detections=len(available),
            gated_candidates=gate_candidates,
        )

        match_by_track = {ti: di for ti, di, _ in association.matches}

        # 3. Update matched tracks / flag misses
        updated_managed: list[ManagedTrack] = []
        for idx, (mt, pred_track) in enumerate(zip(active, predicted_tracks)):
            det_idx = match_by_track.get(idx)

            if det_idx is not None:
                det          = available[det_idx]
                updated_trk  = self._hooks.update(pred_track, det)
                new_misses   = 0
                hit          = True
                # Velocity init on second detection for tentative tracks
                if mt.status == TENTATIVE and mt.first_detection is not None:
                    updated_trk = self._init_velocity(updated_trk, det, mt)
            else:
                updated_trk = pred_track
                new_misses  = mt.consecutive_misses + 1
                hit         = False

            # Update hit history (sliding window of N)
            new_history = (mt.hit_history + [hit])[-self._cfg.N:]

            new_status = self._next_status(mt, new_history, new_misses)

            first_det      = mt.first_detection
            first_det_time = mt.first_det_time
            if mt.status == TENTATIVE and mt.first_detection is None and hit:
                first_det      = available[det_idx]
                first_det_time = time_s

            updated_managed.append(replace(
                mt,
                track             = replace(updated_trk, status=new_status),
                status            = new_status,
                hit_history       = new_history,
                consecutive_misses= new_misses,
                age               = mt.age + 1,
                first_detection   = first_det,
                first_det_time    = first_det_time,
            ))

        # 4. Spawn tentative tracks for unmatched detections
        for det_idx in association.unmatched_detection_indices:
            det = available[det_idx]
            new_mt = self._spawn(det, time_s)
            updated_managed.append(new_mt)

        # 5. Remove deleted tracks
        updated_managed = [mt for mt in updated_managed if mt.status != DELETED]

        # 6. Merge duplicates
        updated_managed = self._merge_duplicates(updated_managed)

        self._tracks = updated_managed

        return [mt.track for mt in self._tracks if mt.status in (CONFIRMED, COASTING)]

    def all_tracks(self) -> list[ManagedTrack]:
        return list(self._tracks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_status(self, mt: ManagedTrack, history: list[bool], misses: int) -> str:
        if misses >= self._cfg.K_del:
            return DELETED

        if mt.status == TENTATIVE:
            hits_in_window = sum(history[-self._cfg.N:])
            if hits_in_window >= self._cfg.M:
                return CONFIRMED
            return TENTATIVE

        if mt.status in (CONFIRMED, COASTING):
            if misses > 0:
                return COASTING
            return CONFIRMED

        return mt.status

    def _spawn(self, det: Detection, time_s: float) -> ManagedTrack:
        """Initialize a tentative track from a single detection."""
        sid = det.sensor_id
        z   = det.z
        r, phi = float(z[0]), float(z[1])

        # Sensor origin
        s = self._sensor_origin(sid)
        pN = s[0] + r * np.cos(phi)
        pE = s[1] + r * np.sin(phi)

        x0 = np.array([pN, pE, 0.0, 0.0], dtype=float)

        # Initial covariance from sensor noise propagated to position space
        _, _, R = self._model.predict(x0, sid)
        sigma_r   = np.sqrt(R[0, 0])
        sigma_phi = np.sqrt(R[1, 1])
        J = np.array([
            [np.cos(phi), -r * np.sin(phi)],
            [np.sin(phi),  r * np.cos(phi)],
        ])
        pos_cov = J @ np.diag([sigma_r**2, sigma_phi**2]) @ J.T
        P0 = np.diag([pos_cov[0, 0], pos_cov[1, 1], 100.0, 100.0])

        track = Track(
            track_id    = self._next_id,
            x           = x0,
            P           = P0,
            last_time_s = time_s,
            missed_count= 0,
            status      = TENTATIVE,
            truth_id    = det.truth_id,
        )
        self._next_id += 1

        return ManagedTrack(
            track          = track,
            status         = TENTATIVE,
            hit_history    = [True],
            first_detection= det,
            first_det_time = time_s,
        )

    def _init_velocity(self, track: Track, det: Detection, mt: ManagedTrack) -> Track:
        """Estimate velocity from first two detections via finite difference."""
        if mt.first_det_time is None or mt.first_detection is None:
            return track

        dt = det.time_s - mt.first_det_time
        if dt < 1e-6:
            return track

        # Convert both detections to NED
        s = self._sensor_origin(det.sensor_id)
        r1, phi1 = mt.first_detection.z[0], mt.first_detection.z[1]
        r2, phi2 = det.z[0], det.z[1]
        pN1 = s[0] + r1 * np.cos(phi1)
        pE1 = s[1] + r1 * np.sin(phi1)
        pN2 = s[0] + r2 * np.cos(phi2)
        pE2 = s[1] + r2 * np.sin(phi2)

        vN = (pN2 - pN1) / dt
        vE = (pE2 - pE1) / dt

        x_new    = track.x.copy()
        x_new[2] = vN
        x_new[3] = vE
        return replace(track, x=x_new)

    def _merge_duplicates(self, tracks: list[ManagedTrack]) -> list[ManagedTrack]:
        """
        Merge pairs of tracks whose Mahalanobis distance is below threshold.
        Keep the older / higher-status track; discard the duplicate.
        """
        if len(tracks) < 2:
            return tracks

        status_rank = {CONFIRMED: 3, COASTING: 2, TENTATIVE: 1, DELETED: 0}
        to_remove: set[int] = set()

        for i in range(len(tracks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(tracks)):
                if j in to_remove:
                    continue
                ti, tj = tracks[i].track, tracks[j].track
                diff = ti.x[:2] - tj.x[:2]
                S    = ti.P[:2, :2] + tj.P[:2, :2]
                try:
                    d2 = float(diff @ np.linalg.inv(S) @ diff)
                except np.linalg.LinAlgError:
                    continue
                if d2 < self._cfg.merge_threshold:
                    # Discard the lower-status / younger track
                    rank_i = status_rank.get(tracks[i].status, 0)
                    rank_j = status_rank.get(tracks[j].status, 0)
                    discard = j if rank_i >= rank_j else i
                    to_remove.add(discard)

        return [mt for idx, mt in enumerate(tracks) if idx not in to_remove]

    def _sensor_origin(self, sensor_id: str) -> np.ndarray:
        """Return NED position of the sensor (uses measurement model internals)."""
        # Predict at a dummy state to get origin implicitly via h(0) - but
        # easier to just hard-code the known positions from the spec.
        origins = {
            "radar":  np.array([0.0, 0.0]),
            "camera": np.array([-80.0, 120.0]),
        }
        if sensor_id in origins:
            return origins[sensor_id]
        # AIS: use vessel position from measurement model
        return self._model._manager.vessel_pos

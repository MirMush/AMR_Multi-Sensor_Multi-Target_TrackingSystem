from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    """Unified detection container used by the T6 pipeline."""

    detection_id: str
    time_s: float
    sensor_id: str
    z: np.ndarray
    R: np.ndarray | None = None
    truth_id: int | None = None
    is_false_alarm: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """Minimal track state required for T6 gating + association."""

    track_id: int
    x: np.ndarray
    P: np.ndarray
    last_time_s: float
    missed_count: int = 0
    status: str = "confirmed"
    truth_id: int | None = None


@dataclass
class GateCandidate:
    """Candidate pair that passed Mahalanobis gating."""

    track_index: int
    detection_index: int
    d2: float
    gamma: float
    innovation: np.ndarray


@dataclass
class AssociationResult:
    """Global data-association output."""

    matches: list[tuple[int, int, float]]
    unmatched_track_indices: list[int]
    unmatched_detection_indices: list[int]
    cost_matrix: np.ndarray


@dataclass
class FusionCycleResult:
    """Results for one multi-sensor T6 cycle."""

    updated_tracks: list[Track]
    association: AssociationResult
    gated_candidates: list[GateCandidate]
    available_detections: list[Detection]
    skipped_detections: list[Detection]
    predicted_tracks: list[Track]


@dataclass
class FakeScan:
    """Synthetic scan packet used by the sandbox script."""

    time_s: float
    detections: list[Detection]
    sensor_available: dict[str, bool]
    truth_states: dict[int, np.ndarray]
    vessel_pos_ned: np.ndarray


@dataclass
class FakeScenarioConfig:
    """Configuration for synthetic Task 6 scenario generation."""

    duration_s: float = 120.0
    dt_s: float = 1.0
    seed: int = 7
    include_false_alarms: bool = True

"""Task 6 sandbox modules for gating and data association."""

from .association import associate_gnn
from .baseline_ekf import CVEKFHooks
from .fake_data import FakeScenarioConfig, FakeScan, generate_task6_fake_scans, initialize_tracks_from_truth
from .fusion_cycle import run_fusion_cycle
from .gating import compute_gate_candidates
from .measurement_models import CoordFrameMeasurementModel, MeasurementModel
from .types import AssociationResult, Detection, FusionCycleResult, GateCandidate, Track

__all__ = [
    "AssociationResult",
    "CVEKFHooks",
    "CoordFrameMeasurementModel",
    "Detection",
    "FakeScenarioConfig",
    "FakeScan",
    "FusionCycleResult",
    "GateCandidate",
    "MeasurementModel",
    "Track",
    "associate_gnn",
    "compute_gate_candidates",
    "generate_task6_fake_scans",
    "initialize_tracks_from_truth",
    "run_fusion_cycle",
]

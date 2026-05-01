from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Detection, FakeScenarioConfig, FakeScan, Track
from .utils import normalize_angle


@dataclass(frozen=True)
class _TargetSpec:
    target_id: int
    p0_n: float
    p0_e: float
    v_n: float
    v_e: float
    has_ais: bool


RADAR_POS = np.array([0.0, 0.0])
CAMERA_POS = np.array([-80.0, 120.0])

SENSOR_PARAMS = {
    "radar": {
        "range_max_m": 1000.0,
        "fov_deg": 360.0,
        "boresight_deg": 0.0,
        "pd": 0.88,
        "lambda_fa": 5.0,
        "sigma_r_m": 5.0,
        "sigma_b_deg": 0.3,
    },
    "camera": {
        "range_max_m": 500.0,
        "fov_deg": 180.0,
        "boresight_deg": 45.0,
        "pd": 0.85,
        "lambda_fa": 2.0,
        "sigma_r_m": 8.0,
        "sigma_b_deg": 0.15,
    },
    "ais": {
        "range_max_m": 5000.0,
        "fov_deg": 360.0,
        "boresight_deg": 0.0,
        "pd": 0.98,
        "lambda_fa": 0.0,
        "sigma_r_m": 6.0,
        "sigma_b_deg": 0.6,
    },
}


def _sensor_R(sensor_id: str) -> np.ndarray:
    params = SENSOR_PARAMS[sensor_id]
    sigma_r = params["sigma_r_m"]
    sigma_b = np.deg2rad(params["sigma_b_deg"])
    return np.diag([sigma_r**2, sigma_b**2])


def _target_specs() -> list[_TargetSpec]:
    # Four crossing targets for conflict-heavy T6 association.
    return [
        _TargetSpec(target_id=0, p0_n=420.0, p0_e=-420.0, v_n=-3.0, v_e=3.0, has_ais=True),
        _TargetSpec(target_id=1, p0_n=-420.0, p0_e=420.0, v_n=3.0, v_e=-3.0, has_ais=True),
        _TargetSpec(target_id=2, p0_n=450.0, p0_e=450.0, v_n=-3.3, v_e=-3.2, has_ais=False),
        _TargetSpec(target_id=3, p0_n=-450.0, p0_e=-450.0, v_n=3.2, v_e=3.4, has_ais=False),
    ]


def _truth_state(target: _TargetSpec, time_s: float) -> np.ndarray:
    p_n = target.p0_n + target.v_n * time_s
    p_e = target.p0_e + target.v_e * time_s
    return np.array([p_n, p_e, target.v_n, target.v_e], dtype=float)


def _vessel_position(time_s: float) -> np.ndarray:
    # Smooth vessel motion to make AIS geometry time-varying.
    north = 120.0 + 130.0 * np.cos(0.016 * time_s)
    east = -320.0 + 110.0 * np.sin(0.02 * time_s)
    return np.array([north, east], dtype=float)


def _sensor_position(sensor_id: str, vessel_pos_ned: np.ndarray) -> np.ndarray:
    if sensor_id == "radar":
        return RADAR_POS
    if sensor_id == "camera":
        return CAMERA_POS
    if sensor_id == "ais":
        return vessel_pos_ned
    raise ValueError(f"Unknown sensor_id: {sensor_id}")


def _is_visible(sensor_id: str, target_pos_ned: np.ndarray, vessel_pos_ned: np.ndarray) -> bool:
    sensor_pos = _sensor_position(sensor_id, vessel_pos_ned)
    delta = target_pos_ned - sensor_pos
    rng = np.linalg.norm(delta)
    params = SENSOR_PARAMS[sensor_id]
    if rng > params["range_max_m"]:
        return False

    if params["fov_deg"] >= 360.0:
        return True

    bearing = np.arctan2(delta[1], delta[0])
    boresight = np.deg2rad(params["boresight_deg"])
    half_fov = 0.5 * np.deg2rad(params["fov_deg"])
    return abs(normalize_angle(float(bearing - boresight))) <= half_fov


def _measurement_from_state(
    sensor_id: str,
    target_state: np.ndarray,
    vessel_pos_ned: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sensor_pos = _sensor_position(sensor_id, vessel_pos_ned)
    delta_n = target_state[0] - sensor_pos[0]
    delta_e = target_state[1] - sensor_pos[1]
    true_range = np.sqrt(delta_n**2 + delta_e**2)
    true_bearing = np.arctan2(delta_e, delta_n)

    params = SENSOR_PARAMS[sensor_id]
    noisy_range = true_range + rng.normal(0.0, params["sigma_r_m"])
    noisy_bearing = normalize_angle(
        true_bearing + rng.normal(0.0, np.deg2rad(params["sigma_b_deg"]))
    )
    return np.array([noisy_range, noisy_bearing], dtype=float)


def _false_alarm_measurement(
    sensor_id: str,
    vessel_pos_ned: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    params = SENSOR_PARAMS[sensor_id]
    false_range = rng.uniform(40.0, params["range_max_m"])
    if params["fov_deg"] >= 360.0:
        false_bearing = rng.uniform(-np.pi, np.pi)
    else:
        boresight = np.deg2rad(params["boresight_deg"])
        half_fov = 0.5 * np.deg2rad(params["fov_deg"])
        false_bearing = boresight + rng.uniform(-half_fov, half_fov)
    return np.array([false_range, normalize_angle(false_bearing)], dtype=float)


def _sensor_availability(time_s: float) -> dict[str, bool]:
    # Intentional outages to exercise T6 availability handling.
    camera_available = not (40.0 <= time_s <= 55.0 or 90.0 <= time_s <= 100.0)
    ais_available = not (60.0 <= time_s <= 80.0)
    return {"radar": True, "camera": camera_available, "ais": ais_available}


def generate_task6_fake_scans(config: FakeScenarioConfig) -> list[FakeScan]:
    """
    Generate synthetic multi-sensor scans for a T6-only sandbox.

    The output mimics conflict-heavy Scenario D behavior with crossing targets.
    """

    rng = np.random.default_rng(config.seed)
    targets = _target_specs()

    scans: list[FakeScan] = []
    scan_idx = 0
    time_grid = np.arange(0.0, config.duration_s + config.dt_s, config.dt_s)
    for time_s in time_grid:
        vessel_pos = _vessel_position(float(time_s))
        truth_states = {t.target_id: _truth_state(t, float(time_s)) for t in targets}
        availability = _sensor_availability(float(time_s))

        detections: list[Detection] = []
        for sensor_id in ("radar", "camera", "ais"):
            params = SENSOR_PARAMS[sensor_id]
            for target in targets:
                if sensor_id == "ais" and not target.has_ais:
                    continue
                state = truth_states[target.target_id]
                if not _is_visible(sensor_id, state[:2], vessel_pos):
                    continue
                if rng.random() > params["pd"]:
                    continue

                z = _measurement_from_state(sensor_id, state, vessel_pos, rng)
                detections.append(
                    Detection(
                        detection_id=f"{sensor_id}_{scan_idx}_{len(detections)}",
                        time_s=float(time_s),
                        sensor_id=sensor_id,
                        z=z,
                        R=_sensor_R(sensor_id),
                        truth_id=target.target_id,
                        is_false_alarm=False,
                    )
                )

            if config.include_false_alarms and params["lambda_fa"] > 0.0:
                num_false = int(rng.poisson(params["lambda_fa"]))
                for _ in range(num_false):
                    z_false = _false_alarm_measurement(sensor_id, vessel_pos, rng)
                    detections.append(
                        Detection(
                            detection_id=f"{sensor_id}_{scan_idx}_{len(detections)}",
                            time_s=float(time_s),
                            sensor_id=sensor_id,
                            z=z_false,
                            R=_sensor_R(sensor_id),
                            truth_id=None,
                            is_false_alarm=True,
                        )
                    )

        scans.append(
            FakeScan(
                time_s=float(time_s),
                detections=detections,
                sensor_available=availability,
                truth_states=truth_states,
                vessel_pos_ned=vessel_pos,
            )
        )
        scan_idx += 1

    return scans


def initialize_tracks_from_truth(
    scans: list[FakeScan],
    seed: int = 123,
    position_sigma_m: float = 20.0,
    velocity_sigma_mps: float = 0.8,
) -> list[Track]:
    """
    Build initial tracks from t=0 truth + perturbation.

    This bypasses T7 track initiation on purpose to keep this sandbox focused
    on Task 6 gating and association only.
    """

    if not scans:
        return []

    rng = np.random.default_rng(seed)
    first = scans[0]
    tracks: list[Track] = []
    for truth_id, truth_state in sorted(first.truth_states.items()):
        x0 = truth_state.copy()
        x0[0] += rng.normal(0.0, position_sigma_m)
        x0[1] += rng.normal(0.0, position_sigma_m)
        x0[2] += rng.normal(0.0, velocity_sigma_mps)
        x0[3] += rng.normal(0.0, velocity_sigma_mps)

        P0 = np.diag(
            [
                position_sigma_m**2,
                position_sigma_m**2,
                velocity_sigma_mps**2,
                velocity_sigma_mps**2,
            ]
        )
        tracks.append(
            Track(
                track_id=truth_id,
                x=x0,
                P=P0,
                last_time_s=first.time_s,
                missed_count=0,
                status="confirmed",
                truth_id=truth_id,
            )
        )
    return tracks

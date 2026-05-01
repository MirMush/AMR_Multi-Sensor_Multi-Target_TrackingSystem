#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASK6_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (TASK6_ROOT, PROJECT_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tracking import (  # noqa: E402
    CVEKFHooks,
    CoordFrameMeasurementModel,
    FakeScenarioConfig,
    generate_task6_fake_scans,
    initialize_tracks_from_truth,
    run_fusion_cycle,
)

RADAR_POS = np.array([0.0, 0.0], dtype=float)
CAMERA_POS = np.array([-80.0, 120.0], dtype=float)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Task 6 sandbox behavior.")
    parser.add_argument("--duration-s", type=float, default=120.0, help="Scenario duration in seconds.")
    parser.add_argument("--dt-s", type=float, default=1.0, help="Scan interval in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for fake data generation.")
    parser.add_argument(
        "--gate-probability",
        type=float,
        default=0.99,
        help="Mahalanobis gating probability (chi-square threshold).",
    )
    parser.add_argument(
        "--no-false-alarms",
        action="store_true",
        help="Disable false alarms in fake data generation.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/task6_viz",
        help="Directory where plot images are saved.",
    )
    return parser


def _sensor_origin(sensor_id: str, vessel_pos_ned: np.ndarray) -> np.ndarray:
    if sensor_id == "radar":
        return RADAR_POS
    if sensor_id == "camera":
        return CAMERA_POS
    if sensor_id == "ais":
        return vessel_pos_ned
    raise ValueError(f"Unknown sensor_id: {sensor_id}")


def _detection_to_ned(sensor_id: str, z: np.ndarray, vessel_pos_ned: np.ndarray) -> np.ndarray:
    origin = _sensor_origin(sensor_id, vessel_pos_ned)
    r = float(z[0])
    b = float(z[1])
    north = origin[0] + r * np.cos(b)
    east = origin[1] + r * np.sin(b)
    return np.array([north, east], dtype=float)


def _to_array(paths: dict[int, list[np.ndarray]], key: int) -> np.ndarray:
    return np.array(paths[key], dtype=float) if paths[key] else np.zeros((0, 2), dtype=float)


def _plot_map(
    out_path: Path,
    truth_paths: dict[int, list[np.ndarray]],
    track_paths: dict[int, list[np.ndarray]],
    vessel_path: np.ndarray,
    det_points: dict[str, list[np.ndarray]],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for idx, target_id in enumerate(sorted(truth_paths)):
        truth_arr = _to_array(truth_paths, target_id)
        track_arr = _to_array(track_paths, target_id)
        color = colors[idx % len(colors)]

        if truth_arr.shape[0] > 0:
            ax.plot(
                truth_arr[:, 1],
                truth_arr[:, 0],
                linestyle="--",
                linewidth=2.0,
                color=color,
                alpha=0.9,
                label=f"Truth T{target_id}",
            )
        if track_arr.shape[0] > 0:
            ax.plot(
                track_arr[:, 1],
                track_arr[:, 0],
                linestyle="-",
                linewidth=1.8,
                color=color,
                alpha=1.0,
                label=f"Track T{target_id}",
            )

    sensor_style = {
        "used_radar": ("tab:cyan", ".", "Used radar det"),
        "used_camera": ("tab:pink", ".", "Used camera det"),
        "used_ais": ("tab:olive", ".", "Used AIS det"),
        "skipped": ("tab:gray", "x", "Skipped det (sensor off)"),
        "false_alarm": ("black", "x", "False alarm"),
    }
    for key, points in det_points.items():
        if not points:
            continue
        arr = np.array(points, dtype=float)
        color, marker, label = sensor_style[key]
        ax.scatter(
            arr[:, 1],
            arr[:, 0],
            s=12,
            alpha=0.18 if marker == "." else 0.35,
            c=color,
            marker=marker,
            linewidths=0.7,
            label=label,
        )

    ax.scatter(0.0, 0.0, c="black", marker="*", s=180, label="Radar position")
    ax.scatter(CAMERA_POS[1], CAMERA_POS[0], c="gold", marker="^", s=140, label="Camera position")

    if vessel_path.shape[0] > 0:
        ax.plot(vessel_path[:, 1], vessel_path[:, 0], color="tab:gray", linewidth=1.2, label="Vessel path")

    ax.set_title("Task 6 Sandbox Map: Truth vs Track + Detections")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stats(
    out_path: Path,
    times: np.ndarray,
    det_count: np.ndarray,
    avail_count: np.ndarray,
    skipped_count: np.ndarray,
    gated_count: np.ndarray,
    match_count: np.ndarray,
    unmatched_track_count: np.ndarray,
    unmatched_det_count: np.ndarray,
    consistent_count: np.ndarray,
    conflicting_count: np.ndarray,
    camera_avail: np.ndarray,
    ais_avail: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(times, det_count, label="All detections", color="tab:blue")
    axes[0].plot(times, avail_count, label="Available detections", color="tab:green")
    axes[0].plot(times, skipped_count, label="Skipped detections", color="tab:gray")
    axes[0].set_ylabel("Detections")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, gated_count, label="Gated pairs", color="tab:orange")
    axes[1].plot(times, match_count, label="Matches", color="tab:purple")
    axes[1].plot(times, unmatched_track_count, label="Unmatched tracks", color="tab:red")
    axes[1].plot(times, unmatched_det_count, label="Unmatched detections", color="tab:brown")
    axes[1].plot(times, consistent_count, label="Truth-consistent matches", color="tab:green", alpha=0.8)
    axes[1].plot(times, conflicting_count, label="Conflicting matches", color="black", alpha=0.7)
    axes[1].axvspan(50.0, 70.0, alpha=0.12, color="tab:red", label="Crossing window")
    axes[1].set_ylabel("Association")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    axes[2].step(times, camera_avail, where="post", label="Camera available", color="tab:pink")
    axes[2].step(times, ais_avail, where="post", label="AIS available", color="tab:olive")
    axes[2].set_ylabel("Availability")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_yticks([0.0, 1.0], ["Off", "On"])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle("Task 6 Sandbox Diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = FakeScenarioConfig(
        duration_s=args.duration_s,
        dt_s=args.dt_s,
        seed=args.seed,
        include_false_alarms=not args.no_false_alarms,
    )
    scans = generate_task6_fake_scans(config)
    tracks = initialize_tracks_from_truth(scans, seed=args.seed + 101)

    measurement_model = CoordFrameMeasurementModel()
    ekf_hooks = CVEKFHooks(measurement_model=measurement_model, sigma_a_mps2=0.4)

    truth_paths: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(tracks))}
    track_paths: dict[int, list[np.ndarray]] = {track.track_id: [track.x[:2].copy()] for track in tracks}
    vessel_points: list[np.ndarray] = []

    det_points: dict[str, list[np.ndarray]] = {
        "used_radar": [],
        "used_camera": [],
        "used_ais": [],
        "skipped": [],
        "false_alarm": [],
    }

    time_hist: list[float] = []
    det_count_hist: list[int] = []
    avail_count_hist: list[int] = []
    skipped_count_hist: list[int] = []
    gated_count_hist: list[int] = []
    match_count_hist: list[int] = []
    unmatched_track_hist: list[int] = []
    unmatched_det_hist: list[int] = []
    consistent_hist: list[int] = []
    conflicting_hist: list[int] = []
    camera_avail_hist: list[float] = []
    ais_avail_hist: list[float] = []

    for scan in scans:
        vessel_points.append(scan.vessel_pos_ned.copy())
        measurement_model.set_vessel_position(scan.vessel_pos_ned[0], scan.vessel_pos_ned[1])

        for target_id, state in scan.truth_states.items():
            if target_id not in truth_paths:
                truth_paths[target_id] = []
            truth_paths[target_id].append(state[:2].copy())

        cycle_result = run_fusion_cycle(
            time_s=scan.time_s,
            tracks=tracks,
            detections=scan.detections,
            sensor_available=scan.sensor_available,
            measurement_model=measurement_model,
            ekf_hooks=ekf_hooks,
            gate_probability=args.gate_probability,
        )
        tracks = cycle_result.updated_tracks

        consistent = 0
        conflicting = 0
        for track_idx, det_idx, _ in cycle_result.association.matches:
            track = tracks[track_idx]
            det = cycle_result.available_detections[det_idx]
            if det.truth_id is None:
                conflicting += 1
            elif track.truth_id == det.truth_id:
                consistent += 1
            else:
                conflicting += 1

        for detection in cycle_result.available_detections:
            point = _detection_to_ned(detection.sensor_id, detection.z, scan.vessel_pos_ned)
            if detection.is_false_alarm:
                det_points["false_alarm"].append(point)
            else:
                det_points[f"used_{detection.sensor_id}"].append(point)
        for detection in cycle_result.skipped_detections:
            point = _detection_to_ned(detection.sensor_id, detection.z, scan.vessel_pos_ned)
            det_points["skipped"].append(point)

        for track in tracks:
            if track.track_id not in track_paths:
                track_paths[track.track_id] = []
            track_paths[track.track_id].append(track.x[:2].copy())

        time_hist.append(scan.time_s)
        det_count_hist.append(len(scan.detections))
        avail_count_hist.append(len(cycle_result.available_detections))
        skipped_count_hist.append(len(cycle_result.skipped_detections))
        gated_count_hist.append(len(cycle_result.gated_candidates))
        match_count_hist.append(len(cycle_result.association.matches))
        unmatched_track_hist.append(len(cycle_result.association.unmatched_track_indices))
        unmatched_det_hist.append(len(cycle_result.association.unmatched_detection_indices))
        consistent_hist.append(consistent)
        conflicting_hist.append(conflicting)
        camera_avail_hist.append(1.0 if scan.sensor_available.get("camera", True) else 0.0)
        ais_avail_hist.append(1.0 if scan.sensor_available.get("ais", True) else 0.0)

    map_out = out_dir / "task6_map.png"
    stats_out = out_dir / "task6_stats.png"

    vessel_path = np.array(vessel_points, dtype=float) if vessel_points else np.zeros((0, 2), dtype=float)
    _plot_map(map_out, truth_paths, track_paths, vessel_path, det_points)
    _plot_stats(
        stats_out,
        np.array(time_hist, dtype=float),
        np.array(det_count_hist, dtype=float),
        np.array(avail_count_hist, dtype=float),
        np.array(skipped_count_hist, dtype=float),
        np.array(gated_count_hist, dtype=float),
        np.array(match_count_hist, dtype=float),
        np.array(unmatched_track_hist, dtype=float),
        np.array(unmatched_det_hist, dtype=float),
        np.array(consistent_hist, dtype=float),
        np.array(conflicting_hist, dtype=float),
        np.array(camera_avail_hist, dtype=float),
        np.array(ais_avail_hist, dtype=float),
    )

    print("Visualization complete.")
    print(f"- Map plot:   {map_out}")
    print(f"- Stats plot: {stats_out}")


if __name__ == "__main__":
    main()

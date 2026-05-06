#!/usr/bin/env python3
"""
T6 — Gating and data association validation on Scenario D.

Tests the core multi-target pipeline:
  - Mahalanobis-distance gating per track per sensor
  - Data association (Hungarian algorithm)
  - Track initialization from detections
  - Reports: gating statistics, association accuracy, MOTP, CE
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent
TASK6_ROOT = PROJECT_ROOT / "task6"
for _p in (PROJECT_ROOT, TASK6_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from task6.tracking import CVEKFHooks, CoordFrameMeasurementModel, run_fusion_cycle
from task6.tracking.types import Detection, Track

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSON_PATH = PROJECT_ROOT / "harbour_sim_output" / "scenario_D.json"
DT = 1.0
GATE_PROBABILITY = 0.99
SIGMA_A = 0.05

RADAR_POS = np.array([0.0, 0.0])
CAMERA_POS = np.array([-80.0, 120.0])

_det_counter = 0


def load_scenario(json_path: Path):
    with open(json_path) as f:
        return json.load(f)


def get_vessel_pos(vessel_positions, t: float):
    times = np.array([row[0] for row in vessel_positions], dtype=float)
    idx = int(np.argmin(np.abs(times - t)))
    return float(vessel_positions[idx][1]), float(vessel_positions[idx][2])


def make_detections(t, meas_list, mm: CoordFrameMeasurementModel) -> list[Detection]:
    global _det_counter
    dets = []
    for m in meas_list:
        sid = m["sensor_id"]
        if sid not in ("radar", "camera"):
            continue
        z        = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
        _, _, R  = mm.predict(np.zeros(4), sid)
        dets.append(Detection(
            detection_id   = f"{sid}_{_det_counter}",
            time_s         = t,
            sensor_id      = sid,
            z              = z,
            R              = R,
            truth_id       = m["target_id"] if not m["is_false_alarm"] else None,
            is_false_alarm = m["is_false_alarm"],
        ))
        _det_counter += 1
    return dets


def compute_motp(tracks: list[Track], gt_states: dict) -> float | None:
    """Match confirmed tracks to ground truth via Hungarian, return mean distance."""
    if not tracks or not gt_states:
        return None
    track_pos = np.array([t.x[:2] for t in tracks])
    gt_pos = np.array(list(gt_states.values()))[:, :2]
    M, N = len(track_pos), len(gt_pos)
    cost = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cost[i, j] = np.linalg.norm(track_pos[i] - gt_pos[j])
    rows, cols = linear_sum_assignment(cost)
    return float(np.mean(cost[rows, cols])) if len(rows) else None


def det_to_ned(sensor_id: str, z: np.ndarray, vessel_ned: np.ndarray) -> np.ndarray:
    """Convert [range, bearing] measurement to NED position."""
    if sensor_id == "radar":
        origin = RADAR_POS
    elif sensor_id == "camera":
        origin = CAMERA_POS
    else:
        origin = vessel_ned
    return origin + np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])


def run_task6_validation() -> None:
    global _det_counter
    _det_counter = 0

    # Load scenario
    with open(JSON_PATH) as f:
        data = json.load(f)
    t_end = float(data["t_end"])
    vessel_positions = data["vessel_positions"]
    gt_data = data["ground_truth"]

    # Ground truth lookup
    gt_times = {int(k): np.array([r[0] for r in v], dtype=float) for k, v in gt_data.items()}
    gt_arr = {int(k): np.array([r[1:] for r in v], dtype=float) for k, v in gt_data.items()}

    def get_gt(tid, t):
        if tid not in gt_times:
            return None
        idx = int(np.argmin(np.abs(gt_times[tid] - t)))
        state = gt_arr[tid][idx]
        return None if np.any(np.isnan(state)) else state

    # Measurement grouping
    meas_sorted = sorted(
        [(float(m["time"]), m) for m in data["measurements"]
         if m["sensor_id"] in ("radar", "camera")],
        key=lambda x: x[0],
    )

    def _win(t_hi: float) -> list:
        return [m for ts, m in meas_sorted if t_hi - DT < ts <= t_hi]

    # -----------------------------------------------------------------------
    # Initialize tracks from ground truth at t=0
    # -----------------------------------------------------------------------
    tracks = []
    for tid_str, rows in gt_data.items():
        tid = int(tid_str)
        row0 = next((r for r in rows if not any(np.isnan(r[1:]))), None)
        if row0 is None:
            continue
        x0 = np.array(row0[1:], dtype=float)
        P0 = np.diag([50.0**2, 50.0**2, 5.0**2, 5.0**2])
        tracks.append(Track(
            track_id=tid,
            x=x0,
            P=P0,
            last_time_s=float(row0[0]),
            truth_id=tid,
        ))

    # -----------------------------------------------------------------------
    # Setup fusion components
    # -----------------------------------------------------------------------
    mm = CoordFrameMeasurementModel()
    ekf_hooks = CVEKFHooks(measurement_model=mm, sigma_a_mps2=SIGMA_A)

    # History for metrics
    time_hist = []
    det_count_hist = []
    gated_count_hist = []
    match_count_hist = []
    unmatched_track_hist = []
    unmatched_det_hist = []
    consistent_hist = []
    conflicting_hist = []
    motp_hist = []
    ce_hist = []

    # For maps
    truth_paths: dict[int, list] = defaultdict(list)
    track_paths: dict[int, list] = defaultdict(list)
    vessel_path: list = []
    det_ned: dict[str, list] = {"radar": [], "camera": [], "false_alarm": []}

    print("\n" + "="*70)
    print("T6 — Gating & Data Association Validation on Scenario D")
    print(f"Gate probability: {GATE_PROBABILITY}, Initial tracks: {len(tracks)}")
    print("="*70)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for t_idx, t in enumerate(np.arange(DT, t_end + DT, DT)):
        t = round(float(t), 6)

        # Vessel position
        pN, pE = get_vessel_pos(vessel_positions, t)
        vessel_ned = np.array([pN, pE])
        vessel_path.append(vessel_ned.copy())
        mm.set_vessel_position(pN, pE)

        # Ground truth paths
        for tid in gt_times:
            s = get_gt(tid, t)
            if s is not None:
                truth_paths[tid].append(s[:2].copy())

        # Detections
        dets = make_detections(t, _win(t), mm)

        # Fusion cycle
        cycle_result = run_fusion_cycle(
            time_s=t,
            tracks=tracks,
            detections=dets,
            sensor_available={"radar": True, "camera": True},
            measurement_model=mm,
            ekf_hooks=ekf_hooks,
            gate_probability=GATE_PROBABILITY,
        )
        tracks = cycle_result.updated_tracks

        # Track paths
        for tr in tracks:
            if tr.track_id not in track_paths:
                track_paths[tr.track_id] = []
            track_paths[tr.track_id].append(tr.x[:2].copy())

        # Detection positions for map
        for d in cycle_result.available_detections:
            pt = det_to_ned(d.sensor_id, d.z, vessel_ned)
            if d.is_false_alarm:
                det_ned["false_alarm"].append(pt)
            else:
                det_ned[d.sensor_id].append(pt)

        # Association accuracy
        consistent = 0
        conflicting = 0
        for track_idx, det_idx, _ in cycle_result.association.matches:
            track = tracks[track_idx]
            det = cycle_result.available_detections[det_idx]
            if det.truth_id is not None and track.truth_id == det.truth_id:
                consistent += 1
            else:
                conflicting += 1

        # Metrics
        active_gt = {tid: get_gt(tid, t) for tid in gt_times if get_gt(tid, t) is not None}
        motp_val = compute_motp(tracks, active_gt)
        ce_val = abs(len(tracks) - len(active_gt))

        motp_hist.append(motp_val)
        ce_hist.append(ce_val)
        time_hist.append(t)
        det_count_hist.append(len(dets))
        gated_count_hist.append(len(cycle_result.gated_candidates))
        match_count_hist.append(len(cycle_result.association.matches))
        unmatched_track_hist.append(len(cycle_result.association.unmatched_track_indices))
        unmatched_det_hist.append(len(cycle_result.association.unmatched_detection_indices))
        consistent_hist.append(consistent)
        conflicting_hist.append(conflicting)

        if t_idx % 20 == 0:
            print(
                f"[t={t:6.1f}s] det={len(dets):2d} gated={len(cycle_result.gated_candidates):2d} "
                f"match={len(cycle_result.association.matches):2d} "
                f"unmatch_tr={len(cycle_result.association.unmatched_track_indices):2d} "
                f"unmatch_det={len(cycle_result.association.unmatched_detection_indices):2d}"
            )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    motp_valid = [v for v in motp_hist if v is not None]
    ce_arr = np.array(ce_hist, dtype=float)
    motp_mean = float(np.mean(motp_valid)) if motp_valid else float("nan")
    ce_mean = float(np.mean(ce_arr))
    
    n_cycles = max(len(time_hist), 1)
    total_gated = sum(gated_count_hist)
    total_matches = sum(match_count_hist)
    total_consistent = sum(consistent_hist)
    total_conflicting = sum(conflicting_hist)

    print("\n" + "="*70)
    print("SUMMARY — T6 Gating & Association")
    print(f"  Cycles                         : {n_cycles}")
    print(f"  Mean detections/cycle          : {sum(det_count_hist) / n_cycles:.2f}")
    print(f"  Mean gated pairs/cycle         : {total_gated / n_cycles:.2f}")
    print(f"  Mean matches/cycle             : {total_matches / n_cycles:.2f}")
    print(f"  Total consistent matches       : {total_consistent}")
    print(f"  Total conflicting matches      : {total_conflicting}")
    if total_consistent + total_conflicting > 0:
        pct = 100.0 * total_consistent / (total_consistent + total_conflicting)
        print(f"  Association accuracy           : {pct:.1f}%")
    print(f"  MOTP (mean position error)     : {motp_mean:.2f} m  (target < 15 m)  {'✓ PASS' if motp_mean < 15 else '✗ FAIL'}")
    print(f"  CE   (cardinality error)       : {ce_mean:.3f}    (target < 0.5)   {'✓ PASS' if ce_mean < 0.5 else '✗ FAIL'}")
    print("="*70)

    # -----------------------------------------------------------------------
    # Plots 
    # -----------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "harbour_sim_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("T6 — Gating & Data Association (Scenario D)", fontsize=14)
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    t_arr = np.array(time_hist)

    # Panel 0: Trajectory map (NED)
    ax = axes[0]
    for i, (tid, path) in enumerate(sorted(truth_paths.items())):
        if not path:
            continue
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0], "--", lw=2.0, color=colors[i % len(colors)],
                alpha=0.8, label=f"Truth T{tid}")

    for i, (tid, path) in enumerate(sorted(track_paths.items())):
        if not path:
            continue
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0], "-", lw=1.8, color=colors[i % len(colors)],
                label=f"Track T{tid}")

    for key, pts in det_ned.items():
        if not pts:
            continue
        arr = np.array(pts)
        styles = {
            "radar": ("tab:cyan", ".", 0.2),
            "camera": ("tab:pink", ".", 0.2),
            "false_alarm": ("black", "x", 0.4),
        }
        col, mk, alpha = styles[key]
        ax.scatter(arr[:, 1], arr[:, 0], s=12, c=col, marker=mk, alpha=alpha, label=f"{key}")

    if vessel_path:
        vp = np.array(vessel_path)
        ax.plot(vp[:, 1], vp[:, 0], color="gray", lw=1, alpha=0.6, label="Vessel")

    ax.scatter([0], [0], c="black", marker="*", s=180, zorder=5, label="Radar")
    ax.scatter([CAMERA_POS[1]], [CAMERA_POS[0]], c="gold", marker="^", s=140, zorder=5, label="Camera")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Trajectories (NED)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")

    # Panel 1: MOTP & CE
    ax = axes[1]
    motp_plot = np.array([v if v is not None else np.nan for v in motp_hist])
    ax.plot(t_arr, motp_plot, "o-", color="tab:blue", markersize=3, label=f"MOTP (mean={motp_mean:.2f} m)")
    ax.axhline(15, color="red", linestyle="--", linewidth=1.2, label="MOTP target (15 m)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("MOTP [m]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Position Accuracy (MOTP)")

    # Panel 2: Association & gating metrics
    ax = axes[2]
    ax.plot(t_arr, match_count_hist, "o-", label="Matches", color="tab:purple", markersize=3)
    ax.plot(t_arr, consistent_hist, "d-", label="Consistent", color="tab:green", markersize=3, alpha=0.8)
    ax.plot(t_arr, conflicting_hist, "s-", label="Conflicting", color="tab:red", markersize=3, alpha=0.7)
    ax.plot(t_arr, gated_count_hist, "^-", label="Gated pairs", color="tab:orange", markersize=2.5, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Count")
    ax.set_title("Association & Gating")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "task6_scenario_D.png"
    fig.savefig(out_path, dpi=180)
    plt.show()

    print(f"\nPlot saved:")
    print(f"  {out_path}")


if __name__ == "__main__":
    run_task6_validation()

#!/usr/bin/env python3
"""
Task 6 — Gating and data association on Scenario D (scenario_D.json).

Runs the full pipeline and saves two plots:
  - task6_map.png   : trajectories + detections in NED
  - task6_stats.png : association diagnostics over time
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASK6_ROOT   = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (TASK6_ROOT, PROJECT_ROOT):
    p = str(_path)
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy.optimize import linear_sum_assignment  # noqa: E402

from tracking import (  # noqa: E402
    CVEKFHooks,
    CoordFrameMeasurementModel,
    run_fusion_cycle,
)
from tracking.types import Detection, Track  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSON_PATH        = PROJECT_ROOT / "harbour_sim_output" / "scenario_D.json"
OUT_DIR          = PROJECT_ROOT / "harbour_sim_output"
DT               = 1.0
GATE_PROBABILITY = 0.99
SIGMA_A          = 0.05
VERBOSE_EVERY    = 10

RADAR_POS  = np.array([0.0, 0.0])
CAMERA_POS = np.array([-80.0, 120.0])

# ---------------------------------------------------------------------------
# Load JSON
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    data = json.load(f)

t_end            = float(data["t_end"])
vessel_positions = data["vessel_positions"]
gt_data          = data["ground_truth"]

vp_times = np.array([row[0] for row in vessel_positions], dtype=float)

def get_vessel_pos(t: float) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(vp_times - t)))
    return float(vessel_positions[idx][1]), float(vessel_positions[idx][2])

# Ground truth lookup per target per tick
gt_times_by_target = {}
gt_array_by_target = {}
for tid_str, rows in gt_data.items():
    gt_times_by_target[int(tid_str)] = np.array([r[0] for r in rows], dtype=float)
    gt_array_by_target[int(tid_str)] = np.array([r[1:] for r in rows], dtype=float)

def get_gt_state(tid: int, t: float) -> np.ndarray | None:
    if tid not in gt_times_by_target:
        return None
    idx = int(np.argmin(np.abs(gt_times_by_target[tid] - t)))
    state = gt_array_by_target[tid][idx]
    if np.any(np.isnan(state)):
        return None
    return state

# ---------------------------------------------------------------------------
# Group measurements by timestamp
# ---------------------------------------------------------------------------
meas_sorted: list[tuple[float, dict]] = sorted(
    [(float(m["time"]), m) for m in data["measurements"]
     if m["sensor_id"] in ("radar", "camera")],
    key=lambda x: x[0],
)

def _meas_in_window(t_hi: float) -> list[dict]:
    return [m for ts, m in meas_sorted if t_hi - DT < ts <= t_hi]

# ---------------------------------------------------------------------------
# Detection factory
# ---------------------------------------------------------------------------
_det_counter = 0

def make_detections(t: float, meas_list: list[dict], mm: CoordFrameMeasurementModel) -> list[Detection]:
    global _det_counter
    detections = []
    for m in meas_list:
        sid = m["sensor_id"]
        z   = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
        _, _, R = mm.predict(np.zeros(4), sid)
        detections.append(Detection(
            detection_id  = f"{sid}_{_det_counter}",
            time_s        = t,
            sensor_id     = sid,
            z             = z,
            R             = R,
            truth_id      = m["target_id"] if not m["is_false_alarm"] else None,
            is_false_alarm= m["is_false_alarm"],
        ))
        _det_counter += 1
    return detections

# ---------------------------------------------------------------------------
# Initialize tracks from ground truth at t=0
# ---------------------------------------------------------------------------
def initialize_tracks(gt_data: dict) -> list[Track]:
    tracks = []
    for tid_str, rows in gt_data.items():
        tid  = int(tid_str)
        row0 = next((r for r in rows if not any(np.isnan(r[1:]))), None)
        if row0 is None:
            continue
        x0 = np.array(row0[1:], dtype=float)
        P0 = np.diag([50.0**2, 50.0**2, 5.0**2, 5.0**2])
        tracks.append(Track(
            track_id    = tid,
            x           = x0,
            P           = P0,
            last_time_s = float(row0[0]),
            truth_id    = tid,
        ))
    return tracks

# ---------------------------------------------------------------------------
# MOTP helper: match tracks to ground truth via Hungarian, return mean dist
# ---------------------------------------------------------------------------
def compute_motp(tracks: list[Track], gt_states: dict[int, np.ndarray]) -> float | None:
    """
    Match confirmed tracks to active ground-truth targets via minimum-distance
    Hungarian assignment. Returns mean position error over all matched pairs,
    or None if there are no tracks or no targets.
    """
    if not tracks or not gt_states:
        return None

    track_pos = np.array([t.x[:2] for t in tracks])          # (M, 2)
    gt_pos    = np.array(list(gt_states.values()))[:, :2]     # (N, 2)

    # Cost matrix: Euclidean distance between every track-target pair
    M, N = len(track_pos), len(gt_pos)
    cost = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cost[i, j] = np.linalg.norm(track_pos[i] - gt_pos[j])

    rows, cols = linear_sum_assignment(cost)
    if len(rows) == 0:
        return None
    return float(np.mean(cost[rows, cols]))

# ---------------------------------------------------------------------------
# Helper: convert [range, bearing] + sensor origin → NED position for plotting
# ---------------------------------------------------------------------------
def det_to_ned(sensor_id: str, z: np.ndarray, vessel_pos: np.ndarray) -> np.ndarray:
    origin = vessel_pos if sensor_id == "ais" else (RADAR_POS if sensor_id == "radar" else CAMERA_POS)
    return origin + np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
measurement_model = CoordFrameMeasurementModel()
ekf_hooks         = CVEKFHooks(measurement_model=measurement_model, sigma_a_mps2=SIGMA_A)
tracks            = initialize_tracks(gt_data)

# History for plots
truth_paths  : dict[int, list] = {t.truth_id: [] for t in tracks}
track_paths  : dict[int, list] = {t.track_id: [t.x[:2].copy()] for t in tracks}
vessel_points: list            = []

det_points = {"used_radar": [], "used_camera": [], "false_alarm": []}

time_hist             = []
det_count_hist        = []
gated_count_hist      = []
match_count_hist      = []
unmatched_track_hist  = []
unmatched_det_hist    = []
consistent_hist       = []
conflicting_hist      = []
motp_hist             = []   # MOTP per tick (None when no tracks/targets)
ce_hist               = []   # Cardinality Error per tick

# Summary counters
total_gated       = 0
total_matches     = 0
truth_consistent  = 0
truth_conflicting = 0

print("=" * 60)
print("Task 6 — Scenario D (scenario_D.json)")
print(f"Targets: {len(tracks)},  t_end: {t_end}s,  gate_prob: {GATE_PROBABILITY}")
print("=" * 60)

for idx, t in enumerate(np.arange(1.0, t_end + DT, DT)):
    t = round(float(t), 6)

    pN_v, pE_v = get_vessel_pos(t)
    vessel_ned = np.array([pN_v, pE_v])
    measurement_model.set_vessel_position(pN_v, pE_v)
    vessel_points.append(vessel_ned.copy())

    # Ground truth positions this tick
    for tid, path in truth_paths.items():
        state = get_gt_state(tid, t)
        if state is not None:
            path.append(state[:2].copy())

    detections = make_detections(t, _meas_in_window(t), measurement_model)

    cycle_result = run_fusion_cycle(
        time_s            = t,
        tracks            = tracks,
        detections        = detections,
        sensor_available  = {"radar": True, "camera": True},
        measurement_model = measurement_model,
        ekf_hooks         = ekf_hooks,
        gate_probability  = GATE_PROBABILITY,
    )
    tracks = cycle_result.updated_tracks

    # Track paths
    for track in tracks:
        if track.track_id not in track_paths:
            track_paths[track.track_id] = []
        track_paths[track.track_id].append(track.x[:2].copy())

    # Detection points for map
    for det in cycle_result.available_detections:
        pt = det_to_ned(det.sensor_id, det.z, vessel_ned)
        if det.is_false_alarm:
            det_points["false_alarm"].append(pt)
        else:
            det_points[f"used_{det.sensor_id}"].append(pt)

    # Association accuracy
    consistent  = 0
    conflicting = 0
    for track_idx, det_idx, _ in cycle_result.association.matches:
        track = tracks[track_idx]
        det   = cycle_result.available_detections[det_idx]
        if det.truth_id is not None and track.truth_id == det.truth_id:
            consistent += 1
        else:
            conflicting += 1

    total_gated       += len(cycle_result.gated_candidates)
    total_matches     += len(cycle_result.association.matches)
    truth_consistent  += consistent
    truth_conflicting += conflicting

    # --- MOTP and CE ---
    active_gt = {tid: get_gt_state(tid, t) for tid in gt_times_by_target
                 if get_gt_state(tid, t) is not None}
    motp_val = compute_motp(tracks, active_gt)
    ce_val   = abs(len(tracks) - len(active_gt))

    motp_hist.append(motp_val)
    ce_hist.append(ce_val)

    time_hist.append(t)
    det_count_hist.append(len(detections))
    gated_count_hist.append(len(cycle_result.gated_candidates))
    match_count_hist.append(len(cycle_result.association.matches))
    unmatched_track_hist.append(len(cycle_result.association.unmatched_track_indices))
    unmatched_det_hist.append(len(cycle_result.association.unmatched_detection_indices))
    consistent_hist.append(consistent)
    conflicting_hist.append(conflicting)

    if idx % max(VERBOSE_EVERY, 1) == 0:
        print(
            f"[t={t:6.1f}s] det={len(detections):2d} "
            f"gated={len(cycle_result.gated_candidates):2d} "
            f"match={len(cycle_result.association.matches):2d} "
            f"unmatched_tracks={len(cycle_result.association.unmatched_track_indices):2d} "
            f"unmatched_det={len(cycle_result.association.unmatched_detection_indices):2d}"
        )

n_cycles = max(len(time_hist), 1)

motp_valid = [v for v in motp_hist if v is not None]
ce_arr     = np.array(ce_hist, dtype=float)
motp_mean  = float(np.mean(motp_valid)) if motp_valid else float("nan")
ce_mean    = float(np.mean(ce_arr))

print("\n" + "=" * 60)
print("Summary")
print(f"  Cycles                   : {n_cycles}")
print(f"  Mean gated pairs/cycle   : {total_gated / n_cycles:.2f}")
print(f"  Mean matches/cycle       : {total_matches / n_cycles:.2f}")
print(f"  Truth-consistent matches : {truth_consistent}")
print(f"  Truth-conflicting matches: {truth_conflicting}")
if truth_consistent + truth_conflicting > 0:
    pct = 100.0 * truth_consistent / (truth_consistent + truth_conflicting)
    print(f"  Association accuracy     : {pct:.1f}%")
print(f"  MOTP (mean position err) : {motp_mean:.2f} m  (target < 15 m)")
print(f"  CE   (cardinality error) : {ce_mean:.3f}    (target < 0.5)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Plot 1 — Trajectory map
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for i, (tid, path) in enumerate(sorted(truth_paths.items())):
    if not path:
        continue
    arr = np.array(path)
    c   = colors[i % len(colors)]
    ax.plot(arr[:, 1], arr[:, 0], "--", lw=2, color=c, alpha=0.7, label=f"Truth T{tid}")

for i, (tid, path) in enumerate(sorted(track_paths.items())):
    if not path:
        continue
    arr = np.array(path)
    c   = colors[i % len(colors)]
    ax.plot(arr[:, 1], arr[:, 0], "-",  lw=1.8, color=c, label=f"Track T{tid}")

for key, pts in det_points.items():
    if not pts:
        continue
    arr    = np.array(pts)
    styles = {
        "used_radar":  ("tab:cyan",  ".", "Radar det"),
        "used_camera": ("tab:pink",  ".", "Camera det"),
        "false_alarm": ("black",     "x", "False alarm"),
    }
    col, mk, lbl = styles[key]
    ax.scatter(arr[:, 1], arr[:, 0], s=12, c=col, marker=mk,
               alpha=0.25 if mk == "." else 0.4, label=lbl)

if vessel_points:
    vp = np.array(vessel_points)
    ax.plot(vp[:, 1], vp[:, 0], color="gray", lw=1, label="Vessel path")

ax.scatter([0], [0], c="black", marker="*", s=180, zorder=5, label="Radar")
ax.scatter([CAMERA_POS[1]], [CAMERA_POS[0]], c="gold", marker="^", s=140, zorder=5, label="Camera")
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_title("Scenario D — Trajectories + Detections")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
map_out = OUT_DIR / "task6_map.png"
fig.savefig(map_out, dpi=180)
plt.show()

# ---------------------------------------------------------------------------
# Plot 2 — Association diagnostics + MOTP + CE
# ---------------------------------------------------------------------------
times = np.array(time_hist)

# Replace None MOTP values with nan for plotting
motp_plot = np.array([v if v is not None else np.nan for v in motp_hist])

fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

axes[0].plot(times, det_count_hist,       label="All detections",      color="tab:blue")
axes[0].plot(times, gated_count_hist,     label="Gated pairs",         color="tab:orange")
axes[0].plot(times, match_count_hist,     label="Matches",             color="tab:purple")
axes[0].set_ylabel("Count")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(times, unmatched_track_hist, label="Unmatched tracks",    color="tab:red")
axes[1].plot(times, unmatched_det_hist,   label="Unmatched dets",      color="tab:brown")
axes[1].plot(times, consistent_hist,      label="Consistent matches",  color="tab:green")
axes[1].plot(times, conflicting_hist,     label="Conflicting matches", color="black", alpha=0.7)
axes[1].axvspan(50, 70, alpha=0.1, color="tab:red", label="Crossing window")
axes[1].set_ylabel("Count")
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2, fontsize=8)

axes[2].plot(times, motp_plot, color="tab:blue", label="MOTP")
axes[2].axhline(15, color="r", linestyle="--", label="Target (15 m)")
axes[2].set_ylabel("MOTP [m]")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

axes[3].plot(times, ce_arr, color="tab:orange", label="CE")
axes[3].axhline(0.5, color="r", linestyle="--", label="Target (0.5)")
axes[3].set_ylabel("Cardinality Error")
axes[3].set_xlabel("Time [s]")
axes[3].grid(True, alpha=0.3)
axes[3].legend()

fig.suptitle("Scenario D — Association Diagnostics")
fig.tight_layout()
stats_out = OUT_DIR / "task6_stats.png"
fig.savefig(stats_out, dpi=180)
plt.show()

print(f"\nPlots saved to:\n  {map_out}\n  {stats_out}")

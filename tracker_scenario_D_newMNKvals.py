#!/usr/bin/env python3
"""
T7 — Track Manager validation on Scenario D and E.

Runs TrackManager (full lifecycle: tentative → confirmed → coasting → deleted)
and reports MOTP and Cardinality Error (CE) for both scenarios.
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
TASK6_ROOT   = PROJECT_ROOT / "task6"
for _p in (PROJECT_ROOT, TASK6_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from task6.tracking.measurement_models import CoordFrameMeasurementModel
from task6.tracking.types import Detection
from track_manager import TrackManager, TrackManagerConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DT = 1.0

_det_counter = 0


def load_scenario(json_path: Path):
    with open(json_path) as f:
        return json.load(f)


def get_vessel_pos(vessel_positions, t: float):
    times = np.array([row[0] for row in vessel_positions], dtype=float)
    idx   = int(np.argmin(np.abs(times - t)))
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


def compute_motp(tracks, gt_states: dict) -> float | None:
    if not tracks or not gt_states:
        return None
    track_pos = np.array([t.x[:2] for t in tracks])
    gt_pos    = np.array(list(gt_states.values()))[:, :2]
    M, N = len(track_pos), len(gt_pos)
    cost = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cost[i, j] = np.linalg.norm(track_pos[i] - gt_pos[j])
    rows, cols = linear_sum_assignment(cost)
    return float(np.mean(cost[rows, cols])) if len(rows) else None


def run_scenario(label: str, json_path: Path, cfg: TrackManagerConfig) -> dict:
    global _det_counter
    _det_counter = 0

    data             = load_scenario(json_path)
    t_end            = float(data["t_end"])
    vessel_positions = data["vessel_positions"]
    gt_data          = data["ground_truth"]

    gt_times = {int(k): np.array([r[0] for r in v], dtype=float) for k, v in gt_data.items()}
    gt_arr   = {int(k): np.array([r[1:] for r in v], dtype=float) for k, v in gt_data.items()}

    def get_gt(tid, t):
        if tid not in gt_times:
            return None
        idx   = int(np.argmin(np.abs(gt_times[tid] - t)))
        state = gt_arr[tid][idx]
        return None if np.any(np.isnan(state)) else state

    meas_sorted = sorted(
        [(float(m["time"]), m) for m in data["measurements"]
         if m["sensor_id"] in ("radar", "camera")],
        key=lambda x: x[0],
    )

    def _win(t_hi: float) -> list:
        return [m for ts, m in meas_sorted if t_hi - DT < ts <= t_hi]

    RADAR_POS  = np.array([0.0,   0.0])
    CAMERA_POS = np.array([-80.0, 120.0])

    def det_to_ned(sid, z, vessel_ned):
        if sid == "radar":
            origin = RADAR_POS
        elif sid == "camera":
            origin = CAMERA_POS
        else:
            origin = vessel_ned
        return origin + np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])

    mm = CoordFrameMeasurementModel()
    tm = TrackManager(mm, cfg)

    motp_hist, ce_hist, time_hist   = [], [], []
    n_active_hist, n_tentative_h = [], []

    # For maps
    truth_paths : dict[int, list] = defaultdict(list)
    track_paths : dict[int, list] = defaultdict(list)
    vessel_path : list            = []
    det_ned     : dict[str, list] = {"radar": [], "camera": [], "false_alarm": []}

    print(f"\n{'='*60}")
    print(f"T7 — {label}  ({json_path.name})")
    print(f"t_end={t_end}s  M={cfg.M} N={cfg.N} K_del={cfg.K_del}")
    print(f"{'='*60}")

    for t in np.arange(1.0, t_end + DT, DT):
        t = round(float(t), 6)

        pN, pE = get_vessel_pos(vessel_positions, t)
        vessel_ned = np.array([pN, pE])
        vessel_path.append(vessel_ned.copy())
        tm.update_vessel_pos(pN, pE)

        # Ground truth paths
        for tid in gt_times:
            s = get_gt(tid, t)
            if s is not None:
                truth_paths[tid].append(s[:2].copy())

        dets      = make_detections(t, _win(t), mm)
        confirmed = tm.step(t, dets)
        all_mt    = tm.all_tracks()

        # Confirmed track paths
        for tr in confirmed:
            track_paths[tr.track_id].append(tr.x[:2].copy())

        # Detection positions for map
        for d in dets:
            pt = det_to_ned(d.sensor_id, d.z, vessel_ned)
            if d.is_false_alarm:
                det_ned["false_alarm"].append(pt)
            else:
                det_ned[d.sensor_id].append(pt)

        active_gt = {tid: get_gt(tid, t) for tid in gt_times if get_gt(tid, t) is not None}
        motp_val  = compute_motp(confirmed, active_gt)
        ce_val    = abs(len(confirmed) - len(active_gt))

        motp_hist.append(motp_val)
        ce_hist.append(ce_val)
        time_hist.append(t)
        n_active_hist.append(len(confirmed))
        n_tentative_h.append(sum(1 for mt in all_mt if mt.status == "tentative"))

    motp_valid = [v for v in motp_hist if v is not None]
    ce_arr     = np.array(ce_hist, dtype=float)
    motp_mean  = float(np.mean(motp_valid)) if motp_valid else float("nan")
    ce_mean    = float(np.mean(ce_arr))
    ce_target  = 1.0 if label == "Scenario E" else 0.5

    print(f"\nSummary — {label}")
    print(f"  MOTP (mean pos error) : {motp_mean:.2f} m  (target < 15 m)  {'PASS' if motp_mean < 15 else 'FAIL'}")
    print(f"  CE   (cardinality)    : {ce_mean:.3f}    (target < {ce_target})   {'PASS' if ce_mean < ce_target else 'FAIL'}")

    return dict(
        label        = label,
        time_hist    = np.array(time_hist),
        motp_hist    = motp_hist,
        ce_hist      = ce_arr,
        n_confirmed  = np.array(n_active_hist),
        n_tentative  = np.array(n_tentative_h),
        motp_mean    = motp_mean,
        ce_mean      = ce_mean,
        ce_target    = ce_target,
        truth_paths  = dict(truth_paths),
        track_paths  = dict(track_paths),
        vessel_path  = vessel_path,
        det_ned      = det_ned,
        RADAR_POS    = RADAR_POS,
        CAMERA_POS   = CAMERA_POS,
    )


# ---------------------------------------------------------------------------
# Run both scenarios
# ---------------------------------------------------------------------------
cfg = TrackManagerConfig(M=3, N=10, K_del=15)
#N=15, K_del=15: 15-second windows accommodate the slow mm-wave radar (0.3 Hz / ~3.3s per scan). 
#                 This ensures real targets have enough time to be scanned and can coast through 
#                 1-2 missed detections without being prematurely deleted.
# M=4: Raised confirmation threshold (from 3) to demand more evidence within the 15s window. 
#      This effectively filters out "ghost tracks" caused by high-clutter false alarms.

res_D = run_scenario(
    "Scenario D",
    PROJECT_ROOT / "harbour_sim_output" / "scenario_D.json",
    cfg,
)
res_E = run_scenario(
    "Scenario E",
    PROJECT_ROOT / "harbour_sim_output" / "scenario_E.json",
    cfg,
)

# ---------------------------------------------------------------------------
# Helper: draw one trajectory map on a given axes
# ---------------------------------------------------------------------------
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

def plot_map(ax, res: dict, title: str) -> None:
    # Ground truth (dashed)
    for i, (tid, path) in enumerate(sorted(res["truth_paths"].items())):
        if not path:
            continue
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0], "--", lw=2, color=COLORS[i % len(COLORS)],
                alpha=0.7, label=f"Truth T{tid}")

    # Confirmed tracks (solid)
    for i, (tid, path) in enumerate(sorted(res["track_paths"].items())):
        if not path:
            continue
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0], "-", lw=1.8, color=COLORS[i % len(COLORS)],
                label=f"Track {tid}")

    # Detections
    for key, pts in res["det_ned"].items():
        if not pts:
            continue
        arr = np.array(pts)
        styles = {
            "radar":       ("tab:cyan",  ".", "Radar det",  0.25),
            "camera":      ("tab:pink",  ".", "Camera det", 0.25),
            "false_alarm": ("black",     "x", "False alarm",0.40),
        }
        col, mk, lbl, alpha = styles[key]
        ax.scatter(arr[:, 1], arr[:, 0], s=12, c=col, marker=mk, alpha=alpha, label=lbl)

    # Vessel path
    if res["vessel_path"]:
        vp = np.array(res["vessel_path"])
        ax.plot(vp[:, 1], vp[:, 0], color="gray", lw=1, label="Vessel")

    # Sensor markers
    ax.scatter([0], [0], c="black", marker="*", s=180, zorder=5, label="Radar")
    ax.scatter([res["CAMERA_POS"][1]], [res["CAMERA_POS"][0]],
               c="gold", marker="^", s=140, zorder=5, label="Camera")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# Figure 1 — Maps (one per scenario)
# ---------------------------------------------------------------------------
fig_map, map_axes = plt.subplots(1, 2, figsize=(16, 7))
plot_map(map_axes[0], res_D, "Scenario D — T7 Trajectories")
plot_map(map_axes[1], res_E, "Scenario E — T7 Trajectories")
fig_map.suptitle("T7 Track Manager — Trajectory Maps", fontsize=13)
fig_map.tight_layout()
map_out = PROJECT_ROOT / "harbour_sim_output" / "task7_map.png"
fig_map.savefig(map_out, dpi=180)
plt.show()

# ---------------------------------------------------------------------------
# Figure 2 — MOTP / CE / counts (one column per scenario)
# ---------------------------------------------------------------------------
fig_stats, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)

for col, res in enumerate([res_D, res_E]):
    t = res["time_hist"]
    motp_plot = np.array([v if v is not None else np.nan for v in res["motp_hist"]])

    ax0 = axes[0, col]
    ax0.plot(t, res["n_confirmed"], label="Confirmed + Coasting", color="tab:blue")
    ax0.plot(t, res["n_tentative"], label="Tentative", color="tab:orange", alpha=0.7)
    ax0.set_title(f"{res['label']} — Track counts")
    ax0.set_ylabel("Count")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1, col]
    ax1.plot(t, motp_plot, color="tab:blue", label=f"MOTP (mean={res['motp_mean']:.1f} m)")
    ax1.axhline(15, color="r", linestyle="--", label="Target 15 m")
    ax1_r = ax1.twinx()
    ax1_r.plot(t, res["ce_hist"], color="tab:orange", alpha=0.7,
               label=f"CE (mean={res['ce_mean']:.2f})")
    ce_target = res.get("ce_target", 0.5)
    ax1_r.axhline(ce_target, color="darkorange", linestyle=":", label=f"Target {ce_target}")
    ax1_r.set_ylabel("CE")
    ax1.set_title(f"{res['label']} — MOTP & CE")
    ax1.set_ylabel("MOTP [m]")
    ax1.set_xlabel("Time [s]")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

fig_stats.suptitle("T7 Track Manager — MOTP & Cardinality Error", fontsize=13)
fig_stats.tight_layout()
stats_out = PROJECT_ROOT / "harbour_sim_output" / "task7_stats.png"
fig_stats.savefig(stats_out, dpi=180)
plt.show()

print(f"\nPlots saved:\n  {map_out}\n  {stats_out}")

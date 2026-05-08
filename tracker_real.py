#!/usr/bin/env python3
"""
T7 — Track Manager validation on real harbour data.

No ground truth available. AIS positions (targets 1, 3, 18) are used as
visual reference only. GNSS (target_id=-1) provides vessel position.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
TASK6_ROOT   = PROJECT_ROOT / "task6"
for _p in (PROJECT_ROOT, TASK6_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from task6.tracking.measurement_models import CoordFrameMeasurementModel
from task6.tracking.types import Detection
from track_manager import TrackManager, TrackManagerConfig

DT = 1.0
RADAR_POS  = np.array([0.0,   0.0])
CAMERA_POS = np.array([-80.0, 120.0])

_det_counter = 0


def load_scenario(json_path: Path):
    with open(json_path) as f:
        return json.load(f)


def make_detections(t, meas_list, mm: CoordFrameMeasurementModel) -> list[Detection]:
    global _det_counter
    dets = []
    for m in meas_list:
        sid = m["sensor_id"]
        if sid not in ("radar", "camera"):
            continue
        if m.get("range_m") is None:
            continue
        z       = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
        _, _, R = mm.predict(np.zeros(4), sid)
        dets.append(Detection(
            detection_id   = f"{sid}_{_det_counter}",
            time_s         = t,
            sensor_id      = sid,
            z              = z,
            R              = R,
            truth_id       = None,
            is_false_alarm = False,
        ))
        _det_counter += 1
    return dets


def det_to_ned(sid: str, z: np.ndarray) -> np.ndarray:
    origin = RADAR_POS if sid == "radar" else CAMERA_POS
    return origin + np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])


def main() -> None:
    global _det_counter
    _det_counter = 0

    data = load_scenario(PROJECT_ROOT / "scenario_real.json")
    t_end = float(data["t_end"])

    # Vessel position from GNSS (target_id = -1)
    vessel_positions = sorted(
        [(float(m["time"]), float(m["north_m"]), float(m["east_m"]))
         for m in data["measurements"]
         if m["sensor_id"] == "gnss" and m["target_id"] == -1],
        key=lambda x: x[0],
    )

    def get_vessel_pos(t: float):
        times = np.array([r[0] for r in vessel_positions])
        idx   = int(np.argmin(np.abs(times - t)))
        return float(vessel_positions[idx][1]), float(vessel_positions[idx][2])

    # AIS reference positions per target (for map overlay only)
    ais_paths: dict[int, list] = defaultdict(list)
    for m in sorted(data["measurements"], key=lambda x: x["time"]):
        if m["sensor_id"] == "ais" and m["target_id"] != -1:
            ais_paths[m["target_id"]].append(
                np.array([float(m["north_m"]), float(m["east_m"])])
            )

    # Radar + camera measurements sorted by time
    meas_sorted = sorted(
        [(float(m["time"]), m) for m in data["measurements"]
         if m["sensor_id"] in ("radar", "camera") and m.get("range_m") is not None],
        key=lambda x: x[0],
    )

    def _win(t_hi: float) -> list:
        return [m for ts, m in meas_sorted if t_hi - DT < ts <= t_hi]

    mm  = CoordFrameMeasurementModel()
    cfg = TrackManagerConfig(M=3, N=15, K_del=10)
    tm  = TrackManager(mm, cfg)

    track_paths : dict[int, list] = defaultdict(list)
    vessel_path : list            = []
    det_ned_r   : list            = []
    det_ned_c   : list            = []
    time_hist, n_confirmed_hist, n_tentative_hist = [], [], []

    print(f"\n{'='*60}")
    print(f"T7 — Real harbour data  ({t_end:.0f}s)")
    print(f"M={cfg.M}  N={cfg.N}  K_del={cfg.K_del}")
    print(f"{'='*60}")

    for t in np.arange(1.0, t_end + DT, DT):
        t = round(float(t), 6)

        pN, pE = get_vessel_pos(t)
        vessel_path.append(np.array([pN, pE]))
        mm.set_vessel_position(pN, pE)

        dets      = make_detections(t, _win(t), mm)
        confirmed = tm.step(t, dets)
        all_mt    = tm.all_tracks()

        for tr in confirmed:
            track_paths[tr.track_id].append(tr.x[:2].copy())

        for d in dets:
            pt = det_to_ned(d.sensor_id, d.z)
            (det_ned_r if d.sensor_id == "radar" else det_ned_c).append(pt)

        time_hist.append(t)
        n_confirmed_hist.append(len(confirmed))
        n_tentative_hist.append(sum(1 for mt in all_mt if mt.status == "tentative"))

        if int(t) % 100 == 0:
            print(f"  t={t:6.0f}s  confirmed={len(confirmed):2d}  tentative={n_tentative_hist[-1]:2d}  dets={len(dets):2d}")

    print(f"\nTotal unique confirmed tracks: {len(track_paths)}")
    print(f"Mean confirmed at any time:   {np.mean(n_confirmed_hist):.2f}")

    # -----------------------------------------------------------------------
    # Figure 1 — Trajectory map
    # -----------------------------------------------------------------------
    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]

    fig_map, ax = plt.subplots(figsize=(12, 10))

    # AIS reference paths
    AIS_COLORS = {1: "gold", 3: "lime", 18: "cyan"}
    for tid, path in ais_paths.items():
        if not path:
            continue
        arr = np.array(path)
        col = AIS_COLORS.get(tid, "white")
        ax.plot(arr[:, 1], arr[:, 0], "--", lw=2, color=col, alpha=0.9,
                label=f"AIS T{tid}")

    # Confirmed tracks
    for i, (tid, path) in enumerate(sorted(track_paths.items())):
        if len(path) < 5:   # skip very short tracks (noise)
            continue
        arr = np.array(path)
        ax.plot(arr[:, 1], arr[:, 0], "-", lw=1.5,
                color=COLORS[i % len(COLORS)], alpha=0.8, label=f"Track {tid}")

    # Detections (subsampled for readability)
    step = 5
    if det_ned_r:
        arr = np.array(det_ned_r)[::step]
        ax.scatter(arr[:, 1], arr[:, 0], s=8, c="tab:cyan",  marker=".", alpha=0.2, label="Radar det")
    if det_ned_c:
        arr = np.array(det_ned_c)[::step]
        ax.scatter(arr[:, 1], arr[:, 0], s=8, c="tab:pink",  marker=".", alpha=0.2, label="Camera det")

    # Vessel path
    if vessel_path:
        vp = np.array(vessel_path)
        ax.plot(vp[:, 1], vp[:, 0], color="gray", lw=1, alpha=0.6, label="Vessel")

    ax.scatter([0],              [0],              c="black", marker="*", s=200, zorder=5, label="Radar")
    ax.scatter([CAMERA_POS[1]], [CAMERA_POS[0]], c="gold",  marker="^", s=160, zorder=5, label="Camera")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("T7 — Real Harbour Data: Confirmed Tracks vs AIS Reference")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig_map.tight_layout()
    map_out = PROJECT_ROOT / "harbour_sim_output" / "real_map.png"
    fig_map.savefig(map_out, dpi=180)

    # -----------------------------------------------------------------------
    # Figure 2 — Track counts over time
    # -----------------------------------------------------------------------
    fig_stats, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_arr = np.array(time_hist)

    axes[0].plot(t_arr, n_confirmed_hist, label="Confirmed + Coasting", color="tab:blue")
    axes[0].plot(t_arr, n_tentative_hist, label="Tentative", color="tab:orange", alpha=0.7)
    axes[0].set_ylabel("Track count")
    axes[0].set_title("Real Data — Track counts over time")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # AIS timestamps as vertical markers
    for tid, col in AIS_COLORS.items():
        ais_times = sorted(
            float(m["time"]) for m in data["measurements"]
            if m["sensor_id"] == "ais" and m["target_id"] == tid
        )
        if ais_times:
            axes[1].scatter(ais_times, [tid] * len(ais_times),
                            s=4, c=col, marker="|", label=f"AIS T{tid}")

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("AIS target ID")
    axes[1].set_title("AIS reception timeline")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig_stats.tight_layout()
    stats_out = PROJECT_ROOT / "harbour_sim_output" / "real_stats.png"
    fig_stats.savefig(stats_out, dpi=180)

    plt.show()
    print(f"\nPlots saved:\n  {map_out}\n  {stats_out}")


if __name__ == "__main__":
    main()

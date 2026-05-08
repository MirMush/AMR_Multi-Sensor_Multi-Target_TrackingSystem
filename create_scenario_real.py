#!/usr/bin/env python3
"""
Convert raw CSV sensor files → corrected scenario_real.json

Dataset: departure of Dana IV, harbour of Copenhagen, 5 March 2026.
NED origin (= radar + camera position): 55.69014690 N / 12.59998830 E

Bearing corrections applied:
  Radar  : bearing_NED_rad = (bearing_radar_deg + RADAR_FRAME_ROT_DEG) * π/180
           (radar frame is rotated 16° from NED)
  Camera : bearing_NED_rad = atan2(X, Z) + CAMERA_FRAME_ROT_RAD
           (camera frame is rotated 28° from NED)

Noise:
  GNSS / AIS : sigma_pos = 6 m  (variance = 36 m²)
  Radar      : covariance read directly from CSV
  Camera     : sigma_x, sigma_z read from CSV

Usage:
  python create_scenario_real.py
  python create_scenario_real.py --radar-rot-deg 16 --cam-rot-deg 28
"""
import argparse
import json
import math
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent / "Experimental data"
OUTPUT   = Path(__file__).parent / "scenario_real_corrected.json"

# Rotation to apply to convert sensor-frame bearing to NED compass bearing.
# Radar: radar reports 0° = its East axis (not North), so there is an inherent
#        −90° convention offset on top of the 16° physical mounting rotation.
#        Net correction: −90° + 16° = −74°
# Camera: 28° physical mounting rotation only.
RADAR_FRAME_ROT_DEG  = -74.0   # = −90° (convention) + 16° (mounting)
CAMERA_FRAME_ROT_DEG =  28.0

# GNSS / AIS position noise
SIGMA_POS_M = 6.0   # metres (1-sigma)

# NED origin = radar + camera mounting position
NED_ORIGIN_LAT =  55.69014690   # °N
NED_ORIGIN_LON =  12.59998830   # °E


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_measurements(radar_rot_deg: float, cam_rot_deg: float) -> tuple[list[dict], list]:
    radar_rot_rad = math.radians(radar_rot_deg)
    cam_rot_rad   = math.radians(cam_rot_deg)
    measurements: list[dict] = []
    vessel_positions: list   = []

    # ── GNSS ────────────────────────────────────────────────────────────────
    for row in read_csv(DATA_DIR / "gnss.csv"):
        t = float(row["time"])
        N = float(row["N"])
        E = float(row["E"])
        vessel_positions.append([t, N, E])
        measurements.append({
            "sensor_id":      "gnss",
            "time":           round(t, 6),
            "is_false_alarm": False,
            "target_id":      -1,
            "range_m":        None,
            "bearing_rad":    None,
            "north_m":        N,
            "east_m":         E,
            "metadata": {
                "heading_deg": float(row["heading"]),
                "sigma_pos_m": SIGMA_POS_M,
            },
        })

    # ── AIS ─────────────────────────────────────────────────────────────────
    for row in read_csv(DATA_DIR / "ais.csv"):
        measurements.append({
            "sensor_id":      "ais",
            "time":           round(float(row["time"]), 6),
            "is_false_alarm": False,
            "target_id":      int(float(row["ais_id"])),
            "range_m":        None,
            "bearing_rad":    None,
            "north_m":        float(row["N"]),
            "east_m":         float(row["E"]),
            "metadata": {
                "mmsi":        int(float(row["mmsi"])),
                "sigma_pos_m": SIGMA_POS_M,
            },
        })

    # ── Radar ────────────────────────────────────────────────────────────────
    # bearing in CSV is in the radar frame; add radar_rot to convert to NED.
    # Covariance matrix: [[cov_range, cov_range_bearing],
    #                     [cov_range_bearing, cov_bearing]]
    for row in read_csv(DATA_DIR / "mm_wave_radar.csv"):
        bearing_rad = math.radians(float(row["bearing"])) + radar_rot_rad
        measurements.append({
            "sensor_id":      "radar",
            "time":           round(float(row["time"]), 6),
            "is_false_alarm": False,
            "target_id":      -1,
            "range_m":        round(float(row["range"]), 8),
            "bearing_rad":    round(bearing_rad, 7),
            "north_m":        None,
            "east_m":         None,
            "metadata": {
                "cluster_id":        int(float(row["cluster_id"])),
                "cov_range":         float(row["cov_range"]),
                "cov_bearing":       float(row["cov_bearing"]),
                "cov_range_bearing": float(row["cov_range_bearing"]),
            },
        })

    # ── Camera ───────────────────────────────────────────────────────────────
    # X, Z are in the camera frame. range = hypot(X, Z).
    # bearing_camera = atan2(X, Z); add cam_rot to convert to NED.
    for row in read_csv(DATA_DIR / "camera.csv"):
        X = float(row["X"])
        Z = float(row["Z"])
        range_m     = math.hypot(X, Z)
        bearing_rad = math.atan2(X, Z) + cam_rot_rad
        measurements.append({
            "sensor_id":      "camera",
            "time":           round(float(row["time"]), 6),
            "is_false_alarm": False,
            "target_id":      -1,
            "range_m":        round(range_m, 7),
            "bearing_rad":    round(bearing_rad, 7),
            "north_m":        None,
            "east_m":         None,
            "metadata": {
                "track_id":  int(float(row["ID"])),
                "X_m":       X,
                "Z_m":       Z,
                "sigma_x_m": float(row["sigma_x"]),
                "sigma_z_m": float(row["sigma_z"]),
            },
        })

    measurements.sort(key=lambda m: m["time"])
    vessel_positions.sort(key=lambda v: v[0])
    return measurements, vessel_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-rot-deg", type=float, default=RADAR_FRAME_ROT_DEG,
                        help=f"Net bearing correction for radar (degrees). Default: {RADAR_FRAME_ROT_DEG} (−90° convention + 16° mounting)")
    parser.add_argument("--cam-rot-deg", type=float, default=CAMERA_FRAME_ROT_DEG,
                        help=f"Camera frame rotation relative to NED (degrees). Default: {CAMERA_FRAME_ROT_DEG}")
    args = parser.parse_args()

    measurements, vessel_positions = build_measurements(args.radar_rot_deg, args.cam_rot_deg)

    t_end = max(m["time"] for m in measurements)

    doc = {
        "scenario_name": "Real World — Dana IV departure, Copenhagen harbour, 2026-03-05",
        "ned_origin": {
            "lat_deg": NED_ORIGIN_LAT,
            "lon_deg": NED_ORIGIN_LON,
            "note":    "55.69014690 N / 12.59998830 E — radar and camera mounting position",
        },
        "t_end":   round(t_end, 3),
        "dt_true": 1.0,
        "sensor_configs": {
            "gnss": {
                "note":       "gnss.csv — NED position + heading of own vessel (Dana IV)",
                "sigma_pos_m": SIGMA_POS_M,
                "origin_ned":  [0.0, 0.0],
            },
            "ais": {
                "note":       "ais.csv — NED positions of AIS-broadcasting vessels",
                "sigma_pos_m": SIGMA_POS_M,
            },
            "radar": {
                "note":             "mm_wave_radar.csv — polar detections in radar frame",
                "frame_rot_deg":    args.radar_rot_deg,
                "bearing_ned_note": "bearing_NED = bearing_radar + frame_rot_deg",
                "origin_ned":       [0.0, 0.0],
            },
            "camera": {
                "note":             "camera.csv — X,Z detections in camera frame",
                "frame_rot_deg":    args.cam_rot_deg,
                "bearing_ned_note": "bearing_NED = atan2(X, Z) + frame_rot_deg",
                "origin_ned":       [-80.0, 120.0],
            },
        },
        "ground_truth":    {},
        "vessel_positions": vessel_positions,
        "measurements":    measurements,
    }

    with open(OUTPUT, "w") as f:
        json.dump(doc, f, indent=2)

    counts = {}
    for m in measurements:
        counts[m["sensor_id"]] = counts.get(m["sensor_id"], 0) + 1
    print(f"Written : {OUTPUT}")
    print(f"  t_end            = {t_end:.1f} s")
    print(f"  measurements     : {counts}")
    print(f"  vessel_positions : {len(vessel_positions)}")
    print(f"  radar rot        : {args.radar_rot_deg}°")
    print(f"  camera rot       : {args.cam_rot_deg}°")
    print(f"  GNSS/AIS sigma   : {SIGMA_POS_M} m")


if __name__ == "__main__":
    main()

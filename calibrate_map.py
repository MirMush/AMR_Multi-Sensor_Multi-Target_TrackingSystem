#!/usr/bin/env python3
"""
Calibrate the harbour map overlay by plotting raw measurements on top.
Adjust EXTENT until measurements align with the map geography.

EXTENT = [East_min, East_max, North_min, North_max]  (metres, NED)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MAP_PATH = Path("Experimental data/new_map.png")
JSON_PATH = Path("scenario_real_corrected.json")

# ── Adjust these until the overlay looks right ──────────────────────────────
EXTENT = [-520, 3000, -1000, 3000]   # [E_min, E_max, N_min, N_max]
# ────────────────────────────────────────────────────────────────────────────

RADAR_POS  = np.array([0.0,    0.0])
CAMERA_POS = np.array([-80.0, 120.0])

with open(JSON_PATH) as f:
    data = json.load(f)

# GNSS vessel positions
gnss = np.array([
    [float(m["north_m"]), float(m["east_m"])]
    for m in data["measurements"]
    if m["sensor_id"] == "gnss" and m.get("north_m") is not None
    and m.get("target_id", -1) == -1
])

# Radar detections → NED
radar_ned = np.array([
    [m["range_m"] * np.cos(m["bearing_rad"]),
     m["range_m"] * np.sin(m["bearing_rad"])]
    for m in data["measurements"]
    if m["sensor_id"] == "radar" and m.get("range_m") is not None
])  # [N, E]

# Camera detections → NED
camera_ned = np.array([
    [CAMERA_POS[0] + m["range_m"] * np.cos(m["bearing_rad"]),
     CAMERA_POS[1] + m["range_m"] * np.sin(m["bearing_rad"])]
    for m in data["measurements"]
    if m["sensor_id"] == "camera" and m.get("range_m") is not None
])  # [N, E]

# AIS positions
ais_ned = {
    tid: np.array([[float(m["north_m"]), float(m["east_m"])]
                   for m in data["measurements"]
                   if m["sensor_id"] == "ais" and m.get("north_m") is not None
                   and m["target_id"] == tid])
    for tid in (1, 3, 18)
}

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

img = plt.imread(str(MAP_PATH))
ax.imshow(img, extent=EXTENT, aspect="auto", zorder=0)

# GNSS vessel
ax.plot(gnss[:, 1], gnss[:, 0], color="magenta", lw=1.2, alpha=0.7,
        zorder=2, label="GNSS vessel")

# Radar
if len(radar_ned):
    ax.scatter(radar_ned[:, 1], radar_ned[:, 0], s=4, c="#FF1493",
               alpha=0.3, zorder=3, label="Radar det")

# Camera
if len(camera_ned):
    ax.scatter(camera_ned[:, 1], camera_ned[:, 0], s=4, c="orange",
               alpha=0.4, zorder=3, label="Camera det")

# AIS
AIS_COLORS = {1: "gold", 3: "lime", 18: "deepskyblue"}
for tid, pts in ais_ned.items():
    if len(pts):
        ax.scatter(pts[:, 1], pts[:, 0], s=30, c=AIS_COLORS[tid],
                   marker="D", zorder=4, label=f"AIS T{tid}")

# Sensor origins
ax.scatter([RADAR_POS[1]],  [RADAR_POS[0]],  c="black", marker="*",
           s=300, zorder=5, label="Radar origin")
ax.scatter([CAMERA_POS[1]], [CAMERA_POS[0]], c="orange", marker="^",
           s=200, zorder=5, label="Camera origin")

ax.set_xlim(EXTENT[0], EXTENT[1])
ax.set_ylim(EXTENT[2], EXTENT[3])
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_title(f"Map calibration — EXTENT={EXTENT}")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3, zorder=1)

out = Path("harbour_sim_output/calibration.png")
fig.savefig(out, dpi=180)
plt.show()
print(f"Saved: {out}")
print(f"Current EXTENT: {EXTENT}")

#!/usr/bin/env python3
"""
T3 — Single-target EKF tracker, Scenario A (radar only).

Reads scenario_A.json, runs the EKF, and reports:
  - RMSE (steady-state, after first 5 updates)
  - NIS consistency (fraction within 95% chi2(2) bounds)
  - Trajectory plot
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from collections import defaultdict

from coord_frame_manager import CoordFrameManager
from EKF import EKF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSON_PATH = "harbour_sim_output/scenario_A.json"
SENSOR_ID = "radar"
GAMMA     = chi2.ppf(0.99, df=2)   # gating threshold  (~9.21)
NIS_LOW   = chi2.ppf(0.025, df=2)  # lower NIS bound   (~0.05)
NIS_HIGH  = chi2.ppf(0.975, df=2)  # upper NIS bound   (~7.38)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    data = json.load(f)

t_end            = float(data["t_end"])
vessel_positions = data["vessel_positions"]
gt_states        = data["ground_truth"]["0"]   # single target, ID 0

# Group radar measurements by timestamp: { t -> [m1, m2, ...] }
radar_by_time = defaultdict(list)
for m in data["measurements"]:
    if m["sensor_id"] == SENSOR_ID:
        radar_by_time[round(m["time"], 6)].append(m)

# Ground truth lookup
gt_times = np.array([row[0] for row in gt_states])
gt_array = np.array([row[1:] for row in gt_states])

def get_gt(t):
    idx = np.argmin(np.abs(gt_times - t))
    return gt_array[idx]

# Vessel position lookup
vp_times_arr = np.array([row[0] for row in vessel_positions])

def get_vessel_pos(t):
    idx = np.argmin(np.abs(vp_times_arr - t))
    return vessel_positions[idx][1], vessel_positions[idx][2]

# ---------------------------------------------------------------------------
# Initialize tracker
# ---------------------------------------------------------------------------
cfm = CoordFrameManager()
ekf = EKF(cfm=cfm, sigma_a=0.05)

initialized = False

# Storage for results
times_log  = []
est_pos    = []
gt_pos     = []
nis_values = []
nis_times  = []

# ---------------------------------------------------------------------------
# Main loop — 1 Hz ticks (fastest sensor rate)
# ---------------------------------------------------------------------------
dt = 1.0

for t in np.arange(1.0, t_end + dt, dt):
    t = round(t, 6)

    # Always update vessel position so cfm has the latest GNSS fix
    pN_v, pE_v = get_vessel_pos(t)
    cfm.update_vessel_pos(pN_v, pE_v)

    # All radar measurements that arrived at this tick
    measurements_at_t = radar_by_time.get(t, [])

    # --- Wait for first radar hit to initialize EKF ---
    if not initialized:
        for m in measurements_at_t:
            z = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
            ekf.initialize_from_measurement(z, SENSOR_ID)
            initialized = True
            break
        continue

    # --- Predict to current tick ---
    ekf.predict(dt)

    # --- Find best measurement via gating (lowest Mahalanobis distance) ---
    H     = cfm.H(ekf.x, SENSOR_ID)
    R     = cfm.R(SENSOR_ID)
    S     = H @ ekf.P @ H.T + R
    S_inv = np.linalg.inv(S)

    best_z  = None
    best_d2 = np.inf

    for m in measurements_at_t:
        z    = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
        y    = z - cfm.h(ekf.x, SENSOR_ID)
        y[1] = float((y[1] + np.pi) % (2 * np.pi) - np.pi)  # wrap bearing
        d2   = float(y @ S_inv @ y)

        if d2 < GAMMA and d2 < best_d2:
            best_d2 = d2
            best_z  = z

    # --- Update with best measurement (if any passed the gate) ---
    if best_z is not None:
        nis = ekf.update(best_z, SENSOR_ID)
        nis_values.append(nis)
        nis_times.append(t)

    # --- Log state at every tick ---
    times_log.append(t)
    est_pos.append(ekf.x[:2].copy())
    gt_pos.append(get_gt(t)[:2])

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
est_pos    = np.array(est_pos)
gt_pos     = np.array(gt_pos)
nis_values = np.array(nis_values)
nis_times  = np.array(nis_times)
times_log  = np.array(times_log)

errors  = np.linalg.norm(est_pos - gt_pos, axis=1)
rmse_ss = np.sqrt(np.mean(errors[5:]**2))
within  = np.mean((nis_values >= NIS_LOW) & (nis_values <= NIS_HIGH))

print("=" * 50)
print("Scenario A — radar-only single-target EKF")
print("=" * 50)
print(f"Updates accepted        : {len(nis_values)}")
print(f"Steady-state RMSE       : {rmse_ss:.2f} m  (target < 12 m)")
print(f"NIS consistency         : {within*100:.1f}%  (target > 90%)")
print(f"NIS bounds              : [{NIS_LOW:.2f}, {NIS_HIGH:.2f}]")
print(f"Gating threshold gamma  : {GAMMA:.2f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Scenario A — Single-target radar-only EKF")

# Trajectory
ax = axes[0]
ax.plot(gt_pos[:, 1],  gt_pos[:, 0],  "k--", label="Ground truth")
ax.plot(est_pos[:, 1], est_pos[:, 0], "b-",  label="EKF estimate")
ax.scatter([0], [0], c="r", marker="^", zorder=5, label="Radar")
ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.set_title("Trajectory (NED)")
ax.legend()
ax.grid(True)
ax.set_aspect("equal")

# Position error
ax = axes[1]
ax.plot(times_log, errors, "b-")
ax.axhline(12, color="r", linestyle="--", label="Target RMSE (12 m)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position error (m)")
ax.set_title("Position Error over Time")
ax.legend()
ax.grid(True)

# NIS
ax = axes[2]
ax.plot(nis_times, nis_values, "b.", label="NIS")
ax.axhline(NIS_HIGH, color="r", linestyle="--", label=f"Upper 95% ({NIS_HIGH:.2f})")
ax.axhline(NIS_LOW,  color="g", linestyle="--", label=f"Lower 95% ({NIS_LOW:.2f})")
ax.set_xlabel("Time (s)")
ax.set_ylabel("NIS")
ax.set_title(f"NIS consistency ({within*100:.1f}% within bounds)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("harbour_sim_output/scenario_A_results.png", dpi=150)
plt.show()
print("Plot saved to harbour_sim_output/scenario_A_results.png")

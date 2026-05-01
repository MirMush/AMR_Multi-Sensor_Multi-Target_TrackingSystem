"""
Scenario B multi-sensor single-target EKF fusion.

Implements and compares:
1) Sequential update: radar -> camera (same predicted prior each scan)
2) Centralised update: stacked [radar; camera] joint measurement update
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, chi2

from coord_frame_manager import CoordFrameManager
from EKF import EKF

JSON_PATH = "harbour_sim_output/scenario_B.json"
DT = 1.0
GATE_PROB = 0.99
SIGMA_A = 0.03
INIT_POLICY = "radar_only"  # options: "first_valid", "radar_only"


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def detect_sensor_ids(data):
    ids = {m["sensor_id"] for m in data["measurements"]}
    radar_id = "radar" if "radar" in ids else None
    camera_id = "camera" if "camera" in ids else None
    if radar_id is None or camera_id is None:
        raise ValueError(f"Could not detect both radar and camera in measurements. Found: {sorted(ids)}")
    return radar_id, camera_id


def group_measurements_by_sensor_and_time(measurements, sensor_ids):
    grouped = {sid: defaultdict(list) for sid in sensor_ids}
    for m in measurements:
        sid = m["sensor_id"]
        if sid in grouped and m.get("range_m") is not None and m.get("bearing_rad") is not None:
            grouped[sid][round(float(m["time"]), 6)].append(m)
    return grouped


def select_best_gated_measurement(ekf, cfm, sensor_id, measurements_at_t, gate_prob=0.99):
    """Reusable gate + best-candidate selection by minimum Mahalanobis distance."""
    dof = 2
    gamma = chi2.ppf(gate_prob, df=dof)

    H = cfm.H(ekf.x, sensor_id)
    R = cfm.R(sensor_id)
    S = H @ ekf.P @ H.T + R
    S_inv = np.linalg.inv(S)
    z_pred = cfm.h(ekf.x, sensor_id)

    best_z = None
    best_d2 = np.inf

    for m in measurements_at_t:
        z = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
        if not cfm.measurement_in_fov_and_range(sensor_id, z):
            continue

        y = z - z_pred
        y[1] = wrap_angle(y[1])
        d2 = float(y @ S_inv @ y)
        if d2 < gamma and d2 < best_d2:
            best_d2 = d2
            best_z = z

    return best_z, best_d2, dof


def ekf_update_stacked(ekf, cfm, sensors, z_list):
    """Single centralised EKF update with stacked measurement model."""
    h_blocks = []
    H_blocks = []
    R_blocks = []

    for sid, z in zip(sensors, z_list):
        h_blocks.append(cfm.h(ekf.x, sid))
        H_blocks.append(cfm.H(ekf.x, sid))
        R_blocks.append(cfm.R(sid))

    z_stack = np.concatenate(z_list)
    h_stack = np.concatenate(h_blocks)
    H_stack = np.vstack(H_blocks)

    rdim = z_stack.size
    R_stack = np.zeros((rdim, rdim), dtype=float)
    offset = 0
    for R in R_blocks:
        n = R.shape[0]
        R_stack[offset : offset + n, offset : offset + n] = R
        offset += n

    y = z_stack - h_stack
    for k in range(1, y.size, 2):
        y[k] = wrap_angle(y[k])

    S = H_stack @ ekf.P @ H_stack.T + R_stack
    K = ekf.P @ H_stack.T @ np.linalg.inv(S)

    ekf.x = ekf.x + K @ y
    I = np.eye(ekf.P.shape[0])
    ekf.P = (I - K @ H_stack) @ ekf.P @ (I - K @ H_stack).T + K @ R_stack @ K.T
    ekf.P = 0.5 * (ekf.P + ekf.P.T)

    nis = float(y.T @ np.linalg.inv(S) @ y)
    dof = int(y.size)
    return nis, dof


def nis_consistency(nis_values, dof_values, alpha=0.95):
    if len(nis_values) == 0:
        return np.nan
    lo = np.array([chi2.ppf((1 - alpha) / 2, df=df) for df in dof_values])
    hi = np.array([chi2.ppf(1 - (1 - alpha) / 2, df=df) for df in dof_values])
    nis = np.array(nis_values)
    return float(np.mean((nis >= lo) & (nis <= hi)))


def compute_rmse(est_pos, gt_pos, skip=5):
    e = np.linalg.norm(est_pos - gt_pos, axis=1)
    return float(np.sqrt(np.mean(e[skip:] ** 2))), e


def clopper_pearson_interval(successes, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lo), float(hi)


def plot_single_architecture(
    out_path,
    title,
    gt_pos,
    est_pos,
    times,
    errors,
    nis_values,
    dof_values,
    radar_pos,
    camera_pos,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title)

    ax = axes[0]
    ax.plot(gt_pos[:, 1], gt_pos[:, 0], "k--", label="Ground truth")
    ax.plot(est_pos[:, 1], est_pos[:, 0], "b-", label="Estimate")
    ax.scatter([radar_pos[1]], [radar_pos[0]], c="k", marker="^", label="Radar")
    ax.scatter([camera_pos[1]], [camera_pos[0]], c="g", marker="s", label="Camera")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Trajectory (NED)")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(times, errors, "b-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Position Error")
    ax.grid(True)

    ax = axes[2]
    ax.plot(nis_values, "b.", alpha=0.8, label="NIS")
    if len(nis_values) > 0:
        lo = np.array([chi2.ppf(0.025, df=df) for df in dof_values])
        hi = np.array([chi2.ppf(0.975, df=df) for df in dof_values])
        ax.plot(lo, "g--", linewidth=1.2, label="95% lower")
        ax.plot(hi, "r--", linewidth=1.2, label="95% upper")
    ax.set_xlabel("Update index")
    ax.set_ylabel("NIS")
    ax.set_title("NIS with 95% bounds")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    t_end = float(data["t_end"])
    gt_states = data["ground_truth"]["0"]
    vessel_positions = data["vessel_positions"]
    sensor_configs = data.get("sensor_configs", {})

    radar_id, camera_id = detect_sensor_ids(data)
    grouped = group_measurements_by_sensor_and_time(data["measurements"], [radar_id, camera_id])

    gt_times = np.array([row[0] for row in gt_states], dtype=float)
    gt_arr = np.array([row[1:] for row in gt_states], dtype=float)

    vp_times = np.array([row[0] for row in vessel_positions], dtype=float)

    def get_gt(t):
        return gt_arr[np.argmin(np.abs(gt_times - t))]

    def get_vessel(t):
        i = np.argmin(np.abs(vp_times - t))
        return float(vessel_positions[i][1]), float(vessel_positions[i][2])

    cfm_seq = CoordFrameManager(sensor_configs)
    cfm_cen = CoordFrameManager(sensor_configs)
    ekf_seq = EKF(cfm=cfm_seq, sigma_a=SIGMA_A)
    ekf_cen = EKF(cfm=cfm_cen, sigma_a=SIGMA_A)

    # identical initialization from earliest valid radar/camera hit
    initialized = False
    init_sensor = None
    init_time = None

    times = []
    gt_pos = []
    seq_pos = []
    cen_pos = []

    seq_nis = []
    seq_dof = []
    cen_nis = []
    cen_dof = []

    seq_accept_both = 0
    seq_accept_radar_only = 0
    seq_accept_camera_only = 0

    cen_accept_both = 0
    cen_accept_radar_only = 0
    cen_accept_camera_only = 0

    overlap_times_total = 0
    overlap_times_with_raw = []

    for t in np.arange(1.0, t_end + DT, DT):
        t = round(float(t), 6)

        vN, vE = get_vessel(t)
        cfm_seq.update_vessel_pos(vN, vE)
        cfm_cen.update_vessel_pos(vN, vE)

        radar_meas = grouped[radar_id].get(t, [])
        camera_meas = grouped[camera_id].get(t, [])
        if radar_meas and camera_meas:
            overlap_times_total += 1
            overlap_times_with_raw.append(t)

        if not initialized:
            init_z = None
            init_sid = None
            if INIT_POLICY == "radar_only":
                search_order = ((radar_id, radar_meas),)
            elif INIT_POLICY == "first_valid":
                search_order = ((radar_id, radar_meas), (camera_id, camera_meas))
            else:
                raise ValueError(f"Unknown INIT_POLICY '{INIT_POLICY}'")

            for sid, mset in search_order:
                for m in mset:
                    z_try = np.array([m["range_m"], m["bearing_rad"]], dtype=float)
                    if cfm_seq.measurement_in_fov_and_range(sid, z_try):
                        init_z = z_try
                        init_sid = sid
                        break
                if init_z is not None:
                    break

            if init_z is not None:
                ekf_seq.initialize_from_measurement(init_z, init_sid)
                ekf_cen.initialize_from_measurement(init_z, init_sid)
                initialized = True
                init_sensor = init_sid
                init_time = t
            continue

        ekf_seq.predict(DT)
        ekf_cen.predict(DT)

        # sequential radar -> camera
        z_r_seq, _, dof_r = select_best_gated_measurement(ekf_seq, cfm_seq, radar_id, radar_meas, GATE_PROB)
        seq_used_r = z_r_seq is not None
        if z_r_seq is not None:
            seq_nis.append(float(ekf_seq.update(z_r_seq, radar_id)))
            seq_dof.append(dof_r)

        z_c_seq, _, dof_c = select_best_gated_measurement(ekf_seq, cfm_seq, camera_id, camera_meas, GATE_PROB)
        seq_used_c = z_c_seq is not None
        if z_c_seq is not None:
            seq_nis.append(float(ekf_seq.update(z_c_seq, camera_id)))
            seq_dof.append(dof_c)

        if seq_used_r and seq_used_c:
            seq_accept_both += 1
        elif seq_used_r:
            seq_accept_radar_only += 1
        elif seq_used_c:
            seq_accept_camera_only += 1

        # centralised stacked update: pick individually gated measurements from same prior
        z_r_cen, _, _ = select_best_gated_measurement(ekf_cen, cfm_cen, radar_id, radar_meas, GATE_PROB)
        z_c_cen, _, _ = select_best_gated_measurement(ekf_cen, cfm_cen, camera_id, camera_meas, GATE_PROB)
        cen_used_r = z_r_cen is not None
        cen_used_c = z_c_cen is not None

        z_list = []
        sid_list = []
        if z_r_cen is not None:
            sid_list.append(radar_id)
            z_list.append(z_r_cen)
        if z_c_cen is not None:
            sid_list.append(camera_id)
            z_list.append(z_c_cen)

        if len(z_list) == 1:
            cen_nis.append(float(ekf_cen.update(z_list[0], sid_list[0])))
            cen_dof.append(2)
        elif len(z_list) == 2:
            nis, dof = ekf_update_stacked(ekf_cen, cfm_cen, sid_list, z_list)
            cen_nis.append(nis)
            cen_dof.append(dof)

        if cen_used_r and cen_used_c:
            cen_accept_both += 1
        elif cen_used_r:
            cen_accept_radar_only += 1
        elif cen_used_c:
            cen_accept_camera_only += 1

        times.append(t)
        gt_pos.append(get_gt(t)[:2])
        seq_pos.append(ekf_seq.x[:2].copy())
        cen_pos.append(ekf_cen.x[:2].copy())

    times = np.array(times)
    gt_pos = np.array(gt_pos)
    seq_pos = np.array(seq_pos)
    cen_pos = np.array(cen_pos)

    seq_rmse, seq_err = compute_rmse(seq_pos, gt_pos)
    cen_rmse, cen_err = compute_rmse(cen_pos, gt_pos)

    traj_delta = np.linalg.norm(seq_pos - cen_pos, axis=1)
    max_traj_delta = float(np.max(traj_delta)) if len(traj_delta) else 0.0
    mean_traj_delta = float(np.mean(traj_delta)) if len(traj_delta) else 0.0

    seq_cons = nis_consistency(seq_nis, seq_dof, alpha=0.95)
    cen_cons = nis_consistency(cen_nis, cen_dof, alpha=0.95)

    seq_nis = np.array(seq_nis, dtype=float)
    cen_nis = np.array(cen_nis, dtype=float)
    seq_dof = np.array(seq_dof, dtype=int)
    cen_dof = np.array(cen_dof, dtype=int)

    seq_lo = np.array([chi2.ppf(0.025, df=df) for df in seq_dof]) if len(seq_dof) else np.array([])
    seq_hi = np.array([chi2.ppf(0.975, df=df) for df in seq_dof]) if len(seq_dof) else np.array([])
    cen_lo = np.array([chi2.ppf(0.025, df=df) for df in cen_dof]) if len(cen_dof) else np.array([])
    cen_hi = np.array([chi2.ppf(0.975, df=df) for df in cen_dof]) if len(cen_dof) else np.array([])

    print("=" * 72)
    print("Scenario B — Radar + Camera Fusion")
    print("=" * 72)
    print(f"Initialization policy                  : {INIT_POLICY}")
    if init_sensor is not None:
        print(f"Initialized at time/sensor             : t={init_time:.1f}s, {init_sensor}")
    print(f"Detected sensors                       : radar='{radar_id}', camera='{camera_id}'")
    print(f"Sequential updates accepted (NIS logs) : {len(seq_nis)}")
    print(f"Centralised updates accepted (NIS logs): {len(cen_nis)}")
    print(f"Raw radar/camera overlap timestamps    : {overlap_times_total}")
    print(f"Sequential accepted both sensors/tick  : {seq_accept_both}")
    print(f"Centralised accepted both sensors/tick : {cen_accept_both}")
    print(f"Sequential radar-only ticks            : {seq_accept_radar_only}")
    print(f"Sequential camera-only ticks           : {seq_accept_camera_only}")
    print(f"Centralised radar-only ticks           : {cen_accept_radar_only}")
    print(f"Centralised camera-only ticks          : {cen_accept_camera_only}")
    print(f"Sequential RMSE (position)             : {seq_rmse:.3f} m")
    print(f"Centralised RMSE (position)            : {cen_rmse:.3f} m")
    print(f"Mean |seq-cen| position delta          : {mean_traj_delta:.6f} m")
    print(f"Max  |seq-cen| position delta          : {max_traj_delta:.6f} m")
    print(f"Sequential NIS consistency (95%)       : {100*seq_cons:.2f}%")
    print(f"Centralised NIS consistency (95%)      : {100*cen_cons:.2f}%")
    if len(seq_nis):
        print(f"Sequential NIS min/max                 : {seq_nis.min():.4f} / {seq_nis.max():.4f}")
        print(f"Sequential lower/upper min             : {seq_lo.min():.4f} / {seq_hi.min():.4f}")
        print(f"Sequential lower/upper max             : {seq_lo.max():.4f} / {seq_hi.max():.4f}")
    if len(cen_nis):
        print(f"Centralised NIS min/max                : {cen_nis.min():.4f} / {cen_nis.max():.4f}")
        print(f"Centralised lower/upper min            : {cen_lo.min():.4f} / {cen_hi.min():.4f}")
        print(f"Centralised lower/upper max            : {cen_lo.max():.4f} / {cen_hi.max():.4f}")
    seq_success = int(np.sum((seq_nis >= seq_lo) & (seq_nis <= seq_hi))) if len(seq_nis) else 0
    cen_success = int(np.sum((cen_nis >= cen_lo) & (cen_nis <= cen_hi))) if len(cen_nis) else 0
    seq_cp = clopper_pearson_interval(seq_success, len(seq_nis), alpha=0.05)
    cen_cp = clopper_pearson_interval(cen_success, len(cen_nis), alpha=0.05)
    if len(seq_nis):
        print(
            "Sequential consistency CI (binomial 95%) : "
            f"[{100*seq_cp[0]:.2f}%, {100*seq_cp[1]:.2f}%], n={len(seq_nis)}"
        )
        print(f"P(all in-bounds | true p=0.95, n={len(seq_nis)}) : {0.95 ** len(seq_nis):.4f}")
    if len(cen_nis):
        print(
            "Centralised consistency CI (binomial 95%): "
            f"[{100*cen_cp[0]:.2f}%, {100*cen_cp[1]:.2f}%], n={len(cen_nis)}"
        )
        print(f"P(all in-bounds | true p=0.95, n={len(cen_nis)}) : {0.95 ** len(cen_nis):.4f}")

    eps = 1e-6
    if abs(seq_rmse - cen_rmse) <= eps:
        better_rmse = "tie"
    else:
        better_rmse = "sequential" if seq_rmse < cen_rmse else "centralised"

    if abs(seq_cons - cen_cons) <= eps:
        better_cons = "tie"
    else:
        better_cons = "sequential" if seq_cons > cen_cons else "centralised"
    print(f"Lower RMSE architecture                : {better_rmse}")
    print(f"Better NIS consistency architecture    : {better_cons}")
    print("=" * 72)

    out_seq = "harbour_sim_output/scenario_B_sequential_results.png"
    out_cen = "harbour_sim_output/scenario_B_centralised_results.png"

    plot_single_architecture(
        out_seq,
        "Scenario B: Sequential EKF Fusion",
        gt_pos,
        seq_pos,
        times,
        seq_err,
        seq_nis,
        seq_dof,
        cfm_seq.radar_pos,
        cfm_seq.camera_pos,
    )
    plot_single_architecture(
        out_cen,
        "Scenario B: Centralised EKF Fusion",
        gt_pos,
        cen_pos,
        times,
        cen_err,
        cen_nis,
        cen_dof,
        cfm_seq.radar_pos,
        cfm_seq.camera_pos,
    )
    print(f"Plot saved to {out_seq}")
    print(f"Plot saved to {out_cen}")


if __name__ == "__main__":
    main()

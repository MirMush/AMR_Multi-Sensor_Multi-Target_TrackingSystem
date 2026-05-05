# Task 6 Pipeline Setup (Clear Version)

This setup is a **Task 6 sandbox only**:

1. Gating with Mahalanobis distance
2. Global data association (Hungarian / GNN)
3. Sensor availability handling
4. Unmatched track/detection outputs

It uses fake data so you can debug T6 logic before full integration of T3-T5 and T7.

## 1) Required files only

Keep these for the Task 6 pipeline:

- `coord_frame_manager.py`
- `task6/tracking/types.py`
- `task6/tracking/utils.py`
- `task6/tracking/measurement_models.py`
- `task6/tracking/baseline_ekf.py`
- `task6/tracking/gating.py`
- `task6/tracking/association.py`
- `task6/tracking/fusion_cycle.py`
- `task6/tracking/fake_data.py`
- `task6/scripts/run_task6_sandbox.py`
- `task6/scripts/visualize_task6_sandbox.py`
- `task6/TASK6_SETUP.md`

Generated files like `__pycache__/` and `outputs/` are not required and can be deleted safely.

## 2) One-cycle step-by-step (what actually happens)

Each scan (time `t`) goes through this flow in `task6/tracking/fusion_cycle.py`:

1. Input:
   - current tracks (`x`, `P`)
   - detections from radar/camera/AIS
   - sensor availability flags
2. Predict:
   - each track is predicted to time `t` using `ekf_hooks.predict()`
3. Availability filter:
   - detections from unavailable sensors are skipped
4. Gating:
   - for every `(track_i, detection_j)` compute `d^2`
   - keep only pairs with `d^2 <= gamma`
5. Association:
   - build one global cost matrix from gated pairs
   - run Hungarian algorithm
6. Update and miss handling:
   - matched tracks: `ekf_hooks.update()`
   - unmatched tracks: `missed_count += 1`
   - unmatched detections are returned (for future track initiation in T7)

## 3) What is Mahalanobis distance

For a track and one detection:

- Innovation: `y = z - h(x^-)`
- Innovation covariance: `S = H P^- H^T + R`
- Distance: `d^2 = y^T S^-1 y`

Meaning:

1. It measures how "surprising" a detection is for the predicted track.
2. It scales by uncertainty:
   - large `S` (high uncertainty) makes the same error less severe
   - small `S` (high confidence) makes the same error more severe
3. It is better than Euclidean distance because it uses covariance and sensor noise.

Gate rule:

- `d^2 <= gamma`
- `gamma` comes from chi-square distribution with probability `P_G` (default `0.99`)

So the gate is an uncertainty-aware ellipse, not a fixed-radius circle.

## 4) Run it

From repo root:

```bash
python task6/scripts/run_task6_sandbox.py
```

With visual output:

```bash
python task6/scripts/visualize_task6_sandbox.py
```

Outputs:

- `outputs/task6_viz/task6_map.png`
- `outputs/task6_viz/task6_stats.png`

## 5) How to read the plots

`task6_map.png`:

1. Dashed lines: ground-truth target paths
2. Solid lines: estimated tracks
3. Dots/crosses: detections (real, false alarms, skipped)
4. Star/triangle: radar and camera fixed positions

`task6_stats.png`:

1. Top: total detections vs available vs skipped
2. Middle: gated pairs, matches, unmatched tracks/detections
3. Bottom: camera/AIS availability (on/off)

## 6) How to plug real tasks later

To integrate T3-T5 and T7:

1. Replace `CVEKFHooks` with your real EKF implementation.
2. Keep `run_fusion_cycle()` as the T6 orchestration function.
3. Feed real detections into `Detection` objects.
4. Wrap T7 lifecycle around unmatched outputs.

Stable interfaces you should keep:

- `measurement_model.predict(x, sensor_id) -> (z_hat, H, R)`
- `ekf_hooks.predict(track, time_s)`
- `ekf_hooks.update(track, detection)`

If these stay stable, T6 logic remains reusable.

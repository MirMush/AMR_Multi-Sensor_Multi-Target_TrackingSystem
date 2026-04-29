#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking import (  # noqa: E402
    CVEKFHooks,
    CoordFrameMeasurementModel,
    FakeScenarioConfig,
    generate_task6_fake_scans,
    initialize_tracks_from_truth,
    run_fusion_cycle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 6 sandbox with synthetic multi-sensor data.")
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
        "--verbose-every",
        type=int,
        default=10,
        help="Print one-line cycle status every N scans.",
    )
    parser.add_argument(
        "--no-false-alarms",
        action="store_true",
        help="Disable false alarms in fake data generation.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

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

    total_gated = 0
    total_matches = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0
    total_skipped = 0
    truth_consistent_matches = 0
    truth_conflicting_matches = 0

    print("Task 6 sandbox started.")
    print(
        f"Scans={len(scans)}, init_tracks={len(tracks)}, "
        f"gate_probability={args.gate_probability:.3f}, false_alarms={not args.no_false_alarms}"
    )

    for idx, scan in enumerate(scans):
        measurement_model.set_vessel_position(scan.vessel_pos_ned[0], scan.vessel_pos_ned[1])
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

        total_gated += len(cycle_result.gated_candidates)
        total_matches += len(cycle_result.association.matches)
        total_unmatched_tracks += len(cycle_result.association.unmatched_track_indices)
        total_unmatched_detections += len(cycle_result.association.unmatched_detection_indices)
        total_skipped += len(cycle_result.skipped_detections)

        for track_idx, det_idx, _ in cycle_result.association.matches:
            track = tracks[track_idx]
            det = cycle_result.available_detections[det_idx]
            if det.truth_id is None:
                truth_conflicting_matches += 1
            elif track.truth_id == det.truth_id:
                truth_consistent_matches += 1
            else:
                truth_conflicting_matches += 1

        if idx % max(args.verbose_every, 1) == 0:
            print(
                f"[t={scan.time_s:6.1f}s] det={len(scan.detections):2d} "
                f"avail={len(cycle_result.available_detections):2d} "
                f"skip={len(cycle_result.skipped_detections):2d} "
                f"gated={len(cycle_result.gated_candidates):2d} "
                f"match={len(cycle_result.association.matches):2d} "
                f"unmatched_tracks={len(cycle_result.association.unmatched_track_indices):2d} "
                f"unmatched_det={len(cycle_result.association.unmatched_detection_indices):2d}"
            )

    total_cycles = max(len(scans), 1)
    print("\nSummary")
    print(f"- Cycles: {len(scans)}")
    print(f"- Mean gated pairs per cycle: {total_gated / total_cycles:.2f}")
    print(f"- Mean matches per cycle: {total_matches / total_cycles:.2f}")
    print(f"- Mean unmatched tracks per cycle: {total_unmatched_tracks / total_cycles:.2f}")
    print(f"- Mean unmatched detections per cycle: {total_unmatched_detections / total_cycles:.2f}")
    print(f"- Total skipped detections due to sensor availability: {total_skipped}")
    print(f"- Truth-consistent matches: {truth_consistent_matches}")
    print(f"- Truth-conflicting matches (includes false alarms): {truth_conflicting_matches}")

    avg_missed = float(np.mean([track.missed_count for track in tracks])) if tracks else 0.0
    print(f"- Final mean track missed_count: {avg_missed:.2f}")


if __name__ == "__main__":
    main()

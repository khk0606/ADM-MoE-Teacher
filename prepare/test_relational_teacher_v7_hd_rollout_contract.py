#!/usr/bin/env python3
"""CPU-only gate tests for the Teacher-v7 High-Desk K=1 canary."""

from __future__ import annotations

from copy import deepcopy

from relational_teacher_v7_hd_rollout_contract import (
    compute_checks,
    compute_scene_checks,
)


def good_metrics():
    cases = []
    for scene_id in ("room_0101", "room_0102"):
        for stratum in (
            "bed_01", "high_desk_chair_06", "legacy_chair_06",
            "chair_01" if scene_id == "room_0101" else "chair_05",
        ):
            for _ in range(2):
                cases.append({
                    "scene_id": scene_id,
                    "training_stratum": stratum,
                    "target_mae": {"base": 1.0, "candidate": 0.99},
                })
    return {
        "relational_cases": cases,
        "v5_replay_cases": [
            {"relative_degradation": value} for value in (0.01, -0.01, 0.02)
        ],
        "high_desk": {"target_mae_base": 1.0, "target_mae_candidate": 0.98},
        "overall": {
            "target_mae_base": 1.0,
            "target_mae_candidate": 0.99,
            "semantic_violation_base": 0.3,
            "semantic_violation_candidate": 0.3,
            "prompt_invariance_mse_base": 0.1,
            "prompt_invariance_mse_candidate": 0.104,
            "v5_replay_relative_degradation": 0.02,
        },
    }


def main() -> None:
    metrics = good_metrics()
    scenes = compute_scene_checks(metrics)
    assert scenes == {"room_0101": True, "room_0102": True}
    assert all(compute_checks(metrics, scenes, True, True).values())

    tampered = deepcopy(metrics)
    tampered["high_desk"]["target_mae_candidate"] = 1.01
    checks = compute_checks(tampered, compute_scene_checks(tampered), True, True)
    assert not checks["high_desk_target_mae_improves"]

    tampered = deepcopy(metrics)
    tampered["v5_replay_cases"][0]["relative_degradation"] = 0.16
    checks = compute_checks(tampered, compute_scene_checks(tampered), True, True)
    assert not checks["each_v5_replay_case_retained_15pct"]
    print("[PASS] v7 K=1 canary 16-case High-Desk and v5 gates")
    print("[PASS] High-Desk/v5 tamper guards")


if __name__ == "__main__":
    main()

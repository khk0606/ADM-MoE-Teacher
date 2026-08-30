#!/usr/bin/env python3
"""CPU tests for Teacher-v7 full train-only K=3 gates."""

from __future__ import annotations

from copy import deepcopy

from relational_teacher_v7_hd_full_k3_contract import full_checks


def metrics(candidate: float, high_candidate: float):
    target_groups = {
        f"room_01{scene}|stratum_{stratum}": {
            "count": 12,
            "base_mae": 1.0,
            "candidate_mae": candidate,
            "relative_change": candidate - 1.0,
            "case_win_rate": 0.75,
        }
        for scene in (1, 2) for stratum in range(4)
    }
    v5_groups = {
        target: {
            "count": count,
            "base_mae": 1.0,
            "candidate_mae": 1.02,
            "relative_degradation": 0.02,
        }
        for target, count in (("chair", 18), ("bed", 1), ("whiteboard", 6))
    }
    return {
        "overall": {
            "target_mae_base": 1.0,
            "target_mae_candidate": candidate,
            "case_win_rate": 0.75,
            "semantic_violation_base": 1.0,
            "semantic_violation_candidate": 0.99,
            "prompt_invariance_mse_base": 1.0,
            "prompt_invariance_mse_candidate": 1.02,
            "v5_replay_relative_degradation": 0.02,
        },
        "high_desk": {
            "target_mae_base": 1.0,
            "target_mae_candidate": high_candidate,
            "case_win_rate": 0.75,
        },
        "target_groups": target_groups,
        "v5_target_groups": v5_groups,
    }


def main() -> None:
    generations = [metrics(0.98, 0.99), metrics(0.99, 0.98), metrics(1.01, 1.01)]
    pooled = metrics(0.99, 0.995)
    checks = full_checks(pooled, generations, True)
    assert checks and all(checks.values())

    tampered = deepcopy(pooled)
    tampered["target_groups"]["room_011|stratum_0"]["relative_change"] = 0.11
    assert not full_checks(tampered, generations, True)[
        "all_eight_scene_strata_bounded_10pct"
    ]
    tampered = deepcopy(pooled)
    tampered["v5_target_groups"]["chair"]["relative_degradation"] = 0.11
    assert not full_checks(tampered, generations, True)[
        "all_three_v5_target_groups_bounded_10pct"
    ]
    assert not full_checks(pooled, generations, False)["reverse_diffusion_repeatable"]
    print("[PASS] Teacher-v7 full K=3 pooled, stratum and replay gates")
    print("[PASS] Teacher-v7 full K=3 failure and repeatability guards")


if __name__ == "__main__":
    main()

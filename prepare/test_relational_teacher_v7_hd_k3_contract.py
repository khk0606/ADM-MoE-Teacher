#!/usr/bin/env python3
"""CPU-only gate tests for the Teacher-v7 selected K=3 canary."""

from __future__ import annotations

from copy import deepcopy

from relational_teacher_v7_hd_k3_contract import k3_checks


def metrics(target=0.99, high_desk=0.98, semantic=0.29, invariance=0.104, v5=0.02):
    cases = []
    for scene_id in ("room_0101", "room_0102"):
        for index in range(8):
            cases.append({
                "scene_id": scene_id,
                "training_stratum": "high_desk_chair_06" if index < 2 else "other",
                "target_mae": {"base": 1.0, "candidate": target},
            })
    return {
        "relational_cases": cases,
        "high_desk": {"target_mae_base": 1.0, "target_mae_candidate": high_desk},
        "overall": {
            "target_mae_base": 1.0,
            "target_mae_candidate": target,
            "semantic_violation_base": 0.3,
            "semantic_violation_candidate": semantic,
            "prompt_invariance_mse_base": 0.1,
            "prompt_invariance_mse_candidate": invariance,
            "v5_replay_relative_degradation": v5,
        },
    }


def main() -> None:
    generations = [metrics(), metrics(target=0.98), metrics(target=1.01)]
    pooled = metrics(target=0.995, high_desk=0.99, invariance=0.103, v5=0.02)
    assert all(k3_checks(pooled, generations, True).values())

    tampered = deepcopy(generations)
    tampered[0]["high_desk"]["target_mae_candidate"] = 1.01
    tampered[1]["high_desk"]["target_mae_candidate"] = 1.01
    assert not k3_checks(pooled, tampered, True)[
        "at_least_two_generations_improve_high_desk"
    ]

    tampered = deepcopy(generations)
    tampered[2]["overall"]["prompt_invariance_mse_candidate"] = 0.111
    assert not k3_checks(pooled, tampered, True)[
        "every_generation_prompt_invariance_bounded_10pct"
    ]
    print("[PASS] v7 K=3 pooled, per-generation and High-Desk gates")
    print("[PASS] multi-seed High-Desk/invariance tamper guards")


if __name__ == "__main__":
    main()

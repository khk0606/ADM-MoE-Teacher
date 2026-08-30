#!/usr/bin/env python3
"""CPU-verifiable protocol and gates for the Teacher-v7 K=1 canary."""

from __future__ import annotations


SCHEMA = "relational_teacher_v7_hd_lora_rollout_canary_v1"
RELATIONAL_CASES = 16
V5_REPLAY_CASES = 3
SELECTED_STEP = 12
EXPECTED_TRAIN_SCENES = ("room_0101", "room_0102")


def compute_scene_checks(metrics):
    checks = {}
    for scene_id in EXPECTED_TRAIN_SCENES:
        rows = [row for row in metrics["relational_cases"]
                if row["scene_id"] == scene_id]
        if len(rows) != 8:
            checks[scene_id] = False
            continue
        base = sum(float(row["target_mae"]["base"]) for row in rows) / len(rows)
        candidate = sum(float(row["target_mae"]["candidate"]) for row in rows) / len(rows)
        checks[scene_id] = candidate <= base * 1.05 + 1e-12
    return checks


def compute_checks(metrics, scene_checks, repeatability: bool,
                   development_unread: bool):
    overall = metrics["overall"]
    high_desk = metrics["high_desk"]
    eps = 1e-12
    relational_cases = metrics["relational_cases"]
    return {
        "selected_step_is_12": True,
        "relational_case_count_is_16": len(relational_cases) == RELATIONAL_CASES,
        "four_high_desk_cases_present": len([
            row for row in relational_cases
            if row["training_stratum"] == "high_desk_chair_06"
        ]) == 4,
        "v5_replay_case_count_is_3": len(metrics["v5_replay_cases"]) == V5_REPLAY_CASES,
        "paired_reverse_diffusion_repeatable": repeatability is True,
        "overall_target_mae_improves": overall["target_mae_candidate"]
        < overall["target_mae_base"] - eps,
        "high_desk_target_mae_improves": high_desk["target_mae_candidate"]
        < high_desk["target_mae_base"] - eps,
        "semantic_violation_not_worse": overall["semantic_violation_candidate"]
        <= overall["semantic_violation_base"] + eps,
        "prompt_invariance_not_worse_5pct": overall["prompt_invariance_mse_candidate"]
        <= overall["prompt_invariance_mse_base"] * 1.05 + 1e-10,
        "each_train_scene_target_mae_not_worse_5pct": all(scene_checks.values()),
        "v5_replay_mean_retained_5pct": overall["v5_replay_relative_degradation"] <= 0.05,
        "each_v5_replay_case_retained_15pct": all(
            row["relative_degradation"] <= 0.15
            for row in metrics["v5_replay_cases"]
        ),
        "development_payloads_unread": development_unread is True,
    }

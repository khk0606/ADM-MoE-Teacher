#!/usr/bin/env python3
"""CPU-verifiable gates for the selected Teacher-v7 K=3 canary."""

from __future__ import annotations


SCHEMA = "relational_teacher_v7_hd_lora_selected_k3_canary_v1"
SELECTED_STEP = 12
GENERATIONS = 3
DIFFUSION_STEPS = 500
SEED = 20260911
RELATIONAL_CASES = 16
V5_CASES = 3
SEED_POLICY = {
    "generation_0": "sealed K=1 stable_seeds(seed, domain, case_id)",
    "generation_1_2": "stable_seeds(seed, domain + '_k3', case_id + '|g=N')",
}


def scene_bounded(metrics, factor: float) -> bool:
    for scene_id in ("room_0101", "room_0102"):
        rows = [row for row in metrics["relational_cases"]
                if row["scene_id"] == scene_id]
        if len(rows) not in {8, 24}:
            return False
        base = sum(float(row["target_mae"]["base"]) for row in rows) / len(rows)
        candidate = sum(float(row["target_mae"]["candidate"]) for row in rows) / len(rows)
        if candidate > base * factor + 1e-12:
            return False
    return True


def k3_checks(pooled, generations, repeatability: bool):
    overall = pooled["overall"]
    high_desk = pooled["high_desk"]
    eps = 1e-12
    target_wins = sum(
        row["overall"]["target_mae_candidate"]
        < row["overall"]["target_mae_base"] - eps
        for row in generations
    )
    high_desk_wins = sum(
        row["high_desk"]["target_mae_candidate"]
        < row["high_desk"]["target_mae_base"] - eps
        for row in generations
    )
    semantic_wins = sum(
        row["overall"]["semantic_violation_candidate"]
        <= row["overall"]["semantic_violation_base"] + eps
        for row in generations
    )
    return {
        "pooled_overall_target_mae_improves": overall["target_mae_candidate"]
        < overall["target_mae_base"] - eps,
        "pooled_high_desk_target_mae_improves": high_desk["target_mae_candidate"]
        < high_desk["target_mae_base"] - eps,
        "pooled_semantic_violation_not_worse": overall["semantic_violation_candidate"]
        <= overall["semantic_violation_base"] + eps,
        "pooled_prompt_invariance_not_worse_5pct": overall["prompt_invariance_mse_candidate"]
        <= overall["prompt_invariance_mse_base"] * 1.05 + 1e-10,
        "pooled_v5_replay_retained_5pct": overall["v5_replay_relative_degradation"] <= 0.05,
        "pooled_each_train_scene_bounded_5pct": scene_bounded(pooled, 1.05),
        "at_least_two_generations_improve_target_mae": target_wins >= 2,
        "at_least_two_generations_improve_high_desk": high_desk_wins >= 2,
        "at_least_two_generations_retain_semantics": semantic_wins >= 2,
        "every_generation_prompt_invariance_bounded_10pct": all(
            row["overall"]["prompt_invariance_mse_candidate"]
            <= row["overall"]["prompt_invariance_mse_base"] * 1.10 + 1e-10
            for row in generations
        ),
        "every_generation_v5_replay_bounded_10pct": all(
            row["overall"]["v5_replay_relative_degradation"] <= 0.10
            for row in generations
        ),
        "every_generation_each_train_scene_bounded_10pct": all(
            scene_bounded(row, 1.10) for row in generations
        ),
        "base_and_candidate_reverse_diffusion_repeatable": repeatability is True,
    }

#!/usr/bin/env python3
"""CPU-verifiable gates for Teacher-v7 full train-only K=3."""

from __future__ import annotations


SCHEMA = "relational_teacher_v7_hd_lora_full_train_k3_v1"
GENERATIONS = 3
DIFFUSION_STEPS = 500
SEED = 20260911
SELECTED_STEP = 12
TRAIN_MOTIONS = 48
RELATIONAL_CASES = 96
HIGH_DESK_CASES_PER_GENERATION = 24
V5_CASES = 25
SEED_POLICY = {
    "sealed_canary_cases": "reuse all three paired K=3 maps and seed pairs exactly",
    "remaining_relational_cases": (
        "generation_seeds(seed, 'full_relational_pair', "
        "scene|stratum|motion, generation); both prompts share the pair"
    ),
    "remaining_v5_cases": (
        "generation_seeds(seed, 'full_v5_replay', sample_id, generation)"
    ),
}


def full_checks(pooled, generations, repeatability: bool):
    overall = pooled["overall"]
    high_desk = pooled["high_desk"]
    eps = 1e-12
    return {
        "pooled_overall_target_mae_improves": overall["target_mae_candidate"]
        < overall["target_mae_base"] - eps,
        "pooled_high_desk_target_mae_improves": high_desk["target_mae_candidate"]
        < high_desk["target_mae_base"] - eps,
        "pooled_relational_case_win_rate_at_least_half": overall["case_win_rate"] >= 0.5,
        "pooled_high_desk_case_win_rate_at_least_half": high_desk["case_win_rate"] >= 0.5,
        "pooled_semantic_violation_not_worse": overall["semantic_violation_candidate"]
        <= overall["semantic_violation_base"] + eps,
        "pooled_prompt_invariance_not_worse_5pct": overall["prompt_invariance_mse_candidate"]
        <= overall["prompt_invariance_mse_base"] * 1.05 + 1e-10,
        "pooled_v5_replay_retained_5pct": overall["v5_replay_relative_degradation"] <= 0.05,
        "all_eight_scene_strata_bounded_10pct": (
            len(pooled["target_groups"]) == 8
            and all(row["relative_change"] <= 0.10
                    for row in pooled["target_groups"].values())
        ),
        "all_three_v5_target_groups_bounded_10pct": (
            set(pooled["v5_target_groups"]) == {"chair", "bed", "whiteboard"}
            and all(row["relative_degradation"] <= 0.10
                    for row in pooled["v5_target_groups"].values())
        ),
        "at_least_two_generations_improve_target_mae": sum(
            row["overall"]["target_mae_candidate"]
            < row["overall"]["target_mae_base"] - eps for row in generations
        ) >= 2,
        "at_least_two_generations_improve_high_desk": sum(
            row["high_desk"]["target_mae_candidate"]
            < row["high_desk"]["target_mae_base"] - eps for row in generations
        ) >= 2,
        "every_generation_prompt_invariance_bounded_10pct": all(
            row["overall"]["prompt_invariance_mse_candidate"]
            <= row["overall"]["prompt_invariance_mse_base"] * 1.10 + 1e-10
            for row in generations
        ),
        "every_generation_v5_replay_bounded_10pct": all(
            row["overall"]["v5_replay_relative_degradation"] <= 0.10
            for row in generations
        ),
        "reverse_diffusion_repeatable": repeatability is True,
    }

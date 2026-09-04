#!/usr/bin/env python3
"""Locked contract for Teacher-v10.1 full-field GT supervision."""

from __future__ import annotations

from typing import Mapping, Sequence

from relational_teacher_v10_supervised_capacity_contract import (
    ABSOLUTE_PRESENCE_LIMITS,
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    DIFFUSION_STEPS,
    EVAL_TIMESTEPS,
    EXPECTED_INSTANCES,
    GRAD_CLIP,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    PROMPT_IDS,
    ROLES,
    ROLLOUT_K,
    SCENES,
    SOURCE_SCENE,
    TIMESTEP_CYCLE,
    V5_RELATIVE_CAP,
    WEIGHT_DECAY,
    canonical_sha256,
    rollout_gate,
    rollout_rank_key,
    select_rollout_candidate,
)


SCHEMA = "relational_teacher_v101_fullfield_supervision_v1"
V10_SCHEMA = "relational_teacher_v10_supervised_capacity_v1"
TRAIN_STEPS = 1000
MONITOR_STEPS = (100, 250, 400, 600, 800, 1000)
LEARNING_RATE = 1e-4
REPLAY_START_STEP = 1
REPLAY_WEIGHT = 0.25
SHORTLIST_COUNT = 3
BACKGROUND_THRESHOLD = 0.05
ACTIVE_THRESHOLD = 0.30
HARD_NEGATIVE_POINTS = 512

LOSS_WEIGHTS = {
    "full_field_known": 4.0,
    "background_absolute": 2.0,
    "hard_false_positive": 2.0,
    "instance_full": 1.0,
    "instance_active": 4.0,
    "worst_instance_active": 2.0,
    "verified_union": 1.0,
    "explicit_negative_absolute": 2.0,
    "prompt_invariance": 0.25,
    "lora_regularizer": 1e-5,
}

POLICY = {
    "schema": "relational_teacher_v101_fullfield_policy_v1",
    "initialization": "sealed_v5r4_plus_fresh_zero_output_rank16_lora",
    "failed_v10_authority": "broad_false_positive_full_field_redesign",
    "supervision": "all_sittable_x0_over_every_non_unknown_point",
    "unknown_sittable": "only_unknown_sittable_points_are_ignored",
    "optimizer": "AdamW",
    "two_scene_gradient_accumulation_per_update": True,
    "prompt_pair_same_gt_timestep_and_noise": True,
    "timestep_cycle": list(TIMESTEP_CYCLE),
    "steps": TRAIN_STEPS,
    "learning_rate": LEARNING_RATE,
    "rank": LORA_RANK,
    "alpha": LORA_ALPHA,
    "background_threshold": BACKGROUND_THRESHOLD,
    "hard_negative_points": HARD_NEGATIVE_POINTS,
    "loss_weights": LOSS_WEIGHTS,
    "v5_replay": {
        "start_step": REPLAY_START_STEP,
        "weight": REPLAY_WEIGHT,
        "relative_cap": V5_RELATIVE_CAP,
    },
    "monitor_steps": list(MONITOR_STEPS),
    "shortlist": "prefer_v5_retained_then_worst_object_recall_mae_and_dense_mae",
    "actual_rollout": "two_scenes_x_two_prompts_x_k3_x_500_steps",
    "checkpoint": "write_only_when_actual_rollout_gate_passes",
    "development_scene_arrays": False,
    "paper_test": False,
}
POLICY_ID = canonical_sha256(POLICY)


def monitor_rank_key(row: Mapping[str, object]) -> tuple:
    base = sum(float(value) for value in row["base_v5_dense"]) / 3.0
    candidate = sum(float(value) for value in row["candidate_v5_dense"]) / 3.0
    retained = candidate <= base * V5_RELATIVE_CAP
    return (
        0 if retained else 1,
        -float(row["worst_instance_soft_recall"]),
        float(row["worst_instance_active_support_mae"]),
        float(row["known_dense_mae"]),
        candidate / max(base, 1e-12),
        int(row["step"]),
    )


def shortlist_monitor_steps(rows: Sequence[Mapping[str, object]]) -> list[int]:
    if [int(row["step"]) for row in rows] != list(MONITOR_STEPS):
        raise ValueError("Teacher-v10.1 monitor inventory changed")
    return [
        int(row["step"])
        for row in sorted(rows, key=monitor_rank_key)[:SHORTLIST_COUNT]
    ]


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_sha256",
    "monitor_rank_key",
    "rollout_gate",
    "rollout_rank_key",
    "select_rollout_candidate",
    "shortlist_monitor_steps",
]

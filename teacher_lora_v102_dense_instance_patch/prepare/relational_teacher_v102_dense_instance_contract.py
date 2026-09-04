#!/usr/bin/env python3
"""Locked Teacher-v10.2 dense per-instance support contract."""

from __future__ import annotations

from relational_teacher_v101_fullfield_contract import (
    ABSOLUTE_PRESENCE_LIMITS,
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    DIFFUSION_STEPS,
    EVAL_TIMESTEPS,
    EXPECTED_INSTANCES,
    GRAD_CLIP,
    LORA_ALPHA,
    LORA_RANK,
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
from relational_teacher_v101_fullfield_contract import (
    monitor_rank_key as _v101_monitor_rank_key,
)


SCHEMA = "relational_teacher_v102_dense_instance_supervision_v1"
V101_SCHEMA = "relational_teacher_v101_fullfield_supervision_v1"
MODEL_SEED = 20261031
TRAIN_STEPS = 1000
MONITOR_STEPS = (100, 250, 400, 600, 800, 1000)
LEARNING_RATE = 1e-4
REPLAY_START_STEP = 1
REPLAY_WEIGHT = 0.25
SHORTLIST_COUNT = 3
SUPPORT_THRESHOLD = 0.05
ACTIVE_THRESHOLD = 0.30
HARD_NEGATIVE_POINTS = 512

LOSS_WEIGHTS = {
    "full_field_known": 1.0,
    "background_absolute": 0.5,
    "hard_false_positive": 0.5,
    "instance_dense_support": 2.0,
    "instance_dense_active": 6.0,
    "worst_instance_dense_active": 3.0,
    "verified_union": 1.0,
    "explicit_negative_absolute": 1.0,
    "prompt_invariance": 0.25,
    "lora_regularizer": 1e-5,
}

POLICY = {
    "schema": "relational_teacher_v102_dense_instance_policy_v1",
    "initialization": "sealed_v5r4_plus_fresh_zero_output_rank16_lora",
    "failed_v101_authority": "room0102_normal_chair_dense_support_collapse",
    "supervision": "equal_macro_full_dense_support_for_each_verified_instance",
    "instance_support_threshold": SUPPORT_THRESHOLD,
    "instance_active_threshold": ACTIVE_THRESHOLD,
    "positive_support_hard_negative_exclusion": "union_of_all_three_instance_targets",
    "unknown_sittable": "only_unknown_sittable_points_are_ignored",
    "optimizer": "AdamW",
    "two_scene_gradient_accumulation_per_update": True,
    "prompt_pair_same_gt_timestep_and_noise": True,
    "timestep_cycle": list(TIMESTEP_CYCLE),
    "steps": TRAIN_STEPS,
    "learning_rate": LEARNING_RATE,
    "rank": LORA_RANK,
    "alpha": LORA_ALPHA,
    "hard_negative_points": HARD_NEGATIVE_POINTS,
    "loss_weights": LOSS_WEIGHTS,
    "v5_replay": {
        "start_step": REPLAY_START_STEP,
        "weight": REPLAY_WEIGHT,
        "relative_cap": V5_RELATIVE_CAP,
    },
    "monitor_steps": list(MONITOR_STEPS),
    "shortlist": "v5_retained_then_worst_object_recall_mae_and_dense_mae",
    "actual_rollout": "two_scenes_x_two_prompts_x_k3_x_500_steps",
    "checkpoint": "write_only_when_actual_rollout_gate_passes",
    "development_scene_arrays": False,
    "paper_test": False,
}
POLICY_ID = canonical_sha256(POLICY)


def monitor_rank_key(row):
    return _v101_monitor_rank_key(row)


def shortlist_monitor_steps(rows):
    if [int(row["step"]) for row in rows] != list(MONITOR_STEPS):
        raise ValueError("Teacher-v10.2 monitor inventory changed")
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

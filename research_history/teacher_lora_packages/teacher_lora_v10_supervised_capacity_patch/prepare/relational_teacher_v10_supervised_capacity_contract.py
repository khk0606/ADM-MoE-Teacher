#!/usr/bin/env python3
"""Locked policy for fresh supervised Teacher-v10 all-sittable training."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


SCHEMA = "relational_teacher_v10_supervised_capacity_v1"
V9812_SCHEMA = "relational_teacher_v9812_two_scene_multiupdate_calibration_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
SCENES = (SOURCE_SCENE, AUDIT_SCENE)
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
ROLES = ("bed", "normal_chair", "high_chair")
EXPECTED_INSTANCES = {
    SOURCE_SCENE: ("bed_01", "chair_01", "chair_06"),
    AUDIT_SCENE: ("bed_01", "chair_05", "chair_06"),
}

MODEL_SEED = 20261029
DIFFUSION_STEPS = 500
TRAIN_STEPS = 1200
MONITOR_STEPS = (100, 250, 500, 750, 1000, 1200)
TIMESTEP_CYCLE = (499, 475, 425, 350, 250, 175, 100, 50, 25, 10, 0)
EVAL_TIMESTEPS = (50, 250, 450)
LORA_RANK = 16
LORA_ALPHA = 16.0
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
SHORTLIST_COUNT = 3
ROLLOUT_K = 3
REPLAY_START_STEP = 601
REPLAY_WEIGHT = 0.05
V5_RELATIVE_CAP = 1.05
PROMPT_INVARIANCE_CAP = 0.02

LOSS_WEIGHTS = {
    "instance_full": 1.0,
    "instance_active": 4.0,
    "verified_union": 1.0,
    "environment": 0.25,
    "explicit_negative_absolute": 1.0,
    "prompt_invariance": 0.25,
    "lora_regularizer": 1e-5,
}

ABSOLUTE_PRESENCE_LIMITS = {
    "minimum_soft_recall": 0.75,
    "minimum_topk_overlap": 0.50,
    "maximum_active_support_mae": 0.10,
    "maximum_hotspot_centroid_distance_xy": 0.60,
    "maximum_negative_mean": 0.10,
    "maximum_negative_max": 0.80,
}

POLICY = {
    "schema": "relational_teacher_v10_supervised_capacity_policy_v1",
    "initialization": "sealed_v5r4_plus_fresh_zero_output_lora",
    "supervision": "q_sample_from_all_sittable_x0_label",
    "optimizer": "AdamW",
    "two_scene_gradient_accumulation_per_update": True,
    "prompt_pair_same_gt_timestep_and_noise": True,
    "timestep_cycle": list(TIMESTEP_CYCLE),
    "rank": LORA_RANK,
    "alpha": LORA_ALPHA,
    "steps": TRAIN_STEPS,
    "learning_rate": LEARNING_RATE,
    "loss_weights": LOSS_WEIGHTS,
    "unknown_sittable": "ignored",
    "explicit_negative": "absolute_zero_target",
    "v5_replay": {
        "start_step": REPLAY_START_STEP,
        "weight": REPLAY_WEIGHT,
        "relative_cap": V5_RELATIVE_CAP,
    },
    "monitor_steps": list(MONITOR_STEPS),
    "shortlist_count": SHORTLIST_COUNT,
    "actual_rollout": "two_scenes_x_two_prompts_x_k3_x_500_steps",
    "checkpoint": "write_only_when_actual_rollout_gate_passes",
    "development_scene_arrays": False,
    "paper_test": False,
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


POLICY_ID = canonical_sha256(POLICY)


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " is non-finite")
    return result


def monitor_rank_key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
    return (
        -_finite(row["worst_instance_soft_recall"], "monitor recall"),
        _finite(row["worst_instance_active_support_mae"], "monitor MAE"),
        _finite(row["known_dense_mae"], "monitor dense MAE"),
        int(row["step"]),
    )


def shortlist_monitor_steps(rows: Sequence[Mapping[str, object]]) -> list[int]:
    if [int(row["step"]) for row in rows] != list(MONITOR_STEPS):
        raise ValueError("monitor step inventory changed")
    ranked = sorted(rows, key=monitor_rank_key)
    return [int(row["step"]) for row in ranked[:SHORTLIST_COUNT]]


def rollout_gate(
    *,
    all_three_counts: Mapping[str, Mapping[str, int]],
    prompt_invariance: Mapping[str, Sequence[float]],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    checkpoint_state_changed: bool,
) -> dict[str, bool]:
    if set(all_three_counts) != set(SCENES) or set(prompt_invariance) != set(SCENES):
        raise ValueError("rollout scene inventory changed")
    checks: dict[str, bool] = {
        "lora_state_changed": bool(checkpoint_state_changed),
    }
    for scene in SCENES:
        if set(all_three_counts[scene]) != {"watch", "write"}:
            raise ValueError(scene + " prompt-count inventory changed")
        invariance = [_finite(value, scene + " invariance") for value in prompt_invariance[scene]]
        if len(invariance) != ROLLOUT_K:
            raise ValueError(scene + " prompt-invariance K changed")
        checks[scene + "_watch_all_three_at_least_two_of_three"] = (
            int(all_three_counts[scene]["watch"]) >= 2
        )
        checks[scene + "_write_all_three_at_least_two_of_three"] = (
            int(all_three_counts[scene]["write"]) >= 2
        )
        checks[scene + "_prompt_invariance_bounded"] = max(invariance) <= PROMPT_INVARIANCE_CAP
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 fixed-probe inventory changed")
    base_mean = sum(_finite(value, "base v5") for value in base_v5_dense) / 3.0
    candidate_mean = sum(_finite(value, "candidate v5") for value in candidate_v5_dense) / 3.0
    checks["v5_fixed_probe_retained_5pct"] = candidate_mean <= base_mean * V5_RELATIVE_CAP
    return checks


def rollout_rank_key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
    counts = row["all_three_counts"]
    minimum_count = min(
        int(counts[scene][prompt])
        for scene in SCENES
        for prompt in ("watch", "write")
    )
    pooled = row["pooled_metrics"]
    worst_recall = min(
        _finite(pooled[scene][role]["soft_recall"], "pooled recall")
        for scene in SCENES
        for role in ROLES
    )
    worst_mae = max(
        _finite(pooled[scene][role]["active_support_mae"], "pooled MAE")
        for scene in SCENES
        for role in ROLES
    )
    return (-float(minimum_count), -worst_recall, worst_mae, int(row["step"]))


def select_rollout_candidate(rows: Sequence[Mapping[str, object]]) -> int | None:
    eligible = [row for row in rows if row.get("eligible") is True]
    if not eligible:
        return None
    eligible.sort(key=rollout_rank_key)
    return int(eligible[0]["step"])


__all__ = [
    "ABSOLUTE_PRESENCE_LIMITS",
    "AUDIT_SCENE",
    "DEVELOPMENT_SCENE",
    "DIFFUSION_STEPS",
    "EVAL_TIMESTEPS",
    "EXPECTED_INSTANCES",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "LORA_ALPHA",
    "LORA_RANK",
    "LOSS_WEIGHTS",
    "MODEL_SEED",
    "MONITOR_STEPS",
    "POLICY",
    "POLICY_ID",
    "PROMPT_IDS",
    "REPLAY_START_STEP",
    "REPLAY_WEIGHT",
    "ROLES",
    "ROLLOUT_K",
    "SCENES",
    "SCHEMA",
    "SHORTLIST_COUNT",
    "SOURCE_SCENE",
    "TIMESTEP_CYCLE",
    "TRAIN_STEPS",
    "V5_RELATIVE_CAP",
    "V9812_SCHEMA",
    "WEIGHT_DECAY",
    "canonical_sha256",
    "monitor_rank_key",
    "rollout_gate",
    "rollout_rank_key",
    "select_rollout_candidate",
    "shortlist_monitor_steps",
]

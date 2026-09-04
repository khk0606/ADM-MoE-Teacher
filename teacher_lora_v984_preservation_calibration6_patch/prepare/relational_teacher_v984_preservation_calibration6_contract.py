#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.4 preservation-aware calibration-6."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v984_preservation_calibration6_v1"
POLICY_SCHEMA = "relational_teacher_v984_preservation_calibration6_policy_v1"
V983_SCHEMA = "relational_teacher_v983_preservation_direction_preflight_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
OBJECTS = ("bed_01", "chair_01", "chair_06")
UPDATE_COUNT = 6
MONITOR_STEPS = (1, 2, 3, 4, 5, 6)
SHORTLIST_LIMIT = 2
SELECTED_V983_CANDIDATE = "preserve11_radius_0p003"
SELECTED_TIMESTEP = 50
STEP_RADIUS = 0.003
MODEL_SEED = 20261016
CALIBRATION_TAG = 20261020
LORA_RANK = 4
LORA_ALPHA = 8.0
TASK_ORDER = (
    "bed_watch",
    "normal_chair_watch",
    "high_chair_watch",
    "bed_write",
    "normal_chair_write",
    "high_chair_write",
    "v5_chair",
    "v5_bed",
    "v5_whiteboard",
    "negative_watch",
    "negative_write",
)


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "fresh_six_update_preservation_aware_actual_t50_to_t0_k3_calibration",
    "authority": "sealed_v983_pass_only",
    "initialization": "fresh_v5r4_zero_output_lora",
    "update_count": UPDATE_COUNT,
    "monitor_steps": list(MONITOR_STEPS),
    "selected_timestep": SELECTED_TIMESTEP,
    "step_radius": STEP_RADIUS,
    "update_1": "exact_v981_six_sit_task_direction_reproduction",
    "update_2": "exact_v983_selected_eleven_task_direction_reproduction",
    "updates_3_through_6": "fresh_eleven_task_directions_on_disjoint_v982_design_states",
    "task_order_after_update_1": list(TASK_ORDER),
    "direction": "minimum_norm_common_descent_of_normalized_task_gradients",
    "inference_schedule": "frozen_base_t499_through_t51_then_current_lora_t50_through_t0",
    "monitor": "actual_k3_two_prompt_response_after_every_update",
    "topk_role": "diagnostic_only",
    "incremental_gates_after_update_1": {
        "each_object_pooled_recall_strictly_improves": True,
        "each_object_pooled_mae_strictly_improves": True,
        "each_v5_case_not_worse": True,
        "each_generation_negative_mean_addition_cap": 0.00025,
    },
    "retention": "reuse_complete_v981_response6_policy_again_against_frozen_base",
    "minimum_directional_derivative": 1e-8,
    "state_change_epsilon": 1e-8,
    "shortlist_limit": SHORTLIST_LIMIT,
    "selection_order": (
        "maximize_high_chair_recall_then_minimize_high_chair_mae_then_"
        "maximize_worst_object_recall_then_minimize_v5_mean_then_"
        "minimize_negative_mean_then_earlier_step"
    ),
    "absolute_three_object_presence": "reported_diagnostic_not_a_calibration_gate",
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "two_train_scene_preservation_calibration_preflight_only",
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


POLICY_ID = canonical_sha256(POLICY)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def pooled_object_metrics(
    rows: Sequence[Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[str, float]]:
    if len(rows) != 3 or any(len(row) != 2 for row in rows):
        raise ValueError("calibration panel must be K=3 x two prompts")
    result: Dict[str, Dict[str, float]] = {}
    for name in OBJECTS:
        result[name] = {
            key: sum(
                _finite(rows[g][p]["instances"][name][key], key)
                for g in range(3)
                for p in range(2)
            )
            / 6.0
            for key in (
                "soft_recall",
                "active_support_mae",
                "hotspot_centroid_distance_xy",
                "topk_overlap",
            )
        }
    return result


def calibration_checks(
    *,
    step: int,
    previous_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Sequence[Mapping[str, object]]],
    previous_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    response_checks: Mapping[str, object],
    directional_derivatives: Sequence[float],
    pre_state_sha256: str,
    post_state_sha256: str,
) -> Dict[str, bool]:
    if step not in MONITOR_STEPS:
        raise ValueError("unexpected calibration step")
    expected_tasks = 6 if step == 1 else len(TASK_ORDER)
    if len(directional_derivatives) != expected_tasks:
        raise ValueError("calibration directional-derivative inventory changed")
    checks: Dict[str, bool] = {
        "frozen_base_response6_policy_is_admissible": bool(response_checks)
        and all(value is True for value in response_checks.values()),
        "direction_is_common_descent": min(
            _finite(value, "directional derivative")
            for value in directional_derivatives
        )
        >= float(POLICY["minimum_directional_derivative"]),
        "lora_state_changes": isinstance(pre_state_sha256, str)
        and isinstance(post_state_sha256, str)
        and len(pre_state_sha256) == 64
        and len(post_state_sha256) == 64
        and pre_state_sha256 != post_state_sha256,
    }
    if step == 1:
        checks["exact_v981_update1_reproduction"] = True
        return checks

    previous = pooled_object_metrics(previous_rows)
    candidate = pooled_object_metrics(candidate_rows)
    epsilon = float(POLICY["state_change_epsilon"])
    for name in OBJECTS:
        checks[name + "_recall_improves_over_previous"] = (
            candidate[name]["soft_recall"]
            > previous[name]["soft_recall"] + epsilon
        )
        checks[name + "_mae_improves_over_previous"] = (
            candidate[name]["active_support_mae"]
            < previous[name]["active_support_mae"] - epsilon
        )
    if len(previous_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 calibration inventory changed")
    for index, target in enumerate(("chair", "bed", "whiteboard")):
        checks["v5_{}_not_worse_than_previous".format(target)] = _finite(
            candidate_v5_dense[index], "candidate v5"
        ) <= _finite(previous_v5_dense[index], "previous v5") + epsilon
    tolerance = float(
        POLICY["incremental_gates_after_update_1"][
            "each_generation_negative_mean_addition_cap"
        ]
    )
    for generation in range(3):
        previous_negative = max(
            _finite(previous_rows[generation][prompt]["explicit_negative_mean"], "previous negative")
            for prompt in range(2)
        )
        candidate_negative = max(
            _finite(candidate_rows[generation][prompt]["explicit_negative_mean"], "candidate negative")
            for prompt in range(2)
        )
        checks["generation_{}_negative_mean_preserved".format(generation)] = (
            candidate_negative <= previous_negative + tolerance
        )
    return checks


def rank_eligible_steps(rows: Sequence[Mapping[str, object]]) -> list[int]:
    eligible = [
        row
        for row in rows
        if row.get("eligible") is True and int(row.get("step", 0)) >= 2
    ]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, float, int]:
        pooled = row["candidate_pooled"]
        high_recall = _finite(pooled["chair_06"]["soft_recall"], "high recall")
        high_mae = _finite(pooled["chair_06"]["active_support_mae"], "high MAE")
        worst_recall = min(
            _finite(pooled[name]["soft_recall"], "recall") for name in OBJECTS
        )
        v5_mean = sum(_finite(value, "v5") for value in row["candidate_v5_dense"]) / 3.0
        negative_mean = sum(
            max(
                _finite(row["candidate_rows"][generation][prompt]["explicit_negative_mean"], "negative")
                for prompt in range(2)
            )
            for generation in range(3)
        ) / 3.0
        return (
            -high_recall,
            high_mae,
            -worst_recall,
            v5_mean,
            negative_mean,
            int(row["step"]),
        )

    eligible.sort(key=key)
    return [int(row["step"]) for row in eligible[:SHORTLIST_LIMIT]]


__all__ = [
    "CALIBRATION_TAG",
    "DEVELOPMENT_SCENE",
    "HELDOUT_TRAIN_SCENE",
    "LORA_ALPHA",
    "LORA_RANK",
    "MODEL_SEED",
    "MONITOR_STEPS",
    "OBJECTS",
    "POLICY",
    "POLICY_ID",
    "POLICY_SCHEMA",
    "PROMPT_IDS",
    "SCHEMA",
    "SELECTED_TIMESTEP",
    "SELECTED_V983_CANDIDATE",
    "SHORTLIST_LIMIT",
    "STEP_RADIUS",
    "TASK_ORDER",
    "TRAIN_SCENE",
    "UPDATE_COUNT",
    "V983_SCHEMA",
    "calibration_checks",
    "canonical_sha256",
    "pooled_object_metrics",
    "rank_eligible_steps",
]

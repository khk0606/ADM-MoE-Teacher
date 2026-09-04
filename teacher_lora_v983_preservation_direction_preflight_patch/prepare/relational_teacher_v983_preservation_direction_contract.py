#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.3 preservation-aware direction preflight."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v983_preservation_direction_preflight_v1"
POLICY_SCHEMA = "relational_teacher_v983_preservation_direction_policy_v1"
V982_SCHEMA = "relational_teacher_v982_rollout_state_calibration6_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
OBJECTS = ("bed_01", "chair_01", "chair_06")
SELECTED_TIMESTEP = 50
MODEL_SEED = 20261016
PREFLIGHT_TAG = 20261019
LORA_RANK = 4
LORA_ALPHA = 8.0
STEP_RADII = (0.00025, 0.0005, 0.001, 0.002, 0.003)
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
    "decision_unit": "hypothetical_second_update_actual_t50_to_t0_k3_response",
    "authority": "sealed_v982_failure_audit_only",
    "initialization": "fresh_v5r4_zero_output_lora_then_exact_v981_update1_reproduction",
    "design_state": "exact_v982_update2_frozen_base_t50_state",
    "selected_timestep": SELECTED_TIMESTEP,
    "step_radii": list(STEP_RADII),
    "task_order": list(TASK_ORDER),
    "direction": "minimum_norm_common_descent_of_eleven_normalized_gradients",
    "sit_tasks": "six_prompt_by_verified_object_losses",
    "v5_tasks": "three_fixed_replay_mse_losses",
    "negative_tasks": "two_prompt_specific_physical_mean_losses_on_explicit_negatives",
    "candidate_schedule": "frozen_base_t499_through_t51_then_step1_plus_candidate_direction_t50_through_t0",
    "topk_role": "diagnostic_only",
    "gates": {
        "reuse_v981_response6_gates_against_frozen_base": True,
        "each_object_pooled_recall_strictly_improves_over_step1": True,
        "each_object_pooled_mae_strictly_improves_over_step1": True,
        "each_v5_case_not_worse_than_step1": True,
        "each_generation_negative_mean_not_worse_than_step1_tolerance": 0.00025,
        "minimum_directional_derivative": 1e-8,
        "map_change_epsilon": 1e-8,
    },
    "selection_order": (
        "maximize_high_chair_recall_gain_over_step1_then_minimize_high_chair_mae_"
        "then_minimize_v5_mean_then_minimize_negative_mean_then_smaller_radius"
    ),
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "fresh_preservation_aware_rollout_state_calibration6_only",
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
        raise ValueError("preservation panel must be K=3 x two prompts")
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


def preservation_checks(
    *,
    step1_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Sequence[Mapping[str, object]]],
    step1_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    response_checks: Mapping[str, object],
    directional_derivatives: Sequence[float],
    maximum_incremental_map_delta: float,
) -> Dict[str, bool]:
    if not response_checks:
        raise ValueError("frozen-Base response checks are absent")
    if len(step1_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 preservation inventory changed")
    if len(directional_derivatives) != len(TASK_ORDER):
        raise ValueError("eleven directional derivatives are required")
    epsilon = float(POLICY["gates"]["map_change_epsilon"])
    step1 = pooled_object_metrics(step1_rows)
    candidate = pooled_object_metrics(candidate_rows)
    checks: Dict[str, bool] = {
        "frozen_base_response6_policy_is_admissible": all(
            value is True for value in response_checks.values()
        ),
        "eleven_task_direction_is_common_descent": min(
            _finite(value, "directional derivative")
            for value in directional_derivatives
        )
        >= float(POLICY["gates"]["minimum_directional_derivative"]),
        "candidate_increment_changes_final_maps": _finite(
            maximum_incremental_map_delta, "incremental map delta"
        )
        > epsilon,
    }
    for name in OBJECTS:
        checks[name + "_recall_improves_over_step1"] = (
            candidate[name]["soft_recall"]
            > step1[name]["soft_recall"] + epsilon
        )
        checks[name + "_mae_improves_over_step1"] = (
            candidate[name]["active_support_mae"]
            < step1[name]["active_support_mae"] - epsilon
        )
    for index, name in enumerate(("chair", "bed", "whiteboard")):
        checks["v5_{}_not_worse_than_step1".format(name)] = _finite(
            candidate_v5_dense[index], "candidate v5"
        ) <= _finite(step1_v5_dense[index], "step1 v5") + epsilon
    tolerance = float(
        POLICY["gates"]["each_generation_negative_mean_not_worse_than_step1_tolerance"]
    )
    for generation in range(3):
        step1_negative = max(
            _finite(step1_rows[generation][p]["explicit_negative_mean"], "step1 negative")
            for p in range(2)
        )
        candidate_negative = max(
            _finite(candidate_rows[generation][p]["explicit_negative_mean"], "candidate negative")
            for p in range(2)
        )
        checks["generation_{}_negative_mean_preserved".format(generation)] = (
            candidate_negative <= step1_negative + tolerance
        )
    return checks


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, float]:
        step1 = row["step1_pooled"]
        candidate = row["candidate_pooled"]
        high_gain = _finite(candidate["chair_06"]["soft_recall"], "high recall") - _finite(
            step1["chair_06"]["soft_recall"], "step1 high recall"
        )
        high_mae = _finite(candidate["chair_06"]["active_support_mae"], "high MAE")
        v5_mean = sum(_finite(value, "v5") for value in row["candidate_v5_dense"]) / 3.0
        negative_mean = sum(
            max(
                _finite(row["candidate_rows"][generation][p]["explicit_negative_mean"], "negative")
                for p in range(2)
            )
            for generation in range(3)
        ) / 3.0
        return (
            -high_gain,
            high_mae,
            v5_mean,
            negative_mean,
            _finite(row["radius"], "radius"),
        )

    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


__all__ = [
    "DEVELOPMENT_SCENE",
    "HELDOUT_TRAIN_SCENE",
    "LORA_ALPHA",
    "LORA_RANK",
    "MODEL_SEED",
    "OBJECTS",
    "POLICY",
    "POLICY_ID",
    "POLICY_SCHEMA",
    "PREFLIGHT_TAG",
    "PROMPT_IDS",
    "SCHEMA",
    "SELECTED_TIMESTEP",
    "STEP_RADII",
    "TASK_ORDER",
    "TRAIN_SCENE",
    "V982_SCHEMA",
    "canonical_sha256",
    "pooled_object_metrics",
    "preservation_checks",
    "rank_candidates",
]

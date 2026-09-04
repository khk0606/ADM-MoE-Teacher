#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.2 rollout-state calibration-6."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v982_rollout_state_calibration6_v1"
POLICY_SCHEMA = "relational_teacher_v982_rollout_state_calibration6_policy_v1"
V981_SCHEMA = "relational_teacher_v981_rollout_state_response6_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
OBJECTS = ("bed_01", "chair_01", "chair_06")
UPDATE_COUNT = 6
MONITOR_STEPS = (1, 2, 3, 4, 5, 6)
SHORTLIST_LIMIT = 2
SELECTED_NAME = "t50_radius_0p003"
SELECTED_TIMESTEP = 50
STEP_RADIUS = 0.003
# This must remain the v9.8 construction seed.  The zero-output LoRA A
# matrices are random, so changing it would change the selected direction.
MODEL_SEED = 20261016
CALIBRATION_TAG = 20261018
LORA_RANK = 4
LORA_ALPHA = 8.0


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "fresh_six_common_descent_updates_with_actual_t50_to_t0_k3_monitors",
    "train_scene": TRAIN_SCENE,
    "selected_response": SELECTED_NAME,
    "selected_timestep": SELECTED_TIMESTEP,
    "step_radius": STEP_RADIUS,
    "update_count": UPDATE_COUNT,
    "monitor_steps": list(MONITOR_STEPS),
    "shortlist_limit": SHORTLIST_LIMIT,
    "prompt_ids": list(PROMPT_IDS),
    "objects": list(OBJECTS),
    "initialization": "fresh_sealed_v5r4_plus_zero_output_lora",
    "update_1": "exact_v98_selected_design_state_direction_reproduction",
    "updates_2_through_6": "new_disjoint_frozen_base_t50_design_states",
    "update_rule": "recompute_six_task_minimum_norm_common_descent_then_apply_exact_euclidean_radius",
    "inference_schedule_for_monitor": "frozen_base_t499_through_t51_then_current_lora_t50_through_t0",
    "monitor_domain": "sealed_v97_k3_base_trajectories_and_prompt_pair",
    "topk_role": "diagnostic_only",
    "eligibility": {
        "reuse_v981_response6_retention_and_response_checks": True,
        "post_first_update_required": True,
        "each_object_pooled_recall_strictly_improves": True,
        "each_object_pooled_mae_strictly_improves": True,
        "minimum_directional_derivative": 1e-8,
        "state_change_epsilon": 1e-8,
    },
    "selection_order": (
        "maximize_high_chair_recall_then_minimize_high_chair_mae_then_"
        "maximize_worst_object_recall_then_minimize_macro_mae_then_earlier_step"
    ),
    "absolute_three_object_presence": "reported_diagnostic_not_a_calibration_gate",
    "checkpoint_policy": "no_model_state_is_serialized",
    "pass_authority": "two_train_scene_rollout_state_calibration_preflight_only",
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


def calibration_design_seeds(update: int) -> tuple[int, int]:
    """Return deterministic, mutually disjoint design seeds for updates 2..6."""

    if update < 2 or update > UPDATE_COUNT:
        raise ValueError("new design seeds exist only for updates 2 through 6")
    digest = hashlib.sha256(
        f"{MODEL_SEED}|v982|calibration|{update}".encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


def pooled_object_metrics(
    rows: Sequence[Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[str, float]]:
    if len(rows) != 3 or any(len(row) != 2 for row in rows):
        raise ValueError("calibration monitor must be K=3 x two prompts")
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


def calibration_step_checks(
    *,
    step: int,
    base_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Sequence[Mapping[str, object]]],
    response_checks: Mapping[str, object],
    directional_derivatives: Sequence[float],
    pre_state_sha256: str,
    post_state_sha256: str,
) -> Dict[str, bool]:
    if step not in MONITOR_STEPS:
        raise ValueError("unexpected calibration step")
    if not response_checks:
        raise ValueError("response checks are absent")
    checks = {
        "response6_policy_is_admissible": all(
            value is True for value in response_checks.values()
        ),
        "six_task_direction_is_common_descent": len(directional_derivatives) == 6
        and min(
            _finite(value, "directional derivative")
            for value in directional_derivatives
        )
        >= float(POLICY["eligibility"]["minimum_directional_derivative"]),
        "lora_state_changes": isinstance(pre_state_sha256, str)
        and isinstance(post_state_sha256, str)
        and len(pre_state_sha256) == 64
        and len(post_state_sha256) == 64
        and pre_state_sha256 != post_state_sha256,
    }
    base = pooled_object_metrics(base_rows)
    candidate = pooled_object_metrics(candidate_rows)
    epsilon = float(POLICY["eligibility"]["state_change_epsilon"])
    for name in OBJECTS:
        checks[name + "_pooled_recall_strictly_improves"] = (
            candidate[name]["soft_recall"]
            > base[name]["soft_recall"] + epsilon
        )
        checks[name + "_pooled_mae_strictly_improves"] = (
            candidate[name]["active_support_mae"]
            < base[name]["active_support_mae"] - epsilon
        )
    return checks


def rank_eligible_steps(rows: Sequence[Mapping[str, object]]) -> list[int]:
    eligible = [
        row
        for row in rows
        if row.get("eligible") is True and int(row.get("step", 0)) >= 2
    ]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, int]:
        pooled = row["candidate_pooled"]
        high_recall = _finite(pooled["chair_06"]["soft_recall"], "high recall")
        high_mae = _finite(pooled["chair_06"]["active_support_mae"], "high MAE")
        worst_recall = min(
            _finite(pooled[name]["soft_recall"], "recall") for name in OBJECTS
        )
        macro_mae = sum(
            _finite(pooled[name]["active_support_mae"], "MAE")
            for name in OBJECTS
        ) / 3.0
        return (-high_recall, high_mae, -worst_recall, macro_mae, int(row["step"]))

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
    "SELECTED_NAME",
    "SELECTED_TIMESTEP",
    "SHORTLIST_LIMIT",
    "STEP_RADIUS",
    "TRAIN_SCENE",
    "UPDATE_COUNT",
    "V981_SCHEMA",
    "calibration_design_seeds",
    "calibration_step_checks",
    "canonical_sha256",
    "pooled_object_metrics",
    "rank_eligible_steps",
]

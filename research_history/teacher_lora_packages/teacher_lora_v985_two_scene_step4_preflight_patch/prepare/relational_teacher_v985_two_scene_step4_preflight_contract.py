#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.5 two-scene step-4 preflight."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v985_two_scene_step4_preflight_v1"
POLICY_SCHEMA = "relational_teacher_v985_two_scene_step4_preflight_policy_v1"
V984_SCHEMA = "relational_teacher_v984_preservation_calibration6_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
ROLES = ("bed", "normal_chair", "high_chair")
EXPECTED_INSTANCES = {
    SOURCE_SCENE: ("bed_01", "chair_01", "chair_06"),
    AUDIT_SCENE: ("bed_01", "chair_05", "chair_06"),
}
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_TIMESTEP = 50
STEP_RADIUS = 0.003
MODEL_SEED = 20261016
PREFLIGHT_TAG = 20261021
LORA_RANK = 4
LORA_ALPHA = 8.0


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "fresh_exact_step4_room0102_actual_t50_to_t0_k3_preflight",
    "authority": "sealed_v984_pass_selected_step4_only",
    "initialization": "fresh_v5r4_zero_output_lora",
    "reconstruction": "exact_v984_updates_1_through_4_on_room0101",
    "audit_scene": AUDIT_SCENE,
    "audit_schedule": "frozen_base_t499_through_t51_then_reconstructed_step4_lora_t50_through_t0",
    "generation_count": 3,
    "prompt_ids": list(PROMPT_IDS),
    "object_roles": list(ROLES),
    "topk_role": "diagnostic_only",
    "absolute_three_object_presence": "reported_diagnostic_not_a_preflight_gate",
    "gates": {
        "each_map_object_recall_absolute_tolerance": 0.005,
        "each_map_object_mae_absolute_tolerance": 0.005,
        "each_map_object_centroid_addition_cap_m": 0.10,
        "each_role_pooled_recall_strictly_improves": True,
        "each_role_pooled_mae_strictly_improves": True,
        "high_chair_at_least_two_generations_improve": True,
        "prompt_invariance_relative_cap": 1.05,
        "explicit_negative_mean_addition_cap": 0.005,
        "explicit_negative_max_addition_cap": 0.02,
        "v5_fixed_probe_relative_cap": 1.01,
        "epsilon": 1e-8,
    },
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "fresh_two_scene_preservation_response_preflight_only",
    "fail_authority": "cross_scene_direction_diagnosis_only",
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


def pooled_by_role(
    rows: Sequence[Sequence[Mapping[str, object]]],
    instance_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    if len(rows) != 3 or any(len(row) != 2 for row in rows):
        raise ValueError("two-scene preflight panel must be K=3 x two prompts")
    if len(instance_names) != 3:
        raise ValueError("two-scene verified instance inventory changed")
    result: Dict[str, Dict[str, float]] = {}
    for role, name in zip(ROLES, instance_names):
        for generation in range(3):
            for prompt in range(2):
                instances = rows[generation][prompt].get("instances")
                if not isinstance(instances, Mapping) or set(instances) != set(instance_names):
                    raise ValueError("two-scene metric instance inventory changed")
        result[role] = {
            key: sum(
                _finite(rows[g][p]["instances"][name][key], role + " " + key)
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


def absolute_presence_checks(
    metrics: Mapping[str, object], instance_names: Sequence[str]
) -> Dict[str, bool]:
    if len(instance_names) != 3:
        raise ValueError("presence instance inventory changed")
    instances = metrics.get("instances")
    if not isinstance(instances, Mapping) or set(instances) != set(instance_names):
        raise ValueError("presence metric inventory changed")
    checks: Dict[str, bool] = {}
    for role, name in zip(ROLES, instance_names):
        row = instances[name]
        checks[role + "_soft_recall_at_least_075"] = _finite(
            row["soft_recall"], role + " recall"
        ) >= 0.75
        checks[role + "_active_support_mae_at_most_010"] = _finite(
            row["active_support_mae"], role + " MAE"
        ) <= 0.10
        checks[role + "_hotspot_centroid_at_most_060m"] = _finite(
            row["hotspot_centroid_distance_xy"], role + " centroid"
        ) <= 0.60
        topk = _finite(row["topk_overlap"], role + " top-k")
        if not 0.0 <= topk <= 1.0:
            raise ValueError("top-k diagnostic is outside [0,1]")
    checks["explicit_negative_mean_at_most_010"] = _finite(
        metrics["explicit_negative_mean"], "negative mean"
    ) <= 0.10
    checks["explicit_negative_max_at_most_080"] = _finite(
        metrics["explicit_negative_max"], "negative max"
    ) <= 0.80
    return checks


def two_scene_preflight_checks(
    *,
    base_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Sequence[Mapping[str, object]]],
    instance_names: Sequence[str],
    base_prompt_invariance: Sequence[float],
    candidate_prompt_invariance: Sequence[float],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    maximum_map_delta: float,
) -> Dict[str, bool]:
    base = pooled_by_role(base_rows, instance_names)
    candidate = pooled_by_role(candidate_rows, instance_names)
    if len(base_prompt_invariance) != 3 or len(candidate_prompt_invariance) != 3:
        raise ValueError("prompt-invariance inventory changed")
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 fixed-probe inventory changed")
    limits = POLICY["gates"]
    epsilon = float(limits["epsilon"])
    checks: Dict[str, bool] = {}

    for generation in range(3):
        for prompt, prompt_name in enumerate(("watch", "write")):
            for role, name in zip(ROLES, instance_names):
                before = base_rows[generation][prompt]["instances"][name]
                after = candidate_rows[generation][prompt]["instances"][name]
                prefix = "g{}_{}_{}".format(generation, prompt_name, role)
                checks[prefix + "_recall_retained"] = _finite(
                    after["soft_recall"], prefix + " recall"
                ) >= _finite(before["soft_recall"], prefix + " base recall") - float(
                    limits["each_map_object_recall_absolute_tolerance"]
                )
                checks[prefix + "_mae_retained"] = _finite(
                    after["active_support_mae"], prefix + " MAE"
                ) <= _finite(before["active_support_mae"], prefix + " base MAE") + float(
                    limits["each_map_object_mae_absolute_tolerance"]
                )
                checks[prefix + "_centroid_retained"] = _finite(
                    after["hotspot_centroid_distance_xy"], prefix + " centroid"
                ) <= _finite(
                    before["hotspot_centroid_distance_xy"], prefix + " base centroid"
                ) + float(limits["each_map_object_centroid_addition_cap_m"])
                topk = _finite(after["topk_overlap"], prefix + " top-k")
                if not 0.0 <= topk <= 1.0:
                    raise ValueError("top-k diagnostic is outside [0,1]")

    for role in ROLES:
        checks[role + "_pooled_recall_strictly_improves"] = (
            candidate[role]["soft_recall"] > base[role]["soft_recall"] + epsilon
        )
        checks[role + "_pooled_mae_strictly_improves"] = (
            candidate[role]["active_support_mae"]
            < base[role]["active_support_mae"] - epsilon
        )

    high_name = instance_names[2]
    high_improving = 0
    for generation in range(3):
        before_recall = sum(
            _finite(base_rows[generation][p]["instances"][high_name]["soft_recall"], "base high recall")
            for p in range(2)
        ) / 2.0
        after_recall = sum(
            _finite(candidate_rows[generation][p]["instances"][high_name]["soft_recall"], "candidate high recall")
            for p in range(2)
        ) / 2.0
        before_mae = sum(
            _finite(base_rows[generation][p]["instances"][high_name]["active_support_mae"], "base high MAE")
            for p in range(2)
        ) / 2.0
        after_mae = sum(
            _finite(candidate_rows[generation][p]["instances"][high_name]["active_support_mae"], "candidate high MAE")
            for p in range(2)
        ) / 2.0
        high_improving += int(
            after_recall > before_recall + epsilon and after_mae < before_mae - epsilon
        )
    checks["high_chair_at_least_two_generations_improve"] = high_improving >= 2

    for generation in range(3):
        checks["generation_{}_prompt_invariance_retained".format(generation)] = _finite(
            candidate_prompt_invariance[generation], "candidate invariance"
        ) <= _finite(base_prompt_invariance[generation], "base invariance") * float(
            limits["prompt_invariance_relative_cap"]
        ) + epsilon
        checks["generation_{}_negative_mean_retained".format(generation)] = max(
            _finite(candidate_rows[generation][p]["explicit_negative_mean"], "candidate negative mean")
            for p in range(2)
        ) <= max(
            _finite(base_rows[generation][p]["explicit_negative_mean"], "base negative mean")
            for p in range(2)
        ) + float(limits["explicit_negative_mean_addition_cap"])
        checks["generation_{}_negative_max_retained".format(generation)] = max(
            _finite(candidate_rows[generation][p]["explicit_negative_max"], "candidate negative max")
            for p in range(2)
        ) <= max(
            _finite(base_rows[generation][p]["explicit_negative_max"], "base negative max")
            for p in range(2)
        ) + float(limits["explicit_negative_max_addition_cap"])

    checks["v5_fixed_probe_retained_1pct"] = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) <= sum(_finite(value, "base v5") for value in base_v5_dense) * float(
        limits["v5_fixed_probe_relative_cap"]
    ) + epsilon
    checks["candidate_final_maps_changed"] = _finite(
        maximum_map_delta, "maximum map delta"
    ) > epsilon
    return checks


__all__ = [
    "AUDIT_SCENE",
    "DEVELOPMENT_SCENE",
    "EXPECTED_INSTANCES",
    "LORA_ALPHA",
    "LORA_RANK",
    "MODEL_SEED",
    "POLICY",
    "POLICY_ID",
    "POLICY_SCHEMA",
    "PREFLIGHT_TAG",
    "PROMPT_IDS",
    "RECONSTRUCTION_STEPS",
    "ROLES",
    "SCHEMA",
    "SELECTED_TIMESTEP",
    "SELECTED_V984_STEP",
    "SOURCE_SCENE",
    "STEP_RADIUS",
    "V984_SCHEMA",
    "absolute_presence_checks",
    "canonical_sha256",
    "pooled_by_role",
    "two_scene_preflight_checks",
]

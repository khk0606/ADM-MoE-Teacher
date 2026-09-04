#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.7 two-scene actual radius response."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


SCHEMA = "relational_teacher_v987_two_scene_radius_response_v1"
POLICY_SCHEMA = "relational_teacher_v987_two_scene_radius_response_policy_v1"
V986_SCHEMA = "relational_teacher_v986_cross_scene_direction_diagnosis_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_TIMESTEP = 50
SELECTED_DIRECTION = "bed_guard_0p10"
RADIUS_GRID = (0.0005, 0.001, 0.002, 0.003)
MODEL_SEED = 20261016
RESPONSE_TAG = 20261023
LORA_RANK = 4
LORA_ALPHA = 8.0


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "actual_two_scene_t50_to_t0_k3_radius_response_grid",
    "authority": "sealed_v986_pass_selected_bed_guard_0p10_only",
    "initialization": "fresh_v5r4_zero_output_lora",
    "reconstruction": "exact_v984_updates_1_through_4_on_room0101",
    "direction": "exact_v986_selected_19_task_bed_guard_0p10",
    "radius_grid": list(RADIUS_GRID),
    "common_start_state": "exact_reconstructed_step4_for_every_radius",
    "inference_schedule": "frozen_base_t499_through_t51_then_candidate_lora_t50_through_t0",
    "scenes": [SOURCE_SCENE, AUDIT_SCENE],
    "generation_count": 3,
    "prompt_ids": list(PROMPT_IDS),
    "scene_gate": "exact_v985_strict_three_role_response_policy_applied_independently",
    "topk_role": "diagnostic_only",
    "absolute_three_object_presence": "reported_diagnostic_not_a_response_gate",
    "selection_order": (
        "maximize_worst_scene_bed_recall_gain_then_"
        "maximize_worst_scene_high_chair_recall_gain_then_"
        "maximize_worst_scene_normal_chair_recall_gain_then_"
        "minimize_v5_mean_then_smaller_radius"
    ),
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "fresh_two_scene_common_direction_calibration_only",
    "fail_authority": "cross_scene_objective_or_inference_redesign_only",
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


def radius_response_checks(
    *,
    source_checks: Mapping[str, object],
    audit_checks: Mapping[str, object],
    selected_direction_reproduced: bool,
    candidate_state_changed: bool,
) -> dict[str, bool]:
    if not source_checks or not audit_checks:
        raise ValueError("both scene check mappings are required")
    if not all(isinstance(value, bool) for value in source_checks.values()):
        raise ValueError("source-scene checks are not booleans")
    if not all(isinstance(value, bool) for value in audit_checks.values()):
        raise ValueError("audit-scene checks are not booleans")
    return {
        "selected_v986_direction_exactly_reproduced": selected_direction_reproduced,
        "candidate_lora_state_changes_from_step4": candidate_state_changed,
        "room0101_response_is_admissible": all(source_checks.values()),
        "room0102_response_is_admissible": all(audit_checks.values()),
    }


def rank_eligible_radii(rows: Sequence[Mapping[str, object]]) -> list[float]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def gain(row: Mapping[str, object], scene: str, role: str) -> float:
        before = row["scene_base_pooled"][scene][role]["soft_recall"]
        after = row["scene_candidate_pooled"][scene][role]["soft_recall"]
        return _finite(after, "candidate recall") - _finite(before, "base recall")

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, float]:
        bed = min(gain(row, scene, "bed") for scene in (SOURCE_SCENE, AUDIT_SCENE))
        high = min(
            gain(row, scene, "high_chair") for scene in (SOURCE_SCENE, AUDIT_SCENE)
        )
        normal = min(
            gain(row, scene, "normal_chair") for scene in (SOURCE_SCENE, AUDIT_SCENE)
        )
        v5_mean = sum(_finite(value, "v5") for value in row["candidate_v5_dense"]) / 3.0
        radius = _finite(row["radius"], "radius")
        return (-bed, -high, -normal, v5_mean, radius)

    eligible.sort(key=key)
    return [float(row["radius"]) for row in eligible]


__all__ = [
    "AUDIT_SCENE",
    "DEVELOPMENT_SCENE",
    "LORA_ALPHA",
    "LORA_RANK",
    "MODEL_SEED",
    "POLICY",
    "POLICY_ID",
    "PROMPT_IDS",
    "RADIUS_GRID",
    "RECONSTRUCTION_STEPS",
    "RESPONSE_TAG",
    "SCHEMA",
    "SELECTED_DIRECTION",
    "SELECTED_TIMESTEP",
    "SELECTED_V984_STEP",
    "SOURCE_SCENE",
    "V986_SCHEMA",
    "canonical_sha256",
    "radius_response_checks",
    "rank_eligible_radii",
]

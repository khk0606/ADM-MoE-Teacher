#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.10 rollout-aligned recovery bracket."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


SCHEMA = "relational_teacher_v9810_rollout_aligned_recovery_v1"
POLICY_SCHEMA = "relational_teacher_v9810_rollout_aligned_recovery_policy_v1"
V989_SCHEMA = "relational_teacher_v989_rollout_aligned_radius_v1"
V988_SCHEMA = "relational_teacher_v988_rollout_aligned_direction_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_TIMESTEP = 50
SELECTED_DIRECTION = "audit_bed_guard_0p10"
FAILED_RADIUS_GRID = (0.00025, 0.0005, 0.001, 0.002, 0.003)
RADIUS_GRID = (0.004, 0.005, 0.006, 0.007, 0.008)
MODEL_SEED = 20261016
RESPONSE_TAG = 20261026
LORA_RANK = 4
LORA_ALPHA = 8.0


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "actual_two_scene_t50_to_t0_k3_monotone_recovery_bracket",
    "authority": "sealed_v989_only_room0102_bed_pair_failed_with_monotone_recovery",
    "initialization": "fresh_v5r4_zero_output_lora",
    "reconstruction": "exact_v984_updates_1_through_4_on_room0101",
    "direction": "exact_v988_selected_51_task_audit_bed_guard_0p10",
    "failed_radius_grid": list(FAILED_RADIUS_GRID),
    "recovery_radius_grid": list(RADIUS_GRID),
    "bracket_basis": {
        "room0102_bed_recall": "strictly_increased_across_every_v989_radius",
        "room0102_bed_mae": "strictly_decreased_across_every_v989_radius",
        "lower_bound": "above_failed_v989_maximum_radius",
        "upper_bound": "bounded_at_0p008",
    },
    "common_start_state": "exact_reconstructed_step4_for_every_radius",
    "inference_schedule": "frozen_base_t499_through_t51_then_candidate_lora_t50_through_t0",
    "scenes": [SOURCE_SCENE, AUDIT_SCENE],
    "generation_count": 3,
    "prompt_ids": list(PROMPT_IDS),
    "scene_gate": "exact_v985_strict_three_role_response_policy_applied_independently",
    "topk_role": "diagnostic_only",
    "absolute_three_object_presence": "reported_diagnostic_not_a_response_gate",
    "selection_order": "smallest_admissible_radius_then_v5_mean",
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "fresh_two_scene_rollout_aligned_calibration_only",
    "fail_authority": "cross_scene_training_objective_redesign_only",
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


def validate_failed_recovery_signature(value: Mapping[str, object]) -> None:
    rows = value.get("response_rows")
    if not isinstance(rows, list) or len(rows) != len(FAILED_RADIUS_GRID):
        raise ValueError("sealed v9.8.9 radius inventory changed")
    recalls = []
    maes = []
    for expected_radius, row in zip(FAILED_RADIUS_GRID, rows):
        if _finite(row.get("radius"), "failed radius") != expected_radius:
            raise ValueError("sealed v9.8.9 radius order changed")
        if row.get("eligible") is not False:
            raise ValueError("sealed v9.8.9 failure eligibility changed")
        if row.get("scene_failed_checks", {}).get(SOURCE_SCENE) != []:
            raise ValueError("sealed v9.8.9 source scene no longer passes")
        if row.get("scene_failed_checks", {}).get(AUDIT_SCENE) != [
            "bed_pooled_mae_strictly_improves",
            "bed_pooled_recall_strictly_improves",
        ]:
            raise ValueError("sealed v9.8.9 audit failure is no longer Bed-only")
        pooled = row["scene_candidate_pooled"][AUDIT_SCENE]["bed"]
        recalls.append(_finite(pooled["soft_recall"], "audit Bed recall"))
        maes.append(_finite(pooled["active_support_mae"], "audit Bed MAE"))
    if not all(right > left for left, right in zip(recalls, recalls[1:])):
        raise ValueError("v9.8.9 audit Bed recall recovery is not strictly monotone")
    if not all(right < left for left, right in zip(maes, maes[1:])):
        raise ValueError("v9.8.9 audit Bed MAE recovery is not strictly monotone")


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
        "selected_v988_direction_exactly_reproduced": selected_direction_reproduced,
        "candidate_lora_state_changes_from_step4": candidate_state_changed,
        "room0101_response_is_admissible": all(source_checks.values()),
        "room0102_response_is_admissible": all(audit_checks.values()),
    }


def rank_eligible_radii(rows: Sequence[Mapping[str, object]]) -> list[float]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float]:
        radius = _finite(row["radius"], "radius")
        v5_mean = sum(_finite(value, "v5") for value in row["candidate_v5_dense"]) / 3.0
        return (radius, v5_mean)

    eligible.sort(key=key)
    return [float(row["radius"]) for row in eligible]


__all__ = [
    "AUDIT_SCENE",
    "DEVELOPMENT_SCENE",
    "FAILED_RADIUS_GRID",
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
    "V988_SCHEMA",
    "V989_SCHEMA",
    "canonical_sha256",
    "radius_response_checks",
    "rank_eligible_radii",
    "validate_failed_recovery_signature",
]

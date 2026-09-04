#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.6 cross-scene direction diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence

import numpy as np


SCHEMA = "relational_teacher_v986_cross_scene_direction_diagnosis_v1"
POLICY_SCHEMA = "relational_teacher_v986_cross_scene_direction_policy_v1"
V985_SCHEMA = "relational_teacher_v985_two_scene_step4_preflight_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_TIMESTEP = 50
STEP_RADIUS = 0.003
MODEL_SEED = 20261016
DIAGNOSIS_TAG = 20261022
LORA_RANK = 4
LORA_ALPHA = 8.0
FW_ITERATIONS = 16384
MINIMUM_DIRECTIONAL_DERIVATIVE = 1e-8
RAW_GRAM_ASYMMETRY_CAP = 1e-8
BED_GUARD_MIXES = (0.10, 0.25, 0.50)

TASK_ORDER = (
    "source_bed_watch",
    "source_normal_chair_watch",
    "source_high_chair_watch",
    "source_bed_write",
    "source_normal_chair_write",
    "source_high_chair_write",
    "source_negative_watch",
    "source_negative_write",
    "audit_bed_watch",
    "audit_normal_chair_watch",
    "audit_high_chair_watch",
    "audit_bed_write",
    "audit_normal_chair_write",
    "audit_high_chair_write",
    "audit_negative_watch",
    "audit_negative_write",
    "v5_chair",
    "v5_bed",
    "v5_whiteboard",
)
AUDIT_BED_TASKS = ("audit_bed_watch", "audit_bed_write")
CANDIDATE_NAMES = (
    "all19_common",
    "bed_guard_0p10",
    "bed_guard_0p25",
    "bed_guard_0p50",
    "audit_bed_only",
)

POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "exact_step4_two_scene_19_task_gradient_geometry",
    "authority": "sealed_v985_cross_scene_failure_only",
    "initialization": "fresh_v5r4_zero_output_lora",
    "reconstruction": "exact_v984_updates_1_through_4_on_room0101",
    "design_scenes": [SOURCE_SCENE, AUDIT_SCENE],
    "design_state": "independent_frozen_base_t50_states_for_watch_and_write",
    "task_order": list(TASK_ORDER),
    "gradient_normalization": "one_unit_lora_gradient_per_scalar_task",
    "gram_construction": {
        "accumulation_dtype": "float64",
        "raw_max_asymmetry_cap": RAW_GRAM_ASYMMETRY_CAP,
        "canonicalization": "half_of_raw_plus_raw_transpose",
    },
    "direction_candidates": list(CANDIDATE_NAMES),
    "audit_bed_guard_mixes": list(BED_GUARD_MIXES),
    "selection_gate": {
        "every_one_of_19_directional_derivatives_at_least": MINIMUM_DIRECTIONAL_DERIVATIVE,
        "both_room0102_bed_derivatives_at_least": MINIMUM_DIRECTIONAL_DERIVATIVE,
        "finite_nonzero_direction": True,
    },
    "selection_order": (
        "maximize_minimum_room0102_bed_derivative_then_"
        "maximize_minimum_all_task_derivative_then_lower_bed_mix"
    ),
    "diagnostic_only": True,
    "no_candidate_parameter_update": True,
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "actual_two_scene_k3_direction_response_grid_only",
    "fail_authority": "objective_or_model_capacity_redesign_only",
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


POLICY_ID = canonical_sha256(POLICY)


def cross_scene_design_seeds(scene_id: str) -> tuple[int, int]:
    if scene_id not in (SOURCE_SCENE, AUDIT_SCENE):
        raise ValueError("cross-scene design seed requested for an unauthorized scene")
    digest = hashlib.sha256(
        "{}|{}|{}|cross-scene-t50".format(MODEL_SEED, DIAGNOSIS_TAG, scene_id).encode(
            "utf-8"
        )
    ).digest()
    mask = (1 << 63) - 1
    return (
        int.from_bytes(digest[:8], "big") & mask,
        int.from_bytes(digest[8:16], "big") & mask,
    )


def _validated_gram(gram: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(gram, dtype=np.float64)
    count = len(TASK_ORDER)
    if matrix.shape != (count, count):
        raise ValueError("cross-scene Gram matrix shape changed")
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=1e-9, atol=1e-10
    ):
        raise ValueError("cross-scene Gram matrix is non-finite or asymmetric")
    if not np.allclose(np.diag(matrix), np.ones(count), rtol=2e-5, atol=2e-6):
        raise ValueError("task gradients are not unit normalized")
    if float(np.linalg.eigvalsh(matrix).min()) < -2e-5:
        raise ValueError("cross-scene Gram matrix is not positive semidefinite")
    return matrix


def _unit_coefficients(matrix: np.ndarray, coefficients: np.ndarray) -> tuple[np.ndarray, float]:
    vector = np.asarray(coefficients, dtype=np.float64)
    if vector.shape != (len(TASK_ORDER),) or not np.isfinite(vector).all():
        raise ValueError("direction coefficients changed")
    norm_sq = float(vector @ matrix @ vector)
    if not math.isfinite(norm_sq) or norm_sq <= 1e-20:
        raise ValueError("direction is zero/non-finite")
    norm = math.sqrt(norm_sq)
    return vector / norm, norm


def _frank_wolfe_min_norm_weights(matrix: np.ndarray) -> np.ndarray:
    """Pure NumPy copy of the sealed deterministic common-descent solver."""

    count = matrix.shape[0]
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    for _ in range(FW_ITERATIONS):
        gradient = matrix @ weights
        vertex = np.zeros(count, dtype=np.float64)
        vertex[int(np.argmin(gradient))] = 1.0
        delta = vertex - weights
        denominator = float(delta @ matrix @ delta)
        if denominator <= 1e-20:
            break
        gamma = float(np.clip(-(delta @ matrix @ weights) / denominator, 0.0, 1.0))
        weights = weights + gamma * delta
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return weights


def candidate_coefficients(gram: Sequence[Sequence[float]]) -> Dict[str, np.ndarray]:
    matrix = _validated_gram(gram)
    common = _frank_wolfe_min_norm_weights(matrix)
    common_unit, _ = _unit_coefficients(matrix, common)
    bed = np.zeros(len(TASK_ORDER), dtype=np.float64)
    for name in AUDIT_BED_TASKS:
        bed[TASK_ORDER.index(name)] = 0.5
    bed_unit, _ = _unit_coefficients(matrix, bed)
    result = {"all19_common": common_unit}
    for mix in BED_GUARD_MIXES:
        blended = (1.0 - mix) * common_unit + mix * bed_unit
        unit, _ = _unit_coefficients(matrix, blended)
        result["bed_guard_{:.2f}".format(mix).replace(".", "p")] = unit
    result["audit_bed_only"] = bed_unit
    if tuple(result) != CANDIDATE_NAMES:
        raise AssertionError("direction candidate order changed")
    return result


def diagnose_directions(gram: Sequence[Sequence[float]]) -> list[Dict[str, object]]:
    matrix = _validated_gram(gram)
    rows = []
    for name, coefficients in candidate_coefficients(matrix).items():
        norm_sq = float(coefficients @ matrix @ coefficients)
        if not math.isclose(norm_sq, 1.0, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError("candidate direction is not unit length")
        derivatives = matrix @ coefficients
        bed_values = [float(derivatives[TASK_ORDER.index(task)]) for task in AUDIT_BED_TASKS]
        checks = {
            "direction_is_finite_and_nonzero": bool(np.isfinite(coefficients).all()),
            "every_task_is_common_descent": float(derivatives.min())
            >= MINIMUM_DIRECTIONAL_DERIVATIVE,
            "both_room0102_bed_tasks_are_descent": min(bed_values)
            >= MINIMUM_DIRECTIONAL_DERIVATIVE,
        }
        rows.append(
            {
                "name": name,
                "effective_coefficients": coefficients.tolist(),
                "directional_derivatives": derivatives.tolist(),
                "minimum_directional_derivative": float(derivatives.min()),
                "minimum_audit_bed_derivative": min(bed_values),
                "checks": checks,
                "failed_checks": sorted(key for key, passed in checks.items() if not passed),
                "eligible": all(checks.values()),
            }
        )
    return rows


def rank_eligible_directions(rows: Sequence[Mapping[str, object]]) -> list[str]:
    index = {name: position for position, name in enumerate(CANDIDATE_NAMES)}
    eligible = [row for row in rows if row.get("eligible") is True]
    eligible.sort(
        key=lambda row: (
            -float(row["minimum_audit_bed_derivative"]),
            -float(row["minimum_directional_derivative"]),
            index[str(row["name"])],
        )
    )
    return [str(row["name"]) for row in eligible]


def conflict_pairs(gram: Sequence[Sequence[float]]) -> list[list[object]]:
    matrix = _validated_gram(gram)
    return [
        [TASK_ORDER[left], TASK_ORDER[right], float(matrix[left, right])]
        for left in range(len(TASK_ORDER))
        for right in range(left + 1, len(TASK_ORDER))
        if float(matrix[left, right]) < 0.0
    ]


__all__ = [
    "AUDIT_BED_TASKS",
    "AUDIT_SCENE",
    "BED_GUARD_MIXES",
    "CANDIDATE_NAMES",
    "DEVELOPMENT_SCENE",
    "DIAGNOSIS_TAG",
    "FW_ITERATIONS",
    "LORA_ALPHA",
    "LORA_RANK",
    "MINIMUM_DIRECTIONAL_DERIVATIVE",
    "MODEL_SEED",
    "POLICY",
    "POLICY_ID",
    "PROMPT_IDS",
    "RAW_GRAM_ASYMMETRY_CAP",
    "RECONSTRUCTION_STEPS",
    "SCHEMA",
    "SELECTED_TIMESTEP",
    "SELECTED_V984_STEP",
    "SOURCE_SCENE",
    "STEP_RADIUS",
    "TASK_ORDER",
    "V985_SCHEMA",
    "candidate_coefficients",
    "canonical_sha256",
    "conflict_pairs",
    "cross_scene_design_seeds",
    "diagnose_directions",
    "rank_eligible_directions",
]

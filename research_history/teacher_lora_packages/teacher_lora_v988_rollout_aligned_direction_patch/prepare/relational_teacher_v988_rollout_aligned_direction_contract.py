#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.8 rollout-aligned direction diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence

import numpy as np


SCHEMA = "relational_teacher_v988_rollout_aligned_direction_v1"
POLICY_SCHEMA = "relational_teacher_v988_rollout_aligned_direction_policy_v1"
V987_SCHEMA = "relational_teacher_v987_two_scene_radius_response_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
ROLES = ("bed", "normal_chair", "high_chair")
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_TIMESTEP = 50
MODEL_SEED = 20261016
DIAGNOSIS_TAG = 20261024
LORA_RANK = 4
LORA_ALPHA = 8.0
FW_ITERATIONS = 32768
MINIMUM_DIRECTIONAL_DERIVATIVE = 1e-8
RAW_GRAM_ASYMMETRY_CAP = 1e-8
AUDIT_BED_GUARD_MIXES = (0.05, 0.10, 0.20, 0.30, 0.40)


def _task_order() -> tuple[str, ...]:
    result = []
    for scene_label in ("source", "audit"):
        for generation in range(3):
            for prompt in ("watch", "write"):
                for role in ROLES:
                    result.append(
                        "{}_g{}_{}_{}".format(scene_label, generation, prompt, role)
                    )
            for prompt in ("watch", "write"):
                result.append("{}_g{}_{}_negative".format(scene_label, generation, prompt))
    result.extend(("v5_chair", "v5_bed", "v5_whiteboard"))
    return tuple(result)


TASK_ORDER = _task_order()
AUDIT_BED_TASKS = tuple(
    "audit_g{}_{}_bed".format(generation, prompt)
    for generation in range(3)
    for prompt in ("watch", "write")
)
CANDIDATE_NAMES = (
    "all51_common",
    "audit_bed_guard_0p05",
    "audit_bed_guard_0p10",
    "audit_bed_guard_0p20",
    "audit_bed_guard_0p30",
    "audit_bed_guard_0p40",
    "audit_bed_only",
)


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "exact_actual_k3_t50_two_scene_51_task_gradient_geometry",
    "authority": "sealed_v987_room0102_bed_regression_only",
    "initialization": "fresh_v5r4_zero_output_lora",
    "reconstruction": "exact_v984_updates_1_through_4_on_room0101",
    "trajectory_states": (
        "exact_frozen_base_t50_states_from_v987_k3_generations_for_both_scenes"
    ),
    "task_order": list(TASK_ORDER),
    "task_inventory": {
        "per_scene_generation_prompt_role": 36,
        "per_scene_generation_prompt_negative": 12,
        "v5_fixed_probe": 3,
        "total": 51,
    },
    "gradient_normalization": "one_unit_lora_gradient_per_scalar_task",
    "gram_construction": {
        "accumulation_dtype": "float64",
        "raw_max_asymmetry_cap": RAW_GRAM_ASYMMETRY_CAP,
        "canonicalization": "half_of_raw_plus_raw_transpose",
    },
    "direction_candidates": list(CANDIDATE_NAMES),
    "audit_bed_guard_mixes": list(AUDIT_BED_GUARD_MIXES),
    "selection_gate": {
        "every_one_of_51_directional_derivatives_at_least": (
            MINIMUM_DIRECTIONAL_DERIVATIVE
        ),
        "every_one_of_6_room0102_bed_derivatives_at_least": (
            MINIMUM_DIRECTIONAL_DERIVATIVE
        ),
        "finite_nonzero_direction": True,
    },
    "selection_order": (
        "maximize_minimum_room0102_bed_derivative_then_"
        "maximize_minimum_all_task_derivative_then_lower_bed_mix"
    ),
    "diagnostic_only": True,
    "no_candidate_parameter_update": True,
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "actual_two_scene_k3_rollout_aligned_radius_grid_only",
    "fail_authority": "cross_scene_objective_or_inference_redesign_only",
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


POLICY_ID = canonical_sha256(POLICY)


def _validated_gram(gram: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(gram, dtype=np.float64)
    count = len(TASK_ORDER)
    if matrix.shape != (count, count):
        raise ValueError("rollout-aligned Gram matrix shape changed")
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=1e-9, atol=1e-10
    ):
        raise ValueError("rollout-aligned Gram matrix is non-finite or asymmetric")
    if not np.allclose(np.diag(matrix), np.ones(count), rtol=2e-5, atol=2e-6):
        raise ValueError("rollout-aligned task gradients are not unit normalized")
    if float(np.linalg.eigvalsh(matrix).min()) < -5e-5:
        raise ValueError("rollout-aligned Gram matrix is not positive semidefinite")
    return matrix


def _unit_coefficients(
    matrix: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, float]:
    vector = np.asarray(coefficients, dtype=np.float64)
    if vector.shape != (len(TASK_ORDER),) or not np.isfinite(vector).all():
        raise ValueError("rollout-aligned direction coefficients changed")
    norm_sq = float(vector @ matrix @ vector)
    if not math.isfinite(norm_sq) or norm_sq <= 1e-20:
        raise ValueError("rollout-aligned direction is zero/non-finite")
    norm = math.sqrt(norm_sq)
    return vector / norm, norm


def _frank_wolfe_min_norm_weights(matrix: np.ndarray) -> np.ndarray:
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


def candidate_coefficients(
    gram: Sequence[Sequence[float]],
) -> Dict[str, np.ndarray]:
    matrix = _validated_gram(gram)
    common = _frank_wolfe_min_norm_weights(matrix)
    common_unit, _ = _unit_coefficients(matrix, common)
    bed = np.zeros(len(TASK_ORDER), dtype=np.float64)
    for name in AUDIT_BED_TASKS:
        bed[TASK_ORDER.index(name)] = 1.0 / len(AUDIT_BED_TASKS)
    bed_unit, _ = _unit_coefficients(matrix, bed)
    result = {"all51_common": common_unit}
    for mix in AUDIT_BED_GUARD_MIXES:
        blended = (1.0 - mix) * common_unit + mix * bed_unit
        result["audit_bed_guard_{:.2f}".format(mix).replace(".", "p")] = (
            _unit_coefficients(matrix, blended)[0]
        )
    result["audit_bed_only"] = bed_unit
    if tuple(result) != CANDIDATE_NAMES:
        raise AssertionError("rollout-aligned candidate order changed")
    return result


def diagnose_directions(
    gram: Sequence[Sequence[float]],
) -> list[Dict[str, object]]:
    matrix = _validated_gram(gram)
    rows = []
    bed_indices = [TASK_ORDER.index(name) for name in AUDIT_BED_TASKS]
    for name, coefficients in candidate_coefficients(matrix).items():
        derivatives = matrix @ coefficients
        minimum_all = float(derivatives.min())
        minimum_bed = float(derivatives[bed_indices].min())
        finite = bool(np.isfinite(derivatives).all())
        checks = {
            "finite_directional_derivatives": finite,
            "every_task_is_common_descent": finite
            and minimum_all >= MINIMUM_DIRECTIONAL_DERIVATIVE,
            "every_audit_bed_task_is_common_descent": finite
            and minimum_bed >= MINIMUM_DIRECTIONAL_DERIVATIVE,
        }
        rows.append(
            {
                "name": name,
                "effective_coefficients": coefficients.tolist(),
                "directional_derivatives": derivatives.tolist(),
                "minimum_all_task_derivative": minimum_all,
                "minimum_audit_bed_derivative": minimum_bed,
                "checks": checks,
                "failed_checks": sorted(
                    key for key, passed in checks.items() if not passed
                ),
                "eligible": all(checks.values()),
            }
        )
    return rows


def rank_eligible_directions(rows: Sequence[Mapping[str, object]]) -> list[str]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def mix_value(name: str) -> float:
        if name == "all51_common":
            return 0.0
        if name == "audit_bed_only":
            return 1.0
        return float(name.rsplit("_", 1)[-1].replace("p", "."))

    eligible.sort(
        key=lambda row: (
            -float(row["minimum_audit_bed_derivative"]),
            -float(row["minimum_all_task_derivative"]),
            mix_value(str(row["name"])),
        )
    )
    return [str(row["name"]) for row in eligible]


def conflict_pairs(gram: Sequence[Sequence[float]]) -> int:
    matrix = _validated_gram(gram)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return int((upper < 0.0).sum())


__all__ = [
    "AUDIT_BED_GUARD_MIXES",
    "AUDIT_BED_TASKS",
    "AUDIT_SCENE",
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
    "ROLES",
    "SCHEMA",
    "SELECTED_TIMESTEP",
    "SELECTED_V984_STEP",
    "SOURCE_SCENE",
    "TASK_ORDER",
    "V987_SCHEMA",
    "canonical_sha256",
    "candidate_coefficients",
    "conflict_pairs",
    "diagnose_directions",
    "rank_eligible_directions",
]

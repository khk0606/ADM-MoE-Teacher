#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.12 two-scene multi-update calibration."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


SCHEMA = "relational_teacher_v9812_two_scene_multiupdate_calibration_v1"
POLICY_SCHEMA = "relational_teacher_v9812_two_scene_multiupdate_calibration_policy_v1"
V9811_SCHEMA = "relational_teacher_v9811_rollout_aligned_replication_v1"
V9810_SCHEMA = "relational_teacher_v9810_rollout_aligned_recovery_v1"
V988_SCHEMA = "relational_teacher_v988_rollout_aligned_direction_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
ROLES = ("bed", "normal_chair", "high_chair")
SELECTED_TIMESTEP = 50
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_DIRECTION = "audit_bed_guard_0p10"
SELECTED_RADIUS = 0.006
MODEL_SEED = 20261016
CALIBRATION_TAG = 20261028
GENERATION_COUNT = 3
UPDATE_COUNT = 6
MONITOR_STEPS = (1, 2, 3, 4, 5, 6)
STEP_RADIUS = 0.001
SHORTLIST_LIMIT = 1
HIGH_CHAIR_GUARD_MIXES = (0.05, 0.10, 0.20, 0.30, 0.40)
MINIMUM_DIRECTIONAL_DERIVATIVE = 1e-8
STATE_CHANGE_EPSILON = 1e-8
ROLE_REGRESSION_TOLERANCE = 0.005
LORA_RANK = 4
LORA_ALPHA = 8.0


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_pair(kind: str, generation: int) -> tuple[int, int]:
    if kind not in {"design", "audit"}:
        raise ValueError("unknown seed-table kind")
    if generation < 0 or generation >= GENERATION_COUNT:
        raise ValueError("generation must be 0, 1 or 2")
    digest = hashlib.sha256(
        "{}|v9812|{}|{}".format(MODEL_SEED, kind, generation).encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


DESIGN_SEED_TABLE = tuple(
    _seed_pair("design", generation) for generation in range(GENERATION_COUNT)
)
AUDIT_SEED_TABLE = tuple(
    _seed_pair("audit", generation) for generation in range(GENERATION_COUNT)
)


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "two_scene_six_update_rollout_aligned_actual_k3_calibration",
    "authority": "sealed_v9811_disjoint_seed_replication_pass_only",
    "initialization": "exact_reconstructed_v9810_radius_0p006_selected_state",
    "selected_timestep": SELECTED_TIMESTEP,
    "update_count": UPDATE_COUNT,
    "monitor_steps": list(MONITOR_STEPS),
    "step_radius": STEP_RADIUS,
    "design_seed_table": [list(row) for row in DESIGN_SEED_TABLE],
    "audit_seed_table": [list(row) for row in AUDIT_SEED_TABLE],
    "seed_exclusion": "design_and_audit_disjoint_from_each_other_and_v97_v9811",
    "gradient_tasks": "two_scenes_x_k3_x_two_prompts_x_three_roles_plus_negatives_plus_v5",
    "direction": "minimum_norm_common_descent_with_high_chair_guard_grid",
    "high_chair_guard_mixes": list(HIGH_CHAIR_GUARD_MIXES),
    "minimum_directional_derivative": MINIMUM_DIRECTIONAL_DERIVATIVE,
    "inference_schedule": "frozen_base_t499_through_t51_then_current_lora_t50_through_t0",
    "monitor": "actual_two_scene_k3_two_prompt_generation_after_every_update",
    "scene_gate": "exact_v985_strict_three_role_response_policy_applied_independently",
    "selected_state_preservation": {
        "per_scene_role_recall_drop_at_most": ROLE_REGRESSION_TOLERANCE,
        "per_scene_role_mae_increase_at_most": ROLE_REGRESSION_TOLERANCE,
    },
    "absolute_three_object_gate": "each_scene_and_prompt_at_least_two_of_three_generations",
    "topk_role": "diagnostic_only",
    "shortlist_limit": SHORTLIST_LIMIT,
    "selection_order": (
        "maximize_minimum_all_three_count_then_maximize_worst_high_chair_recall_"
        "then_minimize_worst_high_chair_mae_then_minimize_v5_mean_then_earlier_step"
    ),
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "shortlisted_state_reconstruction_and_checkpoint_export_gate_only",
    "fail_authority": "cross_scene_objective_or_model_capacity_redesign_only",
}


POLICY_ID = canonical_sha256(POLICY)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def all_three_counts(presence: Mapping[str, object]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        rows = presence.get(scene)
        if not isinstance(rows, list) or len(rows) != GENERATION_COUNT:
            raise ValueError(scene + " presence must be K=3")
        result[scene] = {}
        for prompt_index, prompt in enumerate(("watch", "write")):
            count = 0
            for generation in rows:
                if not isinstance(generation, list) or len(generation) != 2:
                    raise ValueError(scene + " presence prompt inventory changed")
                checks = generation[prompt_index]
                if not isinstance(checks, Mapping) or not checks:
                    raise ValueError(scene + " presence check mapping is absent")
                if all(value is True for value in checks.values()):
                    count += 1
            result[scene][prompt] = count
    return result


def _validated_gram(gram: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(gram, np.float64)
    if (
        matrix.shape != (51, 51)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-10)
        or not np.allclose(np.diag(matrix), np.ones(51), rtol=2e-5, atol=2e-6)
    ):
        raise ValueError("calibration Gram matrix is invalid")
    return matrix


def _unit_coefficients(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norm_sq = float(vector @ matrix @ vector)
    if not math.isfinite(norm_sq) or norm_sq <= 1e-20:
        raise ValueError("calibration coefficient direction is zero/non-finite")
    return vector / math.sqrt(norm_sq)


def direction_candidates(
    gram: Sequence[Sequence[float]],
    task_order: Sequence[str],
) -> list[dict[str, object]]:
    matrix = _validated_gram(gram)
    if len(task_order) != 51 or len(set(task_order)) != 51:
        raise ValueError("calibration task order changed")
    weights = np.full(51, 1.0 / 51.0, np.float64)
    for _ in range(32768):
        gradient = matrix @ weights
        vertex = np.zeros(51, np.float64)
        vertex[int(np.argmin(gradient))] = 1.0
        delta = vertex - weights
        denominator = float(delta @ matrix @ delta)
        if denominator <= 1e-20:
            break
        gamma = float(np.clip(-(delta @ matrix @ weights) / denominator, 0.0, 1.0))
        weights = weights + gamma * delta
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    common = _unit_coefficients(matrix, weights)
    high_indices = [
        index for index, name in enumerate(task_order) if name.endswith("_high_chair")
    ]
    if len(high_indices) != 12:
        raise ValueError("high-chair task inventory changed")
    high = np.zeros(51, np.float64)
    high[high_indices] = 1.0 / len(high_indices)
    high = _unit_coefficients(matrix, high)
    candidates = {"all51_common": common}
    for mix in HIGH_CHAIR_GUARD_MIXES:
        name = "high_chair_guard_{:.2f}".format(mix).replace(".", "p")
        candidates[name] = _unit_coefficients(
            matrix, (1.0 - mix) * common + mix * high
        )
    rows = []
    for name, coefficients in candidates.items():
        derivatives = matrix @ coefficients
        minimum_all = float(derivatives.min())
        minimum_high = float(derivatives[high_indices].min())
        checks = {
            "every_task_is_common_descent": minimum_all
            >= MINIMUM_DIRECTIONAL_DERIVATIVE,
            "every_high_chair_task_is_common_descent": minimum_high
            >= MINIMUM_DIRECTIONAL_DERIVATIVE,
        }
        rows.append(
            {
                "name": name,
                "effective_coefficients": coefficients.tolist(),
                "directional_derivatives": derivatives.tolist(),
                "minimum_all_task_derivative": minimum_all,
                "minimum_high_chair_derivative": minimum_high,
                "checks": checks,
                "failed_checks": sorted(
                    key for key, passed in checks.items() if not passed
                ),
                "eligible": all(checks.values()),
            }
        )
    return rows


def select_direction(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    eligible = [row for row in rows if row.get("eligible") is True]
    if not eligible:
        raise ValueError("no common-descent calibration direction is admissible")
    eligible.sort(
        key=lambda row: (
            -_finite(row["minimum_high_chair_derivative"], "minimum high derivative"),
            -_finite(row["minimum_all_task_derivative"], "minimum all derivative"),
            str(row["name"]),
        )
    )
    return eligible[0]


def calibration_checks(
    *,
    step: int,
    scene_checks: Mapping[str, Mapping[str, bool]],
    start_pooled: Mapping[str, Mapping[str, Mapping[str, float]]],
    candidate_pooled: Mapping[str, Mapping[str, Mapping[str, float]]],
    presence: Mapping[str, object],
    directional_derivatives: Sequence[float],
    pre_state_sha256: str,
    post_state_sha256: str,
) -> dict[str, bool]:
    if step not in MONITOR_STEPS:
        raise ValueError("unexpected calibration step")
    if len(directional_derivatives) != 51:
        raise ValueError("calibration requires exactly 51 task derivatives")
    checks = {
        "direction_is_common_descent": min(
            _finite(value, "directional derivative")
            for value in directional_derivatives
        )
        >= MINIMUM_DIRECTIONAL_DERIVATIVE,
        "lora_state_changes": isinstance(pre_state_sha256, str)
        and isinstance(post_state_sha256, str)
        and len(pre_state_sha256) == 64
        and len(post_state_sha256) == 64
        and pre_state_sha256 != post_state_sha256,
    }
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        current_checks = scene_checks.get(scene)
        if not isinstance(current_checks, Mapping) or not current_checks:
            raise ValueError(scene + " strict scene checks are absent")
        checks[scene + "_strict_response_is_admissible"] = all(
            value is True for value in current_checks.values()
        )
        for role in ROLES:
            start = start_pooled[scene][role]
            candidate = candidate_pooled[scene][role]
            checks[scene + "_" + role + "_recall_preserved"] = (
                _finite(candidate["soft_recall"], "candidate recall")
                >= _finite(start["soft_recall"], "start recall")
                - ROLE_REGRESSION_TOLERANCE
            )
            checks[scene + "_" + role + "_mae_preserved"] = (
                _finite(candidate["active_support_mae"], "candidate MAE")
                <= _finite(start["active_support_mae"], "start MAE")
                + ROLE_REGRESSION_TOLERANCE
            )
    counts = all_three_counts(presence)
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        for prompt in ("watch", "write"):
            checks[scene + "_" + prompt + "_all_three_at_least_two_of_three"] = (
                counts[scene][prompt] >= 2
            )
    return checks


def rank_shortlist(rows: Sequence[Mapping[str, object]]) -> list[int]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, int]:
        counts = row["all_three_counts"]
        minimum_count = min(
            int(counts[scene][prompt])
            for scene in (SOURCE_SCENE, AUDIT_SCENE)
            for prompt in ("watch", "write")
        )
        pooled = row["scene_candidate_pooled"]
        worst_high_recall = min(
            _finite(pooled[scene]["high_chair"]["soft_recall"], "high recall")
            for scene in (SOURCE_SCENE, AUDIT_SCENE)
        )
        worst_high_mae = max(
            _finite(pooled[scene]["high_chair"]["active_support_mae"], "high MAE")
            for scene in (SOURCE_SCENE, AUDIT_SCENE)
        )
        v5_mean = sum(_finite(value, "v5") for value in row["candidate_v5_dense"]) / 3.0
        return (-minimum_count, -worst_high_recall, worst_high_mae, v5_mean, int(row["step"]))

    eligible.sort(key=key)
    return [int(row["step"]) for row in eligible[:SHORTLIST_LIMIT]]


__all__ = [
    "AUDIT_SCENE", "AUDIT_SEED_TABLE", "CALIBRATION_TAG", "DESIGN_SEED_TABLE",
    "DEVELOPMENT_SCENE", "GENERATION_COUNT", "HIGH_CHAIR_GUARD_MIXES",
    "LORA_ALPHA", "LORA_RANK", "MINIMUM_DIRECTIONAL_DERIVATIVE", "MODEL_SEED",
    "MONITOR_STEPS", "POLICY", "POLICY_ID", "PROMPT_IDS", "RECONSTRUCTION_STEPS",
    "ROLES", "SCHEMA",
    "SELECTED_DIRECTION", "SELECTED_RADIUS", "SELECTED_TIMESTEP", "SHORTLIST_LIMIT",
    "SELECTED_V984_STEP", "SOURCE_SCENE", "STATE_CHANGE_EPSILON", "STEP_RADIUS",
    "UPDATE_COUNT", "V9810_SCHEMA", "V9811_SCHEMA", "V988_SCHEMA", "all_three_counts",
    "calibration_checks", "canonical_sha256", "direction_candidates",
    "rank_shortlist", "select_direction",
]

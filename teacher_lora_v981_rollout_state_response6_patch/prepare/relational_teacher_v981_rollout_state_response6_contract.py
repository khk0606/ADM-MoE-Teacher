#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.1 selected rollout-state K=3 response."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v981_rollout_state_response6_v1"
POLICY_SCHEMA = "relational_teacher_v981_rollout_state_response6_policy_v1"
V98_SCHEMA = "relational_teacher_v98_rollout_state_preflight_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
OBJECTS = ("bed_01", "chair_01", "chair_06")
GENERATION_COUNT = 3
SELECTED_NAME = "t50_radius_0p003"
SELECTED_TIMESTEP = 50
SELECTED_RADIUS = 0.003
# Exact v9.8 model/LoRA construction seed.  A fresh numerical seed would
# change the random zero-output LoRA A matrices and invalidate the selected
# direction hash even though B starts at zero.
SEED = 20261016
LORA_RANK = 4
LORA_ALPHA = 8.0


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "selected_t50_radius_0p003_resumed_final_500_step_k3_maps",
    "selected_candidate": SELECTED_NAME,
    "selected_timestep": SELECTED_TIMESTEP,
    "selected_radius": SELECTED_RADIUS,
    "generation_count": GENERATION_COUNT,
    "prompt_ids": list(PROMPT_IDS),
    "objects": list(OBJECTS),
    "adapter_activation": "frozen_base_t499_through_t51_then_selected_lora_t50_through_t0",
    "generation_0": "byte_exact_reuse_of_sealed_v98_audit_base_and_selected_candidate",
    "generation_1_2": "exact_v97_base_trajectory_capture_and_paired_t50_resume",
    "topk_role": "diagnostic_only",
    "retention": {
        "each_map_object_recall_absolute_tolerance": 0.005,
        "each_map_object_mae_absolute_tolerance": 0.005,
        "each_map_object_centroid_addition_cap_m": 0.10,
        "prompt_invariance_relative_cap": 1.05,
        "explicit_negative_mean_addition_cap": 0.005,
        "explicit_negative_max_addition_cap": 0.02,
        "v5_fixed_probe_relative_cap": 1.01,
        "epsilon": 1e-8,
    },
    "required_response": {
        "high_chair_at_least_two_generations_improve_recall_and_mae": True,
        "high_chair_pooled_recall_strictly_improves": True,
        "high_chair_pooled_mae_strictly_improves": True,
        "pooled_worst_object_recall_strictly_improves": True,
        "pooled_macro_mae_strictly_improves": True,
        "all_six_maps_retain_each_object": True,
        "each_generation_retains_prompt_invariance_and_negatives": True,
        "v5_fixed_probe_retained": True,
    },
    "absolute_three_object_presence": "reported_diagnostic_not_a_response_gate",
    "pass_authority": "fresh_rollout_state_calibration6_only",
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


def _validate_rows(rows: Sequence[Sequence[Mapping[str, object]]]) -> None:
    if len(rows) != GENERATION_COUNT or any(len(row) != 2 for row in rows):
        raise ValueError("response rows must be K=3 x two prompts")
    for generation in range(GENERATION_COUNT):
        for prompt in range(2):
            instances = rows[generation][prompt].get("instances")
            if not isinstance(instances, Mapping) or set(instances) != set(OBJECTS):
                raise ValueError("response instance inventory changed")


def pooled_object_metrics(
    rows: Sequence[Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[str, float]]:
    _validate_rows(rows)
    result: Dict[str, Dict[str, float]] = {}
    for name in OBJECTS:
        result[name] = {
            key: sum(
                _finite(rows[g][p]["instances"][name][key], key)
                for g in range(GENERATION_COUNT)
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


def response6_checks(
    *,
    base_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Sequence[Mapping[str, object]]],
    base_prompt_invariance: Sequence[float],
    candidate_prompt_invariance: Sequence[float],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    directional_derivatives: Sequence[float],
    maximum_map_delta: float,
) -> Dict[str, bool]:
    _validate_rows(base_rows)
    _validate_rows(candidate_rows)
    if len(base_prompt_invariance) != 3 or len(candidate_prompt_invariance) != 3:
        raise ValueError("prompt-invariance inventory changed")
    limits = POLICY["retention"]
    epsilon = float(limits["epsilon"])
    checks: Dict[str, bool] = {}

    for generation in range(GENERATION_COUNT):
        for prompt, prompt_name in enumerate(("watch", "write")):
            for name in OBJECTS:
                base = base_rows[generation][prompt]["instances"][name]
                candidate = candidate_rows[generation][prompt]["instances"][name]
                prefix = "g{}_{}_{}".format(generation, prompt_name, name)
                checks[prefix + "_recall_retained"] = _finite(
                    candidate["soft_recall"], prefix + " recall"
                ) >= _finite(base["soft_recall"], prefix + " base recall") - float(
                    limits["each_map_object_recall_absolute_tolerance"]
                )
                checks[prefix + "_mae_retained"] = _finite(
                    candidate["active_support_mae"], prefix + " MAE"
                ) <= _finite(base["active_support_mae"], prefix + " base MAE") + float(
                    limits["each_map_object_mae_absolute_tolerance"]
                )
                checks[prefix + "_centroid_retained"] = _finite(
                    candidate["hotspot_centroid_distance_xy"], prefix + " centroid"
                ) <= _finite(
                    base["hotspot_centroid_distance_xy"], prefix + " base centroid"
                ) + float(limits["each_map_object_centroid_addition_cap_m"])
                topk = _finite(candidate["topk_overlap"], prefix + " top-k")
                if not 0.0 <= topk <= 1.0:
                    raise ValueError("top-k diagnostic is outside [0,1]")

    base_pooled = pooled_object_metrics(base_rows)
    candidate_pooled = pooled_object_metrics(candidate_rows)
    high_improving_generations = 0
    for generation in range(GENERATION_COUNT):
        base_recall = sum(
            _finite(base_rows[generation][p]["instances"]["chair_06"]["soft_recall"], "base high recall")
            for p in range(2)
        ) / 2.0
        candidate_recall = sum(
            _finite(candidate_rows[generation][p]["instances"]["chair_06"]["soft_recall"], "candidate high recall")
            for p in range(2)
        ) / 2.0
        base_mae = sum(
            _finite(base_rows[generation][p]["instances"]["chair_06"]["active_support_mae"], "base high MAE")
            for p in range(2)
        ) / 2.0
        candidate_mae = sum(
            _finite(candidate_rows[generation][p]["instances"]["chair_06"]["active_support_mae"], "candidate high MAE")
            for p in range(2)
        ) / 2.0
        improves = candidate_recall > base_recall + epsilon and candidate_mae < base_mae - epsilon
        high_improving_generations += int(improves)
    checks["high_chair_at_least_two_generations_improve"] = high_improving_generations >= 2
    checks["high_chair_pooled_recall_strictly_improves"] = (
        candidate_pooled["chair_06"]["soft_recall"]
        > base_pooled["chair_06"]["soft_recall"] + epsilon
    )
    checks["high_chair_pooled_mae_strictly_improves"] = (
        candidate_pooled["chair_06"]["active_support_mae"]
        < base_pooled["chair_06"]["active_support_mae"] - epsilon
    )
    checks["pooled_worst_object_recall_strictly_improves"] = min(
        candidate_pooled[name]["soft_recall"] for name in OBJECTS
    ) > min(base_pooled[name]["soft_recall"] for name in OBJECTS) + epsilon
    checks["pooled_macro_mae_strictly_improves"] = sum(
        candidate_pooled[name]["active_support_mae"] for name in OBJECTS
    ) < sum(base_pooled[name]["active_support_mae"] for name in OBJECTS) - epsilon

    for generation in range(GENERATION_COUNT):
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

    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 fixed probe inventory changed")
    checks["v5_fixed_probe_retained_1pct"] = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) <= sum(_finite(value, "base v5") for value in base_v5_dense) * float(
        limits["v5_fixed_probe_relative_cap"]
    ) + epsilon
    if len(directional_derivatives) != 6:
        raise ValueError("six directional derivatives required")
    checks["selected_direction_is_common_descent"] = min(
        _finite(value, "directional derivative") for value in directional_derivatives
    ) >= 1e-8
    checks["candidate_final_maps_changed"] = _finite(
        maximum_map_delta, "maximum map delta"
    ) > epsilon
    return checks


__all__ = [
    "DEVELOPMENT_SCENE",
    "GENERATION_COUNT",
    "HELDOUT_TRAIN_SCENE",
    "LORA_ALPHA",
    "LORA_RANK",
    "OBJECTS",
    "POLICY",
    "POLICY_ID",
    "POLICY_SCHEMA",
    "PROMPT_IDS",
    "SCHEMA",
    "SEED",
    "SELECTED_NAME",
    "SELECTED_RADIUS",
    "SELECTED_TIMESTEP",
    "TRAIN_SCENE",
    "V98_SCHEMA",
    "canonical_sha256",
    "pooled_object_metrics",
    "response6_checks",
]

#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8 rollout-state response preflight."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v98_rollout_state_preflight_v1"
POLICY_SCHEMA = "relational_teacher_v98_rollout_state_policy_v1"
FAILED_V97_SCHEMA = "relational_teacher_v97_early_rollout_k3_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
OBJECTS = ("bed_01", "chair_01", "chair_06")
CAPTURE_TIMESTEPS = (400, 200, 50)
STEP_RADII = (0.001, 0.003, 0.01)
SEED = 20261016
LORA_RANK = 4
LORA_ALPHA = 8.0


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "resumed_final_500_step_reverse_diffusion_map",
    "train_scene": TRAIN_SCENE,
    "design_seed_domain": "new_source_disjoint_design_trajectory",
    "audit_seed_domain": "exact_v97_generation_0_base_reproduction",
    "capture_timesteps": list(CAPTURE_TIMESTEPS),
    "step_radii": list(STEP_RADII),
    "object_tasks": [
        "bed_watch",
        "normal_chair_watch",
        "high_chair_watch",
        "bed_write",
        "normal_chair_write",
        "high_chair_write",
    ],
    "direction": "minimum_norm_common_descent_of_six_normalized_object_tasks",
    "selection": {
        "each_object_each_prompt_recall_tolerance": 0.005,
        "each_object_each_prompt_mae_tolerance": 0.005,
        "each_object_hotspot_centroid_addition_cap_m": 0.10,
        "high_chair_pooled_recall_strictly_improves": True,
        "high_chair_pooled_mae_strictly_improves": True,
        "pooled_worst_object_recall_strictly_improves": True,
        "pooled_macro_mae_strictly_improves": True,
        "prompt_invariance_relative_cap": 1.05,
        "explicit_negative_mean_addition_cap": 0.005,
        "explicit_negative_max_addition_cap": 0.02,
        "v5_fixed_probe_relative_cap": 1.01,
        "minimum_directional_derivative": 1e-8,
        "map_change_epsilon": 1e-8,
    },
    "topk_role": "diagnostic_only",
    "selection_order": (
        "maximize_high_chair_recall_gain_then_maximize_worst_object_recall_gain_"
        "then_minimize_macro_mae_then_smaller_radius_then_later_capture"
    ),
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


def pooled(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    if len(rows) != 2:
        raise ValueError("response panel must contain watch/write")
    result: Dict[str, Dict[str, float]] = {}
    for name in OBJECTS:
        result[name] = {
            key: sum(_finite(row["instances"][name][key], key) for row in rows) / 2.0
            for key in (
                "soft_recall",
                "active_support_mae",
                "hotspot_centroid_distance_xy",
                "topk_overlap",
            )
        }
    return result


def response_checks(
    *,
    base_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    base_prompt_invariance: float,
    candidate_prompt_invariance: float,
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    directional_derivatives: Sequence[float],
    maximum_map_delta: float,
) -> Dict[str, bool]:
    if len(base_rows) != 2 or len(candidate_rows) != 2:
        raise ValueError("watch/write response inventory changed")
    limits = POLICY["selection"]
    checks: Dict[str, bool] = {}
    for prompt_index, prompt in enumerate(("watch", "write")):
        for name in OBJECTS:
            base = base_rows[prompt_index]["instances"][name]
            candidate = candidate_rows[prompt_index]["instances"][name]
            key = prompt + "_" + name
            checks[key + "_recall_retained"] = _finite(
                candidate["soft_recall"], key + " recall"
            ) >= _finite(base["soft_recall"], key + " base recall") - float(
                limits["each_object_each_prompt_recall_tolerance"]
            )
            checks[key + "_mae_retained"] = _finite(
                candidate["active_support_mae"], key + " MAE"
            ) <= _finite(base["active_support_mae"], key + " base MAE") + float(
                limits["each_object_each_prompt_mae_tolerance"]
            )
            checks[key + "_centroid_retained"] = _finite(
                candidate["hotspot_centroid_distance_xy"], key + " centroid"
            ) <= _finite(
                base["hotspot_centroid_distance_xy"], key + " base centroid"
            ) + float(limits["each_object_hotspot_centroid_addition_cap_m"])
            topk = _finite(candidate["topk_overlap"], key + " top-k")
            if not 0.0 <= topk <= 1.0:
                raise ValueError("top-k diagnostic is outside [0,1]")

    base_pooled = pooled(base_rows)
    candidate_pooled = pooled(candidate_rows)
    epsilon = float(limits["map_change_epsilon"])
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
    checks["prompt_invariance_retained_5pct"] = _finite(
        candidate_prompt_invariance, "candidate prompt invariance"
    ) <= _finite(base_prompt_invariance, "base prompt invariance") * float(
        limits["prompt_invariance_relative_cap"]
    ) + epsilon
    checks["explicit_negative_mean_retained"] = max(
        _finite(row["explicit_negative_mean"], "candidate negative mean")
        for row in candidate_rows
    ) <= max(
        _finite(row["explicit_negative_mean"], "base negative mean")
        for row in base_rows
    ) + float(limits["explicit_negative_mean_addition_cap"])
    checks["explicit_negative_max_retained"] = max(
        _finite(row["explicit_negative_max"], "candidate negative max")
        for row in candidate_rows
    ) <= max(
        _finite(row["explicit_negative_max"], "base negative max")
        for row in base_rows
    ) + float(limits["explicit_negative_max_addition_cap"])
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 probe inventory changed")
    checks["v5_fixed_probe_retained_1pct"] = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) <= sum(_finite(value, "base v5") for value in base_v5_dense) * float(
        limits["v5_fixed_probe_relative_cap"]
    ) + epsilon
    if len(directional_derivatives) != 6:
        raise ValueError("six-task directional derivative inventory changed")
    checks["six_task_common_descent"] = min(
        _finite(value, "directional derivative") for value in directional_derivatives
    ) >= float(limits["minimum_directional_derivative"])
    checks["final_map_changed"] = _finite(
        maximum_map_delta, "maximum map delta"
    ) > epsilon
    return checks


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, int]:
        base = row["base_pooled"]
        candidate = row["candidate_pooled"]
        high_gain = _finite(candidate["chair_06"]["soft_recall"], "high recall") - _finite(
            base["chair_06"]["soft_recall"], "base high recall"
        )
        worst_gain = min(
            _finite(candidate[name]["soft_recall"], "recall") for name in OBJECTS
        ) - min(_finite(base[name]["soft_recall"], "base recall") for name in OBJECTS)
        macro_mae = sum(
            _finite(candidate[name]["active_support_mae"], "MAE") for name in OBJECTS
        ) / 3.0
        return (
            -high_gain,
            -worst_gain,
            macro_mae,
            _finite(row["radius"], "radius"),
            -int(row["capture_timestep"]),
        )

    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


def design_seeds() -> tuple[int, int]:
    digest = hashlib.sha256(f"{SEED}|v98|design".encode("utf-8")).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


__all__ = [
    "CAPTURE_TIMESTEPS",
    "DEVELOPMENT_SCENE",
    "FAILED_V97_SCHEMA",
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
    "STEP_RADII",
    "TRAIN_SCENE",
    "canonical_sha256",
    "design_seeds",
    "pooled",
    "rank_candidates",
    "response_checks",
]

#!/usr/bin/env python3
"""Pure contract for Teacher-v10.3 two-scene on-policy response gate."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Mapping, Sequence

from relational_teacher_v102_dense_instance_contract import (
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    EXPECTED_INSTANCES,
    LORA_ALPHA,
    LORA_RANK,
    PROMPT_IDS,
    ROLES,
    SCENES,
    SOURCE_SCENE,
    V5_RELATIVE_CAP,
    canonical_sha256,
)


SCHEMA = "relational_teacher_v103_onpolicy_response_v1"
V102_SCHEMA = "relational_teacher_v102_dense_instance_supervision_v1"
MODEL_SEED = 20261101
CAPTURE_TIMESTEPS = (350, 150, 50)
STEP_RADII = (0.0005, 0.001, 0.002, 0.004)
FW_ITERATIONS = 8192
SUPPORT_TASK_WEIGHT = 2.0
ACTIVE_TASK_WEIGHT = 6.0
HARD_NEGATIVE_POINTS = 512


def _task_order() -> tuple[str, ...]:
    result = []
    for scene in SCENES:
        for prompt in PROMPT_IDS:
            for role in ROLES:
                result.append(scene + "|" + prompt + "|" + role)
            result.append(scene + "|" + prompt + "|negative")
        result.append(scene + "|prompt_invariance")
    result.extend(("v5|chair", "v5|bed", "v5|whiteboard"))
    return tuple(result)


TASK_ORDER = _task_order()
if len(TASK_ORDER) != 21:
    raise AssertionError("Teacher-v10.3 task inventory changed")

POLICY = {
    "schema": "relational_teacher_v103_onpolicy_response_policy_v1",
    "authority": "sealed_v102_actual_k3_failure",
    "initialization": "sealed_v5r4_plus_fresh_zero_output_rank16_lora",
    "decision_unit": "resumed_actual_reverse_diffusion_final_map",
    "scenes": list(SCENES),
    "prompts": list(PROMPT_IDS),
    "design_seed_domain": "new_v103_design_trajectory",
    "audit_seed_domain": "new_disjoint_v103_audit_trajectory",
    "capture_timesteps": list(CAPTURE_TIMESTEPS),
    "step_radii": list(STEP_RADII),
    "task_order": list(TASK_ORDER),
    "task_inventory": {
        "scene_prompt_role_dense_support": 12,
        "scene_prompt_negative": 4,
        "scene_prompt_invariance": 2,
        "v5_fixed_probe": 3,
        "total": 21,
    },
    "dense_instance_task": {
        "support_weight": SUPPORT_TASK_WEIGHT,
        "active_weight": ACTIVE_TASK_WEIGHT,
        "positive_support_hard_negative_exclusion": "all_three_instance_union",
    },
    "direction": "minimum_norm_common_descent_of_21_unit_task_gradients",
    "selection": {
        "each_scene_prompt_object_recall_tolerance": 0.005,
        "each_scene_prompt_object_mae_tolerance": 0.005,
        "each_scene_prompt_worst_recall_strictly_improves": True,
        "each_scene_prompt_macro_mae_strictly_improves": True,
        "global_worst_recall_strictly_improves": True,
        "global_macro_mae_strictly_improves": True,
        "prompt_invariance_relative_cap": 1.05,
        "explicit_negative_mean_addition_cap": 0.005,
        "explicit_negative_max_addition_cap": 0.02,
        "v5_fixed_probe_relative_cap": 1.01,
        "minimum_directional_derivative": 1e-8,
        "map_change_epsilon": 1e-8,
    },
    "absolute_all_three": "diagnostic_only_during_response_gate",
    "checkpoint": "forbidden",
    "optimizer": "forbidden",
    "pass_authority": "fresh_two_scene_on_policy_multiupdate_calibration_only",
    "fail_authority": "lora_capacity_or_placement_diagnosis_only",
    "development_scene_arrays": False,
    "paper_test": False,
}
POLICY_ID = canonical_sha256(POLICY)


def stable_seed_pair(domain: str, scene: str, prompt_id: str) -> tuple[int, int]:
    if domain not in ("design", "audit"):
        raise ValueError("Teacher-v10.3 seed domain changed")
    if scene not in SCENES or prompt_id not in PROMPT_IDS:
        raise ValueError("Teacher-v10.3 seed key changed")
    digest = hashlib.sha256(
        "{}|{}|{}|{}".format(MODEL_SEED, domain, scene, prompt_id).encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _role_values(
    rows: Mapping[str, Mapping[str, Mapping[str, object]]],
    scene: str,
    prompt_id: str,
    role: str,
) -> Mapping[str, object]:
    name = EXPECTED_INSTANCES[scene][ROLES.index(role)]
    return rows[scene][prompt_id]["instances"][name]


def pooled_roles(
    rows: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for role in ROLES:
        values = [
            _role_values(rows, scene, prompt_id, role)
            for scene in SCENES
            for prompt_id in PROMPT_IDS
        ]
        result[role] = {
            key: sum(_finite(value[key], role + " " + key) for value in values)
            / len(values)
            for key in (
                "soft_recall",
                "active_support_mae",
                "topk_overlap",
                "hotspot_centroid_distance_xy",
            )
        }
    return result


def response_checks(
    *,
    base_rows: Mapping[str, Mapping[str, Mapping[str, object]]],
    candidate_rows: Mapping[str, Mapping[str, Mapping[str, object]]],
    base_prompt_invariance: Mapping[str, float],
    candidate_prompt_invariance: Mapping[str, float],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    directional_derivatives: Sequence[float],
    maximum_map_delta: float,
) -> Dict[str, bool]:
    limits = POLICY["selection"]
    epsilon = float(limits["map_change_epsilon"])
    checks: Dict[str, bool] = {}
    base_all = []
    candidate_all = []
    for scene in SCENES:
        for prompt_id in PROMPT_IDS:
            base_group = []
            candidate_group = []
            for role in ROLES:
                base = _role_values(base_rows, scene, prompt_id, role)
                candidate = _role_values(candidate_rows, scene, prompt_id, role)
                key = scene + "_" + prompt_id + "_" + role
                base_recall = _finite(base["soft_recall"], key + " base recall")
                candidate_recall = _finite(
                    candidate["soft_recall"], key + " candidate recall"
                )
                base_mae = _finite(base["active_support_mae"], key + " base MAE")
                candidate_mae = _finite(
                    candidate["active_support_mae"], key + " candidate MAE"
                )
                checks[key + "_recall_retained"] = candidate_recall >= (
                    base_recall
                    - float(limits["each_scene_prompt_object_recall_tolerance"])
                )
                checks[key + "_mae_retained"] = candidate_mae <= (
                    base_mae
                    + float(limits["each_scene_prompt_object_mae_tolerance"])
                )
                base_group.append((base_recall, base_mae))
                candidate_group.append((candidate_recall, candidate_mae))
                base_all.append((base_recall, base_mae))
                candidate_all.append((candidate_recall, candidate_mae))
            checks[scene + "_" + prompt_id + "_worst_recall_improves"] = min(
                value[0] for value in candidate_group
            ) > min(value[0] for value in base_group) + epsilon
            checks[scene + "_" + prompt_id + "_macro_mae_improves"] = sum(
                value[1] for value in candidate_group
            ) < sum(value[1] for value in base_group) - epsilon
        checks[scene + "_prompt_invariance_retained"] = _finite(
            candidate_prompt_invariance[scene], scene + " candidate invariance"
        ) <= _finite(
            base_prompt_invariance[scene], scene + " base invariance"
        ) * float(limits["prompt_invariance_relative_cap"]) + epsilon
        checks[scene + "_negative_mean_retained"] = max(
            _finite(candidate_rows[scene][prompt]["explicit_negative_mean"], "negative mean")
            for prompt in PROMPT_IDS
        ) <= max(
            _finite(base_rows[scene][prompt]["explicit_negative_mean"], "base negative mean")
            for prompt in PROMPT_IDS
        ) + float(limits["explicit_negative_mean_addition_cap"])
        checks[scene + "_negative_max_retained"] = max(
            _finite(candidate_rows[scene][prompt]["explicit_negative_max"], "negative max")
            for prompt in PROMPT_IDS
        ) <= max(
            _finite(base_rows[scene][prompt]["explicit_negative_max"], "base negative max")
            for prompt in PROMPT_IDS
        ) + float(limits["explicit_negative_max_addition_cap"])
    checks["global_worst_recall_improves"] = min(
        value[0] for value in candidate_all
    ) > min(value[0] for value in base_all) + epsilon
    checks["global_macro_mae_improves"] = sum(
        value[1] for value in candidate_all
    ) < sum(value[1] for value in base_all) - epsilon
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("Teacher-v10.3 v5 probe inventory changed")
    checks["v5_fixed_probe_retained_1pct"] = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) <= sum(_finite(value, "base v5") for value in base_v5_dense) * float(
        limits["v5_fixed_probe_relative_cap"]
    ) + epsilon
    if len(directional_derivatives) != len(TASK_ORDER):
        raise ValueError("Teacher-v10.3 directional derivative inventory changed")
    checks["all_21_tasks_are_common_descent"] = min(
        _finite(value, "directional derivative")
        for value in directional_derivatives
    ) >= float(limits["minimum_directional_derivative"])
    checks["resumed_final_map_changed"] = _finite(
        maximum_map_delta, "maximum map delta"
    ) > epsilon
    return checks


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple:
        base_rows = row["base_rows"]
        candidate_rows = row["candidate_rows"]
        worst_gain = min(
            _role_values(candidate_rows, scene, prompt, role)["soft_recall"]
            - _role_values(base_rows, scene, prompt, role)["soft_recall"]
            for scene in SCENES
            for prompt in PROMPT_IDS
            for role in ROLES
        )
        macro_mae = sum(
            _role_values(candidate_rows, scene, prompt, role)["active_support_mae"]
            for scene in SCENES
            for prompt in PROMPT_IDS
            for role in ROLES
        )
        v5_ratio = sum(row["candidate_v5_dense"]) / max(
            sum(row["base_v5_dense"]), 1e-12
        )
        return (
            -float(worst_gain),
            float(macro_mae),
            float(v5_ratio),
            float(row["radius"]),
            -int(row["capture_timestep"]),
        )

    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_sha256",
    "pooled_roles",
    "rank_candidates",
    "response_checks",
    "stable_seed_pair",
]

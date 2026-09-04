#!/usr/bin/env python3
"""Pure contract for Teacher-v10.3.1 on-policy K=3 calibration."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Mapping, Sequence

from relational_teacher_v103_onpolicy_response_contract import (
    EXPECTED_INSTANCES,
    PROMPT_IDS,
    ROLES,
    SCENES,
    TASK_ORDER,
    canonical_sha256,
    response_checks,
)


SCHEMA = "relational_teacher_v1031_onpolicy_calibration6_v1"
V103_SCHEMA = "relational_teacher_v103_onpolicy_response_v1"
MODEL_SEED = 20261102
SELECTED_TIMESTEP = 150
SELECTED_RADIUS = 0.004
GENERATION_COUNT = 3
UPDATE_COUNT = 6
MONITOR_STEPS = (1, 2, 3, 4, 5, 6)
STEP_RADIUS = 0.001
FW_ITERATIONS = 8192
PER_GENERATION_RECALL_TOLERANCE = 0.01
PER_GENERATION_MAE_TOLERANCE = 0.01
NEGATIVE_MEAN_ADDITION_CAP = 0.005
NEGATIVE_MAX_ADDITION_CAP = 0.02
STATE_CHANGE_EPSILON = 1e-8
SHORTLIST_LIMIT = 2


def stable_seed_pair(
    domain: str, scene: str, prompt_id: str, generation: int
) -> tuple[int, int]:
    if domain not in ("design", "audit"):
        raise ValueError("Teacher-v10.3.1 seed domain changed")
    if scene not in SCENES or prompt_id not in PROMPT_IDS:
        raise ValueError("Teacher-v10.3.1 seed key changed")
    if generation < 0 or generation >= GENERATION_COUNT:
        raise ValueError("Teacher-v10.3.1 generation changed")
    digest = hashlib.sha256(
        "{}|v1031|{}|{}|{}|{}".format(
            MODEL_SEED, domain, scene, prompt_id, generation
        ).encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


POLICY = {
    "schema": "relational_teacher_v1031_onpolicy_calibration6_policy_v1",
    "authority": "sealed_v103_selected_t150_radius_0p004_response_pass",
    "initialization": "fresh_v5r4_plus_exact_reconstructed_v103_selected_lora",
    "decision_unit": "disjoint_actual_k3_resumed_final_maps_after_each_update",
    "scenes": list(SCENES),
    "prompts": list(PROMPT_IDS),
    "selected_timestep": SELECTED_TIMESTEP,
    "selected_radius": SELECTED_RADIUS,
    "generation_count": GENERATION_COUNT,
    "update_count": UPDATE_COUNT,
    "monitor_steps": list(MONITOR_STEPS),
    "step_radius": STEP_RADIUS,
    "task_order": list(TASK_ORDER),
    "gradient_design": "K3_mean_per_locked_21_tasks",
    "direction": "minimum_norm_common_descent_recomputed_at_every_update",
    "audit": "fresh_disjoint_K3_for_both_scenes_and_both_prompts",
    "per_generation_guards": {
        "recall_drop_tolerance": PER_GENERATION_RECALL_TOLERANCE,
        "mae_increase_tolerance": PER_GENERATION_MAE_TOLERANCE,
        "negative_mean_addition_cap": NEGATIVE_MEAN_ADDITION_CAP,
        "negative_max_addition_cap": NEGATIVE_MAX_ADDITION_CAP,
    },
    "absolute_all_three": "diagnostic_only_during_short_calibration",
    "shortlist_limit": SHORTLIST_LIMIT,
    "checkpoint": "forbidden_until_separate_full_schedule_export_gate",
    "optimizer": "forbidden_trust_region_updates_only",
    "pass_authority": "extended_onpolicy_training_with_actual_k3_monitor_only",
    "fail_authority": "lora_capacity_or_placement_diagnosis_only",
    "development_scene_arrays": False,
    "paper_test": False,
}
POLICY_ID = canonical_sha256(POLICY)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _role_value(
    rows: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
    scene: str,
    generation: int,
    prompt_index: int,
    role: str,
) -> Mapping[str, object]:
    name = EXPECTED_INSTANCES[scene][ROLES.index(role)]
    return rows[scene][generation][prompt_index]["instances"][name]


def all_three_counts(
    presence: Mapping[str, Sequence[Sequence[Mapping[str, bool]]]],
) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for scene in SCENES:
        scene_rows = presence.get(scene)
        if not isinstance(scene_rows, Sequence) or len(scene_rows) != GENERATION_COUNT:
            raise ValueError(scene + " K=3 presence inventory changed")
        result[scene] = {}
        for prompt_index, prompt_id in enumerate(PROMPT_IDS):
            result[scene][prompt_id] = sum(
                all(value is True for value in generation[prompt_index].values())
                for generation in scene_rows
            )
    return result


def calibration_checks(
    *,
    start_rows: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
    candidate_rows: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
    pooled_response_checks: Mapping[str, bool],
    pre_state_sha256: str,
    post_state_sha256: str,
) -> Dict[str, bool]:
    checks = {
        "pooled_response_is_admissible": bool(pooled_response_checks)
        and all(value is True for value in pooled_response_checks.values()),
        "lora_state_changes": isinstance(pre_state_sha256, str)
        and isinstance(post_state_sha256, str)
        and len(pre_state_sha256) == 64
        and len(post_state_sha256) == 64
        and pre_state_sha256 != post_state_sha256,
    }
    for scene in SCENES:
        for generation in range(GENERATION_COUNT):
            for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                prefix = "{}_g{}_{}".format(scene, generation, prompt_id)
                for role in ROLES:
                    base = _role_value(
                        start_rows, scene, generation, prompt_index, role
                    )
                    candidate = _role_value(
                        candidate_rows, scene, generation, prompt_index, role
                    )
                    checks[prefix + "_" + role + "_recall_retained"] = _finite(
                        candidate["soft_recall"], "candidate recall"
                    ) >= _finite(base["soft_recall"], "start recall") - float(
                        PER_GENERATION_RECALL_TOLERANCE
                    )
                    checks[prefix + "_" + role + "_mae_retained"] = _finite(
                        candidate["active_support_mae"], "candidate MAE"
                    ) <= _finite(base["active_support_mae"], "start MAE") + float(
                        PER_GENERATION_MAE_TOLERANCE
                    )
                start_metric = start_rows[scene][generation][prompt_index]
                current_metric = candidate_rows[scene][generation][prompt_index]
                checks[prefix + "_negative_mean_retained"] = _finite(
                    current_metric["explicit_negative_mean"], "candidate negative mean"
                ) <= _finite(
                    start_metric["explicit_negative_mean"], "start negative mean"
                ) + NEGATIVE_MEAN_ADDITION_CAP
                checks[prefix + "_negative_max_retained"] = _finite(
                    current_metric["explicit_negative_max"], "candidate negative max"
                ) <= _finite(
                    start_metric["explicit_negative_max"], "start negative max"
                ) + NEGATIVE_MAX_ADDITION_CAP
    return checks


def rank_shortlist(rows: Sequence[Mapping[str, object]]) -> list[int]:
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple:
        base = row["start_pooled"]
        candidate = row["candidate_pooled"]
        worst_gain = min(
            _finite(
                candidate[scene][prompt]["instances"][EXPECTED_INSTANCES[scene][index]][
                    "soft_recall"
                ],
                "candidate recall",
            )
            - _finite(
                base[scene][prompt]["instances"][EXPECTED_INSTANCES[scene][index]][
                    "soft_recall"
                ],
                "start recall",
            )
            for scene in SCENES
            for prompt in PROMPT_IDS
            for index in range(3)
        )
        macro_mae = sum(
            _finite(
                candidate[scene][prompt]["instances"][EXPECTED_INSTANCES[scene][index]][
                    "active_support_mae"
                ],
                "candidate MAE",
            )
            for scene in SCENES
            for prompt in PROMPT_IDS
            for index in range(3)
        )
        v5_ratio = sum(row["candidate_v5_dense"]) / max(
            sum(row["start_v5_dense"]), 1e-12
        )
        return (-worst_gain, macro_mae, v5_ratio, int(row["step"]))

    eligible.sort(key=key)
    return [int(row["step"]) for row in eligible[:SHORTLIST_LIMIT]]


__all__ = [name for name in globals() if name.isupper()] + [
    "all_three_counts",
    "calibration_checks",
    "canonical_sha256",
    "rank_shortlist",
    "response_checks",
    "stable_seed_pair",
]

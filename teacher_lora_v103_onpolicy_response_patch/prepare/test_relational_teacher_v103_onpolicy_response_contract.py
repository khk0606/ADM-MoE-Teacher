#!/usr/bin/env python3
"""CPU/static contract for Teacher-v10.3 on-policy response gate."""

from __future__ import annotations

import ast
from pathlib import Path

from relational_teacher_v103_onpolicy_response_contract import (
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    ROLES,
    SCENES,
    TASK_ORDER,
    canonical_sha256,
    rank_candidates,
    response_checks,
    stable_seed_pair,
)


def _metric(recall: float, mae: float) -> dict:
    return {
        "soft_recall": recall,
        "active_support_mae": mae,
        "topk_overlap": 0.4,
        "hotspot_centroid_distance_xy": 0.1,
    }


def _rows(recall: float, mae: float) -> dict:
    result = {}
    names = {
        "room_0101": ("bed_01", "chair_01", "chair_06"),
        "room_0102": ("bed_01", "chair_05", "chair_06"),
    }
    for scene in SCENES:
        result[scene] = {}
        for prompt in PROMPT_IDS:
            result[scene][prompt] = {
                "instances": {
                    name: _metric(recall + 0.001 * index, mae - 0.001 * index)
                    for index, name in enumerate(names[scene])
                },
                "explicit_negative_mean": 0.02,
                "explicit_negative_max": 0.2,
            }
    return result


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID or len(TASK_ORDER) != 21:
        raise AssertionError("Teacher-v10.3 policy/task inventory changed")
    design = [
        stable_seed_pair("design", scene, prompt)
        for scene in SCENES
        for prompt in PROMPT_IDS
    ]
    audit = [
        stable_seed_pair("audit", scene, prompt)
        for scene in SCENES
        for prompt in PROMPT_IDS
    ]
    if len(set(design + audit)) != 8:
        raise AssertionError("Teacher-v10.3 seed domains overlap")

    base = _rows(0.40, 0.30)
    candidate = _rows(0.42, 0.28)
    checks = response_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_prompt_invariance={scene: 0.01 for scene in SCENES},
        candidate_prompt_invariance={scene: 0.009 for scene in SCENES},
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.0601, 0.0601, 0.0601],
        directional_derivatives=[0.1] * len(TASK_ORDER),
        maximum_map_delta=0.01,
    )
    if not all(checks.values()):
        raise AssertionError("valid Teacher-v10.3 response was rejected")
    broken = _rows(0.42, 0.28)
    broken["room_0102"]["sit_write_v1"]["instances"]["chair_05"][
        "soft_recall"
    ] = 0.30
    rejected = response_checks(
        base_rows=base,
        candidate_rows=broken,
        base_prompt_invariance={scene: 0.01 for scene in SCENES},
        candidate_prompt_invariance={scene: 0.009 for scene in SCENES},
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.0601, 0.0601, 0.0601],
        directional_derivatives=[0.1] * len(TASK_ORDER),
        maximum_map_delta=0.01,
    )
    if rejected["room_0102_sit_write_v1_normal_chair_recall_retained"]:
        raise AssertionError("missing room_0102 Normal Chair was accepted")
    drifted = response_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_prompt_invariance={scene: 0.01 for scene in SCENES},
        candidate_prompt_invariance={scene: 0.009 for scene in SCENES},
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.07, 0.07, 0.07],
        directional_derivatives=[0.1] * len(TASK_ORDER),
        maximum_map_delta=0.01,
    )
    if drifted["v5_fixed_probe_retained_1pct"]:
        raise AssertionError("Teacher-v10.3 accepted destructive v5 drift")
    conflicted = response_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_prompt_invariance={scene: 0.01 for scene in SCENES},
        candidate_prompt_invariance={scene: 0.009 for scene in SCENES},
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.0601, 0.0601, 0.0601],
        directional_derivatives=[0.1] * (len(TASK_ORDER) - 1) + [-0.01],
        maximum_map_delta=0.01,
    )
    if conflicted["all_21_tasks_are_common_descent"]:
        raise AssertionError("Teacher-v10.3 accepted a conflicting task direction")
    ranked = rank_candidates(
        [
            {
                "name": "worse",
                "eligible": True,
                "base_rows": base,
                "candidate_rows": candidate,
                "base_v5_dense": [0.06] * 3,
                "candidate_v5_dense": [0.0601] * 3,
                "radius": 0.002,
                "capture_timestep": 150,
            },
            {
                "name": "better",
                "eligible": True,
                "base_rows": base,
                "candidate_rows": _rows(0.43, 0.27),
                "base_v5_dense": [0.06] * 3,
                "candidate_v5_dense": [0.0601] * 3,
                "radius": 0.001,
                "capture_timestep": 350,
            },
        ]
    )
    if ranked != ["better", "worse"]:
        raise AssertionError("Teacher-v10.3 response ranking changed")

    prepare = Path(__file__).resolve().parent
    runner = (
        prepare / "preflight_relational_teacher_v103_onpolicy_response.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(runner)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizer_calls = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("Adam", "AdamW", "SGD")
        )
    ]
    checkpoint_calls = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("save_merged_legacy_state", "torch_save")
        )
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("save", "save_checkpoint")
        )
    ]
    if optimizer_calls or checkpoint_calls:
        raise AssertionError("optimizer/checkpoint entered Teacher-v10.3 response gate")
    for literal in (
        "dense_instance_all_sittable_objective(",
        "_capture_trajectory(",
        "_resume_trajectory(",
        "flattened_task_gradients(task_losses, parameters)",
        "frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)",
        "apply_flat_direction(parameters, direction, radius)",
        "if tuple(observed_order) != TASK_ORDER:",
        '"candidate_decision_uses_resumed_final_maps": True',
        '"absolute_all_three_is_diagnostic_only": True',
        '"no_optimizer_created": True',
        '"no_model_checkpoint_saved": True',
        '"room_0201_arrays_unread": True',
    ):
        if literal not in runner:
            raise AssertionError("Teacher-v10.3 runner guard changed: " + literal)

    print("[PASS] Teacher-v10.3 on-policy response CPU/static contract")
    print("[PASS] 21-task common descent and disjoint trajectory policy")
    print("[PASS] Normal-Chair regression, v5 drift and checkpoint fail closed")


if __name__ == "__main__":
    main()

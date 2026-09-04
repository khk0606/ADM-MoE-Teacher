#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v9.8.8 rollout-aligned diagnosis."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from relational_teacher_v988_rollout_aligned_direction_contract import (
    AUDIT_BED_TASKS,
    CANDIDATE_NAMES,
    POLICY,
    POLICY_ID,
    TASK_ORDER,
    canonical_sha256,
    conflict_pairs,
    diagnose_directions,
    rank_eligible_directions,
)


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("rollout-aligned policy ID changed")
    if len(TASK_ORDER) != 51 or len(set(TASK_ORDER)) != 51:
        raise AssertionError("51-task inventory changed")
    if len(AUDIT_BED_TASKS) != 6 or not set(AUDIT_BED_TASKS).issubset(TASK_ORDER):
        raise AssertionError("six audit-Bed tasks changed")
    if len(CANDIDATE_NAMES) != 7 or len(set(CANDIDATE_NAMES)) != 7:
        raise AssertionError("direction candidate inventory changed")

    identity = np.eye(len(TASK_ORDER), dtype=np.float64)
    rows = diagnose_directions(identity)
    if [row["name"] for row in rows] != list(CANDIDATE_NAMES):
        raise AssertionError("direction order changed")
    if rank_eligible_directions(rows)[0] != "audit_bed_guard_0p40":
        raise AssertionError("audit-Bed-aware ranking changed")
    if rows[-1]["eligible"] is not False:
        raise AssertionError("audit-Bed-only regression was accepted")
    if conflict_pairs(identity) != 0:
        raise AssertionError("conflict counter changed")
    invalid = identity.copy()
    invalid[0, 1] = 0.2
    try:
        diagnose_directions(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("asymmetric Gram matrix was accepted")

    runner = Path(__file__).with_name(
        "preflight_relational_teacher_v988_rollout_aligned_direction.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
    ]
    applies = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_flat_direction"
    ]
    policy_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "create_model_and_diffusion"
    ]
    if len(loads) != 2 or len(applies) != 1:
        raise AssertionError("two-scene/reconstruction-only mutation contract changed")
    if (
        not policy_calls
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("diagnosis may not create optimizer/checkpoint state")
    for literal in (
        'value.get("status") != "FAIL"',
        'row.get("scene_failed_checks", {}).get(SOURCE_SCENE) != []',
        'row.get("scene_failed_checks", {}).get(AUDIT_SCENE) != expected_audit',
        'np.array_equal(base_normalized, v987_arrays["base_normalized"])',
        'for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):',
        'for generation in range(3):',
        'flattened_task_gradients(tasks, parameters)',
        'if tuple(actual_task_order) != TASK_ORDER:',
        '"no_candidate_parameter_update_applied": True',
        '"authorizes_actual_two_scene_k3_rollout_aligned_radius_grid"',
    ):
        if literal not in source:
            raise AssertionError("rollout-aligned runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.8 rollout-aligned direction CPU contract")
    print("[PASS] 51-task geometry, Bed guards and fail-closed ranking")
    print("[PASS] exact K=3 state reproduction, two-scene and no-checkpoint guards")


if __name__ == "__main__":
    main()

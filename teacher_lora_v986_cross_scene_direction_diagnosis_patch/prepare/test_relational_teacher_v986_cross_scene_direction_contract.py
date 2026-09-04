#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v9.8.6 cross-scene diagnosis."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from relational_teacher_v986_cross_scene_direction_contract import (
    AUDIT_BED_TASKS,
    AUDIT_SCENE,
    CANDIDATE_NAMES,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    RAW_GRAM_ASYMMETRY_CAP,
    SOURCE_SCENE,
    TASK_ORDER,
    canonical_sha256,
    conflict_pairs,
    cross_scene_design_seeds,
    diagnose_directions,
    rank_eligible_directions,
)


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("cross-scene policy ID changed")
    if MODEL_SEED != 20261016 or len(TASK_ORDER) != 19 or RAW_GRAM_ASYMMETRY_CAP != 1e-8:
        raise AssertionError("cross-scene protocol changed")
    if TASK_ORDER[8] != AUDIT_BED_TASKS[0] or TASK_ORDER[11] != AUDIT_BED_TASKS[1]:
        raise AssertionError("room_0102 Bed task indices changed")
    if cross_scene_design_seeds(SOURCE_SCENE) == cross_scene_design_seeds(AUDIT_SCENE):
        raise AssertionError("two scenes reuse diagnosis seeds")
    try:
        cross_scene_design_seeds("room_0201")
    except ValueError:
        pass
    else:
        raise AssertionError("development scene received an authorized seed")

    identity = np.eye(len(TASK_ORDER), dtype=np.float64)
    rows = diagnose_directions(identity)
    if [row["name"] for row in rows] != list(CANDIDATE_NAMES):
        raise AssertionError("direction candidate order changed")
    if rows[-1]["eligible"] is not False or rows[-1]["failed_checks"] != [
        "every_task_is_common_descent"
    ]:
        raise AssertionError("Bed-only direction did not fail non-Bed preservation")
    ranking = rank_eligible_directions(rows)
    if not ranking or ranking[0] != "bed_guard_0p50":
        raise AssertionError("strict eligible direction ranking changed")

    conflicting = identity.copy()
    conflicting[0, 1] = conflicting[1, 0] = -0.4
    pairs = conflict_pairs(conflicting)
    if pairs != [[TASK_ORDER[0], TASK_ORDER[1], -0.4]]:
        raise AssertionError("conflict-pair diagnosis changed")
    malformed = identity.copy()
    malformed[0, 0] = 0.5
    try:
        diagnose_directions(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("non-normalized task gradients were accepted")

    runner = Path(__file__).with_name(
        "preflight_relational_teacher_v986_cross_scene_direction.py"
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
        raise AssertionError("exactly two scenes and reconstruction-only update are required")
    if (
        len(policy_calls) < 1
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("diagnosis may not create optimizer/checkpoint state")
    for literal in (
        'value.get("audit_failed_checks") != expected_audit_failures',
        "for step in RECONSTRUCTION_STEPS:",
        "apply_flat_direction(parameters, direction, STEP_RADIUS)",
        "source_gradients, source_losses = _scene_task_gradients(",
        "audit_gradients, audit_losses = _scene_task_gradients(",
        "v5_gradients = flattened_task_gradients(v5_tasks, parameters)",
        "gradients.shape[0] != len(TASK_ORDER)",
        "gradient64 = gradients.double()",
        "raw_gram_max_asymmetry = float(np.abs(raw_gram - raw_gram.T).max())",
        "gram = 0.5 * (raw_gram + raw_gram.T)",
        '"no_candidate_parameter_update_applied": True',
        '"authorizes_actual_two_scene_k3_direction_response_grid": status == "PASS"',
    ):
        if literal not in source:
            raise AssertionError("cross-scene runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.6 cross-scene direction CPU contract")
    print("[PASS] Bed-only direction fails while strict 19-task directions rank deterministically")
    print("[PASS] two-scene, reconstruction-only and no-checkpoint static guards")


if __name__ == "__main__":
    main()

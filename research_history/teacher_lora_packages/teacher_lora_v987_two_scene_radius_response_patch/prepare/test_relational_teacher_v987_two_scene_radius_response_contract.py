#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v9.8.7 radius response."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v987_two_scene_radius_response_contract import (
    AUDIT_SCENE,
    POLICY,
    POLICY_ID,
    RADIUS_GRID,
    SELECTED_DIRECTION,
    SOURCE_SCENE,
    canonical_sha256,
    radius_response_checks,
    rank_eligible_radii,
)


def pooled(recall: float) -> dict:
    return {
        role: {
            "soft_recall": recall + offset,
            "active_support_mae": 0.5 - offset,
        }
        for role, offset in (("bed", 0.0), ("normal_chair", 0.1), ("high_chair", 0.05))
    }


def candidate(radius: float, bed_gain: float, eligible: bool = True) -> dict:
    base = {SOURCE_SCENE: pooled(0.2), AUDIT_SCENE: pooled(0.15)}
    after = copy.deepcopy(base)
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        after[scene]["bed"]["soft_recall"] += bed_gain
        after[scene]["normal_chair"]["soft_recall"] += bed_gain / 2.0
        after[scene]["high_chair"]["soft_recall"] += bed_gain / 3.0
    return {
        "radius": radius,
        "eligible": eligible,
        "scene_base_pooled": base,
        "scene_candidate_pooled": after,
        "candidate_v5_dense": [0.05 + radius, 0.06 + radius, 0.07 + radius],
    }


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("radius-response policy ID changed")
    if RADIUS_GRID != (0.0005, 0.001, 0.002, 0.003):
        raise AssertionError("radius grid changed")
    if SELECTED_DIRECTION != "bed_guard_0p10":
        raise AssertionError("selected v9.8.6 direction changed")

    passed = radius_response_checks(
        source_checks={"a": True},
        audit_checks={"b": True},
        selected_direction_reproduced=True,
        candidate_state_changed=True,
    )
    if not all(passed.values()):
        raise AssertionError("valid two-scene response was rejected")
    failed = radius_response_checks(
        source_checks={"a": True},
        audit_checks={"bed": False},
        selected_direction_reproduced=True,
        candidate_state_changed=True,
    )
    if failed["room0102_response_is_admissible"] is not False:
        raise AssertionError("room_0102 regression was accepted")

    rows = [
        candidate(0.0005, 0.002),
        candidate(0.001, 0.004),
        candidate(0.002, 0.003),
        candidate(0.003, 0.02, eligible=False),
    ]
    if rank_eligible_radii(rows) != [0.001, 0.002, 0.0005]:
        raise AssertionError("eligible radius ranking changed")

    runner = Path(__file__).with_name(
        "evaluate_relational_teacher_v987_two_scene_radius_response.py"
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
    if len(loads) != 2 or len(applies) != 2:
        raise AssertionError("two-scene/reconstruction-plus-radius update contract changed")
    if (
        len(policy_calls) < 1
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("response grid may not create optimizer/checkpoint state")
    for literal in (
        'value.get("selected_candidate") != SELECTED_DIRECTION',
        "for step in RECONSTRUCTION_STEPS:",
        "for radius_index, radius in enumerate(RADIUS_GRID):",
        "_restore_state(named_lora, step4_state)",
        "distinct radii produced duplicate LoRA states",
        "selected_direction_sha256 != sealed_selected[\"direction_sha256\"]",
        "two_scene_preflight_checks(",
        "radius_response_checks(",
        '"at_least_one_radius_is_admissible": selected_radius is not None',
        '"authorizes_fresh_two_scene_common_direction_calibration": status == "PASS"',
    ):
        if literal not in source:
            raise AssertionError("radius-response runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.7 two-scene radius-response CPU contract")
    print("[PASS] audit-scene regression and ineligible radius fail closed")
    print("[PASS] common-start, actual K=3 and no-checkpoint static guards")


if __name__ == "__main__":
    main()

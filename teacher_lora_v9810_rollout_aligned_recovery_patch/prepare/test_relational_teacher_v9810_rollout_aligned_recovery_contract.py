#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v9.8.10 recovery bracket."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v9810_rollout_aligned_recovery_contract import (
    AUDIT_SCENE,
    FAILED_RADIUS_GRID,
    POLICY,
    POLICY_ID,
    RADIUS_GRID,
    SELECTED_DIRECTION,
    SOURCE_SCENE,
    canonical_sha256,
    radius_response_checks,
    rank_eligible_radii,
    validate_failed_recovery_signature,
)


def pooled(recall: float) -> dict:
    return {
        role: {
            "soft_recall": recall + offset,
            "active_support_mae": 0.5 - offset,
        }
        for role, offset in (("bed", 0.0), ("normal_chair", 0.1), ("high_chair", 0.05))
    }


def candidate(radius: float, eligible: bool = True) -> dict:
    base = {SOURCE_SCENE: pooled(0.2), AUDIT_SCENE: pooled(0.15)}
    after = copy.deepcopy(base)
    return {
        "radius": radius,
        "eligible": eligible,
        "scene_base_pooled": base,
        "scene_candidate_pooled": after,
        "candidate_v5_dense": [0.05 + radius, 0.06 + radius, 0.07 + radius],
    }


def failed_v989() -> dict:
    rows = []
    for index, radius in enumerate(FAILED_RADIUS_GRID):
        row = candidate(radius, eligible=False)
        row["scene_failed_checks"] = {
            SOURCE_SCENE: [],
            AUDIT_SCENE: [
                "bed_pooled_mae_strictly_improves",
                "bed_pooled_recall_strictly_improves",
            ],
        }
        bed = row["scene_candidate_pooled"][AUDIT_SCENE]["bed"]
        bed["soft_recall"] = 0.20 + index * 0.001
        bed["active_support_mae"] = 0.60 - index * 0.001
        rows.append(row)
    return {"response_rows": rows}


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("recovery policy ID changed")
    if FAILED_RADIUS_GRID != (0.00025, 0.0005, 0.001, 0.002, 0.003):
        raise AssertionError("sealed failed radius grid changed")
    if RADIUS_GRID != (0.004, 0.005, 0.006, 0.007, 0.008):
        raise AssertionError("recovery radius bracket changed")
    if SELECTED_DIRECTION != "audit_bed_guard_0p10":
        raise AssertionError("selected rollout-aligned direction changed")
    validate_failed_recovery_signature(failed_v989())
    tampered = failed_v989()
    tampered["response_rows"][3]["scene_candidate_pooled"][AUDIT_SCENE]["bed"][
        "soft_recall"
    ] = 0.19
    try:
        validate_failed_recovery_signature(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("non-monotone v9.8.9 failure was accepted")

    passed = radius_response_checks(
        source_checks={"a": True},
        audit_checks={"b": True},
        selected_direction_reproduced=True,
        candidate_state_changed=True,
    )
    if not all(passed.values()):
        raise AssertionError("valid two-scene recovery was rejected")
    failed = radius_response_checks(
        source_checks={"a": True},
        audit_checks={"bed": False},
        selected_direction_reproduced=True,
        candidate_state_changed=True,
    )
    if failed["room0102_response_is_admissible"] is not False:
        raise AssertionError("audit Bed regression was accepted")

    rows = [
        candidate(0.004),
        candidate(0.005),
        candidate(0.006),
        candidate(0.007, eligible=False),
        candidate(0.008),
    ]
    if rank_eligible_radii(rows) != [0.004, 0.005, 0.006, 0.008]:
        raise AssertionError("smallest-admissible radius ranking changed")

    runner = Path(__file__).with_name(
        "evaluate_relational_teacher_v9810_rollout_aligned_recovery.py"
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
        raise AssertionError("recovery bracket may not create optimizer/checkpoint state")
    for literal in (
        "v989 = _validate_v989(v989_file)",
        "validate_failed_recovery_signature(value)",
        "for step in RECONSTRUCTION_STEPS:",
        "for radius_index, radius in enumerate(RADIUS_GRID):",
        "_restore_state(named_lora, step4_state)",
        "distinct radii produced duplicate LoRA states",
        'selected_direction_sha256 != sealed_selected["direction_sha256"]',
        "two_scene_preflight_checks(",
        "radius_response_checks(",
        '"at_least_one_recovery_radius_is_admissible": selected_radius is not None',
        '"authorizes_fresh_two_scene_rollout_aligned_calibration": status == "PASS"',
    ):
        if literal not in source:
            raise AssertionError("recovery runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.10 rollout-aligned recovery CPU contract")
    print("[PASS] sealed monotone Bed-only failure and smallest-pass ranking verified")
    print("[PASS] actual two-scene K=3 and no-checkpoint static guards")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v9.8.11 new-seed replication."""

from __future__ import annotations

import ast
from pathlib import Path

from relational_teacher_v9811_rollout_aligned_replication_contract import (
    GENERATION_COUNT,
    POLICY,
    POLICY_ID,
    REPLICATION_SEED_TABLE,
    SELECTED_DIRECTION,
    SELECTED_RADIUS,
    canonical_sha256,
    replication_checks,
    replication_rollout_seeds,
)


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("replication policy ID changed")
    if SELECTED_DIRECTION != "audit_bed_guard_0p10" or SELECTED_RADIUS != 0.006:
        raise AssertionError("selected v9.8.10 candidate changed")
    expected = tuple(replication_rollout_seeds(i) for i in range(GENERATION_COUNT))
    if expected != REPLICATION_SEED_TABLE or len(set(expected)) != GENERATION_COUNT:
        raise AssertionError("replication seed table changed")
    if any(len(row) != 2 or any(value < 0 for value in row) for row in expected):
        raise AssertionError("replication seed row is invalid")

    passed = replication_checks(
        source_checks={"a": True},
        audit_checks={"b": True},
        selected_state_exact=True,
        selected_v5_exact=True,
        deterministic_repeats_exact=True,
        seed_table_disjoint=True,
    )
    if not all(passed.values()):
        raise AssertionError("valid new-seed replication was rejected")
    failed = replication_checks(
        source_checks={"a": True},
        audit_checks={"bed": False},
        selected_state_exact=True,
        selected_v5_exact=True,
        deterministic_repeats_exact=True,
        seed_table_disjoint=True,
    )
    if failed["room0102_new_seed_response_is_admissible"] is not False:
        raise AssertionError("new-seed audit regression was accepted")
    failed_state = replication_checks(
        source_checks={"a": True},
        audit_checks={"b": True},
        selected_state_exact=False,
        selected_v5_exact=True,
        deterministic_repeats_exact=True,
        seed_table_disjoint=True,
    )
    if failed_state["selected_v9810_state_exactly_reproduced"] is not False:
        raise AssertionError("wrong selected state was accepted")

    runner = Path(__file__).with_name(
        "evaluate_relational_teacher_v9811_rollout_aligned_replication.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
    ]
    applies = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_flat_direction"
    ]
    policy_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "create_model_and_diffusion"
    ]
    if len(loads) != 2 or len(applies) != 2:
        raise AssertionError("two-scene/reconstruction-plus-selected-state contract changed")
    if (
        not policy_calls
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("replication may not create optimizer/checkpoint state")
    for literal in (
        "v9810 = _validate_v9810(v9810_file)",
        "for step in RECONSTRUCTION_STEPS:",
        "selection_cache",
        "replication_cache",
        "replication_rollout_seeds(generation)",
        "seed_table_disjoint",
        "selected_state_exact",
        "selected_v5_exact",
        "two_scene_preflight_checks(",
        "replication_checks(",
        '"new_seed_response_is_admissible": response_eligible',
        '"authorizes_fresh_two_scene_rollout_aligned_multiupdate_calibration"',
    ):
        if literal not in source:
            raise AssertionError("replication runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.11 disjoint-seed rollout replication CPU contract")
    print("[PASS] fixed radius 0.006 state/v5 identity and independent scene gates")
    print("[PASS] actual two-scene K=3, disjoint seeds and no-checkpoint static guards")


if __name__ == "__main__":
    main()

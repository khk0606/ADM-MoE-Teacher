#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v10 supervised training."""

from __future__ import annotations

import ast
from pathlib import Path

from relational_teacher_v10_supervised_capacity_contract import (
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    SCENES,
    canonical_sha256,
    rollout_gate,
    select_rollout_candidate,
    shortlist_monitor_steps,
)


def _pooled(recall: float, mae: float) -> dict:
    return {
        scene: {
            role: {
                "soft_recall": recall,
                "active_support_mae": mae,
                "topk_overlap": 0.8,
                "hotspot_centroid_distance_xy": 0.1,
            }
            for role in ("bed", "normal_chair", "high_chair")
        }
        for scene in SCENES
    }


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("Teacher-v10 policy ID changed")
    monitors = [
        {
            "step": step,
            "worst_instance_soft_recall": 0.2 + step / 2000.0,
            "worst_instance_active_support_mae": 0.7 - step / 2400.0,
            "known_dense_mae": 0.4 - step / 4000.0,
        }
        for step in MONITOR_STEPS
    ]
    if shortlist_monitor_steps(monitors) != [1200, 1000, 750]:
        raise AssertionError("one-step shortlist ranking changed")

    counts = {
        scene: {"watch": 2, "write": 3}
        for scene in SCENES
    }
    invariance = {scene: [0.001, 0.002, 0.003] for scene in SCENES}
    checks = rollout_gate(
        all_three_counts=counts,
        prompt_invariance=invariance,
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.061, 0.061, 0.061],
        checkpoint_state_changed=True,
    )
    if not all(checks.values()):
        raise AssertionError("valid actual K=3 panel was rejected")
    missing = {scene: dict(values) for scene, values in counts.items()}
    missing[SCENES[1]]["write"] = 1
    failed = rollout_gate(
        all_three_counts=missing,
        prompt_invariance=invariance,
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.061, 0.061, 0.061],
        checkpoint_state_changed=True,
    )
    if failed[SCENES[1] + "_write_all_three_at_least_two_of_three"] is not False:
        raise AssertionError("missing-object rollout was accepted")
    degraded = rollout_gate(
        all_three_counts=counts,
        prompt_invariance=invariance,
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.07, 0.07, 0.07],
        checkpoint_state_changed=True,
    )
    if degraded["v5_fixed_probe_retained_5pct"] is not False:
        raise AssertionError("v5 regression was accepted")

    rows = [
        {
            "step": step,
            "eligible": step in (750, 1200),
            "all_three_counts": counts,
            "pooled_metrics": _pooled(0.75 + step / 10000.0, 0.1),
        }
        for step in (750, 1000, 1200)
    ]
    if select_rollout_candidate(rows) != 1200:
        raise AssertionError("actual-rollout selection changed")

    runner = Path(__file__).with_name(
        "run_relational_teacher_v10_supervised_capacity.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizer_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "AdamW"
    ]
    loads = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
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
    checkpoint_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "save_merged_legacy_state"
    ]
    if len(optimizer_calls) != 1 or len(loads) != 1 or len(model_calls) != 1 or len(checkpoint_calls) != 1:
        raise AssertionError("Teacher-v10 optimizer/data/model/checkpoint inventory changed")
    if not policy_calls or policy_calls[0].lineno >= loads[0].lineno or policy_calls[0].lineno >= model_calls[0].lineno:
        raise AssertionError("Teacher-v10 policy is not locked before arrays/model")
    for literal in (
        "for scene_index, scene in enumerate(SCENES):",
        "prediction = predict_xstart(",
        "direct_all_sittable_objective(",
        "for step in range(1, args.steps + 1):",
        "if step >= REPLAY_START_STEP:",
        "shortlisted_steps = shortlist_monitor_steps(monitor_rows)",
        "selected_step = select_rollout_candidate(rollout_rows)",
        "if selected_step is not None:",
        "save_merged_legacy_state(model, checkpoint_file)",
        '"room_0201_arrays_unread": True',
    ):
        if literal not in source:
            raise AssertionError("Teacher-v10 runner lacks guard: " + literal)
    if "apply_flat_direction" in source:
        raise AssertionError("old rollout-state calibration entered Teacher-v10")

    print("[PASS] Teacher-v10 fresh supervised CPU contract")
    print("[PASS] direct-label optimizer, two-scene/timestep and K=3 gates")
    print("[PASS] checkpoint only after actual-rollout PASS")


if __name__ == "__main__":
    main()

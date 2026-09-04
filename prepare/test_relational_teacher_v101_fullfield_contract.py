#!/usr/bin/env python3
"""CPU/static contract for Teacher-v10.1 full-field training."""

from __future__ import annotations

import ast
from pathlib import Path

from relational_teacher_v101_fullfield_contract import (
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    SCENES,
    canonical_sha256,
    rollout_gate,
    shortlist_monitor_steps,
)


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("Teacher-v10.1 policy ID changed")
    rows = []
    for index, step in enumerate(MONITOR_STEPS):
        rows.append(
            {
                "step": step,
                "worst_instance_soft_recall": 0.5 + index * 0.05,
                "worst_instance_active_support_mae": 0.4 - index * 0.04,
                "known_dense_mae": 0.3 - index * 0.03,
                "base_v5_dense": [0.06, 0.06, 0.06],
                "candidate_v5_dense": (
                    [0.08, 0.08, 0.08] if step == 1000 else [0.061, 0.061, 0.061]
                ),
            }
        )
    if shortlist_monitor_steps(rows) != [800, 600, 400]:
        raise AssertionError("v5-aware shortlist ranking changed")

    counts = {scene: {"watch": 2, "write": 2} for scene in SCENES}
    invariance = {scene: [0.001, 0.002, 0.003] for scene in SCENES}
    checks = rollout_gate(
        all_three_counts=counts,
        prompt_invariance=invariance,
        base_v5_dense=[0.06, 0.06, 0.06],
        candidate_v5_dense=[0.061, 0.061, 0.061],
        checkpoint_state_changed=True,
    )
    if not all(checks.values()):
        raise AssertionError("valid Teacher-v10.1 actual rollout was rejected")

    prepare = Path(__file__).resolve().parent
    runner = prepare / "run_relational_teacher_v101_fullfield_supervision.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizer_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW"
    ]
    load_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    model_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "create_model_and_diffusion"
    ]
    checkpoint_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "save_merged_legacy_state"
    ]
    policy_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    if (
        len(optimizer_calls) != 1
        or len(load_calls) != 1
        or len(model_calls) != 1
        or len(checkpoint_calls) != 1
        or not policy_calls
        or policy_calls[0].lineno >= load_calls[0].lineno
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("Teacher-v10.1 execution inventory changed")
    for literal in (
        "for step in range(1, TRAIN_STEPS + 1):",
        "for scene_index, scene in enumerate(SCENES):",
        "fullfield_all_sittable_objective(",
        "if step < REPLAY_START_STEP:",
        "torch.optim.AdamW(",
        "shortlisted_steps = shortlist_monitor_steps(monitor_rows)",
        "selected_step = select_rollout_candidate(rollout_rows)",
        "if selected_step is not None:",
        "save_merged_legacy_state(model, checkpoint_file)",
        '"every_non_unknown_point_receives_gt_supervision": True',
        '"room_0201_arrays_unread": True',
    ):
        if literal not in source:
            raise AssertionError("Teacher-v10.1 runner guard changed: " + literal)
    if "apply_flat_direction" in source:
        raise AssertionError("manual calibration entered Teacher-v10.1")

    objective = (
        prepare / "relational_teacher_v101_fullfield_objective.py"
    ).read_text(encoding="utf-8")
    ast.parse(objective)
    for literal in (
        "known = ~unknown",
        "full_field_known = _point_mse(physical, target, known)",
        "safe_background = known &",
        "torch.topk(values.detach()",
        '"worst_instance_active"',
        "prompt_invariance = _point_mse(physical[0:1], physical[1:2], known[0:1])",
    ):
        if literal not in objective:
            raise AssertionError("Teacher-v10.1 objective guard changed: " + literal)

    print("[PASS] Teacher-v10.1 full-field CPU/static contract")
    print("[PASS] non-unknown full label, hard background and worst-object loss")
    print("[PASS] step-one v5 replay, two-scene K=3 and checkpoint-if-pass guards")


if __name__ == "__main__":
    main()

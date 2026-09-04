#!/usr/bin/env python3
"""CPU/static contract for Teacher-v10.2 dense-instance training."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from relational_teacher_v102_dense_instance_contract import (
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
        raise AssertionError("Teacher-v10.2 policy ID changed")
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
        raise AssertionError("Teacher-v10.2 v5-aware shortlist ranking changed")

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
        raise AssertionError("valid Teacher-v10.2 actual rollout was rejected")

    prepare = Path(__file__).resolve().parent
    objective = (
        prepare / "relational_teacher_v102_dense_instance_objective.py"
    ).read_text(encoding="utf-8")
    ast.parse(objective)
    required_objective = (
        "instance_support_points = (",
        "instance_targets.max(dim=-1).values >= SUPPORT_THRESHOLD",
        "all_instance_support = instance_support_points.any(dim=1)",
        "instance_active_channels = instance_targets >= ACTIVE_THRESHOLD",
        "& ~all_instance_support",
        "support_mask = instance_support_points[",
        "active_mask = instance_active_channels[",
        '"instance_dense_support"',
        '"instance_dense_active"',
        '"worst_instance_dense_active"',
        "torch.topk(values.detach()",
    )
    for literal in required_objective:
        if literal not in objective:
            raise AssertionError("Teacher-v10.2 objective guard changed: " + literal)
    forbidden_objective = (
        "verified_masks =",
        "& point_mask.unsqueeze(-1)",
        "point_mask = verified_masks",
    )
    for literal in forbidden_objective:
        if literal in objective:
            raise AssertionError("v10.1 surface-only mask re-entered v10.2")

    runner = (
        prepare / "run_relational_teacher_v102_dense_instance_supervision.py"
    ).read_text(encoding="utf-8")
    ast.parse(runner)
    for literal in (
        '"fullfield_all_sittable_objective": dense_instance_all_sittable_objective',
        '"_validate_v10_failure": _validate_v101_failure',
        '"V10_SCHEMA": V101_SCHEMA',
        '"dense_instance_positive_support_is_equal_macro": True',
        '"all_instance_positive_support_excluded_from_hard_background": True',
        '"at_least_one_dense_instance_candidate_passes_actual_k3"',
        'checkpoint_file = output_dir / "teacher_v102_dense_instance.pt"',
        '"checkpoint_written_iff_actual_gate_passes"',
        '"room_0201_arrays_unread": True',
    ):
        if literal not in runner:
            raise AssertionError("Teacher-v10.2 runner guard changed: " + literal)
    if "apply_flat_direction" in runner:
        raise AssertionError("manual calibration entered Teacher-v10.2")

    base_path = next(
        (
            Path(entry) / "run_relational_teacher_v101_fullfield_supervision.py"
            for entry in sys.path
            if (Path(entry) / "run_relational_teacher_v101_fullfield_supervision.py").is_file()
        ),
        None,
    )
    if base_path is None:
        raise AssertionError("Teacher-v10.1 base pipeline is missing")
    base_pipeline = base_path.resolve().read_text(encoding="utf-8")
    base_tree = ast.parse(base_pipeline)
    calls = [node for node in ast.walk(base_tree) if isinstance(node, ast.Call)]
    if (
        len(
            [
                node
                for node in calls
                if isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW"
            ]
        )
        != 1
        or len(
            [
                node
                for node in calls
                if isinstance(node.func, ast.Name)
                and node.func.id == "save_merged_legacy_state"
            ]
        )
        != 1
    ):
        raise AssertionError("Teacher-v10.2 base execution inventory changed")
    for literal in (
        "for step in range(1, TRAIN_STEPS + 1):",
        "for scene_index, scene in enumerate(SCENES):",
        "shortlisted_steps = shortlist_monitor_steps(monitor_rows)",
        "selected_step = select_rollout_candidate(rollout_rows)",
        "if selected_step is not None:",
        "save_merged_legacy_state(model, checkpoint_file)",
    ):
        if literal not in base_pipeline:
            raise AssertionError("Teacher-v10.2 base pipeline guard changed: " + literal)

    print("[PASS] Teacher-v10.2 dense-instance CPU/static contract")
    print("[PASS] three full dense supports are equal-macro and never hard negatives")
    print("[PASS] fresh v5r4, two-scene K=3 and checkpoint-if-pass guards")


if __name__ == "__main__":
    main()

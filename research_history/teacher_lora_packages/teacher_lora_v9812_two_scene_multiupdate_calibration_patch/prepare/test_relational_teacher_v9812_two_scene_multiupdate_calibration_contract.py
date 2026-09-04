#!/usr/bin/env python3
"""CPU/static tests for Teacher-v9.8.12 multi-update calibration."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from relational_teacher_v9812_two_scene_multiupdate_calibration_contract import (
    AUDIT_SCENE,
    AUDIT_SEED_TABLE,
    DESIGN_SEED_TABLE,
    POLICY,
    POLICY_ID,
    SOURCE_SCENE,
    UPDATE_COUNT,
    all_three_counts,
    calibration_checks,
    canonical_sha256,
    direction_candidates,
    rank_shortlist,
    select_direction,
)


def _pooled(recall: float = 0.8, mae: float = 0.08) -> dict:
    return {
        scene: {
            role: {
                "soft_recall": recall,
                "active_support_mae": mae,
                "hotspot_centroid_distance_xy": 0.1,
                "topk_overlap": 0.5,
            }
            for role in ("bed", "normal_chair", "high_chair")
        }
        for scene in (SOURCE_SCENE, AUDIT_SCENE)
    }


def _presence(missing: tuple[str, int, int] | None = None) -> dict:
    rows = {}
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        rows[scene] = []
        for generation in range(3):
            prompt_rows = []
            for prompt in range(2):
                checks = {"bed": True, "normal": True, "high": True, "negative": True}
                if missing == (scene, generation, prompt):
                    checks["high"] = False
                prompt_rows.append(checks)
            rows[scene].append(prompt_rows)
    return rows


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("calibration policy ID changed")
    seed_values = [value for table in (DESIGN_SEED_TABLE, AUDIT_SEED_TABLE) for row in table for value in row]
    if len(seed_values) != len(set(seed_values)):
        raise AssertionError("design/audit seed tables overlap")

    gram = np.eye(51, dtype=np.float64)
    task_order = ["task_{:02d}".format(i) for i in range(39)] + [
        "task_{:02d}_high_chair".format(i) for i in range(12)
    ]
    rows = direction_candidates(gram, task_order)
    if len(rows) != 6 or not all(row["eligible"] for row in rows):
        raise AssertionError("valid common-descent direction grid was rejected")
    selected = select_direction(rows)
    if min(selected["directional_derivatives"]) <= 0.0:
        raise AssertionError("selected synthetic direction is not common descent")

    start = _pooled()
    candidate = _pooled(0.82, 0.07)
    presence = _presence()
    checks = calibration_checks(
        step=1,
        scene_checks={SOURCE_SCENE: {"strict": True}, AUDIT_SCENE: {"strict": True}},
        start_pooled=start,
        candidate_pooled=candidate,
        presence=presence,
        directional_derivatives=[0.1] * 51,
        pre_state_sha256="a" * 64,
        post_state_sha256="b" * 64,
    )
    if not all(checks.values()) or all_three_counts(presence) != {
        SOURCE_SCENE: {"watch": 3, "write": 3},
        AUDIT_SCENE: {"watch": 3, "write": 3},
    }:
        raise AssertionError("valid all-three calibration result was rejected")
    missing = _presence((AUDIT_SCENE, 0, 1))
    failed = calibration_checks(
        step=1,
        scene_checks={SOURCE_SCENE: {"strict": True}, AUDIT_SCENE: {"strict": True}},
        start_pooled=start,
        candidate_pooled=candidate,
        presence=missing,
        directional_derivatives=[0.1] * 51,
        pre_state_sha256="a" * 64,
        post_state_sha256="b" * 64,
    )
    if failed[AUDIT_SCENE + "_write_all_three_at_least_two_of_three"] is not True:
        raise AssertionError("two-of-three all-three policy changed")
    missing_two = _presence((AUDIT_SCENE, 0, 1))
    missing_two[AUDIT_SCENE][1][1]["high"] = False
    failed = calibration_checks(
        step=1,
        scene_checks={SOURCE_SCENE: {"strict": True}, AUDIT_SCENE: {"strict": True}},
        start_pooled=start,
        candidate_pooled=candidate,
        presence=missing_two,
        directional_derivatives=[0.1] * 51,
        pre_state_sha256="a" * 64,
        post_state_sha256="b" * 64,
    )
    if failed[AUDIT_SCENE + "_write_all_three_at_least_two_of_three"] is not False:
        raise AssertionError("missing-object generations were accepted")

    shortlist_rows = []
    for step in range(1, UPDATE_COUNT + 1):
        shortlist_rows.append(
            {
                "step": step,
                "eligible": step in (2, 4, 6),
                "all_three_counts": all_three_counts(presence),
                "scene_candidate_pooled": _pooled(0.80 + step / 100.0, 0.08),
                "candidate_v5_dense": [0.06, 0.06, 0.06],
            }
        )
    if rank_shortlist(shortlist_rows) != [6]:
        raise AssertionError("shortlist ranking changed")

    runner = Path(__file__).with_name(
        "run_relational_teacher_v9812_two_scene_multiupdate_calibration.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
    ]
    policy_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "create_model_and_diffusion"
    ]
    if len(loads) != 2 or len(model_calls) != 1:
        raise AssertionError("two-scene/model call inventory changed")
    if (
        not policy_calls
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("policy is not locked before scene arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("calibration may not create an optimizer/checkpoint")
    for literal in (
        "v9811 = _validate_v9811(v9811_file)",
        "for step in RECONSTRUCTION_STEPS:",
        "for step in MONITOR_STEPS:",
        "direction_candidates(current_gram, V988_TASK_ORDER)",
        "two_scene_preflight_checks(",
        "all_three_counts(scene_presence)",
        "rank_shortlist(monitor_rows)",
        '"at_least_one_all_three_state_is_shortlisted": bool(shortlisted_steps)',
        '"authorizes_shortlisted_state_checkpoint_export_gate"',
    ):
        if literal not in source:
            raise AssertionError("calibration runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.12 two-scene multi-update CPU contract")
    print("[PASS] 51-task common descent, per-update K=3 and hard all-three gates")
    print("[PASS] shortlist-only authority, two-scene load and no-checkpoint guards")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU/static tests for Teacher-v9.8.2 rollout-state calibration-6."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v981_rollout_state_response6_contract import response6_checks
from relational_teacher_v982_rollout_state_calibration6_contract import (
    MODEL_SEED,
    MONITOR_STEPS,
    OBJECTS,
    POLICY,
    POLICY_ID,
    STEP_RADIUS,
    UPDATE_COUNT,
    calibration_design_seeds,
    calibration_step_checks,
    canonical_sha256,
    pooled_object_metrics,
    rank_eligible_steps,
)


def metric(recall: float, mae: float) -> dict:
    return {
        "instances": {
            name: {
                "soft_recall": recall,
                "active_support_mae": mae,
                "hotspot_centroid_distance_xy": 0.2,
                "topk_overlap": 0.3,
            }
            for name in OBJECTS
        },
        "explicit_negative_mean": 0.03,
        "explicit_negative_max": 0.4,
    }


def panel(recall: float, mae: float) -> list:
    return [[metric(recall, mae), metric(recall + 0.01, mae - 0.01)] for _ in range(3)]


def response(base: list, candidate: list) -> dict:
    return response6_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_prompt_invariance=(0.01, 0.02, 0.03),
        candidate_prompt_invariance=(0.0101, 0.0201, 0.0301),
        base_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.0501, 0.0601, 0.0701),
        directional_derivatives=(0.2,) * 6,
        maximum_map_delta=0.02,
    )


def row(step: int, base: list, candidate: list) -> dict:
    response_checks_value = response(base, candidate)
    checks = calibration_step_checks(
        step=step,
        base_rows=base,
        candidate_rows=candidate,
        response_checks=response_checks_value,
        directional_derivatives=(0.2,) * 6,
        pre_state_sha256="a" * 64,
        post_state_sha256=(str(step) * 64)[:64],
    )
    return {
        "step": step,
        "base_pooled": pooled_object_metrics(base),
        "candidate_pooled": pooled_object_metrics(candidate),
        "eligible": all(checks.values()),
    }


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("calibration policy ID changed")
    if (MODEL_SEED, UPDATE_COUNT, MONITOR_STEPS, STEP_RADIUS) != (
        20261016,
        6,
        (1, 2, 3, 4, 5, 6),
        0.003,
    ):
        raise AssertionError("calibration protocol changed")
    seeds = [calibration_design_seeds(step) for step in range(2, 7)]
    if len(set(seeds)) != 5:
        raise AssertionError("calibration design seeds collide")

    base = panel(0.2, 0.6)
    candidates = [panel(0.2 + step * 0.01, 0.6 - step * 0.01) for step in range(1, 7)]
    rows = [row(step, base, candidates[step - 1]) for step in range(1, 7)]
    if not all(one["eligible"] for one in rows):
        raise AssertionError("valid calibration panels were rejected")
    shortlist = rank_eligible_steps(rows)
    if shortlist != [6, 5] or 1 in shortlist:
        raise AssertionError("post-first calibration ranking changed")

    regression = copy.deepcopy(candidates[2])
    regression[1][1]["instances"]["chair_06"]["soft_recall"] = 0.1
    regression_checks = calibration_step_checks(
        step=3,
        base_rows=base,
        candidate_rows=regression,
        response_checks=response(base, regression),
        directional_derivatives=(0.2,) * 6,
        pre_state_sha256="a" * 64,
        post_state_sha256="b" * 64,
    )
    if regression_checks["response6_policy_is_admissible"] is not False:
        raise AssertionError("one-map High Chair regression passed")

    runner = Path(__file__).with_name(
        "run_relational_teacher_v982_rollout_state_calibration6.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(loads) != 1:
        raise AssertionError("calibration runner must load exactly one scene")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("calibration runner may not create optimizer/checkpoint state")
    if source.index("atomic_write_json(policy_file") > source.index(
        "create_model_and_diffusion"
    ):
        raise AssertionError("calibration policy is written after model creation")
    for literal in (
        "for step in range(1, UPDATE_COUNT + 1)",
        "calibration_design_seeds(step)",
        "apply_flat_direction(parameters, direction, STEP_RADIUS)",
        "response6_checks(",
        "rank_eligible_steps(monitor_rows)",
        "update-1 K=3 response does not reproduce v9.8.1",
    ):
        if literal not in source:
            raise AssertionError("calibration runner lacks guard: " + literal)
    print("[PASS] Teacher-v9.8.2 rollout-state calibration-6 CPU contract")
    print("[PASS] post-first shortlist and one-map regression fail closed")
    print("[PASS] fresh six-update, one-scene and no-checkpoint static guards")


if __name__ == "__main__":
    main()

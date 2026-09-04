#!/usr/bin/env python3
"""CPU/static tests for Teacher-v9.8.4 preservation calibration-6."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v981_rollout_state_response6_contract import response6_checks
from relational_teacher_v984_preservation_calibration6_contract import (
    MODEL_SEED,
    MONITOR_STEPS,
    OBJECTS,
    POLICY,
    POLICY_ID,
    STEP_RADIUS,
    TASK_ORDER,
    UPDATE_COUNT,
    calibration_checks,
    canonical_sha256,
    pooled_object_metrics,
    rank_eligible_steps,
)


def metric(recall: float, mae: float, negative: float = 0.03) -> dict:
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
        "explicit_negative_mean": negative,
        "explicit_negative_max": 0.4,
    }


def panel(recall: float, mae: float, negative: float = 0.03) -> list:
    return [
        [metric(recall, mae, negative), metric(recall + 0.01, mae - 0.01, negative)]
        for _ in range(3)
    ]


def response(base: list, candidate: list) -> dict:
    return response6_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_prompt_invariance=(0.01, 0.02, 0.03),
        candidate_prompt_invariance=(0.009, 0.019, 0.029),
        base_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.0501, 0.0601, 0.0701),
        directional_derivatives=(0.2,) * 6,
        maximum_map_delta=0.02,
    )


def row(step: int, candidate: list, v5: list[float]) -> dict:
    return {
        "step": step,
        "candidate_pooled": pooled_object_metrics(candidate),
        "candidate_rows": candidate,
        "candidate_v5_dense": v5,
        "eligible": True,
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
    if len(TASK_ORDER) != 11 or TASK_ORDER[-5:] != (
        "v5_chair",
        "v5_bed",
        "v5_whiteboard",
        "negative_watch",
        "negative_write",
    ):
        raise AssertionError("calibration task inventory changed")

    base = panel(0.20, 0.60)
    step1 = panel(0.30, 0.50, 0.031)
    step2 = panel(0.32, 0.48, 0.0311)
    step1_checks = calibration_checks(
        step=1,
        previous_rows=base,
        candidate_rows=step1,
        previous_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.0501, 0.0601, 0.0701),
        response_checks=response(base, step1),
        directional_derivatives=(0.2,) * 6,
        pre_state_sha256="a" * 64,
        post_state_sha256="b" * 64,
    )
    step2_checks = calibration_checks(
        step=2,
        previous_rows=step1,
        candidate_rows=step2,
        previous_v5_dense=(0.0501, 0.0601, 0.0701),
        candidate_v5_dense=(0.0499, 0.0599, 0.0699),
        response_checks=response(base, step2),
        directional_derivatives=(0.2,) * 11,
        pre_state_sha256="b" * 64,
        post_state_sha256="c" * 64,
    )
    if not all(step1_checks.values()) or not all(step2_checks.values()):
        raise AssertionError("valid preservation calibration step was rejected")

    v5_regression = calibration_checks(
        step=2,
        previous_rows=step1,
        candidate_rows=step2,
        previous_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.051, 0.059, 0.069),
        response_checks=response(base, step2),
        directional_derivatives=(0.2,) * 11,
        pre_state_sha256="b" * 64,
        post_state_sha256="c" * 64,
    )
    if v5_regression["v5_chair_not_worse_than_previous"] is not False:
        raise AssertionError("one-case v5 regression passed")
    negative = copy.deepcopy(step2)
    negative[1][0]["explicit_negative_mean"] = 0.032
    negative_regression = calibration_checks(
        step=2,
        previous_rows=step1,
        candidate_rows=negative,
        previous_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.049, 0.059, 0.069),
        response_checks=response(base, negative),
        directional_derivatives=(0.2,) * 11,
        pre_state_sha256="b" * 64,
        post_state_sha256="c" * 64,
    )
    if negative_regression["generation_1_negative_mean_preserved"] is not False:
        raise AssertionError("generation-specific negative regression passed")

    step3 = panel(0.34, 0.46, 0.0311)
    ranking = rank_eligible_steps(
        [
            row(1, step1, [0.0501, 0.0601, 0.0701]),
            row(2, step2, [0.0499, 0.0599, 0.0699]),
            row(3, step3, [0.0498, 0.0598, 0.0698]),
        ]
    )
    if ranking != [3, 2] or 1 in ranking:
        raise AssertionError("post-first preservation shortlist changed")

    runner = Path(__file__).with_name(
        "run_relational_teacher_v984_preservation_calibration6.py"
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
        raise AssertionError("calibration must load exactly one scene")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("calibration may not create optimizer/checkpoint state")
    if source.index("atomic_write_json(policy_file") > source.index(
        "create_model_and_diffusion"
    ):
        raise AssertionError("calibration policy is written after model creation")
    for literal in (
        "for step in MONITOR_STEPS:",
        "calibration_design_seeds(step)",
        "torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)",
        "apply_flat_direction(parameters, direction, STEP_RADIUS)",
        "response6_checks(",
        "calibration_checks(",
        "rank_eligible_steps(monitor_rows)",
        "update-2 K=3 maps do not reproduce selected v9.8.3",
        'f"[CALIBRATION] step={step}/6 eligible={eligible} "',
    ):
        if literal not in source:
            raise AssertionError("calibration runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.4 preservation calibration-6 CPU contract")
    print("[PASS] post-first shortlist and v5/negative regressions fail closed")
    print("[PASS] six-update, one-scene, pre-model policy and no-checkpoint guards")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU/static tests for Teacher-v9.8.3 preservation direction preflight."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v981_rollout_state_response6_contract import response6_checks
from relational_teacher_v983_preservation_direction_contract import (
    MODEL_SEED,
    OBJECTS,
    POLICY,
    POLICY_ID,
    STEP_RADII,
    TASK_ORDER,
    canonical_sha256,
    pooled_object_metrics,
    preservation_checks,
    rank_candidates,
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


def candidate_row(name: str, radius: float, step1: list, candidate: list) -> dict:
    return {
        "name": name,
        "radius": radius,
        "step1_pooled": pooled_object_metrics(step1),
        "candidate_pooled": pooled_object_metrics(candidate),
        "candidate_v5_dense": [0.049, 0.059, 0.069],
        "candidate_rows": candidate,
        "eligible": True,
    }


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("preservation policy ID changed")
    if MODEL_SEED != 20261016 or STEP_RADII != (
        0.00025,
        0.0005,
        0.001,
        0.002,
        0.003,
    ):
        raise AssertionError("preservation protocol changed")
    if len(TASK_ORDER) != 11 or TASK_ORDER[-5:] != (
        "v5_chair",
        "v5_bed",
        "v5_whiteboard",
        "negative_watch",
        "negative_write",
    ):
        raise AssertionError("preservation task inventory changed")

    base = panel(0.20, 0.60)
    step1 = panel(0.30, 0.50, 0.031)
    candidate = panel(0.32, 0.48, 0.0311)
    response_checks_value = response(base, candidate)
    checks = preservation_checks(
        step1_rows=step1,
        candidate_rows=candidate,
        step1_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.049, 0.059, 0.069),
        response_checks=response_checks_value,
        directional_derivatives=(0.2,) * 11,
        maximum_incremental_map_delta=0.01,
    )
    if not all(checks.values()):
        raise AssertionError("valid eleven-task response was rejected")

    v5_regression = preservation_checks(
        step1_rows=step1,
        candidate_rows=candidate,
        step1_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.051, 0.059, 0.069),
        response_checks=response_checks_value,
        directional_derivatives=(0.2,) * 11,
        maximum_incremental_map_delta=0.01,
    )
    if v5_regression["v5_chair_not_worse_than_step1"] is not False:
        raise AssertionError("one-case v5 regression passed")
    negative_regression = copy.deepcopy(candidate)
    negative_regression[2][0]["explicit_negative_mean"] = 0.032
    negative_checks = preservation_checks(
        step1_rows=step1,
        candidate_rows=negative_regression,
        step1_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.049, 0.059, 0.069),
        response_checks=response(base, negative_regression),
        directional_derivatives=(0.2,) * 11,
        maximum_incremental_map_delta=0.01,
    )
    if negative_checks["generation_2_negative_mean_preserved"] is not False:
        raise AssertionError("generation-specific negative regression passed")

    stronger = panel(0.34, 0.46, 0.0311)
    ranking = rank_candidates(
        [
            candidate_row("small", 0.00025, step1, candidate),
            candidate_row("strong", 0.0005, step1, stronger),
        ]
    )
    if ranking != ["strong", "small"]:
        raise AssertionError("preservation ranking changed")

    runner = Path(__file__).with_name(
        "preflight_relational_teacher_v983_preservation_direction.py"
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
        raise AssertionError("preservation preflight must load exactly one scene")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("preservation preflight may not create optimizer/checkpoint state")
    if source.index("atomic_write_json(policy_file") > source.index(
        "create_model_and_diffusion"
    ):
        raise AssertionError("preservation policy is written after model creation")
    for literal in (
        "flattened_task_gradients(all_tasks, parameters)",
        "torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)",
        "for radius_index, radius in enumerate(STEP_RADII)",
        "response6_checks(",
        "rank_candidates(candidate_rows)",
        "reconstructed update-1 K=3 maps differ from v9.8.1",
        "eligible = all(checks.values())",
        'f"[PRESERVE R={radius:.5f}] eligible={eligible} "',
    ):
        if literal not in source:
            raise AssertionError("preservation runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.3 preservation direction CPU contract")
    print("[PASS] v5/negative regression and deterministic ranking fail closed")
    print("[PASS] eleven-task, one-scene, pre-model policy and no-checkpoint guards")


if __name__ == "__main__":
    main()

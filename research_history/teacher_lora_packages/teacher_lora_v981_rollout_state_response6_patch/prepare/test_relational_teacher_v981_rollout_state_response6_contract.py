#!/usr/bin/env python3
"""CPU/static tests for Teacher-v9.8.1 selected rollout-state response-6."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v981_rollout_state_response6_contract import (
    OBJECTS,
    POLICY,
    POLICY_ID,
    SEED,
    SELECTED_NAME,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    canonical_sha256,
    response6_checks,
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


def evaluate(base: list, candidate: list) -> dict:
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


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("policy ID changed")
    if (SELECTED_NAME, SELECTED_TIMESTEP, SELECTED_RADIUS) != (
        "t50_radius_0p003",
        50,
        0.003,
    ):
        raise AssertionError("selected v9.8 candidate changed")
    if SEED != 20261016:
        raise AssertionError("v9.8 LoRA construction seed changed")
    if POLICY["topk_role"] != "diagnostic_only":
        raise AssertionError("Top-k became a response gate")

    base = [[metric(0.2, 0.6), metric(0.21, 0.59)] for _ in range(3)]
    candidate = [[metric(0.23, 0.56), metric(0.24, 0.55)] for _ in range(3)]
    checks = evaluate(base, candidate)
    if not all(checks.values()):
        raise AssertionError("valid K=3 response was rejected")

    one_seed = copy.deepcopy(candidate)
    for generation in (1, 2):
        for prompt in range(2):
            one_seed[generation][prompt]["instances"]["chair_06"]["soft_recall"] = 0.2
            one_seed[generation][prompt]["instances"]["chair_06"]["active_support_mae"] = 0.6
    failed = evaluate(base, one_seed)
    if failed["high_chair_at_least_two_generations_improve"] is not False:
        raise AssertionError("single-generation High Chair response passed")

    regression = copy.deepcopy(candidate)
    regression[2][1]["instances"]["bed_01"]["soft_recall"] = 0.1
    failed = evaluate(base, regression)
    if failed["g2_write_bed_01_recall_retained"] is not False:
        raise AssertionError("one-map Bed regression passed")

    runner = Path(__file__).with_name(
        "evaluate_relational_teacher_v981_rollout_state_response6.py"
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
        raise AssertionError("runner must load one train scene")
    if "torch.save" in source or "save_trainable_state" in source or "torch.optim" in source:
        raise AssertionError("response-6 must not write/train a model")
    if source.index("atomic_write_json(policy_file") > source.index("install_lora(model"):
        raise AssertionError("response-6 policy is written after LoRA installation")
    for literal in (
        "_capture_trajectory",
        "_resume_trajectory",
        "v98_candidate_normalized",
        "v97_base_normalized",
        "tensor_sha256(direction)",
    ):
        if literal not in source:
            raise AssertionError("runner lacks exact response binding: " + literal)
    print("[PASS] Teacher-v9.8.1 selected rollout-state response-6 CPU contract")
    print("[PASS] single-seed response and one-map object regression fail closed")
    print("[PASS] exact v9.8 direction/reuse, one-scene and no-checkpoint guards")


if __name__ == "__main__":
    main()

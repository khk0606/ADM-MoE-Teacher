#!/usr/bin/env python3
"""CPU and static contracts for Teacher-v9.7 early rollout K=3."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v97_early_rollout_k3_contract import (
    GENERATION_COUNT,
    OBJECTS,
    ROLLOUT_POLICY,
    ROLLOUT_POLICY_ID,
    SNAPSHOT_STEPS,
    canonical_sha256,
    continuous_presence_checks,
    pooled_object_metrics,
    rank_eligible_steps,
    rollout_step_checks,
    stable_rollout_seeds,
)


def _instance(recall: float, mae: float, centroid: float, topk: float) -> dict:
    return {
        "soft_recall": recall,
        "active_support_mae": mae,
        "hotspot_centroid_distance_xy": centroid,
        "topk_overlap": topk,
    }


def _metric(
    *,
    recall: float,
    mae: float,
    topk: float,
    negative_mean: float = 0.03,
    negative_max: float = 0.40,
) -> dict:
    return {
        "instances": {
            name: _instance(recall, mae, 0.20, topk) for name in OBJECTS
        },
        "explicit_negative_mean": negative_mean,
        "explicit_negative_max": negative_max,
    }


def _rows(recall: float, mae: float, topk: float) -> list[list[dict]]:
    return [
        [_metric(recall=recall, mae=mae, topk=topk) for _ in range(2)]
        for _ in range(GENERATION_COUNT)
    ]


def main() -> None:
    if canonical_sha256(ROLLOUT_POLICY) != ROLLOUT_POLICY_ID:
        raise AssertionError("rollout policy ID is not canonical")
    if ROLLOUT_POLICY["topk_role"] != (
        "diagnostic_only_not_a_gate_due_discrete_one_point_quantization"
    ):
        raise AssertionError("exact Top-k unexpectedly became a decision gate")

    base = _rows(0.80, 0.090, 0.90)
    # Top-k is intentionally much worse while every continuous metric improves.
    candidate = _rows(0.84, 0.080, 0.10)
    presence = [
        [continuous_presence_checks(candidate[generation][prompt]) for prompt in range(2)]
        for generation in range(GENERATION_COUNT)
    ]
    checks = rollout_step_checks(
        base_rows=base,
        candidate_rows=candidate,
        candidate_presence=presence,
        base_prompt_invariance=(0.02, 0.03, 0.04),
        candidate_prompt_invariance=(0.0201, 0.0301, 0.0401),
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if not all(checks.values()):
        raise AssertionError("valid continuous K=3 rollout was rejected")

    # A background diagnostic does not redefine whether all three Sit objects
    # are present. It is rejected by its own retention gate instead.
    leaked = copy.deepcopy(candidate)
    leaked[0][0]["explicit_negative_mean"] = 0.5
    leaked_presence = [
        [continuous_presence_checks(leaked[generation][prompt]) for prompt in range(2)]
        for generation in range(GENERATION_COUNT)
    ]
    leaked_checks = rollout_step_checks(
        base_rows=base,
        candidate_rows=leaked,
        candidate_presence=leaked_presence,
        base_prompt_invariance=(0.02, 0.03, 0.04),
        candidate_prompt_invariance=(0.0201, 0.0301, 0.0401),
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if leaked_checks["watch_at_least_two_generations_have_all_three"] is not True:
        raise AssertionError("background leakage was confused with object presence")
    if leaked_checks["every_generation_negative_mean_retained"] is not False:
        raise AssertionError("large background leakage escaped its independent gate")

    missing = copy.deepcopy(candidate)
    for prompt in range(2):
        missing[0][prompt]["instances"]["chair_06"]["soft_recall"] = 0.20
        missing[0][prompt]["instances"]["chair_06"]["active_support_mae"] = 0.50
        missing[1][prompt]["instances"]["chair_06"]["soft_recall"] = 0.20
        missing[1][prompt]["instances"]["chair_06"]["active_support_mae"] = 0.50
    missing_presence = [
        [continuous_presence_checks(missing[generation][prompt]) for prompt in range(2)]
        for generation in range(GENERATION_COUNT)
    ]
    failed = rollout_step_checks(
        base_rows=base,
        candidate_rows=missing,
        candidate_presence=missing_presence,
        base_prompt_invariance=(0.02, 0.03, 0.04),
        candidate_prompt_invariance=(0.0201, 0.0301, 0.0401),
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if failed["watch_at_least_two_generations_have_all_three"] is not False:
        raise AssertionError("watch with only one all-three generation was accepted")
    if failed["write_at_least_two_generations_have_all_three"] is not False:
        raise AssertionError("write with only one all-three generation was accepted")
    if failed["high_chair_pooled_soft_recall_at_least_075"] is not False:
        raise AssertionError("missing High Chair passed the pooled recall gate")

    pooled = pooled_object_metrics(candidate)
    rows = []
    for step, recall, mae in ((3, 0.80, 0.090), (6, 0.85, 0.085), (12, 0.85, 0.080)):
        step_pooled = copy.deepcopy(pooled)
        for name in OBJECTS:
            step_pooled[name]["soft_recall"] = recall
            step_pooled[name]["active_support_mae"] = mae
        rows.append({"step": step, "eligible": True, "candidate_pooled": step_pooled})
    if rank_eligible_steps(rows) != [12, 6, 3]:
        raise AssertionError("eligible early-step ranking changed")

    seeds = [stable_rollout_seeds(index) for index in range(GENERATION_COUNT)]
    if len(set(seeds)) != GENERATION_COUNT or stable_rollout_seeds(0) != seeds[0]:
        raise AssertionError("K=3 stable seed derivation changed")

    runner = Path(__file__).with_name(
        "evaluate_relational_teacher_v97_early_rollout_k3.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(calls) != 1:
        raise AssertionError("runner must contain exactly one scene-array load")
    expression = calls[0].args[-1]
    slice_value = expression.slice if isinstance(expression, ast.Subscript) else None
    if isinstance(slice_value, ast.Index):
        slice_value = slice_value.value
    if not (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "records"
        and isinstance(slice_value, ast.Name)
        and slice_value.id == "TRAIN_SCENE"
    ):
        raise AssertionError("scene-array load is not statically bound to room_0101")
    if "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("v9.7 must not serialize model state")
    if source.index("atomic_write_json(rollout_policy_file") > source.index(
        "optimizer = torch.optim.AdamW"
    ):
        raise AssertionError("rollout metric policy is written after optimizer creation")
    for literal in ("28", "SNAPSHOT_STEPS", "stable_rollout_seeds"):
        if literal not in source:
            raise AssertionError("runner is missing sealed rollout protocol: " + literal)

    print("[PASS] Teacher-v9.7 continuous K=3 selection contract")
    print("[PASS] exact Top-k is diagnostic while missing-object continuous failures close the gate")
    print("[PASS] room_0101-only load, pre-optimizer policy and no-checkpoint static guards")


if __name__ == "__main__":
    main()

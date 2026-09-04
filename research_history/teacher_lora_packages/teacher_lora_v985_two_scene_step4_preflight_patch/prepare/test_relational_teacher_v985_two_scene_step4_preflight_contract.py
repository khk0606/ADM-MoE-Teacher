#!/usr/bin/env python3
"""CPU/static contract tests for Teacher-v9.8.5 two-scene preflight."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v985_two_scene_step4_preflight_contract import (
    AUDIT_SCENE,
    EXPECTED_INSTANCES,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    RECONSTRUCTION_STEPS,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    STEP_RADIUS,
    canonical_sha256,
    pooled_by_role,
    two_scene_preflight_checks,
)


NAMES = EXPECTED_INSTANCES[AUDIT_SCENE]


def metric(recall: float, mae: float, negative_mean: float = 0.03) -> dict:
    return {
        "instances": {
            name: {
                "soft_recall": recall,
                "active_support_mae": mae,
                "hotspot_centroid_distance_xy": 0.2,
                "topk_overlap": 0.3,
            }
            for name in NAMES
        },
        "explicit_negative_mean": negative_mean,
        "explicit_negative_max": 0.4,
    }


def panel(recall: float, mae: float, negative_mean: float = 0.03) -> list:
    return [
        [
            metric(recall + generation * 0.01, mae, negative_mean),
            metric(recall + generation * 0.01, mae, negative_mean),
        ]
        for generation in range(3)
    ]


def evaluate(base: list, candidate: list) -> dict:
    return two_scene_preflight_checks(
        base_rows=base,
        candidate_rows=candidate,
        instance_names=NAMES,
        base_prompt_invariance=(0.01, 0.02, 0.03),
        candidate_prompt_invariance=(0.009, 0.019, 0.029),
        base_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.049, 0.059, 0.069),
        maximum_map_delta=0.02,
    )


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("two-scene policy ID changed")
    if (
        MODEL_SEED,
        SELECTED_V984_STEP,
        RECONSTRUCTION_STEPS,
        SELECTED_TIMESTEP,
        STEP_RADIUS,
    ) != (20261016, 4, (1, 2, 3, 4), 50, 0.003):
        raise AssertionError("two-scene reconstruction protocol changed")

    base = panel(0.30, 0.50)
    candidate = panel(0.32, 0.48)
    checks = evaluate(base, candidate)
    if not all(checks.values()):
        raise AssertionError("valid two-scene response was rejected")
    pooled = pooled_by_role(candidate, NAMES)
    if set(pooled) != {"bed", "normal_chair", "high_chair"}:
        raise AssertionError("room_0102 roles changed")

    missing_high = copy.deepcopy(candidate)
    for generation in range(3):
        for prompt in range(2):
            missing_high[generation][prompt]["instances"]["chair_06"]["soft_recall"] = 0.29
            missing_high[generation][prompt]["instances"]["chair_06"]["active_support_mae"] = 0.51
    missing_checks = evaluate(base, missing_high)
    if missing_checks["high_chair_pooled_recall_strictly_improves"] is not False:
        raise AssertionError("missing High Chair response passed")

    one_map_regression = copy.deepcopy(candidate)
    one_map_regression[2][1]["instances"]["chair_05"]["soft_recall"] = 0.28
    regression_checks = evaluate(base, one_map_regression)
    if regression_checks["g2_write_normal_chair_recall_retained"] is not False:
        raise AssertionError("one-map normal-Chair regression passed")

    v5_regression = two_scene_preflight_checks(
        base_rows=base,
        candidate_rows=candidate,
        instance_names=NAMES,
        base_prompt_invariance=(0.01, 0.02, 0.03),
        candidate_prompt_invariance=(0.009, 0.019, 0.029),
        base_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.06, 0.07, 0.08),
        maximum_map_delta=0.02,
    )
    if v5_regression["v5_fixed_probe_retained_1pct"] is not False:
        raise AssertionError("v5 regression passed")

    runner = Path(__file__).with_name(
        "preflight_relational_teacher_v985_two_scene_step4.py"
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
    if len(loads) != 2:
        raise AssertionError("preflight must load exactly two train scenes")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("preflight may not create optimizer/checkpoint state")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
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
    if (
        len(policy_calls) < 1
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise AssertionError("policy is not locked before arrays/model")
    for literal in (
        "for step in RECONSTRUCTION_STEPS:",
        "_direction_exact(direction_row, v984[\"direction_rows\"][step - 1], step)",
        "candidate_v5_predictions\"][SELECTED_V984_STEP - 1]",
        "two_scene_preflight_checks(",
        "room0102_step4_response_is_admissible",
        'f"[AUDIT-CANDIDATE] generation={generation} prompt={PROMPT_IDS[prompt_index]}"',
    ):
        if literal not in source:
            raise AssertionError("two-scene runner lacks guard: " + literal)

    print("[PASS] Teacher-v9.8.5 two-scene step-4 preflight CPU contract")
    print("[PASS] missing-object, one-map and v5 regressions fail closed")
    print("[PASS] exact four-update, two-scene and no-checkpoint static guards")


if __name__ == "__main__":
    main()

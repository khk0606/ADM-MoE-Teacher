#!/usr/bin/env python3
"""CPU contract tests for Teacher-v9.4 common-descent audit."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from relational_teacher_v94_common_descent import frank_wolfe_min_norm_weights
from relational_teacher_v94_common_descent_contract import (
    EXPECTED_V93_FAILURES,
    FW_ITERATIONS,
    STEP_RADII,
    candidate_checks,
    rank_candidates,
)


def _panel(value: float) -> dict:
    prompts = []
    for _ in range(2):
        prompts.append(
            {
                "instances": {
                    name: {
                        "topk_overlap": 0.3,
                        "soft_recall": 0.8,
                        "active_support_mae": 0.1,
                    }
                    for name in ("bed_01", "chair_01", "chair_06")
                },
                "explicit_negative_mean": 0.01,
                "explicit_negative_max": 0.1,
            }
        )
    return {
        "per_task_exact_topk_swap": [[value] * 3, [value] * 3],
        "per_prompt_metrics": prompts,
        "background_trust": 0.0,
        "prompt_invariance": 0.01,
        "negative_mean": 0.01,
        "negative_max": 0.1,
    }


def main() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2 ** -0.5, 2 ** -0.5],
            [1.0, 0.0],
            [0.0, 1.0],
            [2 ** -0.5, 2 ** -0.5],
        ],
        dtype=np.float64,
    )
    gram = vectors @ vectors.T
    weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
    common = weights @ vectors
    direction = common / np.linalg.norm(common)
    if not np.all(vectors @ direction > 0.0):
        raise AssertionError("compatible task gradients lost common descent")
    opposite = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    opposite_weights = frank_wolfe_min_norm_weights(opposite @ opposite.T, FW_ITERATIONS)
    if np.linalg.norm(opposite_weights @ opposite) > 1e-10:
        raise AssertionError("incompatible task gradients did not collapse to zero")

    before = _panel(0.04)
    after = _panel(0.039)
    checks = candidate_checks(
        before=before,
        after=after,
        base_v5_dense=[0.1, 0.1, 0.1],
        candidate_v5_dense=[0.1, 0.1, 0.1],
        common_direction_valid=True,
    )
    if not all(checks.values()):
        raise AssertionError("admissible common-descent panel was rejected")
    bad = _panel(0.039)
    bad["per_task_exact_topk_swap"][1][2] = 0.041
    bad_checks = candidate_checks(
        before=before,
        after=bad,
        base_v5_dense=[0.1, 0.1, 0.1],
        candidate_v5_dense=[0.1, 0.1, 0.1],
        common_direction_valid=True,
    )
    if bad_checks["all_six_same_panel_swap_losses_decrease"] is not False:
        raise AssertionError("one regressing task did not fail closed")

    rows = []
    for index, radius in enumerate(STEP_RADII):
        row_after = _panel(0.039 - index * 0.0001)
        rows.append(
            {
                "name": "common_descent_radius_" + str(radius).replace(".", "p"),
                "step_radius": radius,
                "before": before,
                "after": row_after,
                "eligible": index in (1, 2),
            }
        )
    order = rank_candidates(rows)
    if order != [rows[2]["name"], rows[1]["name"]]:
        raise AssertionError("common-descent candidate ranking changed")
    if set(EXPECTED_V93_FAILURES) != {
        "exact_swap_lr010", "exact_swap_lr020", "exact_swap_lr040"
    }:
        raise AssertionError("v9.3 failure binding inventory changed")

    runner = Path(__file__).with_name("preflight_relational_teacher_v94_common_descent.py")
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(calls) != 1 or "torch.save" in source:
        raise AssertionError("array-load/checkpoint static guard changed")
    print("[PASS] Teacher-v9.4 six-task common-descent CPU contract")
    print("[PASS] minimum-norm geometry, strict six-task gate and ranking")
    print("[PASS] one train-scene load and no checkpoint writer")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU contract tests for Teacher-v9.4.1 metric-first replication."""

from __future__ import annotations

import ast
from pathlib import Path

from relational_teacher_v941_metric_replication_contract import (
    OBJECTS,
    REPLICATION_PANELS,
    SELECTED_RADIUS,
    replication_checks,
)


def _row(topk: float) -> dict:
    prompts = []
    for _ in range(2):
        prompts.append(
            {
                "instances": {
                    name: {
                        "topk_overlap": topk,
                        "soft_recall": 0.8,
                        "active_support_mae": 0.1,
                    }
                    for name in OBJECTS
                },
                "explicit_negative_mean": 0.01,
                "explicit_negative_max": 0.1,
            }
        )
    return {
        "per_prompt_metrics": prompts,
        "prompt_invariance": 0.01,
        "background_trust": 1e-6,
        "negative_mean": 0.01,
        "negative_max": 0.1,
    }


def main() -> None:
    if SELECTED_RADIUS != 0.01 or len(REPLICATION_PANELS) != 3:
        raise AssertionError("sealed replication protocol changed")
    base = [_row(0.3) for _ in range(3)]
    candidate = [_row(0.31) for _ in range(3)]
    checks = replication_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_v5_dense=[0.1, 0.1, 0.1],
        candidate_v5_dense=[0.1, 0.1, 0.1],
    )
    if not all(checks.values()):
        raise AssertionError("valid metric replication was rejected")
    bad = [_row(0.31) for _ in range(3)]
    bad[1]["per_prompt_metrics"][0]["instances"]["chair_06"]["topk_overlap"] = 0.29
    bad_checks = replication_checks(
        base_rows=base,
        candidate_rows=bad,
        base_v5_dense=[0.1, 0.1, 0.1],
        candidate_v5_dense=[0.1, 0.1, 0.1],
    )
    if bad_checks["high_chair_every_case_topk_not_worse"] is not False:
        raise AssertionError("one High-Chair regression did not fail closed")
    runner = Path(__file__).with_name("evaluate_relational_teacher_v941_metric_replication.py")
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(loads) != 1 or "torch.save" in source:
        raise AssertionError("array-load/checkpoint static guard changed")
    print("[PASS] Teacher-v9.4.1 metric-first replication CPU contract")
    print("[PASS] all-object pooled gain, per-case non-regression and retention gates")
    print("[PASS] one train-scene load and no checkpoint writer")


if __name__ == "__main__":
    main()


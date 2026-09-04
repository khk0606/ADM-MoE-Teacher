#!/usr/bin/env python3
"""CPU-only contract tests for Teacher-v9.6 object-balanced consensus."""

from __future__ import annotations

import ast
import copy
import sys
import types
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from relational_teacher_v96_object_balanced_consensus_contract import (
    AUDIT_PANELS,
    DESIGN_PANELS,
    DIRECTION_NAMES,
    OBJECTS,
    STEP_RADII,
    build_direction_specs,
    candidate_checks,
    rank_candidates,
)


ROOT = Path(__file__).resolve().parent


def _labels() -> list[str]:
    return [
        f"{panel['name']}|{prompt}|{name}"
        for panel in DESIGN_PANELS for prompt in ("watch", "write") for name in OBJECTS
    ]


def _test_direction_construction() -> None:
    labels = _labels()
    gradients = []
    object_offsets = {
        "bed_01": np.asarray([0.0, 0.35, -0.05]),
        "chair_01": np.asarray([0.0, -0.2, 0.3]),
        "chair_06": np.asarray([0.0, -0.1, -0.3]),
    }
    for index, label in enumerate(labels):
        name = label.rsplit("|", 1)[1]
        panel_phase = ((index // 6) - 3.5) * 0.025
        prompt_phase = 0.03 if "|watch|" in label else -0.03
        row = np.asarray([1.0, panel_phase, prompt_phase]) + object_offsets[name]
        row /= np.linalg.norm(row)
        gradients.append(row)
    matrix = np.asarray(gradients) @ np.asarray(gradients).T
    specs = build_direction_specs(matrix, labels)
    assert tuple(row["name"] for row in specs) == DIRECTION_NAMES
    assert all(row["valid"] is True for row in specs)
    assert all(len(row["task_coefficients"]) == 48 for row in specs)
    assert all(float(row["minimum_task_directional_derivative"]) > 0.0 for row in specs)
    assert any(float(row["blend_alpha"]) > 0.0 for row in specs[1:])


def _row(topk: float = 0.2) -> dict[str, object]:
    prompts = []
    for _ in range(2):
        prompts.append(
            {
                "instances": {
                    name: {
                        "topk_overlap": topk,
                        "soft_recall": 0.7,
                        "active_support_mae": 0.2,
                    }
                    for name in OBJECTS
                }
            }
        )
    return {
        "per_prompt_metrics": prompts,
        "prompt_invariance": 0.01,
        "background_trust": 0.0,
        "negative_mean": 0.02,
        "negative_max": 0.2,
    }


def _candidate_rows(gain: float) -> list[dict[str, object]]:
    rows = [_row() for _ in AUDIT_PANELS]
    for panel in (0, 1):
        for prompt in range(2):
            for name in OBJECTS:
                rows[panel]["per_prompt_metrics"][prompt]["instances"][name]["topk_overlap"] += gain
    for row in rows:
        row["background_trust"] = 1e-7
    return rows


def _test_metrics_and_regression() -> None:
    base = [_row() for _ in AUDIT_PANELS]
    candidate = _candidate_rows(0.01)
    checks = candidate_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_v5_dense=[0.05, 0.06, 0.07],
        candidate_v5_dense=[0.0501, 0.0601, 0.0701],
        direction_valid=True,
    )
    assert checks and all(checks.values())
    regressed = copy.deepcopy(candidate)
    regressed[3]["per_prompt_metrics"][0]["instances"]["chair_06"]["topk_overlap"] = 0.19
    failed = candidate_checks(
        base_rows=base,
        candidate_rows=regressed,
        base_v5_dense=[0.05, 0.06, 0.07],
        candidate_v5_dense=[0.0501, 0.0601, 0.0701],
        direction_valid=True,
    )
    assert failed["high_chair_every_case_topk_not_worse"] is False


def _test_ranking_and_grid() -> None:
    base = [_row() for _ in AUDIT_PANELS]
    rows = []
    for direction in DIRECTION_NAMES:
        for radius in STEP_RADII:
            name = direction + "_radius_" + str(radius).replace(".", "p")
            gain = 0.01
            eligible = False
            if direction == "object_balanced" and radius == 0.001:
                gain, eligible = 0.015, True
            if direction == "high_chair_priority" and radius == 0.0015:
                gain, eligible = 0.02, True
            rows.append(
                {
                    "name": name,
                    "direction_name": direction,
                    "step_radius": radius,
                    "base_rows": base,
                    "candidate_rows": _candidate_rows(gain),
                    "eligible": eligible,
                }
            )
    order = rank_candidates(rows)
    assert order == [
        "high_chair_priority_radius_0p0015",
        "object_balanced_radius_0p001",
    ]


def _test_fail_closed_runner() -> None:
    runner = ROOT / "preflight_relational_teacher_v96_object_balanced_consensus.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    assert len(loads) == 1
    assert "torch.save" not in source
    assert "serialized_model_state\": False" in source
    assert "development_arrays_read\": False" in source
    assert "authorizes_balanced_response6" in source


def main() -> None:
    _test_direction_construction()
    _test_metrics_and_regression()
    _test_ranking_and_grid()
    _test_fail_closed_runner()
    print("[PASS] Teacher-v9.6 48-task object-balanced direction contract")
    print("[PASS] exact 4x4 candidate grid, ranking and single-case rejection")
    print("[PASS] one train-scene load and no checkpoint writer")


if __name__ == "__main__":
    main()

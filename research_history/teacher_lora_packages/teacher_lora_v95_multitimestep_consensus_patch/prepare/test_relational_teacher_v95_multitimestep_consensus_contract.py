#!/usr/bin/env python3
"""CPU-only contract tests for the Teacher-v9.5 consensus gate."""

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
    # The pure Frank-Wolfe helper does not touch torch. This keeps the contract
    # runnable in lightweight packaging environments while Ubuntu uses real torch.
    sys.modules["torch"] = types.ModuleType("torch")

from relational_teacher_v94_common_descent import frank_wolfe_min_norm_weights
from relational_teacher_v95_multitimestep_consensus_contract import (
    AUDIT_PANELS,
    STEP_RADII,
    consensus_candidate_checks,
    rank_candidates,
)


ROOT = Path(__file__).resolve().parent


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
                    for name in ("bed_01", "chair_01", "chair_06")
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
            for name in ("bed_01", "chair_01", "chair_06"):
                rows[panel]["per_prompt_metrics"][prompt]["instances"][name]["topk_overlap"] += gain
    for row in rows:
        row["background_trust"] = 1e-7
    return rows


def _test_metric_contract() -> None:
    base = [_row() for _ in AUDIT_PANELS]
    candidate = _candidate_rows(0.01)
    checks = consensus_candidate_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_v5_dense=[0.05, 0.06, 0.07],
        candidate_v5_dense=[0.0501, 0.0601, 0.0701],
        common_direction_valid=True,
    )
    assert checks and all(checks.values())

    regressed = copy.deepcopy(candidate)
    regressed[3]["per_prompt_metrics"][0]["instances"]["bed_01"]["topk_overlap"] = 0.19
    failed = consensus_candidate_checks(
        base_rows=base,
        candidate_rows=regressed,
        base_v5_dense=[0.05, 0.06, 0.07],
        candidate_v5_dense=[0.0501, 0.0601, 0.0701],
        common_direction_valid=True,
    )
    assert failed["bed_every_case_topk_not_worse"] is False


def _test_ranking() -> None:
    base = [_row() for _ in AUDIT_PANELS]
    rows = []
    for radius, gain, eligible in zip(STEP_RADII, (0.01, 0.02, 0.02), (True, True, True)):
        rows.append(
            {
                "name": "r" + str(radius),
                "step_radius": radius,
                "base_rows": base,
                "candidate_rows": _candidate_rows(gain),
                "eligible": eligible,
            }
        )
    assert rank_candidates(rows) == ["r0.003", "r0.01", "r0.001"]


def _test_common_direction_geometry() -> None:
    # Four groups pull in different directions but retain a nonzero common
    # component. The minimum-norm convex combination must descend all tasks.
    four_gradients = np.asarray(
        [
            [1.0, 0.8, 0.1],
            [1.0, -0.7, 0.2],
            [1.0, 0.1, -0.8],
            [1.0, -0.2, 0.7],
        ],
        dtype=np.float64,
    )
    gradients = np.repeat(four_gradients, 6, axis=0)
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
    assert gradients.shape == (24, 3)
    gram = gradients @ gradients.T
    weights = frank_wolfe_min_norm_weights(gram, 1024)
    direction = weights @ gradients
    direction /= np.linalg.norm(direction)
    derivatives = gradients @ direction
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert float(derivatives.min()) > 0.0


def _test_fail_closed_runner_shape() -> None:
    runner = ROOT / "preflight_relational_teacher_v95_multitimestep_consensus.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    assert len(loads) == 1
    assert "torch.save" not in source
    assert "room_0201" in source
    assert "serialized_model_state\": False" in source


def main() -> None:
    _test_metric_contract()
    _test_ranking()
    _test_common_direction_geometry()
    _test_fail_closed_runner_shape()
    print("[PASS] Teacher-v9.5 multi-timestep consensus metric contract")
    print("[PASS] 24-task geometry, strict ranking and regression rejection")
    print("[PASS] one train-scene load and no checkpoint writer")


if __name__ == "__main__":
    main()

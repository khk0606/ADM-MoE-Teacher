#!/usr/bin/env python3
"""CPU-only contracts for deterministic MoE labels and anti-collapse loss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from prepare.train_moe_iiw_planner import (  # noqa: E402
    balanced_state_mode_labels,
    router_anti_collapse_objective,
    routing_metrics,
)


def test_balanced_state_modes() -> None:
    states = np.asarray(
        [
            [-1.1, -1.0, 0.9, 0.1],
            [-0.9, -1.1, 1.0, 0.0],
            [-1.0, -0.8, 0.8, 0.2],
            [1.0, 0.8, -0.9, 0.1],
            [0.9, 1.1, -1.0, 0.0],
            [1.2, 1.0, -0.8, 0.2],
        ],
        dtype=np.float32,
    )
    first, metadata = balanced_state_mode_labels(states, 2)
    second, repeated_metadata = balanced_state_mode_labels(states.copy(), 2)
    if not np.array_equal(first, second):
        raise AssertionError("state-mode clustering is not deterministic")
    if np.bincount(first, minlength=2).tolist() != [3, 3]:
        raise AssertionError("state modes are not balanced 3/3")
    if first[:3].tolist() != [0, 0, 0] or first[3:].tolist() != [1, 1, 1]:
        raise AssertionError("obvious two-mode state geometry was not recovered")
    if metadata != repeated_metadata:
        raise AssertionError("state-mode metadata is not deterministic")
    if metadata["uses_gt_motion"] or metadata["uses_gt_iiw"]:
        raise AssertionError("state-mode construction claims target leakage")
    print("[PASS] deterministic balanced 3/3 state-mode labels")


def test_multiscene_mode_metadata_is_json_serializable() -> None:
    positions = np.linspace(-2.0, 2.0, 23, dtype=np.float32)
    states = np.stack(
        [
            positions,
            np.sin(positions),
            np.cos(positions),
            np.square(positions),
        ],
        axis=1,
    )
    labels, metadata = balanced_state_mode_labels(states, 2)
    counts = np.bincount(labels, minlength=2).tolist()
    if sorted(counts) != [11, 12]:
        raise AssertionError("23-sample state modes are not balanced 12/11")
    json.dumps(
        {
            "state_mode_metadata": metadata,
            "mode_labels_in_sample_order": labels.tolist(),
        }
    )
    if not all(
        isinstance(index, int)
        for group in metadata["groups"]
        for index in group
    ):
        raise AssertionError("state-mode groups retain NumPy integer scalars")
    print("[PASS] 23-sample state-mode metadata is JSON serializable")


def test_router_anti_collapse_gradient() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    logits = torch.zeros(6, 2, requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)
    terms, diagnostics = router_anti_collapse_objective(
        logits, probabilities, labels, margin=0.2
    )
    total = (
        terms["router_supervised_ce"]
        + terms["router_load_balance"]
        + 0.01 * terms["router_confidence_entropy"]
        + terms["router_separation_margin"]
    )
    total.backward()
    if logits.grad is None or float(logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("anti-collapse loss gives router no gradient")
    if not torch.allclose(
        diagnostics["soft_load"], torch.tensor([0.5, 0.5]), atol=1e-7
    ):
        raise AssertionError("uniform full-batch load was computed incorrectly")
    if not torch.isfinite(total):
        raise AssertionError("anti-collapse total is non-finite")
    print("[PASS] collapsed router receives finite specialization gradient")


def test_perfect_routing_metrics() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    logits = torch.tensor(
        [[5.0, -5.0]] * 3 + [[-5.0, 5.0]] * 3,
        dtype=torch.float32,
    )
    probabilities = torch.softmax(logits, dim=-1)
    metrics = routing_metrics(logits, probabilities, labels)
    if metrics["accuracy"] != 1.0:
        raise AssertionError("perfect router accuracy contract failed")
    if metrics["hard_counts"] != [3, 3]:
        raise AssertionError("perfect router hard counts are not 3/3")
    if metrics["min_assigned_probability"] <= 0.99:
        raise AssertionError("perfect router assigned probability is too small")
    if metrics["confusion_expected_rows_predicted_columns"] != [[3, 0], [0, 3]]:
        raise AssertionError("router confusion matrix contract failed")
    print("[PASS] router use/accuracy metrics expose both experts")


def test_invalid_unbalanced_labels_rejected() -> None:
    logits = torch.zeros(6, 2)
    probabilities = torch.softmax(logits, dim=-1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    try:
        router_anti_collapse_objective(logits, probabilities, labels, 0.2)
    except AssertionError:
        print("[PASS] unbalanced full-batch pseudo-labels are rejected")
        return
    raise AssertionError("unbalanced pseudo-labels were silently accepted")


def main() -> None:
    test_balanced_state_modes()
    test_multiscene_mode_metadata_is_json_serializable()
    test_router_anti_collapse_gradient()
    test_perfect_routing_metrics()
    test_invalid_unbalanced_labels_rejected()
    print("[PASS] MoE IIW training-side unit contract")


if __name__ == "__main__":
    main()

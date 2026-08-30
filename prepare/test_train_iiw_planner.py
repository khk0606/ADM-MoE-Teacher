#!/usr/bin/env python3
"""Lightweight loss/metric tests for ``train_iiw_planner.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from prepare.train_iiw_planner import (  # noqa: E402
    sample_retrieval_metrics,
    mean_within_scene_prediction_delta,
    sparse_objective,
    tensor_metrics,
)


def test_soft_target_optimum() -> None:
    torch.manual_seed(31)
    target = torch.rand(2, 4, 23, 6).clamp(0.02, 0.98)
    logits = torch.logit(target).requires_grad_(True)
    body_weight = torch.tensor([2.0, 4.0, 8.0, 3.0, 5.0, 7.0])
    loss, parts = sparse_objective(
        logits,
        target,
        body_weight,
        {
            "bce": 1.0,
            "foreground": 0.0,
            "background": 0.0,
            "dice": 0.0,
            "temporal": 0.0,
        },
    )
    loss.backward()
    if logits.grad is None or float(logits.grad.abs().max()) > 1e-7:
        raise AssertionError("weighted soft BCE shifted the p=target optimum")
    if not torch.isfinite(parts["bce"]):
        raise AssertionError("soft BCE is non-finite")
    print("[PASS] sparse BCE preserves the continuous IIW optimum")


def test_sparse_gradient_and_metrics() -> None:
    target = torch.zeros(2, 3, 17, 6)
    target[0, :, 2:6, 0] = torch.tensor([0.4, 0.8, 1.0])[:, None]
    target[1, :, 9:14, 4] = torch.tensor([1.0, 0.8, 0.4])[:, None]
    logits = torch.full_like(target, -3.0, requires_grad=True)
    loss, _ = sparse_objective(
        logits,
        target,
        torch.full((6,), 20.0),
        {
            "bce": 1.0,
            "foreground": 0.5,
            "background": 0.25,
            "dice": 0.5,
            "temporal": 0.5,
        },
    )
    loss.backward()
    active = target > 0.0
    if float(logits.grad[active].abs().sum()) <= 0.0:
        raise AssertionError("active IIW positions received no gradient")
    metrics = tensor_metrics(target, target, 0.7)
    if metrics["f1_at_0_7"] != 1.0 or metrics["mae"] != 0.0:
        raise AssertionError("perfect-prediction metric contract failed")
    retrieval = sample_retrieval_metrics(target, target)
    if retrieval["sample_retrieval_accuracy"] != 1.0:
        raise AssertionError("diagonal sample retrieval contract failed")
    print("[PASS] sparse active regions receive gradients and metrics are exact")


def test_multi_scene_retrieval_is_point_order_safe() -> None:
    first = torch.zeros(2, 2, 5, 6)
    first[0, :, 0, 0] = 1.0
    first[1, :, 1, 0] = 1.0
    second = torch.zeros(2, 2, 5, 6)
    second[0, :, 3, 0] = 1.0
    second[1, :, 4, 0] = 1.0
    target = torch.cat([first, second], dim=0)
    scene_ids = ["room_0001", "room_0001", "room_0002", "room_0002"]
    metrics = sample_retrieval_metrics(target, target, scene_ids)
    if metrics["sample_retrieval_accuracy"] != 1.0:
        raise AssertionError("scene-aware self retrieval failed")
    delta = mean_within_scene_prediction_delta(target, scene_ids)
    if not delta > 0.0:
        raise AssertionError("within-scene state delta was not detected")
    print("[PASS] multi-scene retrieval never compares unrelated point orders")


def main() -> None:
    test_soft_target_optimum()
    test_sparse_gradient_and_metrics()
    test_multi_scene_retrieval_is_point_order_safe()
    print("[PASS] IIWPlanner training objective unit contract")


if __name__ == "__main__":
    main()

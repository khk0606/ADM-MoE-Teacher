#!/usr/bin/env python3
"""Deterministic unit contract for ``models/iiw_planner.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PATCH_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PATCH_ROOT / "models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from iiw_planner import IIWPlanner, NATIVE_BODY_PART_NAMES  # noqa: E402


def make_planner() -> IIWPlanner:
    return IIWPlanner(
        scene_dim=6,
        text_dim=12,
        state_dim=4,
        num_phases=8,
        num_bodies=6,
        hidden_dim=32,
        point_dim=24,
        context_dim=28,
    )


def make_inputs(batch_size: int = 2, num_points: int = 19):
    torch.manual_seed(20260813)
    text_features = torch.randn(batch_size, 12)
    scene_points = torch.randn(batch_size, num_points, 6)
    scene_points[..., 3:] = torch.sigmoid(scene_points[..., 3:])
    state = torch.randn(batch_size, 4)
    state[:, 2:4] = F.normalize(state[:, 2:4], dim=-1)
    return scene_points, text_features, state


def test_shape_range_and_body_order() -> None:
    assert NATIVE_BODY_PART_NAMES == (
        "base",
        "spine",
        "right_hand",
        "left_hand",
        "right_foot",
        "left_foot",
    )
    planner = make_planner().eval()
    inputs = make_inputs()
    with torch.no_grad():
        logits = planner.forward_logits(*inputs)
        probabilities = planner(*inputs)
    assert logits.shape == (2, 8, 19, 6)
    assert probabilities.shape == logits.shape
    assert torch.isfinite(probabilities).all()
    assert float(probabilities.min()) >= 0.0
    assert float(probabilities.max()) <= 1.0
    torch.testing.assert_close(probabilities, torch.sigmoid(logits))
    assert planner.config["num_phases"] == 8
    assert planner.config["num_bodies"] == 6
    print("[PASS] IIWPlanner output shape, range, body order, and config")


def test_point_permutation_equivariance() -> None:
    planner = make_planner().eval()
    scene_points, text_features, state = make_inputs()
    permutation = torch.tensor(
        [8, 1, 14, 0, 5, 18, 7, 2, 12, 3, 16, 4, 6, 10, 9, 11, 13, 15, 17]
    )
    with torch.no_grad():
        original = planner(scene_points, text_features, state)
        permuted = planner(scene_points[:, permutation], text_features, state)
    torch.testing.assert_close(
        permuted,
        original[:, :, permutation],
        atol=2e-6,
        rtol=2e-6,
    )
    print("[PASS] scene point permutation preserves exact IIW alignment")


def test_conditioning_and_gradients() -> None:
    planner = make_planner()
    scene_points, text_features, state = make_inputs()
    text_features.requires_grad_(True)
    scene_points.requires_grad_(True)
    state.requires_grad_(True)
    logits = planner.forward_logits(scene_points, text_features, state)
    weights = torch.linspace(0.1, 1.0, logits.numel()).reshape_as(logits)
    loss = (logits * weights).mean()
    loss.backward()
    for name, value in (
        ("scene_points", scene_points),
        ("text_features", text_features),
        ("state", state),
    ):
        if value.grad is None or float(value.grad.norm()) <= 0.0:
            raise AssertionError("{} received no gradient".format(name))

    with torch.no_grad():
        base = planner(scene_points, text_features, state)
        changed_text = planner(scene_points, text_features.flip(0), state)
        changed_state = planner(scene_points, text_features, state.flip(0))
    if float((base - changed_text).abs().max()) <= 1e-6:
        raise AssertionError("text conditioning has no effect")
    if float((base - changed_state).abs().max()) <= 1e-6:
        raise AssertionError("state conditioning has no effect")
    if float((base[:, 1:] - base[:, :-1]).abs().max()) <= 1e-6:
        raise AssertionError("phase queries collapsed to one prediction")
    print("[PASS] text, scene, state, and phase conditioning are differentiable")


def test_short_supervised_overfit() -> None:
    torch.manual_seed(77)
    planner = make_planner().train()
    scene_points, text_features, state = make_inputs(batch_size=2, num_points=13)

    # A deterministic sparse target with different point, phase, body, and
    # sample structure; this checks the supervised learning path, not quality.
    target = torch.zeros((2, 8, 13, 6))
    target[0, :, 2:5, 0] = torch.linspace(0.2, 1.0, 8)[:, None]
    target[0, 4:, 8:11, 3] = 0.9
    target[1, :, 6:10, 1] = torch.linspace(1.0, 0.2, 8)[:, None]
    target[1, :4, 0:3, 5] = 0.8

    optimizer = torch.optim.Adam(planner.parameters(), lr=2e-2)
    with torch.no_grad():
        initial = float(
            F.binary_cross_entropy_with_logits(
                planner.forward_logits(scene_points, text_features, state), target
            )
        )
    for _ in range(100):
        logits = planner.forward_logits(scene_points, text_features, state)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(
            F.binary_cross_entropy_with_logits(
                planner.forward_logits(scene_points, text_features, state), target
            )
        )
    if not final < initial * 0.45:
        raise AssertionError(
            "short supervised overfit was insufficient: {:.6f} -> {:.6f}".format(
                initial, final
            )
        )
    print(
        "[PASS] compact planner short supervised overfit: "
        "{:.6f} -> {:.6f}".format(initial, final)
    )


def expect_failure(label: str, function) -> None:
    try:
        function()
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid {} was accepted".format(label))


def test_strict_validation() -> None:
    planner = make_planner()
    scene_points, text_features, state = make_inputs()
    expect_failure(
        "text width",
        lambda: planner(scene_points, text_features[:, :-1], state),
    )
    expect_failure(
        "point width",
        lambda: planner(scene_points[..., :-1], text_features, state),
    )
    expect_failure(
        "state width",
        lambda: planner(scene_points, text_features, state[:, :-1]),
    )
    expect_failure(
        "batch size",
        lambda: planner(scene_points, text_features[:1], state),
    )
    bad_scene = scene_points.clone()
    bad_scene[0, 0, 0] = float("nan")
    expect_failure(
        "non-finite scene",
        lambda: planner(bad_scene, text_features, state),
    )
    expect_failure(
        "integer text",
        lambda: planner(scene_points, text_features.long(), state),
    )
    print("[PASS] invalid planner tensor contracts are rejected")


def main() -> None:
    test_shape_range_and_body_order()
    test_point_permutation_equivariance()
    test_conditioning_and_gradients()
    test_short_supervised_overfit()
    test_strict_validation()
    print("[PASS] IIWPlanner unit contract")


if __name__ == "__main__":
    main()

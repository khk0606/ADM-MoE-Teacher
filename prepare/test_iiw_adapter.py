#!/usr/bin/env python3
"""Deterministic unit contract for ``models/iiw_adapter.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PATCH_ROOT / "models"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from iiw_adapter import IIWAdapter, NATIVE_BODY_PART_NAMES  # noqa: E402


def make_inputs() -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(20260812)
    batch_size, num_phases, num_points, num_parts = 2, 8, 19, 6
    horizon = 14
    scene_xyz = torch.randn(batch_size, num_points, 3)
    iiw_plan = torch.rand(
        batch_size, num_phases, num_points, num_parts
    )

    frame_to_phase = torch.full(
        (batch_size, horizon), -1, dtype=torch.long
    )
    x_mask = torch.ones((batch_size, horizon), dtype=torch.bool)
    valid_lengths = (12, 9)
    for batch_index, valid_length in enumerate(valid_lengths):
        phase = torch.arange(valid_length, dtype=torch.long)
        phase = torch.div(
            phase * num_phases, valid_length, rounding_mode="floor"
        )
        frame_to_phase[batch_index, :valid_length] = phase
        x_mask[batch_index, :valid_length] = False
    # A valid clamped phase on padding is also permitted by the contract.
    frame_to_phase[0, valid_lengths[0]:] = num_phases - 1
    return scene_xyz, iiw_plan, frame_to_phase, x_mask


def make_adapter(zero_init: bool = True) -> IIWAdapter:
    return IIWAdapter(
        latent_dim=24,
        num_phases=8,
        num_body_parts=6,
        body_hidden_dim=12,
        hidden_dim=32,
        embedding_dim=5,
        zero_init=zero_init,
    )


def test_native_order_and_spatial_summary() -> None:
    assert NATIVE_BODY_PART_NAMES == (
        "base",
        "spine",
        "right_hand",
        "left_hand",
        "right_foot",
        "left_foot",
    )
    adapter = make_adapter()
    scene = torch.tensor([[[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]])
    plan = torch.zeros((1, 8, 2, 6))
    plan[0, 0, 0, 0] = 1.0
    summary = adapter.summarize(scene, plan)
    assert summary.shape == (1, 8, 6, 8)
    torch.testing.assert_close(summary[0, 0, 0, 0], torch.tensor(0.5))
    torch.testing.assert_close(summary[0, 0, 0, 1], torch.tensor(1.0))
    torch.testing.assert_close(
        summary[0, 0, 0, 2:5], torch.tensor([1.0, 2.0, 3.0])
    )
    torch.testing.assert_close(
        summary[0, 0, 0, 5:8], torch.zeros(3), atol=1e-5, rtol=0.0
    )
    assert torch.count_nonzero(summary[0, 1:]) == 0
    print("[PASS] native body order and spatial descriptor contract")


def test_zero_initialization_and_padding() -> None:
    inputs = make_inputs()
    adapter = make_adapter(zero_init=True)
    output = adapter(*inputs)
    x_mask = inputs[-1]
    assert output.shape == (2, 14, 24)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(output[x_mask]) == 0
    print("[PASS] zero-init preserves baseline and padding is exactly zero")


def test_zero_init_receives_gradient() -> None:
    inputs = make_inputs()
    adapter = make_adapter(zero_init=True)
    output = adapter(*inputs)
    valid = (~inputs[-1]).unsqueeze(-1).to(output.dtype)
    loss = ((output - valid) ** 2).sum()
    loss.backward()
    gradient = adapter.output_projection.weight.grad
    if gradient is None or float(gradient.norm().item()) <= 0.0:
        raise AssertionError("zero-initialized output projection got no gradient")
    print("[PASS] zero-init output projection receives nonzero gradient")


def test_trained_adapter_keeps_zero_plan_exactly_zero() -> None:
    scene_xyz, iiw_plan, frame_to_phase, x_mask = make_inputs()
    torch.manual_seed(20260813)
    adapter = make_adapter(zero_init=False)

    # Prove that the guarantee does not depend on zero-initialized biases.
    with torch.no_grad():
        adapter.output_projection.bias.fill_(0.37)
    zero_plan = torch.zeros_like(iiw_plan)
    zero_output = adapter(
        scene_xyz, zero_plan, frame_to_phase, x_mask
    )
    if torch.count_nonzero(zero_output) != 0:
        raise AssertionError("trained adapter changed the zero-IIW baseline")

    # The null anchor must not disconnect a future differentiable IIW planner.
    differentiable_plan = iiw_plan.clone().requires_grad_(True)
    output = adapter(
        scene_xyz, differentiable_plan, frame_to_phase, x_mask
    )
    loss = output[~x_mask].square().mean()
    loss.backward()
    gradient = differentiable_plan.grad
    if gradient is None or float(gradient.norm().item()) <= 0.0:
        raise AssertionError("null anchoring disconnected the IIW-plan gradient")
    print(
        "[PASS] trained adapter keeps zero IIW exact and nonzero IIW differentiable"
    )


def test_phase_order_affects_frame_tokens() -> None:
    scene_xyz, iiw_plan, frame_to_phase, x_mask = make_inputs()
    torch.manual_seed(31)
    adapter = make_adapter(zero_init=False)
    forward = adapter(scene_xyz, iiw_plan, frame_to_phase, x_mask)
    reverse = adapter(
        scene_xyz, iiw_plan.flip(1), frame_to_phase, x_mask
    )
    valid_difference = (forward - reverse)[~x_mask].abs().max()
    if float(valid_difference.item()) <= 1e-6:
        raise AssertionError("reversing IIW phases did not change valid tokens")
    assert torch.count_nonzero(forward[x_mask]) == 0
    assert torch.count_nonzero(reverse[x_mask]) == 0
    print("[PASS] temporal phase order changes valid frame conditioning")


def expect_failure(label: str, function) -> None:
    try:
        function()
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid {} was accepted".format(label))


def test_strict_validation() -> None:
    scene_xyz, iiw_plan, frame_to_phase, x_mask = make_inputs()
    adapter = make_adapter()

    bad_range = iiw_plan.clone()
    bad_range[0, 0, 0, 0] = 1.25
    expect_failure(
        "IIW range",
        lambda: adapter(scene_xyz, bad_range, frame_to_phase, x_mask),
    )

    bad_phase = frame_to_phase.clone()
    bad_phase[0, 0] = -1
    expect_failure(
        "valid-frame phase sentinel",
        lambda: adapter(scene_xyz, iiw_plan, bad_phase, x_mask),
    )

    bad_mask = x_mask.clone()
    bad_mask[0, 8] = True
    bad_mask[0, 9] = False
    expect_failure(
        "non-prefix padding mask",
        lambda: adapter(scene_xyz, iiw_plan, frame_to_phase, bad_mask),
    )

    bad_points = iiw_plan[:, :, :-1, :]
    expect_failure(
        "point count",
        lambda: adapter(scene_xyz, bad_points, frame_to_phase, x_mask),
    )

    bad_order = frame_to_phase.clone()
    bad_order[0, 5] = bad_order[0, 4] - 1
    expect_failure(
        "decreasing phase order",
        lambda: adapter(scene_xyz, iiw_plan, bad_order, x_mask),
    )
    print(
        "[PASS] invalid range, phase, order, mask, and point contracts are rejected"
    )


def main() -> None:
    test_native_order_and_spatial_summary()
    test_zero_initialization_and_padding()
    test_zero_init_receives_gradient()
    test_trained_adapter_keeps_zero_plan_exactly_zero()
    test_phase_order_affects_frame_tokens()
    test_strict_validation()
    print("[PASS] IIWAdapter unit contract")


if __name__ == "__main__":
    main()

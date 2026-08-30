#!/usr/bin/env python3
"""CPU unit contract for terminal-phase IIW -> cached ADM map*."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Type

import torch


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from models.iiw_map_router import IIWMapRouter  # noqa: E402


def expect_error(error_type: Type[BaseException], function: Callable[[], None]) -> None:
    try:
        function()
    except error_type:
        return
    raise AssertionError("expected {}".format(error_type.__name__))


def test_shapes_and_exact_multiply() -> None:
    torch.manual_seed(20260813)
    base = torch.rand(2, 13, 6)
    iiw = torch.rand(2, 8, 13, 6)
    output = IIWMapRouter()(base, iiw)

    assert output["pc_weight"].shape == (2, 13)
    assert output["mapstar"].shape == (2, 13, 6)
    assert output["phase_pc_weight"].shape == (2, 8, 13)
    assert output["phase_mapstar"].shape == (2, 8, 13, 6)

    expected_phase_weight = iiw.max(dim=-1)[0]
    expected_weight = expected_phase_weight[:, -1, :]
    expected_mapstar = base.detach() * expected_weight.unsqueeze(-1)
    expected_phase_maps = (
        base.detach().unsqueeze(1) * expected_phase_weight.unsqueeze(-1)
    )
    assert torch.equal(output["phase_pc_weight"], expected_phase_weight)
    assert torch.equal(output["pc_weight"], expected_weight)
    assert torch.equal(output["mapstar"], expected_mapstar)
    assert torch.equal(output["phase_mapstar"], expected_phase_maps)


def test_zero_and_one_invariants() -> None:
    base = torch.linspace(0.0, 1.0, 5 * 6).reshape(1, 5, 6)
    router = IIWMapRouter()

    zero = router(base, torch.zeros(1, 8, 5, 6))
    assert torch.equal(zero["pc_weight"], torch.zeros(1, 5))
    assert torch.equal(zero["mapstar"], torch.zeros_like(base))
    assert torch.equal(
        zero["phase_pc_weight"], torch.zeros(1, 8, 5)
    )
    assert torch.equal(
        zero["phase_mapstar"], torch.zeros(1, 8, 5, 6)
    )

    one = router(base, torch.ones(1, 8, 5, 6))
    assert torch.equal(one["pc_weight"], torch.ones(1, 5))
    assert torch.equal(one["mapstar"], base)
    assert torch.equal(one["phase_pc_weight"], torch.ones(1, 8, 5))
    assert torch.equal(
        one["phase_mapstar"], base.unsqueeze(1).expand(1, 8, 5, 6)
    )


def test_terminal_phase_order_sensitivity() -> None:
    base = torch.ones(1, 4, 6)
    iiw = torch.zeros(1, 3, 4, 6)
    iiw[:, 0, :, 2] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    iiw[:, 1, :, 2] = 0.5
    iiw[:, 2, :, 2] = torch.tensor([0.9, 0.8, 0.7, 0.6])
    router = IIWMapRouter()

    forward = router(base, iiw)
    reversed_output = router(base, iiw.flip(1))
    assert not torch.equal(
        forward["pc_weight"], reversed_output["pc_weight"]
    )
    assert torch.equal(
        forward["pc_weight"], iiw[:, -1].max(dim=-1)[0]
    )
    assert torch.equal(
        reversed_output["pc_weight"], iiw[:, 0].max(dim=-1)[0]
    )


def test_gradient_boundary() -> None:
    # Each point has one unique winning body channel, avoiding max ties so the
    # expected IIW gradient is unambiguous and strictly nonzero.
    base = torch.full((1, 7, 6), 0.75, requires_grad=True)
    raw = torch.full((1, 4, 7, 6), 0.1)
    raw[:, -1, :, 3] = 0.8
    iiw = raw.requires_grad_()

    output = IIWMapRouter()(base, iiw)
    output["mapstar"].sum().backward()
    assert base.grad is None
    assert iiw.grad is not None
    assert float(iiw.grad.abs().sum().item()) > 0.0
    assert float(iiw.grad[:, -1, :, 3].abs().sum().item()) > 0.0
    assert float(iiw.grad[:, :-1].abs().sum().item()) == 0.0


def test_strict_validation() -> None:
    router = IIWMapRouter()
    base = torch.rand(2, 5, 6)
    iiw = torch.rand(2, 8, 5, 6)

    expect_error(TypeError, lambda: router([base], iiw))
    expect_error(ValueError, lambda: router(base[:, :, :5], iiw))
    expect_error(ValueError, lambda: router(base, iiw[:, :, :, :5]))
    expect_error(ValueError, lambda: router(base, iiw[:, :, :4]))
    expect_error(TypeError, lambda: router(base.double(), iiw))
    expect_error(TypeError, lambda: router(base.to(torch.int64), iiw))

    below = iiw.clone()
    below[0, 0, 0, 0] = -1e-4
    expect_error(ValueError, lambda: router(base, below))
    above = iiw.clone()
    above[0, 0, 0, 0] = 1.0001
    expect_error(ValueError, lambda: router(base, above))
    nan_plan = iiw.clone()
    nan_plan[0, 0, 0, 0] = float("nan")
    expect_error(ValueError, lambda: router(base, nan_plan))
    inf_base = base.clone()
    inf_base[0, 0, 0] = float("inf")
    expect_error(ValueError, lambda: router(inf_base, iiw))


def main() -> None:
    test_shapes_and_exact_multiply()
    test_zero_and_one_invariants()
    test_terminal_phase_order_sensitivity()
    test_gradient_boundary()
    test_strict_validation()
    print("[PASS] terminal phase then max-native-body pc_weight reducer")
    print("[PASS] exact detached-base map* multiplication")
    print("[PASS] per-phase scalar weights and maps for visualization")
    print("[PASS] zero/one and temporal-order invariants")
    print("[PASS] frozen base gradient boundary and IIW gradient path")
    print("[PASS] strict map-router tensor contract")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unit contract for the non-invasive ``IIWConditionedCMDM`` wrapper."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from models.iiw_cmdm import IIWConditionedCMDM  # noqa: E402


class FakeCMDM(nn.Module):
    def __init__(self, fail_after_adapter: bool = False) -> None:
        super().__init__()
        self.motion_adapter = nn.Linear(3, 5, bias=False)
        self.output_layer = nn.Linear(5, 2, bias=False)
        self.fail_after_adapter = fail_after_adapter
        self.seen_kwargs = None
        with torch.no_grad():
            self.motion_adapter.weight.copy_(
                torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0
            )
            self.output_layer.weight.copy_(
                torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10.0
            )

    def forward(self, x, timesteps, **kwargs):
        del timesteps
        self.seen_kwargs = set(kwargs)
        value = self.motion_adapter(x)
        if self.fail_after_adapter:
            raise RuntimeError("deliberate downstream failure")
        return self.output_layer(value)


class SlowIdentityCMDM(nn.Module):
    """Expose overlapping hook lifetimes if the wrapper lock is removed."""

    def __init__(self) -> None:
        super().__init__()
        self.motion_adapter = nn.Identity()

    def forward(self, x, timesteps, **kwargs):
        del timesteps, kwargs
        value = self.motion_adapter(x)
        time.sleep(0.02)
        return value


def make_case():
    torch.manual_seed(20260812)
    x = torch.randn(2, 6, 3)
    timesteps = torch.tensor([100, 500], dtype=torch.long)
    mask = torch.tensor(
        [
            [False, False, False, False, True, True],
            [False, False, False, False, False, True],
        ],
        dtype=torch.bool,
    )
    residual = torch.randn(2, 6, 5)
    residual = residual.masked_fill(mask.unsqueeze(-1), 0.0)
    return x, timesteps, mask, residual


def test_no_key_exact_parity() -> None:
    x, timesteps, mask, _ = make_case()
    base = FakeCMDM()
    wrapper = IIWConditionedCMDM(base)
    direct = base(x, timesteps, x_mask=mask, c_text=["a", "b"])
    wrapped = wrapper(x, timesteps, x_mask=mask, c_text=["a", "b"])
    assert torch.equal(direct, wrapped)
    assert len(base.motion_adapter._forward_hooks) == 0
    print("[PASS] no-key wrapper call has exact legacy parity")


def test_injection_and_gradient() -> None:
    x, timesteps, mask, residual = make_case()
    base = FakeCMDM()
    wrapper = IIWConditionedCMDM(base)
    residual.requires_grad_(True)
    expected = base.output_layer(base.motion_adapter(x) + residual)
    actual = wrapper(
        x,
        timesteps,
        x_mask=mask,
        c_text=["a", "b"],
        c_iiw_residual=residual,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    if "c_iiw_residual" in base.seen_kwargs:
        raise AssertionError("new keyword leaked into the unmodified CMDM")
    actual.sum().backward()
    if residual.grad is None or float(residual.grad.norm().item()) <= 0.0:
        raise AssertionError("gradient did not reach c_iiw_residual")
    assert len(base.motion_adapter._forward_hooks) == 0
    print("[PASS] residual is injected once and receives gradient")


def test_validation_and_exception_cleanup() -> None:
    x, timesteps, mask, residual = make_case()
    base = FakeCMDM()
    wrapper = IIWConditionedCMDM(base)

    invalid = residual.clone()
    invalid[0, -1, 0] = 1e-12
    try:
        wrapper(
            x,
            timesteps,
            x_mask=mask,
            c_iiw_residual=invalid,
        )
    except ValueError as error:
        assert "exactly zero" in str(error)
    else:
        raise AssertionError("nonzero padded residual was accepted")
    assert len(base.motion_adapter._forward_hooks) == 0

    failing = FakeCMDM(fail_after_adapter=True)
    failing_wrapper = IIWConditionedCMDM(failing)
    try:
        failing_wrapper(
            x,
            timesteps,
            x_mask=mask,
            c_iiw_residual=residual,
        )
    except RuntimeError as error:
        assert "deliberate" in str(error)
    else:
        raise AssertionError("expected downstream exception")
    assert len(failing.motion_adapter._forward_hooks) == 0
    print("[PASS] validation and downstream exceptions remove the hook")


def test_shared_wrapper_serializes_hooks() -> None:
    base = SlowIdentityCMDM()
    wrapper = IIWConditionedCMDM(base)
    mask = torch.zeros((1, 3), dtype=torch.bool)
    barrier = threading.Barrier(2)

    def run(value: float):
        x = torch.zeros((1, 3, 4))
        residual = torch.full_like(x, value)
        barrier.wait()
        return wrapper(
            x,
            torch.tensor([1]),
            x_mask=mask,
            c_iiw_residual=residual,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, 1.0)
        second = executor.submit(run, 2.0)
        first_value = first.result()
        second_value = second.result()
    assert torch.equal(first_value, torch.ones_like(first_value))
    assert torch.equal(second_value, torch.full_like(second_value, 2.0))
    assert len(base.motion_adapter._forward_hooks) == 0
    print("[PASS] shared-wrapper concurrent calls cannot stack hooks")


def main() -> None:
    test_no_key_exact_parity()
    test_injection_and_gradient()
    test_validation_and_exception_cleanup()
    test_shared_wrapper_serializes_hooks()
    print("[PASS] non-invasive IIW CMDM wrapper contract")


if __name__ == "__main__":
    main()

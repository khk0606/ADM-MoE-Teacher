#!/usr/bin/env python3
"""CPU unit tests for the target-instance map* ceiling evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare.evaluate_oracle_instance_mapstar import (  # noqa: E402
    build_instance_oracle_contacts,
    paired_summary,
)


def fixture():
    base = torch.linspace(0.01, 0.99, 2 * 7 * 6).reshape(2, 7, 6)
    masks = torch.zeros(2, 7, 3, dtype=torch.bool)
    masks[0, 0:2, 0] = True
    masks[0, 2:5, 1] = True
    masks[0, 5:7, 2] = True
    masks[1, 0:3, 0] = True
    masks[1, 3:5, 1] = True
    masks[1, 5:7, 2] = True
    target = torch.tensor([1, 2], dtype=torch.long)
    return base, masks, target


def test_exact_instance_mask_and_multiplication() -> None:
    base, masks, target = fixture()
    output = build_instance_oracle_contacts(base, masks, target)
    expected_weight = torch.stack([masks[0, :, 1], masks[1, :, 2]]).float()
    assert torch.equal(output["pc_weight"], expected_weight)
    expected_map = base * expected_weight.unsqueeze(-1)
    assert torch.equal(output["oracle_instance"], expected_map)
    assert torch.equal(output["zero_weight"], torch.zeros_like(base))
    assert torch.equal(output["identity_weight"], base)
    assert torch.equal(output["legacy_base"], base)
    print("[PASS] exact target-instance binary pc_weight and map* multiplication")


def test_base_is_detached_and_non_target_is_zero() -> None:
    base, masks, target = fixture()
    base.requires_grad_(True)
    output = build_instance_oracle_contacts(base, masks, target)
    assert not output["oracle_instance"].requires_grad
    outside = output["oracle_instance"] * (
        1.0 - output["pc_weight"].unsqueeze(-1)
    )
    assert torch.equal(outside, torch.zeros_like(outside))
    print("[PASS] cached Base ADM is detached and every non-target point is zero")


def test_invalid_target_and_overlapping_masks_are_rejected() -> None:
    base, masks, target = fixture()
    invalid = target.clone()
    invalid[0] = 3
    try:
        build_instance_oracle_contacts(base, masks, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid target index was accepted")
    overlapping = masks.clone()
    overlapping[0, 0, 1] = True
    try:
        build_instance_oracle_contacts(base, overlapping, target)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping candidate masks were accepted")
    print("[PASS] invalid targets and overlapping masks are rejected")


def test_paired_summary_is_exact() -> None:
    losses = {
        "oracle_instance": [1.0, 2.0],
        "zero_weight": [2.0, 3.0],
        "identity_weight": [4.0, 5.0],
        "legacy_base": [4.0, 5.0],
    }
    result = paired_summary(losses, ["sample"], [0, 1])
    assert result["motion_mean_loss"]["oracle_instance"] == 1.5
    assert result["motion_mean_loss"]["zero_weight"] == 2.5
    assert result["oracle_minus_zero"] == -1.0
    assert result["oracle_beats_zero"] is True
    assert result["oracle_relative_improvement_vs_zero"] == 0.4
    assert result["oracle_pair_win_count_vs_zero"] == 2
    assert result["oracle_pair_count_vs_zero"] == 2
    assert result["identity_legacy_max_abs_diff"] == 0.0
    print("[PASS] paired fixed-noise loss summary")


def main() -> None:
    test_exact_instance_mask_and_multiplication()
    test_base_is_detached_and_non_target_is_zero()
    test_invalid_target_and_overlapping_masks_are_rejected()
    test_paired_summary_is_exact()
    print("[PASS] oracle target-instance map* unit contract")


if __name__ == "__main__":
    main()

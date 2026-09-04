#!/usr/bin/env python3
"""CPU contract for Teacher-v9.6 affordance viewer helpers."""

from __future__ import annotations

import numpy as np

from relational_teacher_v96_affordance_viewer_common import (
    affordance_colors,
    build_xy_interpolation_plan,
    interpolate_xy_heatmap,
    scalar_channel,
    signed_difference_colors,
    topk_change_masks,
)


def main() -> None:
    values = np.zeros((12, 6), dtype=np.float32)
    values[:, 0] = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    values[:, 5] = np.float32(0.25)
    assert np.array_equal(scalar_channel(values, "pelvis"), values[:, 0])
    assert np.array_equal(scalar_channel(values, "any_joint"), values.max(axis=-1))
    assert affordance_colors(values[:, 0]).shape == (12, 3)
    signed = signed_difference_colors(np.asarray((-0.5, 0.0, 0.5), np.float32), 0.5)
    assert tuple(signed[0]) != tuple(signed[2])
    assert tuple(signed[1]) == (35, 35, 45)

    gx, gy = np.meshgrid(
        np.linspace(-1.0, 1.0, 4, dtype=np.float32),
        np.linspace(-1.0, 1.0, 3, dtype=np.float32),
    )
    xyz = np.stack((gx.reshape(-1), gy.reshape(-1), np.zeros(12, np.float32)), axis=-1)
    plan = build_xy_interpolation_plan(xyz, resolution=32, neighbors=3)
    constant = interpolate_xy_heatmap(np.full(12, 0.375, np.float32), plan)
    assert constant.shape == (32, 32)
    assert np.allclose(constant, 0.375, rtol=0.0, atol=2e-6)

    target = np.zeros((12, 6), dtype=np.float32)
    target[:8, 0] = np.linspace(1.0, 0.3, 8, dtype=np.float32)
    mask = np.ones(12, dtype=bool)
    base = np.zeros_like(target)
    candidate = np.zeros_like(target)
    base[[0, 2], 0] = (0.9, 0.8)
    candidate[[0, 1], 0] = (0.9, 0.85)
    changes = topk_change_masks(base, candidate, target, mask)
    assert changes["left"].sum() == 1 and changes["left"][2]
    assert changes["entered"].sum() == 1 and changes["entered"][1]
    print("[PASS] Teacher-v9.6 viewer CPU contract")
    print("[PASS] fixed palette, signed difference, XY interpolation and exact Top-k swaps")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU contract for the Teacher-v9.8.12 Viser rendering helpers."""

from __future__ import annotations

import numpy as np

from relational_teacher_v9812_affordance_viewer_common import (
    affordance_colors,
    build_xy_interpolation_plan,
    interpolate_xy_heatmap,
    rgba_heatmap,
    scalar_channel,
    signed_difference_colors,
)


def main() -> None:
    values = np.zeros((12, 6), dtype=np.float32)
    values[:, 0] = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    values[:, 5] = np.float32(0.25)
    assert np.array_equal(scalar_channel(values, "pelvis"), values[:, 0])
    assert np.array_equal(scalar_channel(values, "any_joint"), values.max(axis=-1))

    physical = affordance_colors(np.asarray((0.0, 0.5, 1.0), np.float32))
    assert physical.shape == (3, 3)
    assert tuple(physical[0]) == (20, 10, 45)
    assert tuple(physical[-1]) == (190, 25, 20)
    signed = signed_difference_colors(
        np.asarray((-0.5, 0.0, 0.5), np.float32), 0.5
    )
    assert tuple(signed[0]) != tuple(signed[2])
    assert tuple(signed[1]) == (35, 35, 45)

    gx, gy = np.meshgrid(
        np.linspace(-1.0, 1.0, 4, dtype=np.float32),
        np.linspace(-1.0, 1.0, 3, dtype=np.float32),
    )
    xyz = np.stack(
        (gx.reshape(-1), gy.reshape(-1), np.zeros(12, np.float32)), axis=-1
    )
    plan = build_xy_interpolation_plan(xyz, resolution=32, neighbors=3)
    constant = interpolate_xy_heatmap(np.full(12, 0.375, np.float32), plan)
    assert constant.shape == (32, 32)
    assert np.allclose(constant, 0.375, rtol=0.0, atol=2e-6)

    ramp = interpolate_xy_heatmap(xyz[:, 0], plan)
    assert float(ramp[:, -1].mean()) > float(ramp[:, 0].mean())
    assert float(ramp[-1].mean() - ramp[0].mean()) < 1e-5

    rgba = rgba_heatmap(affordance_colors(constant), 0.8)
    assert rgba.shape == (32, 32, 4)
    assert np.all(rgba[..., 3] == 204)
    print("[PASS] Teacher-v9.8.12 affordance viewer CPU contract")
    print("[PASS] fixed palette, signed delta and Viser-aligned XY heatmap")


if __name__ == "__main__":
    main()

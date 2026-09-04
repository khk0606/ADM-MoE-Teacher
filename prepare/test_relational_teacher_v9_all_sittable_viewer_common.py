#!/usr/bin/env python3
"""CPU-only tests for Teacher-v9 continuous viewer display math."""

import numpy as np

from relational_teacher_v9_all_sittable_viewer_common import (
    CHANNEL_ORDER,
    affordance_colors,
    build_xy_interpolation_plan,
    interpolate_xy_heatmap,
    rgba_heatmap,
    scalar_channel,
)


def expect_value_error(callable_value) -> None:
    try:
        callable_value()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    stops = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    colors = affordance_colors(stops)
    assert colors.shape == (6, 3) and colors.dtype == np.uint8
    expected = np.asarray(
        (
            (20, 10, 45),
            (35, 90, 205),
            (15, 195, 205),
            (80, 225, 75),
            (250, 205, 45),
            (190, 25, 20),
        ),
        dtype=np.uint8,
    )
    assert np.array_equal(colors, expected)
    assert np.array_equal(
        affordance_colors(np.asarray((-5.0, 5.0), dtype=np.float32)),
        expected[[0, -1]],
    )

    # Corners are intentionally asymmetric.  Image row zero must be -Y and
    # the final row +Y; a vertical flip therefore fails exact corner checks.
    xyz = np.asarray(
        (
            (-2.0, -3.0, 0.0),
            (4.0, -3.0, 0.1),
            (-2.0, 5.0, 0.2),
            (4.0, 5.0, 0.3),
        ),
        dtype=np.float32,
    )
    plan = build_xy_interpolation_plan(
        xyz, resolution=32, neighbors=4, chunk_size=13
    )
    corner_values = np.asarray((0.10, 0.35, 0.70, 0.95), dtype=np.float32)
    raster = interpolate_xy_heatmap(corner_values, plan)
    assert raster.shape == (32, 32) and raster.dtype == np.float32
    assert np.allclose(
        (raster[0, 0], raster[0, -1], raster[-1, 0], raster[-1, -1]),
        (0.10, 0.35, 0.70, 0.95),
        rtol=0.0,
        atol=1e-6,
    )
    x_ramp = interpolate_xy_heatmap(xyz[:, 0], plan)
    y_ramp = interpolate_xy_heatmap(xyz[:, 1], plan)
    assert float(x_ramp[:, -1].mean()) > float(x_ramp[:, 0].mean())
    assert float(y_ramp[-1].mean()) > float(y_ramp[0].mean())
    constant = interpolate_xy_heatmap(np.full(4, 0.375, dtype=np.float32), plan)
    assert np.allclose(constant, 0.375, rtol=0.0, atol=1e-6)
    assert np.allclose(plan["xy_min"], (-2.0, -3.0), rtol=0.0, atol=0.0)
    assert np.allclose(plan["xy_max"], (4.0, 5.0), rtol=0.0, atol=0.0)

    rgba = rgba_heatmap(affordance_colors(constant), 0.60)
    assert rgba.shape == (32, 32, 4) and rgba.dtype == np.uint8
    assert np.all(rgba[..., 3] == 153)

    affordance = np.arange(24, dtype=np.float32).reshape(4, 6) / 24.0
    assert np.array_equal(
        scalar_channel(affordance, "any_joint"), affordance.max(axis=-1)
    )
    for index, name in enumerate(CHANNEL_ORDER):
        assert np.array_equal(scalar_channel(affordance, name), affordance[:, index])
    expect_value_error(lambda: scalar_channel(affordance, "distance"))
    expect_value_error(
        lambda: build_xy_interpolation_plan(xyz, resolution=31, neighbors=4)
    )
    expect_value_error(
        lambda: build_xy_interpolation_plan(xyz, resolution=32, neighbors=5)
    )
    expect_value_error(lambda: interpolate_xy_heatmap(np.ones(3), plan))

    print("[PASS] Teacher-v9 fixed physical 0-to-1 affordance palette")
    print("[PASS] Viser-aligned +X/right and +Y/final-row heatmap orientation")
    print("[PASS] exact-corner, constant-field and RGBA interpolation behavior")
    print("[PASS] six-channel selector and invalid-input guards")


if __name__ == "__main__":
    main()

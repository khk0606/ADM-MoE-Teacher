#!/usr/bin/env python3
"""Display-only continuous XY heatmap helpers for Teacher-v9."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


CHANNEL_ORDER = (
    "pelvis",
    "left_foot",
    "right_foot",
    "neck",
    "left_wrist",
    "right_wrist",
)


def affordance_colors(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), dtype=np.float32)
    palette = np.asarray(
        (
            (20, 10, 45),
            (35, 90, 205),
            (15, 195, 205),
            (80, 225, 75),
            (250, 205, 45),
            (190, 25, 20),
        ),
        dtype=np.float32,
    )
    return np.stack(
        [np.interp(values, stops, palette[:, channel]) for channel in range(3)],
        axis=-1,
    ).round().astype(np.uint8)


def scalar_channel(affordance: np.ndarray, channel: str) -> np.ndarray:
    affordance = np.asarray(affordance, dtype=np.float32)
    if affordance.ndim != 2 or affordance.shape[1] != 6:
        raise ValueError("viewer affordance must be [N,6]")
    if channel == "any_joint":
        return affordance.max(axis=-1)
    if channel not in CHANNEL_ORDER:
        raise ValueError("unknown affordance channel: " + str(channel))
    return affordance[:, CHANNEL_ORDER.index(channel)]


def build_xy_interpolation_plan(
    xyz: np.ndarray,
    resolution: int = 128,
    neighbors: int = 8,
    chunk_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Create deterministic inverse-distance rasterization in scene XY."""

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("viewer expected finite XYZ points")
    resolution = int(resolution)
    neighbors = int(neighbors)
    if not 32 <= resolution <= 256:
        raise ValueError("heatmap resolution must be in [32,256]")
    if not 1 <= neighbors <= min(32, xyz.shape[0]):
        raise ValueError("heatmap neighbor count is invalid")
    xy = xyz[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    if np.any(xy_max <= xy_min):
        raise ValueError("scene XY extent is degenerate")
    grid_x = np.linspace(xy_min[0], xy_max[0], resolution, dtype=np.float32)
    # Image row zero maps to the Viser plane's negative-Y edge.  Ascending Y
    # keeps the raster and point cloud in the same world orientation.
    grid_y = np.linspace(xy_min[1], xy_max[1], resolution, dtype=np.float32)
    gx, gy = np.meshgrid(grid_x, grid_y)
    query = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1)
    indices_all = np.empty((query.shape[0], neighbors), dtype=np.int32)
    weights_all = np.empty((query.shape[0], neighbors), dtype=np.float32)
    for start in range(0, query.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query.shape[0])
        delta = query[start:stop, None, :] - xy[None, :, :]
        distance2 = np.sum(delta * delta, axis=-1, dtype=np.float32)
        indices = np.argpartition(distance2, neighbors - 1, axis=1)[:, :neighbors]
        nearest = np.take_along_axis(distance2, indices, axis=1)
        order = np.argsort(nearest, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        nearest = np.take_along_axis(nearest, order, axis=1)
        weights = 1.0 / np.maximum(nearest, np.float32(1e-10))
        exact = nearest[:, 0] <= np.float32(1e-10)
        if np.any(exact):
            weights[exact] = 0.0
            weights[exact, 0] = 1.0
        weights /= weights.sum(axis=1, keepdims=True)
        indices_all[start:stop] = indices
        weights_all[start:stop] = weights
    return {
        "indices": indices_all,
        "weights": weights_all,
        "resolution": np.asarray(resolution, dtype=np.int32),
        "xy_min": xy_min,
        "xy_max": xy_max,
        "floor_z": np.asarray(np.percentile(xyz[:, 2], 1.0), dtype=np.float32),
    }


def interpolate_xy_heatmap(
    values: np.ndarray, plan: Mapping[str, np.ndarray]
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    indices = np.asarray(plan["indices"], dtype=np.int32)
    weights = np.asarray(plan["weights"], dtype=np.float32)
    resolution = int(np.asarray(plan["resolution"]).item())
    if values.ndim != 1 or values.shape[0] <= int(indices.max()):
        raise ValueError("heatmap scalar/plan shape mismatch")
    raster = np.sum(values[indices] * weights, axis=1, dtype=np.float32)
    return raster.reshape(resolution, resolution)


def rgba_heatmap(colors: np.ndarray, opacity: float) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim != 3 or colors.shape[-1] != 3:
        raise ValueError("heatmap colors must be HxWx3")
    alpha = np.full(
        colors.shape[:2] + (1,),
        int(round(255.0 * float(np.clip(opacity, 0.0, 1.0)))),
        dtype=np.uint8,
    )
    return np.concatenate((colors, alpha), axis=-1)


__all__ = [
    "CHANNEL_ORDER",
    "affordance_colors",
    "build_xy_interpolation_plan",
    "interpolate_xy_heatmap",
    "rgba_heatmap",
    "scalar_channel",
]

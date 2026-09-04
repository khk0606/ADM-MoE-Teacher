#!/usr/bin/env python3
"""Pure NumPy rendering helpers for the Teacher-v9.8.12 Viser viewer."""

from __future__ import annotations

import math
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


def scalar_channel(affordance: np.ndarray, channel: str) -> np.ndarray:
    values = np.asarray(affordance, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("affordance must have shape [N,6]")
    if channel == "any_joint":
        return values.max(axis=-1)
    if channel not in CHANNEL_ORDER:
        raise ValueError("unknown affordance channel: " + str(channel))
    return values[:, CHANNEL_ORDER.index(channel)]


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
        [np.interp(values, stops, palette[:, axis]) for axis in range(3)], axis=-1
    ).round().astype(np.uint8)


def signed_difference_colors(values: np.ndarray, limit: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    limit = float(limit)
    if not np.isfinite(values).all() or not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("signed-difference values/limit are invalid")
    normalized = np.clip(values / limit, -1.0, 1.0)
    stops = np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0), dtype=np.float32)
    palette = np.asarray(
        (
            (25, 75, 200),
            (80, 175, 245),
            (35, 35, 45),
            (255, 175, 65),
            (210, 35, 35),
        ),
        dtype=np.float32,
    )
    return np.stack(
        [np.interp(normalized, stops, palette[:, axis]) for axis in range(3)],
        axis=-1,
    ).round().astype(np.uint8)


def build_xy_interpolation_plan(
    xyz: np.ndarray,
    *,
    resolution: int = 128,
    neighbors: int = 8,
    chunk_size: int = 256,
) -> Dict[str, np.ndarray]:
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
    grid_y = np.linspace(xy_min[1], xy_max[1], resolution, dtype=np.float32)
    gx, gy = np.meshgrid(grid_x, grid_y)
    query = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1)
    all_indices = np.empty((query.shape[0], neighbors), dtype=np.int32)
    all_weights = np.empty((query.shape[0], neighbors), dtype=np.float32)
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
        all_indices[start:stop] = indices
        all_weights[start:stop] = weights
    return {
        "indices": all_indices,
        "weights": all_weights,
        "resolution": np.asarray(resolution, dtype=np.int32),
        "xy_min": xy_min,
        "xy_max": xy_max,
        "floor_z": np.asarray(np.percentile(xyz[:, 2], 1.0), dtype=np.float32),
    }


def interpolate_xy_heatmap(values: np.ndarray, plan: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    indices = np.asarray(plan["indices"], dtype=np.int32)
    weights = np.asarray(plan["weights"], dtype=np.float32)
    resolution = int(np.asarray(plan["resolution"]).item())
    if values.ndim != 1 or values.shape[0] <= int(indices.max()):
        raise ValueError("heatmap scalar/plan shape mismatch")
    return np.sum(values[indices] * weights, axis=1, dtype=np.float32).reshape(
        resolution, resolution
    )


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
    "signed_difference_colors",
]

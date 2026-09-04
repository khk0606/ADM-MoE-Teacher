#!/usr/bin/env python3
"""Small deterministic helpers for six-task common-descent construction."""

from __future__ import annotations

import numpy as np
import torch


def frank_wolfe_min_norm_weights(
    gram: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Return a deterministic minimum-norm convex combination.

    For a nonzero minimum-norm point in the convex hull of task gradients,
    every task has a nonnegative directional derivative along that point.
    """

    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("gradient Gram matrix must be square with at least two tasks")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("gradient Gram matrix is invalid")
    if iterations <= 0:
        raise ValueError("Frank-Wolfe iteration count must be positive")
    count = matrix.shape[0]
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    for _ in range(iterations):
        gradient = matrix @ weights
        vertex_index = int(np.argmin(gradient))
        vertex = np.zeros(count, dtype=np.float64)
        vertex[vertex_index] = 1.0
        delta = vertex - weights
        denominator = float(delta @ matrix @ delta)
        if denominator <= 1e-20:
            break
        gamma = float(np.clip(-(delta @ matrix @ weights) / denominator, 0.0, 1.0))
        weights = weights + gamma * delta
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return weights


def flattened_task_gradients(
    task_losses: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    """Compute one live flat gradient row per scalar task loss."""

    if task_losses.ndim != 1 or task_losses.numel() < 2:
        raise ValueError("task loss vector must contain at least two scalars")
    rows = []
    for index, loss in enumerate(task_losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index + 1 < task_losses.numel(),
            allow_unused=True,
        )
        pieces = [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
        row = torch.cat(pieces)
        norm = torch.linalg.vector_norm(row)
        if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
            raise RuntimeError("a common-descent task has a zero/non-finite gradient")
        rows.append(row / norm)
    return torch.stack(rows)


def apply_flat_direction(
    parameters: list[torch.nn.Parameter],
    unit_direction: torch.Tensor,
    radius: float,
) -> None:
    """Apply an exact Euclidean trust-region step to the LoRA parameters."""

    if radius <= 0.0:
        raise ValueError("step radius must be positive")
    expected = sum(parameter.numel() for parameter in parameters)
    if unit_direction.ndim != 1 or unit_direction.numel() != expected:
        raise ValueError("flat direction size differs from LoRA parameters")
    if not torch.isfinite(unit_direction).all():
        raise ValueError("flat direction contains NaN/Inf")
    if not torch.isclose(
        torch.linalg.vector_norm(unit_direction),
        unit_direction.new_tensor(1.0),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("common-descent direction must have unit norm")
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            update = unit_direction[offset : offset + count].reshape_as(parameter)
            parameter.add_(update, alpha=-float(radius))
            offset += count
    if offset != expected:
        raise AssertionError("flat direction application was incomplete")


__all__ = [
    "apply_flat_direction",
    "flattened_task_gradients",
    "frank_wolfe_min_norm_weights",
]

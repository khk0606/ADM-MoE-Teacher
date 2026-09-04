#!/usr/bin/env python3
"""Evaluation-aligned exact top-k swap objective for Teacher-v9.3."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from relational_teacher_v9_lora_objective import (
    all_sittable_objective,
    physical_prediction,
)
from relational_teacher_v91_active_support_objective import active_support_macro_loss


ACTIVE_THRESHOLD = 0.30
TOP_FRACTION = 0.25
SWAP_MARGIN = 0.02
SWAP_TEMPERATURE = 0.01


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    """Match NumPy lexsort's value-descending/index-ascending tie policy."""

    # Indices are inherently non-differentiable.  Select them with the exact
    # NumPy policy used by the metric, then gather live Torch scores so all
    # selected values still carry gradients.
    array = values.detach().cpu().numpy()
    indices = np.arange(array.shape[0], dtype=np.int64)
    order = np.lexsort((indices, -array))
    return torch.from_numpy(order.astype(np.int64, copy=False)).to(values.device)


def evaluation_target_topk(
    target_any: torch.Tensor,
    object_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact target set and all object points used by the metric."""

    object_indices = torch.nonzero(object_mask, as_tuple=False).flatten()
    active_indices = torch.nonzero(
        object_mask & (target_any >= ACTIVE_THRESHOLD), as_tuple=False
    ).flatten()
    if object_indices.numel() <= 1 or active_indices.numel() <= 0:
        raise ValueError("exact top-k object/active support is empty")
    count = max(1, int(math.ceil(float(active_indices.numel()) * TOP_FRACTION)))
    target_order = _stable_descending(target_any[active_indices])
    return active_indices[target_order[:count]], object_indices


def _is_member(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (values[:, None] == reference[None, :]).any(dim=1)


def exact_topk_swap_macro_loss(
    physical: torch.Tensor,
    instance_targets: torch.Tensor,
    verified_object_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directly swap false predicted top-k points with missing GT top-k points.

    The target and prediction sets are exactly those used by
    ``one_instance_metrics``.  Selection indices are discrete, but gradients
    flow through the selected false-positive and missing-positive scores.
    """

    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance targets must be [B,3,N,6]")
    if verified_object_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified object masks must be [B,3,N]")
    if physical.shape != (
        instance_targets.shape[0],
        instance_targets.shape[2],
        instance_targets.shape[3],
    ):
        raise ValueError("physical prediction/target shape differs")
    loss_rows = []
    overlap_rows = []
    for batch_index in range(physical.shape[0]):
        prediction_any = physical[batch_index].amax(dim=-1)
        object_losses = []
        object_overlaps = []
        for slot in range(3):
            target_any = instance_targets[batch_index, slot].amax(dim=-1)
            target_top, object_indices = evaluation_target_topk(
                target_any, verified_object_masks[batch_index, slot]
            )
            count = int(target_top.numel())
            prediction_order = _stable_descending(prediction_any[object_indices])
            prediction_top = object_indices[prediction_order[:count]]
            prediction_is_target = _is_member(prediction_top, target_top)
            target_is_prediction = _is_member(target_top, prediction_top)
            false_top = prediction_top[~prediction_is_target]
            missing_top = target_top[~target_is_prediction]
            if false_top.numel() != missing_top.numel():
                raise AssertionError("top-k false/missing set cardinality differs")
            overlap = physical.new_tensor(
                float(count - int(false_top.numel())) / float(count)
            )
            object_overlaps.append(overlap)
            if false_top.numel() > 0:
                false_scores = torch.sort(
                    prediction_any[false_top], descending=True
                ).values
                missing_scores = torch.sort(
                    prediction_any[missing_top], descending=False
                ).values
            else:
                outside = object_indices[~_is_member(object_indices, target_top)]
                if outside.numel() <= 0:
                    raise ValueError("exact top-k object has no outside points")
                false_scores = prediction_any[outside].max().reshape(1)
                missing_scores = prediction_any[target_top].min().reshape(1)
            temperature = physical.new_tensor(SWAP_TEMPERATURE)
            object_losses.append(
                F.softplus(
                    (
                        physical.new_tensor(SWAP_MARGIN)
                        + false_scores
                        - missing_scores
                    )
                    / temperature
                ).mean()
                * temperature
            )
        loss_rows.append(torch.stack(object_losses))
        overlap_rows.append(torch.stack(object_overlaps))
    per_instance_loss = torch.stack(loss_rows)
    per_instance_overlap = torch.stack(overlap_rows)
    return per_instance_loss.mean(), per_instance_loss, per_instance_overlap


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    mask = point_mask.to(prediction.dtype).unsqueeze(-1)
    denominator = mask.sum() * prediction.shape[-1]
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("masked trust loss is empty")
    return ((prediction - target).square() * mask).sum() / denominator


def corrected_v93_objective(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    result = dict(
        all_sittable_objective(prediction, frozen_prediction, batch, mean, std)
    )
    physical = physical_prediction(prediction, mean, std)
    frozen_physical = physical_prediction(frozen_prediction, mean, std)
    active, active_rows = active_support_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    swap, swap_rows, overlap_rows = exact_topk_swap_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    verified_union = batch["verified_object_mask"].any(dim=1)
    background_mask = (
        (~verified_union)
        & (~batch["unknown_sittable_mask"])
        & (~batch["explicit_negative_mask"])
    )
    result.update(
        {
            "active_support_macro": active,
            "per_instance_active_support": active_rows,
            "exact_topk_swap_macro": swap,
            "per_instance_exact_topk_swap": swap_rows,
            "per_instance_exact_topk_overlap": overlap_rows,
            "background_trust": _masked_mse(
                physical, frozen_physical, background_mask
            ),
            "absolute_negative": _masked_mse(
                physical,
                torch.zeros_like(physical),
                batch["explicit_negative_mask"],
            ),
        }
    )
    return result


__all__ = [
    "ACTIVE_THRESHOLD",
    "SWAP_MARGIN",
    "SWAP_TEMPERATURE",
    "TOP_FRACTION",
    "corrected_v93_objective",
    "evaluation_target_topk",
    "exact_topk_swap_macro_loss",
]

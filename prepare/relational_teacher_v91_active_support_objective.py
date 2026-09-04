#!/usr/bin/env python3
"""Active-support and within-object ranking losses for Teacher-v9.1."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from relational_teacher_v9_lora_objective import (
    all_sittable_objective,
    physical_prediction,
)


ACTIVE_THRESHOLD = 0.30
RANK_MARGIN = 0.05
RANK_FRACTION = 0.25


def _channel_masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("channel-masked tensors must have identical shapes")
    denominator = mask.to(prediction.dtype).sum()
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("channel-masked loss has empty support")
    return (
        (prediction - target).square() * mask.to(prediction.dtype)
    ).sum() / denominator


def active_support_macro_loss(
    physical: torch.Tensor,
    instance_targets: torch.Tensor,
    verified_object_masks: torch.Tensor,
    *,
    threshold: float = ACTIVE_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-macro MSE on GT-active channels for all three objects."""

    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance targets must be [B,3,N,6]")
    if physical.shape != (
        instance_targets.shape[0],
        instance_targets.shape[2],
        instance_targets.shape[3],
    ):
        raise ValueError("physical prediction shape changed")
    if verified_object_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified object masks must be [B,3,N]")
    rows = []
    for batch_index in range(physical.shape[0]):
        instances = []
        for slot in range(3):
            target = instance_targets[batch_index, slot]
            mask = (
                (target >= float(threshold))
                & verified_object_masks[batch_index, slot].unsqueeze(-1)
            )
            instances.append(
                _channel_masked_mse(physical[batch_index], target, mask)
            )
        rows.append(torch.stack(instances))
    per_instance = torch.stack(rows)
    return per_instance.mean(), per_instance


def within_object_ranking_loss(
    physical: torch.Tensor,
    instance_targets: torch.Tensor,
    verified_object_masks: torch.Tensor,
    *,
    threshold: float = ACTIVE_THRESHOLD,
    fraction: float = RANK_FRACTION,
    margin: float = RANK_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank GT top-contact points above hard non-top points per object."""

    if not 0.0 < float(fraction) <= 0.5 or float(margin) <= 0.0:
        raise ValueError("ranking hyperparameters are invalid")
    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance targets must be [B,3,N,6]")
    if verified_object_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified object masks must be [B,3,N]")
    rows = []
    for batch_index in range(physical.shape[0]):
        instances = []
        prediction_any = physical[batch_index].amax(dim=-1)
        for slot in range(3):
            target_any = instance_targets[batch_index, slot].amax(dim=-1)
            object_mask = verified_object_masks[batch_index, slot]
            active_indices = torch.nonzero(
                object_mask & (target_any >= float(threshold)), as_tuple=False
            ).flatten()
            object_indices = torch.nonzero(object_mask, as_tuple=False).flatten()
            if active_indices.numel() <= 0 or object_indices.numel() <= 1:
                raise ValueError("ranking object support is empty")
            count = max(1, int(math.ceil(float(active_indices.numel()) * float(fraction))))
            count = min(count, int(active_indices.numel()))
            target_order = torch.topk(
                target_any[active_indices], count, largest=True, sorted=False
            ).indices
            positive_indices = active_indices[target_order]
            negative_mask = object_mask.clone()
            negative_mask[positive_indices] = False
            negative_indices = torch.nonzero(negative_mask, as_tuple=False).flatten()
            if negative_indices.numel() <= 0:
                raise ValueError("ranking negative support is empty")
            hard_count = min(count, int(negative_indices.numel()))
            positive_scores = prediction_any[positive_indices]
            hard_negative_scores = torch.topk(
                prediction_any[negative_indices],
                hard_count,
                largest=True,
                sorted=True,
            ).values
            positive_scores = torch.sort(
                positive_scores, descending=False
            ).values[:hard_count]
            temperature = 0.05
            instances.append(
                F.softplus(
                    (
                        physical.new_tensor(float(margin))
                        + hard_negative_scores
                        - positive_scores
                    )
                    / temperature
                ).mean()
                * temperature
            )
        rows.append(torch.stack(instances))
    per_instance = torch.stack(rows)
    return per_instance.mean(), per_instance


def corrected_objective(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Original sealed objective plus support/ranking diagnostics."""

    result = dict(
        all_sittable_objective(
            prediction, frozen_prediction, batch, mean, std
        )
    )
    physical = physical_prediction(prediction, mean, std)
    active, active_rows = active_support_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    ranking, ranking_rows = within_object_ranking_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    result.update(
        {
            "active_support_macro": active,
            "per_instance_active_support": active_rows,
            "within_object_ranking": ranking,
            "per_instance_ranking": ranking_rows,
        }
    )
    return result


__all__ = [
    "ACTIVE_THRESHOLD",
    "RANK_FRACTION",
    "RANK_MARGIN",
    "active_support_macro_loss",
    "corrected_objective",
    "within_object_ranking_loss",
]

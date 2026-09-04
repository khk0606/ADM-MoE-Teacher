#!/usr/bin/env python3
"""Hotspot-ordering and trust-region losses for Teacher-v9.2."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from relational_teacher_v9_lora_objective import (
    all_sittable_objective,
    physical_prediction,
)
from relational_teacher_v91_active_support_objective import active_support_macro_loss


TOP_FRACTION = 0.25
MARGIN = 0.05
MARGIN_TEMPERATURE = 0.02
LISTWISE_TEMPERATURE = 0.05


def _top_partition(
    target_any: torch.Tensor,
    object_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(object_mask, as_tuple=False).flatten()
    if indices.numel() < 4:
        raise ValueError("verified object has too few points")
    count = max(1, int(math.ceil(float(indices.numel()) * TOP_FRACTION)))
    order = torch.argsort(target_any[indices], descending=True)
    positive = indices[order[:count]]
    negative = indices[order[count:]]
    if negative.numel() <= 0:
        raise ValueError("hotspot partition has no negative points")
    return positive, negative


def hotspot_margin_macro_loss(
    physical: torch.Tensor,
    instance_targets: torch.Tensor,
    verified_object_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push the weakest GT-top-quartile point above hard non-top points."""

    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance targets must be [B,3,N,6]")
    if verified_object_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified object masks must be [B,3,N]")
    rows = []
    for batch_index in range(physical.shape[0]):
        prediction_any = physical[batch_index].amax(dim=-1)
        instances = []
        for slot in range(3):
            target_any = instance_targets[batch_index, slot].amax(dim=-1)
            positive, negative = _top_partition(
                target_any, verified_object_masks[batch_index, slot]
            )
            positive_scores = prediction_any[positive]
            negative_scores = prediction_any[negative]
            temperature = physical.new_tensor(MARGIN_TEMPERATURE)
            # Unnormalised log-sum-exp deliberately bounds the actual
            # extrema: soft_min <= min(positive) and soft_max >= max(negative).
            # This makes the response test conservative instead of allowing a
            # broad, low-amplitude object response to look like a hotspot.
            soft_min_positive = -temperature * torch.logsumexp(
                -positive_scores / temperature, dim=0
            )
            soft_max_negative = temperature * torch.logsumexp(
                negative_scores / temperature, dim=0
            )
            instances.append(
                F.softplus(
                    (
                        physical.new_tensor(MARGIN)
                        + soft_max_negative
                        - soft_min_positive
                    )
                    / temperature
                )
                * temperature
            )
        rows.append(torch.stack(instances))
    per_instance = torch.stack(rows)
    return per_instance.mean(), per_instance


def hotspot_listwise_macro_loss(
    physical: torch.Tensor,
    instance_targets: torch.Tensor,
    verified_object_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the complete within-object hotspot ordering distribution."""

    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance targets must be [B,3,N,6]")
    if verified_object_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified object masks must be [B,3,N]")
    if physical.ndim != 3 or physical.shape[0] != instance_targets.shape[0]:
        raise ValueError("physical prediction must be [B,N,6]")
    if physical.shape[1:] != instance_targets.shape[2:]:
        raise ValueError("physical prediction/target shape differs")
    rows = []
    for batch_index in range(physical.shape[0]):
        prediction_any = physical[batch_index].amax(dim=-1)
        instances = []
        for slot in range(3):
            indices = torch.nonzero(
                verified_object_masks[batch_index, slot], as_tuple=False
            ).flatten()
            if indices.numel() < 4:
                raise ValueError("verified object has too few listwise points")
            target_any = instance_targets[batch_index, slot].amax(dim=-1)
            target_probability = torch.softmax(
                target_any[indices] / LISTWISE_TEMPERATURE, dim=0
            ).detach()
            prediction_log_probability = torch.log_softmax(
                prediction_any[indices] / LISTWISE_TEMPERATURE, dim=0
            )
            instances.append(
                -(target_probability * prediction_log_probability).sum()
            )
        rows.append(torch.stack(instances))
    per_instance = torch.stack(rows)
    return per_instance.mean(), per_instance


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


def corrected_v92_objective(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    result = dict(
        all_sittable_objective(
            prediction, frozen_prediction, batch, mean, std
        )
    )
    physical = physical_prediction(prediction, mean, std)
    frozen_physical = physical_prediction(frozen_prediction, mean, std)
    active, active_rows = active_support_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    margin, margin_rows = hotspot_margin_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    listwise, listwise_rows = hotspot_listwise_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    verified_object_union = batch["verified_object_mask"].any(dim=1)
    background_mask = (~verified_object_union) & (~batch["unknown_sittable_mask"])
    background_trust = _masked_mse(
        physical, frozen_physical, background_mask
    )
    zeros = torch.zeros_like(physical)
    absolute_negative = _masked_mse(
        physical, zeros, batch["explicit_negative_mask"]
    )
    result.update(
        {
            "active_support_macro": active,
            "per_instance_active_support": active_rows,
            "hotspot_margin_macro": margin,
            "per_instance_hotspot_margin": margin_rows,
            "hotspot_listwise_macro": listwise,
            "per_instance_hotspot_listwise": listwise_rows,
            "background_trust": background_trust,
            "absolute_negative": absolute_negative,
        }
    )
    return result


__all__ = [
    "LISTWISE_TEMPERATURE",
    "MARGIN",
    "MARGIN_TEMPERATURE",
    "TOP_FRACTION",
    "corrected_v92_objective",
    "hotspot_listwise_macro_loss",
    "hotspot_margin_macro_loss",
]

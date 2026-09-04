#!/usr/bin/env python3
"""Direct all-sittable x0 objective used by Teacher-v10."""

from __future__ import annotations

from typing import Mapping

import torch

from relational_teacher_v9_lora_objective import physical_prediction


ACTIVE_THRESHOLD = 0.30


def _point_mse(
    prediction: torch.Tensor, target: torch.Tensor, point_mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target must be matching [B,N,6]")
    if point_mask.shape != prediction.shape[:2]:
        raise ValueError("point mask must be [B,N]")
    mask = point_mask.to(prediction.dtype).unsqueeze(-1)
    denominator = mask.sum() * prediction.shape[-1]
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("point-masked objective has empty support")
    return ((prediction - target).square() * mask).sum() / denominator


def _channel_mse(
    prediction: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or channel_mask.shape != prediction.shape:
        raise ValueError("channel-masked objective shape changed")
    mask = channel_mask.to(prediction.dtype)
    denominator = mask.sum()
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("channel-masked objective has empty support")
    return ((prediction - target).square() * mask).sum() / denominator


def direct_all_sittable_objective(
    normalized_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise the actual all-sittable label, equally across three objects."""

    physical = physical_prediction(normalized_prediction, mean, std)
    all_target = batch["all_target"]
    instance_targets = batch["instance_targets"]
    verified_masks = batch["verified_object_mask"].bool()
    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance target inventory must be [B,3,N,6]")
    if verified_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified object mask inventory changed")

    full_rows = []
    active_rows = []
    active_mean_rows = []
    for batch_index in range(physical.shape[0]):
        full_instances = []
        active_instances = []
        active_means = []
        for slot in range(3):
            target = instance_targets[batch_index : batch_index + 1, slot]
            prediction = physical[batch_index : batch_index + 1]
            point_mask = verified_masks[batch_index : batch_index + 1, slot]
            full_instances.append(_point_mse(prediction, target, point_mask))
            active_mask = (target >= ACTIVE_THRESHOLD) & point_mask.unsqueeze(-1)
            active_instances.append(_channel_mse(prediction, target, active_mask))
            active_means.append(prediction[active_mask].mean())
        full_rows.append(torch.stack(full_instances))
        active_rows.append(torch.stack(active_instances))
        active_mean_rows.append(torch.stack(active_means))
    per_instance_full = torch.stack(full_rows)
    per_instance_active = torch.stack(active_rows)
    per_instance_active_mean = torch.stack(active_mean_rows)

    verified_positive = batch["verified_positive_mask"].bool()
    environment = batch["environment_aux_mask"].bool()
    negative = batch["explicit_negative_mask"].bool()
    verified_union = _point_mse(physical, all_target, verified_positive)
    environment_loss = _point_mse(physical, all_target, environment)
    zero = torch.zeros_like(physical)
    negative_absolute = _point_mse(physical, zero, negative)
    if physical.shape[0] != 2 or not torch.equal(
        verified_positive[0], verified_positive[1]
    ):
        raise ValueError("watch/write training pair changed")
    prompt_invariance = _point_mse(
        physical[0:1], physical[1:2], verified_positive[0:1]
    )
    return {
        "instance_full": per_instance_full.mean(),
        "instance_active": per_instance_active.mean(),
        "per_instance_full": per_instance_full,
        "per_instance_active": per_instance_active,
        "per_instance_active_mean": per_instance_active_mean,
        "verified_union": verified_union,
        "environment": environment_loss,
        "explicit_negative_absolute": negative_absolute,
        "prompt_invariance": prompt_invariance,
        "physical": physical,
    }


__all__ = ["ACTIVE_THRESHOLD", "direct_all_sittable_objective"]

#!/usr/bin/env python3
"""Dense per-instance support objective for Teacher-v10.2."""

from __future__ import annotations

from typing import Mapping

import torch

from relational_teacher_v10_supervised_objective import _channel_mse, _point_mse
from relational_teacher_v9_lora_objective import physical_prediction
from relational_teacher_v102_dense_instance_contract import (
    ACTIVE_THRESHOLD,
    HARD_NEGATIVE_POINTS,
    SUPPORT_THRESHOLD,
)


def _hard_false_positive_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    safe_background: torch.Tensor,
) -> torch.Tensor:
    prediction_scalar = prediction.max(dim=-1).values
    target_scalar = target.max(dim=-1).values
    rows = []
    for batch_index in range(prediction.shape[0]):
        mask = safe_background[batch_index]
        values = prediction_scalar[batch_index, mask]
        targets = target_scalar[batch_index, mask]
        if values.numel() == 0:
            raise ValueError("Teacher-v10.2 safe background is empty")
        count = min(HARD_NEGATIVE_POINTS, int(values.numel()))
        _, indices = torch.topk(values.detach(), k=count, largest=True, sorted=False)
        rows.append((values[indices] - targets[indices]).square().mean())
    return torch.stack(rows).mean()


def dense_instance_all_sittable_objective(
    normalized_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise every dense positive cell of each verified instance equally."""
    physical = physical_prediction(normalized_prediction, mean, std)
    target = batch["all_target"]
    unknown = batch["unknown_sittable_mask"].bool()
    known = ~unknown
    instance_targets = batch["instance_targets"]
    if (
        physical.shape != target.shape
        or unknown.shape != physical.shape[:2]
        or instance_targets.shape
        != (physical.shape[0], 3, physical.shape[1], physical.shape[2])
    ):
        raise ValueError("Teacher-v10.2 dense-instance batch shape changed")
    if physical.shape[0] != 2 or not torch.equal(known[0], known[1]):
        raise ValueError("Teacher-v10.2 watch/write pair changed")

    # Every point of every verified instance's dense motion-contact map is a
    # positive.  In particular this is deliberately not intersected with the
    # small object-surface mask that caused the v10.1 Normal-Chair collapse.
    instance_support_points = (
        instance_targets.max(dim=-1).values >= SUPPORT_THRESHOLD
    )
    all_instance_support = instance_support_points.any(dim=1)
    instance_active_channels = instance_targets >= ACTIVE_THRESHOLD
    for batch_index in range(physical.shape[0]):
        for slot in range(3):
            if not bool(instance_support_points[batch_index, slot].any()):
                raise ValueError("Teacher-v10.2 instance support is empty")
            if not bool(instance_active_channels[batch_index, slot].any()):
                raise ValueError("Teacher-v10.2 instance active support is empty")

    full_field_known = _point_mse(physical, target, known)
    safe_background = (
        known
        & ~all_instance_support
        & (target.max(dim=-1).values < SUPPORT_THRESHOLD)
    )
    background_absolute = _point_mse(physical, target, safe_background)
    hard_false_positive = _hard_false_positive_loss(
        physical, target, safe_background
    )

    support_rows = []
    active_rows = []
    active_mean_rows = []
    for batch_index in range(physical.shape[0]):
        support_instances = []
        active_instances = []
        active_means = []
        for slot in range(3):
            one_prediction = physical[batch_index : batch_index + 1]
            one_target = instance_targets[batch_index : batch_index + 1, slot]
            support_mask = instance_support_points[
                batch_index : batch_index + 1, slot
            ]
            active_mask = instance_active_channels[
                batch_index : batch_index + 1, slot
            ]
            support_instances.append(
                _point_mse(one_prediction, one_target, support_mask)
            )
            active_instances.append(
                _channel_mse(one_prediction, one_target, active_mask)
            )
            active_means.append(one_prediction[active_mask].mean())
        support_rows.append(torch.stack(support_instances))
        active_rows.append(torch.stack(active_instances))
        active_mean_rows.append(torch.stack(active_means))
    per_instance_support = torch.stack(support_rows)
    per_instance_active = torch.stack(active_rows)
    per_instance_active_mean = torch.stack(active_mean_rows)

    verified_positive = batch["verified_positive_mask"].bool()
    negative = batch["explicit_negative_mask"].bool()
    verified_union = _point_mse(physical, target, verified_positive)
    explicit_negative_absolute = _point_mse(
        physical, torch.zeros_like(physical), negative
    )
    prompt_invariance = _point_mse(physical[0:1], physical[1:2], known[0:1])
    return {
        "full_field_known": full_field_known,
        "background_absolute": background_absolute,
        "hard_false_positive": hard_false_positive,
        "instance_dense_support": per_instance_support.mean(),
        "instance_dense_active": per_instance_active.mean(),
        "worst_instance_dense_active": per_instance_active.max(dim=1).values.mean(),
        "per_instance_support": per_instance_support,
        "per_instance_active": per_instance_active,
        "per_instance_active_mean": per_instance_active_mean,
        "verified_union": verified_union,
        "explicit_negative_absolute": explicit_negative_absolute,
        "prompt_invariance": prompt_invariance,
        "physical": physical,
    }


__all__ = ["dense_instance_all_sittable_objective"]

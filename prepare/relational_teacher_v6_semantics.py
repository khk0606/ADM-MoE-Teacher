#!/usr/bin/env python3
"""Differentiable semantic-only losses for relational Teacher-v6 adaptation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from relational_teacher_v6_contract import (
    CATEGORY_IDS,
    SITTABLE_CATEGORY_IDS,
    SIT_NEGATIVE_OBJECT_CATEGORY_IDS,
)


def _straight_through_unit_clamp(values: torch.Tensor) -> torch.Tensor:
    clipped = values.clamp(0.0, 1.0)
    return values + (clipped - values).detach()


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("cannot score an empty instance")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top fraction must be in (0,1]")
    count = max(1, int(math.ceil(values.numel() * fraction)))
    return torch.topk(values.reshape(-1), k=count, largest=True).values.mean()


def _instance_ids_for_categories(
    instance_ids: torch.Tensor,
    category_ids: torch.Tensor,
    categories: Iterable[int],
) -> Sequence[int]:
    category_set = {int(value) for value in categories}
    values = []
    for instance_id in torch.unique(instance_ids).detach().cpu().tolist():
        instance_id = int(instance_id)
        if instance_id == 0:
            continue
        mask = instance_ids == instance_id
        cats = torch.unique(category_ids[mask]).detach().cpu().tolist()
        if len(cats) != 1:
            raise ValueError(f"instance {instance_id} has mixed categories")
        if int(cats[0]) in category_set:
            values.append(instance_id)
    return tuple(sorted(values))


def all_sittable_semantic_loss(
    prediction_normalized: torch.Tensor,
    instance_ids: torch.Tensor,
    category_ids: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    candidate_any_min: float = 0.30,
    candidate_pelvis_min: float = 0.15,
    negative_any_max: float = 0.10,
    negative_pelvis_max: float = 0.05,
    top_fraction: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Activate every Chair/Bed instance and suppress non-Sit purpose objects.

    The loss is instance-balanced: a large Bed cannot dominate a small Chair.
    Environment points are intentionally not suppressed because feet may contact
    the floor in a valid Sit affordance map.
    """

    if prediction_normalized.ndim != 3 or prediction_normalized.shape[-1] != 6:
        raise ValueError("prediction must be [B,N,6]")
    if instance_ids.shape != prediction_normalized.shape[:2]:
        raise ValueError("instance_ids must be [B,N]")
    if category_ids.shape != prediction_normalized.shape[:2]:
        raise ValueError("category_ids must be [B,N]")
    if mean.shape[-1] != 6 or std.shape[-1] != 6:
        raise ValueError("mean/std must end in six channels")
    if not (0.0 <= candidate_pelvis_min <= candidate_any_min <= 1.0):
        raise ValueError("invalid candidate activation thresholds")
    if not (0.0 <= negative_pelvis_max <= negative_any_max <= 1.0):
        raise ValueError("invalid negative activation thresholds")

    raw = _straight_through_unit_clamp(prediction_normalized * std + mean)
    zero = prediction_normalized.sum() * 0.0
    candidate_any_loss = zero
    candidate_pelvis_loss = zero
    negative_any_loss = zero
    negative_pelvis_loss = zero
    candidate_count = 0
    negative_count = 0
    metrics = []

    for batch_index in range(prediction_normalized.shape[0]):
        candidate_instances = _instance_ids_for_categories(
            instance_ids[batch_index],
            category_ids[batch_index],
            SITTABLE_CATEGORY_IDS,
        )
        if not candidate_instances:
            raise ValueError("semantic Sit row has no Chair/Bed instance")
        for instance_id in candidate_instances:
            mask = instance_ids[batch_index] == instance_id
            any_score = _top_fraction_mean(
                raw[batch_index, mask].amax(dim=-1), top_fraction
            )
            pelvis_score = _top_fraction_mean(
                raw[batch_index, mask, 0], top_fraction
            )
            candidate_any_loss = candidate_any_loss + F.relu(
                raw.new_tensor(candidate_any_min) - any_score
            )
            candidate_pelvis_loss = candidate_pelvis_loss + F.relu(
                raw.new_tensor(candidate_pelvis_min) - pelvis_score
            )
            candidate_count += 1
            metrics.append(
                {
                    "batch_index": batch_index,
                    "instance_id": instance_id,
                    "role": "sittable",
                    "any_score": any_score.detach(),
                    "pelvis_score": pelvis_score.detach(),
                }
            )

        negative_instances = _instance_ids_for_categories(
            instance_ids[batch_index],
            category_ids[batch_index],
            SIT_NEGATIVE_OBJECT_CATEGORY_IDS,
        )
        for instance_id in negative_instances:
            mask = instance_ids[batch_index] == instance_id
            any_score = _top_fraction_mean(
                raw[batch_index, mask].amax(dim=-1), top_fraction
            )
            pelvis_score = _top_fraction_mean(
                raw[batch_index, mask, 0], top_fraction
            )
            negative_any_loss = negative_any_loss + F.relu(
                any_score - raw.new_tensor(negative_any_max)
            )
            negative_pelvis_loss = negative_pelvis_loss + F.relu(
                pelvis_score - raw.new_tensor(negative_pelvis_max)
            )
            negative_count += 1
            metrics.append(
                {
                    "batch_index": batch_index,
                    "instance_id": instance_id,
                    "role": "sit_negative_object",
                    "any_score": any_score.detach(),
                    "pelvis_score": pelvis_score.detach(),
                }
            )

    candidate_any_loss = candidate_any_loss / float(candidate_count)
    candidate_pelvis_loss = candidate_pelvis_loss / float(candidate_count)
    if negative_count:
        negative_any_loss = negative_any_loss / float(negative_count)
        negative_pelvis_loss = negative_pelvis_loss / float(negative_count)
    total = (
        candidate_any_loss
        + candidate_pelvis_loss
        + negative_any_loss
        + negative_pelvis_loss
    )
    return {
        "total": total,
        "candidate_any": candidate_any_loss,
        "candidate_pelvis": candidate_pelvis_loss,
        "negative_any": negative_any_loss,
        "negative_pelvis": negative_pelvis_loss,
        "candidate_instance_count": raw.new_tensor(float(candidate_count)),
        "negative_instance_count": raw.new_tensor(float(negative_count)),
        "metrics": metrics,
    }


def paired_prompt_invariance_loss(
    watch_prediction: torch.Tensor,
    write_prediction: torch.Tensor,
    category_ids: torch.Tensor,
) -> torch.Tensor:
    """Keep the Base Sit feasibility map purpose-invariant on candidates."""

    if watch_prediction.shape != write_prediction.shape:
        raise ValueError("paired predictions differ in shape")
    if category_ids.shape != watch_prediction.shape[:2]:
        raise ValueError("category_ids shape differs from predictions")
    candidate = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in SITTABLE_CATEGORY_IDS:
        candidate |= category_ids == int(category_id)
    if not torch.all(candidate.any(dim=1)):
        raise ValueError("every prompt pair must contain a candidate")
    delta = (watch_prediction - write_prediction).square().mean(dim=-1)
    return (delta * candidate.to(delta.dtype)).sum() / candidate.sum().clamp_min(1)


def frozen_teacher_preservation_loss(
    adapted_prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    category_ids: torch.Tensor,
) -> torch.Tensor:
    """Preserve the frozen v5 map outside Chair/Bed candidate regions."""

    if adapted_prediction.shape != frozen_prediction.shape:
        raise ValueError("adapted/frozen prediction shape mismatch")
    if category_ids.shape != adapted_prediction.shape[:2]:
        raise ValueError("category_ids shape mismatch")
    candidate = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in SITTABLE_CATEGORY_IDS:
        candidate |= category_ids == int(category_id)
    preserve = ~candidate
    delta = (adapted_prediction - frozen_prediction.detach()).square().mean(dim=-1)
    return (delta * preserve.to(delta.dtype)).sum() / preserve.sum().clamp_min(1)

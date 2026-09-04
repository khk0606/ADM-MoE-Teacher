#!/usr/bin/env python3
"""Torch objective for scene-level Teacher-v9 all-sittable adaptation.

The key invariant is that Bed, normal Chair and High Chair contribute one loss
each, irrespective of object point count.  Unknown Sit objects never enter a
Teacher-v9 task loss.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def physical_prediction(
    normalized_prediction: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Convert to [0,1] with a straight-through clamp for stable gradients."""

    raw = normalized_prediction * std + mean
    clipped = raw.clamp(0.0, 1.0)
    return raw + (clipped - raw).detach()


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target must be matching [B,N,C] tensors")
    if point_mask.shape != prediction.shape[:2]:
        raise ValueError("point mask must be [B,N]")
    mask = point_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = mask.sum() * prediction.shape[-1]
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("masked loss received an empty mask")
    return ((prediction - target).square() * mask).sum() / denominator


def equal_instance_macro_loss(
    prediction: torch.Tensor,
    instance_targets: torch.Tensor,
    verified_object_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return macro loss plus exact [B,3] per-instance losses."""

    if instance_targets.ndim != 4 or instance_targets.shape[1] != 3:
        raise ValueError("instance targets must be exactly [B,3,N,C]")
    if prediction.shape != (
        instance_targets.shape[0],
        instance_targets.shape[2],
        instance_targets.shape[3],
    ):
        raise ValueError("prediction and instance-target shapes differ")
    if verified_object_masks.shape != instance_targets.shape[:3]:
        raise ValueError("verified masks must be [B,3,N]")
    losses = []
    for batch_index in range(prediction.shape[0]):
        rows = []
        for instance_index in range(3):
            rows.append(
                _masked_mse(
                    prediction[batch_index : batch_index + 1],
                    instance_targets[
                        batch_index : batch_index + 1,
                        instance_index,
                    ],
                    verified_object_masks[
                        batch_index : batch_index + 1,
                        instance_index,
                    ],
                )
            )
        losses.append(torch.stack(rows))
    per_instance = torch.stack(losses)
    return per_instance.mean(), per_instance


def paired_prompt_invariance_loss(
    prediction: torch.Tensor,
    verified_positive_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[0] % 2 != 0:
        raise ValueError("prompt-invariance batch must contain exact pairs")
    if verified_positive_mask.shape != prediction.shape[:2]:
        raise ValueError("verified positive mask must be [B,N]")
    if not torch.equal(
        verified_positive_mask[0::2], verified_positive_mask[1::2]
    ):
        raise ValueError("watch/write pair masks differ")
    return _masked_mse(
        prediction[0::2],
        prediction[1::2],
        verified_positive_mask[0::2],
    )


def explicit_negative_addition_loss(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    explicit_negative_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize only positive additions over frozen v5r4 on TV/Desk/Board."""

    addition = torch.relu(prediction - frozen_prediction)
    zeros = torch.zeros_like(addition)
    return _masked_mse(addition, zeros, explicit_negative_mask)


def preservation_loss(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != frozen_prediction.shape:
        raise ValueError("preservation tensors differ")
    return (prediction - frozen_prediction).square().mean()


def all_sittable_objective(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    physical = physical_prediction(prediction, mean, std)
    frozen_physical = physical_prediction(frozen_prediction, mean, std)
    primary, per_instance = equal_instance_macro_loss(
        physical,
        batch["instance_targets"],
        batch["verified_object_mask"],
    )
    union = _masked_mse(
        physical,
        batch["all_target"],
        batch["verified_positive_mask"],
    )
    environment = _masked_mse(
        physical,
        batch["all_target"],
        batch["environment_aux_mask"],
    )
    negative = explicit_negative_addition_loss(
        physical,
        frozen_physical,
        batch["explicit_negative_mask"],
    )
    prompt = paired_prompt_invariance_loss(
        physical,
        batch["verified_positive_mask"],
    )
    return {
        "instance_macro_primary": primary,
        "per_instance_primary": per_instance,
        "verified_union": union,
        "environment_auxiliary": environment,
        "explicit_negative_addition": negative,
        "paired_prompt_invariance": prompt,
    }


__all__ = [
    "all_sittable_objective",
    "equal_instance_macro_loss",
    "explicit_negative_addition_loss",
    "paired_prompt_invariance_loss",
    "physical_prediction",
    "preservation_loss",
]

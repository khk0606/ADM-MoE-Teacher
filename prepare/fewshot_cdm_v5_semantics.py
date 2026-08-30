"""Version-5 prompt policy and differentiable semantic priors.

The motion-derived contact tensor remains the only dense GT target.  The Sit
multi-candidate term below is deliberately a weak *semantic prior*: it asks a
Sit prompt to leave a bounded, non-dominant Bed candidate without pretending
that a Chair-sitting motion is a Bed-sitting motion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


PROMPT_POLICY_ID = "object_agnostic_action_v5"
PROMPT_BY_TARGET = {
    "chair": "Sit somewhere.",
    "bed": "Lie down somewhere.",
    "whiteboard": "Write on a nearby vertical surface with the right hand.",
}
FORBIDDEN_PROMPT_WORDS = ("chair", "bed", "whiteboard", "tv")
RIGHT_WRIST_CHANNEL = 5


def prompt_for_target(target: str) -> str:
    try:
        prompt = PROMPT_BY_TARGET[target]
    except KeyError as exc:
        raise ValueError(f"unknown target for v5 prompt policy: {target}") from exc
    lowered = prompt.lower()
    leaked = [word for word in FORBIDDEN_PROMPT_WORDS if word in lowered]
    if leaked:
        raise AssertionError(f"v5 prompt leaks scene object names: {leaked}")
    return prompt


def _top_fraction_mean(values: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("cannot score an empty candidate region")
    count = max(1, int(np.ceil(values.numel() * fraction)))
    return torch.topk(values.reshape(-1), k=count, largest=True).values.mean()


def _straight_through_unit_clamp(values: torch.Tensor) -> torch.Tensor:
    """Use bounded affordance values without creating dead gradient zones.

    The semantic prior is evaluated in raw affordance space, so its forward
    values must remain in [0, 1].  A regular clamp has zero derivative when a
    noisy diffusion prediction lies below zero or above one, exactly where the
    weak Sit/Bed prior must be able to correct it.  This expression has the
    same forward value as ``values.clamp(0, 1)`` and an identity backward
    derivative.
    """

    clipped = values.clamp(0.0, 1.0)
    return values + (clipped - values).detach()


def sit_multicandidate_loss(
    prediction_normalized: torch.Tensor,
    instance_ids: torch.Tensor,
    targets: List[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    bed_any_min: float,
    bed_any_max: float,
    bed_pelvis_min: float,
    bed_pelvis_max: float,
    chair_bed_margin: float,
) -> Dict[str, torch.Tensor]:
    """Bound a weak Bed candidate for Sit while keeping Chair primary.

    Only Chair/Sit rows participate.  No Bed contact labels are copied into the
    Chair GT and the ordinary diffusion loss is unchanged.  Hinge bands avoid
    both zero Bed activation and an incorrect Bed takeover.
    """

    if prediction_normalized.ndim != 3 or prediction_normalized.shape[-1] != 6:
        raise ValueError("prediction must be [B,N,6]")
    if instance_ids.shape != prediction_normalized.shape[:2]:
        raise ValueError("instance_ids must be [B,N]")
    if len(targets) != prediction_normalized.shape[0]:
        raise ValueError("targets length differs from batch")
    if not (0.0 <= bed_any_min < bed_any_max <= 1.0):
        raise ValueError("invalid Bed any-joint band")
    if not (0.0 <= bed_pelvis_min < bed_pelvis_max <= 1.0):
        raise ValueError("invalid Bed pelvis band")
    if chair_bed_margin <= 0.0:
        raise ValueError("chair_bed_margin must be positive")

    denormalized = prediction_normalized * std + mean
    raw = _straight_through_unit_clamp(denormalized)
    zero = prediction_normalized.sum() * 0.0
    band = zero
    primary = zero
    count = 0
    metrics = []
    for index, target in enumerate(targets):
        if target != "chair":
            continue
        chair_mask = instance_ids[index] == 1
        bed_mask = instance_ids[index] == 2
        if not torch.any(chair_mask) or not torch.any(bed_mask):
            raise ValueError("Sit semantic prior requires Chair and Bed points")
        chair_any = _top_fraction_mean(raw[index, chair_mask].amax(dim=-1))
        bed_any = _top_fraction_mean(raw[index, bed_mask].amax(dim=-1))
        chair_pelvis = _top_fraction_mean(raw[index, chair_mask, 0])
        bed_pelvis = _top_fraction_mean(raw[index, bed_mask, 0])
        band = band + (
            F.relu(raw.new_tensor(bed_any_min) - bed_any)
            + F.relu(bed_any - raw.new_tensor(bed_any_max))
            + F.relu(raw.new_tensor(bed_pelvis_min) - bed_pelvis)
            + F.relu(bed_pelvis - raw.new_tensor(bed_pelvis_max))
        )
        margin = raw.new_tensor(chair_bed_margin)
        primary = primary + F.relu(margin + bed_any - chair_any)
        primary = primary + F.relu(margin + bed_pelvis - chair_pelvis)
        count += 1
        metrics.append((chair_any, bed_any, chair_pelvis, bed_pelvis))

    if count == 0:
        return {
            "total": zero,
            "bed_band": zero,
            "chair_primary": zero,
            "count": zero.detach(),
        }
    band = band / float(count)
    primary = primary / float(count)
    stacked = torch.stack([torch.stack(row) for row in metrics]).mean(dim=0)
    return {
        "total": band + primary,
        "bed_band": band,
        "chair_primary": primary,
        "count": raw.new_tensor(float(count)),
        "chair_any_top10": stacked[0],
        "bed_any_top10": stacked[1],
        "chair_pelvis_top10": stacked[2],
        "bed_pelvis_top10": stacked[3],
    }

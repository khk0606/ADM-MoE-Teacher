"""Routing utilities for history-conditioned full-scene affordance maps."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryFullMapRouter(nn.Module):
    """Select or combine furniture-specific full-scene affordance maps.

    Inputs:
        candidate_maps:
            [B,K,N,J]. One full-scene affordance map per furniture.

        object_logits:
            [B,K]. Furniture-selection logits from HistoryFullMapMoE.

    Routing modes:
        soft:
            Weighted sum using softmax probabilities. Used for training.

        hard:
            Select only the argmax furniture map. Used for inference.

        straight-through:
            Forward output is identical to hard routing, but gradients
            propagate through softmax probabilities.
    """

    def __init__(self, num_candidates: int = 3) -> None:
        super().__init__()

        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")

        self.num_candidates = int(num_candidates)

    def _validate(
        self,
        candidate_maps: torch.Tensor,
        object_logits: torch.Tensor,
        temperature: float,
    ) -> None:
        if candidate_maps.ndim != 4:
            raise ValueError(
                "candidate_maps must be [B,K,N,J], "
                f"got {tuple(candidate_maps.shape)}"
            )

        if object_logits.ndim != 2:
            raise ValueError(
                "object_logits must be [B,K], "
                f"got {tuple(object_logits.shape)}"
            )

        batch_size, num_candidates, _, _ = candidate_maps.shape

        if num_candidates != self.num_candidates:
            raise ValueError(
                f"candidate_maps K must be {self.num_candidates}, "
                f"got {num_candidates}"
            )

        if object_logits.shape != (batch_size, self.num_candidates):
            raise ValueError(
                "object_logits shape mismatch: "
                f"expected {(batch_size, self.num_candidates)}, "
                f"got {tuple(object_logits.shape)}"
            )

        if not torch.is_floating_point(candidate_maps):
            raise TypeError("candidate_maps must be floating point")

        if not torch.is_floating_point(object_logits):
            raise TypeError("object_logits must be floating point")

        if not torch.isfinite(candidate_maps).all():
            raise ValueError("candidate_maps contains NaN/Inf")

        if not torch.isfinite(object_logits).all():
            raise ValueError("object_logits contains NaN/Inf")

        if bool((candidate_maps < 0.0).any()):
            raise ValueError("candidate_maps contains values below zero")

        if bool((candidate_maps > 1.0).any()):
            raise ValueError("candidate_maps contains values above one")

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

    def forward(
        self,
        candidate_maps: torch.Tensor,
        object_logits: torch.Tensor,
        hard: bool = False,
        straight_through: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if hard and straight_through:
            raise ValueError(
                "hard and straight_through are mutually exclusive"
            )

        self._validate(
            candidate_maps,
            object_logits,
            temperature,
        )

        probabilities = F.softmax(
            object_logits / temperature,
            dim=-1,
        )

        selected_index = probabilities.argmax(dim=-1)

        hard_weights = F.one_hot(
            selected_index,
            num_classes=self.num_candidates,
        ).to(candidate_maps.dtype)

        if hard:
            routing_weights = hard_weights
            routing_mode = "hard"

        elif straight_through:
            routing_weights = (
                hard_weights
                - probabilities.detach()
                + probabilities
            )
            routing_mode = "straight_through"

        else:
            routing_weights = probabilities
            routing_mode = "soft"

        # [B,K,N,J] × [B,K,1,1] → [B,N,J]
        map_star = (
            candidate_maps
            * routing_weights[:, :, None, None]
        ).sum(dim=1)

        batch_indices = torch.arange(
            candidate_maps.shape[0],
            device=candidate_maps.device,
        )

        # 디버깅 및 inference 검증용 argmax map
        selected_candidate_map = candidate_maps[
            batch_indices,
            selected_index,
        ]

        return {
            "map_star": map_star,
            "probabilities": probabilities,
            "routing_weights": routing_weights,
            "selected_index": selected_index,
            "selected_candidate_map": selected_candidate_map,
            "candidate_maps": candidate_maps,
            "routing_mode": routing_mode,
        }
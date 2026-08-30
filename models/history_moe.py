"""Lightweight HistoryMoE pilot for object-level point-cloud routing."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryMoE(nn.Module):
    """Predict one object logit per candidate furniture instance.

    The pilot uses three candidate-specific experts in fixed instance-ID order
    [Chair=1, Bed=2, Whiteboard=3]. Each expert receives:

    - frozen CLIP text feature;
    - raw point-cloud object/global PointNet-style features;
    - history state [start_x, start_y, direction_x, direction_y];
    - relative object geometry and direction alignment.

    ``scene_points`` must contain Z-up XYZ and RGB in [0,1].
    """

    def __init__(
        self,
        candidate_ids: Sequence[int] = (1, 2, 3),
        point_input_dim: int = 6,
        text_input_dim: int = 512,
        point_dim: int = 128,
        text_dim: int = 128,
        history_dim: int = 64,
        global_dim: int = 128,
        candidate_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if len(candidate_ids) == 0:
            raise ValueError("candidate_ids must not be empty")

        self.register_buffer(
            "candidate_ids",
            torch.tensor(list(candidate_ids), dtype=torch.long),
            persistent=True,
        )
        self.point_dim = int(point_dim)

        self.point_encoder = nn.Sequential(
            nn.Linear(point_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, point_dim),
            nn.GELU(),
        )
        self.text_encoder = MLP(text_input_dim, text_dim, text_dim)
        self.history_encoder = MLP(4, history_dim, history_dim)
        self.global_encoder = MLP(2 * point_dim, global_dim, global_dim)
        self.candidate_embedding = nn.Embedding(
            len(candidate_ids), candidate_embedding_dim
        )

        # center_xy(2), extent_xyz(3), delta_xy(2), distance(1),
        # direction_alignment(1), point_fraction(1)
        geometry_dim = 10
        expert_input_dim = (
            2 * point_dim
            + text_dim
            + history_dim
            + global_dim
            + candidate_embedding_dim
            + geometry_dim
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(expert_input_dim),
                    nn.Linear(expert_input_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                for _ in candidate_ids
            ]
        )

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_ids.numel())

    def _validate(
        self,
        scene_points: torch.Tensor,
        instance_ids: torch.Tensor,
        text_feature: torch.Tensor,
        history_state: torch.Tensor,
    ) -> None:
        if scene_points.ndim != 3 or scene_points.shape[-1] != 6:
            raise ValueError("scene_points must be [B,N,6]")
        if instance_ids.shape != scene_points.shape[:2]:
            raise ValueError("instance_ids must be [B,N]")
        if text_feature.ndim != 2 or text_feature.shape[0] != scene_points.shape[0]:
            raise ValueError("text_feature must be [B,D]")
        if history_state.shape != (scene_points.shape[0], 4):
            raise ValueError("history_state must be [B,4]")
        if not torch.isfinite(scene_points).all():
            raise ValueError("scene_points contains NaN/Inf")
        if not torch.isfinite(text_feature).all():
            raise ValueError("text_feature contains NaN/Inf")
        if not torch.isfinite(history_state).all():
            raise ValueError("history_state contains NaN/Inf")

    def forward(
        self,
        scene_points: torch.Tensor,
        instance_ids: torch.Tensor,
        text_feature: torch.Tensor,
        history_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate(scene_points, instance_ids, text_feature, history_state)

        point_feature = self.point_encoder(scene_points)  # [B,N,D]
        global_mean = point_feature.mean(dim=1)
        global_max = point_feature.max(dim=1).values
        global_context = self.global_encoder(
            torch.cat([global_mean, global_max], dim=-1)
        )
        text_context = self.text_encoder(text_feature.float())

        # Normalize only the direction part of history. Start position remains
        # in the same chair-local metric coordinates as the scene.
        start_xy = history_state[:, 0:2]
        direction_xy = F.normalize(history_state[:, 2:4], dim=-1, eps=1e-6)
        normalized_history = torch.cat([start_xy, direction_xy], dim=-1)
        history_context = self.history_encoder(normalized_history)

        candidate_ids = self.candidate_ids.to(instance_ids.device)
        candidate_masks = torch.stack(
            [instance_ids == candidate_id for candidate_id in candidate_ids],
            dim=1,
        )  # [B,K,N]

        logits = []
        object_features = []
        geometry_features = []
        batch_size, num_points, _ = scene_points.shape
        xyz = scene_points[:, :, 0:3]

        for candidate_index in range(self.num_candidates):
            batch_object_feature = []
            batch_geometry = []
            for batch_index in range(batch_size):
                mask = candidate_masks[batch_index, candidate_index]
                if not bool(mask.any()):
                    raise ValueError(
                        "candidate has no points: instance_id="
                        + str(int(candidate_ids[candidate_index].item()))
                    )

                object_point_feature = point_feature[batch_index, mask]
                object_mean = object_point_feature.mean(dim=0)
                object_max = object_point_feature.max(dim=0).values
                batch_object_feature.append(
                    torch.cat([object_mean, object_max], dim=-1)
                )

                object_xyz = xyz[batch_index, mask]
                center = object_xyz.mean(dim=0)
                extent = object_xyz.max(dim=0).values - object_xyz.min(dim=0).values
                delta_xy = center[0:2] - start_xy[batch_index]
                distance = torch.linalg.norm(delta_xy).clamp_min(1e-6)
                object_direction = delta_xy / distance
                alignment = (object_direction * direction_xy[batch_index]).sum()
                point_fraction = mask.to(scene_points.dtype).sum() / float(num_points)
                geometry = torch.cat(
                    [
                        center[0:2],
                        extent,
                        delta_xy,
                        distance.reshape(1),
                        alignment.reshape(1),
                        point_fraction.reshape(1),
                    ],
                    dim=0,
                )
                batch_geometry.append(geometry)

            object_feature = torch.stack(batch_object_feature, dim=0)
            geometry = torch.stack(batch_geometry, dim=0)
            candidate_index_tensor = torch.full(
                (batch_size,),
                candidate_index,
                dtype=torch.long,
                device=scene_points.device,
            )
            candidate_embedding = self.candidate_embedding(candidate_index_tensor)

            expert_input = torch.cat(
                [
                    object_feature,
                    text_context,
                    history_context,
                    global_context,
                    candidate_embedding,
                    geometry,
                ],
                dim=-1,
            )
            logit = self.experts[candidate_index](expert_input).squeeze(-1)
            logits.append(logit)
            object_features.append(object_feature)
            geometry_features.append(geometry)

        logits_tensor = torch.stack(logits, dim=-1)
        probabilities = F.softmax(logits_tensor, dim=-1)
        return {
            "logits": logits_tensor,
            "probabilities": probabilities,
            "selected_index": probabilities.argmax(dim=-1),
            "candidate_masks": candidate_masks,
            "object_features": torch.stack(object_features, dim=1),
            "geometry_features": torch.stack(geometry_features, dim=1),
        }

"""History-conditioned MoE for full-scene affordance-map generation."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryFullMapMoE(nn.Module):
    """Generate one full-scene affordance map per candidate furniture.

    Inputs:
        scene_points:
            [B,N,6], chair-local Z-up XYZ and RGB in [0,1].

        instance_ids:
            [B,N], scene point instance IDs.

        text_feature:
            [B,512], frozen CLIP text feature.

        history_state:
            [B,4] containing
            [start_x, start_y, direction_x, direction_y].

        base_affordance:
            [B,N,6], frozen ADM base affordance score in [0,1].

    Outputs:
        logits:
            [B,K], furniture-selection logits.

        probabilities:
            [B,K], furniture-selection probabilities.

        candidate_maps:
            [B,K,N,6], one full-scene affordance map per furniture.

        candidate_residuals:
            [B,K,N,6], residual logits applied to the base map.
    """

    def __init__(
        self,
        candidate_ids: Sequence[int] = (1, 2, 3),
        point_input_dim: int = 6,
        base_input_dim: int = 6,
        text_input_dim: int = 512,
        contact_dim: int = 6,
        point_dim: int = 128,
        base_dim: int = 64,
        text_dim: int = 128,
        history_dim: int = 64,
        global_dim: int = 128,
        instance_embedding_dim: int = 16,
        candidate_embedding_dim: int = 16,
        fused_point_dim: int = 128,
        num_instance_embeddings: int = 16,
        affordance_eps: float = 1e-4,
    ) -> None:
        super().__init__()

        if len(candidate_ids) == 0:
            raise ValueError("candidate_ids must not be empty")

        if affordance_eps <= 0.0 or affordance_eps >= 0.5:
            raise ValueError("affordance_eps must be in (0,0.5)")

        self.register_buffer(
            "candidate_ids",
            torch.tensor(candidate_ids, dtype=torch.long),
            persistent=True,
        )

        self.contact_dim = int(contact_dim)
        self.num_instance_embeddings = int(num_instance_embeddings)
        self.affordance_eps = float(affordance_eps)

        # Point별 scene XYZRGB feature
        self.point_encoder = nn.Sequential(
            nn.Linear(point_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, point_dim),
            nn.GELU(),
        )

        # Point별 base affordance feature
        self.base_encoder = nn.Sequential(
            nn.Linear(base_input_dim, 32),
            nn.GELU(),
            nn.Linear(32, base_dim),
            nn.GELU(),
        )

        # Text와 history global feature
        self.text_encoder = MLP(
            text_input_dim,
            text_dim,
            text_dim,
        )

        self.history_encoder = MLP(
            4,
            history_dim,
            history_dim,
        )

        # Scene point feature의 mean/max pooling
        self.global_encoder = MLP(
            2 * point_dim,
            global_dim,
            global_dim,
        )

        self.instance_embedding = nn.Embedding(
            num_instance_embeddings,
            instance_embedding_dim,
        )

        self.candidate_embedding = nn.Embedding(
            len(candidate_ids),
            candidate_embedding_dim,
        )

        # Point-history geometry:
        # relative_xy(2), distance(1),
        # forward_alignment(1), lateral_alignment(1)
        point_history_geometry_dim = 5

        point_context_dim = (
            point_dim
            + base_dim
            + text_dim
            + history_dim
            + global_dim
            + instance_embedding_dim
            + point_history_geometry_dim
        )

        self.point_fusion = nn.Sequential(
            nn.LayerNorm(point_context_dim),
            nn.Linear(point_context_dim, 256),
            nn.GELU(),
            nn.Linear(256, fused_point_dim),
            nn.GELU(),
        )

        # 가구별 full-map expert
        map_expert_input_dim = (
            fused_point_dim
            + candidate_embedding_dim
        )

        self.map_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(map_expert_input_dim),
                    nn.Linear(map_expert_input_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, contact_dim),
                )
                for _ in candidate_ids
            ]
        )

        # 마지막 residual layer를 0으로 초기화한다.
        # 따라서 학습 시작 시 candidate map은 base map과 거의 동일하다.
        for expert in self.map_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

        # Gate용 가구 geometry:
        # center_xy(2), extent_xyz(3), delta_xy(2),
        # distance(1), direction_alignment(1), point_fraction(1)
        candidate_geometry_dim = 10

        gate_input_dim = (
            2 * point_dim
            + text_dim
            + history_dim
            + global_dim
            + candidate_embedding_dim
            + candidate_geometry_dim
        )

        self.gate_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(gate_input_dim),
                    nn.Linear(gate_input_dim, 256),
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

    def _validate_inputs(
        self,
        scene_points: torch.Tensor,
        instance_ids: torch.Tensor,
        text_feature: torch.Tensor,
        history_state: torch.Tensor,
        base_affordance: torch.Tensor,
    ) -> None:
        if scene_points.ndim != 3 or scene_points.shape[-1] != 6:
            raise ValueError("scene_points must be [B,N,6]")

        batch_size, num_points, _ = scene_points.shape

        if instance_ids.shape != (batch_size, num_points):
            raise ValueError("instance_ids must be [B,N]")

        if text_feature.ndim != 2:
            raise ValueError("text_feature must be [B,D]")

        if text_feature.shape[0] != batch_size:
            raise ValueError("text_feature batch size mismatch")

        if history_state.shape != (batch_size, 4):
            raise ValueError("history_state must be [B,4]")

        if base_affordance.shape != (
            batch_size,
            num_points,
            self.contact_dim,
        ):
            raise ValueError(
                f"base_affordance must be [B,N,{self.contact_dim}]"
            )

        tensors = {
            "scene_points": scene_points,
            "text_feature": text_feature,
            "history_state": history_state,
            "base_affordance": base_affordance,
        }

        for name, value in tensors.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")

        if bool((instance_ids < 0).any()):
            raise ValueError("instance_ids contains negative values")

        if bool((instance_ids >= self.num_instance_embeddings).any()):
            raise ValueError(
                "instance_ids exceeds num_instance_embeddings"
            )

        if bool((base_affordance < 0.0).any()):
            raise ValueError("base_affordance contains values below zero")

        if bool((base_affordance > 1.0).any()):
            raise ValueError("base_affordance contains values above one")

    def _prepare_history(# Startpoint/history 4개 값을 단순히 모든 point에 복사하는 것뿐만 아니라, 각 point와의 공간 관계를 계산해서 넣음.
        self,
        scene_xyz: torch.Tensor,
        history_state: torch.Tensor,
    ):
        start_xy = history_state[:, 0:2]

        direction_xy = F.normalize(# 출발 방향 정규화
            history_state[:, 2:4],
            dim=-1,
            eps=1e-6,
        )

        normalized_history = torch.cat(
            [start_xy, direction_xy],
            dim=-1,
        )

        # 각 point가 Startpoint 기준으로 어디 있는지 계산
        relative_xy = (
            scene_xyz[:, :, 0:2]
            - start_xy[:, None, :]
        )

        distance_from_start = torch.linalg.norm(
            relative_xy,
            dim=-1,
            keepdim=True,
        )

        point_direction = F.normalize(
            relative_xy,
            dim=-1,
            eps=1e-6,
        )

        forward_alignment = (
            point_direction
            * direction_xy[:, None, :]
        ).sum(dim=-1, keepdim=True)

        lateral_alignment = (
            direction_xy[:, None, 0:1]
            * point_direction[:, :, 1:2]
            - direction_xy[:, None, 1:2]
            * point_direction[:, :, 0:1]
        )

        point_history_geometry = torch.cat(
            [
                relative_xy,
                distance_from_start,
                forward_alignment,
                lateral_alignment,
            ],
            dim=-1,
        )

        return (
            start_xy,
            direction_xy,
            normalized_history,
            point_history_geometry,
        )

    def _prepare_candidate_features(#Chair·Bed·Whiteboard 각각의 위치와 크기, 방향 정보를 계산
        self,
        point_feature: torch.Tensor,
        scene_xyz: torch.Tensor,
        instance_ids: torch.Tensor,
        start_xy: torch.Tensor,
        direction_xy: torch.Tensor,
    ):
        candidate_ids = self.candidate_ids.to(instance_ids.device)

        candidate_masks = torch.stack(
            [
                instance_ids == candidate_id
                for candidate_id in candidate_ids
            ],
            dim=1,
        )

        # [B,K,1]
        candidate_counts = candidate_masks.sum(
            dim=2,
            keepdim=True,
        )

        if bool((candidate_counts == 0).any()):
            raise ValueError("at least one candidate furniture has no points")

        mask_float = candidate_masks.to(point_feature.dtype)

        # Candidate object point-feature mean
        object_mean = (
            point_feature[:, None, :, :]
            * mask_float[:, :, :, None]
        ).sum(dim=2)

        object_mean = object_mean / candidate_counts.to(point_feature.dtype)

        # Candidate object point-feature max
        minimum_value = torch.finfo(point_feature.dtype).min

        masked_point_feature = point_feature[:,None,:,:,].masked_fill(
            ~candidate_masks[:, :, :, None],
            minimum_value,
        )

        object_max = masked_point_feature.max(dim=2).values

        # Candidate XYZ center
        object_center = (
            scene_xyz[:, None, :, :]
            * mask_float[:, :, :, None]
        ).sum(dim=2)

        object_center = object_center / candidate_counts.to(scene_xyz.dtype)

        # Candidate XYZ extent
        maximum_value = torch.finfo(scene_xyz.dtype).max

        object_min = scene_xyz[:, None, :, :].masked_fill(
            ~candidate_masks[:, :, :, None],
            maximum_value,
        ).min(dim=2).values

        object_max_xyz = scene_xyz[:, None, :, :].masked_fill(
            ~candidate_masks[:, :, :, None],
            -maximum_value,
        ).max(dim=2).values

        object_extent = object_max_xyz - object_min

        # Startpoint에서 가구까지의 vector
        delta_xy = (
            object_center[:, :, 0:2]
            - start_xy[:, None, :]
        )
        # Startpoint에서 가구까지의 거라
        candidate_distance = torch.linalg.norm(
            delta_xy,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        candidate_direction = (
            delta_xy / candidate_distance
        )
        #History 방향과 가구 방향 일치도
        direction_alignment = (
            candidate_direction
            * direction_xy[:, None, :]
        ).sum(dim=-1, keepdim=True)

        point_fraction = (
            candidate_counts.to(scene_xyz.dtype) / float(scene_xyz.shape[1])
        )
        #최종 candidate geometry
        candidate_geometry = torch.cat(
            [
                object_center[:, :, 0:2],
                object_extent,
                delta_xy,
                candidate_distance,
                direction_alignment,
                point_fraction,
            ],
            dim=-1,
        )

        object_feature = torch.cat([object_mean, object_max],dim=-1,)
        return (candidate_masks,object_feature,candidate_geometry,)

    def forward(
        self,
        scene_points: torch.Tensor,
        instance_ids: torch.Tensor,
        text_feature: torch.Tensor,
        history_state: torch.Tensor,
        base_affordance: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(
            scene_points,
            instance_ids,
            text_feature,
            history_state,
            base_affordance,
        )

        batch_size, num_points, _ = scene_points.shape
        scene_xyz = scene_points[:, :, 0:3]

        # Scene Encoding
        point_feature = self.point_encoder(scene_points)
        # Base Encoding
        base_feature = self.base_encoder(base_affordance)

        # Scene global feature
        global_mean = point_feature.mean(dim=1)
        global_max = point_feature.max(dim=1).values

        global_context = self.global_encoder(torch.cat([global_mean, global_max],dim=-1,))

        # Text feature
        text_context = self.text_encoder(text_feature.float())

        # History feature
        (
            start_xy,
            direction_xy,
            normalized_history,
            point_history_geometry,
        ) = self._prepare_history(
            scene_xyz,
            history_state,
        )
        history_context = self.history_encoder(normalized_history)

        # Point instance embedding
        instance_feature = self.instance_embedding(instance_ids)

        # Global condition을 모든 point로 확장
        text_per_point = text_context[:, None, :].expand(-1,num_points,-1,)
        history_per_point = history_context[:,None,:,].expand(-1,num_points,-1,)
        global_per_point = global_context[:,None,:,].expand(-1,num_points,-1,)

        point_context = torch.cat(
            [
                point_feature,
                base_feature,
                text_per_point,
                history_per_point,
                global_per_point,
                instance_feature,
                point_history_geometry,
            ],
            dim=-1,
        )
        fused_point_feature = self.point_fusion(point_context)

        # Candidate별 pooled feature와 geometry
        (
            candidate_masks,
            object_feature,
            candidate_geometry,
        ) = self._prepare_candidate_features(
            point_feature,
            scene_xyz,
            instance_ids,
            start_xy,
            direction_xy,
        )

        candidate_indices = torch.arange(
            self.num_candidates,
            device=scene_points.device,
        )

        candidate_embeddings = self.candidate_embedding(candidate_indices)

        # Furniture-selection Gate

        gate_logits = []

        for candidate_index in range(self.num_candidates):
            candidate_embedding = candidate_embeddings[candidate_index].unsqueeze(0).expand(batch_size,-1,)

            gate_input = torch.cat(
                [
                    object_feature[:,candidate_index,:,],
                    text_context,
                    history_context,
                    global_context,
                    candidate_embedding,
                    candidate_geometry[:,candidate_index,:,],
                ],
                dim=-1,
            )

            candidate_logit = self.gate_experts[candidate_index](gate_input).squeeze(-1)

            gate_logits.append(candidate_logit)

        logits = torch.stack(
            gate_logits,
            dim=-1,
        )

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        # Candidate full-scene affordance maps

        clipped_base = base_affordance.clamp(
            min=self.affordance_eps,
            max=1.0 - self.affordance_eps,
        )

        base_logits = torch.log(clipped_base) - torch.log1p(
            -clipped_base
        )

        candidate_maps = []
        candidate_residuals = []

        for candidate_index in range(
            self.num_candidates
        ):
            candidate_embedding = candidate_embeddings[candidate_index].view(1,1,-1,).expand(batch_size,num_points,-1,)
            map_input = torch.cat([fused_point_feature,candidate_embedding,],dim=-1,)
            residual = self.map_experts[candidate_index](map_input)

            # 곱셈 mask가 아닌 additive logit residual.
            # 따라서 base map에 없던 이동 경로도 생성할 수 있다.
            candidate_map = torch.sigmoid(base_logits + residual)

            candidate_residuals.append(residual)
            candidate_maps.append(candidate_map)

        candidate_maps = torch.stack(candidate_maps,dim=1,)
        candidate_residuals = torch.stack(candidate_residuals,dim=1,)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "selected_index": probabilities.argmax(dim=-1),
            "candidate_maps": candidate_maps,
            "candidate_residuals": candidate_residuals,
            "candidate_masks": candidate_masks,
            "point_feature": point_feature,
            "fused_point_feature": fused_point_feature,
        }
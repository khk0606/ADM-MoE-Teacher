"""History-conditioned object routing for an ADM affordance map.

This module does not predict object weights. It consumes logits produced by a
future HistoryMoE and performs calibrated candidate-map construction plus
soft/hard routing.

The default ``candidate_region`` mode assigns nearby scene points to a
candidate furniture footprint and suppresses every non-selected region.  This
prevents a strong affordance halo on Environment points near one object from
surviving when another object is selected.  ``legacy_background`` reproduces
the earlier pilot behavior for ablation only.

candidate region 생성과 map routing
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# 이후에 HistoryMoE와 함께 연결하고 GPU Tensor 연산과 역전파를 사용하기 위해 nn.Module로 구현
class HistoryAffordanceRouter(nn.Module):
    """Build and route object-calibrated affordance candidate maps.

    Args:
        candidate_ids: Point-cloud instance IDs in MoE output order.
        quantile: Per-object/per-joint calibration quantile.
        eps: Minimum quantile scale.
        routing_mode: ``candidate_region`` (default) or
            ``legacy_background`` (ablation).
        context_radius: Maximum chair-local XY distance in metres from a
            candidate footprint for Environment/reference points to join its
            region.

    Input shapes:
        base_affordance: [B, N, J], contact scores in [0, 1].
        instance_ids: [B, N].
        object_logits: [B, K], where K == len(candidate_ids).
        scene_xyz: [B, N, 3], required in ``candidate_region`` mode.

    ``legacy_background`` preserves every point not belonging to a candidate
    object.  ``candidate_region`` instead constructs disjoint spatial regions
    and sets unassigned points to zero in the routed affordance.  The original
    point cloud is never removed; only its affordance channels are gated.
    """

    def __init__(
        self,
        candidate_ids: Sequence[int] = (1, 2, 3),# Chair=1, Bed=2, Whiteboard=3
        quantile: float = 0.95,# 각 candidate object별로 affordance score를 계산할 때, 상위 5% point의 score를 기준으로 normalization
        eps: float = 1e-6,# affordance score를 normalization할 때, 0으로 나누는 것을 방지하기 위한 최소값
        routing_mode: str = "candidate_region",# candidate_region: candidate footprint로부터 얼마나 떨어진 point까지 candidate region에 포함시킬지 결정
        context_radius: float = 1.0,# candidate footprint로부터 얼마나 떨어진 point까지 candidate region에 포함시킬지 결정
    ) -> None:
        super().__init__()
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        if len(candidate_ids) == 0:
            raise ValueError("candidate_ids must not be empty")
        if routing_mode not in ("candidate_region", "legacy_background"):
            raise ValueError(
                "routing_mode must be candidate_region or legacy_background"
            )
        if context_radius <= 0.0:
            raise ValueError("context_radius must be positive")

        self.register_buffer(# candidate_ids를 GPU Tensor로 등록하여, 모델이 GPU로 이동할 때 자동으로 이동하도록 함/ 학습되지는 않지만 모델과 함께 GPU 이동 및 저장
            "candidate_ids",
            torch.tensor(list(candidate_ids), dtype=torch.long),
            persistent=True,
        )
        self.quantile = float(quantile)
        self.eps = float(eps)
        self.routing_mode = routing_mode
        self.context_radius = float(context_radius)

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_ids.numel())

    def _validate(# 입력 검사
        self,
        base_affordance: torch.Tensor,
        instance_ids: torch.Tensor,
        object_logits: torch.Tensor,
        scene_xyz: torch.Tensor = None,
    ) -> None:
        if base_affordance.ndim != 3:
            raise ValueError(
                "base_affordance must be [B,N,J], got "
                + str(tuple(base_affordance.shape))
            )
        if instance_ids.ndim != 2:
            raise ValueError(
                "instance_ids must be [B,N], got "
                + str(tuple(instance_ids.shape))
            )
        if object_logits.ndim != 2:
            raise ValueError(
                "object_logits must be [B,K], got "
                + str(tuple(object_logits.shape))
            )

        batch_size, num_points, _ = base_affordance.shape
        if instance_ids.shape != (batch_size, num_points):
            raise ValueError("base_affordance/instance_ids shape mismatch")
        if object_logits.shape != (batch_size, self.num_candidates):# 현재 후보가 3개이므로 object_logits의 shape은 [B,3]이어야 함
            raise ValueError(
                "object_logits must have K=" + str(self.num_candidates)
            )
        if not torch.is_floating_point(base_affordance):
            raise TypeError("base_affordance must be floating point")
        if not torch.is_floating_point(object_logits):
            raise TypeError("object_logits must be floating point")
        if not torch.isfinite(base_affordance).all():
            raise ValueError("base_affordance contains NaN/Inf")
        if not torch.isfinite(object_logits).all():
            raise ValueError("object_logits contains NaN/Inf")
        if self.routing_mode == "candidate_region":# candidate_region 모드에서는 scene_xyz가 필수
            if scene_xyz is None:
                raise ValueError(
                    "scene_xyz is required in candidate_region mode"
                )
            if scene_xyz.shape != (batch_size, num_points, 3):
                raise ValueError("scene_xyz must be [B,N,3]")
            if not torch.is_floating_point(scene_xyz):
                raise TypeError("scene_xyz must be floating point")
            if not torch.isfinite(scene_xyz).all():
                raise ValueError("scene_xyz contains NaN/Inf")

    def _build_candidate_regions(# Chair, Bed, Whiteboard 공간 영역을 생성하여, 각 point가 어느 candidate footprint에 속하는지 결정
        self,
        scene_xyz: torch.Tensor,
        instance_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Assign scene points to disjoint candidate footprint regions.

        Distance is measured in chair-local XY to each candidate's axis-aligned
        footprint.  A point joins its nearest footprint only when the distance
        is at most ``context_radius``. Candidate instance points are forced to
        own themselves.  The result is a deterministic, non-learned ownership
        mask used only to expand object-level weights to point-level weights.
        """
        batch_size, num_candidates, num_points = instance_masks.shape
        xy = scene_xyz[:, :, 0:2]# XY 좌표만 사용(바닥 평면이 XY 평면)
        all_distances = []

        for batch_index in range(batch_size):
            batch_distances = []
            for candidate_index in range(num_candidates):# 가구별로 반복
                mask = instance_masks[batch_index, candidate_index]# 현재 가구의 point 만 선택
                if not bool(mask.any()):
                    raise ValueError(
                        "candidate has no points while building regions"
                    )
                object_xy = xy[batch_index, mask]# 전체 8192 point 중 현재 가구 point의 XY 좌표만 선택
                lower = object_xy.min(dim=0).values
                upper = object_xy.max(dim=0).values
                below = (lower[None, :] - xy[batch_index]).clamp_min(0.0)# 가구 사각형의 왼쪽 또는 아래쪽으로 얼마나 벗어났는지 계산
                above = (xy[batch_index] - upper[None, :]).clamp_min(0.0)# 오른쪽 또는 위쪽으로 얼마나 벗어났는지 계산
                outside = below + above # 최종적으로 사각형 밖으로 얼마나 벗어났는지 합친 값
                batch_distances.append(torch.linalg.norm(outside, dim=-1))# 마지막으로 거리로 변환
            all_distances.append(torch.stack(batch_distances, dim=0))

        footprint_distances = torch.stack(all_distances, dim=0)  # [B,K,N]
        nearest_distance, ownership_index = footprint_distances.min(dim=1)# 가장 가까운 가구 찾기

        # Candidate mesh points always belong to their own candidate even if two axis-aligned footprint boxes overlap.
        for candidate_index in range(num_candidates):
            ownership_index = torch.where(# 이 point가 실제 Chair mesh point라면 → 무조건 Chair region에 넣기
                instance_masks[:, candidate_index],
                torch.full_like(ownership_index, candidate_index),
                ownership_index,
            )

        within_context = nearest_distance <= self.context_radius # 1.0m 이내에 있는 point만 candidate region에 포함
        within_context = within_context | instance_masks.any(dim=1) # 가구 자체 point는 무조건 candidate region에 포함
        region_masks = torch.stack(
            [
                (ownership_index == candidate_index) & within_context
                for candidate_index in range(num_candidates)
            ],
            dim=1,
        )
        unassigned_mask = ~region_masks.any(dim=1) # 각 가구와의 거리가 1.0m 이상 떨어진 point는 candidate region에 포함되지 않음

        if bool((region_masks.sum(dim=1) > 1).any()):
            raise AssertionError("candidate regions must be disjoint")
        if region_masks.shape != (batch_size, num_candidates, num_points):
            raise AssertionError("candidate region shape mismatch")

        return {
            "region_masks": region_masks,
            "unassigned_mask": unassigned_mask,
            "ownership_index": ownership_index,
            "footprint_distances": footprint_distances,
        }

    def build_candidate_maps(# Base map을 가구별로 나누기
        self,
        base_affordance: torch.Tensor,
        instance_ids: torch.Tensor,
        scene_xyz: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Construct raw and q95-calibrated candidate-region maps."""
        candidate_ids = self.candidate_ids.to(instance_ids.device)
        instance_masks = torch.stack(# Instance ID로 가구 mask 만들기
            [instance_ids == candidate_id for candidate_id in candidate_ids],
            dim=1,
        )  # [B,K,N]
        instance_background_mask = ~instance_masks.any(dim=1)  # [B,N]

        if self.routing_mode == "candidate_region":
            if scene_xyz is None:
                raise ValueError("scene_xyz is required for candidate regions")
            region_data = self._build_candidate_regions(
                scene_xyz, instance_masks
            )
            region_masks = region_data["region_masks"]
            background_map = torch.zeros_like(base_affordance)
        else:
            region_masks = instance_masks
            region_data = {
                "region_masks": region_masks,
                "unassigned_mask": torch.zeros_like(instance_background_mask),
                "ownership_index": torch.full_like(instance_ids, -1),
                "footprint_distances": torch.empty(
                    base_affordance.shape[0],
                    self.num_candidates,
                    base_affordance.shape[1],
                    dtype=base_affordance.dtype,
                    device=base_affordance.device,
                ),
            }
            background_map = base_affordance * instance_background_mask[
                :, :, None
            ].to(base_affordance.dtype)

        batch_size, num_points, num_joints = base_affordance.shape
        raw_maps = []
        calibrated_maps = []
        quantile_scales = []

        for batch_index in range(batch_size):
            batch_raw = []
            batch_calibrated = []
            batch_scales = []
            for candidate_index in range(self.num_candidates):
                instance_mask = instance_masks[batch_index, candidate_index]
                region_mask = region_masks[batch_index, candidate_index]
                if not bool(instance_mask.any()):
                    raise ValueError(
                        "candidate instance has no points: id="
                        + str(int(candidate_ids[candidate_index].item()))
                    )

                # Preserve the original per-object q95 scale so the new region behavior can be compared directly with the earlier pilot.
                values = base_affordance[batch_index, instance_mask, :]  # [Ni,J]
                scale = torch.quantile(values, self.quantile, dim=0)
                scale = scale.clamp_min(self.eps)

                raw = base_affordance[batch_index] * region_mask[:, None].to( # candidate footprint에 속하는 point만 남기고 나머지는 0으로 만들기
                    base_affordance.dtype
                )
                calibrated = torch.clamp(
                    base_affordance[batch_index] / scale[None, :],
                    min=0.0,
                    max=1.0,
                )
                calibrated = calibrated * region_mask[:, None].to(
                    base_affordance.dtype
                )

                batch_raw.append(raw)
                batch_calibrated.append(calibrated)
                batch_scales.append(scale)

            raw_maps.append(torch.stack(batch_raw, dim=0))
            calibrated_maps.append(torch.stack(batch_calibrated, dim=0))
            quantile_scales.append(torch.stack(batch_scales, dim=0))

        raw_maps_tensor = torch.stack(raw_maps, dim=0)  # [B,K,N,J]
        calibrated_maps_tensor = torch.stack(calibrated_maps, dim=0)
        quantile_scales_tensor = torch.stack(quantile_scales, dim=0)  # [B,K,J]
        return {
            "candidate_masks": instance_masks,
            "instance_masks": instance_masks,
            "background_mask": instance_background_mask,
            "instance_background_mask": instance_background_mask,
            "background_map": background_map,
            "raw_candidate_maps": raw_maps_tensor,
            "candidate_maps": calibrated_maps_tensor,
            "quantile_scales": quantile_scales_tensor,
            **region_data,
        }

    def forward( # HistoryMoE weight 적용
        self,
        base_affordance: torch.Tensor,
        instance_ids: torch.Tensor,
        object_logits: torch.Tensor,
        scene_xyz: torch.Tensor = None,
        hard: bool = False,
        straight_through: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Route the calibrated candidate maps.

        ``hard=False`` uses softmax probabilities and is differentiable with
        respect to object_logits. ``hard=True`` uses one-hot argmax and is for
        inference/evaluation. ``straight_through=True`` uses the exact hard
        one-hot map in the forward pass while backpropagating through the soft
        probabilities. It is intended to remove the train/inference map
        mismatch during joint CMDM training.
        """
        if hard and straight_through:
            raise ValueError("hard and straight_through are mutually exclusive")
        self._validate(
            base_affordance, instance_ids, object_logits, scene_xyz
        )
        maps = self.build_candidate_maps(
            base_affordance, instance_ids, scene_xyz
        )

        probabilities = F.softmax(object_logits, dim=-1)
        selected_index = probabilities.argmax(dim=-1)
        if hard:
            routing_weights = F.one_hot(
                selected_index, num_classes=self.num_candidates
            ).to(base_affordance.dtype)
        elif straight_through:
            hard_weights = F.one_hot(
                selected_index, num_classes=self.num_candidates
            ).to(base_affordance.dtype)
            routing_weights = (
                hard_weights - probabilities.detach() + probabilities
            )
        else:
            routing_weights = probabilities

        routed_candidates = (
            maps["candidate_maps"]
            * routing_weights[:, :, None, None]
        ).sum(dim=1)
        routed_raw_candidates = (
            maps["raw_candidate_maps"]
            * routing_weights[:, :, None, None]
        ).sum(dim=1)
        routed_affordance = maps["background_map"] + routed_candidates
        routed_raw_affordance = (
            maps["background_map"] + routed_raw_candidates
        )
        full_candidate_maps = (
            maps["background_map"][:, None, :, :] + maps["candidate_maps"]
        )
        point_routing_weight = (
            maps["region_masks"].to(base_affordance.dtype)
            * routing_weights[:, :, None]
        ).sum(dim=1)

        return {
            **maps,
            "probabilities": probabilities,
            "routing_weights": routing_weights,
            "selected_index": selected_index,
            "point_routing_weight": point_routing_weight,
            "full_candidate_maps": full_candidate_maps,
            "routed_affordance": routed_affordance,
            "routed_raw_affordance": routed_raw_affordance,# 최종 affordance map*
        }

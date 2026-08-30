"""Temporal adapter for point-aligned Inter-object Interaction Weights.

The adapter converts a compact phase-level IIW plan into one conditioning
token per CMDM motion frame.  It is intentionally independent of CMDM so the
pretrained model can keep its original static contact input unchanged.

Input contract
--------------

``scene_xyz``
    ``[B, N, 3]`` scene points in the same order as the ADM point cloud.

``iiw_plan``
    ``[B, Q, N, 6]`` point-aligned IIW values in native body-part order:
    base, spine, right hand, left hand, right foot, left foot.

``frame_to_phase``
    ``[B, T]`` indices mapping each valid motion frame to one of the ``Q``
    phase tokens.  Padded frames may contain ``-1`` or a valid clamped index.

``x_mask``
    ``[B, T]`` boolean padding mask, where ``True`` denotes padding.

The output is ``[B, T, latent_dim]`` and is exactly zero on padded frames.
A null-plan anchor guarantees that all-zero IIW remains an exact no-op even
after training. The default zero-initialized output projection additionally
makes every plan a no-op when first attached to a pretrained CMDM.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


NATIVE_BODY_PART_NAMES: Tuple[str, ...] = (
    "base",
    "spine",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
)


class IIWAdapter(nn.Module):
    """Map phase-level point IIW plans to frame-level CMDM residuals.

    The spatial reduction is linear in the input tensor size and never
    expands phase targets to all motion frames.  For every phase/body pair it
    computes eight interpretable descriptors:

    ``[mean mass, max weight, centroid xyz, standard deviation xyz]``.

    Args:
        latent_dim: Size of the CMDM motion-token embedding.
        num_phases: Required number of temporal IIW phases.
        num_body_parts: Required native IIW body channel count.
        body_hidden_dim: Hidden width used per body-part descriptor.
        hidden_dim: Hidden width used after combining all body parts.
        embedding_dim: Width of learned phase and body identity embeddings.
        eps: Positive denominator/variance stability constant.
        zero_init: Zero-initialize the final projection.  This should remain
            enabled when adding the adapter to a pretrained CMDM.
        validate_inputs: Run strict tensor-contract validation in ``forward``.
    """

    descriptor_dim = 8

    def __init__(
        self,
        latent_dim: int = 512,
        num_phases: int = 8,
        num_body_parts: int = 6,
        body_hidden_dim: int = 64,
        hidden_dim: int = 256,
        embedding_dim: int = 16,
        eps: float = 1e-6,
        zero_init: bool = True,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("num_phases", num_phases),
            ("num_body_parts", num_body_parts),
            ("body_hidden_dim", body_hidden_dim),
            ("hidden_dim", hidden_dim),
            ("embedding_dim", embedding_dim),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if num_body_parts != len(NATIVE_BODY_PART_NAMES):
            raise ValueError(
                "num_body_parts must be {} for the native IIW contract, got {}"
                .format(len(NATIVE_BODY_PART_NAMES), num_body_parts)
            )
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.latent_dim = latent_dim
        self.num_phases = num_phases
        self.num_body_parts = num_body_parts
        self.body_hidden_dim = body_hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.eps = float(eps)
        self.validate_inputs = bool(validate_inputs)

        self.phase_embedding = nn.Embedding(num_phases, embedding_dim)
        self.body_embedding = nn.Embedding(num_body_parts, embedding_dim)

        body_input_dim = self.descriptor_dim + 2 * embedding_dim
        self.body_mlp = nn.Sequential(
            nn.Linear(body_input_dim, body_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(body_hidden_dim),
            nn.Linear(body_hidden_dim, body_hidden_dim),
            nn.GELU(),
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(num_body_parts * body_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

        if zero_init:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _is_integer_tensor(value: torch.Tensor) -> bool:
        integer_dtypes = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        return value.dtype in integer_dtypes

    def _validate_inputs(
        self,
        scene_xyz: torch.Tensor,
        iiw_plan: torch.Tensor,
        frame_to_phase: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> None:
        values = (scene_xyz, iiw_plan, frame_to_phase, x_mask)
        names = ("scene_xyz", "iiw_plan", "frame_to_phase", "x_mask")
        for name, value in zip(names, values):
            if not isinstance(value, torch.Tensor):
                raise TypeError("{} must be a torch.Tensor".format(name))

        if scene_xyz.ndim != 3 or scene_xyz.shape[-1] != 3:
            raise ValueError(
                "scene_xyz must have shape [B,N,3], got {}"
                .format(tuple(scene_xyz.shape))
            )
        if iiw_plan.ndim != 4:
            raise ValueError(
                "iiw_plan must have shape [B,Q,N,6], got {}"
                .format(tuple(iiw_plan.shape))
            )
        if frame_to_phase.ndim != 2:
            raise ValueError(
                "frame_to_phase must have shape [B,T], got {}"
                .format(tuple(frame_to_phase.shape))
            )
        if x_mask.ndim != 2:
            raise ValueError(
                "x_mask must have shape [B,T], got {}"
                .format(tuple(x_mask.shape))
            )

        batch_size, num_points, _ = scene_xyz.shape
        plan_batch, num_phases, plan_points, num_parts = iiw_plan.shape
        if plan_batch != batch_size:
            raise ValueError("scene_xyz and iiw_plan batch sizes differ")
        if plan_points != num_points:
            raise ValueError("scene_xyz and iiw_plan point counts differ")
        if num_points <= 0:
            raise ValueError("scene point cloud must not be empty")
        if num_phases != self.num_phases:
            raise ValueError(
                "iiw_plan phase count must be {}, got {}"
                .format(self.num_phases, num_phases)
            )
        if num_parts != self.num_body_parts:
            raise ValueError(
                "iiw_plan body count must be {}, got {}"
                .format(self.num_body_parts, num_parts)
            )
        if frame_to_phase.shape != x_mask.shape:
            raise ValueError("frame_to_phase and x_mask shapes differ")
        if frame_to_phase.shape[0] != batch_size:
            raise ValueError("frame tensors and scene batch sizes differ")
        if frame_to_phase.shape[1] <= 0:
            raise ValueError("motion horizon must be positive")

        if not torch.is_floating_point(scene_xyz):
            raise TypeError("scene_xyz must be floating point")
        if not torch.is_floating_point(iiw_plan):
            raise TypeError("iiw_plan must be floating point")
        if not self._is_integer_tensor(frame_to_phase):
            raise TypeError("frame_to_phase must use an integer dtype")
        if x_mask.dtype != torch.bool:
            raise TypeError("x_mask must have dtype torch.bool")

        device = scene_xyz.device
        if any(value.device != device for value in values[1:]):
            raise ValueError("all IIWAdapter inputs must share one device")
        parameter_device = self.output_projection.weight.device
        if device != parameter_device:
            raise ValueError(
                "input device {} differs from adapter device {}"
                .format(device, parameter_device)
            )

        if not bool(torch.isfinite(scene_xyz).all().item()):
            raise ValueError("scene_xyz contains NaN or Inf")
        if not bool(torch.isfinite(iiw_plan).all().item()):
            raise ValueError("iiw_plan contains NaN or Inf")
        minimum = float(iiw_plan.min().item())
        maximum = float(iiw_plan.max().item())
        tolerance = 1e-5
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            raise ValueError(
                "iiw_plan must lie in [0,1], got [{:.6f},{:.6f}]"
                .format(minimum, maximum)
            )

        # CMDM production masks are prefix-contiguous: valid frames first,
        # padding last.  Detecting holes here prevents a silent phase shift.
        if x_mask.shape[1] > 1:
            returns_to_valid = x_mask[:, :-1] & (~x_mask[:, 1:])
            if bool(returns_to_valid.any().item()):
                raise ValueError("x_mask must be prefix-contiguous")
        if bool(x_mask.all(dim=1).any().item()):
            raise ValueError("each sample must contain at least one valid frame")

        phase_index = frame_to_phase.to(dtype=torch.long)
        if bool((phase_index < -1).any().item()):
            raise ValueError("frame_to_phase contains a value below -1")
        if bool((phase_index >= self.num_phases).any().item()):
            raise ValueError("frame_to_phase contains an out-of-range phase")
        invalid_valid_frame = (~x_mask) & (phase_index < 0)
        if bool(invalid_valid_frame.any().item()):
            raise ValueError("valid frames must map to a non-negative phase")
        if phase_index.shape[1] > 1:
            adjacent_valid = (~x_mask[:, :-1]) & (~x_mask[:, 1:])
            phase_decrease = phase_index[:, 1:] < phase_index[:, :-1]
            if bool((adjacent_valid & phase_decrease).any().item()):
                raise ValueError(
                    "valid frame_to_phase indices must be non-decreasing"
                )

    def summarize(
        self,
        scene_xyz: torch.Tensor,
        iiw_plan: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B,Q,6,8]`` interpretable spatial descriptors.

        This public method intentionally performs only the spatial reduction;
        ``forward`` owns the complete input contract and temporal gathering.
        Reductions use float32 for stable sums with float16 IIW files.
        """
        if scene_xyz.ndim != 3 or scene_xyz.shape[-1] != 3:
            raise ValueError("scene_xyz must have shape [B,N,3]")
        if iiw_plan.ndim != 4:
            raise ValueError("iiw_plan must have shape [B,Q,N,6]")
        if scene_xyz.shape[0] != iiw_plan.shape[0]:
            raise ValueError("scene_xyz and iiw_plan batch sizes differ")
        if scene_xyz.shape[1] != iiw_plan.shape[2]:
            raise ValueError("scene_xyz and iiw_plan point counts differ")

        xyz = scene_xyz.float()
        weights = iiw_plan.float()
        num_points = weights.shape[2]

        weight_sum = weights.sum(dim=2)  # [B,Q,C]
        mass = weight_sum / float(num_points)
        peak = weights.max(dim=2)[0]
        denominator = weight_sum.clamp_min(self.eps).unsqueeze(-1)

        centroid = torch.einsum("bqnc,bnd->bqcd", weights, xyz)
        centroid = centroid / denominator
        second_moment = torch.einsum(
            "bqnc,bnd->bqcd", weights, xyz * xyz
        )
        second_moment = second_moment / denominator
        variance = (second_moment - centroid * centroid).clamp_min(0.0)
        # Subtract the numerical floor so a one-point distribution has exact
        # zero spread while the square-root derivative stays finite at zero.
        standard_deviation = (
            torch.sqrt(variance + self.eps) - self.eps ** 0.5
        ).clamp_min(0.0)

        # An inactive body/phase has no geometric support.  Force its moments
        # to exact zero instead of exposing the numerical sqrt(eps) value.
        active = (weight_sum > self.eps).unsqueeze(-1)
        centroid = centroid * active.to(dtype=centroid.dtype)
        standard_deviation = standard_deviation * active.to(
            dtype=standard_deviation.dtype
        )

        return torch.cat(
            (
                mass.unsqueeze(-1),
                peak.unsqueeze(-1),
                centroid,
                standard_deviation,
            ),
            dim=-1,
        )

    def encode_phases(
        self,
        scene_xyz: torch.Tensor,
        iiw_plan: torch.Tensor,
    ) -> torch.Tensor:
        """Encode IIW spatial summaries as ``[B,Q,latent_dim]`` tokens.

        The returned token is anchored to the all-zero (no-interaction) plan:

        ``adapter_token(plan) = encoder(plan) - encoder(zero_plan)``.

        This makes a zero IIW plan an exact no-op even after training changes
        MLP biases and identity embeddings.  Unlike a hard activity gate, the
        subtraction keeps a continuous gradient from the token back to every
        differentiable spatial descriptor of a nonzero plan.
        """
        descriptors = self.summarize(scene_xyz, iiw_plan)
        encoded = self._encode_descriptors(descriptors)
        null_encoded = self._encode_descriptors(torch.zeros_like(descriptors))
        return encoded - null_encoded

    def _encode_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Encode a validated ``[B,Q,6,8]`` descriptor tensor."""
        batch_size, num_phases, num_parts, _ = descriptors.shape
        device = descriptors.device
        # The reductions above intentionally run in float32.  Convert only
        # after reduction so a half/bfloat16 adapter still receives inputs in
        # its own parameter dtype and returns a CMDM-compatible residual.
        descriptors = descriptors.to(dtype=self.phase_embedding.weight.dtype)

        phase_ids = torch.arange(num_phases, device=device)
        phase_ids = phase_ids.view(1, num_phases, 1)
        phase_ids = phase_ids.expand(batch_size, num_phases, num_parts)
        body_ids = torch.arange(num_parts, device=device)
        body_ids = body_ids.view(1, 1, num_parts)
        body_ids = body_ids.expand(batch_size, num_phases, num_parts)

        phase_identity = self.phase_embedding(phase_ids)
        body_identity = self.body_embedding(body_ids)
        body_input = torch.cat(
            (descriptors, phase_identity, body_identity), dim=-1
        )
        body_features = self.body_mlp(body_input)
        phase_input = body_features.reshape(
            batch_size, num_phases, num_parts * self.body_hidden_dim
        )
        phase_features = self.phase_mlp(phase_input)
        return self.output_projection(phase_features)

    def forward(
        self,
        scene_xyz: torch.Tensor,
        iiw_plan: torch.Tensor,
        frame_to_phase: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce frame-aligned residual conditioning for CMDM."""
        if self.validate_inputs:
            self._validate_inputs(
                scene_xyz, iiw_plan, frame_to_phase, x_mask
            )

        phase_tokens = self.encode_phases(scene_xyz, iiw_plan)
        phase_index = frame_to_phase.to(dtype=torch.long)
        safe_phase_index = phase_index.clamp(0, self.num_phases - 1)
        gather_index = safe_phase_index.unsqueeze(-1).expand(
            -1, -1, self.latent_dim
        )
        frame_tokens = torch.gather(phase_tokens, 1, gather_index)

        # The final masked fill guarantees padded output is exactly zero
        # whether its phase index is -1 or clamped.
        valid = (~x_mask) & (phase_index >= 0)
        return frame_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)


__all__ = ["IIWAdapter", "NATIVE_BODY_PART_NAMES"]

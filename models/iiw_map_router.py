"""Convert native temporal IIW predictions into a literal ADM ``map*``.

This module deliberately does one small, auditable operation.  The cached
base affordance map is treated as immutable, while the final (terminal) IIW
phase supplies one scalar point-cloud weight per scene point::

    terminal_iiw = native_iiw[:, -1, :, :]       # [B, N, 6]
    pc_weight = terminal_iiw.max(dim=-1).values  # [B, N]
    mapstar = base.detach() * pc_weight[..., None]

The six IIW channels are in native body order: base, spine, right hand, left
hand, right foot, left foot.  The max reduction intentionally says that a
point is active when *any* native body region predicts interaction there.

For diagnostics and visualization, the same body reduction is retained for
all phases and multiplied by the same detached base ADM.  This module does
not perform phase-to-frame adaptation and does not modify CMDM.
"""

from __future__ import annotations

from typing import Dict, Tuple

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


class IIWMapRouter(nn.Module):
    """Build a point-weighted affordance map from native temporal IIW.

    Inputs:
        base_affordance: Cached base ADM, shape ``[B, N, 6]`` in ``[0, 1]``.
        native_iiw: Native IIW prediction, shape ``[B, Q, N, 6]`` in
            ``[0, 1]``.  ``Q`` may be any positive number of ordered phases.

    Returned dictionary:
        ``pc_weight``: Terminal-phase scalar weights, ``[B, N]``.
        ``mapstar``: Production ``base.detach() * pc_weight``, ``[B, N, 6]``.
        ``phase_pc_weight``: Scalar weights for every phase, ``[B, Q, N]``.
        ``phase_mapstar``: Visualization maps for every phase,
            ``[B, Q, N, 6]``.

    The base input is detached before every multiplication.  Consequently a
    downstream loss can train the IIW producer but can never update or
    accidentally fine-tune the cached ADM through this routing operation.
    """

    num_body_parts = len(NATIVE_BODY_PART_NAMES)

    def __init__(self, validate_inputs: bool = True) -> None:
        super().__init__()
        self.validate_inputs = bool(validate_inputs)

    @staticmethod
    def _require_tensor(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("{} must be a torch.Tensor".format(name))

    def _validate(
        self,
        base_affordance: torch.Tensor,
        native_iiw: torch.Tensor,
    ) -> None:
        self._require_tensor("base_affordance", base_affordance)
        self._require_tensor("native_iiw", native_iiw)

        if base_affordance.ndim != 3:
            raise ValueError(
                "base_affordance must have shape [B,N,6], got {}".format(
                    tuple(base_affordance.shape)
                )
            )
        if native_iiw.ndim != 4:
            raise ValueError(
                "native_iiw must have shape [B,Q,N,6], got {}".format(
                    tuple(native_iiw.shape)
                )
            )
        if base_affordance.shape[-1] != self.num_body_parts:
            raise ValueError(
                "base_affordance must have {} native body channels, got {}"
                .format(self.num_body_parts, base_affordance.shape[-1])
            )
        if native_iiw.shape[-1] != self.num_body_parts:
            raise ValueError(
                "native_iiw must have {} native body channels, got {}".format(
                    self.num_body_parts, native_iiw.shape[-1]
                )
            )

        batch_size, num_points, _ = base_affordance.shape
        plan_batch, num_phases, plan_points, _ = native_iiw.shape
        if batch_size <= 0 or num_points <= 0:
            raise ValueError("base_affordance batch and point counts must be positive")
        if num_phases <= 0:
            raise ValueError("native_iiw must contain at least one phase")
        if plan_batch != batch_size or plan_points != num_points:
            raise ValueError(
                "base_affordance/native_iiw batch or point mismatch: {} vs {}"
                .format(tuple(base_affordance.shape), tuple(native_iiw.shape))
            )

        if not torch.is_floating_point(base_affordance):
            raise TypeError("base_affordance must be floating point")
        if not torch.is_floating_point(native_iiw):
            raise TypeError("native_iiw must be floating point")
        if base_affordance.device != native_iiw.device:
            raise ValueError(
                "base_affordance and native_iiw must share one device; got {} and {}"
                .format(base_affordance.device, native_iiw.device)
            )
        if base_affordance.dtype != native_iiw.dtype:
            raise TypeError(
                "base_affordance and native_iiw must share one dtype; got {} and {}"
                .format(base_affordance.dtype, native_iiw.dtype)
            )

        for name, value in (
            ("base_affordance", base_affordance),
            ("native_iiw", native_iiw),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("{} contains NaN or Inf".format(name))
            minimum = float(value.min().item())
            maximum = float(value.max().item())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(
                    "{} must lie in [0,1], got [{:.8f},{:.8f}]".format(
                        name, minimum, maximum
                    )
                )

    def forward(
        self,
        base_affordance: torch.Tensor,
        native_iiw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return production and per-phase ``map*`` tensors."""
        if self.validate_inputs:
            self._validate(base_affordance, native_iiw)

        # [B,Q,N,6] -> [B,Q,N].  Keep this differentiable so motion/map losses
        # can reach the IIW planner through the exact multiplication below.
        phase_pc_weight = native_iiw.max(dim=-1)[0]

        # The production rule is intentionally terminal-phase only.  Do not
        # average or max over time here: phase order must affect the output.
        pc_weight = phase_pc_weight[:, -1, :]

        frozen_base = base_affordance.detach()
        mapstar = frozen_base * pc_weight.unsqueeze(-1)
        phase_mapstar = (
            frozen_base.unsqueeze(1) * phase_pc_weight.unsqueeze(-1)
        )

        return {
            "pc_weight": pc_weight,
            "mapstar": mapstar,
            "phase_pc_weight": phase_pc_weight,
            "phase_mapstar": phase_mapstar,
        }


__all__ = ["IIWMapRouter", "NATIVE_BODY_PART_NAMES"]

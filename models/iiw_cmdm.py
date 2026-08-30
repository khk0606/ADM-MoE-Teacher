"""Non-invasive IIW conditioning wrapper for an existing CMDM instance.

The production ``models/cmdm.py`` may already contain local research edits.
This wrapper therefore leaves that file and its checkpoint keys untouched. It
injects a frame-aligned IIW residual by installing a temporary forward hook on
the existing CMDM ``motion_adapter`` for exactly one model call.

Without ``c_iiw_residual`` the wrapper calls the wrapped model directly, so
the legacy execution path is identical.  Hooked calls are serialized because
PyTorch module forward-hook registries are mutable and are not safe for two
simultaneous residuals on one shared model instance.  The hook is removed in a
``finally`` block, including when validation or CMDM forward raises.

This eager-mode integration is intended for the single-GPU Oracle experiment.
Temporary Python hooks can cause graph breaks under TorchScript/``torch.compile``;
production deployment should eventually promote this optional branch into a
versioned CMDM implementation after the research contract is validated.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn as nn


def inject_iiw_residual(
    motion_embedding: torch.Tensor,
    residual: torch.Tensor,
    x_mask: torch.Tensor,
) -> torch.Tensor:
    """Validate and add ``[B,T,latent_dim]`` IIW conditioning tokens."""
    if not torch.is_tensor(motion_embedding):
        raise TypeError("motion_adapter output must be a torch.Tensor")
    if motion_embedding.ndim != 3:
        raise ValueError(
            "motion_adapter output must have shape [B,T,latent_dim], got {}"
            .format(tuple(motion_embedding.shape))
        )
    if not torch.is_tensor(residual):
        raise TypeError("c_iiw_residual must be a torch.Tensor")
    if residual.shape != motion_embedding.shape:
        raise ValueError(
            "c_iiw_residual must have shape [B,T,latent_dim] equal to the "
            "motion embedding; got {} versus {}".format(
                tuple(residual.shape), tuple(motion_embedding.shape)
            )
        )
    if not torch.is_floating_point(residual):
        raise TypeError("c_iiw_residual must be floating point")
    if residual.device != motion_embedding.device:
        raise ValueError(
            "c_iiw_residual and motion embedding must share one device; "
            "got {} and {}".format(
                residual.device, motion_embedding.device
            )
        )
    if residual.dtype != motion_embedding.dtype:
        raise ValueError(
            "c_iiw_residual and motion embedding must have one dtype; got "
            "{} and {}".format(residual.dtype, motion_embedding.dtype)
        )
    if not bool(torch.isfinite(residual).all().item()):
        raise ValueError("c_iiw_residual contains NaN or Inf")

    if not torch.is_tensor(x_mask):
        raise TypeError("x_mask must be a torch.Tensor when IIW is enabled")
    if x_mask.dtype != torch.bool:
        raise TypeError("x_mask must have bool dtype when IIW is enabled")
    if x_mask.shape != motion_embedding.shape[:2]:
        raise ValueError(
            "x_mask must have shape [B,T] matching c_iiw_residual; got {} "
            "versus {}".format(
                tuple(x_mask.shape), tuple(motion_embedding.shape[:2])
            )
        )
    if x_mask.device != residual.device:
        raise ValueError(
            "x_mask and c_iiw_residual must share one device; got {} and {}"
            .format(x_mask.device, residual.device)
        )
    padded = residual.masked_select(x_mask.unsqueeze(-1).expand_as(residual))
    if padded.numel() > 0 and int(torch.count_nonzero(padded).item()) != 0:
        raise ValueError(
            "c_iiw_residual must be exactly zero on every padded frame"
        )
    return motion_embedding + residual


class IIWConditionedCMDM(nn.Module):
    """Wrap a CMDM without modifying its class, state dict, or source file.

    Args:
        cmdm: Existing initialized and checkpoint-loaded CMDM module.  It must
            expose its motion projection as ``cmdm.motion_adapter``.

    Call contract:
        ``wrapper(x, timesteps, **legacy_kwargs)`` is a direct legacy call.
        Supplying ``c_iiw_residual`` additionally requires boolean ``x_mask``.

    Thread safety:
        Calls on one wrapper are serialized for the full hook lifetime. This
        prevents concurrent calls from stacking residual hooks on the same
        ``motion_adapter``. Separate wrapper/model instances remain parallel.
    """

    def __init__(self, cmdm: nn.Module) -> None:
        super().__init__()
        if not isinstance(cmdm, nn.Module):
            raise TypeError("cmdm must be a torch.nn.Module")
        motion_adapter = getattr(cmdm, "motion_adapter", None)
        if not isinstance(motion_adapter, nn.Module):
            raise TypeError("cmdm.motion_adapter must be a torch.nn.Module")
        self.cmdm = cmdm
        # RLock is deliberately not a Parameter/buffer and therefore cannot
        # alter the wrapped CMDM state dict.  It covers the entire hooked call.
        self._iiw_hook_lock = threading.RLock()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if "c_iiw_residual" not in kwargs:
            # No copy, hook, cast, validation, or extra tensor operation.
            return self.cmdm(*args, **kwargs)

        residual = kwargs["c_iiw_residual"]
        if "x_mask" not in kwargs:
            raise KeyError("x_mask is required when c_iiw_residual is provided")
        x_mask = kwargs["x_mask"]

        # The unmodified CMDM does not consume this new keyword.  Copying the
        # dict preserves the caller's object while keeping the base call clean.
        base_kwargs = dict(kwargs)
        del base_kwargs["c_iiw_residual"]

        with self._iiw_hook_lock:
            handle = None

            def inject_after_motion_adapter(
                module: nn.Module,
                module_inputs: Any,
                module_output: torch.Tensor,
            ) -> torch.Tensor:
                del module, module_inputs
                return inject_iiw_residual(
                    module_output, residual, x_mask
                )

            try:
                handle = self.cmdm.motion_adapter.register_forward_hook(
                    inject_after_motion_adapter
                )
                return self.cmdm(*args, **base_kwargs)
            finally:
                if handle is not None:
                    handle.remove()


_inject_iiw_residual = inject_iiw_residual


__all__ = [
    "IIWConditionedCMDM",
    "inject_iiw_residual",
    "_inject_iiw_residual",
]

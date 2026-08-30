#!/usr/bin/env python3
"""Minimal LoRA utilities for leakage-safe CDM adaptation.

The wrapper is deliberately model-agnostic: every selected ``nn.Linear`` keeps
its original weight frozen and receives a zero-output low-rank residual.  The
export helper folds the residual back into legacy ``*.weight`` keys, so the
existing AMDM loader, rollout audit, and evaluator do not need LoRA-aware model
construction at inference time.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn.functional as F


DEFAULT_EXCLUDED_TOKENS = (
    "scene_model",
    "text_model",
    "clip_model",
    "bert_model",
)


class LoRALinear(torch.nn.Module):

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.enabled = True

        for parameter in self.base.parameters():#원본 CDM freeze
            parameter.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(
            torch.empty(
                self.rank,
                self.base.in_features,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(
                self.base.out_features,
                self.rank,
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.base(value)
        if not self.enabled:
            return output
        residual = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return output + residual * self.scale # 기존 CDM + LoRA residual 보정값

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scale


def _excluded(name: str, excluded_tokens: Iterable[str]) -> bool:
    return any(token in name for token in excluded_tokens)


def install_lora(
    model: torch.nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    excluded_tokens: Iterable[str] = DEFAULT_EXCLUDED_TOKENS,
) -> List[str]:
    """Freeze ``model`` and replace eligible Linear children with LoRA layers."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements: List[Tuple[str, torch.nn.Module, str, torch.nn.Linear]] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            # Use the exact class.  PyTorch MultiheadAttention stores an
            # ``_NonDynamicallyQuantizableLinear`` child and reads its weight
            # directly instead of calling the child module; replacing that
            # subclass would silently violate its forward contract.
            if type(child) is torch.nn.Linear and not _excluded(
                full_name, excluded_tokens
            ):
                replacements.append((full_name, parent, child_name, child))
    if not replacements:
        raise RuntimeError("No eligible CDM Linear layer was found for LoRA")
    names = []
    for full_name, parent, child_name, child in replacements:
        setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
        names.append(full_name)
    return sorted(names)


def iter_lora_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_named_parameters(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    values: Dict[str, torch.nn.Parameter] = {}
    for name, module in iter_lora_modules(model):
        values[f"{name}.lora_A"] = module.lora_A
        values[f"{name}.lora_B"] = module.lora_B
    if not values:
        raise RuntimeError("LoRA parameters are missing")
    return values


def set_lora_enabled(model: torch.nn.Module, enabled: bool) -> None:
    found = False
    for _, module in iter_lora_modules(model):
        module.enabled = bool(enabled)
        found = True
    if not found:
        raise RuntimeError("LoRA modules are missing")


def set_frozen_base_eval_lora_train(model: torch.nn.Module) -> None:
    """Keep every frozen base module in eval mode while training LoRA dropout."""

    model.eval()
    found = False
    for _, module in iter_lora_modules(model):
        module.dropout.train(True)
        found = True
    if not found:
        raise RuntimeError("LoRA modules are missing")


def lora_parameter_energy(model: torch.nn.Module) -> torch.Tensor:
    """Mean squared merged-weight residual; exactly zero at initialization."""

    values = [
        ((module.lora_B @ module.lora_A) * module.scale).square().mean()
        for _, module in iter_lora_modules(model)
    ]
    if not values:
        raise RuntimeError("LoRA modules are missing")
    return torch.stack(values).mean()


def merged_legacy_state_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Return a state dict with all LoRA modules folded into legacy keys."""

    wrappers = dict(iter_lora_modules(model))
    wrapper_prefixes = tuple(f"{name}." for name in wrappers)
    result: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in model.state_dict().items():
        if any(key.startswith(prefix) for prefix in wrapper_prefixes):
            continue
        result[key] = value.detach().clone()
    for name, module in wrappers.items():
        result[f"{name}.weight"] = module.merged_weight().detach().clone()
        if module.base.bias is not None:
            result[f"{name}.bias"] = module.base.bias.detach().clone()
    return result


def save_merged_legacy_state(
    model: torch.nn.Module,
    path,
    excluded_tokens: Iterable[str] = DEFAULT_EXCLUDED_TOKENS,
) -> None:
    """Save a loader-compatible partial CDM state with merged LoRA weights."""

    state = OrderedDict(
        (key, value.cpu())
        for key, value in merged_legacy_state_dict(model).items()
        if not _excluded(key, excluded_tokens)
    )
    if not state:
        raise RuntimeError("refusing to save an empty merged CDM checkpoint")
    torch.save(state, path)


def lora_metadata(model: torch.nn.Module) -> Mapping[str, object]:
    modules = list(iter_lora_modules(model))
    if not modules:
        raise RuntimeError("LoRA metadata requested before installation")
    first = modules[0][1]
    return {
        "rank": first.rank,
        "alpha": first.alpha,
        "scale": first.scale,
        "module_count": len(modules),
        "module_names": [name for name, _ in modules],
        "base_parameters_frozen": True,
        "zero_initialized_output_projection": True,
        "export_format": "merged_legacy_partial_state_dict",
    }

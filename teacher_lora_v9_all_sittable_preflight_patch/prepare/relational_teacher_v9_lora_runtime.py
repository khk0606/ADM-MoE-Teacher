#!/usr/bin/env python3
"""Small CUDA/runtime helpers shared by the Teacher-v9 preflight."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FORWARD_INPUT_KEYS = ("c_pc_feat", "c_pc_xyz", "c_text")


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compose_cdm_config(diffusion_steps: int, device: str):
    from hydra import compose, initialize_config_dir
    from utils.misc import compute_repr_dimesion

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                "task=contact_gen",
                "model=cdm",
                "model.arch=Perceiver",
                "task.dataset.sigma=0.8",
                f"diffusion.steps={diffusion_steps}",
            ],
        )
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    cfg.gpu = int(device.split(":", 1)[1]) if device.startswith("cuda:") else None
    return cfg


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        mean = np.asarray(source["mean"], dtype=np.float32)
        std = np.asarray(source["std"], dtype=np.float32)
    if mean.shape != (1, 6) or std.shape != (1, 6):
        raise ValueError("contact statistics must be exact [1,6] arrays")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError("contact statistics are invalid")
    return mean, std


def deterministic_noise(shape: torch.Size, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator).to(device)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def predict_xstart(
    model,
    diffusion,
    x_start: torch.Tensor,
    timestep: torch.Tensor,
    kwargs: Mapping[str, object],
    noise: torch.Tensor,
) -> torch.Tensor:
    if tuple(sorted(kwargs)) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher forward inputs must be exactly text + scene")
    if getattr(diffusion.model_mean_type, "name", "") != "START_X":
        raise RuntimeError("Teacher-v9 requires START_X diffusion prediction")
    x_t = diffusion.q_sample(x_start, timestep, noise=noise)
    wrapped = diffusion._wrap_model(model) if hasattr(diffusion, "_wrap_model") else model
    prediction = wrapped(
        x_t,
        diffusion._scale_timesteps(timestep),
        **dict(kwargs),
    )
    if prediction.shape != x_start.shape:
        raise ValueError("CDM prediction shape changed")
    return prediction


def gradient_summary(
    loss: torch.Tensor,
    named_parameters: Mapping[str, torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> Dict[str, float]:
    gradients = torch.autograd.grad(
        loss,
        list(named_parameters.values()),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total_sq = 0.0
    a_sq = 0.0
    b_sq = 0.0
    finite = True
    used = 0
    for (name, _), gradient in zip(named_parameters.items(), gradients):
        if gradient is None:
            continue
        used += 1
        finite = finite and bool(torch.isfinite(gradient).all().item())
        square = float(gradient.detach().float().square().sum().item())
        total_sq += square
        if name.endswith("lora_A"):
            a_sq += square
        elif name.endswith("lora_B"):
            b_sq += square
    return {
        "loss": float(loss.detach().item()),
        "gradient_l2": math.sqrt(total_sq),
        "lora_A_gradient_l2": math.sqrt(a_sq),
        "lora_B_gradient_l2": math.sqrt(b_sq),
        "parameters_with_gradient": used,
        "all_finite": finite,
    }


def require_positive_gradient(name: str, summary: Mapping[str, object]) -> None:
    value = float(summary["gradient_l2"])
    loss = float(summary["loss"])
    if (
        not math.isfinite(value)
        or value <= 0.0
        or not math.isfinite(loss)
        or loss < 0.0
        or summary.get("all_finite") is not True
        or int(summary.get("parameters_with_gradient", 0)) <= 0
    ):
        raise RuntimeError(f"{name} has no finite nonzero LoRA gradient")


def perturb_lora_output(
    named_parameters: Mapping[str, torch.nn.Parameter], amount: float
) -> Dict[str, torch.Tensor]:
    saved = {name: value.detach().clone() for name, value in named_parameters.items()}
    with torch.no_grad():
        for name, value in named_parameters.items():
            if name.endswith("lora_B"):
                value.add_(float(amount))
    return saved


def restore_lora(
    named_parameters: Mapping[str, torch.nn.Parameter],
    saved: Mapping[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, value in named_parameters.items():
            value.copy_(saved[name])


__all__ = [
    "FORWARD_INPUT_KEYS",
    "REPO_ROOT",
    "compose_cdm_config",
    "configure_reproducibility",
    "deterministic_noise",
    "gradient_summary",
    "load_stats",
    "perturb_lora_output",
    "predict_xstart",
    "require_positive_gradient",
    "restore_lora",
    "tensor_sha256",
]

#!/usr/bin/env python3
"""Shared paired reverse-diffusion metrics for Teacher-v7 High-Desk."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from relational_teacher_v6_contract import (
    PROMPTS,
    SITTABLE_CATEGORY_IDS,
    SIT_NEGATIVE_OBJECT_CATEGORY_IDS,
)


def stable_seeds(seed: int, domain: str, case_id: str) -> Tuple[int, int]:
    digest = hashlib.sha256(f"{seed}|{domain}|{case_id}".encode("utf-8")).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


def create_model(cfg, original_checkpoint, v5_checkpoint,
                 candidate_checkpoint: Optional[object], device: str):
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    if candidate_checkpoint is not None:
        load_ckpt(model, str(candidate_checkpoint))
    model.eval()
    return model, diffusion


@torch.no_grad()
def sample_contact(model, diffusion, xyz: np.ndarray, feat: np.ndarray, text: str,
                   initial_seed: int, reverse_seed: int, device: str,
                   progress: bool) -> np.ndarray:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(initial_seed)
    noise = torch.randn((1, 8192, 6), generator=generator).to(device)
    torch_device = torch.device(device)
    cuda_devices = []
    if torch_device.type == "cuda":
        cuda_devices = [torch_device.index if torch_device.index is not None
                        else torch.cuda.current_device()]
    kwargs = {
        "c_pc_xyz": torch.from_numpy(xyz[None]).to(device).contiguous(),
        "c_pc_feat": torch.from_numpy(feat[None]).to(device).contiguous(),
        "c_text": [text],
    }
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(reverse_seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed(reverse_seed)
        sample = diffusion.p_sample_loop(
            model, (1, 8192, 6), clip_denoised=False, noise=noise,
            model_kwargs=kwargs, device=device, progress=progress,
        )
    return sample[0].detach().cpu().numpy().astype(np.float32)


def denormalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(value * std.reshape(1, 6) + mean.reshape(1, 6), 0.0, 1.0).astype(
        np.float32
    )


def top_fraction_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot score an empty instance")
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def semantic_metrics(prediction: np.ndarray, instance_ids: np.ndarray,
                     category_ids: np.ndarray) -> Dict[str, object]:
    candidate_rows = []
    negative_rows = []
    for instance_id in sorted(int(value) for value in np.unique(instance_ids)
                              if value != 0):
        mask = instance_ids == instance_id
        categories = np.unique(category_ids[mask])
        if categories.size != 1:
            raise ValueError("instance has mixed semantic categories")
        category = int(categories[0])
        any_score = top_fraction_mean(prediction[mask].max(axis=1))
        pelvis_score = top_fraction_mean(prediction[mask, 0])
        row = {
            "instance_id": instance_id,
            "category_id": category,
            "any_score": any_score,
            "pelvis_score": pelvis_score,
        }
        if category in SITTABLE_CATEGORY_IDS:
            row["violation"] = max(0.0, 0.30 - any_score) + max(
                0.0, 0.15 - pelvis_score
            )
            candidate_rows.append(row)
        elif category in SIT_NEGATIVE_OBJECT_CATEGORY_IDS:
            row["violation"] = max(0.0, any_score - 0.10) + max(
                0.0, pelvis_score - 0.05
            )
            negative_rows.append(row)
    if not candidate_rows or not negative_rows:
        raise ValueError("semantic candidate/negative inventory is incomplete")
    candidate = float(np.mean([row["violation"] for row in candidate_rows]))
    negative = float(np.mean([row["violation"] for row in negative_rows]))
    return {
        "candidate_instances": candidate_rows,
        "negative_instances": negative_rows,
        "candidate_violation": candidate,
        "negative_violation": negative,
        "semantic_violation": candidate + negative,
    }


def target_mae(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        raise ValueError("target instance mask is empty")
    return float(np.abs(prediction[mask] - gt[mask]).mean())


def aggregate_metrics(
    relational_gt: np.ndarray,
    relational_base: np.ndarray,
    relational_candidate: np.ndarray,
    instance_ids: np.ndarray,
    category_ids: np.ndarray,
    target_instance_ids: np.ndarray,
    scene_ids: Sequence[str],
    target_names: Sequence[str],
    strata: Sequence[str],
    prompt_ids: Sequence[str],
    v5_gt: np.ndarray,
    v5_base: np.ndarray,
    v5_candidate: np.ndarray,
    v5_targets: Sequence[str],
) -> Dict[str, object]:
    cases = []
    for index in range(len(relational_gt)):
        mask = instance_ids[index] == int(target_instance_ids[index])
        cases.append({
            "index": index,
            "scene_id": str(scene_ids[index]),
            "target_name": str(target_names[index]),
            "training_stratum": str(strata[index]),
            "prompt_id": str(prompt_ids[index]),
            "target_mae": {
                "base": target_mae(relational_base[index], relational_gt[index], mask),
                "candidate": target_mae(relational_candidate[index], relational_gt[index], mask),
            },
            "semantic": {
                "base": semantic_metrics(relational_base[index], instance_ids[index], category_ids[index]),
                "candidate": semantic_metrics(relational_candidate[index], instance_ids[index], category_ids[index]),
            },
        })

    pair_rows = []
    for scene_id in sorted(set(str(value) for value in scene_ids)):
        for stratum in sorted(set(str(strata[index]) for index in range(len(strata))
                                   if str(scene_ids[index]) == scene_id)):
            indices = [index for index in range(len(scene_ids))
                       if str(scene_ids[index]) == scene_id
                       and str(strata[index]) == stratum]
            by_prompt = {str(prompt_ids[index]): index for index in indices}
            if set(by_prompt) != set(PROMPTS):
                raise ValueError("prompt pair is incomplete")
            watch = by_prompt["sit_watch_v1"]
            write = by_prompt["sit_write_v1"]
            candidate_mask = np.isin(category_ids[watch], list(SITTABLE_CATEGORY_IDS))
            pair_rows.append({
                "scene_id": scene_id,
                "training_stratum": stratum,
                "target_name": str(target_names[watch]),
                "base_mse": float(np.square(
                    relational_base[watch, candidate_mask]
                    - relational_base[write, candidate_mask]).mean()),
                "candidate_mse": float(np.square(
                    relational_candidate[watch, candidate_mask]
                    - relational_candidate[write, candidate_mask]).mean()),
            })

    replay_rows = []
    for index, target in enumerate(v5_targets):
        base_mae = float(np.abs(v5_base[index] - v5_gt[index]).mean())
        candidate_mae = float(np.abs(v5_candidate[index] - v5_gt[index]).mean())
        replay_rows.append({
            "target": str(target),
            "base_mae": base_mae,
            "candidate_mae": candidate_mae,
            "relative_degradation": (candidate_mae - base_mae) / max(base_mae, 1e-12),
        })

    def target_summary(rows):
        base = float(np.mean([row["target_mae"]["base"] for row in rows]))
        candidate = float(np.mean([row["target_mae"]["candidate"] for row in rows]))
        return {
            "target_mae_base": base,
            "target_mae_candidate": candidate,
            "target_mae_relative_change": (candidate - base) / max(base, 1e-12),
            "case_win_rate": float(np.mean([
                row["target_mae"]["candidate"] < row["target_mae"]["base"]
                for row in rows
            ])),
        }

    overall_target = target_summary(cases)
    high_desk_cases = [row for row in cases
                       if row["training_stratum"] == "high_desk_chair_06"]
    if len(high_desk_cases) != 4:
        raise ValueError("High-Desk canary case inventory changed")
    high_desk = target_summary(high_desk_cases)
    overall_target.update({
        "semantic_violation_base": float(np.mean([
            row["semantic"]["base"]["semantic_violation"] for row in cases
        ])),
        "semantic_violation_candidate": float(np.mean([
            row["semantic"]["candidate"]["semantic_violation"] for row in cases
        ])),
        "prompt_invariance_mse_base": float(np.mean([row["base_mse"] for row in pair_rows])),
        "prompt_invariance_mse_candidate": float(np.mean([row["candidate_mse"] for row in pair_rows])),
        "v5_replay_mae_base": float(np.mean([row["base_mae"] for row in replay_rows])),
        "v5_replay_mae_candidate": float(np.mean([row["candidate_mae"] for row in replay_rows])),
    })
    overall_target["v5_replay_relative_degradation"] = (
        overall_target["v5_replay_mae_candidate"] - overall_target["v5_replay_mae_base"]
    ) / max(overall_target["v5_replay_mae_base"], 1e-12)
    return {
        "relational_cases": cases,
        "prompt_pairs": pair_rows,
        "v5_replay_cases": replay_rows,
        "high_desk": high_desk,
        "overall": overall_target,
    }

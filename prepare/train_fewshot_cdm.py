#!/usr/bin/env python3
"""Semantic-safe, rollout-aligned CDM LoRA fine-tuning (version 5).

V5 keeps the original CDM frozen, changes only object-agnostic action prompts,
adds a weak Sit->Bed candidate prior without forging motion GT, and validates
Whiteboard interaction on the native right-wrist channel.  Held-out tensors
remain unread during training and checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot_cdm_common import (  # noqa: E402
    DIAGNOSTIC_IDS,
    build_sparse_contact_weights,
    checkpoint_selection_gate,
    denormalize_contact,
    gt_file_from_entry,
    instance_scores,
    load_index_entries,
    load_scene,
    load_split,
    load_stats,
    normalize_contact,
    semantic_checkpoint_gate,
    sha256_file,
    target_balanced_batches,
)
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    lora_parameter_energy,
    save_merged_legacy_state,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from fewshot_cdm_v5_semantics import (  # noqa: E402
    PROMPT_POLICY_ID,
    prompt_for_target,
    sit_multicandidate_loss,
)
from fewshot_cdm_rollout_cache import (  # noqa: E402
    CONTRACT_SCHEMA,
    atomic_save_prediction,
    atomic_write_json,
    candidate_result_path,
    ensure_contract,
    fingerprint_rows,
    load_bound_candidate_result,
    load_prediction,
    prediction_path,
    sha256_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/history_affordance_v1")
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(
            "data/history_affordance_v1/splits/"
            "chair23_bed2_whiteboard12_multistart24_v1.json"
        ),
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path(
            "data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_"
            "contact_cont_joints_0.8_fur.npz"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path("outputs/CDM-Perceiver-ALL/ckpt/model300000.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/history_affordance_v1/experiments/"
            "fewshot_cdm_chair23_bed2_whiteboard12_v5r4"
        ),
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chair-replay-per-step", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-l2-weight", type=float, default=1e-4)
    parser.add_argument("--chair-teacher-weight", type=float, default=2.0)
    parser.add_argument("--chair-high-timestep-weight", type=float, default=1.0)
    parser.add_argument("--chair-high-teacher-weight", type=float, default=2.0)
    parser.add_argument("--high-timestep-weight", type=float, default=1.0)
    parser.add_argument("--high-timestep-start", type=float, default=0.70)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument(
        "--sit-multicandidate-weight",
        type=float,
        default=0.50,
        help="Weak semantic prior weight; dense Chair motion GT stays unchanged.",
    )
    parser.add_argument("--sit-bed-any-min", type=float, default=0.12)
    parser.add_argument("--sit-bed-any-max", type=float, default=0.40)
    parser.add_argument("--sit-bed-pelvis-min", type=float, default=0.06)
    parser.add_argument("--sit-bed-pelvis-max", type=float, default=0.25)
    parser.add_argument("--sit-chair-bed-margin", type=float, default=0.15)
    parser.add_argument("--foreground-bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--competitor-suppression-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--activation-temperature", type=float, default=0.10)
    parser.add_argument(
        "--novel-min-relative-improvement",
        type=float,
        default=0.05,
        help="Minimum train-grid improvement required for both Bed and Whiteboard.",
    )
    parser.add_argument(
        "--chair-max-relative-degradation",
        type=float,
        default=0.05,
        help="Maximum train-grid Chair degradation allowed at checkpoint selection.",
    )
    parser.add_argument(
        "--target-instance-weight",
        type=float,
        default=4.0,
        help="Additional Bed/Whiteboard loss weight on every target-instance channel.",
    )
    parser.add_argument(
        "--target-foreground-weight",
        type=float,
        default=16.0,
        help="Additional loss weight on active GT channels inside the target instance.",
    )
    parser.add_argument("--active-threshold", type=float, default=0.7)
    parser.add_argument("--novel-train-min-f1", type=float, default=0.30)
    parser.add_argument(
        "--novel-train-min-dominance-rate", type=float, default=1.0
    )
    parser.add_argument(
        "--chair-train-min-dominance-rate", type=float, default=0.90
    )
    parser.add_argument(
        "--sit-bed-candidate-min-rate",
        type=float,
        default=0.80,
        help=(
            "Train-only coverage required for the weak secondary Bed candidate "
            "under the Sit prompt. This is separate from strict Chair dominance."
        ),
    )
    parser.add_argument(
        "--diagnostic-checkpoint-count",
        type=int,
        default=3,
        help=(
            "Always retain this many top-ranked train-only evaluation checkpoints, "
            "even if a scientific gate fails. They are diagnostic only."
        ),
    )
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--candidate-shortlist", type=int, default=3)
    parser.add_argument(
        "--candidate-rollout-k",
        type=int,
        default=5,
        help=(
            "Full reverse-diffusion draws used for final train-only selection. "
            "Strict v5 requires at least five and evaluates the complete train split."
        ),
    )
    parser.add_argument("--candidate-dominance-margin", type=float, default=0.01)
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help=(
            "Reuse an existing v5 one_step_candidates.json and run/resume only "
            "the expensive train-only full-diffusion selector. No training or "
            "held-out sample tensor is read."
        ),
    )
    parser.add_argument(
        "--stop-after-one-step",
        action="store_true",
        help=(
            "Stop after the train-only fixed-grid shortlist is saved. Resume "
            "the expensive full rollout later with --rollout-only."
        ),
    )
    parser.add_argument(
        "--rollout-cache-dir",
        type=Path,
        default=None,
        help=(
            "Crash-safe rollout cache. Defaults to OUTPUT_DIR/rollout_cache. "
            "Its immutable contract rejects stale or incompatible reuse."
        ),
    )
    parser.add_argument(
        "--rollout-baseline-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional compatible rollout cache whose original-CDM "
            "draws may be imported after strict split/input/seed/checkpoint and "
            "prediction-critical source-hash validation. Candidate draws are "
            "never imported through this option."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-nonstandard-pretrained", action="store_true")
    return parser.parse_args()


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


def assert_original_checkpoint(path: Path, allow_nonstandard: bool) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    standard = (
        path.parent.name == "ckpt"
        and path.parent.parent.name == "CDM-Perceiver-ALL"
        and path.name.startswith("model")
        and path.suffix == ".pt"
    )
    if not standard and not allow_nonstandard:
        raise RuntimeError(
            "Refusing a nonstandard initialization. Use the untouched original "
            "outputs/CDM-Perceiver-ALL/ckpt/model*.pt, not a one-sample checkpoint."
        )


def save_trainable_state(model: torch.nn.Module, path: Path) -> None:
    """Export merged LoRA weights using the legacy partial-CDM key contract."""

    save_merged_legacy_state(model, path)


def load_rows(
    dataset_root: Path,
    split: Mapping[str, object],
    partition: str,
    mean: np.ndarray,
    std: np.ndarray,
    target_instance_weight: float,
    target_foreground_weight: float,
    active_threshold: float,
) -> Dict[str, Dict[str, object]]:
    sample_ids = [str(v) for v in split["cdm_fewshot"][partition]]
    entries = load_index_entries(dataset_root, sample_ids)
    split_meta = {str(v["sample_id"]): v for v in split["samples"]}
    scene_cache: Dict[str, Dict[str, np.ndarray]] = {}
    rows: Dict[str, Dict[str, object]] = {}
    for sample_id in sample_ids:
        if sample_id in DIAGNOSTIC_IDS:
            raise AssertionError("diagnostic sample entered training")
        entry = entries[sample_id]
        scene_id = str(entry["scene_id"])
        scene = scene_cache.get(scene_id)
        if scene is None:
            scene = load_scene(dataset_root, entry)
            scene_cache[scene_id] = scene
        gt_file = gt_file_from_entry(dataset_root, entry)
        gt = np.load(gt_file, allow_pickle=False)
        affordance = gt["affordance"].astype(np.float32)
        if affordance.shape != (8192, 6):
            raise ValueError(f"{gt_file}: GT shape mismatch")
        if not np.array_equal(gt["instance_ids"], scene["instance_ids"]):
            raise ValueError(f"{sample_id}: GT/scene instance order mismatch")
        if not np.array_equal(gt["source_indices"], scene["source_indices"]):
            raise ValueError(f"{sample_id}: GT/scene point order mismatch")
        target = str(split_meta[sample_id]["target"])
        target_instance_id = int(entry["target_instance_id"])
        expected_instance_id = {"chair": 1, "bed": 2, "whiteboard": 3}[target]
        if target_instance_id != expected_instance_id:
            raise ValueError(
                f"{sample_id}: target instance {target_instance_id} != "
                f"expected {expected_instance_id}"
            )
        source_text = str(entry["text"])
        text = prompt_for_target(target)
        points = scene["points"]
        loss_weights = build_sparse_contact_weights(
            affordance,
            scene["instance_ids"],
            target_instance_id,
            target,
            target_instance_weight,
            target_foreground_weight,
            active_threshold,
        )
        rows[sample_id] = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "target": target,
            "target_instance_id": target_instance_id,
            "text": text,
            "source_text": source_text,
            "prompt_policy_id": PROMPT_POLICY_ID,
            "gt": affordance,
            "x": normalize_contact(affordance, mean, std),
            "xyz": points[:, :3].astype(np.float32),
            "feat": (points[:, 3:6] / 255.0).astype(np.float32),
            "instance_ids": scene["instance_ids"].astype(np.int64),
            "source_indices": scene["source_indices"].astype(np.int64),
            "loss_weights": loss_weights,
        }
    return rows


def stack_batch(
    rows: Mapping[str, Mapping[str, object]],
    sample_ids: List[str],
    device: str,
):
    selected = [rows[sample_id] for sample_id in sample_ids]
    return {
        "x": torch.from_numpy(np.stack([row["x"] for row in selected])).to(device),
        "xyz": torch.from_numpy(np.stack([row["xyz"] for row in selected])).to(device).contiguous(),
        "feat": torch.from_numpy(np.stack([row["feat"] for row in selected])).to(device).contiguous(),
        "text": [str(row["text"]) for row in selected],
        "gt": torch.from_numpy(np.stack([row["gt"] for row in selected])).to(device),
        "weights": torch.from_numpy(
            np.stack([row["loss_weights"] for row in selected])
        ).to(device),
        "instance_ids": torch.from_numpy(
            np.stack([row["instance_ids"] for row in selected])
        ).to(device),
        "target_instance_ids": torch.tensor(
            [int(row["target_instance_id"]) for row in selected],
            dtype=torch.long,
            device=device,
        ),
        "targets": [str(row["target"]) for row in selected],
    }


def predict_xstart(
    model,
    diffusion,
    x_start: torch.Tensor,
    timestep: torch.Tensor,
    model_kwargs: Mapping[str, object],
    noise: torch.Tensor,
) -> torch.Tensor:
    """Run the exact AMDM START_X denoising path without reducing its loss."""
    if getattr(diffusion.model_mean_type, "name", "") != "START_X":
        raise RuntimeError("v5 sparse loss requires diffusion.predict_xstart=true")
    if getattr(diffusion.loss_type, "name", "") not in {"MSE", "RESCALED_MSE"}:
        raise RuntimeError("v5 sparse loss requires an MSE diffusion objective")
    if getattr(diffusion.model_var_type, "name", "") not in {
        "FIXED_SMALL",
        "FIXED_LARGE",
    }:
        raise RuntimeError("v5 sparse loss does not support learned variance")
    if x_start.shape != noise.shape:
        raise ValueError("x_start/noise shape mismatch")
    x_t = diffusion.q_sample(x_start, timestep, noise=noise)
    wrapped_model = (
        diffusion._wrap_model(model) if hasattr(diffusion, "_wrap_model") else model
    )
    prediction = wrapped_model(
        x_t,
        diffusion._scale_timesteps(timestep),
        **dict(model_kwargs),
    )
    if prediction.shape != x_start.shape:
        raise ValueError(
            f"START_X output {tuple(prediction.shape)} != {tuple(x_start.shape)}"
        )
    return prediction


def weighted_xstart_loss(
    prediction: torch.Tensor,
    x_start: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return one normalized weighted MSE value per batch element."""
    if prediction.shape != x_start.shape or weights.shape != x_start.shape:
        raise ValueError("prediction, target, and weights must share a shape")
    if torch.any(weights <= 0) or not torch.isfinite(weights).all():
        raise ValueError("weights must be finite and strictly positive")
    dims = tuple(range(1, prediction.ndim))
    squared = (prediction - x_start).square()
    return (squared * weights).sum(dim=dims) / weights.sum(dim=dims)


def masked_point_xstart_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample MSE over an explicitly selected point region.

    v5r4 uses this only for Chair-preservation auxiliaries.  Restricting the
    teacher and high-noise reconstruction terms to the Chair instance avoids
    suppressing the intentionally weak Bed candidate under ``Sit somewhere.``.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share a shape")
    if point_mask.shape != prediction.shape[:2]:
        raise ValueError("point_mask must be [B,N]")
    if point_mask.dtype != torch.bool:
        raise ValueError("point_mask must be boolean")
    channel_mask = point_mask.unsqueeze(-1).expand_as(prediction)
    selected_count = channel_mask.sum(dim=(1, 2))
    if torch.any(selected_count <= 0):
        raise ValueError("every sample needs at least one selected point")
    squared = (prediction - target).square()
    return (squared * channel_mask.to(squared.dtype)).sum(dim=(1, 2)) / (
        selected_count.to(squared.dtype)
    )


def contact_activation_probability(
    prediction_normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    active_threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Differentiable probability of exceeding the evaluation threshold."""

    if temperature <= 0.0:
        raise ValueError("activation temperature must be positive")
    if mean.shape != (1, 1, 6) or std.shape != (1, 1, 6):
        raise ValueError("mean/std tensors must be [1,1,6]")
    denormalized = prediction_normalized * std + mean
    return torch.sigmoid((denormalized - active_threshold) / temperature)


def _top_fraction_mean_tensor(values: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("cannot score an empty point region")
    count = max(1, int(np.ceil(values.numel() * fraction)))
    return torch.topk(values.reshape(-1), k=count, largest=True).values.mean()


def semantic_contact_loss(
    prediction_normalized: torch.Tensor,
    gt_affordance: torch.Tensor,
    instance_ids: torch.Tensor,
    target_instance_ids: torch.Tensor,
    targets: List[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    active_threshold: float,
    activation_temperature: float,
    foreground_bce_weight: float,
    dice_weight: float,
    competitor_suppression_weight: float,
    ranking_weight: float,
    ranking_margin: float,
) -> Dict[str, torch.Tensor]:
    """Sparse foreground and furniture-ranking loss for novel interactions.

    MAE/MSE alone rewards near-zero maps because active Bed/Whiteboard contact
    occupies only a small fraction of 8192x6 values.  This objective separately
    supervises active target channels, suppresses competing furniture, and
    requires target top-10% scores to exceed competitors by a positive margin.
    Whiteboard uses both any-joint and the native right-wrist channel (index 5);
    Bed uses pelvis and any-joint, matching the downstream semantic gate.
    """

    if prediction_normalized.shape != gt_affordance.shape:
        raise ValueError("semantic prediction/GT shape mismatch")
    if instance_ids.shape != prediction_normalized.shape[:2]:
        raise ValueError("instance_ids must be [B,N]")
    if len(targets) != prediction_normalized.shape[0]:
        raise ValueError("target list length differs from batch size")
    probability = contact_activation_probability(
        prediction_normalized,
        mean,
        std,
        active_threshold,
        activation_temperature,
    )
    zero = prediction_normalized.sum() * 0.0
    components = {
        "foreground_bce": zero,
        "dice": zero,
        "competitor_suppression": zero,
        "ranking": zero,
    }
    novel_count = 0
    eps = 1e-6
    for index, target in enumerate(targets):
        if target == "chair":
            continue
        if target not in {"bed", "whiteboard"}:
            raise ValueError(f"unknown target: {target}")
        novel_count += 1
        target_points = instance_ids[index] == target_instance_ids[index]
        candidate_points = (instance_ids[index] >= 1) & (instance_ids[index] <= 3)
        competitor_points = candidate_points & ~target_points
        active = (gt_affordance[index] >= active_threshold) & target_points[:, None]
        inactive_target = (~active) & target_points[:, None]
        if not torch.any(active):
            raise ValueError(f"{target}: no active GT target contact")
        p = probability[index]
        components["foreground_bce"] = components["foreground_bce"] + (
            -torch.log(p[active].clamp_min(eps)).mean()
        )
        if torch.any(inactive_target):
            components["foreground_bce"] = components["foreground_bce"] + 0.25 * (
                -torch.log((1.0 - p[inactive_target]).clamp_min(eps)).mean()
            )
        active_float = active.to(dtype=p.dtype)
        target_probability = p * target_points[:, None].to(dtype=p.dtype)
        intersection = (target_probability * active_float).sum()
        denominator = target_probability.sum() + active_float.sum()
        components["dice"] = components["dice"] + (
            1.0 - (2.0 * intersection + eps) / (denominator + eps)
        )
        if torch.any(competitor_points):
            competitor_probability = p[competitor_points]
            components["competitor_suppression"] = components[
                "competitor_suppression"
            ] + (-torch.log((1.0 - competitor_probability).clamp_min(eps)).mean())

        target_any = _top_fraction_mean_tensor(p[target_points].amax(dim=-1))
        competitor_any = []
        for instance_id in (1, 2, 3):
            if instance_id == int(target_instance_ids[index].item()):
                continue
            mask = instance_ids[index] == instance_id
            competitor_any.append(_top_fraction_mean_tensor(p[mask].amax(dim=-1)))
        any_margin_loss = F.relu(
            torch.as_tensor(ranking_margin, device=p.device, dtype=p.dtype)
            + torch.stack(competitor_any).max()
            - target_any
        )
        components["ranking"] = components["ranking"] + any_margin_loss
        if target == "bed":
            target_pelvis = _top_fraction_mean_tensor(p[target_points, 0])
            competitor_pelvis = []
            for instance_id in (1, 3):
                mask = instance_ids[index] == instance_id
                competitor_pelvis.append(_top_fraction_mean_tensor(p[mask, 0]))
            components["ranking"] = components["ranking"] + F.relu(
                torch.as_tensor(ranking_margin, device=p.device, dtype=p.dtype)
                + torch.stack(competitor_pelvis).max()
                - target_pelvis
            )
        elif target == "whiteboard":
            target_right_wrist = _top_fraction_mean_tensor(p[target_points, 5])
            competitor_right_wrist = []
            for instance_id in (1, 2):
                mask = instance_ids[index] == instance_id
                competitor_right_wrist.append(
                    _top_fraction_mean_tensor(p[mask, 5])
                )
            components["ranking"] = components["ranking"] + F.relu(
                torch.as_tensor(ranking_margin, device=p.device, dtype=p.dtype)
                + torch.stack(competitor_right_wrist).max()
                - target_right_wrist
            )
    if novel_count == 0:
        components["total"] = zero
        return components
    for key in tuple(components):
        components[key] = components[key] / float(novel_count)
    components["total"] = (
        foreground_bce_weight * components["foreground_bce"]
        + dice_weight * components["dice"]
        + competitor_suppression_weight * components["competitor_suppression"]
        + ranking_weight * components["ranking"]
    )
    return components


def sample_v5_timesteps(
    targets: List[str],
    num_timesteps: int,
    high_timestep_start: float,
    device: str,
) -> torch.Tensor:
    """Uniform Chair timesteps and a 50/50 uniform/high-noise novel mixture."""

    if not 0.0 < high_timestep_start < 1.0:
        raise ValueError("high timestep start must be in (0,1)")
    result = torch.randint(0, num_timesteps, (len(targets),), device=device)
    high_start = max(1, min(num_timesteps - 1, int(num_timesteps * high_timestep_start)))
    for index, target in enumerate(targets):
        if target in {"bed", "whiteboard"} and bool(torch.rand((), device=device) < 0.5):
            result[index] = torch.randint(high_start, num_timesteps, (), device=device)
    return result


def f1_at_threshold(
    prediction: np.ndarray, target: np.ndarray, threshold: float
) -> float:
    pred_active = np.asarray(prediction) >= threshold
    target_active = np.asarray(target) >= threshold
    tp = int(np.logical_and(pred_active, target_active).sum())
    fp = int(np.logical_and(pred_active, ~target_active).sum())
    fn = int(np.logical_and(~pred_active, target_active).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return float(2.0 * precision * recall / max(1e-12, precision + recall))


@torch.no_grad()
def assert_chair_uniform_loss_parity(
    model,
    diffusion,
    rows,
    device: str,
    seed: int,
) -> float:
    """Prove that v5 leaves the legacy Chair MSE numerically unchanged."""
    chair_ids = sorted(
        sample_id
        for sample_id, row in rows.items()
        if str(row["target"]) == "chair"
    )
    if not chair_ids:
        raise ValueError("Chair parity test requires a Chair training row")
    model.eval()
    batch = stack_batch(rows, [chair_ids[0]], device)
    if not torch.equal(batch["weights"], torch.ones_like(batch["weights"])):
        raise AssertionError("Chair sparse-loss weights are not exactly one")
    timestep = torch.tensor(
        [diffusion.num_timesteps // 2], dtype=torch.long, device=device
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(batch["x"].shape, generator=generator).to(device)
    kwargs = {
        "c_pc_xyz": batch["xyz"],
        "c_pc_feat": batch["feat"],
        "c_text": batch["text"],
    }
    prediction = predict_xstart(
        model, diffusion, batch["x"], timestep, kwargs, noise
    )
    sparse = weighted_xstart_loss(
        prediction, batch["x"], batch["weights"]
    )
    legacy = diffusion.training_losses(
        model,
        batch["x"],
        timestep,
        model_kwargs=kwargs,
        noise=noise,
    )["loss"]
    max_abs_diff = float((sparse - legacy).abs().max().item())
    if not torch.allclose(sparse, legacy, rtol=1e-6, atol=1e-7):
        raise AssertionError(
            "v5 uniform weighted loss does not match legacy diffusion MSE: "
            f"max_abs_diff={max_abs_diff:.9g}"
        )
    return max_abs_diff


@torch.no_grad()
def fixed_grid_loss(
    model,
    diffusion,
    rows,
    device: str,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    active_threshold: float,
):
    """Train-only fixed-noise weighted grid plus semantic x0 diagnostics."""
    model.eval()
    timesteps = sorted(
        set(
            [
                0,
                diffusion.num_timesteps // 10,
                diffusion.num_timesteps // 4,
                diffusion.num_timesteps // 2,
                3 * diffusion.num_timesteps // 4,
                diffusion.num_timesteps - 1,
            ]
        )
    )
    per_timestep = []
    uniform_per_timestep = []
    per_target = {target: [] for target in ("chair", "bed", "whiteboard")}
    per_target_uniform = {
        target: [] for target in ("chair", "bed", "whiteboard")
    }
    predictions = {sample_id: [] for sample_id in rows}
    for timestep in timesteps:
        losses = []
        uniform_losses = []
        for offset, sample_id in enumerate(sorted(rows)):
            batch = stack_batch(rows, [sample_id], device)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + timestep * 1009 + offset)
            noise = torch.randn(batch["x"].shape, generator=generator).to(device)
            prediction = predict_xstart(
                model,
                diffusion,
                batch["x"],
                torch.tensor([timestep], dtype=torch.long, device=device),
                {
                    "c_pc_xyz": batch["xyz"],
                    "c_pc_feat": batch["feat"],
                    "c_text": batch["text"],
                },
                noise,
            )
            value = float(
                weighted_xstart_loss(
                    prediction, batch["x"], batch["weights"]
                ).mean().item()
            )
            uniform_value = float((prediction - batch["x"]).square().mean().item())
            target = str(rows[sample_id]["target"])
            losses.append(value)
            uniform_losses.append(uniform_value)
            per_target[target].append(value)
            per_target_uniform[target].append(uniform_value)
            predictions[sample_id].append(prediction[0].cpu().numpy())
        per_timestep.append(float(np.mean(losses)))
        uniform_per_timestep.append(float(np.mean(uniform_losses)))

    semantic_rows = []
    candidate_names = ("chair", "bed", "whiteboard")
    for sample_id in sorted(rows):
        row = rows[sample_id]
        prediction = denormalize_contact(
            np.mean(predictions[sample_id], axis=0).astype(np.float32), mean, std
        )
        gt = np.asarray(row["gt"], dtype=np.float32)
        instance_ids = np.asarray(row["instance_ids"], dtype=np.int64)
        target = str(row["target"])
        target_mask = instance_ids == int(row["target_instance_id"])
        pelvis_scores = instance_scores(prediction, instance_ids, "pelvis")
        any_scores = instance_scores(prediction, instance_ids, "any_joint")
        right_wrist_scores = instance_scores(
            prediction, instance_ids, "right_wrist"
        )
        semantic_rows.append(
            {
                "sample_id": sample_id,
                "target": target,
                "target_region_mae": float(
                    np.abs(prediction[target_mask] - gt[target_mask]).mean()
                ),
                "f1_at_0_7": f1_at_threshold(
                    prediction[target_mask], gt[target_mask], active_threshold
                ),
                "pelvis_dominates": (
                    max(candidate_names, key=lambda name: pelvis_scores[name])
                    == target
                ),
                "any_joint_dominates": (
                    max(candidate_names, key=lambda name: any_scores[name])
                    == target
                ),
                "right_wrist_dominates": (
                    max(candidate_names, key=lambda name: right_wrist_scores[name])
                    == target
                ),
                "target_pelvis_top10": float(pelvis_scores[target]),
                "target_any_joint_top10": float(any_scores[target]),
                "target_right_wrist_top10": float(right_wrist_scores[target]),
                "sit_bed_any_top10": float(any_scores["bed"]),
                "sit_bed_pelvis_top10": float(pelvis_scores["bed"]),
                "sit_chair_primary": bool(
                    any_scores["chair"] >= any_scores["bed"] + 0.15
                    and pelvis_scores["chair"] >= pelvis_scores["bed"] + 0.15
                ),
                "sit_bed_candidate_visible": bool(
                    any_scores["bed"] >= 0.12
                    and any_scores["bed"] <= 0.40
                    and pelvis_scores["bed"] >= 0.06
                    and pelvis_scores["bed"] <= 0.25
                ),
            }
        )
    semantic_per_target = {}
    for target in candidate_names:
        selected = [row for row in semantic_rows if row["target"] == target]
        semantic_per_target[target] = {
            "count": len(selected),
            "target_region_mae": float(
                np.mean([row["target_region_mae"] for row in selected])
            ),
            "f1_at_0_7": float(np.mean([row["f1_at_0_7"] for row in selected])),
            "pelvis_dominance_rate": float(
                np.mean([row["pelvis_dominates"] for row in selected])
            ),
            "any_joint_dominance_rate": float(
                np.mean([row["any_joint_dominates"] for row in selected])
            ),
            "right_wrist_dominance_rate": float(
                np.mean([row["right_wrist_dominates"] for row in selected])
            ),
            "target_pelvis_top10": float(
                np.mean([row["target_pelvis_top10"] for row in selected])
            ),
            "target_any_joint_top10": float(
                np.mean([row["target_any_joint_top10"] for row in selected])
            ),
            "target_right_wrist_top10": float(
                np.mean([row["target_right_wrist_top10"] for row in selected])
            ),
            "sit_bed_any_top10": float(
                np.mean([row["sit_bed_any_top10"] for row in selected])
            ),
            "sit_bed_pelvis_top10": float(
                np.mean([row["sit_bed_pelvis_top10"] for row in selected])
            ),
            "sit_chair_primary_rate": float(
                np.mean([row["sit_chair_primary"] for row in selected])
            ),
            "sit_bed_candidate_visible_rate": float(
                np.mean([row["sit_bed_candidate_visible"] for row in selected])
            ),
        }
    return {
        "loss_definition": "weighted_normalized_start_x_mse",
        "timesteps": timesteps,
        "losses": per_timestep,
        "mean": float(np.mean(per_timestep)),
        "uniform_losses": uniform_per_timestep,
        "uniform_mean": float(np.mean(uniform_per_timestep)),
        "per_target_mean": {
            key: float(np.mean(value)) for key, value in per_target.items()
        },
        "per_target_uniform_mean": {
            key: float(np.mean(value)) for key, value in per_target_uniform.items()
        },
        "semantic": {"per_target": semantic_per_target, "per_sample": semantic_rows},
    }


def _candidate_probe_ids(
    rows: Mapping[str, Mapping[str, object]],
) -> List[str]:
    """Return the complete train partition; candidate selection has no subset mode."""

    counts = Counter(str(row["target"]) for row in rows.values())
    expected = Counter({"chair": 18, "whiteboard": 6, "bed": 1})
    if counts != expected:
        raise ValueError(
            "strict candidate rollout requires the complete train split: "
            f"expected={expected}, actual={counts}"
        )
    return sorted(rows)


def _rollout_region_metrics(
    prediction: np.ndarray,
    gt: np.ndarray,
    target_mask: np.ndarray,
    active_threshold: float,
) -> Dict[str, float]:
    selected_prediction = np.asarray(prediction[target_mask], dtype=np.float32)
    selected_gt = np.asarray(gt[target_mask], dtype=np.float32)
    return {
        "mae": float(np.abs(selected_prediction - selected_gt).mean()),
        "f1_at_0_7": f1_at_threshold(
            selected_prediction, selected_gt, active_threshold
        ),
    }


_BASELINE_PREDICTION_PROTOCOL_FILES = (
    "prepare/evaluate_fewshot_cdm.py",
    "prepare/fewshot_cdm_common.py",
    "models/cdm.py",
    "diffusion/gaussian_diffusion.py",
    "configs/model/cdm.yaml",
)


def _validate_external_original_baseline_cache(
    source_cache_dir: Optional[Path],
    *,
    args: argparse.Namespace,
    original_checkpoint: Path,
    probe_ids: List[str],
    train_rows: Mapping[str, Mapping[str, object]],
) -> Optional[Mapping[str, object]]:
    """Validate an existing original-CDM rollout cache for exact draw reuse.

    The old cache contract also hashes orchestration files.  Those files do not
    affect the numerical baseline prediction and may legitimately change when
    adding this resume/import feature.  Reuse is therefore gated by the exact
    prediction-critical subset below, plus every input tensor, seed, schedule,
    split, stats file, and original checkpoint.
    """

    if source_cache_dir is None:
        return None
    source_cache_dir = source_cache_dir.expanduser().resolve()
    contract_file = source_cache_dir / "contract.json"
    if not contract_file.is_file():
        raise FileNotFoundError(contract_file)
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    expected = {
        "schema": CONTRACT_SCHEMA,
        "partition": "train_complete",
        "heldout_sample_tensors_read": False,
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "split_file": str(args.split.expanduser().resolve()),
        "split_sha256": sha256_file(args.split.expanduser().resolve()),
        "stats_file": str(args.stats_file.expanduser().resolve()),
        "stats_sha256": sha256_file(args.stats_file.expanduser().resolve()),
        "original_checkpoint": str(original_checkpoint),
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "diffusion_steps": int(args.diffusion_steps),
        "k_samples": int(args.candidate_rollout_k),
        "seed": int(args.seed),
        "seed_partition": "train_audit",
        "probe_sample_ids": probe_ids,
        "sample_fingerprints": fingerprint_rows(train_rows, probe_ids),
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing incompatible original-baseline cache reuse: "
            + json.dumps(mismatches, sort_keys=True)
        )
    recorded_protocol = contract.get("protocol_file_sha256", {})
    current_protocol = {
        name: sha256_file(REPO_ROOT / name)
        for name in _BASELINE_PREDICTION_PROTOCOL_FILES
    }
    protocol_mismatches = {
        name: {
            "expected": current_hash,
            "actual": recorded_protocol.get(name),
        }
        for name, current_hash in current_protocol.items()
        if recorded_protocol.get(name) != current_hash
    }
    if protocol_mismatches:
        raise RuntimeError(
            "Refusing baseline cache generated by different prediction code: "
            + json.dumps(protocol_mismatches, sort_keys=True)
        )
    expected_draws = len(probe_ids) * args.candidate_rollout_k
    for sample_id in probe_ids:
        for draw in range(args.candidate_rollout_k):
            load_prediction(
                prediction_path(source_cache_dir, "original", sample_id, draw)
            )
    metadata = {
        "source_cache_dir": str(source_cache_dir),
        "source_contract_file": str(contract_file),
        "source_contract_sha256": sha256_json(contract),
        "prediction_critical_protocol_sha256": current_protocol,
        "validated_draw_count": expected_draws,
        "candidate_draws_reused": False,
    }
    print(
        f"[PASS] validated external original baseline cache: "
        f"draws={expected_draws} source={source_cache_dir}",
        flush=True,
    )
    return metadata


def select_candidate_by_train_rollout(
    cfg,
    original_checkpoint: Path,
    candidate_records: List[Dict[str, object]],
    train_rows: Mapping[str, Mapping[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    """Select a train-only candidate using actual reverse diffusion.

    One-step START_X metrics only form a shortlist.  Final checkpoint creation
    is forbidden unless an exact 500-step, K>=5 rollout over the complete train
    split activates and ranks the intended furniture.  Seeds and aggregate
    definitions intentionally match the independent train audit, preventing a
    K=1/subset checkpoint from being promoted and then failing the K=5 gate.
    """

    from evaluate_fewshot_cdm import (
        create_model,
        sample_contact_deterministic,
        stable_rollout_seeds,
    )

    ranked = sorted(
        candidate_records,
        key=lambda row: tuple(float(value) for value in row["one_step_rank"]),
        reverse=True,
    )[: args.candidate_shortlist]
    if not ranked:
        raise RuntimeError("one-step gates produced no candidate checkpoint")
    probe_ids = _candidate_probe_ids(train_rows)
    baseline_source = _validate_external_original_baseline_cache(
        args.rollout_baseline_cache_dir,
        args=args,
        original_checkpoint=original_checkpoint,
        probe_ids=probe_ids,
        train_rows=train_rows,
    )
    cache_dir = (
        args.rollout_cache_dir.expanduser().resolve()
        if args.rollout_cache_dir is not None
        else output_dir / "rollout_cache"
    )
    candidate_contract = []
    for record in ranked:
        candidate_path = Path(str(record["checkpoint"])).expanduser().resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        actual_hash = sha256_file(candidate_path)
        recorded_hash = str(record.get("checkpoint_sha256", ""))
        if recorded_hash and recorded_hash != actual_hash:
            raise RuntimeError(
                f"candidate checkpoint changed after training: {candidate_path}"
            )
        candidate_contract.append(
            {
                "step": int(record["step"]),
                "checkpoint": str(candidate_path),
                "checkpoint_sha256": actual_hash,
                "one_step_rank": [
                    float(value) for value in record["one_step_rank"]
                ],
            }
        )
    protocol_files = {
        name: sha256_file(REPO_ROOT / name)
        for name in (
            "prepare/train_fewshot_cdm.py",
            "prepare/fewshot_cdm_rollout_cache.py",
            "prepare/evaluate_fewshot_cdm.py",
            "prepare/fewshot_cdm_common.py",
            "prepare/fewshot_cdm_v5_semantics.py",
            "models/cdm.py",
            "diffusion/gaussian_diffusion.py",
            "configs/model/cdm.yaml",
        )
    }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "partition": "train_complete",
        "heldout_sample_tensors_read": False,
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "split_file": str(args.split.expanduser().resolve()),
        "split_sha256": sha256_file(args.split.expanduser().resolve()),
        "stats_file": str(args.stats_file.expanduser().resolve()),
        "stats_sha256": sha256_file(args.stats_file.expanduser().resolve()),
        "original_checkpoint": str(original_checkpoint),
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "diffusion_steps": int(args.diffusion_steps),
        "k_samples": int(args.candidate_rollout_k),
        "seed": int(args.seed),
        "seed_partition": "train_audit",
        "active_threshold": float(args.active_threshold),
        "candidate_dominance_margin": float(args.candidate_dominance_margin),
        "chair_max_relative_degradation": float(
            args.chair_max_relative_degradation
        ),
        "chair_train_min_dominance_rate": float(
            args.chair_train_min_dominance_rate
        ),
        "sit_bed_candidate_min_rate": float(args.sit_bed_candidate_min_rate),
        "novel_train_min_f1": float(args.novel_train_min_f1),
        "novel_train_min_dominance_rate": float(
            args.novel_train_min_dominance_rate
        ),
        "probe_sample_ids": probe_ids,
        "sample_fingerprints": fingerprint_rows(train_rows, probe_ids),
        "shortlisted_candidates": candidate_contract,
        "external_original_baseline": baseline_source,
        "protocol_file_sha256": protocol_files,
    }
    contract_sha256 = ensure_contract(cache_dir, contract)
    progress_file = cache_dir / "progress.json"
    total_draws = (
        (1 + len(ranked)) * len(probe_ids) * args.candidate_rollout_k
    )
    completed_draws = 0
    cache_hits = 0
    computed_this_run = 0
    compute_seconds: List[float] = []
    rollout_started = time.monotonic()

    def report_progress(
        role: str,
        sample_id: str,
        draw: int,
        cached: bool,
        candidate_step: Optional[int] = None,
    ) -> None:
        nonlocal completed_draws, cache_hits, computed_this_run
        completed_draws += 1
        cache_hits += int(cached)
        computed_this_run += int(not cached)
        elapsed = time.monotonic() - rollout_started
        average = float(np.mean(compute_seconds)) if compute_seconds else None
        remaining = max(0, total_draws - completed_draws)
        eta = None if average is None else average * remaining
        payload = {
            "schema": "history_affordance_v1_fewshot_cdm_rollout_progress_v1",
            "status": "RUNNING",
            "contract_sha256": contract_sha256,
            "completed_draws": completed_draws,
            "planned_draws_upper_bound": total_draws,
            "cache_hits_this_invocation": cache_hits,
            "computed_this_invocation": computed_this_run,
            "elapsed_seconds": elapsed,
            "mean_computed_draw_seconds": average,
            "eta_seconds_upper_bound": eta,
            "current": {
                "role": role,
                "candidate_step": candidate_step,
                "sample_id": sample_id,
                "draw": draw,
                "cache_hit": cached,
            },
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(progress_file, payload)
        eta_text = "unknown" if eta is None else f"{eta / 60.0:.1f}m"
        step_text = "-" if candidate_step is None else str(candidate_step)
        print(
            f"[ROLLOUT-PROGRESS] role={role} step={step_text} "
            f"sample={sample_id} draw={draw + 1}/{args.candidate_rollout_k} "
            f"cache={cached} done={completed_draws}/{total_draws} "
            f"eta_upper={eta_text}",
            flush=True,
        )

    configure_reproducibility(args.seed)
    original_model = None
    original_diffusion = None
    original_predictions: Dict[tuple, np.ndarray] = {}
    for sample_id in probe_ids:
        row = train_rows[sample_id]
        for draw in range(args.candidate_rollout_k):
            cache_file = prediction_path(
                cache_dir, "original", sample_id, draw
            )
            cached = cache_file.is_file()
            if cached:
                prediction = load_prediction(cache_file)
            elif baseline_source is not None:
                source_cache_dir = Path(
                    str(baseline_source["source_cache_dir"])
                )
                prediction = load_prediction(
                    prediction_path(
                        source_cache_dir, "original", sample_id, draw
                    )
                )
                atomic_save_prediction(cache_file, prediction)
                cached = True
            else:
                if original_model is None:
                    print(
                        "[ROLLOUT] loading original CDM for missing baseline draws",
                        flush=True,
                    )
                    original_model, original_diffusion = create_model(
                        cfg, original_checkpoint, args.device
                    )
                initial_seed, reverse_seed = stable_rollout_seeds(
                    args.seed, "train_audit", sample_id, draw
                )
                started = time.monotonic()
                normalized = sample_contact_deterministic(
                    original_model,
                    original_diffusion,
                    row,
                    initial_seed,
                    reverse_seed,
                    args.device,
                )
                prediction = denormalize_contact(normalized, mean, std).astype(
                    np.float32
                )
                atomic_save_prediction(cache_file, prediction)
                compute_seconds.append(time.monotonic() - started)
            original_predictions[(sample_id, draw)] = prediction
            report_progress("original", sample_id, draw, cached)
    if original_model is not None:
        del original_model, original_diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluated = []
    selected = None
    candidate_names = ("chair", "bed", "whiteboard")
    for candidate_record in ranked:
        candidate_path = Path(str(candidate_record["checkpoint"])).expanduser().resolve()
        candidate_step = int(candidate_record["step"])
        candidate_hash = sha256_file(candidate_path)
        cached_result_file = candidate_result_path(cache_dir, candidate_step)
        cached_result = load_bound_candidate_result(
            cached_result_file,
            contract_sha256,
            candidate_hash,
            candidate_step,
        )
        if cached_result is not None:
            evaluated.append(cached_result)
            skipped = len(probe_ids) * args.candidate_rollout_k
            completed_draws += skipped
            cache_hits += skipped
            print(
                f"[ROLLOUT-RESUME] reused complete candidate result "
                f"step={candidate_step} draws={skipped} "
                f"pass={cached_result['passed']}",
                flush=True,
            )
            if cached_result["passed"]:
                selected = cached_result
                break
            continue
        configure_reproducibility(args.seed)
        candidate_model = None
        candidate_diffusion = None
        per_target = {
            target: {
                "fewshot_mae": [],
                "original_mae": [],
                "zero_mae": [],
                "fewshot_f1": [],
                "original_f1": [],
                "ensemble_pelvis_dominance": [],
                "ensemble_any_joint_dominance": [],
                "ensemble_right_wrist_dominance": [],
                "per_draw_pelvis_dominance": [],
                "per_draw_any_joint_dominance": [],
                "per_draw_right_wrist_dominance": [],
                "per_draw_sit_chair_primary": [],
                "per_draw_sit_bed_candidate_visible": [],
                "ensemble_sit_chair_primary": [],
                "ensemble_sit_bed_candidate_visible": [],
            }
            for target in candidate_names
        }
        per_sample = []
        for sample_id in probe_ids:
            row = train_rows[sample_id]
            target = str(row["target"])
            gt = np.asarray(row["gt"], dtype=np.float32)
            instance_ids = np.asarray(row["instance_ids"], dtype=np.int64)
            target_mask = instance_ids == int(row["target_instance_id"])
            candidate_draws = []
            original_draws = []
            per_draw = []
            for draw in range(args.candidate_rollout_k):
                cache_file = prediction_path(
                    cache_dir,
                    "candidate",
                    sample_id,
                    draw,
                    candidate_step,
                )
                cached = cache_file.is_file()
                if cached:
                    prediction = load_prediction(cache_file)
                else:
                    if candidate_model is None:
                        print(
                            f"[ROLLOUT] loading candidate step={candidate_step} "
                            "for missing draws",
                            flush=True,
                        )
                        candidate_model, candidate_diffusion = create_model(
                            cfg, candidate_path, args.device
                        )
                    initial_seed, reverse_seed = stable_rollout_seeds(
                        args.seed, "train_audit", sample_id, draw
                    )
                    started = time.monotonic()
                    normalized = sample_contact_deterministic(
                        candidate_model,
                        candidate_diffusion,
                        row,
                        initial_seed,
                        reverse_seed,
                        args.device,
                    )
                    prediction = denormalize_contact(
                        normalized, mean, std
                    ).astype(np.float32)
                    atomic_save_prediction(cache_file, prediction)
                    compute_seconds.append(time.monotonic() - started)
                report_progress(
                    "candidate",
                    sample_id,
                    draw,
                    cached,
                    candidate_step,
                )
                original = original_predictions[(sample_id, draw)]
                zero = np.zeros_like(gt)
                candidate_draws.append(prediction)
                original_draws.append(original)
                fewshot_metric = _rollout_region_metrics(
                    prediction, gt, target_mask, args.active_threshold
                )
                original_metric = _rollout_region_metrics(
                    original, gt, target_mask, args.active_threshold
                )
                zero_metric = _rollout_region_metrics(
                    zero, gt, target_mask, args.active_threshold
                )
                scores = {
                    channel: instance_scores(prediction, instance_ids, channel)
                    for channel in ("pelvis", "any_joint", "right_wrist")
                }
                margins = {
                    channel: float(
                        scores[channel][target]
                        - max(
                            scores[channel][name]
                            for name in candidate_names
                            if name != target
                        )
                    )
                    for channel in ("pelvis", "any_joint", "right_wrist")
                }
                values = per_target[target]
                values["fewshot_mae"].append(fewshot_metric["mae"])
                values["original_mae"].append(original_metric["mae"])
                values["zero_mae"].append(zero_metric["mae"])
                values["fewshot_f1"].append(fewshot_metric["f1_at_0_7"])
                values["original_f1"].append(original_metric["f1_at_0_7"])
                values["per_draw_pelvis_dominance"].append(
                    margins["pelvis"] >= args.candidate_dominance_margin
                )
                values["per_draw_any_joint_dominance"].append(
                    margins["any_joint"] >= args.candidate_dominance_margin
                )
                values["per_draw_right_wrist_dominance"].append(
                    margins["right_wrist"] >= args.candidate_dominance_margin
                )
                draw_sit_chair_primary = (
                    scores["any_joint"]["chair"]
                    >= scores["any_joint"]["bed"] + args.sit_chair_bed_margin
                    and scores["pelvis"]["chair"]
                    >= scores["pelvis"]["bed"] + args.sit_chair_bed_margin
                )
                draw_sit_bed_candidate_visible = (
                    args.sit_bed_any_min
                    <= scores["any_joint"]["bed"]
                    <= args.sit_bed_any_max
                    and args.sit_bed_pelvis_min
                    <= scores["pelvis"]["bed"]
                    <= args.sit_bed_pelvis_max
                )
                values["per_draw_sit_chair_primary"].append(
                    draw_sit_chair_primary if target == "chair" else True
                )
                values["per_draw_sit_bed_candidate_visible"].append(
                    draw_sit_bed_candidate_visible
                    if target == "chair"
                    else True
                )
                per_draw.append(
                    {
                        "draw": draw,
                        "fewshot": fewshot_metric,
                        "original": original_metric,
                        "zero": zero_metric,
                        "dominance_margin": margins,
                        "sit_chair_primary": draw_sit_chair_primary,
                        "sit_bed_candidate_visible": (
                            draw_sit_bed_candidate_visible
                        ),
                    }
                )
            ensemble_prediction = np.mean(candidate_draws, axis=0).astype(np.float32)
            ensemble_original = np.mean(original_draws, axis=0).astype(np.float32)
            ensemble_scores = {
                channel: instance_scores(
                    ensemble_prediction, instance_ids, channel
                )
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            ensemble_margins = {
                channel: float(
                    ensemble_scores[channel][target]
                    - max(
                        ensemble_scores[channel][name]
                        for name in candidate_names
                        if name != target
                    )
                )
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            values["ensemble_pelvis_dominance"].append(
                ensemble_margins["pelvis"] >= args.candidate_dominance_margin
            )
            values["ensemble_any_joint_dominance"].append(
                ensemble_margins["any_joint"] >= args.candidate_dominance_margin
            )
            values["ensemble_right_wrist_dominance"].append(
                ensemble_margins["right_wrist"]
                >= args.candidate_dominance_margin
            )
            sit_chair_primary = (
                ensemble_scores["any_joint"]["chair"]
                >= ensemble_scores["any_joint"]["bed"]
                + args.sit_chair_bed_margin
                and ensemble_scores["pelvis"]["chair"]
                >= ensemble_scores["pelvis"]["bed"]
                + args.sit_chair_bed_margin
            )
            sit_bed_candidate_visible = (
                args.sit_bed_any_min
                <= ensemble_scores["any_joint"]["bed"]
                <= args.sit_bed_any_max
                and args.sit_bed_pelvis_min
                <= ensemble_scores["pelvis"]["bed"]
                <= args.sit_bed_pelvis_max
            )
            values["ensemble_sit_chair_primary"].append(
                sit_chair_primary if target == "chair" else True
            )
            values["ensemble_sit_bed_candidate_visible"].append(
                sit_bed_candidate_visible if target == "chair" else True
            )
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "ensemble": {
                        "fewshot": _rollout_region_metrics(
                            ensemble_prediction,
                            gt,
                            target_mask,
                            args.active_threshold,
                        ),
                        "original": _rollout_region_metrics(
                            ensemble_original,
                            gt,
                            target_mask,
                            args.active_threshold,
                        ),
                        "dominance_margin": ensemble_margins,
                        "candidate_scores": ensemble_scores,
                        "sit_chair_primary": sit_chair_primary,
                        "sit_bed_candidate_visible": sit_bed_candidate_visible,
                    },
                    "per_draw": per_draw,
                }
            )
        if candidate_model is not None:
            del candidate_model, candidate_diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        aggregate = {
            target: {
                key: float(np.mean(value))
                for key, value in metrics.items()
            }
            for target, metrics in per_target.items()
        }
        chair_original_mae = aggregate["chair"]["original_mae"]
        chair_degradation = (
            aggregate["chair"]["fewshot_mae"] - chair_original_mae
        ) / max(chair_original_mae, 1e-12)
        checks = {
            "chair_mae_retained": (
                chair_degradation <= args.chair_max_relative_degradation
            ),
            "chair_pelvis_dominance": (
                aggregate["chair"]["ensemble_pelvis_dominance"]
                >= args.chair_train_min_dominance_rate
            ),
            "chair_any_joint_dominance": (
                aggregate["chair"]["ensemble_any_joint_dominance"]
                >= args.chair_train_min_dominance_rate
            ),
            "chair_per_draw_dominance": (
                aggregate["chair"]["per_draw_pelvis_dominance"] >= 0.80
                and aggregate["chair"]["per_draw_any_joint_dominance"] >= 0.80
            ),
            "sit_chair_remains_primary": (
                aggregate["chair"]["ensemble_sit_chair_primary"]
                >= args.chair_train_min_dominance_rate
            ),
            "sit_bed_candidate_visible_and_bounded": (
                aggregate["chair"]["ensemble_sit_bed_candidate_visible"]
                >= args.sit_bed_candidate_min_rate
            ),
            "sit_per_draw_chair_primary": (
                aggregate["chair"]["per_draw_sit_chair_primary"]
                >= args.chair_train_min_dominance_rate
            ),
            "sit_per_draw_bed_candidate_visible_and_bounded": (
                aggregate["chair"]["per_draw_sit_bed_candidate_visible"]
                >= args.sit_bed_candidate_min_rate
            ),
            "bed_f1_improved": (
                aggregate["bed"]["fewshot_f1"]
                > aggregate["bed"]["original_f1"]
            ),
            "bed_f1_active": (
                aggregate["bed"]["fewshot_f1"] >= args.novel_train_min_f1
            ),
            "bed_mae_beats_zero": (
                aggregate["bed"]["fewshot_mae"] < aggregate["bed"]["zero_mae"]
            ),
            "bed_pelvis_dominance": (
                aggregate["bed"]["ensemble_pelvis_dominance"]
                >= args.novel_train_min_dominance_rate
            ),
            "bed_any_joint_dominance": (
                aggregate["bed"]["ensemble_any_joint_dominance"]
                >= args.novel_train_min_dominance_rate
            ),
            "bed_per_draw_dominance": (
                aggregate["bed"]["per_draw_pelvis_dominance"] >= 0.80
                and aggregate["bed"]["per_draw_any_joint_dominance"] >= 0.80
            ),
            "whiteboard_f1_improved": (
                aggregate["whiteboard"]["fewshot_f1"]
                > aggregate["whiteboard"]["original_f1"]
            ),
            "whiteboard_f1_active": (
                aggregate["whiteboard"]["fewshot_f1"]
                >= args.novel_train_min_f1
            ),
            "whiteboard_mae_beats_zero": (
                aggregate["whiteboard"]["fewshot_mae"]
                < aggregate["whiteboard"]["zero_mae"]
            ),
            "whiteboard_any_joint_dominance": (
                aggregate["whiteboard"]["ensemble_any_joint_dominance"]
                >= args.novel_train_min_dominance_rate
            ),
            "whiteboard_right_wrist_dominance": (
                aggregate["whiteboard"]["ensemble_right_wrist_dominance"]
                >= args.novel_train_min_dominance_rate
            ),
            "whiteboard_per_draw_any_joint_dominance": (
                aggregate["whiteboard"]["per_draw_any_joint_dominance"] >= 0.80
            ),
            "whiteboard_per_draw_right_wrist_dominance": (
                aggregate["whiteboard"]["per_draw_right_wrist_dominance"]
                >= 0.80
            ),
        }
        rank = [
            min(
                aggregate["bed"]["fewshot_f1"],
                aggregate["whiteboard"]["fewshot_f1"],
            ),
            min(
                aggregate["bed"]["ensemble_any_joint_dominance"],
                aggregate["whiteboard"]["ensemble_any_joint_dominance"],
            ),
            aggregate["whiteboard"]["ensemble_right_wrist_dominance"],
            aggregate["chair"]["per_draw_sit_bed_candidate_visible"],
            aggregate["chair"]["per_draw_sit_chair_primary"],
            aggregate["chair"]["ensemble_sit_bed_candidate_visible"],
            -max(chair_degradation, 0.0),
        ]
        result = {
            "contract_sha256": contract_sha256,
            "step": candidate_step,
            "checkpoint": str(candidate_path),
            "checkpoint_sha256": candidate_hash,
            "one_step_rank": candidate_record["one_step_rank"],
            "rollout_rank": rank,
            "passed": all(checks.values()),
            "checks": checks,
            "chair_relative_mae_degradation": chair_degradation,
            "aggregate": aggregate,
            "per_sample_draw": per_sample,
        }
        atomic_write_json(cached_result_file, result)
        evaluated.append(result)
        if result["passed"]:
            selected = result
        print(
            f"[CANDIDATE-ROLLOUT] step={result['step']} "
            f"bed_f1={aggregate['bed']['fewshot_f1']:.4f} "
            f"whiteboard_f1={aggregate['whiteboard']['fewshot_f1']:.4f} "
            f"chair_deg={chair_degradation:+.4f} pass={result['passed']}",
            flush=True,
        )
        # Candidates are already ordered by the train-only one-step rank.  The
        # first one satisfying the exact K>=5 all-train rollout contract is the
        # selected checkpoint; later candidates need not consume GPU time.
        if selected is not None:
            break
    selection_file = output_dir / "rollout_candidate_selection.json"
    payload = {
        "schema": "history_affordance_v1_fewshot_cdm_candidate_rollout_v1",
        "partition": "train_complete",
        "heldout_sample_tensors_read": False,
        "diffusion_steps": args.diffusion_steps,
        "k_samples": args.candidate_rollout_k,
        "seed_partition": "train_audit",
        "complete_train_partition": True,
        "probe_sample_ids": probe_ids,
        "shortlisted_candidate_count": len(ranked),
        "candidate_count": len(evaluated),
        "evaluated": evaluated,
        "rollout_cache": {
            "directory": str(cache_dir),
            "contract_file": str(cache_dir / "contract.json"),
            "contract_sha256": contract_sha256,
            "progress_file": str(progress_file),
            "crash_safe_per_draw": True,
            "resumable": True,
        },
    }
    if selected is None:
        payload["status"] = "FAIL"
        atomic_write_json(selection_file, payload)
        atomic_write_json(
            progress_file,
            {
                "schema": "history_affordance_v1_fewshot_cdm_rollout_progress_v1",
                "status": "FAIL",
                "contract_sha256": contract_sha256,
                "completed_draws": completed_draws,
                "planned_draws_upper_bound": total_draws,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            "[FAIL] No shortlisted checkpoint passed train-only full-rollout "
            f"selection; see {selection_file}. Held-out evaluation is forbidden.",
            flush=True,
        )
        raise SystemExit(2)
    payload["status"] = "PASS"
    payload["selected"] = selected
    atomic_write_json(selection_file, payload)
    atomic_write_json(
        progress_file,
        {
            "schema": "history_affordance_v1_fewshot_cdm_rollout_progress_v1",
            "status": "PASS",
            "contract_sha256": contract_sha256,
            "completed_draws": completed_draws,
            "planned_draws_upper_bound": total_draws,
            "selected_step": int(selected["step"]),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "selected": selected,
        "selection_file": str(selection_file),
        "selection_file_sha256": sha256_file(selection_file),
        "partition": "train_complete",
        "complete_train_partition": True,
        "k_samples": args.candidate_rollout_k,
        "seed_partition": "train_audit",
        "probe_sample_ids": probe_ids,
        "evaluated_candidate_count": len(evaluated),
        "rollout_cache": payload["rollout_cache"],
    }


def _load_rollout_only_candidates(
    output_dir: Path, expected_steps: int
) -> List[Dict[str, object]]:
    """Strictly recover completed one-step training without touching test data."""

    candidate_file = output_dir / "one_step_candidates.json"
    train_log_file = output_dir / "train_log.jsonl"
    if not candidate_file.is_file() or not train_log_file.is_file():
        raise FileNotFoundError(
            "--rollout-only requires the completed v5 training artifacts "
            f"{candidate_file} and {train_log_file}"
        )
    payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    if payload.get("schema") != (
        "history_affordance_v1_fewshot_cdm_one_step_candidates_v2"
    ):
        raise RuntimeError(f"{candidate_file}: schema mismatch")
    if payload.get("partition") != "train_only":
        raise RuntimeError(f"{candidate_file}: not a train-only artifact")
    if payload.get("heldout_sample_tensors_read") is not False:
        raise RuntimeError(f"{candidate_file}: held-out isolation not proven")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{candidate_file}: no candidate checkpoints")
    for record in candidates:
        path = Path(str(record["checkpoint"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(record.get("checkpoint_sha256", "")):
            raise RuntimeError(f"{path}: candidate hash mismatch")
        if not record.get("selection", {}).get("passed"):
            raise RuntimeError(f"{path}: candidate did not pass one-step gates")
    log_steps = []
    with train_log_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            log_steps.append(int(row["step"]))
    if log_steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(
            f"{train_log_file}: expected exact completed steps 1..{expected_steps}, "
            f"found {len(log_steps)} rows ending at "
            f"{log_steps[-1] if log_steps else None}"
        )
    print(
        f"[PASS] rollout-only recovered {len(candidates)} train-only candidates "
        f"and exact {expected_steps}-step log",
        flush=True,
    )
    return candidates


def _reconstruct_original_context_for_rollout_only(
    cfg,
    checkpoint: Path,
    train_rows: Mapping[str, Mapping[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Mapping[str, object], Mapping[str, object], float]:
    """Recompute only cheap provenance values omitted by the interrupted run."""

    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    print(
        "[ROLLOUT-ONLY] reconstructing original-grid and zero-init LoRA metadata",
        flush=True,
    )
    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(checkpoint))
    model.eval()
    parity_id = sorted(train_rows)[0]
    parity_batch = stack_batch(train_rows, [parity_id], args.device)
    parity_timestep = torch.tensor(
        [diffusion.num_timesteps // 2], dtype=torch.long, device=args.device
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 500000)
    parity_noise = torch.randn(parity_batch["x"].shape, generator=generator).to(
        args.device
    )
    kwargs = {
        "c_pc_xyz": parity_batch["xyz"],
        "c_pc_feat": parity_batch["feat"],
        "c_text": parity_batch["text"],
    }
    with torch.no_grad():
        parity_before = predict_xstart(
            model,
            diffusion,
            parity_batch["x"],
            parity_timestep,
            kwargs,
            parity_noise,
        )
    install_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    with torch.no_grad():
        parity_after = predict_xstart(
            model,
            diffusion,
            parity_batch["x"],
            parity_timestep,
            kwargs,
            parity_noise,
        )
    if not torch.equal(parity_before, parity_after):
        raise AssertionError("rollout-only zero-init LoRA parity failed")
    metadata = dict(lora_metadata(model))
    chair_uniform_parity = assert_chair_uniform_loss_parity(
        model, diffusion, train_rows, args.device, args.seed + 600000
    )
    initial_grid = fixed_grid_loss(
        model,
        diffusion,
        train_rows,
        args.device,
        args.seed + 700000,
        mean,
        std,
        args.active_threshold,
    )
    del model, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "[PASS] rollout-only reconstructed original provenance without "
        "held-out tensor access",
        flush=True,
    )
    return initial_grid, metadata, chair_uniform_parity


def _finalize_v5_run(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    split_file: Path,
    output_dir: Path,
    train_rows: Mapping[str, Mapping[str, object]],
    test_ids: set,
    counts: Counter,
    original_hash: str,
    initial_grid: Mapping[str, object],
    candidate_records: List[Dict[str, object]],
    rollout_selection: Mapping[str, object],
    lora_info: Mapping[str, object],
    chair_uniform_parity: float,
) -> Mapping[str, object]:
    """Atomically promote the rollout winner and write the downstream contract."""

    selected_rollout = rollout_selection["selected"]
    selected_candidate = Path(
        str(selected_rollout["checkpoint"])
    ).expanduser().resolve()
    selected_step = int(selected_rollout["step"])
    selected_record = next(
        row for row in candidate_records if int(row["step"]) == selected_step
    )
    best_grid = dict(selected_record["grid"])
    best_selection = dict(selected_record["selection"])
    checkpoint_file = output_dir / "fewshot_cdm.pt"
    temporary_checkpoint = output_dir / ".fewshot_cdm.pt.tmp"
    temporary_checkpoint.unlink(missing_ok=True)
    shutil.copyfile(selected_candidate, temporary_checkpoint)
    if sha256_file(temporary_checkpoint) != str(
        selected_rollout["checkpoint_sha256"]
    ):
        temporary_checkpoint.unlink(missing_ok=True)
        raise RuntimeError("atomic promotion copy changed candidate bytes")
    temporary_checkpoint.replace(checkpoint_file)
    summary = {
        "schema": "history_affordance_v1_fewshot_cdm_train_v5r4",
        "status": "PASS",
        "selection_data": "train_only",
        "test_partition_read_during_training": False,
        "test_sample_data_read_during_training": False,
        "test_ids_used_only_for_exclusion_assertion": True,
        "initialization": {
            "checkpoint": str(checkpoint),
            "sha256": original_hash,
            "standard_original_path": not args.allow_nonstandard_pretrained,
            "one_sample_diagnostic_checkpoint_used": False,
        },
        "split": {
            "file": str(split_file),
            "sha256": sha256_file(split_file),
            "train_count": len(train_rows),
            "held_out_test_count": len(test_ids),
            "train_target_counts": dict(counts),
        },
        "sampling": {
            "method": "chair3_bed1_whiteboard1_shuffled_cycles",
            "batch_size": 5,
            "chair_replay_per_step": args.chair_replay_per_step,
            "bed_per_step": 1,
            "whiteboard_per_step": 1,
        },
        "regularization": {
            "method": (
                "frozen_original_cdm_plus_zero_init_lora_and_"
                "chair_region_multinoise_teacher_v5r4"
            ),
            "chair_teacher_weight": args.chair_teacher_weight,
            "chair_high_timestep_weight": args.chair_high_timestep_weight,
            "chair_high_teacher_weight": args.chair_high_teacher_weight,
            "chair_teacher_region": "target_chair_instance_only",
            "bed_candidate_region_excluded_from_chair_teacher": True,
            "lora_l2_weight": args.lora_l2_weight,
            "original_checkpoint_sha256": original_hash,
            "lora": dict(lora_info),
            "zero_init_bitwise_original": True,
        },
        "data_objective": {
            "method": "rollout_aligned_sparse_semantic_multinoise_start_x_v5r4",
            "diffusion_prediction_target": "START_X",
            "chair_uses_legacy_uniform_weights": True,
            "base_weight": 1.0,
            "target_instance_additive_weight": args.target_instance_weight,
            "target_foreground_additive_weight": args.target_foreground_weight,
            "active_threshold": args.active_threshold,
            "background_remains_supervised": True,
            "chair_uniform_legacy_parity_max_abs_diff": chair_uniform_parity,
            "semantic_weight": args.semantic_weight,
            "foreground_bce_weight": args.foreground_bce_weight,
            "dice_weight": args.dice_weight,
            "competitor_suppression_weight": args.competitor_suppression_weight,
            "ranking_weight": args.ranking_weight,
            "ranking_margin": args.ranking_margin,
            "activation_temperature": args.activation_temperature,
            "high_timestep_weight": args.high_timestep_weight,
            "high_timestep_start_fraction": args.high_timestep_start,
            "novel_high_timestep_extra_forward": True,
            "chair_high_timestep_extra_forward": True,
            "prompt_policy_id": PROMPT_POLICY_ID,
            "prompt_by_target": {
                target: prompt_for_target(target)
                for target in ("chair", "bed", "whiteboard")
            },
            "whiteboard_semantic_channel": "right_wrist_native_index_5",
            "sit_multicandidate": {
                "supervision_type": "weak_semantic_prior_only",
                "motion_gt_relabelled_or_copied": False,
                "weight": args.sit_multicandidate_weight,
                "bed_any_joint_band": [
                    args.sit_bed_any_min,
                    args.sit_bed_any_max,
                ],
                "bed_pelvis_band": [
                    args.sit_bed_pelvis_min,
                    args.sit_bed_pelvis_max,
                ],
                "chair_primary_margin": args.sit_chair_bed_margin,
            },
        },
        "steps": args.steps,
        "best_step": selected_step,
        "lr": args.lr,
        "diffusion_steps": args.diffusion_steps,
        "initial_train_grid": dict(initial_grid),
        "best_train_grid": best_grid,
        "checkpoint_selection": {
            "method": "one_step_shortlist_then_train_full_rollout",
            "gate_passed": True,
            "novel_min_relative_improvement": args.novel_min_relative_improvement,
            "chair_max_relative_degradation": args.chair_max_relative_degradation,
            "novel_train_min_f1": args.novel_train_min_f1,
            "novel_train_min_dominance_rate": args.novel_train_min_dominance_rate,
            "chair_train_min_dominance_rate": args.chair_train_min_dominance_rate,
            "sit_bed_candidate_min_rate": args.sit_bed_candidate_min_rate,
            "best": best_selection,
            "candidate_count": len(candidate_records),
            "shortlist_size": args.candidate_shortlist,
            "rollout_selection": dict(rollout_selection),
            "rollout_gate_passed": True,
        },
        "relative_train_grid_improvement": (
            float(initial_grid["mean"]) - float(best_grid["mean"])
        )
        / float(initial_grid["mean"]),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print("[PASS] train-only full-rollout selected v5 few-shot CDM checkpoint")
    print(
        f"[OK] train grid: {float(initial_grid['mean']):.8f} -> "
        f"{float(best_grid['mean']):.8f} at step {selected_step}"
    )
    print(f"[OK] target-specific selection: {best_selection}")
    print(f"[OK] saved: {checkpoint_file}")
    print(f"[OK] saved: {output_dir / 'summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("--steps and --eval-every must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.chair_replay_per_step != 3:
        raise ValueError("v5 strictly requires --chair-replay-per-step 3")
    if args.lora_rank <= 0 or args.lora_alpha <= 0.0:
        raise ValueError("LoRA rank/alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0,1)")
    if (
        args.lora_l2_weight < 0.0
        or args.chair_teacher_weight <= 0.0
        or args.chair_high_timestep_weight <= 0.0
        or args.chair_high_teacher_weight <= 0.0
    ):
        raise ValueError("LoRA and Chair multi-noise preservation weights are invalid")
    if args.high_timestep_weight <= 0.0:
        raise ValueError("--high-timestep-weight must be positive")
    if not 0.0 < args.high_timestep_start < 1.0:
        raise ValueError("--high-timestep-start must be in (0,1)")
    if args.semantic_weight <= 0.0:
        raise ValueError("--semantic-weight must be positive")
    if any(
        value < 0.0
        for value in (
            args.foreground_bce_weight,
            args.dice_weight,
            args.competitor_suppression_weight,
            args.ranking_weight,
        )
    ):
        raise ValueError("semantic component weights must be nonnegative")
    if args.ranking_margin <= 0.0 or args.activation_temperature <= 0.0:
        raise ValueError("ranking margin/activation temperature must be positive")
    if args.novel_min_relative_improvement <= 0.0:
        raise ValueError("--novel-min-relative-improvement must be positive")
    if not 0.0 <= args.chair_max_relative_degradation <= 0.10:
        raise ValueError("--chair-max-relative-degradation must be in [0,0.10]")
    if args.target_instance_weight <= 0.0:
        raise ValueError("--target-instance-weight must be positive")
    if args.target_foreground_weight <= 0.0:
        raise ValueError("--target-foreground-weight must be positive")
    if not 0.0 < args.active_threshold < 1.0:
        raise ValueError("--active-threshold must be in (0,1)")
    if not 0.0 < args.novel_train_min_f1 <= 1.0:
        raise ValueError("--novel-train-min-f1 must be in (0,1]")
    if not 0.0 < args.novel_train_min_dominance_rate <= 1.0:
        raise ValueError("--novel-train-min-dominance-rate must be in (0,1]")
    if not 0.0 < args.chair_train_min_dominance_rate <= 1.0:
        raise ValueError("--chair-train-min-dominance-rate must be in (0,1]")
    if not 0.0 < args.sit_bed_candidate_min_rate <= 1.0:
        raise ValueError("--sit-bed-candidate-min-rate must be in (0,1]")
    if args.sit_bed_candidate_min_rate >= args.chair_train_min_dominance_rate:
        raise ValueError(
            "weak Sit-Bed coverage must be lower than strict Chair dominance"
        )
    if args.diagnostic_checkpoint_count <= 0:
        raise ValueError("--diagnostic-checkpoint-count must be positive")
    if args.candidate_shortlist <= 0:
        raise ValueError("candidate shortlist must be positive")
    if args.candidate_rollout_k < 5:
        raise ValueError("strict v5r4 requires --candidate-rollout-k >= 5")
    if args.candidate_dominance_margin <= 0.0:
        raise ValueError("--candidate-dominance-margin must be positive")
    if abs(args.active_threshold - 0.7) > 1e-12:
        raise ValueError("strict v5r4 fixes --active-threshold at 0.7")
    if args.rollout_only and args.stop_after_one_step:
        raise ValueError("--rollout-only and --stop-after-one-step are mutually exclusive")
    expected_sit_values = (0.50, 0.12, 0.40, 0.06, 0.25, 0.15)
    actual_sit_values = (
        args.sit_multicandidate_weight,
        args.sit_bed_any_min,
        args.sit_bed_any_max,
        args.sit_bed_pelvis_min,
        args.sit_bed_pelvis_max,
        args.sit_chair_bed_margin,
    )
    if any(abs(a - b) > 1e-12 for a, b in zip(actual_sit_values, expected_sit_values)):
        raise ValueError(
            "strict v5r4 fixes Sit multi-candidate policy at "
            "weight=.50, any=[.12,.40], pelvis=[.06,.25], margin=.15"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configure_reproducibility(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    split_file = args.split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    checkpoint = args.pretrained_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.rollout_only:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"--rollout-only requires the existing v5 directory: {output_dir}"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Strict v5 requires a fresh output directory; refusing stale "
                f"artifacts: {output_dir}. Use --rollout-only only for a completed "
                "4000-step v5 directory."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    assert_original_checkpoint(checkpoint, args.allow_nonstandard_pretrained)
    split = load_split(split_file)
    mean, std = load_stats(stats_file)
    train_rows = load_rows(
        dataset_root,
        split,
        "train",
        mean,
        std,
        args.target_instance_weight,
        args.target_foreground_weight,
        args.active_threshold,
    )
    # Only IDs are read to prove exclusion.  No held-out manifest, point cloud,
    # text, GT, or tensor is loaded anywhere in this training program.
    test_ids = set(str(v) for v in split["cdm_fewshot"]["test"])
    if set(train_rows) & test_ids:
        raise AssertionError("held-out test data entered training loader")
    counts = Counter(str(row["target"]) for row in train_rows.values())
    if counts != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise AssertionError(f"unexpected train target counts: {counts}")

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    original_hash = sha256_file(checkpoint)
    if args.rollout_only:
        candidate_records = _load_rollout_only_candidates(output_dir, args.steps)
        rollout_selection = select_candidate_by_train_rollout(
            cfg,
            checkpoint,
            candidate_records,
            train_rows,
            mean,
            std,
            args,
            output_dir,
        )
        initial_grid, resume_lora_info, chair_uniform_parity = (
            _reconstruct_original_context_for_rollout_only(
                cfg,
                checkpoint,
                train_rows,
                mean,
                std,
                args,
            )
        )
        _finalize_v5_run(
            args=args,
            checkpoint=checkpoint,
            split_file=split_file,
            output_dir=output_dir,
            train_rows=train_rows,
            test_ids=test_ids,
            counts=counts,
            original_hash=original_hash,
            initial_grid=initial_grid,
            candidate_records=candidate_records,
            rollout_selection=rollout_selection,
            lora_info=resume_lora_info,
            chair_uniform_parity=chair_uniform_parity,
        )
        return

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(checkpoint))
    model.eval()
    parity_id = sorted(train_rows)[0]
    parity_batch = stack_batch(train_rows, [parity_id], args.device)
    parity_timestep = torch.tensor(
        [diffusion.num_timesteps // 2], dtype=torch.long, device=args.device
    )
    parity_generator = torch.Generator(device="cpu")
    parity_generator.manual_seed(args.seed + 500000)
    parity_noise = torch.randn(
        parity_batch["x"].shape, generator=parity_generator
    ).to(args.device)
    with torch.no_grad():
        parity_before = predict_xstart(
            model,
            diffusion,
            parity_batch["x"],
            parity_timestep,
            {
                "c_pc_xyz": parity_batch["xyz"],
                "c_pc_feat": parity_batch["feat"],
                "c_text": parity_batch["text"],
            },
            parity_noise,
        )
    lora_module_names = install_lora(
        model,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    )
    with torch.no_grad():
        parity_after = predict_xstart(
            model,
            diffusion,
            parity_batch["x"],
            parity_timestep,
            {
                "c_pc_xyz": parity_batch["xyz"],
                "c_pc_feat": parity_batch["feat"],
                "c_text": parity_batch["text"],
            },
            parity_noise,
        )
    if not torch.equal(parity_before, parity_after):
        raise AssertionError("zero-init LoRA changed the original CDM output")
    print(
        f"[PASS] zero-init LoRA preserves original CDM bitwise; "
        f"modules={len(lora_module_names)}"
    )
    named_trainable = lora_named_parameters(model)
    trainable = list(named_trainable.values())
    if not named_trainable:
        raise RuntimeError("CDM has no trainable LoRA parameters")
    chair_uniform_parity = assert_chair_uniform_loss_parity(
        model, diffusion, train_rows, args.device, args.seed + 600000
    )
    print(
        "[PASS] Chair all-one weighted START_X loss matches legacy MSE: "
        f"max_abs_diff={chair_uniform_parity:.9g}"
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    replay_counts = {
        "chair": args.chair_replay_per_step,
        "bed": 1,
        "whiteboard": 1,
    }
    batches = target_balanced_batches(
        list(train_rows.values()),
        args.steps,
        args.seed,
        replay_counts=replay_counts,
    )
    for batch in batches:
        target_counts = Counter(str(train_rows[value]["target"]) for value in batch)
        if len(batch) != 5 or target_counts != Counter(replay_counts):
            raise AssertionError("Chair3+Bed1+Whiteboard1 replay contract failed")

    initial_grid = fixed_grid_loss(
        model,
        diffusion,
        train_rows,
        args.device,
        args.seed + 700000,
        mean,
        std,
        args.active_threshold,
    )
    checkpoint_file = output_dir / "fewshot_cdm.pt"
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    diagnostic_dir = output_dir / "diagnostic_checkpoints"
    diagnostic_dir.mkdir(exist_ok=True)
    best_grid = None
    best_selection = None
    best_rank = None
    best_step = 0
    candidate_records = []
    diagnostic_records = []
    logs = []
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    for step, sample_ids in enumerate(batches, start=1):
        # Gradients do not require train mode.  The pretrained CDM and its
        # encoders remain in inference mode so dropout/BN cannot drift; only
        # the optional LoRA dropout is enabled for adaptation.
        set_frozen_base_eval_lora_train(model)
        batch = stack_batch(train_rows, sample_ids, args.device)
        timestep = sample_v5_timesteps(
            batch["targets"],
            diffusion.num_timesteps,
            args.high_timestep_start,
            args.device,
        )
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(batch["x"])
        prediction = predict_xstart(
            model,
            diffusion,
            batch["x"],
            timestep,
            {
                "c_pc_xyz": batch["xyz"],
                "c_pc_feat": batch["feat"],
                "c_text": batch["text"],
            },
            noise,
        )
        per_sample_data_loss = weighted_xstart_loss(
            prediction, batch["x"], batch["weights"]
        )
        data_loss = per_sample_data_loss.mean()
        uniform_data_loss = (prediction - batch["x"]).square().mean()
        semantic = semantic_contact_loss(
            prediction,
            batch["gt"],
            batch["instance_ids"],
            batch["target_instance_ids"],
            batch["targets"],
            mean_tensor,
            std_tensor,
            args.active_threshold,
            args.activation_temperature,
            args.foreground_bce_weight,
            args.dice_weight,
            args.competitor_suppression_weight,
            args.ranking_weight,
            args.ranking_margin,
        )
        sit_semantic = sit_multicandidate_loss(
            prediction,
            batch["instance_ids"],
            batch["targets"],
            mean_tensor,
            std_tensor,
            bed_any_min=args.sit_bed_any_min,
            bed_any_max=args.sit_bed_any_max,
            bed_pelvis_min=args.sit_bed_pelvis_min,
            bed_pelvis_max=args.sit_bed_pelvis_max,
            chair_bed_margin=args.sit_chair_bed_margin,
        )

        novel_indices = [
            index
            for index, target in enumerate(batch["targets"])
            if target in {"bed", "whiteboard"}
        ]
        if len(novel_indices) != 2:
            raise AssertionError("v5 batch must contain one Bed and one Whiteboard")
        novel_index = torch.tensor(novel_indices, dtype=torch.long, device=args.device)
        high_start = int(diffusion.num_timesteps * args.high_timestep_start)
        high_timestep = torch.randint(
            high_start,
            diffusion.num_timesteps,
            (len(novel_indices),),
            device=args.device,
        )
        high_noise = torch.randn_like(batch["x"][novel_index])
        high_prediction = predict_xstart(
            model,
            diffusion,
            batch["x"][novel_index],
            high_timestep,
            {
                "c_pc_xyz": batch["xyz"][novel_index].contiguous(),
                "c_pc_feat": batch["feat"][novel_index].contiguous(),
                "c_text": [batch["text"][index] for index in novel_indices],
            },
            high_noise,
        )
        high_data_loss = weighted_xstart_loss(
            high_prediction,
            batch["x"][novel_index],
            batch["weights"][novel_index],
        ).mean()
        high_semantic = semantic_contact_loss(
            high_prediction,
            batch["gt"][novel_index],
            batch["instance_ids"][novel_index],
            batch["target_instance_ids"][novel_index],
            [batch["targets"][index] for index in novel_indices],
            mean_tensor,
            std_tensor,
            args.active_threshold,
            args.activation_temperature,
            args.foreground_bce_weight,
            args.dice_weight,
            args.competitor_suppression_weight,
            args.ranking_weight,
            args.ranking_margin,
        )

        chair_indices = [
            index for index, target in enumerate(batch["targets"]) if target == "chair"
        ]
        chair_index = torch.tensor(chair_indices, dtype=torch.long, device=args.device)
        chair_target_points = (
            batch["instance_ids"][chair_index]
            == batch["target_instance_ids"][chair_index, None]
        )
        chair_high_timestep = torch.randint(
            high_start,
            diffusion.num_timesteps,
            (len(chair_indices),),
            device=args.device,
        )
        chair_high_noise = torch.randn_like(batch["x"][chair_index])
        chair_high_kwargs = {
            "c_pc_xyz": batch["xyz"][chair_index].contiguous(),
            "c_pc_feat": batch["feat"][chair_index].contiguous(),
            "c_text": [batch["text"][index] for index in chair_indices],
        }
        chair_high_prediction = predict_xstart(
            model,
            diffusion,
            batch["x"][chair_index],
            chair_high_timestep,
            chair_high_kwargs,
            chair_high_noise,
        )
        chair_high_data_loss = masked_point_xstart_loss(
            chair_high_prediction,
            batch["x"][chair_index],
            chair_target_points,
        ).mean()
        chair_high_sit_semantic = sit_multicandidate_loss(
            chair_high_prediction,
            batch["instance_ids"][chair_index],
            [batch["targets"][index] for index in chair_indices],
            mean_tensor,
            std_tensor,
            bed_any_min=args.sit_bed_any_min,
            bed_any_max=args.sit_bed_any_max,
            bed_pelvis_min=args.sit_bed_pelvis_min,
            bed_pelvis_max=args.sit_bed_pelvis_max,
            chair_bed_margin=args.sit_chair_bed_margin,
        )
        set_lora_enabled(model, False)
        try:
            with torch.no_grad():
                teacher_prediction = predict_xstart(
                    model,
                    diffusion,
                    batch["x"][chair_index],
                    timestep[chair_index],
                    {
                        "c_pc_xyz": batch["xyz"][chair_index].contiguous(),
                        "c_pc_feat": batch["feat"][chair_index].contiguous(),
                        "c_text": [batch["text"][index] for index in chair_indices],
                    },
                    noise[chair_index],
                )
                chair_high_teacher_prediction = predict_xstart(
                    model,
                    diffusion,
                    batch["x"][chair_index],
                    chair_high_timestep,
                    chair_high_kwargs,
                    chair_high_noise,
                )
        finally:
            set_lora_enabled(model, True)
        chair_teacher_loss = masked_point_xstart_loss(
            prediction[chair_index],
            teacher_prediction,
            chair_target_points,
        ).mean()
        chair_high_teacher_loss = masked_point_xstart_loss(
            chair_high_prediction,
            chair_high_teacher_prediction,
            chair_target_points,
        ).mean()
        adapter_l2 = lora_parameter_energy(model)
        loss = (
            data_loss
            + args.semantic_weight * semantic["total"]
            + args.sit_multicandidate_weight * sit_semantic["total"]
            + args.high_timestep_weight
            * (high_data_loss + args.semantic_weight * high_semantic["total"])
            + args.chair_teacher_weight * chair_teacher_loss
            + args.chair_high_timestep_weight
            * (
                chair_high_data_loss
                + args.sit_multicandidate_weight
                * chair_high_sit_semantic["total"]
                + args.chair_high_teacher_weight
                * chair_high_teacher_loss
            )
            + args.lora_l2_weight * adapter_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step {step}: non-finite loss")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip).item()
        )
        optimizer.step()
        logs.append(
            {
                "step": step,
                "sample_ids": sample_ids,
                "targets": [str(train_rows[v]["target"]) for v in sample_ids],
                "timesteps": [int(v) for v in timestep.detach().cpu().tolist()],
                "data_loss": float(data_loss.item()),
                "uniform_data_loss": float(uniform_data_loss.item()),
                "semantic_loss": float(semantic["total"].item()),
                "sit_multicandidate_loss": float(sit_semantic["total"].item()),
                "high_timestep_data_loss": float(high_data_loss.item()),
                "high_timestep_semantic_loss": float(high_semantic["total"].item()),
                "chair_teacher_loss": float(chair_teacher_loss.item()),
                "chair_high_timestep_data_loss": float(
                    chair_high_data_loss.item()
                ),
                "chair_high_timestep_sit_loss": float(
                    chair_high_sit_semantic["total"].item()
                ),
                "chair_high_timestep_teacher_loss": float(
                    chair_high_teacher_loss.item()
                ),
                "lora_parameter_energy": float(adapter_l2.item()),
                "total_loss": float(loss.item()),
                "gradient_norm": gradient_norm,
            }
        )
        if step == 1 or step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "data_loss": float(data_loss.item()),
                        "uniform_data_loss": float(uniform_data_loss.item()),
                        "semantic_loss": float(semantic["total"].item()),
                        "sit_multicandidate_loss": float(
                            sit_semantic["total"].item()
                        ),
                        "high_timestep_data_loss": float(high_data_loss.item()),
                        "high_timestep_semantic_loss": float(high_semantic["total"].item()),
                        "chair_teacher_loss": float(chair_teacher_loss.item()),
                        "chair_high_timestep_data_loss": float(
                            chair_high_data_loss.item()
                        ),
                        "chair_high_timestep_sit_loss": float(
                            chair_high_sit_semantic["total"].item()
                        ),
                        "chair_high_timestep_teacher_loss": float(
                            chair_high_teacher_loss.item()
                        ),
                        "lora_parameter_energy": float(adapter_l2.item()),
                        "total_loss": float(loss.item()),
                        "gradient_norm": gradient_norm,
                        "sample_ids": sample_ids,
                    }
                )
            )
        if step % args.eval_every == 0 or step == args.steps:
            grid = fixed_grid_loss(
                model,
                diffusion,
                train_rows,
                args.device,
                args.seed + 700000,
                mean,
                std,
                args.active_threshold,
            )
            loss_selection = checkpoint_selection_gate(
                initial_grid,
                grid,
                args.novel_min_relative_improvement,
                args.chair_max_relative_degradation,
            )
            semantic_selection = semantic_checkpoint_gate(
                grid["semantic"],
                args.novel_train_min_f1,
                args.novel_train_min_dominance_rate,
                args.chair_train_min_dominance_rate,
                args.sit_bed_candidate_min_rate,
            )
            selection = {
                "passed": (
                    loss_selection["passed"] and semantic_selection["passed"]
                ),
                "loss_gate": loss_selection,
                "semantic_gate": semantic_selection,
                "rank": semantic_selection["rank"] + loss_selection["rank"],
            }
            print(
                f"[TRAIN-GRID] step={step} mean={grid['mean']:.8f} "
                f"chair={grid['per_target_mean']['chair']:.8f} "
                f"bed={grid['per_target_mean']['bed']:.8f} "
                f"whiteboard={grid['per_target_mean']['whiteboard']:.8f} "
                f"bed_f1={grid['semantic']['per_target']['bed']['f1_at_0_7']:.4f} "
                f"whiteboard_f1="
                f"{grid['semantic']['per_target']['whiteboard']['f1_at_0_7']:.4f} "
                f"gate={selection['passed']} rank={selection['rank']}"
            )
            candidate_rank = tuple(float(value) for value in selection["rank"])
            # A scientific gate must control promotion, not whether learned
            # weights survive process exit. Retain the top train-only
            # diagnostic checkpoints independently of the strict candidate
            # list so a failed run remains inspectable and visualizable.
            diagnostic_file = diagnostic_dir / f"step_{step:07d}.pt"
            diagnostic_record = {
                "step": step,
                "checkpoint": str(diagnostic_file),
                "one_step_rank": [float(value) for value in candidate_rank],
                "grid": grid,
                "selection": selection,
                "scientific_status": (
                    "CANDIDATE" if selection["passed"] else "DIAGNOSTIC_ONLY"
                ),
            }
            diagnostic_records.append(diagnostic_record)
            diagnostic_records.sort(
                key=lambda row: tuple(float(v) for v in row["one_step_rank"]),
                reverse=True,
            )
            retained_diagnostics = diagnostic_records[
                : args.diagnostic_checkpoint_count
            ]
            retained_steps = {int(row["step"]) for row in retained_diagnostics}
            if step in retained_steps:
                save_trainable_state(model, diagnostic_file)
                diagnostic_record["checkpoint_sha256"] = sha256_file(
                    diagnostic_file
                )
                print(
                    f"[DIAGNOSTIC-CHECKPOINT] saved step={step} "
                    f"gate={selection['passed']} file={diagnostic_file}",
                    flush=True,
                )
            for discarded in diagnostic_records[
                args.diagnostic_checkpoint_count :
            ]:
                Path(str(discarded["checkpoint"])).unlink(missing_ok=True)
            diagnostic_records = retained_diagnostics
            if selection["passed"]:
                candidate_file = candidate_dir / f"step_{step:07d}.pt"
                save_trainable_state(model, candidate_file)
                candidate_records.append(
                    {
                        "step": step,
                        "checkpoint": str(candidate_file),
                        "checkpoint_sha256": sha256_file(candidate_file),
                        "one_step_rank": [float(value) for value in candidate_rank],
                        "grid": grid,
                        "selection": selection,
                    }
                )
                if best_rank is None or candidate_rank > best_rank:
                    best_grid = dict(grid)
                    best_selection = dict(selection)
                    best_rank = candidate_rank
                    best_step = step

    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row) + "\n")
    (output_dir / "one_step_candidates.json").write_text(
        json.dumps(
            {
                "schema": "history_affordance_v1_fewshot_cdm_one_step_candidates_v2",
                "partition": "train_only",
                "heldout_sample_tensors_read": False,
                "prompt_policy_id": PROMPT_POLICY_ID,
                "candidates": candidate_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostic_file = output_dir / "diagnostic_checkpoints.json"
    atomic_write_json(
        diagnostic_file,
        {
            "schema": "history_affordance_v1_fewshot_cdm_diagnostic_shortlist_v1",
            "scientific_scope": "train_only_diagnostic_not_promoted",
            "partition": "train_only",
            "heldout_sample_tensors_read": False,
            "prompt_policy_id": PROMPT_POLICY_ID,
            "retained_count": len(diagnostic_records),
            "records": diagnostic_records,
        },
    )
    if best_step == 0 or best_grid is None or best_selection is None:
        failure_file = output_dir / "shortlist_status.json"
        atomic_write_json(
            failure_file,
            {
                "schema": "history_affordance_v1_fewshot_cdm_v5r4_shortlist_status_v1",
                "status": "FAIL",
                "partition": "train_only",
                "heldout_sample_tensors_read": False,
                "prompt_policy_id": PROMPT_POLICY_ID,
                "completed_steps": args.steps,
                "candidate_count": len(candidate_records),
                "diagnostic_checkpoint_count": len(diagnostic_records),
                "diagnostic_checkpoints": str(diagnostic_file),
                "reason": (
                    "No train-only checkpoint passed weighted Bed/Whiteboard "
                    "loss, Chair replay retention, and semantic gates."
                ),
                "heldout_evaluation_forbidden": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            "[FAIL] No train-only checkpoint passed the v5 one-step gates; "
            f"see {failure_file}. Held-out evaluation is forbidden.",
            flush=True,
        )
        raise SystemExit(2)
    if args.stop_after_one_step:
        shortlist_file = output_dir / "shortlist_status.json"
        atomic_write_json(
            shortlist_file,
            {
                "schema": "history_affordance_v1_fewshot_cdm_v5r4_shortlist_status_v1",
                "status": "PASS",
                "partition": "train_only",
                "heldout_sample_tensors_read": False,
                "prompt_policy_id": PROMPT_POLICY_ID,
                "completed_steps": args.steps,
                "candidate_count": len(candidate_records),
                "diagnostic_checkpoint_count": len(diagnostic_records),
                "diagnostic_checkpoints": str(diagnostic_file),
                "best_one_step": {
                    "step": best_step,
                    "rank": list(best_rank),
                    "selection": best_selection,
                },
                "next_action": "rerun identical command with --rollout-only and without --stop-after-one-step",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print("[PASS] v5 train-only one-step shortlist completed; rollout not started")
        print(f"[OK] saved: {shortlist_file}")
        return
    rollout_selection = select_candidate_by_train_rollout(
        cfg,
        checkpoint,
        candidate_records,
        train_rows,
        mean,
        std,
        args,
        output_dir,
    )
    _finalize_v5_run(
        args=args,
        checkpoint=checkpoint,
        split_file=split_file,
        output_dir=output_dir,
        train_rows=train_rows,
        test_ids=test_ids,
        counts=counts,
        original_hash=original_hash,
        initial_grid=initial_grid,
        candidate_records=candidate_records,
        rollout_selection=rollout_selection,
        lora_info=lora_metadata(model),
        chair_uniform_parity=chair_uniform_parity,
    )


if __name__ == "__main__":
    main()

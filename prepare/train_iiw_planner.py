#!/usr/bin/env python3
"""Supervised overfit test for scene/text/state -> phase IIW prediction.

This is the first deployable-side test after the GT-IIW Oracle experiment.  It
trains only :class:`models.iiw_planner.IIWPlanner` against the native
point-aligned IIW targets.  Neither IIWAdapter nor CMDM is loaded here.  One or
more production scene indices may be combined; the result proves the planner
tensor/loss contract, not held-out generalization.

IIW is sparse, so a plain all-point MSE is not an adequate success criterion.
The objective combines soft-positive-weighted BCE, foreground/background
balanced L1, soft Dice overlap, and phase-delta supervision.  Evaluation also
reports thresholded overlap and a correct-order versus reversed-target margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.iiw_planner import IIWPlanner  # noqa: E402


NUM_POINTS = 8192
NUM_BODIES = 6
TEXT_DIM = 512
STATE_DIM = 4
ACTIVE_THRESHOLD = 0.70
EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_under(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(name + " contains NaN/Inf")


def gradient_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


class IIWPlannerDataset(Dataset):
    """Strict loader for one or more IIW-augmented production indices."""

    def __init__(self, dataset_root: Path, index_names: Sequence[str]):
        self.root = dataset_root.expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not index_names:
            raise ValueError("at least one --index is required")
        self.scenes: Dict[str, Dict[str, np.ndarray]] = {}
        self.samples: List[Dict] = []
        self.num_phases = None
        seen_ids = set()

        for index_name in index_names:
            index_file = resolve_under(self.root, index_name)
            if not index_file.is_file():
                raise FileNotFoundError(index_file)
            index = json.loads(index_file.read_text())
            if not bool(index.get("iiw_gt_ready", False)):
                raise ValueError(str(index_file) + ": IIW GT is not ready")
            if index.get("iiw_gt_method") != "point_aligned_iiw_proxy":
                raise ValueError(str(index_file) + ": unexpected IIW method")
            num_phases = int(index.get("iiw_num_phases", -1))
            if num_phases <= 1:
                raise ValueError(str(index_file) + ": invalid phase count")
            if self.num_phases is None:
                self.num_phases = num_phases
            elif self.num_phases != num_phases:
                raise ValueError("all indices must use the same phase count")

            scene_id = str(index["scene_id"])
            if scene_id in self.scenes:
                raise ValueError("duplicate scene index: " + scene_id)
            self.scenes[scene_id] = self._load_scene(index, scene_id)
            entries = index.get("samples", [])
            if len(entries) != int(index.get("num_samples", -1)):
                raise ValueError(str(index_file) + ": num_samples mismatch")
            for entry in entries:
                sample = self._load_sample(entry, scene_id, num_phases)
                if sample["sample_id"] in seen_ids:
                    raise ValueError("duplicate sample: " + sample["sample_id"])
                seen_ids.add(sample["sample_id"])
                self.samples.append(sample)

        if not self.samples or self.num_phases is None:
            raise ValueError("planner dataset contains no samples")
        states = np.stack([row["state"] for row in self.samples], axis=0)
        self.state_mean = states.mean(axis=0).astype(np.float32)
        self.state_std = states.std(axis=0).astype(np.float32)
        self.state_std = np.maximum(self.state_std, 1e-4).astype(np.float32)

        targets = np.stack([row["target"] for row in self.samples], axis=0)
        self.target_soft_mean = targets.mean(axis=(0, 1, 2)).astype(np.float32)
        if np.any(self.target_soft_mean <= 0.0):
            raise ValueError("one or more IIW body channels have zero target mass")
        self.target_active_fraction = float(
            np.mean(targets >= ACTIVE_THRESHOLD)
        )
        if not (0.0 < self.target_active_fraction < 0.5):
            raise ValueError("unexpected IIW target sparsity")

    def _load_scene(
        self, index: Dict, scene_id: str
    ) -> Dict[str, np.ndarray]:
        scene_dir = resolve_under(self.root, index["scene_adm_input"])
        points_file = scene_dir / "points.npz"
        sidecar_file = scene_dir / "sidecar.npz"
        for path in (points_file, sidecar_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(points_file, allow_pickle=False) as archive:
            points = archive["points"].astype(np.float32)
        with np.load(sidecar_file, allow_pickle=False) as archive:
            source_indices = archive["source_indices"].astype(np.int64)
            instance_ids = archive["instance_ids"].astype(np.int64)
        if points.shape != (NUM_POINTS, 6):
            raise ValueError(scene_id + ": scene point shape mismatch")
        if source_indices.shape != (NUM_POINTS,):
            raise ValueError(scene_id + ": source-index shape mismatch")
        if instance_ids.shape != (NUM_POINTS,):
            raise ValueError(scene_id + ": instance-ID shape mismatch")
        if not np.isfinite(points).all():
            raise ValueError(scene_id + ": scene points contain NaN/Inf")
        result = points.copy()
        result[:, 3:6] /= 255.0
        if np.any(result[:, 3:6] < 0.0) or np.any(result[:, 3:6] > 1.0):
            raise ValueError(scene_id + ": normalized RGB outside [0,1]")
        return {
            "points": result,
            "source_indices": source_indices,
            "instance_ids": instance_ids,
        }

    def _load_sample(
        self, entry: Dict, scene_id: str, num_phases: int
    ) -> Dict:
        sample_id = str(entry["sample_id"])
        if str(entry["scene_id"]) != scene_id:
            raise ValueError(sample_id + ": scene mismatch")
        if not bool(entry.get("iiw_gt_ready", False)):
            raise ValueError(sample_id + ": IIW target is not ready")
        target_file = resolve_under(self.root, entry["iiw_gt"])
        if not target_file.is_file():
            raise FileNotFoundError(target_file)
        with np.load(target_file, allow_pickle=False) as archive:
            target = archive["iiw_native_phase_max"].astype(np.float32)
            frame_to_phase = archive["frame_to_phase"].astype(np.int64)
            target_xyz = archive["scene_xyz_adm"].astype(np.float32)
            target_sources = archive["source_indices"].astype(np.int64)
            target_instances = archive["instance_ids"].astype(np.int64)
            phase_mask = archive["phase_mask"].astype(bool)
        expected = (num_phases, NUM_POINTS, NUM_BODIES)
        if target.shape != expected:
            raise ValueError(sample_id + ": target shape mismatch")
        if phase_mask.shape != (num_phases,) or bool(phase_mask.any()):
            raise ValueError(sample_id + ": all generated phases must be valid")
        scene = self.scenes[scene_id]
        if not np.array_equal(target_xyz, scene["points"][:, :3]):
            raise ValueError(sample_id + ": target/scene point order mismatch")
        if not np.array_equal(target_sources, scene["source_indices"]):
            raise ValueError(sample_id + ": target/source point order mismatch")
        if not np.array_equal(target_instances, scene["instance_ids"]):
            raise ValueError(sample_id + ": target/instance order mismatch")
        if not np.isfinite(target).all():
            raise ValueError(sample_id + ": target contains NaN/Inf")
        if np.any(target < 0.0) or np.any(target > 1.0):
            raise ValueError(sample_id + ": target outside [0,1]")
        if frame_to_phase.ndim != 1 or len(frame_to_phase) < num_phases:
            raise ValueError(sample_id + ": invalid frame-to-phase mapping")
        if np.any(np.diff(frame_to_phase) < 0):
            raise ValueError(sample_id + ": non-monotonic phase mapping")
        if set(frame_to_phase.tolist()) != set(range(num_phases)):
            raise ValueError(sample_id + ": phases do not cover valid frames")
        temporal_delta = float(np.mean(np.abs(np.diff(target, axis=0))))
        if temporal_delta <= 0.0:
            raise ValueError(sample_id + ": target has no temporal variation")

        start_xyz = np.asarray(
            entry["start_position_adm_chair_local_xyz"], dtype=np.float32
        )
        direction = np.asarray(
            entry["history_direction_adm_xy"], dtype=np.float32
        )
        if start_xyz.shape != (3,) or direction.shape != (2,):
            raise ValueError(sample_id + ": state source shape mismatch")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            raise ValueError(sample_id + ": direction is zero")
        direction = direction / norm
        state = np.concatenate([start_xyz[:2], direction]).astype(np.float32)
        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "text": str(entry["text"]),
            "state": state,
            "target": target,
            "temporal_delta": temporal_delta,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        return {
            "scene_points": self.scenes[sample["scene_id"]]["points"].copy(),
            # Keep metric chair-local start coordinates. IIWPlanner uses
            # scene_xy - start_xy internally, so z-scoring here would destroy
            # that geometric relation. Mean/std are saved as metadata only.
            "state": sample["state"].copy(),
            "target": sample["target"].copy(),
            "text": sample["text"],
            "sample_id": sample["sample_id"],
            "scene_id": sample["scene_id"],
        }


def sparse_objective(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if logits.shape != target.shape:
        raise ValueError("planner logits/target shape mismatch")
    assert_finite("planner logits", logits)
    assert_finite("IIW target", target)
    if bool(((target < 0.0) | (target > 1.0)).any().item()):
        raise ValueError("IIW target outside [0,1]")
    prediction = torch.sigmoid(logits)
    # ``pos_weight`` changes the optimum for continuous/soft BCE targets.
    # Element weighting retains the correct optimum p=target while increasing
    # the contribution of sparse high-IIW values.
    raw_bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    element_weight = 1.0 + target * (
        positive_weight.view(1, 1, 1, -1) - 1.0
    )
    bce = (raw_bce * element_weight).sum() / element_weight.sum()
    absolute = torch.abs(prediction - target)
    foreground = (absolute * target).sum() / target.sum().clamp_min(EPS)
    background_weight = 1.0 - target
    background = (absolute * background_weight).sum() / (
        background_weight.sum().clamp_min(EPS)
    )
    reduce_dims = (2,)
    intersection = (prediction * target).sum(dim=reduce_dims)
    denominator = prediction.sum(dim=reduce_dims) + target.sum(
        dim=reduce_dims
    )
    dice = 1.0 - ((2.0 * intersection + EPS) / (denominator + EPS)).mean()
    predicted_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    temporal = F.l1_loss(predicted_delta, target_delta)
    total = (
        weights["bce"] * bce
        + weights["foreground"] * foreground
        + weights["background"] * background
        + weights["dice"] * dice
        + weights["temporal"] * temporal
    )
    return total, {
        "total": total,
        "bce": bce,
        "foreground_mae": foreground,
        "background_mae": background,
        "dice_loss": dice,
        "temporal_l1": temporal,
    }


def tensor_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    assert_finite("prediction", prediction)
    if bool(((prediction < 0.0) | (prediction > 1.0)).any().item()):
        raise ValueError("planner probability outside [0,1]")
    absolute = torch.abs(prediction - target)
    active = target >= threshold
    inactive = ~active
    predicted_active = prediction >= threshold
    true_positive = torch.logical_and(predicted_active, active).sum().float()
    false_positive = torch.logical_and(predicted_active, inactive).sum().float()
    false_negative = torch.logical_and(~predicted_active, active).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(EPS)
    union = torch.logical_or(predicted_active, active).sum().float()
    iou = true_positive / union.clamp_min(1.0)
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    reversed_target = torch.flip(target, dims=(1,))
    correct_l1 = F.l1_loss(prediction, target)
    reversed_l1 = F.l1_loss(prediction, reversed_target)
    return {
        "mae": float(absolute.mean().item()),
        "active_mae": float(absolute[active].mean().item()),
        "inactive_mae": float(absolute[inactive].mean().item()),
        "precision_at_0_7": float(precision.item()),
        "recall_at_0_7": float(recall.item()),
        "f1_at_0_7": float(f1.item()),
        "iou_at_0_7": float(iou.item()),
        "predicted_active_fraction": float(predicted_active.float().mean().item()),
        "target_active_fraction": float(active.float().mean().item()),
        "temporal_delta_l1": float(F.l1_loss(pred_delta, target_delta).item()),
        "predicted_temporal_delta": float(pred_delta.abs().mean().item()),
        "target_temporal_delta": float(target_delta.abs().mean().item()),
        "correct_order_l1": float(correct_l1.item()),
        "reversed_target_l1": float(reversed_l1.item()),
        "order_margin": float((reversed_l1 - correct_l1).item()),
    }


def sample_retrieval_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scene_ids: Sequence[str] = None,
) -> Dict[str, float]:
    """Measure self-target retrieval only among point-aligned scene peers.

    IIW tensors from different scenes have unrelated point orderings.  They
    must never be compared by array index.  When ``scene_ids`` is omitted the
    legacy single-scene behavior is preserved exactly.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("retrieval tensors must be matching [B,Q,N,C]")
    batch_size = prediction.shape[0]
    if batch_size < 2:
        return {
            "sample_retrieval_accuracy": 1.0,
            "sample_retrieval_margin": 0.0,
        }
    if scene_ids is None:
        normalized_scene_ids = ["__single_scene__"] * batch_size
    else:
        normalized_scene_ids = [str(value) for value in scene_ids]
        if len(normalized_scene_ids) != batch_size:
            raise ValueError("scene_ids length does not match retrieval batch")

    correct = 0
    margins = []
    for scene_id in dict.fromkeys(normalized_scene_ids):
        indices = [
            index for index, value in enumerate(normalized_scene_ids)
            if value == scene_id
        ]
        group_prediction = prediction[indices]
        group_target = target[indices]
        group_size = len(indices)
        if group_size == 1:
            correct += 1
            margins.append(0.0)
            continue
        distances = torch.empty((group_size, group_size), dtype=torch.float32)
        for predicted_index in range(group_size):
            difference = torch.abs(
                group_prediction[predicted_index:predicted_index + 1]
                - group_target
            )
            distances[predicted_index] = difference.mean(dim=(1, 2, 3))
        selected = distances.argmin(dim=1)
        expected = torch.arange(group_size, dtype=torch.long)
        correct += int((selected == expected).sum().item())
        own = distances.diagonal()
        mask = torch.eye(group_size, dtype=torch.bool)
        nearest_other = distances.masked_fill(mask, float("inf")).min(dim=1)[0]
        margins.extend((nearest_other - own).tolist())
    return {
        "sample_retrieval_accuracy": float(correct) / float(batch_size),
        "sample_retrieval_margin": float(np.mean(margins)),
    }


def mean_within_scene_prediction_delta(
    prediction: torch.Tensor,
    scene_ids: Sequence[str],
) -> float:
    """Return mean pairwise output delta without crossing point orderings."""
    if prediction.ndim != 4 or prediction.shape[0] != len(scene_ids):
        raise ValueError("prediction/scene_ids contract mismatch")
    pairwise = []
    normalized = [str(value) for value in scene_ids]
    for first in range(len(prediction)):
        for second in range(first + 1, len(prediction)):
            if normalized[first] != normalized[second]:
                continue
            pairwise.append(float(
                torch.mean(torch.abs(prediction[first] - prediction[second])).item()
            ))
    if not pairwise:
        raise ValueError("state-dependence check requires a repeated scene")
    return float(np.mean(pairwise))


def make_diagnostic_plot(
    output_file: Path,
    records: List[Dict],
    target: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print("[CHECK] diagnostic plot skipped: " + str(error))
        return
    target_mass = target.mean(axis=1).T
    prediction_mass = prediction.mean(axis=1).T
    target_active = (target >= ACTIVE_THRESHOLD).mean(axis=1).T
    prediction_active = (prediction >= ACTIVE_THRESHOLD).mean(axis=1).T
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].plot([row["step"] for row in records], [row["total"] for row in records])
    axes[0, 0].set_title("Training objective")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("loss")
    for axis, value, title in (
        (axes[0, 1], target_mass, "GT mean IIW"),
        (axes[0, 2], prediction_mass, "Predicted mean IIW"),
        (axes[1, 1], target_active, "GT active ratio @0.7"),
        (axes[1, 2], prediction_active, "Predicted active ratio @0.7"),
    ):
        image = axis.imshow(value, aspect="auto", origin="lower", cmap="magma")
        axis.set_xlabel("phase")
        axis.set_ylabel("native body channel")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[1, 0].plot(
        np.mean(np.abs(np.diff(target, axis=0)), axis=(1, 2)),
        marker="o",
        label="GT",
    )
    axes[1, 0].plot(
        np.mean(np.abs(np.diff(prediction, axis=0)), axis=(1, 2)),
        marker="o",
        label="prediction",
    )
    axes[1, 0].set_title("Phase-to-phase change")
    axes[1, 0].set_xlabel("phase transition")
    axes[1, 0].legend()
    figure.suptitle("IIWPlanner overfit diagnostic: " + sample_id)
    figure.savefig(str(output_file), dpi=160)
    plt.close(figure)


def main() -> None:
    # AMDM owns the frozen CLIP helpers. Keep this import inside ``main`` so
    # the sparse objective/metrics have a lightweight CPU-only unit contract.
    from models.functions import (
        encode_text_clip,
        load_and_freeze_clip_model,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--point-dim", type=int, default=96)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "strict IIWPlanner checkpoint to continue when the first "
            "supervised run ends with status=CHECK"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if args.lr <= 0.0 or args.max_pos_weight < 1.0:
        raise ValueError("invalid optimizer/sparsity setting")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = IIWPlannerDataset(args.dataset_root, args.index)
    if args.batch_size > len(dataset):
        raise ValueError("batch-size exceeds dataset size")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=min(len(dataset), 6),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    clip_model = load_and_freeze_clip_model("ViT-B/32").to(device)
    clip_model.eval()
    if any(parameter.requires_grad for parameter in clip_model.parameters()):
        raise AssertionError("CLIP must remain frozen")
    unique_texts = sorted({row["text"] for row in dataset.samples})
    with torch.no_grad():
        encoded = encode_text_clip(
            clip_model,
            unique_texts,
            max_length=32,
            device=str(device),
        ).detach().float()
    if encoded.shape != (len(unique_texts), TEXT_DIM):
        raise ValueError("CLIP text feature shape mismatch")
    text_cache = {
        text: encoded[index].clone()
        for index, text in enumerate(unique_texts)
    }
    del encoded
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_config = {
        "scene_dim": 6,
        "text_dim": TEXT_DIM,
        "state_dim": STATE_DIM,
        "num_phases": int(dataset.num_phases),
        "num_bodies": NUM_BODIES,
        "hidden_dim": args.hidden_dim,
        "point_dim": args.point_dim,
        "context_dim": args.context_dim,
    }
    planner = IIWPlanner(**model_config).to(device)
    resumed_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_file = args.resume_checkpoint.expanduser().resolve()
        if not resume_file.is_file():
            raise FileNotFoundError(resume_file)
        saved = torch.load(str(resume_file), map_location="cpu")
        if not isinstance(saved, dict):
            raise TypeError("resume checkpoint must be a mapping")
        saved_config = saved.get("model_config")
        if saved_config != model_config:
            raise ValueError(
                "resume checkpoint architecture differs from current planner"
            )
        if saved.get("target_method") != "point_aligned_iiw_proxy":
            raise ValueError("resume checkpoint target method mismatch")
        if str(saved.get("text_encoder")) != "ViT-B/32":
            raise ValueError("resume checkpoint text encoder mismatch")
        saved_indices = {
            Path(str(value)).name for value in saved.get("index_files", [])
        }
        current_indices = {Path(str(value)).name for value in args.index}
        if saved_indices != current_indices:
            raise ValueError("resume checkpoint index set mismatch")
        state_dict = saved.get("model", saved.get("iiw_planner_state_dict"))
        if not isinstance(state_dict, dict) or not state_dict:
            raise TypeError("resume checkpoint has no planner state dict")
        planner.load_state_dict(state_dict, strict=True)
        resumed_checkpoint = str(resume_file)
        print("[PASS] resumed strict IIWPlanner checkpoint: " + resumed_checkpoint)
    trainable = [parameter for parameter in planner.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("IIWPlanner has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    positive_weight = (
        (1.0 - torch.from_numpy(dataset.target_soft_mean))
        / torch.from_numpy(dataset.target_soft_mean).clamp_min(1e-5)
    ).clamp(1.0, args.max_pos_weight).to(device)
    loss_weights = {
        "bce": args.bce_weight,
        "foreground": args.foreground_weight,
        "background": args.background_weight,
        "dice": args.dice_weight,
        "temporal": args.temporal_weight,
    }

    def prepare(
        raw: Dict,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
    ]:
        scene = raw["scene_points"].to(device, torch.float32).contiguous()
        state = raw["state"].to(device, torch.float32).contiguous()
        target = raw["target"].to(device, torch.float32).contiguous()
        texts = [str(value) for value in raw["text"]]
        text_features = torch.stack([text_cache[value] for value in texts], dim=0)
        batch_size = scene.shape[0]
        if scene.shape != (batch_size, NUM_POINTS, 6):
            raise ValueError("scene batch shape mismatch")
        if state.shape != (batch_size, STATE_DIM):
            raise ValueError("state batch shape mismatch")
        if target.shape != (
            batch_size, dataset.num_phases, NUM_POINTS, NUM_BODIES
        ):
            raise ValueError("target batch shape mismatch")
        for name, value in (
            ("scene", scene), ("state", state),
            ("text feature", text_features), ("target", target),
        ):
            assert_finite(name, value)
        return scene, text_features, state, target, texts

    @torch.no_grad()
    def evaluate() -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        planner.eval()
        prediction_rows = []
        target_rows = []
        sample_ids = []
        scene_ids = []
        objective_rows = []
        for raw in eval_loader:
            scene, text_features, state, target, _ = prepare(raw)
            logits = planner.forward_logits(scene, text_features, state)
            objective, parts = sparse_objective(
                logits, target, positive_weight, loss_weights
            )
            prediction = torch.sigmoid(logits)
            objective_rows.append((scene.shape[0], {
                key: float(value.item()) for key, value in parts.items()
            }))
            prediction_rows.append(prediction.cpu())
            target_rows.append(target.cpu())
            sample_ids.extend([str(value) for value in raw["sample_id"]])
            scene_ids.extend([str(value) for value in raw["scene_id"]])
        prediction = torch.cat(prediction_rows, dim=0)
        target = torch.cat(target_rows, dim=0)
        metrics = tensor_metrics(prediction, target, ACTIVE_THRESHOLD)
        metrics.update(sample_retrieval_metrics(prediction, target, scene_ids))
        total_rows = float(sum(count for count, _ in objective_rows))
        for key in objective_rows[0][1]:
            metrics[key] = float(sum(
                count * row[key] for count, row in objective_rows
            ) / total_rows)
        metrics["mean_pairwise_prediction_delta"] = (
            mean_within_scene_prediction_delta(prediction, scene_ids)
        )
        return metrics, {
            "prediction": prediction.numpy(),
            "target": target.numpy(),
            "sample_ids": np.asarray(sample_ids),
        }

    initial_metrics, _ = evaluate()
    planner.train()
    first_raw = next(iter(train_loader))
    scene, text_features, state, target, _ = prepare(first_raw)
    optimizer.zero_grad(set_to_none=True)
    first_logits = planner.forward_logits(scene, text_features, state)
    with torch.no_grad():
        first_probabilities = planner(scene, text_features, state)
        expected_probabilities = torch.sigmoid(first_logits.detach())
        if first_probabilities.shape != first_logits.shape:
            raise ValueError("IIWPlanner forward probability shape mismatch")
        if not torch.allclose(
            first_probabilities,
            expected_probabilities,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise AssertionError(
                "IIWPlanner forward must equal sigmoid(forward_logits)"
            )
        if bool(
            ((first_probabilities < 0.0) | (first_probabilities > 1.0))
            .any()
            .item()
        ):
            raise AssertionError("IIWPlanner forward returned invalid probability")
    print("[PASS] planner logits/probability API contract")
    first_loss, _ = sparse_objective(
        first_logits, target, positive_weight, loss_weights
    )
    first_loss.backward()
    initial_gradient_l2 = gradient_l2(trainable)
    if not np.isfinite(initial_gradient_l2) or initial_gradient_l2 <= 0.0:
        raise AssertionError("supervised IIW loss does not reach IIWPlanner")
    optimizer.zero_grad(set_to_none=True)
    print(
        "[PASS] supervised sparse IIW loss reaches planner: gradient_l2="
        + "{:.9g}".format(initial_gradient_l2)
    )

    records = []
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        scene, text_features, state, target, _ = prepare(raw)
        planner.train()
        optimizer.zero_grad(set_to_none=True)
        logits = planner.forward_logits(scene, text_features, state)
        loss, parts = sparse_objective(
            logits, target, positive_weight, loss_weights
        )
        loss.backward()
        grad_l2 = gradient_l2(trainable)
        if not np.isfinite(grad_l2) or grad_l2 <= 0.0:
            raise AssertionError("planner gradient is zero/non-finite")
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        record = {
            "step": step,
            "sample_ids": [str(value) for value in raw["sample_id"]],
            "gradient_l2": grad_l2,
        }
        record.update({key: float(value.detach().item()) for key, value in parts.items()})
        records.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(record))

    final_metrics, arrays = evaluate()
    if not final_metrics["total"] < initial_metrics["total"]:
        raise AssertionError(
            "planner objective did not improve: {:.8f} -> {:.8f}".format(
                initial_metrics["total"], final_metrics["total"]
            )
        )
    if not final_metrics["active_mae"] < initial_metrics["active_mae"]:
        raise AssertionError(
            "planner did not improve active-region MAE: {:.8f} -> {:.8f}".format(
                initial_metrics["active_mae"], final_metrics["active_mae"]
            )
        )
    if final_metrics["mean_pairwise_prediction_delta"] <= 1e-6:
        raise AssertionError("planner collapsed to one state-independent IIW")
    if final_metrics["predicted_temporal_delta"] <= 1e-7:
        raise AssertionError("planner collapsed to a static phase output")
    retrieval_scene_count = len({row["scene_id"] for row in dataset.samples})
    chance_accuracy = float(retrieval_scene_count) / float(len(dataset))
    overfit_quality_pass = bool(
        final_metrics["f1_at_0_7"] >= 0.50
        and final_metrics["order_margin"] > 0.0
        and final_metrics["sample_retrieval_accuracy"] > chance_accuracy
    )
    print(
        "[PASS] planner objective improved: {:.8f} -> {:.8f}".format(
            initial_metrics["total"], final_metrics["total"]
        )
    )
    print(
        "[PASS] planner output depends on state: mean_pairwise_delta="
        + "{:.8f}".format(final_metrics["mean_pairwise_prediction_delta"])
    )
    print(
        (
            "[PASS] state-conditioned target retrieval beats chance: "
            if final_metrics["sample_retrieval_accuracy"] > chance_accuracy
            else "[CHECK] state-conditioned target retrieval is weak: "
        )
        + "accuracy={:.6f}, chance={:.6f}".format(
            final_metrics["sample_retrieval_accuracy"], chance_accuracy
        )
    )
    print(
        "[PASS] planner preserves temporal variation: mean_abs_delta="
        + "{:.8f}".format(final_metrics["predicted_temporal_delta"])
    )
    print(
        (
            "[PASS] predicted IIW is closer to correct phase order: margin="
            if final_metrics["order_margin"] > 0.0
            else "[CHECK] correct/reversed phase ranking is weak: margin="
        )
        + "{:.8f}".format(final_metrics["order_margin"])
    )
    if overfit_quality_pass:
        print(
            "[PASS] sparse overfit quality: F1@0.7={:.6f}".format(
                final_metrics["f1_at_0_7"]
            )
        )
    else:
        print(
            "[CHECK] sparse overlap is below overfit target: F1@0.7="
            + "{:.6f} (target >= 0.50)".format(
                final_metrics["f1_at_0_7"]
            )
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS" if overfit_quality_pass else "CHECK",
        "scope": "multi-scene supervised IIWPlanner training contract",
        "held_out_generalization": False,
        "target_method": "point_aligned_iiw_proxy",
        "planner_inputs": ["scene_points", "CLIP_text_feature", "state"],
        "target_leakage_into_planner_inputs": False,
        "dataset_root": str(dataset.root),
        "index_files": list(args.index),
        "num_samples": len(dataset),
        "num_phases": dataset.num_phases,
        "num_points": NUM_POINTS,
        "num_bodies": NUM_BODIES,
        "active_threshold": ACTIVE_THRESHOLD,
        "target_active_fraction": dataset.target_active_fraction,
        "target_soft_mean_per_body": dataset.target_soft_mean.tolist(),
        "positive_weight_per_body": positive_weight.detach().cpu().tolist(),
        "state_mean": dataset.state_mean.tolist(),
        "state_std": dataset.state_std.tolist(),
        "model_config": model_config,
        "text_encoder": "ViT-B/32",
        "text_encoder_frozen": True,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "resumed_checkpoint": resumed_checkpoint,
        "loss_weights": loss_weights,
        "initial_gradient_l2": initial_gradient_l2,
        "initial": initial_metrics,
        "final": final_metrics,
        "overfit_quality_pass": overfit_quality_pass,
        "overfit_quality_contract": {
            "f1_at_0_7_min": 0.50,
            "correct_order_margin_positive": True,
            "sample_retrieval_above_chance": True,
        },
        "optimization_records": records,
    }
    summary_file = output_dir / "summary.json"
    checkpoint_file = output_dir / "iiw_planner.pt"
    result_file = output_dir / "iiw_planner_results.npz"
    csv_file = output_dir / "per_sample_metrics.csv"
    plot_file = output_dir / "iiw_planner_diagnostics.png"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "model": planner.state_dict(),
            "iiw_planner_state_dict": planner.state_dict(),
            "step": args.steps,
            "resumed_checkpoint": resumed_checkpoint,
            "index_files": list(args.index),
            "model_config": model_config,
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "text_encoder": "ViT-B/32",
            "target_method": "point_aligned_iiw_proxy",
            "num_phases": dataset.num_phases,
            "native_body_part_count": NUM_BODIES,
            "summary": summary,
        },
        checkpoint_file,
    )
    np.savez_compressed(
        result_file,
        prediction=arrays["prediction"].astype(np.float32),
        target=arrays["target"].astype(np.float32),
        sample_ids=arrays["sample_ids"],
    )
    with csv_file.open("w", newline="") as handle:
        fieldnames = [
            "sample_id", "mae", "active_mae", "inactive_mae",
            "precision_at_0_7", "recall_at_0_7", "f1_at_0_7",
            "iou_at_0_7", "order_margin", "predicted_temporal_delta",
            "target_temporal_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        prediction_tensor = torch.from_numpy(arrays["prediction"])
        target_tensor = torch.from_numpy(arrays["target"])
        for index, sample_id in enumerate(arrays["sample_ids"].tolist()):
            row_metrics = tensor_metrics(
                prediction_tensor[index:index + 1],
                target_tensor[index:index + 1],
                ACTIVE_THRESHOLD,
            )
            writer.writerow({
                "sample_id": str(sample_id),
                **{name: row_metrics[name] for name in fieldnames if name != "sample_id"},
            })
    make_diagnostic_plot(
        plot_file,
        records,
        arrays["target"][0],
        arrays["prediction"][0],
        str(arrays["sample_ids"][0]),
    )
    print(
        "[PASS] scene + text + state -> temporal IIW supervised overfit"
        if overfit_quality_pass
        else "[CHECK] tensor/training contract passed; continue sparse overfit"
    )
    for path in (summary_file, checkpoint_file, result_file, csv_file, plot_file):
        if path.is_file():
            print("[OK] saved: " + str(path))


if __name__ == "__main__":
    main()

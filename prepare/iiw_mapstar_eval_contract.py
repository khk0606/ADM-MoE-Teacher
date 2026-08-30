#!/usr/bin/env python3
"""Neutral data and metric contracts for literal IIW point-map routing.

This module deliberately depends only on the production IIW target loader,
the production motion loader, and the stock CMDM configuration.  It contains
no alternative CMDM-conditioning implementation.  The evaluator can therefore
prove that its only learned point-map consumer is ``c_pc_contact``.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from datasets.history_affordance_v1 import (
    HistoryAffordanceV1ContactMotionDataset,
)
from prepare.train_iiw_planner import (
    IIWPlannerDataset,
    NUM_BODIES,
    NUM_POINTS,
)
from utils.misc import compute_repr_dimesion


NUM_BODY_PARTS = NUM_BODIES
MAX_HORIZON = 196
MOTION_DIM = 66
DEFAULT_GRID = (0, 100, 250, 500, 750, 999)
BODY_NAMES = (
    "base",
    "spine",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
)


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


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("failed to freeze " + type(module).__name__)


def state_digest(module: torch.nn.Module) -> str:
    """Hash every parameter and persistent buffer without copying a model."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def joint_trainer_state_digest(module: torch.nn.Module) -> str:
    """Reproduce the digest stored by the v1 map* joint trainer.

    The frozen evaluator has its own state digest above.  The joint trainer
    predates that helper and serializes tensor shapes as int64 bytes, so this
    compatibility digest is required to bind evaluation to the exact CMDM
    state recorded in a joint checkpoint.
    """
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_coverage(
    module: torch.nn.Module,
    checkpoint_file: Path,
    require_trainable_complete: bool = True,
) -> Dict[str, object]:
    """Verify stock CMDM checkpoint coverage before freezing the model."""
    saved = torch.load(str(checkpoint_file), map_location="cpu")
    if not isinstance(saved, Mapping):
        raise TypeError("CMDM checkpoint must contain a state-dict mapping")
    current = module.state_dict()
    trainable_names = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    external_prefixes = ("text_model.", "clip_model.", "bert_model.")
    loaded_keys: List[str] = []
    missing_trainable: List[str] = []
    missing_frozen: List[str] = []
    mismatched_shapes: List[Dict[str, object]] = []
    for key, value in current.items():
        saved_key = key if key in saved else "module." + key
        if saved_key not in saved:
            if key in trainable_names and not key.startswith(external_prefixes):
                missing_trainable.append(key)
            else:
                missing_frozen.append(key)
            continue
        saved_value = saved[saved_key]
        if not torch.is_tensor(saved_value):
            mismatched_shapes.append(
                {"key": key, "reason": "checkpoint value is not a tensor"}
            )
            continue
        if tuple(saved_value.shape) != tuple(value.shape):
            mismatched_shapes.append(
                {
                    "key": key,
                    "model_shape": list(value.shape),
                    "checkpoint_shape": list(saved_value.shape),
                }
            )
            continue
        loaded_keys.append(key)
    if mismatched_shapes or (require_trainable_complete and missing_trainable):
        raise ValueError(
            "CMDM checkpoint coverage failed: missing_trainable={} "
            "shape_mismatches={}".format(
                len(missing_trainable), len(mismatched_shapes)
            )
        )
    return {
        "model_state_tensors": len(current),
        "loaded_shape_matched_tensors": len(loaded_keys),
        "missing_trainable_tensors": missing_trainable,
        "missing_frozen_tensors": missing_frozen,
        "shape_mismatches": mismatched_shapes,
    }


def load_cmdm_config(repo_root: Path):
    default_cfg = OmegaConf.load(repo_root / "configs/default.yaml")
    cfg = OmegaConf.create(
        {"seed": int(default_cfg.seed), "diffusion": default_cfg.diffusion}
    )
    cfg.task = OmegaConf.load(
        repo_root / "configs/task/contact_motion_gen.yaml"
    )
    cfg.model = OmegaConf.load(repo_root / "configs/model/cmdm.yaml")
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    OmegaConf.resolve(cfg)
    if str(cfg.model.data_repr) != "pos" or int(cfg.model.input_feats) != MOTION_DIM:
        raise ValueError("map* evaluation requires CMDM pos representation (66D)")
    if int(cfg.task.dataset.num_points) != NUM_POINTS:
        raise ValueError("map* evaluation requires exactly 8192 scene points")
    if int(cfg.task.dataset.max_horizon) != MAX_HORIZON:
        raise ValueError("map* evaluation requires max_horizon=196")
    if int(cfg.model.latent_dim) != 512:
        raise ValueError("map* evaluation requires CMDM latent_dim=512")
    return cfg


class IIWMapstarEvalDataset(Dataset):
    """Join production IIW targets and motion tensors by sample ID.

    Both source loaders independently verify point order, cached Base ADM,
    motion shape, history state, and IIW target shape.  This wrapper verifies
    that they describe exactly the same sample before exposing one batch.
    """

    def __init__(self, dataset_root: Path, index_names: Sequence[str]):
        self.root = dataset_root.expanduser().resolve()
        self.index_names = [str(value) for value in index_names]
        self.iiw = IIWPlannerDataset(self.root, self.index_names)
        motion_cfg = SimpleNamespace(
            dataset_root=str(self.root),
            index_files=self.index_names,
            num_points=NUM_POINTS,
            max_horizon=MAX_HORIZON,
        )
        self.motion = HistoryAffordanceV1ContactMotionDataset(
            motion_cfg, phase="test"
        )
        self.num_phases = int(self.iiw.num_phases)

        iiw_by_id = {
            str(self.iiw[index]["sample_id"]): index
            for index in range(len(self.iiw))
        }
        motion_by_id = {
            str(self.motion[index]["info_sample_id"]): index
            for index in range(len(self.motion))
        }
        if set(iiw_by_id) != set(motion_by_id):
            raise ValueError("IIW and motion indices contain different sample IDs")

        # Preserve the index/IIW loader order used by planner training.
        self.samples: List[Dict[str, object]] = []
        self._iiw_indices: List[int] = []
        self._motion_indices: List[int] = []
        for iiw_index in range(len(self.iiw)):
            iiw_row = self.iiw[iiw_index]
            sample_id = str(iiw_row["sample_id"])
            motion_index = motion_by_id[sample_id]
            motion_row = self.motion[motion_index]
            if str(iiw_row["scene_id"]) != str(motion_row["info_scene_id"]):
                raise ValueError(sample_id + ": IIW/motion scene mismatch")
            if str(iiw_row["text"]) != str(motion_row["c_text"]):
                raise ValueError(sample_id + ": IIW/motion text mismatch")
            if not np.array_equal(
                iiw_row["scene_points"], motion_row["c_scene_points"]
            ):
                raise ValueError(sample_id + ": IIW/motion point order mismatch")
            if not np.array_equal(
                iiw_row["state"], motion_row["c_history_state"]
            ):
                raise ValueError(sample_id + ": IIW/motion state mismatch")
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "scene_id": str(iiw_row["scene_id"]),
                    "text": str(iiw_row["text"]),
                }
            )
            self._iiw_indices.append(iiw_index)
            self._motion_indices.append(motion_index)

        scene_to_indices: Dict[str, List[int]] = {}
        for index, sample in enumerate(self.samples):
            scene_to_indices.setdefault(str(sample["scene_id"]), []).append(index)
        self.shuffle_peer_indices = list(range(len(self.samples)))
        for scene_id, indices in scene_to_indices.items():
            if len(indices) < 2:
                raise ValueError(
                    scene_id
                    + ": at least two samples are required for a same-scene control"
                )
            for offset, sample_index in enumerate(indices):
                self.shuffle_peer_indices[sample_index] = indices[
                    (offset + 1) % len(indices)
                ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        iiw_row = self.iiw[self._iiw_indices[index]]
        motion_row = self.motion[self._motion_indices[index]]
        peer = self.shuffle_peer_indices[index]
        peer_iiw = self.iiw[self._iiw_indices[peer]]
        if str(peer_iiw["scene_id"]) != str(iiw_row["scene_id"]):
            raise AssertionError("same-scene control crossed scene boundaries")
        return {
            "x": motion_row["x"].copy(),
            "x_mask": motion_row["x_mask"].copy(),
            "c_pc_xyz": motion_row["c_pc_xyz"].copy(),
            "c_pc_contact": motion_row["c_base_affordance"].copy(),
            "iiw_plan": iiw_row["target"].copy(),
            "iiw_shuffled": peer_iiw["target"].copy(),
            "c_text": str(iiw_row["text"]),
            "sample_id": str(iiw_row["sample_id"]),
            "shuffled_sample_id": str(peer_iiw["sample_id"]),
            "scene_points": iiw_row["scene_points"].copy(),
            "state": iiw_row["state"].copy(),
        }


def plan_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_threshold: float,
) -> Dict[str, object]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction/target plan shapes differ")
    assert_finite("predicted IIW", prediction)
    assert_finite("target IIW", target)
    if float(prediction.min().item()) < -1e-6 or float(
        prediction.max().item()
    ) > 1.0 + 1e-6:
        raise ValueError("planner probabilities lie outside [0,1]")

    absolute = (prediction - target).abs()
    squared = (prediction - target).square()
    gt_active = target >= active_threshold
    pred_active = prediction >= active_threshold
    true_positive = int((gt_active & pred_active).sum().item())
    false_positive = int(((~gt_active) & pred_active).sum().item())
    false_negative = int((gt_active & (~pred_active)).sum().item())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    pred_flat = prediction.reshape(-1).double()
    target_flat = target.reshape(-1).double()
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * target_centered.square().sum()
    )
    correlation = (
        float(
            (pred_centered * target_centered).sum().item()
            / denominator.item()
        )
        if float(denominator.item()) > 0.0
        else 0.0
    )
    phase_mae = absolute.mean(dim=(0, 2, 3)).detach().cpu().tolist()
    body_mae_values = absolute.mean(dim=(0, 1, 2)).detach().cpu().tolist()
    active_error = absolute.masked_select(gt_active)
    inactive_error = absolute.masked_select(~gt_active)
    return {
        "mae": float(absolute.mean().item()),
        "mse": float(squared.mean().item()),
        "correlation": correlation,
        "prediction_mean": float(prediction.mean().item()),
        "target_mean": float(target.mean().item()),
        "prediction_max": float(prediction.max().item()),
        "target_max": float(target.max().item()),
        "active_threshold": float(active_threshold),
        "active_fraction_gt": float(gt_active.float().mean().item()),
        "active_fraction_predicted": float(pred_active.float().mean().item()),
        "active_mae": (
            float(active_error.mean().item()) if active_error.numel() else 0.0
        ),
        "inactive_mae": (
            float(inactive_error.mean().item()) if inactive_error.numel() else 0.0
        ),
        "active_true_positive": true_positive,
        "active_false_positive": false_positive,
        "active_false_negative": false_negative,
        "active_precision": float(precision),
        "active_recall": float(recall),
        "active_f1": float(f1),
        "temporal_delta_predicted": float(
            (prediction[:, 1:] - prediction[:, :-1]).abs().mean().item()
        ),
        "temporal_delta_target": float(
            (target[:, 1:] - target[:, :-1]).abs().mean().item()
        ),
        "phase_mae": [float(value) for value in phase_mae],
        "body_mae": {
            name: float(body_mae_values[body_index])
            for body_index, name in enumerate(BODY_NAMES)
        },
    }


def aggregate_plan_metrics(rows: Sequence[Dict[str, object]]) -> Dict:
    if not rows:
        raise ValueError("cannot aggregate an empty metric set")
    scalar_keys = (
        "mae",
        "mse",
        "correlation",
        "prediction_mean",
        "target_mean",
        "prediction_max",
        "target_max",
        "active_fraction_gt",
        "active_fraction_predicted",
        "active_mae",
        "inactive_mae",
        "active_f1",
        "temporal_delta_predicted",
        "temporal_delta_target",
    )
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in scalar_keys
    }
    true_positive = sum(int(row["active_true_positive"]) for row in rows)
    false_positive = sum(int(row["active_false_positive"]) for row in rows)
    false_negative = sum(int(row["active_false_negative"]) for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    result.update(
        {
            "active_threshold": float(rows[0]["active_threshold"]),
            "active_true_positive": true_positive,
            "active_false_positive": false_positive,
            "active_false_negative": false_negative,
            "micro_active_precision": float(precision),
            "micro_active_recall": float(recall),
            "micro_active_f1": float(
                2.0 * precision * recall / max(precision + recall, 1e-12)
            ),
            "phase_mae": np.asarray(
                [row["phase_mae"] for row in rows], dtype=np.float64
            ).mean(axis=0).tolist(),
            "body_mae": {
                name: float(
                    np.mean([float(row["body_mae"][name]) for row in rows])
                )
                for name in BODY_NAMES
            },
        }
    )
    return result


def validate_planner_output(
    prediction: torch.Tensor,
    batch_size: int,
    num_phases: int,
) -> None:
    expected = (batch_size, num_phases, NUM_POINTS, NUM_BODY_PARTS)
    if tuple(prediction.shape) != expected:
        raise ValueError(
            "planner output shape must be {}, got {}".format(
                expected, tuple(prediction.shape)
            )
        )
    if not torch.is_floating_point(prediction):
        raise TypeError("planner probabilities must be floating point")
    assert_finite("planner probabilities", prediction)
    if float(prediction.min().item()) < -1e-6 or float(
        prediction.max().item()
    ) > 1.0 + 1e-6:
        raise ValueError("planner probabilities lie outside [0,1]")

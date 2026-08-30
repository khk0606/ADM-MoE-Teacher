#!/usr/bin/env python3
"""Evaluate a learned IIW planner and its frozen downstream motion path.

The planner sees only the deployable conditioning tuple

``scene points + CLIP text feature + [start_x, start_y, dir_x, dir_y]``

and predicts a point-aligned phase plan ``[B,Q,N,6]``.  This script compares
that prediction with the motion-derived oracle target, then passes both plans
through the *frozen* Oracle IIW adapter and the *frozen* official CMDM.

All downstream motion losses use the same target motion, diffusion timestep,
and Gaussian noise for these modes:

* ``predicted``: planner output;
* ``oracle``: motion-derived GT IIW;
* ``reversed``: phase-reversed GT IIW;
* ``static_max``: phasewise maximum with temporal schedule removed;
* ``sample_shuffled``: another motion's GT IIW from the same scene;
* ``predicted_shuffled``: another state-conditioned prediction in that scene;
* ``zero``: an all-zero plan;
* ``legacy``: the original CMDM call with no IIW residual.

This is a conditional denoising evaluation, not yet free-running motion
generation: the GT motion mask and deterministic phase schedule are used to
score every mode fairly.  The planner itself never receives GT motion or GT
IIW.  Neither the planner, adapter, nor CMDM is optimized by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models  # noqa: E402,F401
from models.base import create_model_and_diffusion  # noqa: E402
from models.functions import encode_text_clip  # noqa: E402
from models.iiw_adapter import IIWAdapter  # noqa: E402
from models.iiw_cmdm import IIWConditionedCMDM  # noqa: E402
from prepare.test_iiw_oracle import (  # noqa: E402
    DEFAULT_GRID,
    MAX_HORIZON,
    NUM_BODY_PARTS,
    NUM_POINTS,
    IIWOracleDataset,
    checkpoint_coverage,
    load_cmdm_config,
    resolve_under,
    set_seed,
    state_digest,
)
from utils.training import load_ckpt  # noqa: E402


BODY_NAMES = (
    "base",
    "spine",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
)


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(name + " contains NaN/Inf")


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise AssertionError("failed to freeze " + type(module).__name__)


def extract_state_dict(checkpoint: Mapping, primary_key: str) -> Mapping:
    """Read one strict model state dict with one documented fallback."""
    if primary_key in checkpoint:
        state = checkpoint[primary_key]
    elif primary_key == "model" and "iiw_planner_state_dict" in checkpoint:
        # Compatibility with the first planner-overfit checkpoint revision.
        state = checkpoint["iiw_planner_state_dict"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        raise KeyError(
            "checkpoint has none of {!r}, 'iiw_planner_state_dict', or "
            "'model_state_dict'".format(primary_key)
        )
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint model state must be a non-empty mapping")
    # DDP checkpoints are accepted only when every key shares the prefix.
    keys = [str(key) for key in state]
    if keys and all(key.startswith("module.") for key in keys):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


class PlannerEvalDataset(Dataset):
    """Augment the strict Oracle loader with planner scene/state inputs."""

    def __init__(self, dataset_root: Path, index_names: Sequence[str]):
        self.root = dataset_root.expanduser().resolve()
        self.oracle = IIWOracleDataset(self.root, index_names)
        entry_by_id: Dict[str, Dict] = {}
        for name in index_names:
            index_file = resolve_under(self.root, name)
            index = json.loads(index_file.read_text())
            for entry in index["samples"]:
                sample_id = str(entry["sample_id"])
                if sample_id in entry_by_id:
                    raise ValueError("duplicate planner sample: " + sample_id)
                entry_by_id[sample_id] = entry

        self.states: List[np.ndarray] = []
        for sample in self.oracle.samples:
            sample_id = sample["sample_id"]
            if sample_id not in entry_by_id:
                raise KeyError(sample_id + ": missing source index entry")
            entry = entry_by_id[sample_id]
            start = np.asarray(
                entry["start_position_adm_chair_local_xyz"],
                dtype=np.float32,
            )
            direction = np.asarray(
                entry["history_direction_adm_xy"], dtype=np.float32
            )
            if start.shape != (3,) or direction.shape != (2,):
                raise ValueError(sample_id + ": state source shape mismatch")
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm < 1e-8:
                raise ValueError(sample_id + ": direction is invalid")
            direction = direction / norm
            state = np.concatenate((start[:2], direction)).astype(np.float32)
            if state.shape != (4,) or not np.isfinite(state).all():
                raise ValueError(sample_id + ": planner state is invalid")
            self.states.append(state)

    @property
    def num_phases(self) -> int:
        return self.oracle.num_phases

    def __len__(self) -> int:
        return len(self.oracle)

    def __getitem__(self, index: int) -> Dict:
        result = dict(self.oracle[index])
        sample = self.oracle.samples[index]
        scene = self.oracle.scenes[sample["scene_id"]]
        result["scene_points"] = scene["scene_points"].copy()
        result["state"] = self.states[index].copy()
        return result


def plan_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_threshold: float,
) -> Dict[str, object]:
    """Return sparsity-aware point/phase/body metrics for one batch item."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction/target plan shapes differ")
    assert_finite("predicted IIW", prediction)
    assert_finite("target IIW", target)
    if float(prediction.min().item()) < -1e-6 or float(
        prediction.max().item()
    ) > 1.0 + 1e-6:
        raise ValueError("planner probabilities lie outside [0,1]")

    error = prediction - target
    absolute = error.abs()
    squared = error.square()
    gt_active = target >= active_threshold
    pred_active = prediction >= active_threshold
    true_positive = int((gt_active & pred_active).sum().item())
    false_positive = int(((~gt_active) & pred_active).sum().item())
    false_negative = int((gt_active & (~pred_active)).sum().item())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    pred_flat = prediction.reshape(-1).double()
    gt_flat = target.reshape(-1).double()
    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * gt_centered.square().sum()
    )
    correlation = (
        float((pred_centered * gt_centered).sum().item() / denominator.item())
        if float(denominator.item()) > 0.0
        else 0.0
    )

    phase_mae = absolute.mean(dim=(0, 2, 3)).detach().cpu().tolist()
    body_mae_values = absolute.mean(dim=(0, 1, 2)).detach().cpu().tolist()
    body_mae = {
        name: float(body_mae_values[index])
        for index, name in enumerate(BODY_NAMES)
    }
    active_error = absolute.masked_select(gt_active)
    inactive_error = absolute.masked_select(~gt_active)
    active_mae = (
        float(active_error.mean().item()) if active_error.numel() else 0.0
    )
    inactive_mae = (
        float(inactive_error.mean().item()) if inactive_error.numel() else 0.0
    )
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
        "active_fraction_predicted": float(
            pred_active.float().mean().item()
        ),
        "active_mae": active_mae,
        "inactive_mae": inactive_mae,
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
        "body_mae": body_mae,
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
    tp = sum(int(row["active_true_positive"]) for row in rows)
    fp = sum(int(row["active_false_positive"]) for row in rows)
    fn = sum(int(row["active_false_negative"]) for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result.update(
        {
            "active_threshold": float(rows[0]["active_threshold"]),
            "active_true_positive": tp,
            "active_false_positive": fp,
            "active_false_negative": fn,
            "micro_active_precision": float(precision),
            "micro_active_recall": float(recall),
            "micro_active_f1": float(
                2.0 * precision * recall
                / max(precision + recall, 1e-12)
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


def build_adapter(checkpoint: Mapping, device: torch.device) -> IIWAdapter:
    if "iiw_adapter_state_dict" not in checkpoint:
        raise KeyError("adapter checkpoint has no iiw_adapter_state_dict")
    architecture = checkpoint.get("adapter_architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("adapter checkpoint has no architecture mapping")
    accepted = (
        "latent_dim",
        "num_phases",
        "num_body_parts",
        "body_hidden_dim",
        "hidden_dim",
        "embedding_dim",
    )
    kwargs = {key: int(architecture[key]) for key in accepted}
    kwargs["zero_init"] = False
    adapter = IIWAdapter(**kwargs).to(device)
    adapter.load_state_dict(checkpoint["iiw_adapter_state_dict"], strict=True)
    freeze(adapter)
    return adapter


def validate_planner_output(
    prediction: torch.Tensor,
    batch_size: int,
    num_phases: int,
) -> None:
    expected = (batch_size, num_phases, NUM_POINTS, NUM_BODY_PARTS)
    if prediction.shape != expected:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--cmdm-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--allow-partial-cmdm-checkpoint", action="store_true"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--eval-timesteps", type=int, nargs="+", default=list(DEFAULT_GRID)
    )
    parser.add_argument("--active-threshold", type=float, default=0.7)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--require-predicted-beats-zero", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError(
            "batch-size must be 1 for exact per-sample fixed-noise accounting"
        )
    if not 0.0 < args.active_threshold < 1.0:
        raise ValueError("active-threshold must lie strictly inside (0,1)")
    if not args.eval_timesteps or any(
        value < 0 or value >= 1000 for value in args.eval_timesteps
    ):
        raise ValueError("eval-timesteps must lie in [0,999]")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset_root = args.dataset_root.expanduser().resolve()
    planner_file = args.planner_checkpoint.expanduser().resolve()
    adapter_file = args.adapter_checkpoint.expanduser().resolve()
    for path in (planner_file, adapter_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = PlannerEvalDataset(dataset_root, args.index)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    planner_checkpoint = torch.load(str(planner_file), map_location="cpu")
    if not isinstance(planner_checkpoint, Mapping):
        raise TypeError("planner checkpoint must be a mapping")
    planner_summary = planner_checkpoint.get("summary")
    if not isinstance(planner_summary, Mapping):
        raise TypeError("planner checkpoint has no supervised summary mapping")
    if str(planner_summary.get("status")) != "PASS" or not bool(
        planner_summary.get("overfit_quality_pass", False)
    ):
        raise ValueError(
            "planner checkpoint did not pass the supervised sparse-IIW gate; "
            "continue planner training before downstream evaluation"
        )
    if planner_checkpoint.get("target_method") != "point_aligned_iiw_proxy":
        raise ValueError("planner checkpoint uses an unexpected IIW target method")
    if str(planner_checkpoint.get("text_encoder")) != "ViT-B/32":
        raise ValueError("planner checkpoint was not trained with CLIP ViT-B/32")
    trained_indices = planner_checkpoint.get("index_files")
    if not isinstance(trained_indices, (list, tuple)):
        raise TypeError("planner checkpoint has no index_files sequence")
    trained_index_names = {Path(str(value)).name for value in trained_indices}
    evaluated_index_names = {Path(str(value)).name for value in args.index}
    if trained_index_names != evaluated_index_names:
        raise ValueError(
            "planner/evaluation index mismatch: trained={} evaluated={}".format(
                sorted(trained_index_names), sorted(evaluated_index_names)
            )
        )
    model_config = planner_checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise TypeError("planner checkpoint has no model_config mapping")
    from models.iiw_planner import IIWPlanner  # noqa: E402

    planner = IIWPlanner(**dict(model_config)).to(device)
    planner.load_state_dict(
        extract_state_dict(planner_checkpoint, "model"), strict=True
    )
    freeze(planner)
    planner_digest_before = state_digest(planner)
    if int(getattr(planner, "num_phases", dataset.num_phases)) != int(
        dataset.num_phases
    ):
        raise ValueError("planner/dataset phase count mismatch")
    if int(getattr(planner, "state_dim", 4)) != 4:
        raise ValueError("planner state_dim must be 4")

    adapter_checkpoint = torch.load(str(adapter_file), map_location="cpu")
    if not isinstance(adapter_checkpoint, Mapping):
        raise TypeError("adapter checkpoint must be a mapping")
    adapter_summary = adapter_checkpoint.get("summary")
    if not isinstance(adapter_summary, Mapping):
        raise TypeError("adapter checkpoint has no Oracle summary mapping")
    if str(adapter_summary.get("status")) != "PASS":
        raise ValueError("Oracle adapter checkpoint did not pass its training gate")
    if not bool(adapter_summary.get("trained_zero_plan_exact_zero_residual")):
        raise ValueError("Oracle adapter lacks the exact null-plan guarantee")
    if not bool(adapter_summary.get("cmdm_state_unchanged")):
        raise ValueError("Oracle adapter checkpoint does not certify frozen CMDM")
    adapter = build_adapter(adapter_checkpoint, device)
    adapter_digest_before = state_digest(adapter)
    if adapter.num_phases != dataset.num_phases:
        raise ValueError("adapter/dataset phase count mismatch")
    adapter_indices = adapter_summary.get("index_files")
    if not isinstance(adapter_indices, (list, tuple)):
        raise TypeError("Oracle adapter checkpoint has no index_files sequence")
    adapter_index_names = {Path(str(value)).name for value in adapter_indices}
    if adapter_index_names != evaluated_index_names:
        raise ValueError(
            "Oracle adapter/evaluation index mismatch: trained={} evaluated={}".format(
                sorted(adapter_index_names), sorted(evaluated_index_names)
            )
        )

    cmdm_file_value = args.cmdm_checkpoint
    if cmdm_file_value is None:
        source = adapter_checkpoint.get("source_cmdm_checkpoint")
        if not source:
            parser.error(
                "--cmdm-checkpoint is required because the adapter "
                "checkpoint has no source_cmdm_checkpoint"
            )
        cmdm_file_value = Path(str(source))
    cmdm_file = cmdm_file_value.expanduser().resolve()
    if not cmdm_file.is_file():
        raise FileNotFoundError(cmdm_file)

    cfg = load_cmdm_config(REPO_ROOT)
    cmdm_backbone, diffusion = create_model_and_diffusion(
        cfg, device=str(device)
    )
    cmdm_backbone.to(device)
    coverage = checkpoint_coverage(
        cmdm_backbone,
        cmdm_file,
        require_trainable_complete=(not args.allow_partial_cmdm_checkpoint),
    )
    adapter_cmdm_source = adapter_checkpoint.get("source_cmdm_checkpoint")
    if adapter_cmdm_source:
        adapter_cmdm_name = Path(str(adapter_cmdm_source)).expanduser().name
        if adapter_cmdm_name != cmdm_file.name:
            raise ValueError(
                "Oracle adapter/CMDM checkpoint filename mismatch: trained={} "
                "evaluated={}".format(adapter_cmdm_name, cmdm_file.name)
            )
    load_ckpt(cmdm_backbone, str(cmdm_file))
    freeze(cmdm_backbone)
    cmdm_digest_before = state_digest(cmdm_backbone)
    expected_cmdm_digest = adapter_checkpoint.get("cmdm_state_digest")
    if expected_cmdm_digest and str(expected_cmdm_digest) != cmdm_digest_before:
        raise ValueError(
            "CMDM state differs from the state used to train the Oracle adapter"
        )
    cmdm = IIWConditionedCMDM(cmdm_backbone).to(device)
    cmdm.eval()
    if any(
        timestep >= int(diffusion.num_timesteps)
        for timestep in args.eval_timesteps
    ):
        raise ValueError("eval timestep exceeds diffusion schedule")

    unique_texts = sorted(
        {str(sample["text"]) for sample in dataset.oracle.samples}
    )
    with torch.no_grad():
        encoded = encode_text_clip(
            cmdm_backbone.text_model,
            unique_texts,
            max_length=cmdm_backbone.text_max_length,
            device=str(device),
        ).float()
    if encoded.ndim != 2 or encoded.shape[1] != 512:
        raise ValueError(
            "IIWPlanner requires 512D CLIP features, got {}".format(
                tuple(encoded.shape)
            )
        )
    text_cache = {
        text: encoded[index].detach()
        for index, text in enumerate(unique_texts)
    }

    def prepare_batch(raw: Dict) -> Dict:
        result = {
            "x": raw["x"].to(device, torch.float32).contiguous(),
            "x_mask": raw["x_mask"].to(device, torch.bool).contiguous(),
            "scene_points": raw["scene_points"].to(
                device, torch.float32
            ).contiguous(),
            "scene_xyz": raw["c_pc_xyz"].to(
                device, torch.float32
            ).contiguous(),
            "base_contact": raw["c_pc_contact"].to(
                device, torch.float32
            ).contiguous(),
            "gt_plan": raw["iiw_plan"].to(
                device, torch.float32
            ).contiguous(),
            "shuffled_plan": raw["iiw_shuffled"].to(
                device, torch.float32
            ).contiguous(),
            "frame_to_phase": raw["frame_to_phase"].to(
                device, torch.long
            ).contiguous(),
            "state": raw["state"].to(device, torch.float32).contiguous(),
            "texts": [str(value) for value in raw["c_text"]],
            "sample_ids": [str(value) for value in raw["sample_id"]],
            "shuffled_ids": [
                str(value) for value in raw["shuffled_sample_id"]
            ],
        }
        batch_size = int(result["x"].shape[0])
        expected_plan = (
            batch_size,
            dataset.num_phases,
            NUM_POINTS,
            NUM_BODY_PARTS,
        )
        expected_shapes = {
            "x": (batch_size, MAX_HORIZON, 66),
            "x_mask": (batch_size, MAX_HORIZON),
            "scene_points": (batch_size, NUM_POINTS, 6),
            "scene_xyz": (batch_size, NUM_POINTS, 3),
            "base_contact": (batch_size, NUM_POINTS, NUM_BODY_PARTS),
            "gt_plan": expected_plan,
            "shuffled_plan": expected_plan,
            "frame_to_phase": (batch_size, MAX_HORIZON),
            "state": (batch_size, 4),
        }
        for name, shape in expected_shapes.items():
            if tuple(result[name].shape) != shape:
                raise ValueError(
                    "{} shape must be {}, got {}".format(
                        name, shape, tuple(result[name].shape)
                    )
                )
        for name in (
            "x",
            "scene_points",
            "scene_xyz",
            "base_contact",
            "gt_plan",
            "state",
        ):
            assert_finite(name, result[name])
        result["text_features"] = torch.stack(
            [text_cache[text] for text in result["texts"]], dim=0
        ).contiguous()
        return result

    predicted_by_id: Dict[str, torch.Tensor] = {}
    plan_rows: List[Dict[str, object]] = []
    saved_predictions: List[np.ndarray] = []
    saved_targets: List[np.ndarray] = []
    saved_ids: List[str] = []
    with torch.no_grad():
        for raw in loader:
            batch = prepare_batch(raw)
            prediction = planner(
                batch["scene_points"],
                batch["text_features"],
                batch["state"],
            )
            validate_planner_output(
                prediction, int(batch["x"].shape[0]), dataset.num_phases
            )
            for index, sample_id in enumerate(batch["sample_ids"]):
                single_prediction = prediction[index : index + 1]
                single_target = batch["gt_plan"][index : index + 1]
                metrics = plan_metrics(
                    single_prediction,
                    single_target,
                    args.active_threshold,
                )
                metrics["sample_id"] = sample_id
                plan_rows.append(metrics)
                predicted_by_id[sample_id] = (
                    single_prediction[0].detach().cpu().contiguous()
                )
                if args.save_predictions:
                    saved_predictions.append(
                        single_prediction[0].detach().cpu().numpy()
                    )
                    saved_targets.append(
                        single_target[0].detach().cpu().numpy()
                    )
                    saved_ids.append(sample_id)
    if len(predicted_by_id) != len(dataset):
        raise AssertionError("planner prediction cache is incomplete")
    plan_aggregate = aggregate_plan_metrics(plan_rows)
    peer_deltas = []
    for sample_index, peer_index in enumerate(
        dataset.oracle.shuffle_peer_indices
    ):
        sample_id = dataset.oracle.samples[sample_index]["sample_id"]
        peer_id = dataset.oracle.samples[peer_index]["sample_id"]
        peer_deltas.append(
            float(
                (
                    predicted_by_id[sample_id]
                    - predicted_by_id[peer_id]
                ).abs().mean().item()
            )
        )
    plan_aggregate["same_scene_peer_prediction_mae"] = float(
        np.mean(peer_deltas)
    )
    print(
        "[PASS] predicted point-aligned IIW for {} samples".format(
            len(predicted_by_id)
        )
    )
    print(
        "[OK] IIW MAE={:.8f} correlation={:.6f} active_F1={:.6f}".format(
            plan_aggregate["mae"],
            plan_aggregate["correlation"],
            plan_aggregate["micro_active_f1"],
        )
    )

    def plan_for_mode(batch: Dict, mode: str) -> torch.Tensor:
        if mode == "predicted":
            return torch.stack(
                [predicted_by_id[value] for value in batch["sample_ids"]],
                dim=0,
            ).to(device=device, dtype=torch.float32).contiguous()
        if mode == "oracle":
            return batch["gt_plan"]
        if mode == "reversed":
            return batch["gt_plan"].flip(1)
        if mode == "static_max":
            return batch["gt_plan"].max(dim=1, keepdim=True)[0].expand_as(
                batch["gt_plan"]
            )
        if mode == "sample_shuffled":
            return batch["shuffled_plan"]
        if mode == "predicted_shuffled":
            return torch.stack(
                [predicted_by_id[value] for value in batch["shuffled_ids"]],
                dim=0,
            ).to(device=device, dtype=torch.float32).contiguous()
        if mode == "zero":
            return torch.zeros_like(batch["gt_plan"])
        raise ValueError("unknown plan mode: " + mode)

    def residual_for(batch: Dict, mode: str) -> torch.Tensor:
        frame_to_phase = batch["frame_to_phase"]
        if mode == "static_max":
            # Remove phase timing as well as phase order. Otherwise the
            # adapter's learned phase embedding would leak a temporal signal
            # into this nominally static control.
            frame_to_phase = torch.where(
                batch["x_mask"],
                torch.full_like(frame_to_phase, -1),
                torch.zeros_like(frame_to_phase),
            )
        residual = adapter(
            batch["scene_xyz"],
            plan_for_mode(batch, mode),
            frame_to_phase,
            batch["x_mask"],
        )
        expected = (
            batch["x"].shape[0],
            MAX_HORIZON,
            int(cfg.model.latent_dim),
        )
        if tuple(residual.shape) != expected:
            raise ValueError("adapter residual shape mismatch")
        assert_finite("adapter residual", residual)
        return residual

    def motion_loss(
        batch: Dict,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        kwargs = {
            "x_mask": batch["x_mask"],
            "c_pc_xyz": batch["scene_xyz"],
            "c_pc_contact": batch["base_contact"],
            "c_text": batch["texts"],
        }
        if mode != "legacy":
            kwargs["c_iiw_residual"] = residual_for(batch, mode)
        terms = diffusion.training_losses(
            cmdm,
            batch["x"],
            timesteps,
            model_kwargs=kwargs,
            noise=noise,
        )
        loss = terms["loss"].mean()
        assert_finite(mode + " motion loss", loss)
        return loss

    modes = (
        "predicted",
        "oracle",
        "reversed",
        "static_max",
        "sample_shuffled",
        "predicted_shuffled",
        "zero",
        "legacy",
    )
    losses_by_mode: Dict[str, List[float]] = {mode: [] for mode in modes}
    per_timestep = []
    per_sample_mode: Dict[str, Dict[str, List[float]]] = {
        sample["sample_id"]: {mode: [] for mode in modes}
        for sample in dataset.oracle.samples
    }
    with torch.no_grad():
        for timestep in args.eval_timesteps:
            set_seed(args.seed + 100000 + int(timestep))
            values = {mode: [] for mode in modes}
            for raw in loader:
                batch = prepare_batch(raw)
                noise = torch.randn_like(batch["x"])
                timesteps = torch.full(
                    (batch["x"].shape[0],),
                    int(timestep),
                    dtype=torch.long,
                    device=device,
                )
                # Use individual items when reporting per-sample values. The
                # diffusion loss has already reduced a multi-item batch.
                if batch["x"].shape[0] != 1:
                    raise ValueError(
                        "downstream fixed-noise evaluation requires "
                        "--batch-size 1 for exact per-sample accounting"
                    )
                sample_id = batch["sample_ids"][0]
                for mode in modes:
                    value = float(
                        motion_loss(batch, timesteps, noise, mode).item()
                    )
                    values[mode].append(value)
                    losses_by_mode[mode].append(value)
                    per_sample_mode[sample_id][mode].append(value)
            row = {"timestep": int(timestep)}
            for mode in modes:
                row[mode] = float(np.mean(values[mode]))
            per_timestep.append(row)

    mean_losses = {
        mode: float(np.mean(losses_by_mode[mode])) for mode in modes
    }
    per_sample_motion = []
    for sample_id, values in per_sample_mode.items():
        per_sample_motion.append(
            {
                "sample_id": sample_id,
                **{
                    mode: float(np.mean(values[mode])) for mode in modes
                },
            }
        )

    zero_legacy_diff = float(
        np.max(
            np.abs(
                np.asarray(losses_by_mode["zero"], dtype=np.float64)
                - np.asarray(losses_by_mode["legacy"], dtype=np.float64)
            )
        )
    )
    if zero_legacy_diff != 0.0:
        raise AssertionError(
            "zero IIW differs from legacy under fixed noise: "
            + str(zero_legacy_diff)
        )
    predicted_beats_zero = mean_losses["predicted"] < mean_losses["zero"]
    predicted_beats_predicted_shuffled = (
        mean_losses["predicted"] < mean_losses["predicted_shuffled"]
    )
    oracle_beats_zero = mean_losses["oracle"] < mean_losses["zero"]
    if not oracle_beats_zero:
        raise AssertionError("loaded Oracle adapter does not beat zero IIW")
    if args.require_predicted_beats_zero and not predicted_beats_zero:
        raise AssertionError("predicted IIW does not beat zero IIW")

    frozen_checks = {
        "planner_unchanged": state_digest(planner) == planner_digest_before,
        "adapter_unchanged": state_digest(adapter) == adapter_digest_before,
        "cmdm_unchanged": state_digest(cmdm_backbone) == cmdm_digest_before,
    }
    if not all(frozen_checks.values()):
        raise AssertionError("a frozen evaluation model changed state")

    oracle_reference = None
    oracle_summary = adapter_checkpoint.get("summary")
    if isinstance(oracle_summary, Mapping):
        final_grid = oracle_summary.get("final_timestep_grid")
        if isinstance(final_grid, Mapping):
            means = final_grid.get("mean_motion_loss")
            if isinstance(means, Mapping) and "temporal" in means:
                oracle_reference = float(means["temporal"])
    oracle_reference_abs_diff = (
        abs(mean_losses["oracle"] - oracle_reference)
        if oracle_reference is not None
        else None
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS" if predicted_beats_zero else "CHECK",
        "scope": "learned IIW plan and frozen downstream conditional loss",
        "planner_inputs": "scene_points + CLIP text + 4D start/direction state",
        "planner_receives_gt_motion": False,
        "planner_receives_gt_iiw": False,
        "free_running_motion_generation": False,
        "conditional_eval_disclosure": (
            "GT motion, mask, and deterministic phase schedule are used only "
            "for matched fixed-noise denoising-loss evaluation."
        ),
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "num_samples": len(dataset),
        "num_phases": dataset.num_phases,
        "planner_checkpoint": str(planner_file),
        "planner_step": planner_checkpoint.get("step"),
        "supervised_planner_gate_passed": True,
        "planner_model_config": dict(model_config),
        "adapter_checkpoint": str(adapter_file),
        "cmdm_checkpoint": str(cmdm_file),
        "cmdm_checkpoint_coverage": coverage,
        "seed": int(args.seed),
        "eval_timesteps": [int(value) for value in args.eval_timesteps],
        "plan_metrics": plan_aggregate,
        "per_sample_plan_metrics": plan_rows,
        "motion_mean_loss": mean_losses,
        "motion_per_timestep": per_timestep,
        "per_sample_motion_mean_loss": per_sample_motion,
        "predicted_beats_zero": bool(predicted_beats_zero),
        "predicted_beats_predicted_shuffled": bool(
            predicted_beats_predicted_shuffled
        ),
        "predicted_shuffled_ranking_is_hard_pass_requirement": False,
        "oracle_beats_zero": bool(oracle_beats_zero),
        "predicted_oracle_gap": float(
            mean_losses["predicted"] - mean_losses["oracle"]
        ),
        "zero_legacy_max_abs_diff": zero_legacy_diff,
        "oracle_checkpoint_reference_mean": oracle_reference,
        "oracle_checkpoint_reference_abs_diff": oracle_reference_abs_diff,
        "frozen_state_checks": frozen_checks,
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_predictions:
        np.savez_compressed(
            output_dir / "planner_predictions.npz",
            sample_ids=np.asarray(saved_ids),
            predicted=np.stack(saved_predictions, axis=0).astype(np.float32),
            target=np.stack(saved_targets, axis=0).astype(np.float32),
        )

    print("[PASS] zero IIW has exact legacy parity")
    print("[PASS] planner, Oracle adapter, and official CMDM stayed frozen")
    print(
        "[{}] predicted={:.8f} oracle={:.8f} zero={:.8f}".format(
            "PASS" if predicted_beats_zero else "CHECK",
            mean_losses["predicted"],
            mean_losses["oracle"],
            mean_losses["zero"],
        )
    )
    for mode in modes:
        print("[OK] {} grid mean: {:.8f}".format(mode, mean_losses[mode]))
    print("[OK] saved: " + str(summary_file))
    if args.save_predictions:
        print("[OK] saved: " + str(output_dir / "planner_predictions.npz"))


if __name__ == "__main__":
    main()

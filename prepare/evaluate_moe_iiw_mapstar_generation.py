#!/usr/bin/env python3
"""Paired full-diffusion generation ablation for literal MoE-IIW map*.

The fixed-timestep diffusion objective is useful for training diagnostics but
is not a motion-generation metric.  This evaluator therefore runs the stock
CMDM reverse process with exactly the same random stream for each conditioning
mode and compares generated 22-joint motions with the production GT.

The learned path is strictly

    scene + text + state -> MoE-IIW -> point pc_weight
    mapstar = cached Base ADM * pc_weight[..., None]
    CMDM(c_pc_contact=mapstar)

No IIW embedding adapter or alternative CMDM conditioning key is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models  # noqa: E402,F401
from models.base import create_model_and_diffusion  # noqa: E402
from models.functions import encode_text_clip  # noqa: E402
from models.iiw_map_router import IIWMapRouter  # noqa: E402
from models.iiw_planner import (  # noqa: E402
    MoEIIWPlanner,
    build_iiw_planner_from_checkpoint,
)
from prepare.evaluate_moe_iiw_mapstar import (  # noqa: E402
    hard_prediction,
    validate_moe_checkpoint,
)
from prepare.iiw_mapstar_eval_contract import (  # noqa: E402
    IIWMapstarEvalDataset,
    MAX_HORIZON,
    NUM_BODY_PARTS,
    NUM_POINTS,
    assert_finite,
    checkpoint_coverage,
    freeze,
    joint_trainer_state_digest,
    load_cmdm_config,
    set_seed,
    state_digest,
    validate_planner_output,
)
from prepare.audit_moe_iiw_mapstar_joint import (  # noqa: E402
    preflight_external_audit,
)
from utils.training import load_ckpt  # noqa: E402


DEFAULT_MODES = (
    "predicted",
    "zero_weight",
    "legacy_base",
    "predicted_shuffled",
    "oracle",
)
CONTEXT_RADIUS_BY_MODE = {
    "predicted_context_r05": 0.5,
    "predicted_context_r10": 1.0,
    "predicted_context_r15": 1.5,
    "predicted_shuffled_context_r10": 1.0,
    "predicted_context_r10_a025": 1.0,
    "predicted_context_r10_a050": 1.0,
    "predicted_context_r10_a075": 1.0,
    "predicted_shuffled_context_r10_a025": 1.0,
    "predicted_shuffled_context_r10_a050": 1.0,
    "predicted_shuffled_context_r10_a075": 1.0,
}
CONTEXT_FLOOR_BY_MODE = {
    "predicted_context_r05": 1.0,
    "predicted_context_r10": 1.0,
    "predicted_context_r15": 1.0,
    "predicted_shuffled_context_r10": 1.0,
    "predicted_context_r10_a025": 0.25,
    "predicted_context_r10_a050": 0.50,
    "predicted_context_r10_a075": 0.75,
    "predicted_shuffled_context_r10_a025": 0.25,
    "predicted_shuffled_context_r10_a050": 0.50,
    "predicted_shuffled_context_r10_a075": 0.75,
}
TEMPORAL_REDUCER_MODES = (
    "predicted_phaseweighted",
    "predicted_shuffled_phaseweighted",
    "oracle_phaseweighted",
)
ALL_MODES = (
    DEFAULT_MODES
    + tuple(CONTEXT_RADIUS_BY_MODE)
    + TEMPORAL_REDUCER_MODES
)
INSTANCE_NAMES = {
    0: "environment",
    1: "chair",
    2: "bed",
    3: "whiteboard",
    4: "tv",
}
METRIC_NAMES = (
    "mpjpe_global_m",
    "mpjpe_start_aligned_m",
    "mpjpe_root_relative_pose_m",
    "root_ade_xy_m",
    "root_fde_xy_m",
    "sitting_root_position_mae_m",
    "sitting_pelvis_height_mae_m",
)
PRIMARY_METRICS = (
    "mpjpe_global_m",
    "root_ade_xy_m",
    "sitting_root_position_mae_m",
    "sitting_pelvis_height_mae_m",
)


def detect_sitting_start(gt: np.ndarray) -> Dict[str, object]:
    if gt.ndim != 3 or gt.shape[1:] != (22, 3):
        raise ValueError("GT motion must have shape [T,22,3]")
    height = gt[:, 0, 2]
    frame_count = len(height)
    edge = max(3, min(10, frame_count // 8))
    initial_height = float(np.median(height[:edge]))
    final_height = float(np.median(height[-edge:]))
    drop = initial_height - final_height
    if drop < 0.04:
        raise ValueError(
            "GT pelvis height drop is only {:.6f} m; not a sit motion".format(
                drop
            )
        )
    enter_threshold = final_height + 0.15 * drop
    stay_threshold = final_height + 0.25 * drop
    window = max(3, min(8, frame_count // 20))
    start = None
    for frame in range(frame_count - window + 1):
        if (
            height[frame] <= enter_threshold
            and float(np.max(height[frame:frame + window])) <= stay_threshold
        ):
            start = frame
            break
    if start is None:
        start = int(np.argmin(np.abs(height - final_height)))
    return {
        "frame": int(start),
        "method": "gt_stable_near_final_pelvis_height",
        "initial_height_m": initial_height,
        "final_height_m": final_height,
        "height_drop_m": drop,
        "stability_window_frames": int(window),
    }


def motion_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sitting_start: int,
) -> Dict[str, float]:
    if prediction.shape != target.shape or prediction.shape[1:] != (22, 3):
        raise ValueError("motion metrics require matching [T,22,3] arrays")
    if not 0 <= sitting_start < len(target):
        raise ValueError("sitting start lies outside motion")
    joint_error = np.linalg.norm(prediction - target, axis=-1)
    pred_root = prediction[:, 0]
    gt_root = target[:, 0]
    root_error = np.linalg.norm(
        pred_root[:, :2] - gt_root[:, :2], axis=-1
    )
    pred_pose = prediction - pred_root[:, None]
    gt_pose = target - gt_root[:, None]
    pose_error = np.linalg.norm(pred_pose - gt_pose, axis=-1)
    translation = gt_root[0] - pred_root[0]
    aligned = prediction + translation[None, None]
    aligned_error = np.linalg.norm(aligned - target, axis=-1)
    sit_pred = pred_root[sitting_start:]
    sit_gt = gt_root[sitting_start:]
    return {
        "mpjpe_global_m": float(joint_error.mean()),
        "mpjpe_start_aligned_m": float(aligned_error.mean()),
        "mpjpe_root_relative_pose_m": float(pose_error.mean()),
        "root_ade_xy_m": float(root_error.mean()),
        "root_fde_xy_m": float(root_error[-1]),
        "sitting_root_position_mae_m": float(
            np.linalg.norm(sit_pred - sit_gt, axis=-1).mean()
        ),
        "sitting_pelvis_height_mae_m": float(
            np.abs(sit_pred[:, 2] - sit_gt[:, 2]).mean()
        ),
    }


def aggregate_metrics(rows: Iterable[Mapping[str, float]]) -> Dict[str, Dict]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot aggregate empty metric rows")
    result: Dict[str, Dict] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def bootstrap_paired_difference(
    predicted: Sequence[float],
    control: Sequence[float],
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    """Bootstrap independent dataset samples, after averaging K generations."""
    first = np.asarray(predicted, dtype=np.float64)
    second = np.asarray(control, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("paired bootstrap inputs must be equal nonempty vectors")
    delta = first - second  # Negative means predicted is better.
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta), size=(replicates, len(delta)))
    sampled = delta[draw].mean(axis=1)
    mean_control = float(second.mean())
    return {
        "mean_delta_m": float(delta.mean()),
        "median_delta_m": float(np.median(delta)),
        "relative_mean_percent": float(
            100.0 * delta.mean() / max(abs(mean_control), 1e-12)
        ),
        "ci95_low_m": float(np.quantile(sampled, 0.025)),
        "ci95_high_m": float(np.quantile(sampled, 0.975)),
        "sample_win_rate": float(np.mean(delta < 0.0)),
        "sample_tie_rate": float(np.mean(delta == 0.0)),
        "num_dataset_samples": int(len(delta)),
    }


def paired_comparisons(
    rows: Sequence[Mapping[str, object]],
    modes: Sequence[str],
    primary_mode: str,
    seed: int,
    replicates: int,
) -> Dict[str, Dict]:
    by_key: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["mode"]))
        by_key.setdefault(key, []).append(row)
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    result: Dict[str, Dict] = {}
    for control in modes:
        if control == primary_mode:
            continue
        metric_result = {}
        for metric_index, metric in enumerate(METRIC_NAMES):
            predicted_values = []
            control_values = []
            for sample_id in sample_ids:
                predicted_rows = by_key[(sample_id, primary_mode)]
                control_rows = by_key[(sample_id, control)]
                predicted_values.append(float(np.mean([
                    float(row[metric]) for row in predicted_rows
                ])))
                control_values.append(float(np.mean([
                    float(row[metric]) for row in control_rows
                ])))
            metric_result[metric] = bootstrap_paired_difference(
                predicted_values,
                control_values,
                seed + 1000 * metric_index + len(control),
                replicates,
            )
        result[primary_mode + "_vs_" + control] = metric_result
    return result


def mapstar_instance_audit(
    pc_weight: np.ndarray,
    mapstar: np.ndarray,
    instance_ids: np.ndarray,
) -> Dict[str, object]:
    if pc_weight.shape != (NUM_POINTS,):
        raise ValueError("pc_weight shape mismatch")
    if mapstar.shape != (NUM_POINTS, NUM_BODY_PARTS):
        raise ValueError("mapstar shape mismatch")
    if instance_ids.shape != (NUM_POINTS,):
        raise ValueError("instance ID shape mismatch")
    channels = {
        "pelvis": mapstar[:, 0],
        "any_joint": mapstar.max(axis=1),
    }
    instance_result: Dict[str, Dict] = {}
    for instance_id, name in INSTANCE_NAMES.items():
        mask = instance_ids == instance_id
        if not np.any(mask):
            raise ValueError("scene has no points for instance " + name)
        instance_result[name] = {
            "point_count": int(mask.sum()),
            "pc_weight_mean": float(pc_weight[mask].mean()),
            "pc_weight_max": float(pc_weight[mask].max()),
        }
    summaries = {}
    top_count = max(1, int(math.ceil(NUM_POINTS * 0.10)))
    for channel_name, values in channels.items():
        nonnegative = np.maximum(values, 0.0)
        total_mass = float(nonnegative.sum())
        per_instance = {}
        for instance_id, name in INSTANCE_NAMES.items():
            mask = instance_ids == instance_id
            mass = float(nonnegative[mask].sum())
            per_instance[name] = {
                "mean": float(nonnegative[mask].mean()),
                "max": float(nonnegative[mask].max()),
                "mass": mass,
                "mass_share": mass / max(total_mass, 1e-20),
            }
        top_indices = np.argpartition(nonnegative, -top_count)[-top_count:]
        top_ids = instance_ids[top_indices]
        top_fraction = {
            name: float(np.mean(top_ids == instance_id))
            for instance_id, name in INSTANCE_NAMES.items()
        }
        highest_mass = max(
            per_instance, key=lambda name: per_instance[name]["mass"]
        )
        summaries[channel_name] = {
            "highest_mass_instance": highest_mass,
            "top10_instance_fraction": top_fraction,
            "per_instance": per_instance,
        }
    return {
        "instances": instance_result,
        "channels": summaries,
        "chair_dominates_pelvis": (
            summaries["pelvis"]["highest_mass_instance"] == "chair"
        ),
        "chair_dominates_any_joint": (
            summaries["any_joint"]["highest_mass_instance"] == "chair"
        ),
    }


def preserve_selected_object_context(
    base: torch.Tensor,
    pc_weight: torch.Tensor,
    scene_xyz: torch.Tensor,
    instance_ids: torch.Tensor,
    radius: float,
    context_floor: float = 1.0,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Restore Base ADM on nearby environment points, not other furniture.

    The selected furniture is inferred from mean terminal IIW pc_weight over
    the three candidate instances.  Learned weights remain unchanged on all
    furniture.  Only environment points within ``radius`` of the selected
    furniture receive weight one, preserving the Base ADM spatial halo that
    CMDM can use to anchor its root trajectory.
    """
    if tuple(base.shape) != (1, NUM_POINTS, NUM_BODY_PARTS):
        raise ValueError("context base shape mismatch")
    if tuple(pc_weight.shape) != (1, NUM_POINTS):
        raise ValueError("context pc_weight shape mismatch")
    if tuple(scene_xyz.shape) != (1, NUM_POINTS, 3):
        raise ValueError("context xyz shape mismatch")
    if tuple(instance_ids.shape) != (1, NUM_POINTS):
        raise ValueError("context instance-ID shape mismatch")
    if radius <= 0.0 or chunk_size <= 0:
        raise ValueError("context radius and chunk size must be positive")
    if not 0.0 <= context_floor <= 1.0:
        raise ValueError("context floor must lie in [0,1]")
    candidate_scores = []
    for instance_id in (1, 2, 3):
        mask = instance_ids[0] == instance_id
        if not bool(mask.any().item()):
            raise ValueError("candidate instance has no scene points")
        candidate_scores.append(pc_weight[0, mask].mean())
    selected_id = 1 + int(torch.stack(candidate_scores).argmax().item())
    selected_mask = instance_ids[0] == selected_id
    environment_indices = torch.nonzero(
        instance_ids[0] == 0, as_tuple=False
    ).flatten()
    selected_xyz = scene_xyz[0, selected_mask]
    if selected_xyz.numel() == 0 or environment_indices.numel() == 0:
        raise ValueError("selected object or environment region is empty")
    context_indices: List[torch.Tensor] = []
    for start in range(0, int(environment_indices.numel()), chunk_size):
        indices = environment_indices[start:start + chunk_size]
        distance = torch.cdist(scene_xyz[0, indices], selected_xyz).amin(dim=1)
        context_indices.append(indices[distance <= radius])
    context = torch.cat(context_indices, dim=0)
    expanded_weight = pc_weight.clone()
    expanded_weight[0, context] = torch.maximum(
        expanded_weight[0, context],
        torch.full_like(expanded_weight[0, context], context_floor),
    )
    mapstar = base.detach() * expanded_weight.unsqueeze(-1)
    return (
        mapstar.contiguous(),
        expanded_weight.contiguous(),
        selected_id,
        int(context.numel()),
    )


def phaseweighted_pc_weight(
    base: torch.Tensor,
    native_iiw: torch.Tensor,
    minimum_phase_weight: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse all temporal IIW phases without discarding approach history.

    First reduce the six native body channels at each phase.  Then apply a
    monotonic 0.25..1.0 phase ramp and take the maximum over phases.  Early
    approach regions survive at lower magnitude while the final interaction
    remains strongest.  Unlike a plain max, reversing phase order changes the
    resulting static point weights.
    """
    if tuple(base.shape) != (1, NUM_POINTS, NUM_BODY_PARTS):
        raise ValueError("phaseweighted base shape mismatch")
    if native_iiw.ndim != 4 or tuple(native_iiw.shape[0:1]) != (1,):
        raise ValueError("phaseweighted IIW must be [1,Q,N,6]")
    if tuple(native_iiw.shape[2:]) != (NUM_POINTS, NUM_BODY_PARTS):
        raise ValueError("phaseweighted IIW point/body shape mismatch")
    if not 0.0 <= minimum_phase_weight <= 1.0:
        raise ValueError("minimum phase weight must lie in [0,1]")
    if float(native_iiw.min().item()) < -1e-6 or float(
        native_iiw.max().item()
    ) > 1.0 + 1e-6:
        raise ValueError("phaseweighted IIW lies outside [0,1]")
    phase_pc_weight = native_iiw.max(dim=-1)[0]
    phase_count = int(phase_pc_weight.shape[1])
    ramp = torch.linspace(
        minimum_phase_weight,
        1.0,
        phase_count,
        device=native_iiw.device,
        dtype=native_iiw.dtype,
    ).view(1, phase_count, 1)
    pc_weight = (phase_pc_weight * ramp).max(dim=1)[0]
    mapstar = base.detach() * pc_weight.unsqueeze(-1)
    return (
        mapstar.contiguous(),
        pc_weight.contiguous(),
        phase_pc_weight.contiguous(),
    )


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_bars(
    path: Path,
    aggregates: Mapping[str, Mapping[str, Mapping[str, float]]],
    modes: Sequence[str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#3b82f6", "#6b7280", "#ef4444", "#f59e0b", "#10b981"]
    for axis, metric in zip(axes.flat, PRIMARY_METRICS):
        means = [aggregates[mode][metric]["mean"] for mode in modes]
        stds = [aggregates[mode][metric]["std"] for mode in modes]
        axis.bar(range(len(modes)), means, yerr=stds, color=colors[:len(modes)])
        axis.set_xticks(range(len(modes)))
        axis.set_xticklabels(modes, rotation=25, ha="right")
        axis.set_title(metric)
        axis.set_ylabel("error (m), lower is better")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Paired full-diffusion MoE-IIW map* generation ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-v2-audit", type=Path)
    parser.add_argument("--cmdm-checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument(
        "--max-samples", type=int,
        help="smoke-test only; omit for the required complete evaluation",
    )
    parser.add_argument(
        "--sample-id", action="append",
        help="evaluate an explicit sample ID; repeat for multiple smoke samples",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-motions", action="store_true")
    parser.add_argument(
        "--modes", nargs="+", choices=ALL_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--primary-mode", choices=ALL_MODES, default="predicted",
        help="mode treated as the proposed method in paired scientific gates",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_generations <= 0:
        parser.error("--num-generations must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_samples is not None and args.sample_id:
        parser.error("use either --max-samples or --sample-id, not both")
    if args.bootstrap_replicates < 1000:
        parser.error("--bootstrap-replicates must be at least 1000")
    modes = tuple(dict.fromkeys(args.modes))
    if args.primary_mode not in modes or "zero_weight" not in modes:
        parser.error("modes must contain --primary-mode and zero_weight")

    dataset_root = args.dataset_root.expanduser().resolve()
    planner_file = args.planner_checkpoint.expanduser().resolve()
    cmdm_file = args.cmdm_checkpoint.expanduser().resolve()
    stats_file = args.stats.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (dataset_root, planner_file, cmdm_file, stats_file):
        if not path.exists():
            raise FileNotFoundError(path)
    audit_file = None
    if args.joint_v2_audit is not None:
        audit_file = args.joint_v2_audit.expanduser().resolve()
        preflight_external_audit(audit_file, planner_file)

    stats = np.load(stats_file, allow_pickle=False)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    if mean.shape != (1, 66) or std.shape != (1, 66):
        raise ValueError("CMDM motion stats must be [1,66]")
    if not np.isfinite(mean).all() or not np.all(std > 0.0):
        raise ValueError("CMDM motion stats are invalid")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset = IIWMapstarEvalDataset(dataset_root, args.index)
    checkpoint = torch.load(str(planner_file), map_location="cpu")
    quality_gate = validate_moe_checkpoint(
        checkpoint,
        list(args.index),
        checkpoint_file=planner_file,
        joint_v2_audit=audit_file,
    )
    planner = build_iiw_planner_from_checkpoint(
        checkpoint, device=device, strict=True
    )
    if not isinstance(planner, MoEIIWPlanner):
        raise TypeError("checkpoint did not build MoEIIWPlanner")
    freeze(planner)
    planner_digest_before = state_digest(planner)

    cfg = load_cmdm_config(REPO_ROOT)
    cmdm, diffusion = create_model_and_diffusion(cfg, device=str(device))
    cmdm.to(device)
    coverage = checkpoint_coverage(cmdm, cmdm_file, True)
    load_ckpt(cmdm, str(cmdm_file))
    freeze(cmdm)
    cmdm_digest_before = state_digest(cmdm)
    if quality_gate["acceptance_path"] == "external_noninferiority_v2":
        expected = checkpoint.get("cmdm_state_sha256")
        actual = joint_trainer_state_digest(cmdm)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError("CMDM does not match audited joint checkpoint")

    unique_texts = sorted({str(sample["text"]) for sample in dataset.samples})
    with torch.no_grad():
        encoded = encode_text_clip(
            cmdm.text_model,
            unique_texts,
            max_length=cmdm.text_max_length,
            device=str(device),
        ).float()
    text_cache = {
        text: encoded[index].detach()
        for index, text in enumerate(unique_texts)
    }
    map_router = IIWMapRouter().to(device)
    map_router.eval()

    motion_by_id: Dict[str, Dict[str, object]] = {}
    for index in range(len(dataset.motion)):
        row = dataset.motion[index]
        motion_by_id[str(row["info_sample_id"])] = row
    if len(motion_by_id) != len(dataset):
        raise AssertionError("motion sample-ID cache is incomplete")

    predicted_by_id: Dict[str, torch.Tensor] = {}
    routing_by_id: Dict[str, Dict[str, object]] = {}
    map_audits: Dict[str, Dict[str, object]] = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            sample_id = str(raw["sample_id"])
            scene_points = torch.from_numpy(raw["scene_points"])[None].to(
                device, torch.float32
            ).contiguous()
            state = torch.from_numpy(raw["state"])[None].to(
                device, torch.float32
            ).contiguous()
            text_feature = text_cache[str(raw["c_text"])][None].contiguous()
            output = planner.forward_with_routing(
                scene_points, text_feature, state, return_expert_logits=True
            )
            prediction = hard_prediction(output)
            validate_planner_output(prediction, 1, dataset.num_phases)
            predicted_by_id[sample_id] = prediction[0].cpu().contiguous()
            probabilities = output["routing_probabilities"][0].cpu()
            routing_by_id[sample_id] = {
                "probabilities": probabilities.tolist(),
                "selected_expert": int(output["selected_expert"][0].item()),
                "max_probability": float(probabilities.max().item()),
            }
            base = torch.from_numpy(raw["c_pc_contact"])[None].to(
                device, torch.float32
            ).contiguous()
            routed = map_router(base, prediction)
            motion_row = motion_by_id[sample_id]
            map_audits[sample_id] = mapstar_instance_audit(
                routed["pc_weight"][0].cpu().numpy(),
                routed["mapstar"][0].cpu().numpy(),
                np.asarray(motion_row["c_instance_ids"], dtype=np.int64),
            )
    if len(predicted_by_id) != len(dataset):
        raise AssertionError("planner prediction cache is incomplete")

    sample_index_by_id = {
        str(sample["sample_id"]): index
        for index, sample in enumerate(dataset.samples)
    }
    if args.sample_id:
        missing = [
            sample_id for sample_id in args.sample_id
            if sample_id not in sample_index_by_id
        ]
        if missing:
            raise ValueError("unknown --sample-id values: " + str(missing))
        evaluation_indices = [
            sample_index_by_id[sample_id] for sample_id in args.sample_id
        ]
    elif args.max_samples is not None:
        evaluation_indices = list(range(min(int(args.max_samples), len(dataset))))
    else:
        evaluation_indices = list(range(len(dataset)))
    if len(set(evaluation_indices)) != len(evaluation_indices):
        raise ValueError("duplicate evaluation sample IDs are not allowed")
    num_evaluated_samples = len(evaluation_indices)

    def contact_for(index: int, raw: Mapping[str, object], mode: str):
        sample_id = str(raw["sample_id"])
        base = torch.from_numpy(raw["c_pc_contact"])[None].to(
            device, torch.float32
        ).contiguous()
        if mode == "legacy_base":
            return base
        if mode in ("predicted", "predicted_phaseweighted") or (
            mode in CONTEXT_RADIUS_BY_MODE
            and not mode.startswith("predicted_shuffled_")
        ):
            plan = predicted_by_id[sample_id][None].to(device)
        elif mode == "zero_weight":
            plan = torch.zeros(
                (1, dataset.num_phases, NUM_POINTS, NUM_BODY_PARTS),
                device=device, dtype=torch.float32,
            )
        elif mode in ("oracle", "oracle_phaseweighted"):
            plan = torch.from_numpy(raw["iiw_plan"])[None].to(
                device, torch.float32
            ).contiguous()
        elif (
            mode == "predicted_shuffled"
            or mode.startswith("predicted_shuffled_context_")
            or mode == "predicted_shuffled_phaseweighted"
        ):
            peer_index = dataset.shuffle_peer_indices[index]
            peer_id = str(dataset.samples[peer_index]["sample_id"])
            plan = predicted_by_id[peer_id][None].to(device)
        else:
            raise ValueError("unknown mode: " + mode)
        if mode in TEMPORAL_REDUCER_MODES:
            return phaseweighted_pc_weight(base, plan)[0]
        routed = map_router(base, plan)
        if mode in CONTEXT_RADIUS_BY_MODE:
            motion_row = motion_by_id[sample_id]
            instance_ids = torch.from_numpy(
                np.asarray(motion_row["c_instance_ids"], dtype=np.int64)
            )[None].to(device).contiguous()
            scene_xyz = torch.from_numpy(raw["c_pc_xyz"])[None].to(
                device, torch.float32
            ).contiguous()
            context_mapstar, _, selected_id, context_count = (
                preserve_selected_object_context(
                    base,
                    routed["pc_weight"],
                    scene_xyz,
                    instance_ids,
                    CONTEXT_RADIUS_BY_MODE[mode],
                    context_floor=CONTEXT_FLOOR_BY_MODE[mode],
                )
            )
            if selected_id != 1:
                raise AssertionError(
                    sample_id + ": current chair-only dataset selected non-chair"
                )
            if context_count <= 0:
                raise AssertionError("context dilation selected no environment")
            return context_mapstar
        return routed["mapstar"].contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    saved: Dict[str, List[np.ndarray]] = {mode: [] for mode in modes}
    for dataset_index in evaluation_indices:
        raw = dataset[dataset_index]
        sample_id = str(raw["sample_id"])
        motion_row = motion_by_id[sample_id]
        valid_frames = int(np.asarray(motion_row["info_valid_frames"]).item())
        gt_flat = np.asarray(motion_row["x_raw"], np.float32)[:valid_frames]
        gt_motion = gt_flat.reshape(valid_frames, 22, 3)
        sitting = detect_sitting_start(gt_motion)
        x_mask = torch.from_numpy(raw["x_mask"])[None].to(
            device, torch.bool
        ).contiguous()
        scene_xyz = torch.from_numpy(raw["c_pc_xyz"])[None].to(
            device, torch.float32
        ).contiguous()
        text = str(raw["c_text"])
        contacts = {
            mode: contact_for(dataset_index, raw, mode) for mode in modes
        }
        for generation_index in range(args.num_generations):
            paired_seed = args.seed + dataset_index * 1000 + generation_index
            for mode in modes:
                # Re-seeding before both the initial noise and reverse process
                # gives every mode the exact same stochastic trajectory.
                set_seed(paired_seed)
                noise = torch.randn(
                    (1, MAX_HORIZON, 66), device=device
                )
                generated = diffusion.p_sample_loop(
                    cmdm,
                    (1, MAX_HORIZON, 66),
                    clip_denoised=False,
                    noise=noise,
                    model_kwargs={
                        "x_mask": x_mask,
                        "c_pc_xyz": scene_xyz,
                        "c_pc_contact": contacts[mode],
                        "c_text": [text],
                    },
                    device=device,
                    progress=not args.no_progress,
                )
                normalized = generated[0].detach().cpu().numpy().astype(np.float32)
                metric_flat = normalized * std + mean
                motion = metric_flat[:valid_frames].reshape(
                    valid_frames, 22, 3
                )
                if not np.isfinite(motion).all():
                    raise ValueError(sample_id + " generated NaN/Inf")
                metrics = motion_metrics(
                    motion, gt_motion, int(sitting["frame"])
                )
                record: Dict[str, object] = {
                    "sample_id": sample_id,
                    "scene_id": str(dataset.samples[dataset_index]["scene_id"]),
                    "generation_index": generation_index,
                    "seed": paired_seed,
                    "mode": mode,
                    **metrics,
                }
                rows.append(record)
                if args.save_motions:
                    saved[mode].append(motion.astype(np.float32))
                print(json.dumps(record))

    aggregates = {
        mode: aggregate_metrics(
            row for row in rows if str(row["mode"]) == mode
        )
        for mode in modes
    }
    comparisons = paired_comparisons(
        rows, modes, args.primary_mode,
        args.seed + 900000, args.bootstrap_replicates
    )
    zero_comparison = comparisons[
        args.primary_mode + "_vs_zero_weight"
    ]
    primary_mean_wins = {
        metric: zero_comparison[metric]["mean_delta_m"] < 0.0
        for metric in PRIMARY_METRICS
    }
    primary_ci_wins = {
        metric: zero_comparison[metric]["ci95_high_m"] < 0.0
        for metric in PRIMARY_METRICS
    }
    shuffled_control = {
        "predicted": "predicted_shuffled",
        "predicted_context_r10": "predicted_shuffled_context_r10",
        "predicted_context_r10_a025": (
            "predicted_shuffled_context_r10_a025"
        ),
        "predicted_context_r10_a050": (
            "predicted_shuffled_context_r10_a050"
        ),
        "predicted_context_r10_a075": (
            "predicted_shuffled_context_r10_a075"
        ),
        "predicted_phaseweighted": "predicted_shuffled_phaseweighted",
    }.get(args.primary_mode)
    shuffled_comparison = (
        comparisons.get(args.primary_mode + "_vs_" + shuffled_control)
        if shuffled_control is not None else None
    )
    shuffled_primary_mean_wins = (
        {
            metric: shuffled_comparison[metric]["mean_delta_m"] < 0.0
            for metric in PRIMARY_METRICS
        }
        if shuffled_comparison is not None else {}
    )
    semantic_checks = {
        "chair_dominates_pelvis_all_samples": all(
            value["chair_dominates_pelvis"] for value in map_audits.values()
        ),
        "chair_dominates_any_joint_all_samples": all(
            value["chair_dominates_any_joint"] for value in map_audits.values()
        ),
    }
    generation_checks = {
        "predicted_zero_primary_mean_wins": primary_mean_wins,
        "predicted_zero_primary_ci95_wins": primary_ci_wins,
        "all_primary_means_beat_zero": all(primary_mean_wins.values()),
        "all_primary_ci95_upper_bounds_below_zero": all(
            primary_ci_wins.values()
        ),
        "predicted_shuffled_primary_mean_wins": shuffled_primary_mean_wins,
        "all_primary_means_beat_predicted_shuffled": bool(
            shuffled_primary_mean_wins
        ) and all(shuffled_primary_mean_wins.values()),
    }
    base_comparison = comparisons.get(
        args.primary_mode + "_vs_legacy_base"
    )
    base_primary_mean_wins = (
        {
            metric: base_comparison[metric]["mean_delta_m"] < 0.0
            for metric in PRIMARY_METRICS
        }
        if base_comparison is not None else {}
    )
    generation_checks["legacy_base_primary_mean_wins"] = (
        base_primary_mean_wins
    )
    generation_checks["all_primary_means_beat_legacy_base"] = bool(
        base_primary_mean_wins
    ) and all(base_primary_mean_wins.values())
    complete_evaluation = num_evaluated_samples == len(dataset)
    scientific_status = "INCOMPLETE"
    if complete_evaluation:
        scientific_status = (
            "PASS"
            if all(semantic_checks.values())
            and generation_checks["all_primary_means_beat_zero"]
            and generation_checks["all_primary_ci95_upper_bounds_below_zero"]
            and generation_checks[
                "all_primary_means_beat_predicted_shuffled"
            ]
            and generation_checks["all_primary_means_beat_legacy_base"]
            else "CHECK"
        )
    planner_digest_after = state_digest(planner)
    cmdm_digest_after = state_digest(cmdm)
    frozen_checks = {
        "planner_unchanged": planner_digest_before == planner_digest_after,
        "cmdm_unchanged": cmdm_digest_before == cmdm_digest_after,
    }
    if not all(frozen_checks.values()):
        raise AssertionError("generation evaluator modified a frozen model")

    summary = {
        "status": "PASS",
        "scientific_status": scientific_status,
        "scope": "paired training-set full-diffusion generation ablation",
        "held_out_generalization": False,
        "claim_limit": (
            "LC17+HC6 training-set generation only; held-out scenes and "
            "non-chair targets remain untested"
        ),
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "planner_checkpoint": str(planner_file),
        "external_audit": str(audit_file) if audit_file else None,
        "cmdm_checkpoint": str(cmdm_file),
        "checkpoint_quality_gate": quality_gate,
        "cmdm_checkpoint_coverage": coverage,
        "map_definition": "cached Base ADM * terminal-max-body pc_weight",
        "context_ablation_definition": (
            "learned furniture pc_weight unchanged; Base ADM restored only "
            "on environment points within radius of IIW-selected furniture"
        ),
        "context_radius_by_mode_m": CONTEXT_RADIUS_BY_MODE,
        "context_floor_by_mode": CONTEXT_FLOOR_BY_MODE,
        "phaseweighted_reducer": (
            "pc_weight=max_phase(linear_ramp_0.25_to_1.0 * "
            "max_native_body(IIW_phase))"
        ),
        "cmdm_conditioning_keys": [
            "x_mask", "c_pc_xyz", "c_pc_contact", "c_text"
        ],
        "modes": list(modes),
        "primary_mode": args.primary_mode,
        "num_dataset_samples": len(dataset),
        "num_evaluated_samples": num_evaluated_samples,
        "evaluated_sample_ids": [
            str(dataset.samples[index]["sample_id"])
            for index in evaluation_indices
        ],
        "complete_evaluation": complete_evaluation,
        "num_generations_per_sample": args.num_generations,
        "paired_seed_policy": "seed + dataset_index*1000 + generation_index",
        "same_reverse_random_stream_per_mode": True,
        "bootstrap_unit": "dataset sample after averaging K generations",
        "bootstrap_replicates": args.bootstrap_replicates,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "semantic_mapstar_checks": semantic_checks,
        "generation_checks": generation_checks,
        "frozen_checks": frozen_checks,
        "routing": routing_by_id,
        "mapstar_instance_audit": map_audits,
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    write_rows(output_dir / "generation_metrics.csv", rows)
    plot_metric_bars(
        output_dir / "generation_ablation.png", aggregates, modes
    )
    if args.save_motions:
        payload = {
            "sample_id": np.asarray([str(row["sample_id"]) for row in rows]),
            "mode": np.asarray([str(row["mode"]) for row in rows]),
            "generation_index": np.asarray(
                [int(row["generation_index"]) for row in rows], np.int64
            ),
        }
        # Motion lengths differ, so each mode is intentionally stored as an
        # object array only when explicitly requested.
        for mode in modes:
            object_array = np.empty(len(saved[mode]), dtype=object)
            object_array[:] = saved[mode]
            payload[mode + "_motion"] = object_array
        np.savez_compressed(output_dir / "generated_motions.npz", **payload)

    print("[PASS] paired full-diffusion generation evaluation completed")
    print("[{}] scientific status".format(scientific_status))
    for metric in PRIMARY_METRICS:
        result = zero_comparison[metric]
        print(
            "[ZERO] {} delta={:+.6f}m ci95=[{:+.6f},{:+.6f}] "
            "win_rate={:.3f}".format(
                metric,
                result["mean_delta_m"],
                result["ci95_low_m"],
                result["ci95_high_m"],
                result["sample_win_rate"],
            )
        )
    for path in (
        summary_file,
        output_dir / "generation_metrics.csv",
        output_dir / "generation_ablation.png",
    ):
        print("[OK] saved: " + str(path))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the strongest literal target-instance ``Base ADM * pc_weight`` ceiling.

This diagnostic deliberately does not load an IIW planner.  For every sample,
the prepared candidate-instance mask supplies a perfect binary point weight::

    pc_weight[n] = 1  when point n belongs to the GT target instance
                   0  otherwise
    mapstar = cached_base_affordance.detach() * pc_weight[..., None]

The frozen CMDM then receives exactly one of three paired contact conditions:

``oracle_instance``
    The literal target-instance map* above.
``zero_weight``
    An all-zero contact map.
``legacy_base``
    The unchanged cached Base ADM (also checked against identity weighting).

All modes share the same motion target, diffusion timestep, and sampled noise.
The result is an engineering ceiling test, not a learned-model evaluation.  A
single sample cannot establish statistical significance, but failure to beat
zero contact shows that a perfect selector cannot rescue the cached Base ADM.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models  # noqa: E402,F401
from datasets.history_affordance_v1 import (  # noqa: E402
    HistoryAffordanceV1ContactMotionDataset,
)
from models.base import create_model_and_diffusion  # noqa: E402
from prepare.iiw_mapstar_eval_contract import (  # noqa: E402
    DEFAULT_GRID,
    MAX_HORIZON,
    NUM_BODY_PARTS,
    NUM_POINTS,
    assert_finite,
    checkpoint_coverage,
    freeze,
    load_cmdm_config,
    set_seed,
    state_digest,
)
from utils.training import load_ckpt  # noqa: E402


TARGET_NAMES = ("chair", "bed", "whiteboard")
MODES = ("oracle_instance", "zero_weight", "identity_weight", "legacy_base")


def build_instance_oracle_contacts(
    base_affordance: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_index: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build exact binary target-instance weights and paired contact maps."""
    if not isinstance(base_affordance, torch.Tensor):
        raise TypeError("base_affordance must be a tensor")
    if not isinstance(candidate_mask, torch.Tensor):
        raise TypeError("candidate_mask must be a tensor")
    if not isinstance(target_index, torch.Tensor):
        raise TypeError("target_index must be a tensor")
    if base_affordance.ndim != 3 or base_affordance.shape[-1] != NUM_BODY_PARTS:
        raise ValueError("base_affordance must have shape [B,N,6]")
    if candidate_mask.ndim != 3 or candidate_mask.shape[-1] != len(TARGET_NAMES):
        raise ValueError("candidate_mask must have shape [B,N,3]")
    if target_index.ndim != 1:
        raise ValueError("target_index must have shape [B]")
    batch_size, num_points, _ = base_affordance.shape
    if tuple(candidate_mask.shape[:2]) != (batch_size, num_points):
        raise ValueError("base/candidate point shape mismatch")
    if tuple(target_index.shape) != (batch_size,):
        raise ValueError("base/target batch shape mismatch")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be boolean")
    if target_index.dtype != torch.long:
        raise TypeError("target_index must be int64")
    if not torch.is_floating_point(base_affordance):
        raise TypeError("base_affordance must be floating point")
    if not bool(torch.isfinite(base_affordance).all().item()):
        raise ValueError("base_affordance contains NaN/Inf")
    if float(base_affordance.min().item()) < 0.0 or float(
        base_affordance.max().item()
    ) > 1.0:
        raise ValueError("base_affordance must lie in [0,1]")
    if bool((candidate_mask.sum(dim=-1) > 1).any().item()):
        raise ValueError("candidate instance masks overlap")
    if bool(((target_index < 0) | (target_index >= len(TARGET_NAMES))).any().item()):
        raise ValueError("target_index lies outside [0,2]")

    gather_index = target_index.view(batch_size, 1, 1).expand(
        batch_size, num_points, 1
    )
    pc_weight = candidate_mask.gather(2, gather_index).squeeze(2)
    if bool((pc_weight.sum(dim=1) <= 0).any().item()):
        raise ValueError("a target instance mask contains no points")
    pc_weight = pc_weight.to(dtype=base_affordance.dtype).contiguous()

    frozen_base = base_affordance.detach()
    oracle = frozen_base * pc_weight.unsqueeze(-1)
    zero = torch.zeros_like(frozen_base)
    identity = frozen_base * torch.ones_like(pc_weight).unsqueeze(-1)
    if not torch.equal(identity, frozen_base):
        raise AssertionError("identity pc_weight does not reproduce Base ADM")
    if not torch.equal(oracle, frozen_base * pc_weight.unsqueeze(-1)):
        raise AssertionError("oracle map* is not exact Base ADM multiplication")
    outside = oracle * (1.0 - pc_weight.unsqueeze(-1))
    if not torch.equal(outside, torch.zeros_like(outside)):
        raise AssertionError("oracle map* retained a non-target point")

    return {
        "pc_weight": pc_weight,
        "oracle_instance": oracle.contiguous(),
        "zero_weight": zero.contiguous(),
        "identity_weight": identity.contiguous(),
        "legacy_base": frozen_base.contiguous(),
    }


def top_fraction_mean(values: torch.Tensor, fraction: float = 0.1) -> float:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("top_fraction_mean expects a non-empty vector")
    count = max(1, int(np.ceil(int(values.numel()) * float(fraction))))
    return float(torch.topk(values, count, largest=True).values.mean().item())


def paired_summary(
    mode_losses: Mapping[str, Sequence[float]],
    sample_ids: Sequence[str],
    timesteps: Sequence[int],
) -> Dict[str, object]:
    expected = len(sample_ids) * len(timesteps)
    for mode in MODES:
        if mode not in mode_losses or len(mode_losses[mode]) != expected:
            raise ValueError("incomplete loss vector for " + mode)
    means = {
        mode: float(np.mean(np.asarray(mode_losses[mode], dtype=np.float64)))
        for mode in MODES
    }
    oracle = np.asarray(mode_losses["oracle_instance"], dtype=np.float64)
    zero = np.asarray(mode_losses["zero_weight"], dtype=np.float64)
    base = np.asarray(mode_losses["legacy_base"], dtype=np.float64)
    relative_improvement = float(
        (means["zero_weight"] - means["oracle_instance"])
        / max(abs(means["zero_weight"]), 1e-12)
    )
    return {
        "motion_mean_loss": means,
        "oracle_minus_zero": float(np.mean(oracle - zero)),
        "oracle_minus_base": float(np.mean(oracle - base)),
        "oracle_relative_improvement_vs_zero": relative_improvement,
        "oracle_pair_win_count_vs_zero": int(np.sum(oracle < zero)),
        "oracle_pair_count_vs_zero": int(oracle.size),
        "oracle_beats_zero": bool(means["oracle_instance"] < means["zero_weight"]),
        "oracle_beats_base": bool(means["oracle_instance"] < means["legacy_base"]),
        "identity_legacy_max_abs_diff": float(np.max(np.abs(
            np.asarray(mode_losses["identity_weight"], dtype=np.float64) - base
        ))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--cmdm-checkpoint", type=Path, required=True)
    parser.add_argument("--allow-partial-cmdm-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--eval-timesteps", type=int, nargs="+", default=list(DEFAULT_GRID)
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-mapstar", action="store_true")
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=0.01,
        help="strict diagnostic margin against zero contact (default: 1%%)",
    )
    parser.add_argument(
        "--minimum-improved-timesteps",
        type=int,
        default=4,
        help="minimum paired timestep wins required per sample",
    )
    args = parser.parse_args()

    if args.batch_size != 1:
        parser.error("paired per-sample evaluation requires --batch-size 1")
    if not args.eval_timesteps:
        parser.error("--eval-timesteps must not be empty")
    if args.minimum_relative_improvement < 0.0:
        parser.error("--minimum-relative-improvement must be non-negative")
    if not (1 <= args.minimum_improved_timesteps <= len(args.eval_timesteps)):
        parser.error(
            "--minimum-improved-timesteps must lie in [1, number of timesteps]"
        )

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_root = args.dataset_root.expanduser().resolve()
    cmdm_file = args.cmdm_checkpoint.expanduser().resolve()
    if not cmdm_file.is_file():
        raise FileNotFoundError(cmdm_file)

    dataset_cfg = SimpleNamespace(
        dataset_root=str(dataset_root),
        index_files=list(args.index),
        num_points=NUM_POINTS,
        max_horizon=MAX_HORIZON,
    )
    dataset = HistoryAffordanceV1ContactMotionDataset(dataset_cfg, phase="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    cfg = load_cmdm_config(REPO_ROOT)
    cmdm, diffusion = create_model_and_diffusion(cfg, device=str(device))
    cmdm.to(device)
    coverage = checkpoint_coverage(
        cmdm,
        cmdm_file,
        require_trainable_complete=(not args.allow_partial_cmdm_checkpoint),
    )
    load_ckpt(cmdm, str(cmdm_file))
    freeze(cmdm)
    cmdm_digest_before = state_digest(cmdm)
    if any(
        value < 0 or value >= int(diffusion.num_timesteps)
        for value in args.eval_timesteps
    ):
        raise ValueError("eval timestep lies outside diffusion schedule")

    sample_ids = [str(sample["sample_id"]) for sample in dataset.samples]
    mode_losses: Dict[str, List[float]] = {mode: [] for mode in MODES}
    per_timestep: List[Dict[str, object]] = []
    per_sample: Dict[str, Dict[str, object]] = {
        sample_id: {
            "sample_id": sample_id,
            "scene_id": str(dataset.samples[index]["scene_id"]),
            "target_index": int(dataset.samples[index]["target_index"]),
            "target_name": TARGET_NAMES[int(dataset.samples[index]["target_index"])],
            "losses": {mode: [] for mode in MODES},
        }
        for index, sample_id in enumerate(sample_ids)
    }
    saved_ids: List[str] = []
    saved_weights: List[np.ndarray] = []
    saved_maps: List[np.ndarray] = []
    semantic_rows: List[Dict[str, object]] = []

    def prepare(raw: Dict[str, object]) -> Dict[str, object]:
        batch = {
            "x": raw["x"].to(device, torch.float32).contiguous(),
            "x_mask": raw["x_mask"].to(device, torch.bool).contiguous(),
            "scene_xyz": raw["c_pc_xyz"].to(
                device, torch.float32
            ).contiguous(),
            "base": raw["c_base_affordance"].to(
                device, torch.float32
            ).contiguous(),
            "candidate_mask": raw["c_candidate_mask"].to(
                device, torch.bool
            ).contiguous(),
            "target_index": raw["c_target_index"].to(
                device, torch.long
            ).reshape(-1).contiguous(),
            "texts": [str(value) for value in raw["c_text"]],
            "sample_ids": [str(value) for value in raw["info_sample_id"]],
            "scene_ids": [str(value) for value in raw["info_scene_id"]],
        }
        batch_size = int(batch["x"].shape[0])
        expected = {
            "x": (batch_size, MAX_HORIZON, 66),
            "x_mask": (batch_size, MAX_HORIZON),
            "scene_xyz": (batch_size, NUM_POINTS, 3),
            "base": (batch_size, NUM_POINTS, NUM_BODY_PARTS),
            "candidate_mask": (batch_size, NUM_POINTS, len(TARGET_NAMES)),
            "target_index": (batch_size,),
        }
        for name, shape in expected.items():
            if tuple(batch[name].shape) != shape:
                raise ValueError(
                    "{} shape mismatch: {} != {}".format(
                        name, tuple(batch[name].shape), shape
                    )
                )
        for name in ("x", "scene_xyz", "base"):
            assert_finite(name, batch[name])
        return batch

    def loss_for(
        batch: Dict[str, object],
        contacts: Mapping[str, torch.Tensor],
        mode: str,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        kwargs = {
            "x_mask": batch["x_mask"],
            "c_pc_xyz": batch["scene_xyz"],
            "c_pc_contact": contacts[mode],
            "c_text": batch["texts"],
        }
        if set(kwargs) != {"x_mask", "c_pc_xyz", "c_pc_contact", "c_text"}:
            raise AssertionError("unexpected CMDM conditioning key")
        terms = diffusion.training_losses(
            cmdm,
            batch["x"],
            timesteps,
            model_kwargs=kwargs,
            noise=noise,
        )
        loss = terms["loss"].mean()
        assert_finite(mode + " loss", loss)
        return loss

    with torch.no_grad():
        # Save and audit each exact oracle tensor once.
        for raw in loader:
            batch = prepare(raw)
            contacts = build_instance_oracle_contacts(
                batch["base"], batch["candidate_mask"], batch["target_index"]
            )
            sample_id = batch["sample_ids"][0]
            target_index = int(batch["target_index"][0].item())
            target_mask = contacts["pc_weight"][0].bool()
            target_any = batch["base"][0].max(dim=-1)[0].masked_select(target_mask)
            channel_top10 = [
                top_fraction_mean(
                    batch["base"][0, :, channel].masked_select(target_mask)
                )
                for channel in range(NUM_BODY_PARTS)
            ]
            semantic = {
                "sample_id": sample_id,
                "scene_id": batch["scene_ids"][0],
                "target_index": target_index,
                "target_name": TARGET_NAMES[target_index],
                "target_point_count": int(target_mask.sum().item()),
                "target_base_any_joint_mean": float(target_any.mean().item()),
                "target_base_any_joint_max": float(target_any.max().item()),
                "target_base_any_joint_top10_mean": top_fraction_mean(target_any),
                "target_base_channel_top10_mean": channel_top10,
                "oracle_mapstar_min": float(
                    contacts["oracle_instance"][0].min().item()
                ),
                "oracle_mapstar_max": float(
                    contacts["oracle_instance"][0].max().item()
                ),
            }
            semantic_rows.append(semantic)
            per_sample[sample_id].update(semantic)
            if args.save_mapstar:
                saved_ids.append(sample_id)
                saved_weights.append(contacts["pc_weight"][0].cpu().numpy())
                saved_maps.append(contacts["oracle_instance"][0].cpu().numpy())

        for timestep in args.eval_timesteps:
            row: Dict[str, object] = {"timestep": int(timestep), "per_sample": []}
            step_values: Dict[str, List[float]] = {mode: [] for mode in MODES}
            for sample_offset, raw in enumerate(loader):
                batch = prepare(raw)
                contacts = build_instance_oracle_contacts(
                    batch["base"], batch["candidate_mask"], batch["target_index"]
                )
                # One deterministic, paired noise tensor per sample and timestep.
                set_seed(args.seed + 100000 + int(timestep) + 1009 * sample_offset)
                noise = torch.randn_like(batch["x"])
                timesteps = torch.full(
                    (1,), int(timestep), dtype=torch.long, device=device
                )
                sample_id = batch["sample_ids"][0]
                sample_row: Dict[str, object] = {"sample_id": sample_id}
                for mode in MODES:
                    value = float(
                        loss_for(batch, contacts, mode, timesteps, noise).item()
                    )
                    mode_losses[mode].append(value)
                    step_values[mode].append(value)
                    per_sample[sample_id]["losses"][mode].append(value)
                    sample_row[mode] = value
                row["per_sample"].append(sample_row)
            for mode in MODES:
                row[mode] = float(np.mean(step_values[mode]))
            per_timestep.append(row)

    aggregate = paired_summary(
        mode_losses, sample_ids=sample_ids, timesteps=args.eval_timesteps
    )
    if aggregate["identity_legacy_max_abs_diff"] != 0.0:
        raise AssertionError("identity weight and legacy Base losses differ")
    for sample_id in sample_ids:
        sample_means = {
            mode: float(np.mean(per_sample[sample_id]["losses"][mode]))
            for mode in MODES
        }
        per_sample[sample_id]["motion_mean_loss"] = sample_means
        per_sample[sample_id]["oracle_minus_zero"] = float(
            sample_means["oracle_instance"] - sample_means["zero_weight"]
        )
        per_sample[sample_id]["oracle_beats_zero"] = bool(
            sample_means["oracle_instance"] < sample_means["zero_weight"]
        )
        oracle_values = np.asarray(
            per_sample[sample_id]["losses"]["oracle_instance"],
            dtype=np.float64,
        )
        zero_values = np.asarray(
            per_sample[sample_id]["losses"]["zero_weight"],
            dtype=np.float64,
        )
        relative_improvement = float(
            (sample_means["zero_weight"] - sample_means["oracle_instance"])
            / max(abs(sample_means["zero_weight"]), 1e-12)
        )
        improved_timestep_count = int(np.sum(oracle_values < zero_values))
        strict_useful = bool(
            relative_improvement >= args.minimum_relative_improvement
            and improved_timestep_count >= args.minimum_improved_timesteps
        )
        per_sample[sample_id]["oracle_relative_improvement_vs_zero"] = (
            relative_improvement
        )
        per_sample[sample_id]["oracle_improved_timestep_count"] = (
            improved_timestep_count
        )
        per_sample[sample_id]["oracle_strictly_useful"] = strict_useful
        per_sample[sample_id]["oracle_ceiling"] = (
            "USEFUL_DIAGNOSTIC"
            if strict_useful
            else "ZERO_LEVEL_OR_INCONSISTENT"
        )

    frozen_checks = {
        "cmdm_unchanged": state_digest(cmdm) == cmdm_digest_before,
        "identity_weight_bitwise_base": True,
        "zero_weight_exact_zero": True,
        "oracle_exact_base_times_binary_target_mask": True,
        "paired_noise_per_sample_timestep": True,
    }
    if not all(frozen_checks.values()):
        raise AssertionError("oracle evaluator invariant failed")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS",
        "scientific_status": "CHECK",
        "scope": "perfect target-instance binary pc_weight ceiling",
        "map_definition": (
            "mapstar = cached_base_affordance.detach() * "
            "candidate_mask[target_index][...,None]"
        ),
        "interpretation": (
            "Single-sample target types are diagnostic only. "
            "A useful diagnostic requires the configured relative loss margin "
            "and paired timestep-win count; otherwise the current Base ADM is "
            "treated as zero-level or inconsistent for that sample."
        ),
        "strict_decision_thresholds": {
            "minimum_relative_improvement": float(
                args.minimum_relative_improvement
            ),
            "minimum_improved_timesteps": int(args.minimum_improved_timesteps),
            "total_timesteps": len(args.eval_timesteps),
        },
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "num_samples": len(dataset),
        "cmdm_checkpoint": str(cmdm_file),
        "cmdm_checkpoint_coverage": coverage,
        "eval_timesteps": [int(value) for value in args.eval_timesteps],
        "aggregate": aggregate,
        "per_sample": [per_sample[value] for value in sample_ids],
        "semantic_target_signal": semantic_rows,
        "frozen_checks": frozen_checks,
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_mapstar:
        np.savez_compressed(
            output_dir / "oracle_instance_mapstar.npz",
            sample_ids=np.asarray(saved_ids),
            pc_weight=np.stack(saved_weights).astype(np.float32),
            mapstar=np.stack(saved_maps).astype(np.float32),
        )

    print("[PASS] exact binary target-instance pc_weight and map* construction")
    print("[PASS] identity-weight/Base parity and exact zero-contact control")
    print("[PASS] frozen CMDM paired-noise timestep-grid comparison")
    for row in summary["per_sample"]:
        means = row["motion_mean_loss"]
        print(
            "[{}] {} target={} oracle={:.8f} zero={:.8f} base={:.8f} "
            "relative={:+.3%} timestep_wins={}/{}".format(
                row["oracle_ceiling"],
                row["sample_id"],
                row["target_name"],
                means["oracle_instance"],
                means["zero_weight"],
                means["legacy_base"],
                row["oracle_relative_improvement_vs_zero"],
                row["oracle_improved_timestep_count"],
                len(args.eval_timesteps),
            )
        )
    print("[OK] saved: " + str(summary_file))
    if args.save_mapstar:
        print("[OK] saved: " + str(output_dir / "oracle_instance_mapstar.npz"))


if __name__ == "__main__":
    main()

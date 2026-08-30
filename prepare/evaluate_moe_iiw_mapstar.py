#!/usr/bin/env python3
"""Evaluate literal ``base ADM * point pc_weight`` conditioning.

This evaluator intentionally bypasses the IIW embedding-residual path.  A
quality-passed MoE IIW planner predicts ``[B,Q,N,6]`` native-body interaction
weights.  :class:`models.iiw_map_router.IIWMapRouter` reduces the terminal
phase to one scalar per point and builds

``mapstar = cached_base_affordance.detach() * pc_weight[..., None]``.

Only ``mapstar`` is supplied to the unmodified pretrained CMDM through its
existing ``c_pc_contact`` argument.  No embedding-condition path is used.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
from prepare.audit_moe_iiw_mapstar_joint import (  # noqa: E402
    file_sha256,
    load_external_audit,
    preflight_external_audit,
)
from prepare.iiw_mapstar_eval_contract import (  # noqa: E402
    IIWMapstarEvalDataset,
    aggregate_plan_metrics,
    assert_finite,
    checkpoint_coverage,
    DEFAULT_GRID,
    freeze,
    joint_trainer_state_digest,
    load_cmdm_config,
    MAX_HORIZON,
    NUM_BODY_PARTS,
    NUM_POINTS,
    plan_metrics,
    set_seed,
    state_digest,
    validate_planner_output,
)
from utils.training import load_ckpt  # noqa: E402


REQUIRED_MOE_CHECKS = (
    "dense_trunk_bitwise_unchanged",
    "combined_objective_improved",
    "iiw_objective_not_degraded",
    "router_accuracy_perfect",
    "both_experts_hard_selected",
    "both_experts_soft_loaded",
    "router_confident",
    "experts_diverged",
    "sparse_f1_preserved",
    "temporal_order_preserved",
    "state_outputs_not_collapsed",
)

REQUIRED_JOINT_CHECKS = (
    "source_moe_checkpoint_passed",
    "cached_base_bitwise_unchanged",
    "cmdm_bitwise_unchanged",
    "dense_iiw_trunk_bitwise_unchanged",
    "motion_only_router_gradient_nonzero",
    "motion_only_both_expert_gradients_nonzero",
    "hard_eval_routing_perfect",
    "both_experts_hard_selected",
    "both_experts_soft_loaded",
    "router_confident",
    "experts_remain_distinct",
    "iiw_f1_preserved",
    "iiw_temporal_order_preserved",
    "iiw_objective_not_degraded",
    "motion_grid_improved",
    "pc_weight_in_unit_interval",
    "mapstar_in_unit_interval",
)


def validate_moe_checkpoint(
    checkpoint: Mapping,
    index_names: List[str],
    checkpoint_file: Optional[Path] = None,
    joint_v2_audit: Optional[Path] = None,
) -> Dict[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("planner checkpoint must be a mapping")
    marker = str(
        checkpoint.get("model_type", checkpoint.get("planner_type", ""))
    ).lower()
    if "moe" not in marker:
        raise ValueError("map* production evaluation requires a MoE IIW checkpoint")
    if str(checkpoint.get("planner_type")) != "moe_iiw_v1":
        raise ValueError("map* evaluation requires planner_type=moe_iiw_v1")
    if int(checkpoint.get("format_version", -1)) != 2:
        raise ValueError("map* evaluation requires MoE checkpoint format_version=2")
    if str(checkpoint.get("target_method")) != "point_aligned_iiw_proxy":
        raise ValueError("MoE checkpoint IIW target method mismatch")
    if str(checkpoint.get("text_encoder")) != "ViT-B/32":
        raise ValueError("MoE checkpoint text encoder mismatch")
    if int(checkpoint.get("native_body_part_count", -1)) != NUM_BODY_PARTS:
        raise ValueError("MoE checkpoint native body count mismatch")
    summary = checkpoint.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("MoE checkpoint has no summary mapping")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping):
        raise TypeError("MoE checkpoint has no checks mapping")
    is_joint = str(checkpoint.get("joint_type")) == "moe_iiw_mapstar_joint_v1"
    training_status = str(summary.get("status"))
    quality_gate: Dict[str, object]
    if not is_joint:
        if joint_v2_audit is not None:
            raise ValueError(
                "--joint-v2-audit applies only to a joint checkpoint"
            )
        if training_status != "PASS":
            raise ValueError("MoE IIW checkpoint did not pass training")
        failed = [
            name for name in REQUIRED_MOE_CHECKS if not bool(checks.get(name))
        ]
        if failed:
            raise ValueError("MoE checkpoint failed: " + ", ".join(failed))
        quality_gate = {
            "training_status": "PASS",
            "acceptance_path": "strict_v1_pass",
            "external_audit_required": False,
        }
    elif training_status == "PASS":
        if joint_v2_audit is not None:
            raise ValueError(
                "a PASS joint checkpoint does not use a non-inferiority audit"
            )
        failed = [
            name for name in REQUIRED_JOINT_CHECKS if not bool(checks.get(name))
        ]
        if failed:
            raise ValueError("MoE checkpoint failed: " + ", ".join(failed))
        quality_gate = {
            "training_status": "PASS",
            "acceptance_path": "strict_v1_pass",
            "external_audit_required": False,
        }
    elif training_status == "CHECK":
        if int(checkpoint.get("joint_format_version", -1)) != 1:
            raise ValueError(
                "v2 audit requires joint checkpoint joint_format_version=1"
            )
        if joint_v2_audit is None:
            raise ValueError(
                "CHECK joint checkpoint requires --joint-v2-audit"
            )
        if checkpoint_file is None:
            raise ValueError("checkpoint path is required to validate external audit")
        audit = load_external_audit(
            joint_v2_audit, summary, checkpoint_file
        )
        quality_gate = {
            "training_status": "CHECK",
            "acceptance_path": "external_noninferiority_v2",
            "external_audit_required": True,
            "external_audit_file": str(
                joint_v2_audit.expanduser().resolve()
            ),
            "external_audit_sha256": file_sha256(
                joint_v2_audit.expanduser().resolve()
            ),
            "external_audit_policy_id": audit["policy_id"],
            "external_audit_checkpoint_sha256": audit["checkpoint"]["sha256"],
            "external_audit_summary_sha256": audit["source_summary"]["sha256"],
        }
    else:
        raise ValueError(
            "MoE joint checkpoint training status must be PASS or CHECK"
        )
    trained = checkpoint.get("index_files")
    if not isinstance(trained, (list, tuple)):
        raise TypeError("MoE checkpoint has no index_files sequence")
    trained_names = {Path(str(value)).name for value in trained}
    evaluated_names = {Path(str(value)).name for value in index_names}
    if trained_names != evaluated_names:
        raise ValueError(
            "planner/evaluation index mismatch: {} vs {}".format(
                sorted(trained_names), sorted(evaluated_names)
            )
        )
    return quality_gate


def hard_prediction(output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    probabilities = output["routing_probabilities"]
    selected = output["selected_expert"]
    expert_logits = output.get("expert_logits")
    if expert_logits is None:
        raise KeyError("hard routing requires expert_logits")
    weights = F.one_hot(
        selected, num_classes=int(probabilities.shape[1])
    ).to(dtype=expert_logits.dtype)
    logits = (expert_logits * weights[:, :, None, None, None]).sum(dim=1)
    return torch.sigmoid(logits).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--joint-v2-audit",
        type=Path,
        help=(
            "required external v2 audit JSON when the joint checkpoint keeps "
            "its original v1 status=CHECK"
        ),
    )
    parser.add_argument("--cmdm-checkpoint", type=Path, required=True)
    parser.add_argument("--allow-partial-cmdm-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--eval-timesteps", type=int, nargs="+", default=list(DEFAULT_GRID)
    )
    parser.add_argument("--active-threshold", type=float, default=0.7)
    parser.add_argument("--routing", choices=("hard", "soft"), default="hard")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--require-predicted-beats-zero", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size != 1:
        parser.error("fixed-noise per-sample evaluation requires --batch-size 1")
    if not (0.0 < args.active_threshold < 1.0):
        parser.error("--active-threshold must lie in (0,1)")
    if not args.eval_timesteps:
        parser.error("--eval-timesteps must not be empty")

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_root = args.dataset_root.expanduser().resolve()
    planner_file = args.planner_checkpoint.expanduser().resolve()
    cmdm_file = args.cmdm_checkpoint.expanduser().resolve()
    for path in (planner_file, cmdm_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = IIWMapstarEvalDataset(dataset_root, args.index)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if args.joint_v2_audit is not None:
        preflight_external_audit(args.joint_v2_audit, planner_file)
    checkpoint = torch.load(str(planner_file), map_location="cpu")
    checkpoint_quality_gate = validate_moe_checkpoint(
        checkpoint,
        list(args.index),
        checkpoint_file=planner_file,
        joint_v2_audit=args.joint_v2_audit,
    )
    planner = build_iiw_planner_from_checkpoint(
        checkpoint, device=device, strict=True
    )
    if not isinstance(planner, MoEIIWPlanner):
        raise TypeError("checkpoint factory did not build MoEIIWPlanner")
    freeze(planner)
    planner_digest_before = state_digest(planner)

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
    audited_check_joint = (
        checkpoint_quality_gate["acceptance_path"]
        == "external_noninferiority_v2"
    )
    expected_joint_cmdm_digest = (
        checkpoint.get("cmdm_state_sha256") if audited_check_joint else None
    )
    actual_joint_cmdm_digest = None
    if audited_check_joint:
        if not isinstance(expected_joint_cmdm_digest, str):
            raise ValueError("joint checkpoint has no CMDM state SHA-256")
        actual_joint_cmdm_digest = joint_trainer_state_digest(cmdm)
        if actual_joint_cmdm_digest != expected_joint_cmdm_digest:
            raise ValueError(
                "evaluation CMDM state does not match the joint-training CMDM"
            )
    if any(value < 0 or value >= int(diffusion.num_timesteps)
           for value in args.eval_timesteps):
        raise ValueError("eval timestep lies outside diffusion schedule")

    unique_texts = sorted({str(sample["text"]) for sample in dataset.samples})
    with torch.no_grad():
        encoded = encode_text_clip(
            cmdm.text_model,
            unique_texts,
            max_length=cmdm.text_max_length,
            device=str(device),
        ).float()
    if tuple(encoded.shape[1:]) != (512,):
        raise ValueError("planner requires 512D CLIP features")
    text_cache = {
        text: encoded[index].detach()
        for index, text in enumerate(unique_texts)
    }
    map_router = IIWMapRouter().to(device)
    identity_tensor_parity_checked = False

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
            "base": raw["c_pc_contact"].to(
                device, torch.float32
            ).contiguous(),
            "gt_plan": raw["iiw_plan"].to(
                device, torch.float32
            ).contiguous(),
            "shuffled_plan": raw["iiw_shuffled"].to(
                device, torch.float32
            ).contiguous(),
            "state": raw["state"].to(device, torch.float32).contiguous(),
            "texts": [str(value) for value in raw["c_text"]],
            "sample_ids": [str(value) for value in raw["sample_id"]],
            "shuffled_ids": [str(value) for value in raw["shuffled_sample_id"]],
        }
        batch_size = int(result["x"].shape[0])
        expected = {
            "x": (batch_size, MAX_HORIZON, 66),
            "x_mask": (batch_size, MAX_HORIZON),
            "scene_points": (batch_size, NUM_POINTS, 6),
            "scene_xyz": (batch_size, NUM_POINTS, 3),
            "base": (batch_size, NUM_POINTS, NUM_BODY_PARTS),
            "gt_plan": (
                batch_size, dataset.num_phases, NUM_POINTS, NUM_BODY_PARTS
            ),
            "shuffled_plan": (
                batch_size, dataset.num_phases, NUM_POINTS, NUM_BODY_PARTS
            ),
            "state": (batch_size, 4),
        }
        for name, shape in expected.items():
            if tuple(result[name].shape) != shape:
                raise ValueError("{} shape mismatch: {} != {}".format(
                    name, tuple(result[name].shape), shape
                ))
        for name in ("x", "scene_points", "scene_xyz", "base", "gt_plan", "state"):
            assert_finite(name, result[name])
        result["text_features"] = torch.stack(
            [text_cache[text] for text in result["texts"]], dim=0
        ).contiguous()
        return result

    predicted_by_id: Dict[str, torch.Tensor] = {}
    routing_by_id: Dict[str, Dict[str, object]] = {}
    plan_rows: List[Dict[str, object]] = []
    saved_predictions: List[np.ndarray] = []
    saved_targets: List[np.ndarray] = []
    saved_weights: List[np.ndarray] = []
    saved_maps: List[np.ndarray] = []
    saved_ids: List[str] = []
    with torch.no_grad():
        for raw in loader:
            batch = prepare_batch(raw)
            output = planner.forward_with_routing(
                batch["scene_points"],
                batch["text_features"],
                batch["state"],
                return_expert_logits=(args.routing == "hard"),
            )
            prediction = (
                hard_prediction(output)
                if args.routing == "hard"
                else output["prediction"]
            )
            validate_planner_output(
                prediction, int(batch["x"].shape[0]), dataset.num_phases
            )
            routed = map_router(batch["base"], prediction)
            for index, sample_id in enumerate(batch["sample_ids"]):
                pred = prediction[index : index + 1]
                target = batch["gt_plan"][index : index + 1]
                metrics = plan_metrics(pred, target, args.active_threshold)
                metrics["sample_id"] = sample_id
                plan_rows.append(metrics)
                predicted_by_id[sample_id] = pred[0].cpu().contiguous()
                probabilities = output["routing_probabilities"][index].cpu()
                routing_by_id[sample_id] = {
                    "probabilities": probabilities.tolist(),
                    "selected_expert": int(
                        output["selected_expert"][index].item()
                    ),
                    "max_probability": float(probabilities.max().item()),
                }
                if args.save_predictions:
                    saved_predictions.append(pred[0].cpu().numpy())
                    saved_targets.append(target[0].cpu().numpy())
                    saved_weights.append(routed["pc_weight"][index].cpu().numpy())
                    saved_maps.append(routed["mapstar"][index].cpu().numpy())
                    saved_ids.append(sample_id)
    if len(predicted_by_id) != len(dataset):
        raise AssertionError("planner prediction cache is incomplete")

    aggregate = aggregate_plan_metrics(plan_rows)
    peer_deltas = []
    for sample_index, peer_index in enumerate(dataset.shuffle_peer_indices):
        sample_id = dataset.samples[sample_index]["sample_id"]
        peer_id = dataset.samples[peer_index]["sample_id"]
        peer_deltas.append(float(
            (predicted_by_id[sample_id] - predicted_by_id[peer_id]).abs().mean()
        ))
    aggregate["same_scene_peer_prediction_mae"] = float(np.mean(peer_deltas))

    def plan_for(batch: Dict, mode: str) -> torch.Tensor:
        if mode == "predicted":
            return torch.stack(
                [predicted_by_id[value] for value in batch["sample_ids"]], dim=0
            ).to(device).contiguous()
        if mode == "oracle":
            return batch["gt_plan"]
        if mode == "reversed":
            return batch["gt_plan"].flip(1)
        if mode == "sample_shuffled":
            return batch["shuffled_plan"]
        if mode == "predicted_shuffled":
            return torch.stack(
                [predicted_by_id[value] for value in batch["shuffled_ids"]], dim=0
            ).to(device).contiguous()
        if mode == "zero_weight":
            return torch.zeros_like(batch["gt_plan"])
        if mode == "identity_weight":
            return torch.ones_like(batch["gt_plan"])
        raise ValueError("unknown plan mode: " + mode)

    def contact_for(batch: Dict, mode: str) -> torch.Tensor:
        nonlocal identity_tensor_parity_checked
        if mode == "legacy_base":
            return batch["base"]
        routed = map_router(batch["base"], plan_for(batch, mode))
        contact = routed["mapstar"].contiguous()
        expected = (int(batch["x"].shape[0]), NUM_POINTS, NUM_BODY_PARTS)
        if tuple(contact.shape) != expected:
            raise ValueError("map* shape mismatch")
        assert_finite("map*", contact)
        if mode == "identity_weight":
            if not torch.equal(contact, batch["base"]):
                raise AssertionError(
                    "pc_weight=1 tensor does not exactly reproduce Base ADM"
                )
            identity_tensor_parity_checked = True
        return contact

    def motion_loss(
        batch: Dict,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        kwargs = {
            "x_mask": batch["x_mask"],
            "c_pc_xyz": batch["scene_xyz"],
            "c_pc_contact": contact_for(batch, mode),
            "c_text": batch["texts"],
        }
        # Deliberate main-path invariant: the only IIW effect is map* above.
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

    modes = (
        "predicted",
        "oracle",
        "reversed",
        "sample_shuffled",
        "predicted_shuffled",
        "zero_weight",
        "identity_weight",
        "legacy_base",
    )
    losses: Dict[str, List[float]] = {mode: [] for mode in modes}
    per_timestep = []
    with torch.no_grad():
        for timestep in args.eval_timesteps:
            set_seed(args.seed + 100000 + int(timestep))
            row = {"timestep": int(timestep)}
            step_values = {mode: [] for mode in modes}
            for raw in loader:
                batch = prepare_batch(raw)
                noise = torch.randn_like(batch["x"])
                timesteps = torch.full(
                    (1,), int(timestep), dtype=torch.long, device=device
                )
                for mode in modes:
                    value = float(
                        motion_loss(batch, timesteps, noise, mode).item()
                    )
                    losses[mode].append(value)
                    step_values[mode].append(value)
            for mode in modes:
                row[mode] = float(np.mean(step_values[mode]))
            per_timestep.append(row)

    mean_losses = {mode: float(np.mean(values)) for mode, values in losses.items()}
    identity_legacy_diff = float(np.max(np.abs(
        np.asarray(losses["identity_weight"], dtype=np.float64)
        - np.asarray(losses["legacy_base"], dtype=np.float64)
    )))
    if not identity_tensor_parity_checked:
        raise AssertionError("identity-weight tensor parity was not checked")
    predicted_beats_zero = mean_losses["predicted"] < mean_losses["zero_weight"]
    predicted_beats_base = mean_losses["predicted"] < mean_losses["legacy_base"]
    predicted_beats_shuffled = (
        mean_losses["predicted"] < mean_losses["predicted_shuffled"]
    )
    if args.require_predicted_beats_zero and not predicted_beats_zero:
        raise AssertionError("predicted map* does not beat zero-contact map")

    frozen_checks = {
        "planner_unchanged": state_digest(planner) == planner_digest_before,
        "cmdm_unchanged": state_digest(cmdm) == cmdm_digest_before,
    }
    if not all(frozen_checks.values()):
        raise AssertionError("frozen evaluation model changed state")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_counts = np.bincount(
        [row["selected_expert"] for row in routing_by_id.values()],
        minlength=int(planner.num_experts),
    )
    routing_probabilities = np.asarray(
        [row["probabilities"] for row in routing_by_id.values()], dtype=np.float64
    )
    # This first frozen comparison is diagnostic.  A literal map* tensor can
    # be perfectly wired even when a planner trained for the old embedding
    # consumer has not yet learned the best static contact reweighting.  Do
    # not conflate the wiring contract with the downstream quality result.
    wiring_pass = identity_tensor_parity_checked and all(frozen_checks.values())
    quality_pass = predicted_beats_base and predicted_beats_shuffled
    status = "PASS" if wiring_pass else "CHECK"
    summary = {
        "status": status,
        "scope": "terminal-IIW scalar pc_weight multiplied with cached Base ADM",
        "map_definition": "mapstar = base_affordance.detach() * pc_weight[...,None]",
        "pc_weight_definition": "terminal_phase_max_over_native_body_channels",
        "temporal_limitation": (
            "CMDM receives one static terminal-phase map; the other seven phases "
            "are diagnostic only."
        ),
        "uses_iiw_adapter": False,
        "uses_embedding_residual": False,
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "num_samples": len(dataset),
        "planner_checkpoint": str(planner_file),
        "planner_checkpoint_quality_gate": checkpoint_quality_gate,
        "cmdm_checkpoint": str(cmdm_file),
        "joint_cmdm_state_provenance": {
            "required": bool(audited_check_joint),
            "expected_sha256": expected_joint_cmdm_digest,
            "actual_sha256": actual_joint_cmdm_digest,
            "matched": (
                actual_joint_cmdm_digest == expected_joint_cmdm_digest
                if audited_check_joint else None
            ),
        },
        "cmdm_checkpoint_coverage": coverage,
        "routing_mode": args.routing,
        "plan_metrics": aggregate,
        "per_sample_plan_metrics": plan_rows,
        "routing": {
            "per_sample": routing_by_id,
            "hard_counts": hard_counts.tolist(),
            "soft_load": routing_probabilities.mean(axis=0).tolist(),
            "minimum_confidence": float(routing_probabilities.max(axis=1).min()),
        },
        "motion_mean_loss": mean_losses,
        "motion_per_timestep": per_timestep,
        "predicted_beats_zero_weight": bool(predicted_beats_zero),
        "predicted_beats_legacy_base": bool(predicted_beats_base),
        "predicted_beats_predicted_shuffled": bool(predicted_beats_shuffled),
        "wiring_status": "PASS" if wiring_pass else "CHECK",
        "quality_status": "PASS" if quality_pass else "CHECK",
        "identity_base_tensor_bitwise_equal": bool(
            identity_tensor_parity_checked
        ),
        "quality_note": (
            "Frozen comparison only. If predicted map* does not beat legacy "
            "Base ADM, run the dedicated map* joint fine-tuning stage."
        ),
        "identity_legacy_loss_max_abs_diff": identity_legacy_diff,
        "frozen_state_checks": frozen_checks,
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_predictions:
        np.savez_compressed(
            output_dir / "mapstar_predictions.npz",
            sample_ids=np.asarray(saved_ids),
            predicted_iiw=np.stack(saved_predictions).astype(np.float32),
            target_iiw=np.stack(saved_targets).astype(np.float32),
            pc_weight=np.stack(saved_weights).astype(np.float32),
            mapstar=np.stack(saved_maps).astype(np.float32),
        )

    print("[PASS] pc_weight=1 tensor exactly reproduces cached Base ADM")
    print("[PASS] planner and CMDM stayed frozen")
    print(
        "[PASS] planner checkpoint quality gate: "
        + str(checkpoint_quality_gate["acceptance_path"])
    )
    print("[{}] predicted={:.8f} zero={:.8f} base={:.8f}".format(
        status,
        mean_losses["predicted"],
        mean_losses["zero_weight"],
        mean_losses["legacy_base"],
    ))
    for mode in modes:
        print("[OK] {} grid mean: {:.8f}".format(mode, mean_losses[mode]))
    print("[OK] hard expert counts: " + str(hard_counts.tolist()))
    print("[OK] saved: " + str(summary_file))
    if args.save_predictions:
        print("[OK] saved: " + str(output_dir / "mapstar_predictions.npz"))


if __name__ == "__main__":
    main()

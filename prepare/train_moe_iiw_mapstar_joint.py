#!/usr/bin/env python3
"""Multi-scene MoE-IIW -> pc_weight -> map* -> frozen CMDM training.

This is the first joint-training contract for the literal map-routing design::

    scene + text + state -> MoEIIWPlanner -> native IIW [B,Q,N,6]
    terminal IIW -> max over native bodies -> pc_weight [B,N]
    cached base ADM * pc_weight[...,None] -> map* [B,N,6]
    CMDM(c_pc_contact=map*) -> diffusion motion loss

The cached ADM and the pretrained CMDM are immutable.  Only the MoE router
and residual experts are optimized; the inherited dense IIW trunk is frozen.
Training uses a straight-through hard expert choice, while evaluation uses an
actually hard expert choice.  Supervised IIW and explicit anti-collapse losses
remain active so the six-sample motion overfit cannot silently destroy the
already validated IIW planner.

The same strict map multiplication and frozen-model boundaries apply to both
the original HC6 smoke run and the combined LC17+HC6 production run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models  # noqa: E402,F401
from datasets.history_affordance_v1 import (  # noqa: E402
    HistoryAffordanceV1ContactMotionDataset,
)
from models.base import create_model_and_diffusion  # noqa: E402
from models.functions import encode_text_clip  # noqa: E402
from models.iiw_map_router import IIWMapRouter  # noqa: E402
from models.iiw_planner import (  # noqa: E402
    MoEIIWPlanner,
    build_iiw_planner_from_checkpoint,
)
from prepare.train_iiw_planner import (  # noqa: E402
    ACTIVE_THRESHOLD,
    NUM_BODIES,
    NUM_POINTS,
    TEXT_DIM,
    IIWPlannerDataset,
    assert_finite,
    mean_within_scene_prediction_delta,
    sample_retrieval_metrics,
    sparse_objective,
    tensor_metrics,
)
from prepare.train_moe_iiw_planner import (  # noqa: E402
    assigned_expert_sparse_objective,
    balanced_state_mode_labels,
    router_anti_collapse_objective,
    routing_metrics,
)
from utils.misc import compute_repr_dimesion  # noqa: E402
from utils.training import load_ckpt  # noqa: E402


EXPECTED_EXPERTS = 2
EXPECTED_PHASES = 8
MAX_HORIZON = 196
MOTION_DIM = 66
EVALUATION_TIMESTEPS = (0, 100, 250, 500, 750, 999)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cmdm_config(repo_root: Path):
    default_cfg = OmegaConf.load(repo_root / "configs/default.yaml")
    cfg = OmegaConf.create(
        {"seed": int(default_cfg.seed), "diffusion": default_cfg.diffusion}
    )
    cfg.task = OmegaConf.load(repo_root / "configs/task/contact_motion_gen.yaml")
    cfg.model = OmegaConf.load(repo_root / "configs/model/cmdm.yaml")
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    OmegaConf.resolve(cfg)
    if str(cfg.model.data_repr) != "pos" or int(cfg.model.input_feats) != MOTION_DIM:
        raise ValueError("joint map* training requires CMDM pos with 66 features")
    if int(cfg.task.dataset.num_points) != NUM_POINTS:
        raise ValueError("joint map* training requires exactly 8192 scene points")
    if int(cfg.task.dataset.max_horizon) != MAX_HORIZON:
        raise ValueError("joint map* training requires max_horizon=196")
    return cfg


def parameter_gradient_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return total ** 0.5


def expert_gradient_l2(
    planner: MoEIIWPlanner, num_experts: int,
) -> List[float]:
    """Return one gradient norm for every leading-K expert parameter slice."""
    totals = [0.0 for _ in range(num_experts)]
    found = False
    for name, parameter in planner.named_parameters():
        if not name.startswith("expert_"):
            continue
        found = True
        if parameter.ndim < 1 or int(parameter.shape[0]) != num_experts:
            raise ValueError("expert parameter lacks leading K dimension: " + name)
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        for expert in range(num_experts):
            totals[expert] += float(gradient[expert].square().sum().item())
    if not found:
        raise AttributeError("MoE planner exposes no expert_* parameters")
    return [value ** 0.5 for value in totals]


def expert_parameter_delta_l2(
    planner: MoEIIWPlanner, num_experts: int,
) -> float:
    rows: List[List[torch.Tensor]] = [[] for _ in range(num_experts)]
    for name, parameter in planner.named_parameters():
        if not name.startswith("expert_"):
            continue
        if parameter.ndim < 1 or int(parameter.shape[0]) != num_experts:
            raise ValueError("expert parameter lacks leading K dimension: " + name)
        for expert in range(num_experts):
            rows[expert].append(parameter[expert].detach().float().reshape(-1))
    if any(not row for row in rows):
        raise AttributeError("one or more experts expose no parameters")
    flattened = [torch.cat(row) for row in rows]
    return float((flattened[0] - flattened[1]).norm().item())


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash parameters and buffers without retaining a second model in RAM."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_coverage(
    module: torch.nn.Module, checkpoint_file: Path,
) -> Dict[str, object]:
    """Require every non-text trainable CMDM tensor in the checkpoint."""
    saved = torch.load(str(checkpoint_file), map_location="cpu")
    if not isinstance(saved, Mapping):
        raise TypeError("CMDM checkpoint must be a state-dict mapping")
    current = module.state_dict()
    trainable_names = {
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    external_prefixes = ("text_model.", "clip_model.", "bert_model.")
    loaded = []
    missing_trainable = []
    missing_external = []
    mismatched = []
    for name, value in current.items():
        saved_name = name if name in saved else "module." + name
        if saved_name not in saved:
            if name in trainable_names and not name.startswith(external_prefixes):
                missing_trainable.append(name)
            else:
                missing_external.append(name)
            continue
        saved_value = saved[saved_name]
        if not torch.is_tensor(saved_value) or tuple(saved_value.shape) != tuple(value.shape):
            mismatched.append(name)
            continue
        loaded.append(name)
    if missing_trainable or mismatched:
        raise ValueError(
            "CMDM checkpoint coverage failed: missing_trainable={} "
            "shape_mismatches={}".format(
                len(missing_trainable), len(mismatched)
            )
        )
    return {
        "model_state_tensors": len(current),
        "loaded_shape_matched_tensors": len(loaded),
        "missing_trainable_tensors": missing_trainable,
        "missing_external_or_frozen_tensors": missing_external,
        "shape_mismatches": mismatched,
    }


def straight_through_expert_logits(
    routing: Mapping[str, torch.Tensor],
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use hard experts forward; give the train route a softmax gradient."""
    required = ("expert_logits", "routing_probabilities", "routing_logits")
    missing = [name for name in required if name not in routing]
    if missing:
        raise KeyError("MoE routing result misses: " + ", ".join(missing))
    expert_logits = routing["expert_logits"]
    soft = routing["routing_probabilities"]
    if expert_logits.ndim != 5:
        raise ValueError("expert_logits must have shape [B,K,Q,N,6]")
    if soft.shape != expert_logits.shape[:2]:
        raise ValueError("expert logits and routing probabilities disagree")
    expected_soft = torch.softmax(routing["routing_logits"], dim=-1)
    if not torch.allclose(soft, expected_soft, rtol=1e-5, atol=1e-6):
        raise AssertionError("routing probabilities are not softmax(logits)")
    selected = soft.argmax(dim=-1)
    hard = F.one_hot(selected, num_classes=soft.shape[1]).to(dtype=soft.dtype)
    weights = hard + soft - soft.detach() if training else hard
    mixed = (
        expert_logits * weights[:, :, None, None, None]
    ).sum(dim=1).contiguous()
    return mixed, weights


def cmdm_conditioning(batch: Mapping[str, object], mapstar: torch.Tensor) -> Dict:
    """Return the complete and intentionally minimal frozen-CMDM condition."""
    if mapstar.ndim != 3 or mapstar.shape[-1] != NUM_BODIES:
        raise ValueError("mapstar must have shape [B,N,6]")
    return {
        "x_mask": batch["x_mask"],
        "c_pc_xyz": batch["scene_xyz"],
        "c_pc_contact": mapstar.contiguous(),
        "c_text": batch["texts"],
    }


def validate_moe_checkpoint(
    checkpoint: Mapping[str, object],
    index_names: Sequence[str],
) -> None:
    if str(checkpoint.get("planner_type")) != "moe_iiw_v1":
        raise ValueError("joint trainer requires planner_type=moe_iiw_v1")
    if int(checkpoint.get("format_version", -1)) != 2:
        raise ValueError("joint trainer requires MoE checkpoint format_version=2")
    if str(checkpoint.get("target_method")) != "point_aligned_iiw_proxy":
        raise ValueError("MoE checkpoint IIW target method mismatch")
    if str(checkpoint.get("text_encoder")) != "ViT-B/32":
        raise ValueError("MoE checkpoint text encoder mismatch")
    if int(checkpoint.get("num_phases", -1)) != EXPECTED_PHASES:
        raise ValueError("MoE checkpoint phase count mismatch")
    if int(checkpoint.get("native_body_part_count", -1)) != NUM_BODIES:
        raise ValueError("MoE checkpoint body count mismatch")
    summary = checkpoint.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("MoE checkpoint has no summary mapping")
    if summary.get("status") != "PASS":
        raise ValueError("source MoE-IIW checkpoint status is not PASS")
    if not bool(summary.get("overfit_quality_pass", False)):
        raise ValueError("source MoE-IIW checkpoint failed its overfit contract")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping) or not all(bool(value) for value in checks.values()):
        raise ValueError("source MoE-IIW checkpoint has failed strict checks")
    saved = {Path(str(value)).name for value in checkpoint.get("index_files", [])}
    current = {Path(str(value)).name for value in index_names}
    if saved != current:
        raise ValueError("source MoE-IIW checkpoint index set mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--moe-iiw-checkpoint", type=Path, required=True)
    parser.add_argument("--cmdm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--motion-microbatch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--random-diffusion", action="store_true")
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--iiw-weight", type=float, default=1.0)
    parser.add_argument("--assigned-expert-weight", type=float, default=0.5)
    parser.add_argument("--router-weight", type=float, default=0.25)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--routing-margin", type=float, default=0.20)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    args = parser.parse_args()

    if args.steps <= 0 or args.lr <= 0.0:
        raise ValueError("steps and lr must be positive")
    if args.motion_microbatch <= 0:
        raise ValueError("motion microbatch must be positive")
    if not 0 <= args.timestep < 1000:
        raise ValueError("timestep must be in [0,999]")
    if args.routing_margin < 0.0 or args.max_pos_weight < 1.0:
        raise ValueError("invalid routing margin or positive weight")
    loss_weight_names = (
        "motion_weight", "iiw_weight", "assigned_expert_weight",
        "router_weight", "balance_weight", "entropy_weight",
        "separation_weight", "bce_weight", "foreground_weight",
        "background_weight", "dice_weight", "temporal_weight",
    )
    if any(float(getattr(args, name)) < 0.0 for name in loss_weight_names):
        raise ValueError("all objective weights must be non-negative")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset_root = args.dataset_root.expanduser().resolve()
    moe_file = args.moe_iiw_checkpoint.expanduser().resolve()
    cmdm_file = args.cmdm_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (dataset_root, moe_file, cmdm_file):
        if not path.exists():
            raise FileNotFoundError(path)

    # IIW targets and production motion tensors are independently strict;
    # align them by sample ID before any optimization.
    iiw_dataset = IIWPlannerDataset(dataset_root, args.index)
    motion_cfg = OmegaConf.create({
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "num_points": NUM_POINTS,
        "max_horizon": MAX_HORIZON,
    })
    motion_dataset = HistoryAffordanceV1ContactMotionDataset(
        motion_cfg, phase="train"
    )
    num_samples = len(iiw_dataset)
    if len(motion_dataset) != num_samples or num_samples < EXPECTED_EXPERTS:
        raise ValueError("IIW/motion sample count mismatch or too few samples")
    if args.motion_microbatch > num_samples:
        raise ValueError("motion microbatch exceeds dataset size")
    if int(iiw_dataset.num_phases) != EXPECTED_PHASES:
        raise ValueError("strict joint overfit requires exactly eight IIW phases")
    iiw_rows = [iiw_dataset[index] for index in range(num_samples)]
    motion_rows = [motion_dataset[index] for index in range(num_samples)]
    iiw_ids = [str(row["sample_id"]) for row in iiw_rows]
    motion_ids = [str(row["info_sample_id"]) for row in motion_rows]
    if iiw_ids != motion_ids:
        raise ValueError("IIW and motion datasets have different sample order")
    scene_ids = sorted({str(row["scene_id"]) for row in iiw_rows})
    texts = sorted({str(row["text"]) for row in iiw_rows})
    if not scene_ids or not texts:
        raise ValueError("joint dataset contains no scene or prompt")

    def stack(name: str, rows: Sequence[Mapping], dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.stack([row[name] for row in rows])).to(
            device=device, dtype=dtype
        ).contiguous()

    batch: Dict[str, object] = {
        "x": stack("x", motion_rows, torch.float32),
        "x_mask": stack("x_mask", motion_rows, torch.bool),
        "scene_points": stack("scene_points", iiw_rows, torch.float32),
        "scene_xyz": stack("c_pc_xyz", motion_rows, torch.float32),
        "base_affordance": stack("c_base_affordance", motion_rows, torch.float32),
        "state": stack("state", iiw_rows, torch.float32),
        "target_iiw": stack("target", iiw_rows, torch.float32),
        "texts": [str(row["c_text"]) for row in motion_rows],
        "sample_ids": iiw_ids,
    }
    expected_shapes = {
        "x": (num_samples, MAX_HORIZON, MOTION_DIM),
        "x_mask": (num_samples, MAX_HORIZON),
        "scene_points": (num_samples, NUM_POINTS, 6),
        "scene_xyz": (num_samples, NUM_POINTS, 3),
        "base_affordance": (num_samples, NUM_POINTS, NUM_BODIES),
        "state": (num_samples, 4),
        "target_iiw": (
            num_samples, EXPECTED_PHASES, NUM_POINTS, NUM_BODIES
        ),
    }
    for name, shape in expected_shapes.items():
        value = batch[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(name + " shape mismatch")
        if value.dtype != torch.bool:
            assert_finite(name, value)
    # The production and planner loaders must expose identical scene ordering.
    if not torch.equal(
        batch["scene_points"][:, :, :3], batch["scene_xyz"]
    ):
        raise ValueError("IIW and CMDM scene point order differs")
    for scene_id in scene_ids:
        indices = [
            index for index, row in enumerate(iiw_rows)
            if str(row["scene_id"]) == scene_id
        ]
        scene_base = batch["base_affordance"].index_select(
            0, torch.as_tensor(indices, device=device)
        )
        if not torch.equal(scene_base, scene_base[0:1].expand_as(scene_base)):
            raise ValueError(scene_id + ": samples do not share cached Base ADM")
    base_hash_before = hashlib.sha256(
        batch["base_affordance"].detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

    checkpoint = torch.load(str(moe_file), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MoE-IIW checkpoint must be a mapping")
    validate_moe_checkpoint(checkpoint, args.index)
    planner = build_iiw_planner_from_checkpoint(
        checkpoint, device=device, strict=True
    )
    if not isinstance(planner, MoEIIWPlanner):
        raise TypeError("source checkpoint did not build MoEIIWPlanner")
    if int(planner.num_experts) != EXPECTED_EXPERTS:
        raise ValueError("joint training requires two experts")

    # Reconstruct and verify the exact state-only labels used by the PASS
    # source checkpoint.  No motion or IIW target participates in this step.
    states_np = batch["state"].detach().cpu().numpy()
    labels_np, mode_metadata = balanced_state_mode_labels(
        states_np, EXPECTED_EXPERTS
    )
    source_summary = checkpoint["summary"]
    source_ids = [str(value) for value in source_summary.get("sample_ids", [])]
    source_labels = np.asarray(
        source_summary.get("mode_labels_in_sample_order", []), dtype=np.int64
    )
    if source_ids != iiw_ids or not np.array_equal(source_labels, labels_np):
        raise ValueError("source MoE mode labels do not match current data")
    mode_labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)

    # Freeze the inherited dense IIW trunk.  Only explicit router/expert names
    # may remain trainable in this strict stage.
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    router_parameters: List[torch.nn.Parameter] = []
    expert_parameters: List[torch.nn.Parameter] = []
    for name, parameter in planner.named_parameters():
        if name.startswith("router."):
            parameter.requires_grad_(True)
            router_parameters.append(parameter)
        elif name.startswith("expert_"):
            parameter.requires_grad_(True)
            expert_parameters.append(parameter)
    if not router_parameters or not expert_parameters:
        raise RuntimeError("MoE checkpoint exposes no router/expert parameters")
    trainable = router_parameters + expert_parameters
    trainable_names = [
        name for name, value in planner.named_parameters() if value.requires_grad
    ]
    if any(
        not (name.startswith("router.") or name.startswith("expert_"))
        for name in trainable_names
    ):
        raise AssertionError("non-MoE planner parameter unexpectedly trainable")

    cfg = load_cmdm_config(REPO_ROOT)
    cmdm, diffusion = create_model_and_diffusion(cfg, device=str(device))
    cmdm.to(device)
    cmdm_checkpoint_coverage = checkpoint_coverage(cmdm, cmdm_file)
    load_ckpt(cmdm, str(cmdm_file))
    cmdm.eval()
    for parameter in cmdm.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in cmdm.parameters()):
        raise AssertionError("CMDM did not freeze completely")
    cmdm_hash_before = module_state_sha256(cmdm)
    planner_dense_hash_before = hashlib.sha256()
    for name, value in sorted(planner.state_dict().items()):
        if name.startswith("router.") or name.startswith("expert_"):
            continue
        planner_dense_hash_before.update(name.encode("utf-8"))
        planner_dense_hash_before.update(
            value.detach().cpu().contiguous().numpy().tobytes()
        )
    planner_dense_hash_before = planner_dense_hash_before.hexdigest()

    unique_texts = sorted(set(batch["texts"]))
    with torch.no_grad():
        encoded = encode_text_clip(
            cmdm.text_model, unique_texts,
            max_length=cmdm.text_max_length, device=str(device),
        ).detach().float()
    text_cache = {text: encoded[index] for index, text in enumerate(unique_texts)}
    text_features = torch.stack(
        [text_cache[text] for text in batch["texts"]], dim=0
    ).contiguous()
    if text_features.shape != (num_samples, TEXT_DIM):
        raise ValueError("CLIP text feature shape mismatch")

    map_router = IIWMapRouter(validate_inputs=True).to(device)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    positive_weight = (
        (1.0 - torch.from_numpy(iiw_dataset.target_soft_mean))
        / torch.from_numpy(iiw_dataset.target_soft_mean).clamp_min(1e-5)
    ).clamp(1.0, args.max_pos_weight).to(device)
    sparse_weights = {
        "bce": args.bce_weight,
        "foreground": args.foreground_weight,
        "background": args.background_weight,
        "dice": args.dice_weight,
        "temporal": args.temporal_weight,
    }
    anti_weights = {
        "router_supervised_ce": args.router_weight,
        "router_load_balance": args.balance_weight,
        "router_confidence_entropy": args.entropy_weight,
        "router_separation_margin": args.separation_weight,
    }

    def planner_forward(indices: torch.Tensor, training_route: bool) -> Dict:
        routing = planner.forward_with_routing(
            batch["scene_points"].index_select(0, indices),
            text_features.index_select(0, indices),
            batch["state"].index_select(0, indices),
            return_expert_logits=True,
        )
        logits, route_weights = straight_through_expert_logits(
            routing, training=training_route
        )
        prediction = torch.sigmoid(logits)
        routed = map_router(
            batch["base_affordance"].index_select(0, indices), prediction
        )
        exact = (
            batch["base_affordance"].index_select(0, indices).detach()
            * routed["pc_weight"].unsqueeze(-1)
        )
        if not torch.equal(routed["mapstar"], exact):
            raise AssertionError("map* is not exact cached-base times pc_weight")
        return {
            "routing": routing,
            "logits": logits,
            "prediction": prediction,
            "route_weights": route_weights,
            "map_route": routed,
        }

    def supervised_forward(training_route: bool) -> Dict:
        indices = torch.arange(num_samples, device=device)
        result = planner_forward(indices, training_route)
        routing = result["routing"]
        iiw_total, iiw_parts = sparse_objective(
            result["logits"], batch["target_iiw"],
            positive_weight, sparse_weights,
        )
        expert_total, expert_parts = assigned_expert_sparse_objective(
            routing["expert_logits"], mode_labels, batch["target_iiw"],
            positive_weight, sparse_weights,
        )
        anti_parts, _ = router_anti_collapse_objective(
            routing["routing_logits"], routing["routing_probabilities"],
            mode_labels, args.routing_margin,
        )
        auxiliary = args.iiw_weight * iiw_total
        auxiliary = auxiliary + args.assigned_expert_weight * expert_total
        for name, weight in anti_weights.items():
            auxiliary = auxiliary + float(weight) * anti_parts[name]
        return {
            **result,
            "iiw_total": iiw_total,
            "iiw_parts": iiw_parts,
            "expert_total": expert_total,
            "expert_parts": expert_parts,
            "anti_parts": anti_parts,
            "auxiliary_loss": auxiliary,
        }

    def diffusion_inputs(randomize: bool, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        set_seed(seed)
        if randomize:
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (num_samples,), device=device
            )
        else:
            timesteps = torch.full(
                (num_samples,), args.timestep,
                dtype=torch.long, device=device,
            )
        noise = torch.randn_like(batch["x"])
        return timesteps, noise

    def motion_chunk(
        start: int,
        stop: int,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        training_route: bool,
    ) -> Tuple[torch.Tensor, Dict]:
        indices = torch.arange(start, stop, device=device)
        result = planner_forward(indices, training_route)
        local_batch = {
            "x_mask": batch["x_mask"].index_select(0, indices),
            "scene_xyz": batch["scene_xyz"].index_select(0, indices),
            "texts": [batch["texts"][index] for index in range(start, stop)],
        }
        terms = diffusion.training_losses(
            cmdm,
            batch["x"].index_select(0, indices),
            timesteps.index_select(0, indices),
            model_kwargs=cmdm_conditioning(
                local_batch, result["map_route"]["mapstar"]
            ),
            noise=noise.index_select(0, indices),
        )
        loss = terms["loss"].mean()
        assert_finite("CMDM motion loss", loss)
        return loss, result

    def backward_motion_only(seed: int) -> Dict:
        planner.train()
        optimizer.zero_grad(set_to_none=True)
        timesteps, noise = diffusion_inputs(False, seed)
        total_value = 0.0
        for start in range(0, num_samples, args.motion_microbatch):
            stop = min(start + args.motion_microbatch, num_samples)
            loss, _ = motion_chunk(
                start, stop, timesteps, noise, training_route=True
            )
            scale = float(stop - start) / float(num_samples)
            (loss * scale).backward()
            total_value += float(loss.detach().item()) * scale
        router_grad = parameter_gradient_l2(router_parameters)
        expert_grad = parameter_gradient_l2(expert_parameters)
        per_expert = expert_gradient_l2(planner, EXPECTED_EXPERTS)
        values = [router_grad, expert_grad] + per_expert
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise AssertionError(
                "motion-only loss failed to reach router and every expert: "
                + str(values)
            )
        optimizer.zero_grad(set_to_none=True)
        return {
            "motion_loss": total_value,
            "router_gradient_l2": router_grad,
            "expert_gradient_l2": expert_grad,
            "per_expert_gradient_l2": per_expert,
        }

    @torch.no_grad()
    def evaluate(timestep_grid: Sequence[int]) -> Tuple[Dict, Dict]:
        planner.eval()
        full = supervised_forward(training_route=False)
        prediction = full["prediction"]
        metrics = tensor_metrics(
            prediction.cpu(), batch["target_iiw"].cpu(), ACTIVE_THRESHOLD
        )
        row_scene_ids = [str(row["scene_id"]) for row in iiw_rows]
        metrics.update(sample_retrieval_metrics(
            prediction.cpu(), batch["target_iiw"].cpu(), row_scene_ids
        ))
        metrics["mean_pairwise_prediction_delta"] = (
            mean_within_scene_prediction_delta(
                prediction.cpu(), row_scene_ids
            )
        )
        metrics.update({
            "iiw_" + key: float(value.item())
            for key, value in full["iiw_parts"].items()
        })
        metrics["assigned_expert_sparse_total"] = float(
            full["expert_total"].item()
        )
        metrics["auxiliary_loss"] = float(full["auxiliary_loss"].item())
        route_metrics = routing_metrics(
            full["routing"]["routing_logits"],
            full["routing"]["routing_probabilities"], mode_labels,
        )
        motion_losses = []
        per_timestep = []
        for timestep in timestep_grid:
            set_seed(args.seed + 10000 + int(timestep))
            timesteps = torch.full(
                (num_samples,), int(timestep),
                dtype=torch.long, device=device,
            )
            noise = torch.randn_like(batch["x"])
            values = []
            for start in range(0, num_samples, args.motion_microbatch):
                stop = min(start + args.motion_microbatch, num_samples)
                indices = torch.arange(start, stop, device=device)
                local_batch = {
                    "x_mask": batch["x_mask"].index_select(0, indices),
                    "scene_xyz": batch["scene_xyz"].index_select(0, indices),
                    "texts": [batch["texts"][index] for index in range(start, stop)],
                }
                terms = diffusion.training_losses(
                    cmdm,
                    batch["x"].index_select(0, indices),
                    timesteps.index_select(0, indices),
                    model_kwargs=cmdm_conditioning(
                        local_batch,
                        full["map_route"]["mapstar"].index_select(0, indices),
                    ),
                    noise=noise.index_select(0, indices),
                )
                values.extend(terms["loss"].detach().cpu().reshape(-1).tolist())
            mean_value = float(np.mean(values))
            motion_losses.append(mean_value)
            per_timestep.append({"timestep": int(timestep), "motion_loss": mean_value})
        routed = full["map_route"]
        expert_prediction = torch.sigmoid(full["routing"]["expert_logits"])
        expert_output_delta = float(
            (expert_prediction[:, 0] - expert_prediction[:, 1])
            .abs().mean().item()
        )
        output = {
            **metrics,
            "routing": route_metrics,
            "motion_grid": {
                "timesteps": [int(value) for value in timestep_grid],
                "motion_losses": motion_losses,
                "mean_motion_loss": float(np.mean(motion_losses)),
                "per_timestep": per_timestep,
            },
            "pc_weight_min": float(routed["pc_weight"].min().item()),
            "pc_weight_max": float(routed["pc_weight"].max().item()),
            "mapstar_min": float(routed["mapstar"].min().item()),
            "mapstar_max": float(routed["mapstar"].max().item()),
            "expert_parameter_delta_l2": expert_parameter_delta_l2(
                planner, EXPECTED_EXPERTS
            ),
            "expert_output_delta": expert_output_delta,
        }
        arrays = {
            "iiw_prediction": prediction.cpu().numpy(),
            "pc_weight": routed["pc_weight"].cpu().numpy(),
            "mapstar": routed["mapstar"].cpu().numpy(),
            "phase_pc_weight": routed["phase_pc_weight"].cpu().numpy(),
            "phase_mapstar": routed["phase_mapstar"].cpu().numpy(),
            "routing_probabilities": full["routing"][
                "routing_probabilities"
            ].cpu().numpy(),
            "selected_expert": full["routing"]["selected_expert"].cpu().numpy(),
        }
        return output, arrays

    # The initial grid and the motion-only gradient probe intentionally use
    # the same deterministic per-timestep noise as final evaluation.
    initial, _ = evaluate(EVALUATION_TIMESTEPS)
    motion_gradient_seed = args.seed + 10000 + args.timestep
    motion_gradient = backward_motion_only(motion_gradient_seed)
    print(
        "[PASS] motion-only map* loss reaches router and both experts: "
        + json.dumps(motion_gradient)
    )

    optimization_records = []
    for step in range(1, args.steps + 1):
        planner.train()
        optimizer.zero_grad(set_to_none=True)
        supervised = supervised_forward(training_route=True)
        supervised["auxiliary_loss"].backward()

        timesteps, noise = diffusion_inputs(
            args.random_diffusion, args.seed + 30000 + step
        )
        motion_value = 0.0
        for start in range(0, num_samples, args.motion_microbatch):
            stop = min(start + args.motion_microbatch, num_samples)
            loss, _ = motion_chunk(
                start, stop, timesteps, noise, training_route=True
            )
            scale = float(stop - start) / float(num_samples)
            (args.motion_weight * loss * scale).backward()
            motion_value += float(loss.detach().item()) * scale

        router_grad = parameter_gradient_l2(router_parameters)
        expert_grad = parameter_gradient_l2(expert_parameters)
        per_expert_grad = expert_gradient_l2(planner, EXPECTED_EXPERTS)
        all_gradients = [router_grad, expert_grad] + per_expert_grad
        if not all(math.isfinite(value) and value > 0.0 for value in all_gradients):
            raise AssertionError("joint gradients collapsed: " + str(all_gradients))
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        route_metrics = routing_metrics(
            supervised["routing"]["routing_logits"].detach(),
            supervised["routing"]["routing_probabilities"].detach(),
            mode_labels,
        )
        total_value = (
            args.motion_weight * motion_value
            + float(supervised["auxiliary_loss"].detach().item())
        )
        record = {
            "step": step,
            "total_loss": total_value,
            "motion_loss": motion_value,
            "auxiliary_loss": float(supervised["auxiliary_loss"].detach().item()),
            "iiw_loss": float(supervised["iiw_total"].detach().item()),
            "router_gradient_l2": router_grad,
            "expert_gradient_l2": expert_grad,
            "per_expert_gradient_l2": per_expert_grad,
            "routing_accuracy": route_metrics["accuracy"],
            "routing_hard_counts": route_metrics["hard_counts"],
            "routing_soft_load": route_metrics["soft_load"],
            "timesteps": timesteps.detach().cpu().tolist(),
        }
        optimization_records.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(record))

    final, arrays = evaluate(EVALUATION_TIMESTEPS)
    cmdm_hash_after = module_state_sha256(cmdm)
    cmdm_unchanged = cmdm_hash_before == cmdm_hash_after
    planner_dense_digest = hashlib.sha256()
    for name, value in sorted(planner.state_dict().items()):
        if name.startswith("router.") or name.startswith("expert_"):
            continue
        planner_dense_digest.update(name.encode("utf-8"))
        planner_dense_digest.update(
            value.detach().cpu().contiguous().numpy().tobytes()
        )
    planner_dense_hash_after = planner_dense_digest.hexdigest()
    base_hash_after = hashlib.sha256(
        batch["base_affordance"].detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    checks = {
        "source_moe_checkpoint_passed": True,
        "cached_base_bitwise_unchanged": base_hash_before == base_hash_after,
        "cmdm_bitwise_unchanged": cmdm_unchanged,
        "dense_iiw_trunk_bitwise_unchanged": (
            planner_dense_hash_before == planner_dense_hash_after
        ),
        "motion_only_router_gradient_nonzero": (
            motion_gradient["router_gradient_l2"] > 0.0
        ),
        "motion_only_both_expert_gradients_nonzero": all(
            value > 0.0 for value in motion_gradient["per_expert_gradient_l2"]
        ),
        "hard_eval_routing_perfect": final["routing"]["accuracy"] == 1.0,
        "both_experts_hard_selected": all(
            count > 0 for count in final["routing"]["hard_counts"]
        ),
        "both_experts_soft_loaded": min(final["routing"]["soft_load"]) >= 0.20,
        "router_confident": (
            final["routing"]["mean_assigned_probability"] >= 0.60
        ),
        "experts_remain_distinct": (
            final["expert_parameter_delta_l2"] > 1e-8
            and final["expert_output_delta"] > 1e-8
        ),
        "iiw_f1_preserved": final["f1_at_0_7"] >= 0.50,
        "iiw_temporal_order_preserved": final["order_margin"] > 0.0,
        "iiw_objective_not_degraded": (
            final["iiw_total"] <= initial["iiw_total"] + 1e-6
        ),
        "motion_grid_improved": (
            final["motion_grid"]["mean_motion_loss"]
            < initial["motion_grid"]["mean_motion_loss"]
        ),
        "pc_weight_in_unit_interval": (
            0.0 <= final["pc_weight_min"]
            and final["pc_weight_max"] <= 1.0
        ),
        "mapstar_in_unit_interval": (
            0.0 <= final["mapstar_min"] and final["mapstar_max"] <= 1.0
        ),
    }
    status = "PASS" if all(checks.values()) else "CHECK"
    for name, passed in checks.items():
        print(("[PASS] " if passed else "[CHECK] ") + name)

    summary = {
        "status": status,
        "scope": "multi-scene MoE-IIW map* frozen-CMDM training",
        "held_out_generalization": False,
        "scientific_claim_limit": (
            "training-set fit and two-state-mode specialization; held-out "
            "generalization requires a separate split"
        ),
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "scene_ids": scene_ids,
        "texts": texts,
        "sample_ids": iiw_ids,
        "num_samples": num_samples,
        "num_experts": EXPECTED_EXPERTS,
        "num_phases": EXPECTED_PHASES,
        "num_points": NUM_POINTS,
        "num_bodies": NUM_BODIES,
        "map_definition": "mapstar = cached_base_affordance * pc_weight[...,None]",
        "pc_weight_definition": (
            "terminal predicted native IIW phase, max over six native bodies"
        ),
        "planner_inputs": ["scene_points", "CLIP_text_feature", "state"],
        "cmdm_conditioning_keys": ["x_mask", "c_pc_xyz", "c_pc_contact", "c_text"],
        "cmdm_contact_value": "mapstar",
        "cached_base_detached": True,
        "base_adm_recomputed_or_modified": False,
        "cmdm_frozen": True,
        "cmdm_state_sha256_before": cmdm_hash_before,
        "cmdm_state_sha256_after": cmdm_hash_after,
        "cmdm_checkpoint_coverage": cmdm_checkpoint_coverage,
        "cached_base_sha256_before": base_hash_before,
        "cached_base_sha256_after": base_hash_after,
        "dense_iiw_trunk_sha256_before": planner_dense_hash_before,
        "dense_iiw_trunk_sha256_after": planner_dense_hash_after,
        "dense_iiw_trunk_frozen": True,
        "iiw_adapter_used": False,
        "embedding_residual_used": False,
        "terminal_phase_index": EXPECTED_PHASES - 1,
        "body_reduction": "max_over_six_native_iiw_channels",
        "pc_weight_broadcast_over_cmdm_contact_channels": True,
        "trainable_parameter_names": trainable_names,
        "train_routing_forward": "straight_through_hard_expert",
        "eval_routing_forward": "hard_expert",
        "source_moe_iiw_checkpoint": str(moe_file),
        "source_cmdm_checkpoint": str(cmdm_file),
        "source_planner_type": checkpoint["planner_type"],
        "source_checkpoint_format_version": checkpoint["format_version"],
        "state_mode_metadata": mode_metadata,
        "mode_labels_in_sample_order": labels_np.tolist(),
        "steps": args.steps,
        "logical_full_batch_size": num_samples,
        "motion_microbatch": args.motion_microbatch,
        "random_diffusion": bool(args.random_diffusion),
        "fixed_evaluation_timesteps": list(EVALUATION_TIMESTEPS),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "objective_weights": {
            "motion": args.motion_weight,
            "iiw": args.iiw_weight,
            "assigned_expert": args.assigned_expert_weight,
            **anti_weights,
            **{"sparse_" + key: value for key, value in sparse_weights.items()},
        },
        "positive_weight_per_body": positive_weight.detach().cpu().tolist(),
        "motion_only_gradient_contract": motion_gradient,
        "motion_only_gradient_timestep": args.timestep,
        "motion_only_gradient_noise_seed": motion_gradient_seed,
        "initial": initial,
        "final": final,
        "checks": checks,
        "overfit_quality_pass": bool(all(checks.values())),
        "optimization_records": optimization_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    checkpoint_file = output_dir / "moe_iiw_mapstar_joint.pt"
    result_file = output_dir / "moe_iiw_mapstar_joint_results.npz"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({
        "model": planner.state_dict(),
        "iiw_planner_state_dict": planner.state_dict(),
        "planner_type": "moe_iiw_v1",
        "model_type": "moe_iiw_mapstar_joint",
        "joint_type": "moe_iiw_mapstar_joint_v1",
        # Keep the planner sub-checkpoint loadable by the existing v2 factory.
        "format_version": 2,
        "joint_format_version": 1,
        "model_config": checkpoint["model_config"],
        "state_mean": checkpoint["state_mean"],
        "state_std": checkpoint["state_std"],
        "num_experts": EXPECTED_EXPERTS,
        "num_phases": EXPECTED_PHASES,
        "native_body_part_count": NUM_BODIES,
        "text_encoder": "ViT-B/32",
        "target_method": "point_aligned_iiw_proxy",
        "index_files": list(args.index),
        "source_moe_iiw_checkpoint": str(moe_file),
        "source_cmdm_checkpoint": str(cmdm_file),
        "cmdm_frozen": True,
        "cmdm_state_sha256": cmdm_hash_after,
        "mapstar_definition": (
            "base_affordance.detach() * "
            "max_native_body(predicted_iiw[:,7])[...,None]"
        ),
        "iiw_adapter_used": False,
        "embedding_residual_used": False,
        "optimizer_state_dict": optimizer.state_dict(),
        "summary": summary,
    }, checkpoint_file)
    np.savez_compressed(
        result_file,
        iiw_prediction=arrays["iiw_prediction"].astype(np.float32),
        pc_weight=arrays["pc_weight"].astype(np.float32),
        mapstar=arrays["mapstar"].astype(np.float32),
        phase_pc_weight=arrays["phase_pc_weight"].astype(np.float32),
        phase_mapstar=arrays["phase_mapstar"].astype(np.float32),
        routing_probabilities=arrays["routing_probabilities"].astype(np.float32),
        selected_expert=arrays["selected_expert"].astype(np.int64),
        mode_labels=labels_np,
        sample_ids=np.asarray(iiw_ids),
    )
    print(
        "[PASS] multi-scene MoE-IIW map* joint training"
        if status == "PASS"
        else "[CHECK] joint training finished; inspect failed strict checks"
    )
    print("[OK] initial motion grid: {:.8f}".format(
        initial["motion_grid"]["mean_motion_loss"]
    ))
    print("[OK] final motion grid: {:.8f}".format(
        final["motion_grid"]["mean_motion_loss"]
    ))
    for path in (summary_file, checkpoint_file, result_file):
        print("[OK] saved: " + str(path))


if __name__ == "__main__":
    main()

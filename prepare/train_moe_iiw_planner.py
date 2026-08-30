#!/usr/bin/env python3
"""State-routed MoE IIW planner for one or more production scenes.

This script deliberately changes only the learned ``scene + text + state ->
IIW`` path that follows the cached base ADM.  The cached ADM, IIW GT, frozen
CLIP encoder, IIWAdapter, and CMDM are not modified here.

Two deterministic, near-balanced state modes are constructed from the four
deployable state values only; neither GT motion nor GT IIW participates in
mode construction.  Six-sample HC runs retain the original exhaustive 3/3
contract.  Larger LC+HC runs use deterministic capacity-constrained two-means
and may differ by at most one sample (for 23 samples this is 12/11).

This is a wiring/specialization test, not evidence of held-out generalization.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.iiw_planner import IIWPlanner, MoEIIWPlanner  # noqa: E402
from prepare.train_iiw_planner import (  # noqa: E402
    ACTIVE_THRESHOLD,
    NUM_BODIES,
    NUM_POINTS,
    STATE_DIM,
    TEXT_DIM,
    IIWPlannerDataset,
    assert_finite,
    gradient_l2,
    mean_within_scene_prediction_delta,
    sample_retrieval_metrics,
    set_seed,
    sparse_objective,
    tensor_metrics,
)


EXPECTED_SAMPLES = 6
EXPECTED_EXPERTS = 2
EPS = 1e-8


def strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a plain state dict, accepting an all-DDP ``module.`` prefix."""
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint state dict must be a non-empty mapping")
    keys = [str(key) for key in state]
    if all(key.startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state.items()}
    if any(key.startswith("module.") for key in keys):
        raise ValueError("checkpoint has a partial/inconsistent module. prefix")
    return {str(key): value for key, value in state.items()}


def balanced_state_mode_labels(
    states: np.ndarray,
    num_experts: int = EXPECTED_EXPERTS,
) -> Tuple[np.ndarray, Dict]:
    """Cluster state vectors into deterministic near-balanced two modes.

    For K=2 this exhaustively minimizes standardized within-cluster SSE.  The
    first sample is fixed in mode 0 to remove the label-swap ambiguity, and
    lexicographic combination order resolves exact floating-point ties.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != STATE_DIM:
        raise ValueError("states must have shape [B,4]")
    if values.shape[0] < num_experts:
        raise ValueError("number of states must be at least num_experts")
    if num_experts != EXPECTED_EXPERTS:
        raise ValueError("current strict MoE contract requires exactly 2 experts")
    if not np.isfinite(values).all():
        raise ValueError("states contain NaN/Inf")

    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-4)
    normalized = (values - mean) / std
    count = values.shape[0]
    first_size = count // 2
    indices = tuple(range(count))
    best = None
    if count == EXPECTED_SAMPLES:
        # Preserve the exact original HC6 label contract bit-for-bit.
        for first_group in itertools.combinations(indices, first_size):
            if 0 not in first_group:
                continue
            second_group = tuple(
                index for index in indices if index not in first_group
            )
            groups = (tuple(first_group), second_group)
            score = 0.0
            for group in groups:
                rows = normalized[list(group)]
                score += float(np.square(rows - rows.mean(axis=0)).sum())
            candidate = (score, groups)
            if best is None or candidate < best:
                best = candidate
    else:
        # Capacity-constrained deterministic two-means.  Given two centers,
        # sorting the per-sample cost difference gives the optimal assignment
        # for the fixed first-cluster capacity.
        covariance = normalized.T @ normalized
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
        if len(nonzero) and axis[nonzero[0]] < 0.0:
            axis = -axis
        order = np.lexsort((np.arange(count), normalized @ axis))
        first = np.asarray(order[:first_size], dtype=np.int64)
        for _ in range(100):
            second_mask = np.ones(count, dtype=bool)
            second_mask[first] = False
            second = np.flatnonzero(second_mask)
            center0 = normalized[first].mean(axis=0)
            center1 = normalized[second].mean(axis=0)
            cost0 = np.square(normalized - center0).sum(axis=1)
            cost1 = np.square(normalized - center1).sum(axis=1)
            new_first = np.lexsort((np.arange(count), cost0 - cost1))[
                :first_size
            ]
            new_first = np.sort(new_first)
            if np.array_equal(new_first, np.sort(first)):
                first = new_first
                break
            first = new_first
        second_mask = np.ones(count, dtype=bool)
        second_mask[first] = False
        groups = (
            tuple(first.tolist()),
            tuple(np.flatnonzero(second_mask).tolist()),
        )
        # Remove label-swap ambiguity without using any target information.
        if 0 not in groups[0]:
            groups = (groups[1], groups[0])
        score = sum(
            float(np.square(normalized[list(group)] - normalized[list(group)].mean(axis=0)).sum())
            for group in groups
        )
        best = (score, groups)
    if best is None:
        raise RuntimeError("failed to construct balanced state modes")

    labels = np.empty((values.shape[0],), dtype=np.int64)
    for label, group in enumerate(best[1]):
        labels[list(group)] = label
    counts = np.bincount(labels, minlength=num_experts)
    if int(counts.max() - counts.min()) > 1 or int(counts.sum()) != count:
        raise AssertionError("state-mode labels are not near-balanced")
    centers = np.stack(
        [values[labels == label].mean(axis=0) for label in range(num_experts)]
    )
    normalized_centers = np.stack(
        [normalized[labels == label].mean(axis=0) for label in range(num_experts)]
    )
    return labels, {
        "method": (
            "exact_balanced_standardized_state_sse"
            if count == EXPECTED_SAMPLES
            else "deterministic_capacity_constrained_state_two_means"
        ),
        "uses_gt_motion": False,
        "uses_gt_iiw": False,
        "state_mean": mean.astype(np.float32).tolist(),
        "state_std": std.astype(np.float32).tolist(),
        "within_cluster_sse": float(best[0]),
        "counts": counts.tolist(),
        "groups": [list(group) for group in best[1]],
        "centers": centers.astype(np.float32).tolist(),
        "normalized_centers": normalized_centers.astype(np.float32).tolist(),
    }


def router_anti_collapse_objective(
    router_logits: torch.Tensor,
    routing_probabilities: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return supervised routing plus differentiable anti-collapse terms."""
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [B,K]")
    if routing_probabilities.shape != router_logits.shape:
        raise ValueError("router probabilities/logits shape mismatch")
    if labels.shape != (router_logits.shape[0],):
        raise ValueError("mode labels must have shape [B]")
    if router_logits.shape[1] != EXPECTED_EXPERTS:
        raise ValueError("strict contract expects two routing experts")
    assert_finite("router logits", router_logits)
    assert_finite("routing probabilities", routing_probabilities)
    expected_probabilities = torch.softmax(router_logits, dim=-1)
    if not torch.allclose(
        routing_probabilities, expected_probabilities, rtol=1e-5, atol=1e-6
    ):
        raise AssertionError("routing probabilities must equal softmax(logits)")
    if bool(((labels < 0) | (labels >= router_logits.shape[1])).any().item()):
        raise ValueError("mode labels outside expert range")
    counts = torch.bincount(labels, minlength=router_logits.shape[1])
    if int((counts.max() - counts.min()).item()) > 1:
        raise AssertionError("router labels are not near-balanced")

    supervised = F.cross_entropy(router_logits, labels)
    load = routing_probabilities.mean(dim=0)
    uniform = torch.full_like(load, 1.0 / float(load.numel()))
    balance = F.mse_loss(load, uniform)
    entropy_per_sample = -(
        routing_probabilities
        * torch.log(routing_probabilities.clamp_min(EPS))
    ).sum(dim=-1)
    entropy = entropy_per_sample.mean() / math.log(float(load.numel()))
    assigned = routing_probabilities.gather(1, labels[:, None]).squeeze(1)
    masked_other = routing_probabilities.masked_fill(
        F.one_hot(labels, num_classes=load.numel()).bool(), -1.0
    ).max(dim=1).values
    separation = F.relu(float(margin) - (assigned - masked_other)).mean()
    return {
        "router_supervised_ce": supervised,
        "router_load_balance": balance,
        "router_confidence_entropy": entropy,
        "router_separation_margin": separation,
    }, {
        "soft_load": load,
        "normalized_entropy": entropy,
        "assigned_probability": assigned,
        "selected_expert": routing_probabilities.argmax(dim=-1),
    }


def routing_metrics(
    router_logits: torch.Tensor,
    routing_probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> Dict:
    """Compute hard/soft use metrics without hiding expert collapse."""
    with torch.no_grad():
        terms, diagnostics = router_anti_collapse_objective(
            router_logits, routing_probabilities, labels, margin=0.0
        )
        selected = diagnostics["selected_expert"]
        num_experts = router_logits.shape[1]
        confusion = torch.zeros(num_experts, num_experts, dtype=torch.long)
        for expected, predicted in zip(labels.cpu(), selected.cpu()):
            confusion[int(expected), int(predicted)] += 1
        hard_counts = torch.bincount(
            selected, minlength=num_experts
        ).detach().cpu()
        assigned = diagnostics["assigned_probability"]
        return {
            "accuracy": float((selected == labels).float().mean().item()),
            "hard_counts": hard_counts.tolist(),
            "soft_load": diagnostics["soft_load"].detach().cpu().tolist(),
            "mean_assigned_probability": float(assigned.mean().item()),
            "min_assigned_probability": float(assigned.min().item()),
            "normalized_entropy": float(
                diagnostics["normalized_entropy"].item()
            ),
            "confusion_expected_rows_predicted_columns": confusion.tolist(),
            "supervised_ce": float(terms["router_supervised_ce"].item()),
        }


def parameter_l2(parameters: Iterable[torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for value in parameters:
        tensor = value.detach().float()
        total += float(tensor.square().sum().item())
        count += tensor.numel()
    if count == 0:
        raise ValueError("expert parameter collection is empty")
    return total ** 0.5


def expert_parameter_diagnostics(planner: MoEIIWPlanner) -> Dict:
    """Report residual magnitude and pairwise divergence for two experts."""
    named = list(planner.named_parameters())
    expert_tensors = []
    for expert in range(EXPECTED_EXPERTS):
        rows = []
        for name, value in named:
            if not name.startswith("expert_"):
                continue
            if value.ndim < 1 or value.shape[0] != EXPECTED_EXPERTS:
                raise ValueError(
                    "expert parameter must use leading K dimension: " + name
                )
            rows.append(value[expert].reshape(-1))
        if not rows:
            raise AttributeError("MoEIIWPlanner exposes no expert_* parameters")
        expert_tensors.append(torch.cat(rows))
    difference = expert_tensors[0] - expert_tensors[1]
    return {
        "expert_parameter_l2": [
            float(value.detach().float().norm().item())
            for value in expert_tensors
        ],
        "expert_pairwise_parameter_delta_l2": float(
            difference.detach().float().norm().item()
        ),
    }


def assigned_expert_sparse_objective(
    expert_logits: torch.Tensor,
    labels: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
    sparse_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervise the expert assigned to each deterministic state mode.

    Mixture-only supervision can let both experts co-adapt to the same output.
    Selecting one full expert logit tensor per sample supplies an explicit
    specialization gradient while preserving the deployment mixture output.
    """
    if expert_logits.ndim != 5:
        raise ValueError("expert_logits must have shape [B,K,Q,N,C]")
    batch_size, num_experts = expert_logits.shape[:2]
    if labels.shape != (batch_size,) or target.shape[0] != batch_size:
        raise ValueError("assigned-expert batch contract mismatch")
    if tuple(expert_logits.shape[2:]) != tuple(target.shape[1:]):
        raise ValueError("expert/target IIW shape mismatch")
    selected = expert_logits[
        torch.arange(batch_size, device=expert_logits.device), labels
    ]
    total, parts = sparse_objective(
        selected, target, positive_weight, sparse_weights
    )
    per_expert = {}
    for expert in range(num_experts):
        mask = labels == expert
        if not bool(mask.any().item()):
            raise AssertionError("one expert has no assigned sample")
        value, _ = sparse_objective(
            selected[mask], target[mask], positive_weight, sparse_weights
        )
        per_expert["expert_{}_assigned_sparse_total".format(expert)] = value
    return total, per_expert


def main() -> None:
    from models.functions import encode_text_clip, load_and_freeze_clip_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--residual-rank", type=int, default=16)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--routing-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--foreground-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--router-weight", type=float, default=0.25)
    parser.add_argument("--assigned-expert-weight", type=float, default=0.5)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--routing-margin", type=float, default=0.20)
    parser.add_argument(
        "--unfreeze-dense-trunk",
        action="store_true",
        help=(
            "also optimize the inherited dense IIWPlanner; default freezes "
            "it and learns only router plus residual experts"
        ),
    )
    args = parser.parse_args()

    if args.num_experts != EXPECTED_EXPERTS:
        raise ValueError("current production contract requires --num-experts 2")
    if args.steps <= 0 or args.lr <= 0.0:
        raise ValueError("steps and lr must be positive")
    if (
        args.max_pos_weight < 1.0
        or args.routing_margin < 0.0
        or args.residual_rank <= 0
        or args.router_hidden_dim <= 0
        or args.routing_temperature <= 0.0
    ):
        raise ValueError("invalid loss setting")
    for name in (
        "bce_weight", "foreground_weight", "background_weight",
        "dice_weight", "temporal_weight", "router_weight",
        "assigned_expert_weight",
        "balance_weight", "entropy_weight", "separation_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(name + " must be non-negative")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = IIWPlannerDataset(args.dataset_root, args.index)
    num_samples = len(dataset)
    if args.batch_size != num_samples:
        raise ValueError(
            "MoE routing supervision is full-batch; set --batch-size={}"
            .format(num_samples)
        )
    scene_ids = sorted({row["scene_id"] for row in dataset.samples})
    texts = sorted({row["text"] for row in dataset.samples})
    if not scene_ids or not texts:
        raise ValueError("MoE dataset contains no scene or text")
    states_np = np.stack([row["state"] for row in dataset.samples]).astype(np.float32)
    labels_np, mode_metadata = balanced_state_mode_labels(
        states_np, args.num_experts
    )
    mode_metadata["sample_ids_by_mode"] = [
        [
            dataset.samples[index]["sample_id"]
            for index in range(len(dataset))
            if int(labels_np[index]) == mode
        ]
        for mode in range(args.num_experts)
    ]

    raw_rows = [dataset[index] for index in range(len(dataset))]
    scene = torch.from_numpy(
        np.stack([row["scene_points"] for row in raw_rows])
    ).to(device=device, dtype=torch.float32).contiguous()
    state = torch.from_numpy(
        np.stack([row["state"] for row in raw_rows])
    ).to(device=device, dtype=torch.float32).contiguous()
    target = torch.from_numpy(
        np.stack([row["target"] for row in raw_rows])
    ).to(device=device, dtype=torch.float32).contiguous()
    mode_labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    sample_ids = [str(row["sample_id"]) for row in raw_rows]
    if scene.shape != (num_samples, NUM_POINTS, 6):
        raise ValueError("full-batch scene shape mismatch")
    if state.shape != (num_samples, STATE_DIM):
        raise ValueError("full-batch state shape mismatch")
    if target.shape != (
        num_samples, dataset.num_phases, NUM_POINTS, NUM_BODIES
    ):
        raise ValueError("full-batch target shape mismatch")

    clip_model = load_and_freeze_clip_model("ViT-B/32").to(device)
    clip_model.eval()
    if any(parameter.requires_grad for parameter in clip_model.parameters()):
        raise AssertionError("CLIP must remain frozen")
    with torch.no_grad():
        encoded_text = encode_text_clip(
            clip_model, texts, max_length=32, device=str(device)
        ).detach().float()
    if encoded_text.shape != (len(texts), TEXT_DIM):
        raise ValueError("CLIP text feature shape mismatch")
    text_lookup = {
        text: encoded_text[index] for index, text in enumerate(texts)
    }
    text_features = torch.stack(
        [text_lookup[row["text"]] for row in raw_rows], dim=0
    ).contiguous()
    del encoded_text, clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dense_file = args.dense_checkpoint.expanduser().resolve()
    if not dense_file.is_file():
        raise FileNotFoundError(dense_file)
    dense_saved = torch.load(str(dense_file), map_location="cpu")
    if not isinstance(dense_saved, Mapping):
        raise TypeError("dense checkpoint must be a mapping")
    if dense_saved.get("target_method") != "point_aligned_iiw_proxy":
        raise ValueError("dense checkpoint target method mismatch")
    if str(dense_saved.get("text_encoder")) != "ViT-B/32":
        raise ValueError("dense checkpoint text encoder mismatch")
    if int(dense_saved.get("num_phases", -1)) != int(dataset.num_phases):
        raise ValueError("dense checkpoint phase count mismatch")
    if int(dense_saved.get("native_body_part_count", -1)) != NUM_BODIES:
        raise ValueError("dense checkpoint body count mismatch")
    saved_indices = {
        Path(str(value)).name for value in dense_saved.get("index_files", [])
    }
    current_indices = {Path(str(value)).name for value in args.index}
    if saved_indices != current_indices:
        raise ValueError("dense checkpoint index set mismatch")
    dense_summary = dense_saved.get("summary", {})
    if isinstance(dense_summary, Mapping) and dense_summary.get("status") != "PASS":
        raise ValueError("dense initialization checkpoint did not have PASS status")
    dense_config = dense_saved.get("model_config")
    if not isinstance(dense_config, Mapping):
        raise TypeError("dense checkpoint model_config is missing")
    expected_dense_config = {
        "scene_dim": 6,
        "text_dim": TEXT_DIM,
        "state_dim": STATE_DIM,
        "num_phases": int(dataset.num_phases),
        "num_bodies": NUM_BODIES,
        "hidden_dim": int(dense_config.get("hidden_dim", -1)),
        "point_dim": int(dense_config.get("point_dim", -1)),
        "context_dim": int(dense_config.get("context_dim", -1)),
    }
    if dict(dense_config) != expected_dense_config:
        raise ValueError("dense checkpoint architecture/config is not canonical")
    dense_state = strip_module_prefix(
        dense_saved.get("model", dense_saved.get("iiw_planner_state_dict"))
    )

    dense_planner = IIWPlanner(**expected_dense_config).to(device)
    dense_planner.load_state_dict(dense_state, strict=True)
    dense_planner.eval()
    planner = MoEIIWPlanner(
        **expected_dense_config,
        num_experts=args.num_experts,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        residual_rank=args.residual_rank,
        router_hidden_dim=args.router_hidden_dim,
        routing_temperature=args.routing_temperature,
    ).to(device)
    planner.load_dense_state_dict(dense_state)

    positive_weight = (
        (1.0 - torch.from_numpy(dataset.target_soft_mean))
        / torch.from_numpy(dataset.target_soft_mean).clamp_min(1e-5)
    ).clamp(1.0, args.max_pos_weight).to(device)
    sparse_weights = {
        "bce": args.bce_weight,
        "foreground": args.foreground_weight,
        "background": args.background_weight,
        "dice": args.dice_weight,
        "temporal": args.temporal_weight,
    }
    auxiliary_weights = {
        "router_supervised_ce": args.router_weight,
        "router_load_balance": args.balance_weight,
        "router_confidence_entropy": args.entropy_weight,
        "router_separation_margin": args.separation_weight,
    }

    with torch.no_grad():
        dense_logits = dense_planner.forward_logits(scene, text_features, state)
        initialized_routing = planner.forward_with_routing(
            scene, text_features, state, return_expert_logits=False
        )
        initialized_logits = initialized_routing["logits"]
    if not torch.equal(initialized_logits, dense_logits):
        max_difference = float((initialized_logits - dense_logits).abs().max().item())
        raise AssertionError(
            "zero-residual MoE initialization lost exact dense parity: "
            + "max_abs_diff={:.9g}".format(max_difference)
        )
    print("[PASS] dense checkpoint -> zero-residual MoE has exact logit parity")
    del dense_logits, initialized_logits, initialized_routing, dense_planner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dense_parameters = getattr(planner, "dense_parameters", None)
    moe_parameters = getattr(planner, "moe_parameters", None)
    if not callable(dense_parameters) or not callable(moe_parameters):
        raise AttributeError(
            "MoEIIWPlanner must expose dense_parameters()/moe_parameters()"
        )
    if not args.unfreeze_dense_trunk:
        for parameter in dense_parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in dense_parameters()):
            raise AssertionError("failed to freeze inherited dense planner")
    if not any(parameter.requires_grad for parameter in moe_parameters()):
        raise AssertionError("router/residual expert parameters are not trainable")
    trainable = [parameter for parameter in planner.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("MoEIIWPlanner has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    def forward_loss() -> Tuple[torch.Tensor, Dict, Dict, torch.Tensor, Dict]:
        routing = planner.forward_with_routing(
            scene, text_features, state, return_expert_logits=True
        )
        logits = routing["logits"]
        expected_shape = (
            num_samples, dataset.num_phases, NUM_POINTS, NUM_BODIES
        )
        if logits.shape != expected_shape:
            raise ValueError("MoE mixed IIW logit shape mismatch")
        required = {
            "routing_logits", "routing_probabilities", "selected_expert",
            "routing_entropy", "expert_load", "expert_logits",
        }
        missing = sorted(required.difference(routing))
        if missing:
            raise KeyError("routing diagnostics missing: " + ", ".join(missing))
        iiw_total, iiw_parts = sparse_objective(
            logits, target, positive_weight, sparse_weights
        )
        expert_total, expert_parts = assigned_expert_sparse_objective(
            routing["expert_logits"], mode_labels, target,
            positive_weight, sparse_weights,
        )
        anti_parts, anti_diagnostics = router_anti_collapse_objective(
            routing["routing_logits"],
            routing["routing_probabilities"],
            mode_labels,
            args.routing_margin,
        )
        total = iiw_total + args.assigned_expert_weight * expert_total
        for name, weight in auxiliary_weights.items():
            total = total + float(weight) * anti_parts[name]
        anti_parts = dict(anti_parts)
        anti_parts["assigned_expert_sparse_total"] = expert_total
        for name, value in expert_parts.items():
            anti_parts[name] = torch.as_tensor(value, device=device)
        return total, iiw_parts, anti_parts, logits, routing

    @torch.no_grad()
    def evaluate() -> Tuple[Dict, Dict]:
        planner.eval()
        total, iiw_parts, anti_parts, logits, routing = forward_loss()
        prediction = torch.sigmoid(logits)
        metrics = tensor_metrics(prediction.cpu(), target.cpu(), ACTIVE_THRESHOLD)
        row_scene_ids = [str(row["scene_id"]) for row in raw_rows]
        metrics.update(sample_retrieval_metrics(
            prediction.cpu(), target.cpu(), row_scene_ids
        ))
        metrics.update({
            "iiw_" + key: float(value.item())
            for key, value in iiw_parts.items()
        })
        metrics.update({
            key: float(value.item()) for key, value in anti_parts.items()
        })
        metrics["combined_total"] = float(total.item())
        metrics["routing"] = routing_metrics(
            routing["routing_logits"],
            routing["routing_probabilities"],
            mode_labels,
        )
        metrics.update(expert_parameter_diagnostics(planner))
        expert_predictions = torch.sigmoid(routing["expert_logits"])
        if expert_predictions.shape[1] != EXPECTED_EXPERTS:
            raise ValueError("expert prediction count mismatch")
        metrics["expert_pairwise_output_delta"] = float(
            (expert_predictions[:, 0] - expert_predictions[:, 1])
            .abs().mean().item()
        )
        metrics["mean_pairwise_prediction_delta"] = (
            mean_within_scene_prediction_delta(
                prediction.cpu(), row_scene_ids
            )
        )
        return metrics, {
            "prediction": prediction.cpu().numpy(),
            "target": target.cpu().numpy(),
            "router_logits": routing["routing_logits"].cpu().numpy(),
            "routing_probabilities": routing["routing_probabilities"].cpu().numpy(),
            "selected_expert": routing["selected_expert"].cpu().numpy(),
        }

    initial_metrics, _ = evaluate()
    planner.train()
    optimizer.zero_grad(set_to_none=True)
    first_total, _, _, _, _ = forward_loss()
    first_total.backward()
    initial_gradient_l2 = gradient_l2(trainable)
    if not np.isfinite(initial_gradient_l2) or initial_gradient_l2 <= 0.0:
        raise AssertionError("combined IIW/router loss does not reach MoE planner")
    optimizer.zero_grad(set_to_none=True)
    print(
        "[PASS] IIW + anti-collapse losses reach MoE planner: gradient_l2="
        + "{:.9g}".format(initial_gradient_l2)
    )

    records = []
    for step in range(1, args.steps + 1):
        planner.train()
        optimizer.zero_grad(set_to_none=True)
        total, iiw_parts, anti_parts, _, routing = forward_loss()
        total.backward()
        grad_l2 = gradient_l2(trainable)
        if not np.isfinite(grad_l2) or grad_l2 <= 0.0:
            raise AssertionError("MoE planner gradient is zero/non-finite")
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        route = routing_metrics(
            routing["routing_logits"].detach(),
            routing["routing_probabilities"].detach(),
            mode_labels,
        )
        record = {
            "step": step,
            "combined_total": float(total.detach().item()),
            "gradient_l2": grad_l2,
            "routing_accuracy": route["accuracy"],
            "routing_hard_counts": route["hard_counts"],
            "routing_soft_load": route["soft_load"],
            "routing_assigned_probability": route["mean_assigned_probability"],
        }
        record.update({
            "iiw_" + key: float(value.detach().item())
            for key, value in iiw_parts.items()
        })
        record.update({
            key: float(value.detach().item())
            for key, value in anti_parts.items()
        })
        records.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(record))

    final_metrics, arrays = evaluate()
    final_state = planner.state_dict()
    dense_trunk_unchanged = True
    if not args.unfreeze_dense_trunk:
        for name, original in dense_state.items():
            current = final_state[name].detach().cpu()
            if not torch.equal(current, original.detach().cpu()):
                dense_trunk_unchanged = False
                break
    checks = {
        "dense_trunk_bitwise_unchanged": dense_trunk_unchanged,
        "combined_objective_improved": (
            final_metrics["combined_total"] < initial_metrics["combined_total"]
        ),
        "iiw_objective_not_degraded": (
            final_metrics["iiw_total"] <= initial_metrics["iiw_total"] + 1e-6
        ),
        "router_accuracy_perfect": (
            final_metrics["routing"]["accuracy"] == 1.0
        ),
        "both_experts_hard_selected": all(
            count > 0 for count in final_metrics["routing"]["hard_counts"]
        ),
        "both_experts_soft_loaded": (
            min(final_metrics["routing"]["soft_load"]) >= 0.20
        ),
        "router_confident": (
            final_metrics["routing"]["mean_assigned_probability"] >= 0.60
        ),
        "experts_diverged": (
            final_metrics["expert_pairwise_parameter_delta_l2"] > 1e-8
            and final_metrics["expert_pairwise_output_delta"] > 1e-8
        ),
        "sparse_f1_preserved": final_metrics["f1_at_0_7"] >= 0.50,
        "temporal_order_preserved": final_metrics["order_margin"] > 0.0,
        "state_outputs_not_collapsed": (
            final_metrics["mean_pairwise_prediction_delta"] > 1e-6
        ),
    }
    status = "PASS" if all(checks.values()) else "CHECK"
    for name, passed in checks.items():
        print(("[PASS] " if passed else "[CHECK] ") + name)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        **expected_dense_config,
        "num_experts": args.num_experts,
        "residual_rank": args.residual_rank,
        "router_hidden_dim": args.router_hidden_dim,
        "routing_temperature": args.routing_temperature,
    }
    summary = {
        "status": status,
        "scope": "multi-scene full-batch state-routed MoE IIW training contract",
        "held_out_generalization": False,
        "scientific_claim_limit": (
            "training-set fit and two-mode specialization; held-out evidence "
            "requires a separate split"
        ),
        "base_adm_recomputed_or_modified": False,
        "target_method": "point_aligned_iiw_proxy",
        "planner_inputs": ["scene_points", "CLIP_text_feature", "state"],
        "target_leakage_into_planner_inputs": False,
        "dense_initialization_checkpoint": str(dense_file),
        "dataset_root": str(dataset.root),
        "index_files": list(args.index),
        "scene_ids": scene_ids,
        "texts": texts,
        "sample_ids": sample_ids,
        "num_samples": len(dataset),
        "num_phases": dataset.num_phases,
        "num_points": NUM_POINTS,
        "num_bodies": NUM_BODIES,
        "num_experts": args.num_experts,
        "full_batch_training": True,
        "batch_size": args.batch_size,
        "state_mode_metadata": mode_metadata,
        "mode_labels_in_sample_order": labels_np.tolist(),
        "model_type": "moe_iiw_planner",
        "planner_type": "moe_iiw_v1",
        "format_version": 2,
        "model_config": model_config,
        "dense_model_config": expected_dense_config,
        "text_encoder": "ViT-B/32",
        "text_encoder_frozen": True,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "sparse_loss_weights": sparse_weights,
        "auxiliary_loss_weights": auxiliary_weights,
        "assigned_expert_weight": args.assigned_expert_weight,
        "dense_trunk_frozen": not args.unfreeze_dense_trunk,
        "routing_margin": args.routing_margin,
        "positive_weight_per_body": positive_weight.detach().cpu().tolist(),
        "initial_gradient_l2": initial_gradient_l2,
        "initial": initial_metrics,
        "final": final_metrics,
        "checks": checks,
        "overfit_quality_pass": bool(all(checks.values())),
        "optimization_records": records,
    }
    summary_file = output_dir / "summary.json"
    checkpoint_file = output_dir / "moe_iiw_planner.pt"
    result_file = output_dir / "moe_iiw_planner_results.npz"
    csv_file = output_dir / "routing_per_sample.csv"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "model": planner.state_dict(),
            "iiw_planner_state_dict": planner.state_dict(),
            "model_type": "moe_iiw_planner",
            "planner_type": "moe_iiw_v1",
            "format_version": 2,
            "planner_class": "MoEIIWPlanner",
            "model_config": model_config,
            "dense_model_config": expected_dense_config,
            "num_experts": args.num_experts,
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "step": args.steps,
            "index_files": list(args.index),
            "text_encoder": "ViT-B/32",
            "target_method": "point_aligned_iiw_proxy",
            "num_phases": dataset.num_phases,
            "native_body_part_count": NUM_BODIES,
            "dense_initialization_checkpoint": str(dense_file),
            "mode_metadata": mode_metadata,
            "summary": summary,
        },
        checkpoint_file,
    )
    np.savez_compressed(
        result_file,
        prediction=arrays["prediction"].astype(np.float32),
        target=arrays["target"].astype(np.float32),
        router_logits=arrays["router_logits"].astype(np.float32),
        routing_probabilities=arrays["routing_probabilities"].astype(np.float32),
        selected_expert=arrays["selected_expert"].astype(np.int64),
        mode_labels=labels_np,
        states=states_np,
        sample_ids=np.asarray(sample_ids),
    )
    with csv_file.open("w", newline="") as handle:
        fieldnames = [
            "sample_id", "mode_label", "selected_expert",
            "expert_0_probability", "expert_1_probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow({
                "sample_id": sample_id,
                "mode_label": int(labels_np[index]),
                "selected_expert": int(arrays["selected_expert"][index]),
                "expert_0_probability": float(
                    arrays["routing_probabilities"][index, 0]
                ),
                "expert_1_probability": float(
                    arrays["routing_probabilities"][index, 1]
                ),
            })
    print(
        "[PASS] dense IIWPlanner -> state-routed MoE IIW contract"
        if status == "PASS"
        else "[CHECK] MoE training completed; inspect failed strict checks"
    )
    for path in (summary_file, checkpoint_file, result_file, csv_file):
        print("[OK] saved: " + str(path))


if __name__ == "__main__":
    main()

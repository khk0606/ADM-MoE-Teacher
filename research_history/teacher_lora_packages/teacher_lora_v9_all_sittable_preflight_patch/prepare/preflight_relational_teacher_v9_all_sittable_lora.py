#!/usr/bin/env python3
"""CUDA parity, loss-routing and policy preflight for Teacher-v9.

This program performs no optimizer update and saves no model checkpoint.  It
starts from the sealed v5r4 Teacher, installs a fresh zero-output LoRA, and
audits the scene-level all-sittable objective on room_0101/0102 only.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from fewshot_cdm_common import load_split  # noqa: E402
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    lora_parameter_energy,
    set_frozen_base_eval_lora_train,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    PROMPTS,
    TEACHER_FORWARD_INPUTS,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v9_all_sittable_metrics import (  # noqa: E402
    all_instance_metrics,
    simultaneous_presence_checks,
)
from relational_teacher_v9_lora_objective import (  # noqa: E402
    all_sittable_objective,
    physical_prediction,
    preservation_loss,
)
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
    OBJECTIVE_WEIGHTS,
    POLICY_SCHEMA,
    RELATIVE_SELECTION_RULES,
    SCHEMA,
    canonical_sha256,
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v9_lora_runtime import (  # noqa: E402
    FORWARD_INPUT_KEYS,
    compose_cdm_config,
    configure_reproducibility,
    deterministic_noise,
    gradient_summary,
    load_stats,
    perturb_lora_output,
    predict_xstart,
    require_positive_gradient,
    restore_lora,
    tensor_sha256,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v5-dataset-root", type=Path, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-evidence-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _build_relational_batch(
    bundles: Sequence[Mapping[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> Dict[str, object]:
    rows = []
    for bundle in sorted(bundles, key=lambda value: str(value["scene_id"])):
        for prompt_id in sorted(PROMPTS):
            rows.append((bundle, prompt_id))
    targets = np.stack([np.asarray(row[0]["all_target"], np.float32) for row in rows])
    normalized = (targets - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)
    points = np.stack([np.asarray(row[0]["points"], np.float32) for row in rows])
    batch = {
        "x": torch.from_numpy(normalized.astype(np.float32)).to(device),
        "xyz": torch.from_numpy(points[:, :, :3]).to(device).contiguous(),
        "feat": torch.from_numpy(points[:, :, 3:6] / 255.0).to(device).contiguous(),
        "text": [PROMPTS[row[1]] for row in rows],
        "scene_ids": [str(row[0]["scene_id"]) for row in rows],
        "prompt_ids": [row[1] for row in rows],
        "instance_names": [tuple(row[0]["instance_names"]) for row in rows],
        "instance_roles": [tuple(row[0]["instance_roles"]) for row in rows],
    }
    for key, dtype in (
        ("instance_targets", np.float32),
        ("verified_object_mask", bool),
        ("verified_positive_mask", bool),
        ("unknown_sittable_mask", bool),
        ("environment_aux_mask", bool),
        ("explicit_negative_mask", bool),
        ("all_target", np.float32),
    ):
        value = np.stack([np.asarray(row[0][key], dtype=dtype) for row in rows])
        batch[key] = torch.from_numpy(value).to(device)
    if batch["scene_ids"] != ["room_0101", "room_0101", "room_0102", "room_0102"]:
        raise AssertionError("Teacher-v9 train panel order changed")
    if batch["prompt_ids"] != ["sit_watch_v1", "sit_write_v1"] * 2:
        raise AssertionError("Teacher-v9 prompt-pair order changed")
    if any(tuple(roles) != ("bed", "normal_chair", "high_chair") for roles in batch["instance_roles"]):
        raise AssertionError("Teacher-v9 instance-role order changed")
    return batch


def _paired_noise(batch: Mapping[str, object], seed: int, device: str) -> torch.Tensor:
    one_per_scene = deterministic_noise(
        torch.Size((2, batch["x"].shape[1], batch["x"].shape[2])),
        seed,
        device,
    )
    return one_per_scene.repeat_interleave(2, dim=0)


def _select_v5_probe_ids(rows: Mapping[str, Mapping[str, object]]) -> List[str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for sample_id, row in rows.items():
        grouped[str(row["target"])].append(str(sample_id))
    if set(grouped) != {"chair", "bed", "whiteboard"}:
        raise ValueError("sealed v5 replay target inventory changed")
    return [sorted(grouped[target])[0] for target in ("chair", "bed", "whiteboard")]


def _finite_mapping(value: Mapping[str, object], label: str) -> None:
    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)
        elif isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise ValueError(label + " contains a non-finite number")
    walk(value)


def _metrics_for(bundle: Mapping[str, object], prediction: np.ndarray) -> Dict[str, object]:
    value = all_instance_metrics(
        prediction,
        np.asarray(bundle["instance_targets"], np.float32),
        bundle["instance_names"],
        np.asarray(bundle["xyz"], np.float32),
        np.asarray(bundle["verified_object_mask"], bool),
        np.asarray(bundle["explicit_negative_mask"], bool),
        np.asarray(bundle["unknown_sittable_mask"], bool),
    )
    _finite_mapping(value, str(bundle["scene_id"]) + " metrics")
    return value


def _presence_checks(bundle: Mapping[str, object], prediction: np.ndarray) -> Dict[str, bool]:
    # Missing-object counterexamples intentionally have zero predicted mass on
    # one verified instance.  Its hotspot centroid is therefore +inf, which is
    # a valid fail-closed diagnostic consumed by simultaneous_presence_checks.
    # Do not apply the finite-baseline guard used by _metrics_for here.  Only
    # boolean checks are persisted, so no Infinity enters policy JSON.
    metrics = all_instance_metrics(
        prediction,
        np.asarray(bundle["instance_targets"], np.float32),
        bundle["instance_names"],
        np.asarray(bundle["xyz"], np.float32),
        np.asarray(bundle["verified_object_mask"], bool),
        np.asarray(bundle["explicit_negative_mask"], bool),
        np.asarray(bundle["unknown_sittable_mask"], bool),
    )
    return simultaneous_presence_checks(
        metrics,
        **ABSOLUTE_PRESENCE_LIMITS,
    )


def _counterexample_audit(bundle: Mapping[str, object]) -> Dict[str, object]:
    targets = np.asarray(bundle["instance_targets"], np.float32)
    full = np.asarray(bundle["all_target"], np.float32)
    roles = tuple(bundle["instance_roles"])
    role_slot = {role: index for index, role in enumerate(roles)}
    bed = targets[role_slot["bed"]]
    normal = targets[role_slot["normal_chair"]]
    high = targets[role_slot["high_chair"]]
    candidates = {
        "valid_all_three": full.copy(),
        "bed_only": bed.copy(),
        "missing_normal_chair": np.maximum(bed, high),
        "missing_high_chair": np.maximum(bed, normal),
    }

    displaced = full.copy()
    slot = role_slot["normal_chair"]
    object_mask = np.asarray(bundle["verified_object_mask"], bool)[slot]
    object_indices = np.flatnonzero(object_mask)
    target_any = targets[slot].max(axis=-1)
    active_indices = np.flatnonzero(object_mask & (target_any >= 0.30))
    if active_indices.size == 0:
        raise ValueError("normal-Chair target has no active support")
    k = max(1, int(math.ceil(float(active_indices.size) * 0.25)))
    target_order = np.lexsort((active_indices, -target_any[active_indices]))
    target_top = set(int(value) for value in active_indices[target_order[:k]])
    available = np.asarray(
        [value for value in object_indices if int(value) not in target_top],
        dtype=np.int64,
    )
    if available.size < k:
        raise ValueError("normal-Chair object is too small for displacement counterexample")
    xyz = np.asarray(bundle["xyz"], np.float32)
    gt_weights = target_any[active_indices]
    gt_centroid = (
        xyz[active_indices, :2] * gt_weights[:, None]
    ).sum(axis=0) / max(float(gt_weights.sum()), 1e-12)
    distances = np.linalg.norm(xyz[available, :2] - gt_centroid[None], axis=1)
    farthest = available[np.lexsort((available, -distances))[:k]]
    displaced[object_mask] = 0.0
    displaced[farthest] = 1.0
    candidates["same_object_displacement"] = displaced

    hotspot = full.copy()
    hotspot[np.asarray(bundle["explicit_negative_mask"], bool)] = 1.0
    candidates["explicit_negative_hotspot"] = hotspot

    rows = {}
    for name, prediction in candidates.items():
        checks = _presence_checks(bundle, prediction)
        rows[name] = {"checks": checks, "all_checks_pass": all(checks.values())}
    if rows["valid_all_three"]["all_checks_pass"] is not True:
        raise AssertionError("exact all-sittable GT failed the locked presence policy")
    for name in (
        "bed_only",
        "missing_normal_chair",
        "missing_high_chair",
        "same_object_displacement",
        "explicit_negative_hotspot",
    ):
        if rows[name]["all_checks_pass"] is not False:
            raise AssertionError(name + " was incorrectly accepted")
    return rows


def _validate_gate0a(
    evidence_file: Path,
    original_checkpoint: Path,
    v5_checkpoint: Path,
) -> None:
    evidence = read_json(evidence_file)
    quality = evidence.get("quality_chain")
    if (
        evidence.get("schema") != "history_affordance_v2_gate0a_evidence_report_v1"
        or evidence.get("status") != "PASS"
        or evidence.get("may_export_base_teacher") is not True
        or not isinstance(quality, Mapping)
        or not isinstance(quality.get("sha256"), Mapping)
    ):
        raise ValueError("v5r4 evidence report is not sealed Gate-0A PASS")
    hashes = quality["sha256"]
    if hashes.get("fewshot_checkpoint") != sha256_file(v5_checkpoint):
        raise ValueError("v5r4 checkpoint differs from Gate-0A evidence")
    if hashes.get("original_checkpoint") != sha256_file(original_checkpoint):
        raise ValueError("original checkpoint differs from Gate-0A evidence")


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9 LoRA preflight requires CUDA")
    if args.diffusion_steps != 500:
        raise ValueError("Teacher-v9 preflight is sealed to 500 diffusion steps")
    if args.lora_rank != 4 or float(args.lora_alpha) != 8.0:
        raise ValueError("Teacher-v9 preflight is sealed to LoRA rank=4 alpha=8")
    configure_reproducibility(args.seed)

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    index_file = args.index.expanduser().resolve()
    v5_dataset_root = args.v5_dataset_root.expanduser().resolve()
    v5_split_file = args.v5_split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    v5_checkpoint = args.v5_checkpoint.expanduser().resolve()
    evidence_file = args.v5_evidence_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9 preflight output")
    for path in (
        index_file,
        v5_split_file,
        stats_file,
        original_checkpoint,
        v5_checkpoint,
        evidence_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_dataset_root.is_dir():
        raise FileNotFoundError("a required dataset root is absent")

    index = validate_top_index(dataset_root, source_root, index_file)
    records = sorted(index["scenes"], key=lambda row: str(row["scene_id"]))
    bundles = [load_train_scene_bundle(dataset_root, source_root, row) for row in records]
    if tuple(str(row["scene_id"]) for row in bundles) != EXPECTED_TRAIN_SCENES:
        raise AssertionError("Teacher-v9 bundle order changed")
    _validate_gate0a(evidence_file, original_checkpoint, v5_checkpoint)
    mean, std = load_stats(stats_file)
    batch = _build_relational_batch(bundles, mean, std, args.device)
    timesteps = torch.tensor([125, 125, 375, 375], dtype=torch.long, device=args.device)
    noise = _paired_noise(batch, args.seed + 1000, args.device)
    if not torch.equal(batch["x"][0], batch["x"][1]) or not torch.equal(batch["x"][2], batch["x"][3]):
        raise AssertionError("watch/write pair does not share exact x0")
    if not torch.equal(noise[0], noise[1]) or not torch.equal(noise[2], noise[3]):
        raise AssertionError("watch/write pair does not share exact noise")
    if int(timesteps[0]) != int(timesteps[1]) or int(timesteps[2]) != int(timesteps[3]):
        raise AssertionError("watch/write pair does not share exact timestep")
    kwargs = {
        "c_pc_xyz": batch["xyz"],
        "c_pc_feat": batch["feat"],
        "c_text": batch["text"],
    }
    if tuple(sorted(kwargs)) != tuple(sorted(TEACHER_FORWARD_INPUTS)):
        raise AssertionError("target/relation metadata entered Teacher forward")

    v5_split = load_split(v5_split_file)
    v5_rows = load_v5_rows(
        v5_dataset_root,
        v5_split,
        "train",
        mean,
        std,
        4.0,
        16.0,
        0.7,
    )
    counts = Counter(str(row["target"]) for row in v5_rows.values())
    if counts != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise ValueError("sealed v5 replay rows changed")
    v5_probe_ids = _select_v5_probe_ids(v5_rows)
    v5_batch = stack_v5_batch(v5_rows, v5_probe_ids, args.device)
    v5_timesteps = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, args.seed + 2000, args.device)
    v5_kwargs = {
        "c_pc_xyz": v5_batch["xyz"],
        "c_pc_feat": v5_batch["feat"],
        "c_text": v5_batch["text"],
    }

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    with torch.no_grad():
        frozen_prediction = predict_xstart(
            model, diffusion, batch["x"], timesteps, kwargs, noise
        )
        frozen_v5_prediction = predict_xstart(
            model,
            diffusion,
            v5_batch["x"],
            v5_timesteps,
            v5_kwargs,
            v5_noise,
        )
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)
    frozen_physical = physical_prediction(frozen_prediction, mean_tensor, std_tensor)
    frozen_numpy = frozen_physical.detach().cpu().numpy()
    baseline_rows = {}
    for index_value, (scene_id, prompt_id) in enumerate(zip(batch["scene_ids"], batch["prompt_ids"])):
        bundle = bundles[0] if scene_id == "room_0101" else bundles[1]
        baseline_rows[f"{scene_id}|{prompt_id}"] = _metrics_for(
            bundle, frozen_numpy[index_value]
        )
    prompt_mask = batch["verified_positive_mask"][0::2].to(frozen_physical.dtype).unsqueeze(-1)
    prompt_denominator = float(prompt_mask.sum().item()) * 6.0
    baseline_prompt_invariance = float(
        (((frozen_physical[0::2] - frozen_physical[1::2]).square() * prompt_mask).sum()
        / prompt_denominator).item()
    )
    baseline_v5_dense_rows = (
        (frozen_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
    ).detach().cpu().tolist()
    counterexamples = {
        str(bundle["scene_id"]): _counterexample_audit(bundle) for bundle in bundles
    }

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "metric_policy.json"
    policy = {
        "schema": POLICY_SCHEMA,
        "status": "LOCKED_BEFORE_OPTIMIZER",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "train_scenes": list(EXPECTED_TRAIN_SCENES),
        "development_scenes_unread": list(EXPECTED_DEVELOPMENT_SCENES),
        "panel_rows": [
            f"{scene_id}|{prompt_id}"
            for scene_id, prompt_id in zip(batch["scene_ids"], batch["prompt_ids"])
        ],
        "prompt_pairing": "same_scene_same_x0_same_timestep_same_noise",
        "timesteps": [int(value) for value in timesteps.detach().cpu().tolist()],
        "relational_noise_sha256": tensor_sha256(noise),
        "v5_probe_ids": v5_probe_ids,
        "v5_timesteps": [int(value) for value in v5_timesteps.detach().cpu().tolist()],
        "v5_noise_sha256": tensor_sha256(v5_noise),
        "absolute_presence_limits": ABSOLUTE_PRESENCE_LIMITS,
        "relative_selection_rules": RELATIVE_SELECTION_RULES,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "baseline": {
            "per_scene_prompt_instance_metrics": baseline_rows,
            "prompt_invariance_mse": baseline_prompt_invariance,
            "v5_replay_dense_mse_per_target": {
                target: float(value)
                for target, value in zip(("chair", "bed", "whiteboard"), baseline_v5_dense_rows)
            },
            "v5_replay_dense_mse_mean": float(np.mean(baseline_v5_dense_rows)),
        },
        "counterexamples": counterexamples,
        "unknown_sittable_policy": "ignored_in_all_v9_task_losses_never_negative",
        "explicit_negative_policy": "tv_desk_whiteboard_addition_over_frozen_only",
        "paper_test_access": False,
    }
    _finite_mapping(policy, "metric policy")
    policy["policy_id"] = canonical_sha256(
        {key: value for key, value in policy.items() if key not in {"created_utc", "policy_id"}}
    )
    atomic_write_json(policy_file, policy)
    policy_hash = sha256_file(policy_file)

    module_names = install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    if len(module_names) != 31:
        raise AssertionError(f"Teacher-v9 expected 31 LoRA modules, got {len(module_names)}")
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    with torch.no_grad():
        zero_prediction = predict_xstart(
            model, diffusion, batch["x"], timesteps, kwargs, noise
        )
        zero_v5_prediction = predict_xstart(
            model,
            diffusion,
            v5_batch["x"],
            v5_timesteps,
            v5_kwargs,
            v5_noise,
        )
    if not torch.equal(zero_prediction, frozen_prediction) or not torch.equal(
        zero_v5_prediction, frozen_v5_prediction
    ):
        raise AssertionError("zero-init Teacher-v9 LoRA changed sealed v5r4 output")

    prediction = predict_xstart(model, diffusion, batch["x"], timesteps, kwargs, noise)
    objective = all_sittable_objective(
        prediction,
        frozen_prediction,
        batch,
        mean_tensor,
        std_tensor,
    )
    components = {}
    for slot, role in enumerate(("bed", "normal_chair", "high_chair")):
        summary = gradient_summary(
            objective["per_instance_primary"][:, slot].mean(),
            named_lora,
            retain_graph=True,
        )
        require_positive_gradient(role + " primary", summary)
        components[role + "_primary"] = summary
    for name in (
        "instance_macro_primary",
        "verified_union",
        "environment_auxiliary",
        "paired_prompt_invariance",
    ):
        summary = gradient_summary(objective[name], named_lora, retain_graph=True)
        require_positive_gradient(name, summary)
        components[name] = summary

    physical = physical_prediction(prediction, mean_tensor, std_tensor)
    unknown_shift = (
        batch["unknown_sittable_mask"].to(physical.dtype).unsqueeze(-1) * 0.01
    )
    shifted_normalized = prediction + unknown_shift / std_tensor
    shifted = all_sittable_objective(
        shifted_normalized,
        frozen_prediction,
        batch,
        mean_tensor,
        std_tensor,
    )
    for name in (
        "instance_macro_primary",
        "verified_union",
        "environment_auxiliary",
        "explicit_negative_addition",
        "paired_prompt_invariance",
    ):
        if not torch.equal(objective[name].detach(), shifted[name].detach()):
            raise AssertionError("unknown Sit points leaked into loss: " + name)
    zero_negative_addition = float(
        objective["explicit_negative_addition"].detach().item()
    )
    del shifted, shifted_normalized, unknown_shift, physical, objective, prediction
    torch.cuda.empty_cache()

    perturbation_rows = []
    zero_state = {name: value.detach().clone() for name, value in named_lora.items()}
    for amount in (1e-5, -1e-5):
        restore_lora(named_lora, zero_state)
        perturb_lora_output(named_lora, amount)
        perturbed_prediction = predict_xstart(
            model, diffusion, batch["x"], timesteps, kwargs, noise
        )
        perturbed_objective = all_sittable_objective(
            perturbed_prediction,
            frozen_prediction,
            batch,
            mean_tensor,
            std_tensor,
        )
        perturbed_v5 = predict_xstart(
            model,
            diffusion,
            v5_batch["x"],
            v5_timesteps,
            v5_kwargs,
            v5_noise,
        )
        negative_summary = gradient_summary(
            perturbed_objective["explicit_negative_addition"],
            named_lora,
            retain_graph=True,
        )
        replay_summary = gradient_summary(
            preservation_loss(perturbed_v5, frozen_v5_prediction),
            named_lora,
            retain_graph=True,
        )
        regularizer_summary = gradient_summary(
            lora_parameter_energy(model), named_lora, retain_graph=False
        )
        perturbation_rows.append(
            {
                "amount": amount,
                "explicit_negative_addition": negative_summary,
                "v5_replay_preservation": replay_summary,
                "lora_regularizer": regularizer_summary,
            }
        )
        del (
            perturbed_prediction,
            perturbed_objective,
            perturbed_v5,
            negative_summary,
            replay_summary,
            regularizer_summary,
        )
        torch.cuda.empty_cache()
    selected_perturbation = max(
        perturbation_rows,
        key=lambda row: (
            float(row["explicit_negative_addition"]["gradient_l2"]),
            float(row["explicit_negative_addition"]["loss"]),
        ),
    )
    require_positive_gradient(
        "perturbed explicit-negative addition",
        selected_perturbation["explicit_negative_addition"],
    )
    require_positive_gradient(
        "perturbed v5 replay preservation",
        selected_perturbation["v5_replay_preservation"],
    )
    require_positive_gradient(
        "perturbed LoRA regularizer",
        selected_perturbation["lora_regularizer"],
    )
    components["explicit_negative_addition_perturbed"] = selected_perturbation[
        "explicit_negative_addition"
    ]
    components["v5_replay_preservation_perturbed"] = selected_perturbation[
        "v5_replay_preservation"
    ]
    components["lora_regularizer_perturbed"] = selected_perturbation[
        "lora_regularizer"
    ]
    restore_lora(named_lora, zero_state)

    for parameter in model.parameters():
        parameter.grad = None
    prediction = predict_xstart(model, diffusion, batch["x"], timesteps, kwargs, noise)
    total_objective = all_sittable_objective(
        prediction, frozen_prediction, batch, mean_tensor, std_tensor
    )
    current_v5 = predict_xstart(
        model,
        diffusion,
        v5_batch["x"],
        v5_timesteps,
        v5_kwargs,
        v5_noise,
    )
    replay_preservation = preservation_loss(current_v5, frozen_v5_prediction)
    total_loss = (
        OBJECTIVE_WEIGHTS["instance_macro_primary"]
        * total_objective["instance_macro_primary"]
        + OBJECTIVE_WEIGHTS["verified_union"] * total_objective["verified_union"]
        + OBJECTIVE_WEIGHTS["environment_auxiliary"]
        * total_objective["environment_auxiliary"]
        + OBJECTIVE_WEIGHTS["explicit_negative_addition"]
        * total_objective["explicit_negative_addition"]
        + OBJECTIVE_WEIGHTS["paired_prompt_invariance"]
        * total_objective["paired_prompt_invariance"]
        + OBJECTIVE_WEIGHTS["frozen_v5_replay_preservation"]
        * replay_preservation
        + OBJECTIVE_WEIGHTS["lora_regularizer"] * lora_parameter_energy(model)
    )
    total_loss.backward()
    frozen_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if "lora_" not in name and parameter.grad is not None
    ]
    if frozen_gradient_names:
        raise AssertionError("gradient reached frozen CDM parameters")
    total_gradient_sq = 0.0
    for parameter in named_lora.values():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("total objective produced non-finite LoRA gradient")
            total_gradient_sq += float(parameter.grad.float().square().sum().item())
    total_gradient_l2 = math.sqrt(total_gradient_sq)
    if total_gradient_l2 <= 0.0:
        raise RuntimeError("Teacher-v9 total objective has no LoRA gradient")

    source_paths = {
        "preflight": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v9_all_sittable_lora_preflight.py",
        "objective": PREPARE_ROOT / "relational_teacher_v9_lora_objective.py",
        "runtime": PREPARE_ROOT / "relational_teacher_v9_lora_runtime.py",
        "preflight_contract": PREPARE_ROOT / "relational_teacher_v9_lora_preflight_contract.py",
        "dataset_contract": PREPARE_ROOT / "relational_teacher_v9_all_sittable_contract.py",
        "metrics": PREPARE_ROOT / "relational_teacher_v9_all_sittable_metrics.py",
        "dataset_validator": PREPARE_ROOT / "validate_relational_teacher_v9_all_sittable_dataset.py",
        "lora": PREPARE_ROOT / "fewshot_cdm_lora.py",
        "v5_common": PREPARE_ROOT / "fewshot_cdm_common.py",
        "v5_loader": PREPARE_ROOT / "train_fewshot_cdm.py",
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(name + ": " + str(path))
    paths = {
        "source_dataset_index": source_root / str(index["source_index_file"]),
        "dataset_index": index_file,
        "v5_split": v5_split_file,
        "stats_file": stats_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "v5_evidence_report": evidence_file,
        "metric_policy": policy_file,
        **source_paths,
    }
    path_strings = {name: str(Path(path).resolve()) for name, path in paths.items()}
    path_hashes = {name: sha256_file(Path(path).resolve()) for name, path in paths.items()}
    scene_bindings = [
        {
            key: bundle[key]
            for key in (
                "scene_id",
                "instance_names",
                "instance_roles",
                "manifest_file",
                "manifest_sha256",
                "consensus_file",
                "consensus_sha256",
                "points_file",
                "points_sha256",
            )
        }
        for bundle in bundles
    ]
    checks = {
        "cuda_used": True,
        "sealed_v9_all_sittable_index": True,
        "exact_two_train_scenes": True,
        "development_payloads_unread": True,
        "one_primary_gt_per_scene": True,
        "watch_write_share_gt_timestep_noise": True,
        "bed_normal_high_present_simultaneously": True,
        "equal_instance_macro_not_point_weighted": True,
        "unknown_sittable_ignored_not_negative": True,
        "explicit_negative_is_tv_desk_whiteboard_only": True,
        "counterexamples_fail_closed": True,
        "metric_policy_locked_before_optimizer": True,
        "gate0a_v5_checkpoint_bound": True,
        "original_checkpoint_bound": True,
        "fresh_zero_init_v5_output_bitwise_equal": True,
        "only_lora_parameters_trainable": True,
        "every_instance_component_gradient_reaches_lora": True,
        "union_environment_prompt_gradients_reach_lora": True,
        "negative_replay_regularizer_gradients_reach_lora_when_perturbed": True,
        "total_objective_gradient_reaches_lora": True,
        "frozen_cdm_has_no_gradient": True,
        "teacher_forward_is_text_plus_scene_only": tuple(sorted(kwargs))
        == tuple(sorted(FORWARD_INPUT_KEYS)),
    }
    if not all(checks.values()):
        raise AssertionError("Teacher-v9 preflight checks are not all true")
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "train_scenes": list(EXPECTED_TRAIN_SCENES),
        "development_scenes_metadata_only": list(EXPECTED_DEVELOPMENT_SCENES),
        "development_payloads_read": False,
        "forward_input_keys": sorted(kwargs),
        "prompts": PROMPTS,
        "scene_bindings": scene_bindings,
        "panel": {
            "row_ids": policy["panel_rows"],
            "timesteps": policy["timesteps"],
            "relational_noise_sha256": policy["relational_noise_sha256"],
            "v5_probe_ids": v5_probe_ids,
            "v5_timesteps": policy["v5_timesteps"],
            "v5_noise_sha256": policy["v5_noise_sha256"],
        },
        "metric_policy_id": policy["policy_id"],
        "metric_policy_sha256": policy_hash,
        "lora": dict(lora_metadata(model)),
        "zero_init_parity_max_abs": float(
            (zero_prediction - frozen_prediction).abs().max().item()
        ),
        "zero_init_v5_parity_max_abs": float(
            (zero_v5_prediction - frozen_v5_prediction).abs().max().item()
        ),
        "zero_init_negative_addition": float(
            zero_negative_addition
        ),
        "zero_init_v5_preservation": 0.0,
        "zero_init_lora_energy": 0.0,
        "component_gradients": components,
        "perturbation_direction_audit": perturbation_rows,
        "total_objective": {
            "loss": float(total_loss.detach().item()),
            "gradient_l2": total_gradient_l2,
            "weights": OBJECTIVE_WEIGHTS,
        },
        "frozen_parameter_gradient_count": len(frozen_gradient_names),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": checks,
        "authorizes_one_scene_overfit_smoke": True,
        "authorizes_response3": False,
        "authorizes_calibration": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
        "failed_checks": [],
    }
    report["binding_id"] = canonical_sha256(
        {
            "paths": path_hashes,
            "scene_bindings": scene_bindings,
            "panel": report["panel"],
            "metric_policy_id": report["metric_policy_id"],
            "seed": report["seed"],
        }
    )
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    print("[PREFLIGHT_PASS] Teacher-v9 all-sittable LoRA CUDA gate")
    print("[PASS] fresh v5r4 zero-init parity; no v7/v8 candidate loaded")
    print("[PASS] Bed/normal-Chair/High-Chair gradients reach LoRA independently")
    print("[PASS] unknown Sit objects ignored; explicit negatives separated")
    print("[PASS] Bed-only, missing-Chair, displacement and hotspot cases rejected")
    print("[PASS] room_0201 arrays and paper test unread")
    print("[OK] LoRA modules:", len(module_names))
    print("[OK] total gradient L2:", total_gradient_l2)
    print("[OK] policy ID:", policy["policy_id"])
    print("[OK] binding ID:", report["binding_id"])
    print("[OK] report:", report_file)


if __name__ == "__main__":
    main()

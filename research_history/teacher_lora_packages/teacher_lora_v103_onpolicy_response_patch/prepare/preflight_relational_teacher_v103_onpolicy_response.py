#!/usr/bin/env python3
"""Teacher-v10.3 two-scene on-policy rollout-state response gate."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_relational_teacher_v10_supervised_capacity as v10  # noqa: E402
from fewshot_cdm_common import load_split  # noqa: E402
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from relational_teacher_v102_dense_instance_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_INSTANCES,
    SUPPORT_THRESHOLD,
)
from relational_teacher_v102_dense_instance_objective import (  # noqa: E402
    dense_instance_all_sittable_objective,
)
from relational_teacher_v103_onpolicy_response_contract import (  # noqa: E402
    ACTIVE_TASK_WEIGHT,
    CAPTURE_TIMESTEPS,
    DEVELOPMENT_SCENE,
    FW_ITERATIONS,
    HARD_NEGATIVE_POINTS,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    ROLES,
    SCENES,
    SCHEMA,
    STEP_RADII,
    SUPPORT_TASK_WEIGHT,
    TASK_ORDER,
    V102_SCHEMA,
    canonical_sha256,
    pooled_roles,
    rank_candidates,
    response_checks,
    stable_seed_pair,
)
from relational_teacher_v94_common_descent import (  # noqa: E402
    apply_flat_direction,
    flattened_task_gradients,
    frank_wolfe_min_norm_weights,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    PROMPTS,
    atomic_savez,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v9_all_sittable_metrics import (  # noqa: E402
    simultaneous_presence_checks,
)
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v9_lora_runtime import (  # noqa: E402
    FORWARD_INPUT_KEYS,
    compose_cdm_config,
    configure_reproducibility,
    deterministic_noise,
    load_stats,
    predict_xstart,
    tensor_sha256,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v102-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(label + " contains a non-finite number")


def _validate_v102_failure(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V102_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_step") is not None
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_objective_or_architecture_redesign") is not True
        or value.get("failed_checks")
        != ["at_least_one_dense_instance_candidate_passes_actual_k3"]
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v10.2 failure does not authorize v10.3")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != set(hashes)
        or "checkpoint" in paths
    ):
        raise ValueError("failed Teacher-v10.2 path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10.2 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed Teacher-v10.2 unexpectedly contains a checkpoint")
    return value


def _bound_path(authority: Mapping[str, object], name: str) -> Path:
    path = Path(str(authority["paths"][name])).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != authority["path_sha256"][name]:
        raise ValueError("Teacher-v10.2 bound input changed: " + name)
    return path


def _prompt_kwargs(
    bundle: Mapping[str, object], text: str, device: str
) -> Dict[str, object]:
    xyz = torch.from_numpy(np.asarray(bundle["xyz"], np.float32)[None]).to(
        device
    ).contiguous()
    feat = torch.from_numpy(
        np.asarray(bundle["points"], np.float32)[None, :, 3:6] / 255.0
    ).to(device).contiguous()
    result = {"c_pc_xyz": xyz, "c_pc_feat": feat, "c_text": [text]}
    if tuple(sorted(result)) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher-v10.3 forward input changed")
    return result


@torch.no_grad()
def _capture_trajectory(
    model,
    diffusion,
    bundle: Mapping[str, object],
    text: str,
    initial_seed: int,
    reverse_seed: int,
    device: str,
    progress: bool,
) -> Mapping[str, object]:
    noise = deterministic_noise(torch.Size((1, 8192, 6)), initial_seed, device)
    kwargs = _prompt_kwargs(bundle, text, device)
    indices: Sequence[int] = list(range(diffusion.num_timesteps))[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
    states: Dict[int, torch.Tensor] = {}
    rng_states: Dict[int, Mapping[str, torch.Tensor]] = {}
    torch_device = torch.device(device)
    devices = [
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(reverse_seed)
        torch.cuda.manual_seed(reverse_seed)
        image = noise
        for timestep in indices:
            if timestep in CAPTURE_TIMESTEPS:
                states[timestep] = image.detach().cpu().clone()
                rng_states[timestep] = {
                    "cpu": torch.random.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state(torch_device).cpu().clone(),
                }
            t = torch.tensor([timestep], dtype=torch.long, device=device)
            output = diffusion.p_sample(
                model,
                image,
                t,
                clip_denoised=False,
                model_kwargs=kwargs,
            )
            image = output["sample"]
    if set(states) != set(CAPTURE_TIMESTEPS):
        raise AssertionError("Teacher-v10.3 trajectory capture changed")
    return {
        "final": image[0].detach().cpu(),
        "states": states,
        "rng_states": rng_states,
    }


@torch.no_grad()
def _resume_trajectory(
    model,
    diffusion,
    bundle: Mapping[str, object],
    text: str,
    state: torch.Tensor,
    rng_state: Mapping[str, torch.Tensor],
    start_timestep: int,
    device: str,
) -> torch.Tensor:
    kwargs = _prompt_kwargs(bundle, text, device)
    image = state.to(device)
    torch_device = torch.device(device)
    devices = [
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.set_rng_state(rng_state["cpu"].cpu())
        torch.cuda.set_rng_state(rng_state["cuda"].cpu(), torch_device)
        for timestep in range(int(start_timestep), -1, -1):
            t = torch.tensor([timestep], dtype=torch.long, device=device)
            output = diffusion.p_sample(
                model,
                image,
                t,
                clip_denoised=False,
                model_kwargs=kwargs,
            )
            image = output["sample"]
    return image[0].detach().cpu()


def _direct_prediction(model, diffusion, states: torch.Tensor, timestep: int, kwargs):
    wrapped = diffusion._wrap_model(model) if hasattr(diffusion, "_wrap_model") else model
    t = torch.full((2,), int(timestep), dtype=torch.long, device=states.device)
    return wrapped(states, diffusion._scale_timesteps(t), **dict(kwargs))


def _physical(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(
        value.detach().cpu().numpy() * std.reshape(1, 1, 6)
        + mean.reshape(1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)


def _negative_tasks(
    physical: torch.Tensor, batch: Mapping[str, torch.Tensor]
) -> list[torch.Tensor]:
    instance_targets = batch["instance_targets"]
    instance_support = (
        instance_targets.max(dim=-1).values >= SUPPORT_THRESHOLD
    ).any(dim=1)
    known = ~batch["unknown_sittable_mask"].bool()
    target = batch["all_target"]
    safe = known & ~instance_support & (
        target.max(dim=-1).values < SUPPORT_THRESHOLD
    )
    explicit = batch["explicit_negative_mask"].bool()
    scalar = physical.max(dim=-1).values
    rows = []
    for prompt_index in range(2):
        values = scalar[prompt_index, safe[prompt_index]]
        if values.numel() == 0 or not bool(explicit[prompt_index].any().item()):
            raise ValueError("Teacher-v10.3 negative support is empty")
        count = min(HARD_NEGATIVE_POINTS, int(values.numel()))
        _, indices = torch.topk(values.detach(), k=count, largest=True, sorted=False)
        hard = values[indices].square().mean()
        explicit_loss = physical[prompt_index, explicit[prompt_index]].square().mean()
        rows.append(explicit_loss + 0.5 * hard)
    return rows


def _prompt_task(physical: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["verified_positive_mask"][0].bool()
    if not torch.equal(mask, batch["verified_positive_mask"][1].bool()):
        raise ValueError("Teacher-v10.3 prompt support changed")
    return (physical[0, mask] - physical[1, mask]).square().mean()


def _metric_rows(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    if predictions.shape != (2, 2, 8192, 6):
        raise ValueError("Teacher-v10.3 final map shape changed")
    return {
        scene: {
            prompt: v10._metrics(bundles[scene], predictions[scene_index, prompt_index])
            for prompt_index, prompt in enumerate(PROMPT_IDS)
        }
        for scene_index, scene in enumerate(SCENES)
    }


def _prompt_invariance(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, float]:
    result = {}
    for scene_index, scene in enumerate(SCENES):
        mask = np.asarray(bundles[scene]["verified_positive_mask"], bool)
        result[scene] = float(
            np.square(
                predictions[scene_index, 0, mask]
                - predictions[scene_index, 1, mask]
            ).mean()
        )
    return result


def _all_three(
    rows: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> Dict[str, Dict[str, bool]]:
    return {
        scene: {
            prompt: all(
                simultaneous_presence_checks(
                    rows[scene][prompt], **ABSOLUTE_PRESENCE_LIMITS
                ).values()
            )
            for prompt in PROMPT_IDS
        }
        for scene in SCENES
    }


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v10.3 on-policy response requires CUDA")
    failure_file = args.failed_v102_summary.expanduser().resolve()
    authority = _validate_v102_failure(failure_file)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v10.3 output")
    configure_reproducibility(MODEL_SEED)

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "onpolicy_response_policy.json"
    atomic_write_json(policy_file, POLICY)
    if read_json(policy_file) != POLICY or canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("Teacher-v10.3 policy hash changed")

    index_file = _bound_path(authority, "dataset_index")
    source_index = _bound_path(authority, "source_dataset_index")
    stats_file = _bound_path(authority, "stats_file")
    split_file = _bound_path(authority, "v5_split")
    evidence_file = _bound_path(authority, "v5_evidence_report")
    original_checkpoint = _bound_path(authority, "original_checkpoint")
    v5_checkpoint = _bound_path(authority, "v5_checkpoint")
    metric_policy_file = _bound_path(authority, "metric_policy")
    dataset_root = index_file.parent
    source_root = source_index.parent
    v5_root = v5_checkpoint.parents[2]

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != set(SCENES):
        raise ValueError("Teacher-v10.3 train-scene inventory changed")
    bundles = {
        scene: load_train_scene_bundle(dataset_root, source_root, records[scene])
        for scene in SCENES
    }
    for scene in SCENES:
        if tuple(str(name) for name in bundles[scene]["instance_names"]) != EXPECTED_INSTANCES[scene]:
            raise ValueError(scene + " verified instance order changed")

    mean, std = load_stats(stats_file)
    batches = {
        scene: v10._build_batch(bundles[scene], mean, std, args.device)
        for scene in SCENES
    }
    kwargs = {scene: v10._kwargs(batches[scene]) for scene in SCENES}
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    metric_policy = read_json(metric_policy_file)
    split = load_split(split_file)
    v5_rows = load_v5_rows(v5_root, split, "train", mean, std, 4.0, 16.0, 0.7)
    if Counter(str(row["target"]) for row in v5_rows.values()) != Counter(
        {"chair": 18, "whiteboard": 6, "bed": 1}
    ):
        raise ValueError("sealed v5 replay inventory changed")
    v5_ids = list(metric_policy["v5_probe_ids"])
    if [str(v5_rows[name]["target"]) for name in v5_ids] != [
        "chair",
        "bed",
        "whiteboard",
    ]:
        raise ValueError("v5 fixed-probe order changed")
    v5_batch = stack_v5_batch(v5_rows, v5_ids, args.device)
    v5_kwargs = {
        "c_pc_xyz": v5_batch["xyz"],
        "c_pc_feat": v5_batch["feat"],
        "c_text": v5_batch["text"],
    }

    cfg = compose_cdm_config(500, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()

    trajectories: Dict[str, Dict[str, Dict[str, Mapping[str, object]]]] = {
        "design": {},
        "audit": {},
    }
    seed_tables: Dict[str, list] = {"design": [], "audit": []}
    for domain in ("design", "audit"):
        for scene in SCENES:
            trajectories[domain][scene] = {}
            scene_seeds = []
            for prompt_id in PROMPT_IDS:
                seeds = stable_seed_pair(domain, scene, prompt_id)
                scene_seeds.append(list(seeds))
                trajectories[domain][scene][prompt_id] = _capture_trajectory(
                    model,
                    diffusion,
                    bundles[scene],
                    PROMPTS[prompt_id],
                    seeds[0],
                    seeds[1],
                    args.device,
                    not args.no_progress,
                )
                print(
                    "[BASE-{}] scene={} prompt={}".format(
                        domain.upper(), scene, prompt_id
                    ),
                    flush=True,
                )
            seed_tables[domain].append(scene_seeds)
    if seed_tables["design"] == seed_tables["audit"]:
        raise AssertionError("Teacher-v10.3 design/audit seeds overlap")

    audit_base_normalized = np.stack(
        [
            np.stack(
                [
                    trajectories["audit"][scene][prompt]["final"].numpy()
                    for prompt in PROMPT_IDS
                ]
            )
            for scene in SCENES
        ]
    ).astype(np.float32)
    for scene in SCENES:
        for prompt in PROMPT_IDS:
            for timestep in CAPTURE_TIMESTEPS:
                repeated = _resume_trajectory(
                    model,
                    diffusion,
                    bundles[scene],
                    PROMPTS[prompt],
                    trajectories["audit"][scene][prompt]["states"][timestep],
                    trajectories["audit"][scene][prompt]["rng_states"][timestep],
                    timestep,
                    args.device,
                )
                if not torch.equal(
                    repeated, trajectories["audit"][scene][prompt]["final"]
                ):
                    raise AssertionError("Teacher-v10.3 trajectory resume is not exact")
    print("[TRAJECTORY_REPRODUCTION_PASS] 12 partial resumes are exact", flush=True)

    v5_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(
        v5_batch["x"].shape, MODEL_SEED + 900000, args.device
    )
    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"])
        .square()
        .mean(dim=(1, 2))
        .cpu()
        .tolist()
    )

    modules = install_lora(model, LORA_RANK, LORA_ALPHA, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("Teacher-v10.3 LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    parameters = list(named_lora.values())
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    zero_state = v10._lora_cpu_state(named_lora)
    zero_state_sha256 = v10._state_sha256(named_lora)
    with torch.no_grad():
        zero_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not torch.equal(zero_v5, base_v5_prediction):
        raise AssertionError("fresh zero-init LoRA differs from sealed v5r4")

    audit_base = np.clip(
        audit_base_normalized * std.reshape(1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    base_rows = _metric_rows(bundles, audit_base)
    base_pooled = pooled_roles(base_rows)
    base_invariance = _prompt_invariance(bundles, audit_base)
    base_all_three = _all_three(base_rows)

    design_states_array = np.empty((2, 3, 2, 8192, 6), np.float32)
    audit_states_array = np.empty_like(design_states_array)
    direct_before = np.empty((3, 2, 2, 8192, 6), np.float32)
    direct_after = np.empty((3, 4, 2, 2, 8192, 6), np.float32)
    candidate_normalized = np.empty((3, 4, 2, 2, 8192, 6), np.float32)
    candidate_v5_predictions = np.empty((3, 4, 3, 8192, 6), np.float32)
    direction_rows = []
    candidate_rows = []

    for timestep_index, timestep in enumerate(CAPTURE_TIMESTEPS):
        v10._restore_lora_state(named_lora, zero_state)
        set_lora_enabled(model, True)
        set_frozen_base_eval_lora_train(model)
        scalar_tasks = []
        observed_order = []
        for scene_index, scene in enumerate(SCENES):
            design_states = torch.stack(
                [
                    trajectories["design"][scene][prompt]["states"][timestep][0]
                    for prompt in PROMPT_IDS
                ]
            ).to(args.device)
            audit_states = torch.stack(
                [
                    trajectories["audit"][scene][prompt]["states"][timestep][0]
                    for prompt in PROMPT_IDS
                ]
            ).to(args.device)
            design_states_array[scene_index, timestep_index] = (
                design_states.detach().cpu().numpy()
            )
            audit_states_array[scene_index, timestep_index] = (
                audit_states.detach().cpu().numpy()
            )
            set_lora_enabled(model, False)
            with torch.no_grad():
                frozen = _direct_prediction(
                    model, diffusion, design_states, timestep, kwargs[scene]
                )
            set_lora_enabled(model, True)
            prediction = _direct_prediction(
                model, diffusion, design_states, timestep, kwargs[scene]
            )
            if not torch.equal(prediction.detach(), frozen):
                raise AssertionError("zero-init on-policy prediction differs from v5r4")
            objective = dense_instance_all_sittable_objective(
                prediction, batches[scene], mean_tensor, std_tensor
            )
            direct_before[timestep_index, scene_index] = _physical(
                frozen, mean, std
            )
            role_tasks = (
                SUPPORT_TASK_WEIGHT * objective["per_instance_support"]
                + ACTIVE_TASK_WEIGHT * objective["per_instance_active"]
            )
            negative_tasks = _negative_tasks(objective["physical"], batches[scene])
            for prompt_index, prompt in enumerate(PROMPT_IDS):
                for role_index, role in enumerate(ROLES):
                    scalar_tasks.append(role_tasks[prompt_index, role_index])
                    observed_order.append(scene + "|" + prompt + "|" + role)
                scalar_tasks.append(negative_tasks[prompt_index])
                observed_order.append(scene + "|" + prompt + "|negative")
            scalar_tasks.append(_prompt_task(objective["physical"], batches[scene]))
            observed_order.append(scene + "|prompt_invariance")

        v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
        v5_tasks = (v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
        for index, role in enumerate(("chair", "bed", "whiteboard")):
            scalar_tasks.append(v5_tasks[index])
            observed_order.append("v5|" + role)
        if tuple(observed_order) != TASK_ORDER:
            raise AssertionError("Teacher-v10.3 task order changed")
        task_losses = torch.stack(scalar_tasks)
        gradients = flattened_task_gradients(task_losses, parameters)
        raw_gram = gradients.double() @ gradients.double().T
        gram = (0.5 * (raw_gram + raw_gram.T)).detach().cpu().numpy()
        weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
        direction = torch.from_numpy(weights).to(
            device=gradients.device, dtype=gradients.dtype
        ) @ gradients
        direction_norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(direction_norm) or float(direction_norm.item()) <= 1e-12:
            raise RuntimeError("Teacher-v10.3 common direction is zero/non-finite")
        direction = direction / direction_norm
        derivatives = (gradients @ direction).detach().cpu().tolist()
        direction_row = {
            "capture_timestep": timestep,
            "task_order": list(TASK_ORDER),
            "task_losses": [float(value) for value in task_losses.detach().cpu().tolist()],
            "gram": gram.tolist(),
            "weights": weights.tolist(),
            "direction_norm_before_unit": float(direction_norm.item()),
            "directional_derivatives": [float(value) for value in derivatives],
            "minimum_directional_derivative": float(min(derivatives)),
            "direction_sha256": tensor_sha256(direction),
        }
        _finite_tree(direction_row, "Teacher-v10.3 direction")
        direction_rows.append(direction_row)

        for radius_index, radius in enumerate(STEP_RADII):
            v10._restore_lora_state(named_lora, zero_state)
            set_lora_enabled(model, True)
            apply_flat_direction(parameters, direction, radius)
            model.eval()
            for scene_index, scene in enumerate(SCENES):
                design_states = torch.from_numpy(
                    design_states_array[scene_index, timestep_index]
                ).to(args.device)
                with torch.no_grad():
                    after = _direct_prediction(
                        model, diffusion, design_states, timestep, kwargs[scene]
                    )
                direct_after[
                    timestep_index, radius_index, scene_index
                ] = _physical(after, mean, std)
            with torch.no_grad():
                current_v5 = predict_xstart(
                    model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
                )
            current_v5_dense = (
                (current_v5 - v5_batch["x"])
                .square()
                .mean(dim=(1, 2))
                .cpu()
                .tolist()
            )
            candidate_v5_predictions[timestep_index, radius_index] = (
                current_v5.detach().cpu().numpy()
            )
            for scene_index, scene in enumerate(SCENES):
                for prompt_index, prompt in enumerate(PROMPT_IDS):
                    final = _resume_trajectory(
                        model,
                        diffusion,
                        bundles[scene],
                        PROMPTS[prompt],
                        trajectories["audit"][scene][prompt]["states"][timestep],
                        trajectories["audit"][scene][prompt]["rng_states"][timestep],
                        timestep,
                        args.device,
                    )
                    candidate_normalized[
                        timestep_index, radius_index, scene_index, prompt_index
                    ] = final.numpy()
            physical = np.clip(
                candidate_normalized[timestep_index, radius_index]
                * std.reshape(1, 1, 1, 6)
                + mean.reshape(1, 1, 1, 6),
                0.0,
                1.0,
            ).astype(np.float32)
            rows = _metric_rows(bundles, physical)
            invariance = _prompt_invariance(bundles, physical)
            maximum_delta = float(np.abs(physical - audit_base).max())
            checks = response_checks(
                base_rows=base_rows,
                candidate_rows=rows,
                base_prompt_invariance=base_invariance,
                candidate_prompt_invariance=invariance,
                base_v5_dense=base_v5_dense,
                candidate_v5_dense=current_v5_dense,
                directional_derivatives=derivatives,
                maximum_map_delta=maximum_delta,
            )
            name = "t{}_radius_{}".format(
                timestep, str(radius).replace("0.", "0p")
            )
            row = {
                "name": name,
                "capture_timestep": timestep,
                "radius": radius,
                "direction_sha256": direction_row["direction_sha256"],
                "directional_derivatives": [float(value) for value in derivatives],
                "base_rows": base_rows,
                "candidate_rows": rows,
                "base_pooled": base_pooled,
                "candidate_pooled": pooled_roles(rows),
                "base_prompt_invariance": base_invariance,
                "candidate_prompt_invariance": invariance,
                "base_v5_dense": [float(value) for value in base_v5_dense],
                "candidate_v5_dense": [float(value) for value in current_v5_dense],
                "base_all_three_diagnostic": base_all_three,
                "candidate_all_three_diagnostic": _all_three(rows),
                "maximum_final_map_delta": maximum_delta,
                "state_sha256": v10._state_sha256(named_lora),
                "candidate_maps_sha256": tensor_sha256(
                    torch.from_numpy(candidate_normalized[timestep_index, radius_index])
                ),
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(
                    key for key, passed in checks.items() if not passed
                ),
            }
            _finite_tree(row, "Teacher-v10.3 candidate")
            candidate_rows.append(row)
            print(
                "[ON-POLICY t={} R={}] eligible={} worst={:.6f}->{:.6f} v5={:+.3f}%".format(
                    timestep,
                    radius,
                    row["eligible"],
                    min(
                        role["soft_recall"]
                        for role in row["base_pooled"].values()
                    ),
                    min(
                        role["soft_recall"]
                        for role in row["candidate_pooled"].values()
                    ),
                    100.0 * (sum(current_v5_dense) / sum(base_v5_dense) - 1.0),
                ),
                flush=True,
            )
            del current_v5
            torch.cuda.empty_cache()
        del task_losses, gradients, raw_gram, direction

    eligible_order = rank_candidates(candidate_rows)
    selected_candidate = eligible_order[0] if eligible_order else None
    candidates = np.clip(
        candidate_normalized * std.reshape(1, 1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    maps_file = output_dir / "onpolicy_response_maps.npz"
    save_arrays: Dict[str, np.ndarray] = {
        "scene_ids": np.asarray(SCENES),
        "prompt_ids": np.asarray(PROMPT_IDS),
        "capture_timesteps": np.asarray(CAPTURE_TIMESTEPS, np.int64),
        "step_radii": np.asarray(STEP_RADII, np.float64),
        "design_seed_table": np.asarray(seed_tables["design"], np.int64),
        "audit_seed_table": np.asarray(seed_tables["audit"], np.int64),
        "design_states": design_states_array,
        "audit_states": audit_states_array,
        "audit_base_normalized": audit_base_normalized,
        "audit_base": audit_base,
        "direct_before": direct_before,
        "direct_after": direct_after,
        "candidates_normalized": candidate_normalized,
        "candidates": candidates,
        "v5_target": v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        "base_v5_prediction": base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        "candidate_v5_predictions": candidate_v5_predictions,
    }
    for scene in SCENES:
        prefix = "source" if scene == SCENES[0] else "audit"
        bundle = bundles[scene]
        for key, dtype in (
            ("xyz", np.float32),
            ("points", np.float32),
            ("instance_names", None),
            ("verified_object_mask", bool),
            ("verified_positive_mask", bool),
            ("unknown_sittable_mask", bool),
            ("explicit_negative_mask", bool),
            ("instance_targets", np.float32),
            ("all_target", np.float32),
        ):
            save_arrays[prefix + "_" + key] = np.asarray(bundle[key], dtype=dtype)
    atomic_savez(maps_file, **save_arrays)

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v103_onpolicy_response.py",
        "contract": PREPARE_ROOT
        / "relational_teacher_v103_onpolicy_response_contract.py",
        "summarizer": PREPARE_ROOT
        / "summarize_relational_teacher_v103_onpolicy_response.py",
        "v102_failure": failure_file,
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "metric_policy": metric_policy_file,
        "dense_instance_objective": PREPARE_ROOT
        / "relational_teacher_v102_dense_instance_objective.py",
        "common_descent": PREPARE_ROOT
        / "relational_teacher_v94_common_descent.py",
        "response_policy": policy_file,
        "response_maps": maps_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    checks = {
        "sealed_v102_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "policy_written_before_model_and_lora": True,
        "both_train_scene_arrays_loaded": True,
        "design_and_audit_seed_domains_disjoint": seed_tables["design"]
        != seed_tables["audit"],
        "all_12_partial_resumes_bitwise_exact": True,
        "exact_21_task_inventory": len(TASK_ORDER) == 21,
        "candidate_decision_uses_resumed_final_maps": True,
        "absolute_all_three_is_diagnostic_only": True,
        "teacher_forward_is_text_plus_scene_only": all(
            tuple(sorted(kwargs[scene])) == tuple(sorted(FORWARD_INPUT_KEYS))
            for scene in SCENES
        ),
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "at_least_one_onpolicy_response_is_admissible": selected_candidate
        is not None,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": MODEL_SEED,
        "device": args.device,
        "diffusion_steps": 500,
        "train_scenes": list(SCENES),
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "development_arrays_read": False,
        "paper_test_access": False,
        "prompt_ids": list(PROMPT_IDS),
        "forward_input_keys": sorted(FORWARD_INPUT_KEYS),
        "capture_timesteps": list(CAPTURE_TIMESTEPS),
        "step_radii": list(STEP_RADII),
        "design_seed_table": seed_tables["design"],
        "audit_seed_table": seed_tables["audit"],
        "trajectory_accounting": {
            "design_full_draws": 4,
            "audit_full_draws": 4,
            "audit_exact_partial_resumes": 12,
            "candidate_partial_resumes": 48,
        },
        "v102_binding_id": authority.get("binding_id"),
        "policy_id": POLICY_ID,
        "zero_state_sha256": zero_state_sha256,
        "lora": dict(lora_metadata(model)),
        "serialized_model_state": False,
        "base_rows": base_rows,
        "base_pooled": base_pooled,
        "base_prompt_invariance": base_invariance,
        "base_all_three_diagnostic": base_all_three,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "direction_rows": direction_rows,
        "candidates": candidate_rows,
        "eligible_selection_order": eligible_order,
        "selected_candidate": selected_candidate,
        "paths": path_strings,
        "path_sha256": path_hashes,
        "response_maps_sha256": path_hashes["response_maps"],
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "authorizes_onpolicy_multiupdate_calibration": status == "PASS",
        "authorizes_lora_capacity_or_placement_diagnosis": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v102_binding_id": report["v102_binding_id"],
            "policy_id": POLICY_ID,
            "design_seed_table": report["design_seed_table"],
            "audit_seed_table": report["audit_seed_table"],
            "selected_candidate": selected_candidate,
            "response_maps_sha256": report["response_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v10.3 report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    if list(output_dir.rglob("*.pt")) or list(output_dir.rglob("*.pth")):
        raise AssertionError("Teacher-v10.3 unexpectedly saved model state")
    print("[ONPOLICY_RESPONSE_{}] Teacher-v10.3".format(status))
    print("[PASS] fresh v5r4 two-scene design/audit trajectories completed")
    print("[PASS] 21-task directions and 12 resumed-final response candidates completed")
    print("[OK] selected candidate:", selected_candidate)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)
    del model, diffusion, zero_state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

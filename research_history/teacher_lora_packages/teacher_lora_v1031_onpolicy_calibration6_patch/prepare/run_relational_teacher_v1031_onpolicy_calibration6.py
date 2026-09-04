#!/usr/bin/env python3
"""Run Teacher-v10.3.1 two-scene on-policy K=3 calibration."""

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

import preflight_relational_teacher_v103_onpolicy_response as v103  # noqa: E402
import run_relational_teacher_v10_supervised_capacity as v10  # noqa: E402
import validate_relational_teacher_v10_supervised_capacity as v10v  # noqa: E402
from fewshot_cdm_common import load_split  # noqa: E402
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    set_frozen_base_eval_lora_train,
)
from relational_teacher_v102_dense_instance_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_INSTANCES,
)
from relational_teacher_v102_dense_instance_objective import (  # noqa: E402
    dense_instance_all_sittable_objective,
)
from relational_teacher_v103_onpolicy_response_contract import (  # noqa: E402
    ACTIVE_TASK_WEIGHT,
    CAPTURE_TIMESTEPS,
    FW_ITERATIONS as V103_FW_ITERATIONS,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED as V103_MODEL_SEED,
    POLICY_ID as V103_POLICY_ID,
    PROMPT_IDS,
    ROLES,
    SCENES,
    SUPPORT_TASK_WEIGHT,
    TASK_ORDER,
    canonical_sha256,
    response_checks,
)
from relational_teacher_v1031_onpolicy_calibration6_contract import (  # noqa: E402
    FW_ITERATIONS,
    GENERATION_COUNT,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    SCHEMA,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    SHORTLIST_LIMIT,
    STEP_RADIUS,
    UPDATE_COUNT,
    V103_SCHEMA,
    all_three_counts,
    calibration_checks,
    rank_shortlist,
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


METRIC_KEYS = (
    "soft_recall",
    "active_support_mae",
    "topk_overlap",
    "hotspot_centroid_distance_xy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v103-report", type=Path, required=True)
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


def _validate_v103(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V103_SCHEMA
        or value.get("status") != "PASS"
        or value.get("selected_candidate") != "t150_radius_0p004"
        or value.get("policy_id") != V103_POLICY_ID
        or value.get("authorizes_onpolicy_multiupdate_calibration") is not True
        or value.get("authorizes_checkpoint") is not False
        or value.get("serialized_model_state") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("failed_checks") != []
    ):
        raise ValueError("Teacher-v10.3 PASS does not authorize calibration")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v10.3 path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10.3 bound file changed: " + str(name))
    selected = [
        row for row in value["candidates"] if row.get("name") == value["selected_candidate"]
    ]
    if (
        len(selected) != 1
        or selected[0].get("capture_timestep") != SELECTED_TIMESTEP
        or float(selected[0].get("radius")) != SELECTED_RADIUS
        or selected[0].get("eligible") is not True
        or selected[0].get("failed_checks") != []
    ):
        raise ValueError("Teacher-v10.3 selected response changed")
    if list(path.parent.rglob("*.pt")) or list(path.parent.rglob("*.pth")):
        raise ValueError("Teacher-v10.3 response contains a forbidden checkpoint")
    return value


def _bound_path(authority: Mapping[str, object], name: str) -> Path:
    path = Path(str(authority["paths"][name])).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != authority["path_sha256"][name]:
        raise ValueError("Teacher-v10.3 bound input changed: " + name)
    return path


def _raw_rows(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, list]:
    if predictions.shape != (2, GENERATION_COUNT, 2, 8192, 6):
        raise ValueError("Teacher-v10.3.1 K=3 map shape changed")
    return {
        scene: [
            [
                v10._metrics(
                    bundles[scene],
                    predictions[scene_index, generation, prompt_index],
                )
                for prompt_index in range(2)
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene_index, scene in enumerate(SCENES)
    }


def _pooled_rows(raw: Mapping[str, Sequence[Sequence[Mapping[str, object]]]]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for scene in SCENES:
        result[scene] = {}
        for prompt_index, prompt_id in enumerate(PROMPT_IDS):
            instance_rows = {}
            for name in EXPECTED_INSTANCES[scene]:
                instance_rows[name] = {
                    key: float(
                        np.mean(
                            [
                                raw[scene][generation][prompt_index]["instances"][name][key]
                                for generation in range(GENERATION_COUNT)
                            ]
                        )
                    )
                    for key in METRIC_KEYS
                }
            result[scene][prompt_id] = {
                "instances": instance_rows,
                "explicit_negative_mean": float(
                    np.mean(
                        [
                            raw[scene][generation][prompt_index]["explicit_negative_mean"]
                            for generation in range(GENERATION_COUNT)
                        ]
                    )
                ),
                "explicit_negative_max": float(
                    max(
                        raw[scene][generation][prompt_index]["explicit_negative_max"]
                        for generation in range(GENERATION_COUNT)
                    )
                ),
            }
    return result


def _prompt_invariance(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, float]:
    result = {}
    for scene_index, scene in enumerate(SCENES):
        mask = np.asarray(bundles[scene]["verified_positive_mask"], bool)
        result[scene] = float(
            np.mean(
                [
                    np.square(
                        predictions[scene_index, generation, 0, mask]
                        - predictions[scene_index, generation, 1, mask]
                    ).mean()
                    for generation in range(GENERATION_COUNT)
                ]
            )
        )
    return result


def _presence(raw: Mapping[str, Sequence[Sequence[Mapping[str, object]]]]) -> Dict[str, list]:
    return {
        scene: [
            [
                simultaneous_presence_checks(
                    raw[scene][generation][prompt_index],
                    **ABSOLUTE_PRESENCE_LIMITS,
                )
                for prompt_index in range(2)
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene in SCENES
    }


def _state_copy(named: Mapping[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in named.items()}


def _restore_state(
    named: Mapping[str, torch.nn.Parameter], state: Mapping[str, torch.Tensor]
) -> None:
    if set(named) != set(state):
        raise ValueError("Teacher-v10.3.1 LoRA state inventory changed")
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(state[name].to(parameter.device))


def _apply_direction_on_lora_device(
    parameters: list[torch.nn.Parameter],
    direction: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Move a CPU-aggregated direction to the exact LoRA device/dtype."""

    if not parameters:
        raise ValueError("Teacher-v10.3.1 has no LoRA parameters")
    reference = parameters[0]
    if any(
        parameter.device != reference.device or parameter.dtype != reference.dtype
        for parameter in parameters
    ):
        raise RuntimeError("Teacher-v10.3.1 LoRA parameters span devices/dtypes")
    if not direction.is_floating_point():
        raise TypeError("Teacher-v10.3.1 direction is not floating point")
    applied_direction = direction.to(
        device=reference.device,
        dtype=reference.dtype,
    )
    if (
        applied_direction.device != reference.device
        or applied_direction.dtype != reference.dtype
    ):
        raise RuntimeError("Teacher-v10.3.1 direction device/dtype transfer failed")
    apply_flat_direction(parameters, applied_direction, radius)
    return applied_direction


def _reconstruct_v103_direction(
    *,
    model,
    diffusion,
    arrays: Mapping[str, np.ndarray],
    batches: Mapping[str, Mapping[str, torch.Tensor]],
    kwargs: Mapping[str, Mapping[str, object]],
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    v5_batch: Mapping[str, object],
    v5_t: torch.Tensor,
    v5_kwargs: Mapping[str, object],
    v5_noise: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    selected: Mapping[str, object],
    device: str,
) -> torch.Tensor:
    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    tasks = []
    observed = []
    for scene_index, scene in enumerate(SCENES):
        states = torch.from_numpy(
            np.asarray(arrays["design_states"][scene_index, timestep_index], np.float32)
        ).to(device)
        prediction = v103._direct_prediction(
            model, diffusion, states, SELECTED_TIMESTEP, kwargs[scene]
        )
        objective = dense_instance_all_sittable_objective(
            prediction, batches[scene], mean_tensor, std_tensor
        )
        roles = (
            SUPPORT_TASK_WEIGHT * objective["per_instance_support"]
            + ACTIVE_TASK_WEIGHT * objective["per_instance_active"]
        )
        negatives = v103._negative_tasks(objective["physical"], batches[scene])
        for prompt_index, prompt_id in enumerate(PROMPT_IDS):
            for role_index, role in enumerate(ROLES):
                tasks.append(roles[prompt_index, role_index])
                observed.append(scene + "|" + prompt_id + "|" + role)
            tasks.append(negatives[prompt_index])
            observed.append(scene + "|" + prompt_id + "|negative")
        tasks.append(v103._prompt_task(objective["physical"], batches[scene]))
        observed.append(scene + "|prompt_invariance")
    v5_prediction = predict_xstart(
        model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
    )
    v5_tasks = (v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
    for index, role in enumerate(("chair", "bed", "whiteboard")):
        tasks.append(v5_tasks[index])
        observed.append("v5|" + role)
    if tuple(observed) != TASK_ORDER:
        raise AssertionError("Teacher-v10.3 reconstruction task order changed")
    task_losses = torch.stack(tasks)
    gradients = flattened_task_gradients(task_losses, parameters)
    gram = (gradients.double() @ gradients.double().T).detach().cpu().numpy()
    gram = 0.5 * (gram + gram.T)
    weights = frank_wolfe_min_norm_weights(gram, V103_FW_ITERATIONS)
    direction = torch.from_numpy(weights).to(
        device=gradients.device, dtype=gradients.dtype
    ) @ gradients
    direction = direction / torch.linalg.vector_norm(direction)
    sealed_direction = [
        row
        for row in selected["direction_rows"]
        if row.get("capture_timestep") == SELECTED_TIMESTEP
    ]
    if (
        len(sealed_direction) != 1
        or not np.allclose(gram, sealed_direction[0]["gram"], rtol=2e-6, atol=2e-7)
        or not np.allclose(weights, sealed_direction[0]["weights"], rtol=2e-6, atol=2e-7)
        or tensor_sha256(direction) != sealed_direction[0]["direction_sha256"]
    ):
        raise AssertionError("Teacher-v10.3 selected direction did not reproduce")
    return direction


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v10.3.1 calibration requires CUDA")
    report_file = args.v103_report.expanduser().resolve()
    authority = _validate_v103(report_file)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v10.3.1 output")
    # Model and LoRA initialization must reproduce the sealed v10.3 state
    # bit-for-bit.  The new calibration seed is used only by the explicitly
    # derived design/audit seed tables below.
    configure_reproducibility(V103_MODEL_SEED)
    output_dir.mkdir(parents=True)
    policy_file = output_dir / "onpolicy_calibration_policy.json"
    atomic_write_json(policy_file, POLICY)
    if read_json(policy_file) != POLICY or canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("Teacher-v10.3.1 policy hash changed")

    response_maps = _bound_path(authority, "response_maps")
    response_arrays = v10v._load_npz(response_maps)
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
        raise ValueError("Teacher-v10.3.1 scene inventory changed")
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
    v5_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(
        v5_batch["x"].shape, V103_MODEL_SEED + 900000, args.device
    )

    cfg = compose_cdm_config(500, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )

    modules = install_lora(model, LORA_RANK, LORA_ALPHA, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("Teacher-v10.3.1 LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    parameters = list(named_lora.values())
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    zero_state_sha256 = v10._state_sha256(named_lora)
    if zero_state_sha256 != authority.get("zero_state_sha256"):
        raise AssertionError("fresh zero-output LoRA differs from Teacher-v10.3")
    direction = _reconstruct_v103_direction(
        model=model,
        diffusion=diffusion,
        arrays=response_arrays,
        batches=batches,
        kwargs=kwargs,
        mean_tensor=mean_tensor,
        std_tensor=std_tensor,
        v5_batch=v5_batch,
        v5_t=v5_t,
        v5_kwargs=v5_kwargs,
        v5_noise=v5_noise,
        parameters=parameters,
        selected=authority,
        device=args.device,
    )
    _apply_direction_on_lora_device(parameters, direction, SELECTED_RADIUS)
    selected_row = next(
        row for row in authority["candidates"] if row["name"] == authority["selected_candidate"]
    )
    start_state_sha256 = v10._state_sha256(named_lora)
    if start_state_sha256 != selected_row["state_sha256"]:
        raise AssertionError("Teacher-v10.3 selected LoRA state did not reproduce")
    with torch.no_grad():
        selected_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    radius_index = list(authority["step_radii"]).index(SELECTED_RADIUS)
    if not np.array_equal(
        selected_v5_prediction.detach().cpu().numpy(),
        response_arrays["candidate_v5_predictions"][timestep_index, radius_index],
    ):
        raise AssertionError("Teacher-v10.3 selected v5 response did not reproduce")
    print("[V103_RECONSTRUCTION_PASS] exact t150 radius 0.004 state", flush=True)

    caches: Dict[str, Dict[str, list]] = {"design": {}, "audit": {}}
    seed_tables: Dict[str, list] = {"design": [], "audit": []}
    design_states_array = np.empty((2, GENERATION_COUNT, 2, 8192, 6), np.float32)
    audit_states_array = np.empty_like(design_states_array)
    start_normalized = np.empty_like(design_states_array)
    for domain in ("design", "audit"):
        for scene_index, scene in enumerate(SCENES):
            caches[domain][scene] = []
            scene_seeds = []
            for generation in range(GENERATION_COUNT):
                generation_cache = []
                generation_seeds = []
                for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                    seeds = stable_seed_pair(domain, scene, prompt_id, generation)
                    generation_seeds.append(list(seeds))
                    trajectory = v103._capture_trajectory(
                        model,
                        diffusion,
                        bundles[scene],
                        PROMPTS[prompt_id],
                        seeds[0],
                        seeds[1],
                        args.device,
                        not args.no_progress,
                    )
                    state = trajectory["states"][SELECTED_TIMESTEP]
                    rng_state = trajectory["rng_states"][SELECTED_TIMESTEP]
                    generation_cache.append((state, rng_state))
                    target = design_states_array if domain == "design" else audit_states_array
                    target[scene_index, generation, prompt_index] = state[0].numpy()
                    if domain == "audit":
                        start_normalized[scene_index, generation, prompt_index] = trajectory[
                            "final"
                        ].numpy()
                        repeated = v103._resume_trajectory(
                            model,
                            diffusion,
                            bundles[scene],
                            PROMPTS[prompt_id],
                            state,
                            rng_state,
                            SELECTED_TIMESTEP,
                            args.device,
                        )
                        if not torch.equal(repeated, trajectory["final"]):
                            raise AssertionError("Teacher-v10.3.1 audit resume is not exact")
                    print(
                        "[CALIBRATION-{}] scene={} generation={} prompt={}".format(
                            domain.upper(), scene, generation, prompt_id
                        ),
                        flush=True,
                    )
                caches[domain][scene].append(generation_cache)
                scene_seeds.append(generation_seeds)
            seed_tables[domain].append(scene_seeds)
    flat_design = {
        tuple(pair)
        for scene in seed_tables["design"]
        for generation in scene
        for pair in generation
    }
    flat_audit = {
        tuple(pair)
        for scene in seed_tables["audit"]
        for generation in scene
        for pair in generation
    }
    v103_seeds = {
        tuple(pair)
        for domain in (authority["design_seed_table"], authority["audit_seed_table"])
        for scene in domain
        for pair in scene
    }
    if flat_design & flat_audit or flat_design & v103_seeds or flat_audit & v103_seeds:
        raise AssertionError("Teacher-v10.3.1 seed domains overlap")
    print("[TRAJECTORY_REPRODUCTION_PASS] disjoint K=3 audit resumes exact", flush=True)

    start_physical = np.clip(
        start_normalized * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    start_raw = _raw_rows(bundles, start_physical)
    start_pooled = _pooled_rows(start_raw)
    start_invariance = _prompt_invariance(bundles, start_physical)
    start_presence = _presence(start_raw)
    start_v5_dense = (
        (selected_v5_prediction - v5_batch["x"])
        .square()
        .mean(dim=(1, 2))
        .cpu()
        .tolist()
    )

    candidates_normalized = np.empty(
        (UPDATE_COUNT, 2, GENERATION_COUNT, 2, 8192, 6), np.float32
    )
    candidates_physical = np.empty_like(candidates_normalized)
    candidate_v5_predictions = np.empty((UPDATE_COUNT, 3, 8192, 6), np.float32)
    direction_rows = []
    monitor_rows = []
    state_snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    for step in MONITOR_STEPS:
        set_frozen_base_eval_lora_train(model)
        pre_state_sha256 = v10._state_sha256(named_lora)
        gradient_groups = []
        recorded_task_losses = []
        for scene_index, scene in enumerate(SCENES):
            generation_tasks = []
            for generation in range(GENERATION_COUNT):
                states = torch.from_numpy(
                    design_states_array[scene_index, generation]
                ).to(args.device)
                prediction = v103._direct_prediction(
                    model, diffusion, states, SELECTED_TIMESTEP, kwargs[scene]
                )
                objective = dense_instance_all_sittable_objective(
                    prediction, batches[scene], mean_tensor, std_tensor
                )
                roles = (
                    SUPPORT_TASK_WEIGHT * objective["per_instance_support"]
                    + ACTIVE_TASK_WEIGHT * objective["per_instance_active"]
                )
                negatives = v103._negative_tasks(objective["physical"], batches[scene])
                generation_tasks.append(
                    {
                        "roles": roles,
                        "negatives": negatives,
                        "invariance": v103._prompt_task(
                            objective["physical"], batches[scene]
                        ),
                    }
                )
            scalar_tasks = []
            for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                for role_index, role in enumerate(ROLES):
                    scalar_tasks.append(
                        torch.stack(
                            [row["roles"][prompt_index, role_index] for row in generation_tasks]
                        ).mean()
                    )
                scalar_tasks.append(
                    torch.stack(
                        [row["negatives"][prompt_index] for row in generation_tasks]
                    ).mean()
                )
            scalar_tasks.append(
                torch.stack([row["invariance"] for row in generation_tasks]).mean()
            )
            scene_losses = torch.stack(scalar_tasks)
            gradient_groups.append(
                flattened_task_gradients(scene_losses, parameters).detach().cpu()
            )
            recorded_task_losses.extend(
                float(value) for value in scene_losses.detach().cpu().tolist()
            )
            del generation_tasks, scalar_tasks, scene_losses
        live_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
        live_v5_tasks = (live_v5 - v5_batch["x"]).square().mean(dim=(1, 2))
        gradient_groups.append(
            flattened_task_gradients(live_v5_tasks, parameters).detach().cpu()
        )
        recorded_task_losses.extend(
            float(value) for value in live_v5_tasks.detach().cpu().tolist()
        )
        gradients = torch.cat(gradient_groups, dim=0)
        if gradients.shape[0] != len(TASK_ORDER) or len(recorded_task_losses) != len(TASK_ORDER):
            raise AssertionError("Teacher-v10.3.1 task inventory changed")
        gram = (gradients.double() @ gradients.double().T).detach().cpu().numpy()
        gram = 0.5 * (gram + gram.T)
        weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
        direction = torch.from_numpy(weights).to(
            device=gradients.device, dtype=gradients.dtype
        ) @ gradients
        direction_norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(direction_norm) or float(direction_norm.item()) <= 1e-12:
            raise RuntimeError("Teacher-v10.3.1 common direction is zero/non-finite")
        direction = direction / direction_norm
        derivatives = (gradients @ direction).detach().cpu().tolist()
        applied_direction = _apply_direction_on_lora_device(
            parameters, direction, STEP_RADIUS
        )
        if tensor_sha256(applied_direction) != tensor_sha256(direction):
            raise AssertionError("Teacher-v10.3.1 direction changed during device transfer")
        post_state_sha256 = v10._state_sha256(named_lora)
        state_snapshots[step] = _state_copy(named_lora)
        direction_row = {
            "step": step,
            "task_order": list(TASK_ORDER),
            "task_losses": recorded_task_losses,
            "gram": gram.tolist(),
            "weights": weights.tolist(),
            "direction_norm_before_unit": float(direction_norm.item()),
            "directional_derivatives": [float(value) for value in derivatives],
            "minimum_directional_derivative": float(min(derivatives)),
            "direction_sha256": tensor_sha256(direction),
            "pre_state_sha256": pre_state_sha256,
            "post_state_sha256": post_state_sha256,
        }
        _finite_tree(direction_row, "Teacher-v10.3.1 direction")
        direction_rows.append(direction_row)

        model.eval()
        current_normalized = np.empty_like(start_normalized)
        for scene_index, scene in enumerate(SCENES):
            for generation in range(GENERATION_COUNT):
                for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                    state, rng_state = caches["audit"][scene][generation][prompt_index]
                    final = v103._resume_trajectory(
                        model,
                        diffusion,
                        bundles[scene],
                        PROMPTS[prompt_id],
                        state,
                        rng_state,
                        SELECTED_TIMESTEP,
                        args.device,
                    )
                    current_normalized[
                        scene_index, generation, prompt_index
                    ] = final.numpy()
                    print(
                        "[MONITOR {}] scene={} generation={} prompt={}".format(
                            step, scene, generation, prompt_id
                        ),
                        flush=True,
                    )
        with torch.no_grad():
            current_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        current_v5_dense = (
            (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        current_physical = np.clip(
            current_normalized * std.reshape(1, 1, 1, 1, 6)
            + mean.reshape(1, 1, 1, 1, 6),
            0.0,
            1.0,
        ).astype(np.float32)
        candidates_normalized[step - 1] = current_normalized
        candidates_physical[step - 1] = current_physical
        candidate_v5_predictions[step - 1] = current_v5.detach().cpu().numpy()
        current_raw = _raw_rows(bundles, current_physical)
        current_pooled = _pooled_rows(current_raw)
        current_invariance = _prompt_invariance(bundles, current_physical)
        current_presence = _presence(current_raw)
        pooled_checks = response_checks(
            base_rows=start_pooled,
            candidate_rows=current_pooled,
            base_prompt_invariance=start_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=start_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=derivatives,
            maximum_map_delta=float(np.abs(current_physical - start_physical).max()),
        )
        checks = calibration_checks(
            start_rows=start_raw,
            candidate_rows=current_raw,
            pooled_response_checks=pooled_checks,
            pre_state_sha256=pre_state_sha256,
            post_state_sha256=post_state_sha256,
        )
        row = {
            "step": step,
            "pre_state_sha256": pre_state_sha256,
            "post_state_sha256": post_state_sha256,
            "direction_sha256": direction_row["direction_sha256"],
            "start_raw": start_raw,
            "candidate_raw": current_raw,
            "start_pooled": start_pooled,
            "candidate_pooled": current_pooled,
            "start_prompt_invariance": start_invariance,
            "candidate_prompt_invariance": current_invariance,
            "start_v5_dense": [float(value) for value in start_v5_dense],
            "candidate_v5_dense": [float(value) for value in current_v5_dense],
            "presence": current_presence,
            "all_three_counts": all_three_counts(current_presence),
            "pooled_response_checks": pooled_checks,
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(
                name for name, passed in checks.items() if not passed
            ),
        }
        _finite_tree(row, "Teacher-v10.3.1 monitor")
        monitor_rows.append(row)
        print(
            "[CALIBRATION {}] eligible={} all-three={} v5={:+.3f}%".format(
                step,
                row["eligible"],
                row["all_three_counts"],
                100.0 * (sum(current_v5_dense) / sum(start_v5_dense) - 1.0),
            ),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    shortlisted_steps = rank_shortlist(monitor_rows)
    shortlisted_state_sha256 = {
        str(step): monitor_rows[step - 1]["post_state_sha256"]
        for step in shortlisted_steps
    }
    for step in shortlisted_steps:
        _restore_state(named_lora, state_snapshots[step])
        if v10._state_sha256(named_lora) != shortlisted_state_sha256[str(step)]:
            raise AssertionError("Teacher-v10.3.1 shortlisted state changed in memory")
    _restore_state(named_lora, state_snapshots[UPDATE_COUNT])

    maps_file = output_dir / "onpolicy_calibration_maps.npz"
    save_arrays: Dict[str, np.ndarray] = {
        "scene_ids": np.asarray(SCENES),
        "prompt_ids": np.asarray(PROMPT_IDS),
        "design_seed_table": np.asarray(seed_tables["design"], np.int64),
        "audit_seed_table": np.asarray(seed_tables["audit"], np.int64),
        "design_states": design_states_array,
        "audit_states": audit_states_array,
        "start_normalized": start_normalized,
        "start": start_physical,
        "candidates_normalized": candidates_normalized,
        "candidates": candidates_physical,
        "v5_target": v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        "base_v5_prediction": base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        "start_v5_prediction": selected_v5_prediction.detach().cpu().numpy().astype(np.float32),
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
        / "validate_relational_teacher_v1031_onpolicy_calibration6.py",
        "contract": PREPARE_ROOT
        / "relational_teacher_v1031_onpolicy_calibration6_contract.py",
        "summarizer": PREPARE_ROOT
        / "summarize_relational_teacher_v1031_onpolicy_calibration6.py",
        "v103_report": report_file,
        "v103_maps": response_maps,
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
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "calibration_policy": policy_file,
        "calibration_maps": maps_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    checks = {
        "sealed_v103_pass_and_selected_response_bound": True,
        "fresh_v5r4_zero_init": True,
        "selected_v103_state_exactly_reconstructed": True,
        "selected_v103_v5_response_exactly_reconstructed": True,
        "policy_written_before_scene_arrays_and_model": True,
        "design_audit_and_v103_seed_domains_disjoint": True,
        "all_12_audit_partial_resumes_bitwise_exact": True,
        "six_21_task_common_descent_updates_completed": len(direction_rows)
        == UPDATE_COUNT,
        "six_actual_two_scene_k3_monitors_completed": len(monitor_rows)
        == UPDATE_COUNT,
        "shortlisted_states_verified_in_memory": bool(shortlisted_steps),
        "absolute_all_three_is_diagnostic_only": True,
        "teacher_forward_is_text_plus_scene_only": all(
            tuple(sorted(kwargs[scene])) == tuple(sorted(FORWARD_INPUT_KEYS))
            for scene in SCENES
        ),
        "only_two_train_scene_arrays_loaded": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_calibration_state_is_admissible": bool(shortlisted_steps),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": MODEL_SEED,
        "initialization_seed": V103_MODEL_SEED,
        "device": args.device,
        "diffusion_steps": 500,
        "train_scenes": list(SCENES),
        "development_scene_metadata_only": "room_0201",
        "development_arrays_read": False,
        "paper_test_access": False,
        "prompt_ids": list(PROMPT_IDS),
        "generation_count": GENERATION_COUNT,
        "selected_timestep": SELECTED_TIMESTEP,
        "selected_radius": SELECTED_RADIUS,
        "update_count": UPDATE_COUNT,
        "monitor_steps": list(MONITOR_STEPS),
        "step_radius": STEP_RADIUS,
        "shortlist_limit": SHORTLIST_LIMIT,
        "design_seed_table": seed_tables["design"],
        "audit_seed_table": seed_tables["audit"],
        "v103_binding_id": authority["binding_id"],
        "v103_selected_candidate": authority["selected_candidate"],
        "policy_id": POLICY_ID,
        "zero_state_sha256": zero_state_sha256,
        "start_state_sha256": start_state_sha256,
        "final_state_sha256": monitor_rows[-1]["post_state_sha256"],
        "lora": dict(lora_metadata(model)),
        "serialized_model_state": False,
        "start_raw": start_raw,
        "start_pooled": start_pooled,
        "start_prompt_invariance": start_invariance,
        "start_presence": start_presence,
        "start_all_three_counts": all_three_counts(start_presence),
        "start_v5_dense": [float(value) for value in start_v5_dense],
        "direction_rows": direction_rows,
        "monitor_rows": monitor_rows,
        "shortlisted_steps": shortlisted_steps,
        "shortlisted_state_sha256": shortlisted_state_sha256,
        "paths": path_strings,
        "path_sha256": path_hashes,
        "calibration_maps_sha256": path_hashes["calibration_maps"],
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "authorizes_extended_onpolicy_training": status == "PASS",
        "authorizes_lora_capacity_or_placement_diagnosis": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v103_binding_id": report["v103_binding_id"],
            "policy_id": POLICY_ID,
            "start_state_sha256": start_state_sha256,
            "shortlisted_steps": shortlisted_steps,
            "shortlisted_state_sha256": shortlisted_state_sha256,
            "calibration_maps_sha256": report["calibration_maps_sha256"],
            "audit_seed_table": report["audit_seed_table"],
            "status": status,
        }
    )
    _finite_tree(report, "Teacher-v10.3.1 report")
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, report)
    if list(output_dir.rglob("*.pt")) or list(output_dir.rglob("*.pth")):
        raise AssertionError("Teacher-v10.3.1 unexpectedly saved model state")
    print("[ONPOLICY_CALIBRATION6_{}] Teacher-v10.3.1".format(status))
    print("[PASS] exact v10.3 start and six K=3 monitored updates completed")
    print("[PASS] shortlisted states verified in memory; no checkpoint written")
    print("[OK] shortlisted steps:", shortlisted_steps)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", summary_file)


if __name__ == "__main__":
    main()

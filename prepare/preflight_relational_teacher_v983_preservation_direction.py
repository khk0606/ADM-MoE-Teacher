#!/usr/bin/env python3
"""Teacher-v9.8.3 preservation-aware rollout-state direction preflight."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

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
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from preflight_relational_teacher_v98_rollout_state_response import (  # noqa: E402
    _capture_trajectory,
    _direct_prediction,
    _resume_trajectory,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    PROMPTS,
    atomic_savez,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v9_lora_objective import physical_prediction  # noqa: E402
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
from relational_teacher_v91_active_support_objective import corrected_objective  # noqa: E402
from relational_teacher_v94_common_descent import (  # noqa: E402
    apply_flat_direction,
    flattened_task_gradients,
    frank_wolfe_min_norm_weights,
)
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    continuous_presence_checks,
    stable_rollout_seeds,
)
from relational_teacher_v98_rollout_state_contract import (  # noqa: E402
    CAPTURE_TIMESTEPS,
)
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    response6_checks,
)
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    POLICY_ID as V982_POLICY_ID,
    calibration_design_seeds,
)
from relational_teacher_v983_preservation_direction_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    HELDOUT_TRAIN_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PREFLIGHT_TAG,
    PROMPT_IDS,
    SCHEMA,
    SELECTED_TIMESTEP,
    STEP_RADII,
    TASK_ORDER,
    TRAIN_SCENE,
    V982_SCHEMA,
    canonical_sha256,
    pooled_object_metrics,
    preservation_checks,
    rank_candidates,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
    _metrics,
    _prompt_invariance,
    _restore_lora_state,
    _validate_preflight,
)
from run_relational_teacher_v982_rollout_state_calibration6 import (  # noqa: E402
    _state_sha256,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-calibration-summary", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(label + " contains NaN/Inf")


def _validate_bound_files(value: Mapping[str, object], label: str) -> None:
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError(label + " path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError(label + " bound file changed: " + str(name))


def _validate_v982(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    expected_nested = {
        1: [],
        2: ["v5_fixed_probe_retained_1pct"],
        3: ["v5_fixed_probe_retained_1pct"],
        4: ["generation_2_negative_mean_retained", "v5_fixed_probe_retained_1pct"],
        5: [
            "generation_0_negative_mean_retained",
            "generation_1_negative_mean_retained",
            "generation_2_negative_mean_retained",
            "v5_fixed_probe_retained_1pct",
        ],
        6: [
            "generation_0_negative_mean_retained",
            "generation_1_negative_mean_retained",
            "generation_2_negative_mean_retained",
            "v5_fixed_probe_retained_1pct",
        ],
    }
    if (
        value.get("schema") != V982_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("policy_id") != V982_POLICY_ID
        or value.get("shortlisted_steps") != []
        or value.get("failed_checks")
        != ["at_least_one_post_first_update_is_admissible"]
        or value.get("authorizes_two_scene_rollout_state_calibration_preflight")
        is not False
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.2 failure authority changed")
    rows = value.get("monitor_rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("Teacher-v9.8.2 monitor inventory changed")
    for row in rows:
        step = int(row.get("step", 0))
        failures = sorted(
            name for name, passed in row.get("response_checks", {}).items() if not passed
        )
        if failures != expected_nested.get(step):
            raise ValueError("Teacher-v9.8.2 nested failure diagnosis changed")
    _validate_bound_files(value, "Teacher-v9.8.2")
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.2 failure contains forbidden model state")
    return value


def _same_bound_arg(argument: Path, authority: Mapping[str, object], name: str) -> Path:
    path = argument.expanduser().resolve()
    expected = Path(str(authority["paths"][name])).expanduser().resolve()
    if (
        path != expected
        or not path.is_file()
        or sha256_file(path) != authority["path_sha256"][name]
    ):
        raise ValueError(name + " differs from sealed Teacher-v9.8.2")
    return path


def _rows(bundle: Mapping[str, object], values: np.ndarray) -> list:
    if values.shape != (3, 2, 8192, 6):
        raise ValueError("K=3 preservation map shape changed")
    return [
        [_metrics(bundle, values[generation, prompt]) for prompt in range(2)]
        for generation in range(3)
    ]


def _invariance(values: np.ndarray, mask: np.ndarray) -> list[float]:
    return [
        _prompt_invariance(values[generation], mask)
        for generation in range(3)
    ]


def _sit_tasks(
    model,
    diffusion,
    states: torch.Tensor,
    kwargs: Mapping[str, object],
    batch: Mapping[str, object],
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        set_lora_enabled(model, False)
        frozen = _direct_prediction(
            model, diffusion, states, SELECTED_TIMESTEP, kwargs
        )
    set_lora_enabled(model, True)
    prediction = _direct_prediction(
        model, diffusion, states, SELECTED_TIMESTEP, kwargs
    )
    objective = corrected_objective(
        prediction, frozen, batch, mean_tensor, std_tensor
    )
    tasks = (
        objective["per_instance_primary"]
        + 2.0 * objective["per_instance_active_support"]
        + 0.25 * objective["per_instance_ranking"]
    ).reshape(-1)
    return tasks, prediction, frozen


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8.3 preservation preflight requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != MODEL_SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.3 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.3 output")
    configure_reproducibility(args.seed)

    failed_file = args.failed_calibration_summary.expanduser().resolve()
    failed = _validate_v982(failed_file)
    failed_arrays_file = Path(str(failed["paths"]["calibration6_maps"])).resolve()
    failed_arrays = _load_npz(failed_arrays_file)
    response_file = Path(str(failed["paths"]["response6_summary"])).resolve()
    response = read_json(response_file)
    response_maps_file = Path(str(failed["paths"]["response6_maps"])).resolve()
    response_arrays = _load_npz(response_maps_file)
    v98_file = Path(str(failed["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_maps_file = Path(str(failed["paths"]["v98_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)
    v97_file = Path(str(failed["paths"]["v97_summary"])).resolve()
    v97 = read_json(v97_file)
    v97_maps_file = Path(str(failed["paths"]["v97_maps"])).resolve()
    v97_arrays = _load_npz(v97_maps_file)
    if failed.get("v981_binding_id") != response.get("binding_id"):
        raise ValueError("v9.8.1 binding changed")
    if failed.get("v98_binding_id") != v98.get("binding_id"):
        raise ValueError("v9.8 binding changed")
    if failed.get("v97_binding_id") != v97.get("binding_id"):
        raise ValueError("v9.7 binding changed")
    if not np.array_equal(
        failed_arrays["candidates_normalized"][0],
        response_arrays["candidate_normalized"],
    ):
        raise ValueError("v9.8.2 update-1 artifact changed")

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_root.is_dir():
        raise FileNotFoundError("a sealed dataset root is absent")
    index_file = _same_bound_arg(args.index, failed, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, failed, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, failed, "stats_file")
    original_checkpoint = _same_bound_arg(
        args.original_checkpoint, failed, "original_checkpoint"
    )
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, failed, "v5_checkpoint")
    evidence_file = _same_bound_arg(
        args.v5_evidence_report, failed, "v5_evidence_report"
    )
    if dataset_root != index_file.parent:
        raise ValueError("dataset root differs from sealed dataset index")
    if source_root != Path(str(failed["paths"]["source_dataset_index"])).resolve().parent:
        raise ValueError("source dataset root differs from sealed source index")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")
    preflight_file = Path(str(failed["paths"]["preflight_report"])).resolve()
    preflight, original_policy = _validate_preflight(preflight_file)
    if preflight.get("binding_id") != failed.get("preflight_binding_id"):
        raise ValueError("Teacher-v9 preflight binding changed")
    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE or failed.get("scene_binding") != response.get(
        "scene_binding"
    ):
        raise ValueError("room_0101 binding changed")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "preservation_direction_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("preservation direction policy hash changed")

    mean, std = load_stats(stats_file)
    batch = _build_batch(bundle, mean, std, args.device)
    kwargs = _kwargs(batch)
    if tuple(sorted(kwargs)) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher forward input changed")
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    split = load_split(split_file)
    v5_rows = load_v5_rows(v5_root, split, "train", mean, std, 4.0, 16.0, 0.7)
    if Counter(str(row["target"]) for row in v5_rows.values()) != Counter(
        {"chair": 18, "whiteboard": 6, "bed": 1}
    ):
        raise ValueError("sealed v5 replay inventory changed")
    v5_ids = list(original_policy["v5_probe_ids"])
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
    v5_noise = deterministic_noise(v5_batch["x"].shape, MODEL_SEED + 2000, args.device)

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()

    prompt_texts = [PROMPTS[name] for name in PROMPT_IDS]
    base_normalized = np.asarray(v97_arrays["frozen_base_normalized"], np.float32)
    if base_normalized.shape != (3, 2, 8192, 6):
        raise ValueError("v9.7 Base K=3 shape changed")
    audit_cache: Dict[int, list] = {}
    for generation in range(3):
        seed_pair = stable_rollout_seeds(generation)
        audit_cache[generation] = []
        for prompt_index, text in enumerate(prompt_texts):
            final, states, rng_states = _capture_trajectory(
                model,
                diffusion,
                bundle,
                text,
                seed_pair[0],
                seed_pair[1],
                args.device,
                not args.no_progress,
            )
            if not np.array_equal(final.numpy(), base_normalized[generation, prompt_index]):
                raise AssertionError("v9.7 Base trajectory did not reproduce")
            audit_cache[generation].append((states, rng_states))
            print(
                "[BASE-AUDIT] generation={} prompt={}".format(
                    generation, PROMPT_IDS[prompt_index]
                ),
                flush=True,
            )

    update2_seeds = calibration_design_seeds(2)
    if failed.get("design_seed_table", {}).get("2") != list(update2_seeds):
        raise ValueError("v9.8.2 update-2 design seed changed")
    update2_states = []
    for prompt_index, text in enumerate(prompt_texts):
        final, states, rng_states = _capture_trajectory(
            model,
            diffusion,
            bundle,
            text,
            update2_seeds[0],
            update2_seeds[1],
            args.device,
            not args.no_progress,
        )
        resumed = _resume_trajectory(
            model,
            diffusion,
            bundle,
            text,
            states[SELECTED_TIMESTEP],
            rng_states[SELECTED_TIMESTEP],
            SELECTED_TIMESTEP,
            args.device,
        )
        if not torch.equal(final, resumed):
            raise AssertionError("update-2 design Base resume is not exact")
        update2_states.append(states[SELECTED_TIMESTEP])
        print("[BASE-DESIGN] update=2 prompt={}".format(PROMPT_IDS[prompt_index]), flush=True)
    update2_states_tensor = torch.cat(update2_states, dim=0).to(args.device)

    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(
        base_v5_prediction.detach().cpu().numpy(), response_arrays["base_v5_prediction"]
    ):
        raise AssertionError("fresh v5r4 fixed probe did not reproduce")

    modules = install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    parameters = list(named_lora.values())
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    zero_state_sha256 = _state_sha256(named_lora)

    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    step1_states = torch.from_numpy(
        np.asarray(v98_arrays["design_states"][timestep_index], np.float32)
    ).to(args.device)
    step1_tasks, step1_prediction, step1_frozen = _sit_tasks(
        model,
        diffusion,
        step1_states,
        kwargs,
        batch,
        mean_tensor,
        std_tensor,
    )
    step1_gradients = flattened_task_gradients(step1_tasks, parameters)
    step1_gram = (step1_gradients @ step1_gradients.T).detach().cpu().double().numpy()
    step1_weights = frank_wolfe_min_norm_weights(step1_gram, 4096)
    step1_direction = torch.from_numpy(step1_weights).to(
        step1_gradients.device, step1_gradients.dtype
    ) @ step1_gradients
    step1_direction = step1_direction / torch.linalg.vector_norm(step1_direction)
    if tensor_sha256(step1_direction) != response["direction"]["direction_sha256"]:
        raise AssertionError("v9.8 selected update-1 direction did not reproduce")
    apply_flat_direction(parameters, step1_direction, 0.003)
    step1_state = _lora_cpu_state(named_lora)
    step1_state_sha256 = _state_sha256(named_lora)

    step1_normalized = np.empty((3, 2, 8192, 6), np.float32)
    for generation in range(3):
        for prompt_index, text in enumerate(prompt_texts):
            states, rng_states = audit_cache[generation][prompt_index]
            final = _resume_trajectory(
                model,
                diffusion,
                bundle,
                text,
                states[SELECTED_TIMESTEP],
                rng_states[SELECTED_TIMESTEP],
                SELECTED_TIMESTEP,
                args.device,
            )
            step1_normalized[generation, prompt_index] = final.numpy()
    with torch.no_grad():
        step1_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(step1_normalized, response_arrays["candidate_normalized"]):
        raise AssertionError("reconstructed update-1 K=3 maps differ from v9.8.1")
    if not np.array_equal(
        step1_v5_prediction.detach().cpu().numpy(),
        response_arrays["candidate_v5_prediction"],
    ):
        raise AssertionError("reconstructed update-1 v5 response differs from v9.8.1")
    print("[UPDATE1_REPRODUCTION_PASS] exact v9.8.1 state and K=3 response", flush=True)
    del (
        step1_tasks,
        step1_prediction,
        step1_frozen,
        step1_gradients,
        step1_direction,
    )
    gc.collect()
    torch.cuda.empty_cache()

    set_lora_enabled(model, True)
    set_frozen_base_eval_lora_train(model)
    sit_tasks, design_prediction, design_frozen = _sit_tasks(
        model,
        diffusion,
        update2_states_tensor,
        kwargs,
        batch,
        mean_tensor,
        std_tensor,
    )
    live_v5_prediction = predict_xstart(
        model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
    )
    v5_tasks = (live_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
    design_physical = physical_prediction(design_prediction, mean_tensor, std_tensor)
    negative_mask = batch["explicit_negative_mask"].bool()
    negative_tasks = torch.stack(
        [design_physical[prompt][negative_mask[prompt]].mean() for prompt in range(2)]
    )
    all_tasks = torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)
    if all_tasks.numel() != len(TASK_ORDER):
        raise AssertionError("preservation task inventory changed")
    gradients = flattened_task_gradients(all_tasks, parameters)
    gram = (gradients @ gradients.T).detach().cpu().double().numpy()
    weights = frank_wolfe_min_norm_weights(gram, 8192)
    direction = torch.from_numpy(weights).to(
        gradients.device, gradients.dtype
    ) @ gradients
    direction_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction_norm) or float(direction_norm.item()) <= 0.0:
        raise RuntimeError("preservation direction is zero/non-finite")
    direction = direction / direction_norm
    derivatives = (gradients @ direction).detach().cpu().tolist()
    if min(float(value) for value in derivatives) < float(
        POLICY["gates"]["minimum_directional_derivative"]
    ):
        raise RuntimeError("eleven-task direction is not common descent")
    direction_row = {
        "task_order": list(TASK_ORDER),
        "task_losses": [float(value) for value in all_tasks.detach().cpu().tolist()],
        "gram": gram.tolist(),
        "weights": weights.tolist(),
        "directional_derivatives": [float(value) for value in derivatives],
        "direction_norm_before_unit": float(direction_norm.item()),
        "direction_sha256": tensor_sha256(direction),
        "step1_state_sha256": step1_state_sha256,
    }
    del (
        sit_tasks,
        design_prediction,
        design_frozen,
        live_v5_prediction,
        v5_tasks,
        design_physical,
        negative_tasks,
        all_tasks,
        gradients,
    )
    gc.collect()
    torch.cuda.empty_cache()

    mean_k3 = mean.reshape(1, 1, 1, 6)
    std_k3 = std.reshape(1, 1, 1, 6)
    base_physical = np.clip(base_normalized * std_k3 + mean_k3, 0.0, 1.0).astype(
        np.float32
    )
    step1_physical = np.clip(step1_normalized * std_k3 + mean_k3, 0.0, 1.0).astype(
        np.float32
    )
    base_rows = _rows(bundle, base_physical)
    step1_rows = _rows(bundle, step1_physical)
    base_pooled = pooled_object_metrics(base_rows)
    step1_pooled = pooled_object_metrics(step1_rows)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(base_physical, verified_mask)
    step1_invariance = _invariance(step1_physical, verified_mask)
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    step1_v5_dense = (
        (step1_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )

    candidate_normalized = np.empty((len(STEP_RADII), 3, 2, 8192, 6), np.float32)
    candidate_physical = np.empty_like(candidate_normalized)
    candidate_v5_predictions = np.empty((len(STEP_RADII), 3, 8192, 6), np.float32)
    candidate_rows = []
    for radius_index, radius in enumerate(STEP_RADII):
        _restore_lora_state(named_lora, step1_state)
        set_lora_enabled(model, True)
        apply_flat_direction(parameters, direction, radius)
        model.eval()
        with torch.no_grad():
            current_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        current_normalized = np.empty((3, 2, 8192, 6), np.float32)
        for generation in range(3):
            for prompt_index, text in enumerate(prompt_texts):
                states, rng_states = audit_cache[generation][prompt_index]
                final = _resume_trajectory(
                    model,
                    diffusion,
                    bundle,
                    text,
                    states[SELECTED_TIMESTEP],
                    rng_states[SELECTED_TIMESTEP],
                    SELECTED_TIMESTEP,
                    args.device,
                )
                current_normalized[generation, prompt_index] = final.numpy()
        current_physical = np.clip(
            current_normalized * std_k3 + mean_k3, 0.0, 1.0
        ).astype(np.float32)
        candidate_normalized[radius_index] = current_normalized
        candidate_physical[radius_index] = current_physical
        candidate_v5_predictions[radius_index] = current_v5.detach().cpu().numpy()
        current_rows = _rows(bundle, current_physical)
        current_pooled = pooled_object_metrics(current_rows)
        current_invariance = _invariance(current_physical, verified_mask)
        current_v5_dense = (
            (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        maximum_base_delta = float(np.abs(current_physical - base_physical).max())
        maximum_incremental_delta = float(np.abs(current_physical - step1_physical).max())
        response_policy_checks = response6_checks(
            base_rows=base_rows,
            candidate_rows=current_rows,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=derivatives[:6],
            maximum_map_delta=maximum_base_delta,
        )
        checks = preservation_checks(
            step1_rows=step1_rows,
            candidate_rows=current_rows,
            step1_v5_dense=step1_v5_dense,
            candidate_v5_dense=current_v5_dense,
            response_checks=response_policy_checks,
            directional_derivatives=derivatives,
            maximum_incremental_map_delta=maximum_incremental_delta,
        )
        eligible = all(checks.values())
        presence = [
            [continuous_presence_checks(current_rows[g][p]) for p in range(2)]
            for g in range(3)
        ]
        name = "preserve11_radius_" + str(radius).replace("0.", "0p")
        row = {
            "name": name,
            "radius": float(radius),
            "base_rows": base_rows,
            "step1_rows": step1_rows,
            "candidate_rows": current_rows,
            "base_pooled": base_pooled,
            "step1_pooled": step1_pooled,
            "candidate_pooled": current_pooled,
            "base_prompt_invariance": base_invariance,
            "step1_prompt_invariance": step1_invariance,
            "candidate_prompt_invariance": current_invariance,
            "base_v5_dense": [float(value) for value in base_v5_dense],
            "step1_v5_dense": [float(value) for value in step1_v5_dense],
            "candidate_v5_dense": [float(value) for value in current_v5_dense],
            "maximum_base_map_delta": maximum_base_delta,
            "maximum_incremental_map_delta": maximum_incremental_delta,
            "presence": presence,
            "response_checks": response_policy_checks,
            "checks": checks,
            "eligible": eligible,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        _finite_tree(row, "preservation candidate")
        candidate_rows.append(row)
        v5_change_percent = 100.0 * (
            sum(current_v5_dense) / sum(base_v5_dense) - 1.0
        )
        maximum_negative_addition = max(
            max(current_rows[g][p]["explicit_negative_mean"] for p in range(2))
            - max(step1_rows[g][p]["explicit_negative_mean"] for p in range(2))
            for g in range(3)
        )
        print(
            f"[PRESERVE R={radius:.5f}] eligible={eligible} "
            f"high={step1_pooled['chair_06']['soft_recall']:.6f}->"
            f"{current_pooled['chair_06']['soft_recall']:.6f} "
            f"v5={v5_change_percent:+.3f}% "
            f"neg-add={maximum_negative_addition:+.6f}",
            flush=True,
        )
        del current_v5
        torch.cuda.empty_cache()

    eligible_order = rank_candidates(candidate_rows)
    selected_candidate = eligible_order[0] if eligible_order else None
    arrays_file = output_dir / "preservation_direction_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(bundle["xyz"], np.float32),
        points=np.asarray(bundle["points"], np.float32),
        instance_ids=np.asarray(bundle["instance_ids"], np.int64),
        category_ids=np.asarray(bundle["category_ids"], np.int64),
        verified_object_mask=np.asarray(bundle["verified_object_mask"], bool),
        verified_positive_mask=verified_mask,
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        all_sittable_gt=np.asarray(bundle["all_target"], np.float32),
        step_radii=np.asarray(STEP_RADII, np.float64),
        prompt_ids=np.asarray(PROMPT_IDS),
        audit_seed_table=np.asarray([stable_rollout_seeds(g) for g in range(3)], np.int64),
        update2_design_seeds=np.asarray(update2_seeds, np.int64),
        base_normalized=base_normalized,
        step1_normalized=step1_normalized,
        candidates_normalized=candidate_normalized,
        base=base_physical,
        step1=step1_physical,
        candidates=candidate_physical,
        v5_target=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_prediction=base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        step1_v5_prediction=step1_v5_prediction.detach().cpu().numpy().astype(np.float32),
        candidate_v5_predictions=candidate_v5_predictions,
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v983_preservation_direction.py",
        "contract": PREPARE_ROOT / "relational_teacher_v983_preservation_direction_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v983_preservation_direction.py",
        "failed_calibration_summary": failed_file,
        "failed_calibration_maps": failed_arrays_file,
        "response6_summary": response_file,
        "response6_maps": response_maps_file,
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "v97_summary": v97_file,
        "v97_maps": v97_maps_file,
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "preservation_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": Path(str(failed["paths"]["source_dataset_index"])).resolve(),
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "preservation_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    top_checks = {
        "sealed_v982_failure_and_nested_diagnosis_bound": True,
        "fresh_v5r4_zero_init": True,
        "v981_update1_state_and_maps_exactly_reproduced": True,
        "eleven_task_direction_is_common_descent": min(derivatives)
        >= float(POLICY["gates"]["minimum_directional_derivative"]),
        "five_actual_k3_radius_candidates_evaluated": len(candidate_rows)
        == len(STEP_RADII),
        "policy_locked_before_model_and_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_preservation_candidate_is_admissible": selected_candidate is not None,
    }
    status = "PASS" if all(top_checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_seed": MODEL_SEED,
        "preflight_tag": PREFLIGHT_TAG,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": HELDOUT_TRAIN_SCENE,
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "prompt_ids": list(PROMPT_IDS),
        "prompt_text": {name: PROMPTS[name] for name in PROMPT_IDS},
        "forward_input_keys": sorted(kwargs),
        "selected_timestep": SELECTED_TIMESTEP,
        "step_radii": list(STEP_RADII),
        "task_order": list(TASK_ORDER),
        "update2_design_seeds": list(update2_seeds),
        "audit_seed_table": [list(stable_rollout_seeds(g)) for g in range(3)],
        "trajectory_accounting": {
            "audit_base_full_draws": 6,
            "update2_design_base_full_draws": 2,
            "update2_design_exact_partial_repeats": 2,
            "step1_reproduction_partial_resumes": 6,
            "candidate_partial_resumes": 30,
        },
        "preflight_binding_id": preflight["binding_id"],
        "v97_binding_id": v97["binding_id"],
        "v98_binding_id": v98["binding_id"],
        "v981_binding_id": response["binding_id"],
        "v982_binding_id": failed["binding_id"],
        "scene_binding": failed["scene_binding"],
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["preservation_policy"],
        "lora": dict(lora_metadata(model)),
        "zero_state_sha256": zero_state_sha256,
        "step1_state_sha256": step1_state_sha256,
        "serialized_model_state": False,
        "direction": direction_row,
        "base_rows": base_rows,
        "step1_rows": step1_rows,
        "base_pooled": base_pooled,
        "step1_pooled": step1_pooled,
        "base_prompt_invariance": base_invariance,
        "step1_prompt_invariance": step1_invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "step1_v5_dense": [float(value) for value in step1_v5_dense],
        "candidates": candidate_rows,
        "eligible_selection_order": eligible_order,
        "selected_candidate": selected_candidate,
        "checks": top_checks,
        "failed_checks": sorted(name for name, passed in top_checks.items() if not passed),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "preservation_maps_sha256": path_hashes["preservation_maps"],
        "authorizes_preservation_aware_calibration6": status == "PASS",
        "authorizes_checkpoint": False,
        "authorizes_room_0102": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "preflight_binding_id": report["preflight_binding_id"],
            "v97_binding_id": report["v97_binding_id"],
            "v98_binding_id": report["v98_binding_id"],
            "v981_binding_id": report["v981_binding_id"],
            "v982_binding_id": report["v982_binding_id"],
            "policy_id": report["policy_id"],
            "selected_candidate": report["selected_candidate"],
            "preservation_maps_sha256": report["preservation_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.3 report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.3 unexpectedly saved model state")
    print("[PRESERVATION_DIRECTION_PREFLIGHT_{}] Teacher-v9.8.3".format(status))
    print("[PASS] exact v9.8.1 update-1 state and K=3 response reproduced")
    print("[PASS] eleven-task direction and five actual K=3 radii evaluated")
    print("[OK] selected candidate:", selected_candidate)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)
    del model, diffusion
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

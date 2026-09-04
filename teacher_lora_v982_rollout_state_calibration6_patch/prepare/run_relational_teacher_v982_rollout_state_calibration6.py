#!/usr/bin/env python3
"""Fresh six-update rollout-state calibration for Teacher-v9.8.2."""

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
    design_seeds as v98_design_seeds,
)
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    POLICY_ID as V981_POLICY_ID,
    response6_checks,
)
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    CALIBRATION_TAG,
    DEVELOPMENT_SCENE,
    HELDOUT_TRAIN_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    MONITOR_STEPS,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SELECTED_NAME,
    SELECTED_TIMESTEP,
    STEP_RADIUS,
    TRAIN_SCENE,
    UPDATE_COUNT,
    V981_SCHEMA,
    calibration_design_seeds,
    calibration_step_checks,
    canonical_sha256,
    pooled_object_metrics,
    rank_eligible_steps,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _metrics,
    _prompt_invariance,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response6-summary", type=Path, required=True)
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
    parser.add_argument("--updates", type=int, default=UPDATE_COUNT)
    parser.add_argument("--step-radius", type=float, default=STEP_RADIUS)
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


def _validate_v981(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V981_SCHEMA
        or value.get("status") != "PASS"
        or value.get("selected_candidate") != SELECTED_NAME
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or float(value.get("selected_radius")) != STEP_RADIUS
        or value.get("policy_id") != V981_POLICY_ID
        or value.get("failed_checks")
        or value.get("authorizes_rollout_state_calibration6") is not True
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.1 PASS authority changed")
    _validate_bound_files(value, "Teacher-v9.8.1")
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.1 contains forbidden model state")
    return value


def _same_bound_arg(argument: Path, authority: Mapping[str, object], name: str) -> Path:
    path = argument.expanduser().resolve()
    expected = Path(str(authority["paths"][name])).expanduser().resolve()
    if (
        path != expected
        or not path.is_file()
        or sha256_file(path) != authority["path_sha256"][name]
    ):
        raise ValueError(name + " differs from sealed Teacher-v9.8.1 authority")
    return path


def _state_sha256(named: Mapping[str, torch.nn.Parameter]) -> str:
    return canonical_sha256(
        {
            name: tensor_sha256(parameter.detach())
            for name, parameter in sorted(named.items())
        }
    )


def _rows(bundle: Mapping[str, object], values: np.ndarray) -> list:
    if values.shape != (3, 2, 8192, 6):
        raise ValueError("K=3 calibration map shape changed")
    return [
        [_metrics(bundle, values[generation, prompt]) for prompt in range(2)]
        for generation in range(3)
    ]


def _invariance(values: np.ndarray, mask: np.ndarray) -> list[float]:
    return [
        _prompt_invariance(values[generation], mask)
        for generation in range(3)
    ]


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8.2 calibration requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != MODEL_SEED
        or args.updates != UPDATE_COUNT
        or float(args.step_radius) != STEP_RADIUS
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.2 calibration protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.2 output")
    configure_reproducibility(args.seed)

    response_file = args.response6_summary.expanduser().resolve()
    response = _validate_v981(response_file)
    response_maps_file = Path(str(response["paths"]["response6_maps"])).resolve()
    response_arrays = _load_npz(response_maps_file)
    v98_file = Path(str(response["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_maps_file = Path(str(response["paths"]["v98_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)
    v97_file = Path(str(response["paths"]["v97_summary"])).resolve()
    v97 = read_json(v97_file)
    v97_maps_file = Path(str(response["paths"]["v97_maps"])).resolve()
    v97_arrays = _load_npz(v97_maps_file)
    if response.get("v98_binding_id") != v98.get("binding_id"):
        raise ValueError("v9.8 binding changed")
    if response.get("v97_binding_id") != v97.get("binding_id"):
        raise ValueError("v9.7 binding changed")

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_root.is_dir():
        raise FileNotFoundError("a sealed dataset root is absent")
    index_file = _same_bound_arg(args.index, response, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, response, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, response, "stats_file")
    original_checkpoint = _same_bound_arg(
        args.original_checkpoint, response, "original_checkpoint"
    )
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, response, "v5_checkpoint")
    evidence_file = _same_bound_arg(
        args.v5_evidence_report, response, "v5_evidence_report"
    )
    if dataset_root != index_file.parent:
        raise ValueError("dataset root differs from sealed dataset index")
    if source_root != Path(str(response["paths"]["source_dataset_index"])).resolve().parent:
        raise ValueError("source dataset root differs from sealed source index")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")
    preflight_file = Path(str(response["paths"]["preflight_report"])).resolve()
    preflight, original_policy = _validate_preflight(preflight_file)
    if preflight.get("binding_id") != response.get("preflight_binding_id"):
        raise ValueError("Teacher-v9 preflight binding changed")
    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE:
        raise AssertionError("wrong train scene loaded")
    if response.get("scene_binding") != {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]:
        raise ValueError("room_0101 scene binding changed")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "calibration6_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("calibration policy hash changed")

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
        initial_seed, reverse_seed = stable_rollout_seeds(generation)
        audit_cache[generation] = []
        for prompt_index, text in enumerate(prompt_texts):
            final, states, rng_states = _capture_trajectory(
                model,
                diffusion,
                bundle,
                text,
                initial_seed,
                reverse_seed,
                args.device,
                not args.no_progress,
            )
            if not np.array_equal(final.numpy(), base_normalized[generation, prompt_index]):
                raise AssertionError("v9.7 Base K=3 trajectory did not reproduce")
            audit_cache[generation].append((states, rng_states))
            print(
                "[BASE-AUDIT] generation={} prompt={}".format(
                    generation, PROMPT_IDS[prompt_index]
                ),
                flush=True,
            )

    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    step_design_states: Dict[int, torch.Tensor] = {
        1: torch.from_numpy(
            np.asarray(v98_arrays["design_states"][timestep_index], np.float32)
        )
    }
    design_seed_table = {1: list(v98_design_seeds())}
    forbidden_seed_pairs = {tuple(stable_rollout_seeds(g)) for g in range(3)}
    forbidden_seed_pairs.add(tuple(v98_design_seeds()))
    for step in range(2, UPDATE_COUNT + 1):
        seed_pair = calibration_design_seeds(step)
        if seed_pair in forbidden_seed_pairs:
            raise AssertionError("calibration design seed overlaps a sealed audit/design seed")
        forbidden_seed_pairs.add(seed_pair)
        design_seed_table[step] = list(seed_pair)
        states_for_prompts = []
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
                raise AssertionError("new design t50 Base resume is not bitwise exact")
            states_for_prompts.append(states[SELECTED_TIMESTEP])
            print(
                "[BASE-DESIGN] update={} prompt={}".format(
                    step, PROMPT_IDS[prompt_index]
                ),
                flush=True,
            )
        step_design_states[step] = torch.cat(states_for_prompts, dim=0)

    v5_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, MODEL_SEED + 2000, args.device)
    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(
        base_v5_prediction.detach().cpu().numpy(),
        response_arrays["base_v5_prediction"],
    ):
        raise AssertionError("fresh v5r4 fixed-probe prediction did not reproduce")

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

    std_k3 = std.reshape(1, 1, 1, 6)
    mean_k3 = mean.reshape(1, 1, 1, 6)
    base_physical = np.clip(base_normalized * std_k3 + mean_k3, 0.0, 1.0).astype(
        np.float32
    )
    base_rows = _rows(bundle, base_physical)
    base_pooled = pooled_object_metrics(base_rows)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(base_physical, verified_mask)
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )

    candidate_normalized = np.empty((UPDATE_COUNT, 3, 2, 8192, 6), np.float32)
    candidate_physical = np.empty_like(candidate_normalized)
    candidate_v5_predictions = np.empty((UPDATE_COUNT, 3, 8192, 6), np.float32)
    direction_rows = []
    monitor_rows = []

    for step in range(1, UPDATE_COUNT + 1):
        set_lora_enabled(model, True)
        set_frozen_base_eval_lora_train(model)
        design_states = step_design_states[step].to(args.device)
        with torch.no_grad():
            set_lora_enabled(model, False)
            frozen_design = _direct_prediction(
                model, diffusion, design_states, SELECTED_TIMESTEP, kwargs
            )
        set_lora_enabled(model, True)
        prediction = _direct_prediction(
            model, diffusion, design_states, SELECTED_TIMESTEP, kwargs
        )
        objective = corrected_objective(
            prediction, frozen_design, batch, mean_tensor, std_tensor
        )
        task_losses = (
            objective["per_instance_primary"]
            + 2.0 * objective["per_instance_active_support"]
            + 0.25 * objective["per_instance_ranking"]
        ).reshape(-1)
        gradients = flattened_task_gradients(task_losses, parameters)
        gram = (gradients @ gradients.T).detach().cpu().double().numpy()
        weights = frank_wolfe_min_norm_weights(gram, 4096)
        direction = torch.from_numpy(weights).to(
            device=gradients.device, dtype=gradients.dtype
        ) @ gradients
        direction_norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(direction_norm) or float(direction_norm.item()) <= 0.0:
            raise RuntimeError("calibration common direction is zero/non-finite")
        direction = direction / direction_norm
        derivatives = (gradients @ direction).detach().cpu().tolist()
        if min(float(value) for value in derivatives) < float(
            POLICY["eligibility"]["minimum_directional_derivative"]
        ):
            raise RuntimeError("calibration update is not common descent")
        pre_state_sha256 = _state_sha256(named_lora)
        direction_sha256 = tensor_sha256(direction)
        apply_flat_direction(parameters, direction, STEP_RADIUS)
        post_state_sha256 = _state_sha256(named_lora)
        if pre_state_sha256 == post_state_sha256:
            raise AssertionError("LoRA state did not change")
        if step == 1 and direction_sha256 != response["direction"]["direction_sha256"]:
            raise AssertionError("first calibration direction does not reproduce v9.8")

        model.eval()
        with torch.no_grad():
            current_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        candidate_v5_predictions[step - 1] = current_v5.detach().cpu().numpy()
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
                del final
        current_physical = np.clip(
            current_normalized * std_k3 + mean_k3, 0.0, 1.0
        ).astype(np.float32)
        candidate_normalized[step - 1] = current_normalized
        candidate_physical[step - 1] = current_physical
        current_rows = _rows(bundle, current_physical)
        current_pooled = pooled_object_metrics(current_rows)
        current_invariance = _invariance(current_physical, verified_mask)
        current_v5_dense = (
            (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        maximum_delta = float(np.abs(current_physical - base_physical).max())
        response_policy_checks = response6_checks(
            base_rows=base_rows,
            candidate_rows=current_rows,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=derivatives,
            maximum_map_delta=maximum_delta,
        )
        step_checks = calibration_step_checks(
            step=step,
            base_rows=base_rows,
            candidate_rows=current_rows,
            response_checks=response_policy_checks,
            directional_derivatives=derivatives,
            pre_state_sha256=pre_state_sha256,
            post_state_sha256=post_state_sha256,
        )
        presence = [
            [continuous_presence_checks(current_rows[g][p]) for p in range(2)]
            for g in range(3)
        ]
        direction_row = {
            "step": step,
            "design_seeds": design_seed_table[step],
            "task_order": list(response["direction"]["task_order"]),
            "task_losses": [float(value) for value in task_losses.detach().cpu().tolist()],
            "gram": gram.tolist(),
            "weights": weights.tolist(),
            "directional_derivatives": [float(value) for value in derivatives],
            "direction_norm_before_unit": float(direction_norm.item()),
            "direction_sha256": direction_sha256,
            "pre_state_sha256": pre_state_sha256,
            "post_state_sha256": post_state_sha256,
        }
        row = {
            "step": step,
            "base_rows": base_rows,
            "candidate_rows": current_rows,
            "base_pooled": base_pooled,
            "candidate_pooled": current_pooled,
            "base_prompt_invariance": base_invariance,
            "candidate_prompt_invariance": current_invariance,
            "base_v5_dense": [float(value) for value in base_v5_dense],
            "candidate_v5_dense": [float(value) for value in current_v5_dense],
            "maximum_final_map_delta": maximum_delta,
            "presence": presence,
            "response_checks": response_policy_checks,
            "checks": step_checks,
            "eligible": all(step_checks.values()),
            "failed_checks": sorted(
                name for name, passed in step_checks.items() if not passed
            ),
        }
        _finite_tree(direction_row, "calibration direction")
        _finite_tree(row, "calibration monitor")
        direction_rows.append(direction_row)
        monitor_rows.append(row)
        print(
            "[CALIBRATION] step={}/{} eligible={} recall(bed/chair/high)={:.6f}/{:.6f}/{:.6f} v5={:+.3f}%".format(
                step,
                UPDATE_COUNT,
                row["eligible"],
                current_pooled["bed_01"]["soft_recall"],
                current_pooled["chair_01"]["soft_recall"],
                current_pooled["chair_06"]["soft_recall"],
                100.0 * (sum(current_v5_dense) / sum(base_v5_dense) - 1.0),
            ),
            flush=True,
        )

        if step == 1:
            if not np.array_equal(current_normalized, response_arrays["candidate_normalized"]):
                raise AssertionError("update-1 K=3 response does not reproduce v9.8.1")
            if not np.array_equal(
                current_v5.detach().cpu().numpy(),
                response_arrays["candidate_v5_prediction"],
            ):
                raise AssertionError("update-1 v5 response does not reproduce v9.8.1")
            print("[UPDATE1_REPRODUCTION_PASS] exact v9.8.1 K=3 response", flush=True)

        del (
            design_states,
            frozen_design,
            prediction,
            objective,
            task_losses,
            gradients,
            direction,
            current_v5,
        )
        gc.collect()
        torch.cuda.empty_cache()

    shortlisted_steps = rank_eligible_steps(monitor_rows)
    arrays_file = output_dir / "calibration6_maps.npz"
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
        monitor_steps=np.asarray(MONITOR_STEPS, np.int64),
        prompt_ids=np.asarray(PROMPT_IDS),
        seed_table=np.asarray([stable_rollout_seeds(g) for g in range(3)], np.int64),
        base_normalized=base_normalized,
        candidates_normalized=candidate_normalized,
        base=base_physical,
        candidates=candidate_physical,
        v5_target=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_prediction=base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        candidate_v5_predictions=candidate_v5_predictions,
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v982_rollout_state_calibration6.py",
        "contract": PREPARE_ROOT / "relational_teacher_v982_rollout_state_calibration6_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v982_rollout_state_calibration6.py",
        "response6_summary": response_file,
        "response6_maps": response_maps_file,
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "v97_summary": v97_file,
        "v97_maps": v97_maps_file,
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "calibration6_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": Path(str(response["paths"]["source_dataset_index"])).resolve(),
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "calibration6_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    first_step_exact = (
        np.array_equal(candidate_normalized[0], response_arrays["candidate_normalized"])
        and np.array_equal(
            candidate_v5_predictions[0], response_arrays["candidate_v5_prediction"]
        )
    )
    top_checks = {
        "sealed_v981_pass_authority_bound": True,
        "fresh_v5r4_zero_init": True,
        "update1_exactly_reproduces_v981": first_step_exact,
        "six_sequential_common_descent_updates_completed": len(direction_rows)
        == UPDATE_COUNT
        and all(
            min(row["directional_derivatives"])
            >= float(POLICY["eligibility"]["minimum_directional_derivative"])
            for row in direction_rows
        ),
        "new_design_seed_pairs_are_disjoint": len(
            {tuple(value) for value in design_seed_table.values()}
        )
        == UPDATE_COUNT
        and not (
            {tuple(value) for value in design_seed_table.values()}
            & {tuple(stable_rollout_seeds(g)) for g in range(3)}
        ),
        "all_k3_base_trajectories_reproduce_v97": True,
        "each_update_uses_actual_paired_t50_resume": True,
        "calibration_policy_locked_before_model_and_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_state_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_post_first_update_is_admissible": bool(shortlisted_steps),
    }
    status = "PASS" if all(top_checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_seed": MODEL_SEED,
        "calibration_tag": CALIBRATION_TAG,
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
        "update_count": UPDATE_COUNT,
        "monitor_steps": list(MONITOR_STEPS),
        "selected_timestep": SELECTED_TIMESTEP,
        "step_radius": STEP_RADIUS,
        "design_seed_table": {str(key): value for key, value in design_seed_table.items()},
        "audit_seed_table": [list(stable_rollout_seeds(g)) for g in range(3)],
        "trajectory_accounting": {
            "audit_base_full_draws": 6,
            "new_design_base_full_draws": 10,
            "new_design_exact_partial_repeats": 10,
            "candidate_monitor_partial_resumes": 36,
        },
        "preflight_binding_id": preflight["binding_id"],
        "v97_binding_id": v97["binding_id"],
        "v98_binding_id": v98["binding_id"],
        "v981_binding_id": response["binding_id"],
        "scene_binding": response["scene_binding"],
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["calibration6_policy"],
        "lora": dict(lora_metadata(model)),
        "zero_state_sha256": zero_state_sha256,
        "serialized_model_state": False,
        "base_rows": base_rows,
        "base_pooled": base_pooled,
        "base_prompt_invariance": base_invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "direction_rows": direction_rows,
        "monitor_rows": monitor_rows,
        "shortlisted_steps": shortlisted_steps,
        "checks": top_checks,
        "failed_checks": sorted(name for name, passed in top_checks.items() if not passed),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "calibration6_maps_sha256": path_hashes["calibration6_maps"],
        "authorizes_two_scene_rollout_state_calibration_preflight": status == "PASS",
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
            "policy_id": report["policy_id"],
            "shortlisted_steps": report["shortlisted_steps"],
            "calibration6_maps_sha256": report["calibration6_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.2 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.2 unexpectedly saved model state")
    print("[ROLLOUT_STATE_CALIBRATION6_{}] Teacher-v9.8.2".format(status))
    print("[PASS] fresh v5r4 and exact update-1 response reproduction")
    print("[PASS] six online common-descent updates and K=3 monitors completed")
    print("[OK] shortlisted steps:", shortlisted_steps)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    del model, diffusion
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

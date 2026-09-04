#!/usr/bin/env python3
"""Run fresh Teacher-v9.8.4 preservation-aware calibration-6."""

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
    _resume_trajectory,
)
from preflight_relational_teacher_v983_preservation_direction import (  # noqa: E402
    _finite_tree,
    _invariance,
    _load_npz,
    _rows,
    _same_bound_arg,
    _sit_tasks,
    _validate_bound_files,
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
from relational_teacher_v94_common_descent import (  # noqa: E402
    apply_flat_direction,
    flattened_task_gradients,
    frank_wolfe_min_norm_weights,
)
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    continuous_presence_checks,
    stable_rollout_seeds,
)
from relational_teacher_v98_rollout_state_contract import CAPTURE_TIMESTEPS  # noqa: E402
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    response6_checks,
)
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    calibration_design_seeds,
)
from relational_teacher_v983_preservation_direction_contract import (  # noqa: E402
    POLICY_ID as V983_POLICY_ID,
    SCHEMA as V983_REPORT_SCHEMA,
)
from relational_teacher_v984_preservation_calibration6_contract import (  # noqa: E402
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
    SELECTED_TIMESTEP,
    SELECTED_V983_CANDIDATE,
    STEP_RADIUS,
    TASK_ORDER,
    TRAIN_SCENE,
    UPDATE_COUNT,
    calibration_checks,
    canonical_sha256,
    pooled_object_metrics,
    rank_eligible_steps,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
)
from run_relational_teacher_v982_rollout_state_calibration6 import (  # noqa: E402
    _state_sha256,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preservation-preflight-report", type=Path, required=True)
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
    parser.add_argument("--updates", type=int, default=UPDATE_COUNT)
    parser.add_argument("--step-radius", type=float, default=STEP_RADIUS)
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _validate_v983(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V983_REPORT_SCHEMA
        or value.get("status") != "PASS"
        or value.get("policy_id") != V983_POLICY_ID
        or value.get("selected_candidate") != SELECTED_V983_CANDIDATE
        or value.get("eligible_selection_order", [None])[0]
        != SELECTED_V983_CANDIDATE
        or value.get("failed_checks")
        or value.get("authorizes_preservation_aware_calibration6") is not True
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_room_0102") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.3 PASS authority changed")
    selected = [
        row for row in value.get("candidates", [])
        if row.get("name") == SELECTED_V983_CANDIDATE
    ]
    if len(selected) != 1 or selected[0].get("eligible") is not True or selected[0].get(
        "failed_checks"
    ):
        raise ValueError("sealed Teacher-v9.8.3 selected candidate changed")
    _validate_bound_files(value, "Teacher-v9.8.3")
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.3 contains forbidden model state")
    return value


def _direction(
    tasks: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    iterations: int,
) -> tuple[torch.Tensor, Dict[str, object]]:
    gradients = flattened_task_gradients(tasks, parameters)
    gram = (gradients @ gradients.T).detach().cpu().double().numpy()
    weights = frank_wolfe_min_norm_weights(gram, iterations)
    direction = torch.from_numpy(weights).to(
        gradients.device, gradients.dtype
    ) @ gradients
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        raise RuntimeError("calibration common direction is zero/non-finite")
    direction = direction / norm
    derivatives = (gradients @ direction).detach().cpu().tolist()
    if min(float(value) for value in derivatives) < float(
        POLICY["minimum_directional_derivative"]
    ):
        raise RuntimeError("calibration direction is not common descent")
    row = {
        "task_losses": [float(value) for value in tasks.detach().cpu().tolist()],
        "gram": gram.tolist(),
        "weights": weights.tolist(),
        "directional_derivatives": [float(value) for value in derivatives],
        "direction_norm_before_unit": float(norm.item()),
        "direction_sha256": tensor_sha256(direction),
    }
    del gradients
    return direction, row


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8.4 preservation calibration requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.updates != UPDATE_COUNT
        or float(args.step_radius) != STEP_RADIUS
        or args.seed != MODEL_SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.4 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.4 output")
    configure_reproducibility(args.seed)

    v983_file = args.preservation_preflight_report.expanduser().resolve()
    v983 = _validate_v983(v983_file)
    v983_maps_file = Path(str(v983["paths"]["preservation_maps"])).resolve()
    v983_arrays = _load_npz(v983_maps_file)
    response_file = Path(str(v983["paths"]["response6_summary"])).resolve()
    response = read_json(response_file)
    response_maps_file = Path(str(v983["paths"]["response6_maps"])).resolve()
    response_arrays = _load_npz(response_maps_file)
    v98_file = Path(str(v983["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_maps_file = Path(str(v983["paths"]["v98_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)
    v97_file = Path(str(v983["paths"]["v97_summary"])).resolve()
    v97 = read_json(v97_file)
    v97_maps_file = Path(str(v983["paths"]["v97_maps"])).resolve()
    v97_arrays = _load_npz(v97_maps_file)
    preflight_file = Path(str(v983["paths"]["preflight_report"])).resolve()
    preflight = read_json(preflight_file)

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_root.is_dir():
        raise FileNotFoundError("a sealed dataset root is absent")
    index_file = _same_bound_arg(args.index, v983, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v983, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v983, "stats_file")
    original_checkpoint = _same_bound_arg(
        args.original_checkpoint, v983, "original_checkpoint"
    )
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v983, "v5_checkpoint")
    evidence_file = _same_bound_arg(
        args.v5_evidence_report, v983, "v5_evidence_report"
    )
    if dataset_root != index_file.parent:
        raise ValueError("dataset root differs from sealed dataset index")
    source_index = Path(str(v983["paths"]["source_dataset_index"])).resolve()
    if source_root != source_index.parent:
        raise ValueError("source dataset root differs from sealed source index")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE:
        raise ValueError("wrong calibration scene loaded")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "preservation_calibration6_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("preservation calibration policy hash changed")

    mean, std = load_stats(stats_file)
    batch = _build_batch(bundle, mean, std, args.device)
    kwargs = _kwargs(batch)
    if tuple(sorted(kwargs)) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher forward input changed")
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    metric_policy = read_json(Path(str(v983["paths"]["metric_policy"])).resolve())
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
            audit_cache[generation].append(
                (states[SELECTED_TIMESTEP], rng_states[SELECTED_TIMESTEP])
            )
            print(
                f"[BASE-AUDIT] generation={generation} prompt={PROMPT_IDS[prompt_index]}",
                flush=True,
            )

    design_cache: Dict[int, list] = {}
    design_seed_table: Dict[int, list[int]] = {}
    for step in range(2, UPDATE_COUNT + 1):
        seed_pair = calibration_design_seeds(step)
        design_seed_table[step] = list(seed_pair)
        design_cache[step] = []
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
            state = states[SELECTED_TIMESTEP]
            rng_state = rng_states[SELECTED_TIMESTEP]
            repeated = _resume_trajectory(
                model,
                diffusion,
                bundle,
                text,
                state,
                rng_state,
                SELECTED_TIMESTEP,
                args.device,
            )
            if not torch.equal(final, repeated):
                raise AssertionError("calibration Base design resume is not exact")
            design_cache[step].append((state, rng_state))
            print(
                f"[BASE-DESIGN] update={step} prompt={PROMPT_IDS[prompt_index]}",
                flush=True,
            )

    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(
        base_v5_prediction.detach().cpu().numpy(), v983_arrays["base_v5_prediction"]
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
    if zero_state_sha256 != v983["zero_state_sha256"]:
        raise AssertionError("fresh zero-output LoRA state differs from v9.8.3")

    mean_k3 = mean.reshape(1, 1, 1, 6)
    std_k3 = std.reshape(1, 1, 1, 6)
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

    candidates_normalized = np.empty((UPDATE_COUNT, 3, 2, 8192, 6), np.float32)
    candidates_physical = np.empty_like(candidates_normalized)
    candidate_v5_predictions = np.empty((UPDATE_COUNT, 3, 8192, 6), np.float32)
    direction_rows = []
    monitor_rows = []
    previous_rows = base_rows
    previous_v5_dense = base_v5_dense
    step1_exact = False
    step2_exact = False
    selected_v983_index = [
        index
        for index, row in enumerate(v983["candidates"])
        if row["name"] == SELECTED_V983_CANDIDATE
    ][0]
    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)

    for step in MONITOR_STEPS:
        set_frozen_base_eval_lora_train(model)
        pre_state_sha256 = _state_sha256(named_lora)
        if step == 1:
            task_order = list(TASK_ORDER[:6])
            design_states = torch.from_numpy(
                np.asarray(v98_arrays["design_states"][timestep_index], np.float32)
            ).to(args.device)
            tasks, design_prediction, design_frozen = _sit_tasks(
                model,
                diffusion,
                design_states,
                kwargs,
                batch,
                mean_tensor,
                std_tensor,
            )
            direction, direction_row = _direction(tasks, parameters, 4096)
            expected_direction = response["direction"]
        else:
            task_order = list(TASK_ORDER)
            design_states = torch.cat(
                [design_cache[step][prompt][0] for prompt in range(2)], dim=0
            ).to(args.device)
            sit_tasks, design_prediction, design_frozen = _sit_tasks(
                model,
                diffusion,
                design_states,
                kwargs,
                batch,
                mean_tensor,
                std_tensor,
            )
            live_v5_prediction = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
            v5_tasks = (live_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
            design_physical = physical_prediction(
                design_prediction, mean_tensor, std_tensor
            )
            negative_mask = batch["explicit_negative_mask"].bool()
            negative_tasks = torch.stack(
                [
                    design_physical[prompt][negative_mask[prompt]].mean()
                    for prompt in range(2)
                ]
            )
            tasks = torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)
            direction, direction_row = _direction(tasks, parameters, 8192)
            expected_direction = v983["direction"] if step == 2 else None
        direction_row.update(
            {
                "step": step,
                "task_order": task_order,
                "pre_state_sha256": pre_state_sha256,
                "design_seeds": (
                    list(v98["design_seeds"])
                    if step == 1
                    else design_seed_table[step]
                ),
            }
        )
        if expected_direction is not None:
            if (
                direction_row["direction_sha256"]
                != expected_direction["direction_sha256"]
                or not np.allclose(
                    direction_row["task_losses"],
                    expected_direction["task_losses"],
                    rtol=1e-7,
                    atol=1e-8,
                )
                or not np.allclose(
                    direction_row["gram"], expected_direction["gram"], rtol=1e-7, atol=1e-8
                )
                or not np.allclose(
                    direction_row["weights"], expected_direction["weights"], rtol=1e-7, atol=1e-9
                )
                or not np.allclose(
                    direction_row["directional_derivatives"],
                    expected_direction["directional_derivatives"],
                    rtol=1e-7,
                    atol=1e-8,
                )
                or not math.isclose(
                    direction_row["direction_norm_before_unit"],
                    expected_direction["direction_norm_before_unit"],
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )
            ):
                raise AssertionError(f"update-{step} direction did not exactly reproduce")

        apply_flat_direction(parameters, direction, STEP_RADIUS)
        post_state_sha256 = _state_sha256(named_lora)
        if pre_state_sha256 == post_state_sha256:
            raise AssertionError("calibration LoRA state did not change")
        if step == 1 and post_state_sha256 != v983["step1_state_sha256"]:
            raise AssertionError("update-1 LoRA state does not reproduce v9.8.1")
        direction_row["post_state_sha256"] = post_state_sha256
        direction_rows.append(direction_row)
        model.eval()

        current_normalized = np.empty((3, 2, 8192, 6), np.float32)
        for generation in range(3):
            for prompt_index, text in enumerate(prompt_texts):
                state, rng_state = audit_cache[generation][prompt_index]
                final = _resume_trajectory(
                    model,
                    diffusion,
                    bundle,
                    text,
                    state,
                    rng_state,
                    SELECTED_TIMESTEP,
                    args.device,
                )
                current_normalized[generation, prompt_index] = final.numpy()
        with torch.no_grad():
            current_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        current_physical = np.clip(
            current_normalized * std_k3 + mean_k3, 0.0, 1.0
        ).astype(np.float32)
        candidates_normalized[step - 1] = current_normalized
        candidates_physical[step - 1] = current_physical
        candidate_v5_predictions[step - 1] = current_v5.detach().cpu().numpy()
        current_rows = _rows(bundle, current_physical)
        current_pooled = pooled_object_metrics(current_rows)
        current_invariance = _invariance(current_physical, verified_mask)
        current_v5_dense = (
            (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        maximum_base_delta = float(np.abs(current_physical - base_physical).max())
        response_policy_checks = response6_checks(
            base_rows=base_rows,
            candidate_rows=current_rows,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=direction_row["directional_derivatives"][:6],
            maximum_map_delta=maximum_base_delta,
        )
        checks = calibration_checks(
            step=step,
            previous_rows=previous_rows,
            candidate_rows=current_rows,
            previous_v5_dense=previous_v5_dense,
            candidate_v5_dense=current_v5_dense,
            response_checks=response_policy_checks,
            directional_derivatives=direction_row["directional_derivatives"],
            pre_state_sha256=pre_state_sha256,
            post_state_sha256=post_state_sha256,
        )
        presence = [
            [continuous_presence_checks(current_rows[g][p]) for p in range(2)]
            for g in range(3)
        ]

        if step == 1:
            if not np.array_equal(current_normalized, v983_arrays["step1_normalized"]):
                raise AssertionError("update-1 K=3 maps do not reproduce v9.8.1")
            if not np.array_equal(
                current_v5.detach().cpu().numpy(), v983_arrays["step1_v5_prediction"]
            ):
                raise AssertionError("update-1 v5 response does not reproduce v9.8.1")
            step1_exact = True
        elif step == 2:
            if not np.array_equal(
                current_normalized,
                v983_arrays["candidates_normalized"][selected_v983_index],
            ):
                raise AssertionError("update-2 K=3 maps do not reproduce selected v9.8.3")
            if not np.array_equal(
                current_v5.detach().cpu().numpy(),
                v983_arrays["candidate_v5_predictions"][selected_v983_index],
            ):
                raise AssertionError("update-2 v5 response does not reproduce selected v9.8.3")
            step2_exact = True

        eligible = all(checks.values())
        row = {
            "step": step,
            "design_seeds": direction_row["design_seeds"],
            "previous_rows": previous_rows,
            "candidate_rows": current_rows,
            "previous_pooled": pooled_object_metrics(previous_rows),
            "candidate_pooled": current_pooled,
            "base_prompt_invariance": base_invariance,
            "candidate_prompt_invariance": current_invariance,
            "base_v5_dense": [float(value) for value in base_v5_dense],
            "previous_v5_dense": [float(value) for value in previous_v5_dense],
            "candidate_v5_dense": [float(value) for value in current_v5_dense],
            "maximum_base_map_delta": maximum_base_delta,
            "presence": presence,
            "response_checks": response_policy_checks,
            "checks": checks,
            "eligible": eligible,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        _finite_tree(row, "preservation calibration row")
        monitor_rows.append(row)
        print(
            f"[CALIBRATION] step={step}/6 eligible={eligible} "
            f"high={row['previous_pooled']['chair_06']['soft_recall']:.6f}->"
            f"{current_pooled['chair_06']['soft_recall']:.6f} "
            f"v5={100.0 * (sum(current_v5_dense) / sum(base_v5_dense) - 1.0):+.3f}%",
            flush=True,
        )

        previous_rows = current_rows
        previous_v5_dense = current_v5_dense
        del tasks, design_prediction, design_frozen, direction, current_v5
        if step > 1:
            del sit_tasks, live_v5_prediction, v5_tasks, design_physical, negative_tasks
        gc.collect()
        torch.cuda.empty_cache()

    shortlisted_steps = rank_eligible_steps(monitor_rows)
    arrays_file = output_dir / "preservation_calibration6_maps.npz"
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
        audit_seed_table=np.asarray([stable_rollout_seeds(g) for g in range(3)], np.int64),
        design_seed_table=np.asarray(
            [design_seed_table[step] for step in range(2, UPDATE_COUNT + 1)], np.int64
        ),
        base_normalized=base_normalized,
        candidates_normalized=candidates_normalized,
        base=base_physical,
        candidates=candidates_physical,
        v5_target=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_prediction=base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        candidate_v5_predictions=candidate_v5_predictions,
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v984_preservation_calibration6.py",
        "contract": PREPARE_ROOT / "relational_teacher_v984_preservation_calibration6_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v984_preservation_calibration6.py",
        "v983_report": v983_file,
        "v983_maps": v983_maps_file,
        "response6_summary": response_file,
        "response6_maps": response_maps_file,
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "v97_summary": v97_file,
        "v97_maps": v97_maps_file,
        "preflight_report": preflight_file,
        "metric_policy": Path(str(v983["paths"]["metric_policy"])).resolve(),
        "calibration_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "calibration_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    top_checks = {
        "sealed_v983_pass_and_selected_candidate_bound": True,
        "fresh_v5r4_zero_init": True,
        "update1_exactly_reproduces_v981": step1_exact,
        "update2_exactly_reproduces_selected_v983": step2_exact,
        "six_updates_and_actual_k3_monitors_completed": len(monitor_rows) == UPDATE_COUNT,
        "all_directions_are_common_descent": all(
            row["checks"]["direction_is_common_descent"] for row in monitor_rows
        ),
        "policy_locked_before_model_and_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
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
        "task_order": list(TASK_ORDER),
        "design_seed_table": {str(key): value for key, value in design_seed_table.items()},
        "audit_seed_table": [list(stable_rollout_seeds(g)) for g in range(3)],
        "trajectory_accounting": {
            "audit_base_full_draws": 6,
            "design_base_full_draws": 10,
            "design_exact_partial_repeats": 10,
            "candidate_monitor_partial_resumes": 36,
        },
        "preflight_binding_id": preflight["binding_id"],
        "v97_binding_id": v97["binding_id"],
        "v98_binding_id": v98["binding_id"],
        "v981_binding_id": response["binding_id"],
        "v983_binding_id": v983["binding_id"],
        "scene_binding": v983["scene_binding"],
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["calibration_policy"],
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
        "calibration_maps_sha256": path_hashes["calibration_maps"],
        "authorizes_two_scene_preservation_calibration_preflight": status == "PASS",
        "authorizes_checkpoint": False,
        "authorizes_room_0102": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v983_binding_id": report["v983_binding_id"],
            "policy_id": report["policy_id"],
            "shortlisted_steps": report["shortlisted_steps"],
            "calibration_maps_sha256": report["calibration_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.4 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.4 unexpectedly saved model state")
    print(f"[PRESERVATION_CALIBRATION6_{status}] Teacher-v9.8.4")
    print("[PASS] exact v9.8.1 update 1 and selected v9.8.3 update 2 reproduced")
    print("[PASS] six fresh updates and six actual K=3 monitors completed")
    print("[OK] shortlisted steps:", shortlisted_steps)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    del model, diffusion
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

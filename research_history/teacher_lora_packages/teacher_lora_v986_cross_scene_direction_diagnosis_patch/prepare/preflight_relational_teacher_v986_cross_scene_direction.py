#!/usr/bin/env python3
"""Diagnose a strict two-scene common-descent direction at exact v9.8.4 step 4."""

from __future__ import annotations

import argparse
import gc
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
)
from preflight_relational_teacher_v98_rollout_state_response import (  # noqa: E402
    _capture_trajectory,
    _resume_trajectory,
)
from preflight_relational_teacher_v983_preservation_direction import (  # noqa: E402
    _finite_tree,
    _load_npz,
    _same_bound_arg,
    _sit_tasks,
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
)
from relational_teacher_v98_rollout_state_contract import CAPTURE_TIMESTEPS  # noqa: E402
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    calibration_design_seeds,
)
from relational_teacher_v984_preservation_calibration6_contract import (  # noqa: E402
    POLICY_ID as V984_POLICY_ID,
    SCHEMA as V984_REPORT_SCHEMA,
    TASK_ORDER as V984_TASK_ORDER,
)
from relational_teacher_v985_two_scene_step4_preflight_contract import (  # noqa: E402
    EXPECTED_INSTANCES,
    POLICY_ID as V985_POLICY_ID,
)
from preflight_relational_teacher_v985_two_scene_step4 import (  # noqa: E402
    _direction_exact,
    _scene_binding,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
)
from run_relational_teacher_v982_rollout_state_calibration6 import _state_sha256  # noqa: E402
from run_relational_teacher_v984_preservation_calibration6 import _direction  # noqa: E402
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402
from relational_teacher_v986_cross_scene_direction_contract import (  # noqa: E402
    AUDIT_SCENE,
    CANDIDATE_NAMES,
    DEVELOPMENT_SCENE,
    DIAGNOSIS_TAG,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    RAW_GRAM_ASYMMETRY_CAP,
    RECONSTRUCTION_STEPS,
    SCHEMA,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    STEP_RADIUS,
    TASK_ORDER,
    V985_SCHEMA,
    canonical_sha256,
    conflict_pairs,
    cross_scene_design_seeds,
    diagnose_directions,
    rank_eligible_directions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-two-scene-report", type=Path, required=True)
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


def _validate_v985(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    expected_audit_failures = [
        "bed_pooled_mae_strictly_improves",
        "bed_pooled_recall_strictly_improves",
    ]
    if (
        value.get("schema") != V985_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("policy_id") != V985_POLICY_ID
        or value.get("failed_checks") != ["room0102_step4_response_is_admissible"]
        or value.get("audit_failed_checks") != expected_audit_failures
        or value.get("authorizes_cross_scene_direction_diagnosis") is not True
        or value.get("authorizes_two_scene_preservation_response_preflight") is not False
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.5 Bed-only cross-scene failure changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("sealed Teacher-v9.8.5 path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("sealed Teacher-v9.8.5 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("sealed Teacher-v9.8.5 contains forbidden model state")
    return value


def _capture_design_pair(model, diffusion, bundle, device: str, progress: bool):
    scene_id = str(bundle["scene_id"])
    seed_pair = cross_scene_design_seeds(scene_id)
    states = []
    for prompt_id in PROMPT_IDS:
        text = PROMPTS[prompt_id]
        final, trajectory, rng_states = _capture_trajectory(
            model,
            diffusion,
            bundle,
            text,
            seed_pair[0],
            seed_pair[1],
            device,
            progress,
        )
        repeated = _resume_trajectory(
            model,
            diffusion,
            bundle,
            text,
            trajectory[SELECTED_TIMESTEP],
            rng_states[SELECTED_TIMESTEP],
            SELECTED_TIMESTEP,
            device,
        )
        if not torch.equal(final, repeated):
            raise AssertionError(scene_id + " diagnosis Base resume is not exact")
        states.append(trajectory[SELECTED_TIMESTEP])
        print("[DIAGNOSIS-DESIGN] scene={} prompt={}".format(scene_id, prompt_id), flush=True)
        del final, repeated, trajectory, rng_states
    return torch.cat(states, dim=0), seed_pair


def _scene_task_gradients(
    *,
    model,
    diffusion,
    states: torch.Tensor,
    kwargs: Mapping[str, object],
    batch: Mapping[str, object],
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor, list[float]]:
    sit_tasks, prediction, frozen = _sit_tasks(
        model,
        diffusion,
        states,
        kwargs,
        batch,
        mean_tensor,
        std_tensor,
    )
    physical = physical_prediction(prediction, mean_tensor, std_tensor)
    negative_mask = batch["explicit_negative_mask"].bool()
    negative_tasks = torch.stack(
        [physical[prompt][negative_mask[prompt]].mean() for prompt in range(2)]
    )
    tasks = torch.cat((sit_tasks, negative_tasks), dim=0)
    gradients = flattened_task_gradients(tasks, parameters).detach().cpu()
    losses = [float(value) for value in tasks.detach().cpu().tolist()]
    del sit_tasks, prediction, frozen, physical, negative_tasks, tasks
    gc.collect()
    torch.cuda.empty_cache()
    return gradients, losses


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8.6 cross-scene diagnosis requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != MODEL_SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.6 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.6 output")
    configure_reproducibility(args.seed)

    v985_file = args.failed_two_scene_report.expanduser().resolve()
    v985 = _validate_v985(v985_file)
    v984_file = Path(str(v985["paths"]["v984_summary"])).resolve()
    v984 = read_json(v984_file)
    if (
        v984.get("schema") != V984_REPORT_SCHEMA
        or v984.get("status") != "PASS"
        or v984.get("policy_id") != V984_POLICY_ID
        or v984.get("shortlisted_steps") != [4, 3]
        or v984.get("failed_checks")
        or v984.get("binding_id") != v985.get("v984_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.4 authority changed")
    v984_maps_file = Path(str(v985["paths"]["v984_maps"])).resolve()
    v984_arrays = _load_npz(v984_maps_file)
    v98_file = Path(str(v985["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_maps_file = Path(str(v985["paths"]["v98_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _same_bound_arg(args.index, v985, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v985, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v985, "stats_file")
    original_checkpoint = _same_bound_arg(args.original_checkpoint, v985, "original_checkpoint")
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v985, "v5_checkpoint")
    evidence_file = _same_bound_arg(args.v5_evidence_report, v985, "v5_evidence_report")
    source_index = Path(str(v985["paths"]["source_dataset_index"])).resolve()
    if dataset_root != index_file.parent or source_root != source_index.parent:
        raise ValueError("dataset roots differ from sealed Teacher-v9.8.5")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "cross_scene_direction_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("cross-scene direction policy hash changed")

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {SOURCE_SCENE, AUDIT_SCENE}:
        raise ValueError("two train-scene metadata inventory changed")
    source_bundle = load_train_scene_bundle(dataset_root, source_root, records[SOURCE_SCENE])
    audit_bundle = load_train_scene_bundle(dataset_root, source_root, records[AUDIT_SCENE])
    if tuple(source_bundle["instance_names"]) != tuple(EXPECTED_INSTANCES[SOURCE_SCENE]):
        raise ValueError("source verified target order changed")
    if tuple(audit_bundle["instance_names"]) != tuple(EXPECTED_INSTANCES[AUDIT_SCENE]):
        raise ValueError("audit verified target order changed")

    mean, std = load_stats(stats_file)
    source_batch = _build_batch(source_bundle, mean, std, args.device)
    audit_batch = _build_batch(audit_bundle, mean, std, args.device)
    source_kwargs = _kwargs(source_batch)
    audit_kwargs = _kwargs(audit_batch)
    if tuple(sorted(source_kwargs)) != tuple(sorted(FORWARD_INPUT_KEYS)) or tuple(
        sorted(audit_kwargs)
    ) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher forward input changed")
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    metric_policy = read_json(Path(str(v985["paths"]["metric_policy"])).resolve())
    split = load_split(split_file)
    v5_rows = load_v5_rows(v5_root, split, "train", mean, std, 4.0, 16.0, 0.7)
    if Counter(str(row["target"]) for row in v5_rows.values()) != Counter(
        {"chair": 18, "whiteboard": 6, "bed": 1}
    ):
        raise ValueError("sealed v5 replay inventory changed")
    v5_ids = list(metric_policy["v5_probe_ids"])
    if [str(v5_rows[name]["target"]) for name in v5_ids] != ["chair", "bed", "whiteboard"]:
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
    reconstruction_cache: Dict[int, list[torch.Tensor]] = {}
    for step in range(2, SELECTED_V984_STEP + 1):
        seed_pair = calibration_design_seeds(step)
        reconstruction_cache[step] = []
        for prompt_index, text in enumerate(prompt_texts):
            final, states, rng_states = _capture_trajectory(
                model,
                diffusion,
                source_bundle,
                text,
                seed_pair[0],
                seed_pair[1],
                args.device,
                not args.no_progress,
            )
            repeated = _resume_trajectory(
                model,
                diffusion,
                source_bundle,
                text,
                states[SELECTED_TIMESTEP],
                rng_states[SELECTED_TIMESTEP],
                SELECTED_TIMESTEP,
                args.device,
            )
            if not torch.equal(final, repeated):
                raise AssertionError("reconstruction Base resume is not exact")
            reconstruction_cache[step].append(states[SELECTED_TIMESTEP])
            print("[RECONSTRUCTION-DESIGN] update={} prompt={}".format(step, PROMPT_IDS[prompt_index]), flush=True)
            del final, repeated, states, rng_states

    source_diagnosis_states, source_design_seeds = _capture_design_pair(
        model, diffusion, source_bundle, args.device, not args.no_progress
    )
    audit_diagnosis_states, audit_design_seeds = _capture_design_pair(
        model, diffusion, audit_bundle, args.device, not args.no_progress
    )

    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(
        base_v5_prediction.detach().cpu().numpy(), v984_arrays["base_v5_prediction"]
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
    if zero_state_sha256 != v984["zero_state_sha256"]:
        raise AssertionError("fresh zero-output LoRA state differs from v9.8.4")

    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    reconstruction_rows = []
    for step in RECONSTRUCTION_STEPS:
        set_frozen_base_eval_lora_train(model)
        pre_state_sha256 = _state_sha256(named_lora)
        if step == 1:
            task_order = list(V984_TASK_ORDER[:6])
            design_states = torch.from_numpy(
                np.asarray(v98_arrays["design_states"][timestep_index], np.float32)
            ).to(args.device)
            tasks, design_prediction, design_frozen = _sit_tasks(
                model,
                diffusion,
                design_states,
                source_kwargs,
                source_batch,
                mean_tensor,
                std_tensor,
            )
            direction, direction_row = _direction(tasks, parameters, 4096)
            design_seeds = list(v98["design_seeds"])
        else:
            task_order = list(V984_TASK_ORDER)
            design_states = torch.cat(reconstruction_cache[step], dim=0).to(args.device)
            sit_tasks, design_prediction, design_frozen = _sit_tasks(
                model,
                diffusion,
                design_states,
                source_kwargs,
                source_batch,
                mean_tensor,
                std_tensor,
            )
            live_v5_prediction = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
            v5_tasks = (live_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
            design_physical = physical_prediction(design_prediction, mean_tensor, std_tensor)
            negative_mask = source_batch["explicit_negative_mask"].bool()
            negative_tasks = torch.stack(
                [design_physical[prompt][negative_mask[prompt]].mean() for prompt in range(2)]
            )
            tasks = torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)
            direction, direction_row = _direction(tasks, parameters, 8192)
            design_seeds = list(calibration_design_seeds(step))
        apply_flat_direction(parameters, direction, STEP_RADIUS)
        post_state_sha256 = _state_sha256(named_lora)
        direction_row.update(
            {
                "step": step,
                "task_order": task_order,
                "pre_state_sha256": pre_state_sha256,
                "post_state_sha256": post_state_sha256,
                "design_seeds": design_seeds,
            }
        )
        _direction_exact(direction_row, v984["direction_rows"][step - 1], step)
        reconstruction_rows.append(direction_row)
        print("[RECONSTRUCTION_PASS] exact Teacher-v9.8.4 update={}".format(step), flush=True)
        del tasks, design_prediction, design_frozen, direction, design_states
        if step > 1:
            del sit_tasks, live_v5_prediction, v5_tasks, design_physical, negative_tasks
        gc.collect()
        torch.cuda.empty_cache()

    selected_state_sha256 = _state_sha256(named_lora)
    if selected_state_sha256 != v984["direction_rows"][3]["post_state_sha256"]:
        raise AssertionError("reconstructed step-4 state hash changed")

    source_gradients, source_losses = _scene_task_gradients(
        model=model,
        diffusion=diffusion,
        states=source_diagnosis_states.to(args.device),
        kwargs=source_kwargs,
        batch=source_batch,
        mean_tensor=mean_tensor,
        std_tensor=std_tensor,
        parameters=parameters,
    )
    print("[GRADIENTS] room_0101 six Sit plus two negative tasks", flush=True)
    audit_gradients, audit_losses = _scene_task_gradients(
        model=model,
        diffusion=diffusion,
        states=audit_diagnosis_states.to(args.device),
        kwargs=audit_kwargs,
        batch=audit_batch,
        mean_tensor=mean_tensor,
        std_tensor=std_tensor,
        parameters=parameters,
    )
    print("[GRADIENTS] room_0102 six Sit plus two negative tasks", flush=True)
    live_v5_prediction = predict_xstart(
        model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
    )
    v5_tasks = (live_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
    v5_gradients = flattened_task_gradients(v5_tasks, parameters).detach().cpu()
    v5_losses = [float(value) for value in v5_tasks.detach().cpu().tolist()]
    print("[GRADIENTS] v5 Chair/Bed/Whiteboard preservation tasks", flush=True)

    gradients = torch.cat((source_gradients, audit_gradients, v5_gradients), dim=0)
    if gradients.shape[0] != len(TASK_ORDER):
        raise AssertionError("19-task gradient inventory changed")
    gradient64 = gradients.double()
    raw_gram = (gradient64 @ gradient64.T).numpy()
    if not np.isfinite(raw_gram).all():
        raise RuntimeError("raw 19-task Gram matrix contains NaN/Inf")
    raw_gram_max_asymmetry = float(np.abs(raw_gram - raw_gram.T).max())
    if raw_gram_max_asymmetry > RAW_GRAM_ASYMMETRY_CAP:
        raise RuntimeError(
            "raw 19-task Gram asymmetry exceeds numerical cap: {:.12g}".format(
                raw_gram_max_asymmetry
            )
        )
    gram = 0.5 * (raw_gram + raw_gram.T)
    task_losses = source_losses + audit_losses + v5_losses
    rows = diagnose_directions(gram)
    for row in rows:
        coefficients = torch.tensor(row["effective_coefficients"], dtype=gradients.dtype)
        flat_direction = coefficients @ gradients
        direction_norm = float(torch.linalg.vector_norm(flat_direction).item())
        if not np.isfinite(direction_norm) or not np.isclose(direction_norm, 1.0, rtol=2e-5, atol=2e-6):
            raise RuntimeError("diagnostic direction is not finite/unit")
        row["direction_l2"] = direction_norm
        row["direction_sha256"] = tensor_sha256(flat_direction)
        print(
            "[DIRECTION {}] eligible={} bed-min={:.8f} all-min={:.8f}".format(
                row["name"],
                row["eligible"],
                row["minimum_audit_bed_derivative"],
                row["minimum_directional_derivative"],
            ),
            flush=True,
        )
    selection_order = rank_eligible_directions(rows)
    selected_candidate = selection_order[0] if selection_order else None
    conflicts = conflict_pairs(gram)

    geometry_file = output_dir / "cross_scene_gradient_geometry.npz"
    atomic_savez(
        geometry_file,
        task_order=np.asarray(TASK_ORDER),
        task_losses=np.asarray(task_losses, np.float64),
        gram=np.asarray(gram, np.float64),
        candidate_names=np.asarray(CANDIDATE_NAMES),
        effective_coefficients=np.asarray(
            [row["effective_coefficients"] for row in rows], np.float64
        ),
        directional_derivatives=np.asarray(
            [row["directional_derivatives"] for row in rows], np.float64
        ),
        source_design_seeds=np.asarray(source_design_seeds, np.int64),
        audit_design_seeds=np.asarray(audit_design_seeds, np.int64),
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v986_cross_scene_direction.py",
        "contract": PREPARE_ROOT / "relational_teacher_v986_cross_scene_direction_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v986_cross_scene_direction.py",
        "v985_report": v985_file,
        "v984_summary": v984_file,
        "v984_maps": v984_maps_file,
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "metric_policy": Path(str(v985["paths"]["metric_policy"])).resolve(),
        "diagnosis_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "gradient_geometry": geometry_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    top_checks = {
        "sealed_v985_bed_only_cross_scene_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exactly_19_live_normalized_task_gradients": True,
        "at_least_one_strict_cross_scene_direction_exists": selected_candidate is not None,
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "no_candidate_parameter_update_applied": True,
        "room0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
    }
    status = "PASS" if all(top_checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_seed": MODEL_SEED,
        "diagnosis_tag": DIAGNOSIS_TAG,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "source_scene": SOURCE_SCENE,
        "audit_scene": AUDIT_SCENE,
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "development_arrays_read": False,
        "paper_test_access": False,
        "selected_v984_step": SELECTED_V984_STEP,
        "reconstruction_steps": list(RECONSTRUCTION_STEPS),
        "selected_timestep": SELECTED_TIMESTEP,
        "step_radius_reconstruction_only": STEP_RADIUS,
        "prompt_ids": list(PROMPT_IDS),
        "forward_input_keys": sorted(source_kwargs),
        "task_order": list(TASK_ORDER),
        "task_losses": task_losses,
        "source_design_seeds": list(source_design_seeds),
        "audit_design_seeds": list(audit_design_seeds),
        "trajectory_accounting": {
            "room0101_reconstruction_full_draws": 6,
            "room0101_reconstruction_exact_partial_repeats": 6,
            "room0101_diagnosis_full_draws": 2,
            "room0101_diagnosis_exact_partial_repeats": 2,
            "room0102_diagnosis_full_draws": 2,
            "room0102_diagnosis_exact_partial_repeats": 2,
        },
        "v985_binding_id": v985["binding_id"],
        "v984_binding_id": v984["binding_id"],
        "source_scene_binding": _scene_binding(records[SOURCE_SCENE]),
        "audit_scene_binding": _scene_binding(records[AUDIT_SCENE]),
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["diagnosis_policy"],
        "lora": dict(lora_metadata(model)),
        "zero_state_sha256": zero_state_sha256,
        "selected_state_sha256": selected_state_sha256,
        "reconstruction_rows": reconstruction_rows,
        "gram": gram.tolist(),
        "raw_gram_max_asymmetry": raw_gram_max_asymmetry,
        "conflict_pairs": conflicts,
        "conflict_pair_count": len(conflicts),
        "direction_candidates": rows,
        "eligible_selection_order": selection_order,
        "selected_candidate": selected_candidate,
        "checks": top_checks,
        "failed_checks": sorted(name for name, passed in top_checks.items() if not passed),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "gradient_geometry_sha256": path_hashes["gradient_geometry"],
        "serialized_model_state": False,
        "authorizes_actual_two_scene_k3_direction_response_grid": status == "PASS",
        "authorizes_objective_or_capacity_redesign": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v985_binding_id": report["v985_binding_id"],
            "policy_id": report["policy_id"],
            "selected_state_sha256": report["selected_state_sha256"],
            "gradient_geometry_sha256": report["gradient_geometry_sha256"],
            "selected_candidate": report["selected_candidate"],
            "status": report["status"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.6 report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.6 unexpectedly saved model state")
    print("[CROSS_SCENE_DIRECTION_{}] Teacher-v9.8.6 diagnosis".format(status))
    print("[PASS] sealed room_0102 Bed failure and exact step-4 reconstruction verified")
    print("[PASS] 12 Sit, four negative and three v5 task gradients computed")
    print("[OK] conflicts: {} / 171".format(len(conflicts)))
    print("[OK] selected candidate:", selected_candidate)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)
    del model, diffusion, gradients, gradient64, source_gradients, audit_gradients, v5_gradients
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

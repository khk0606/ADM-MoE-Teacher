#!/usr/bin/env python3
"""Diagnose a common direction on the exact two-scene K=3 rollout states."""

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
from preflight_relational_teacher_v985_two_scene_step4 import (  # noqa: E402
    _direction_exact,
    _scene_binding,
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
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    stable_rollout_seeds,
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
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
)
from run_relational_teacher_v982_rollout_state_calibration6 import (  # noqa: E402
    _state_sha256,
)
from run_relational_teacher_v984_preservation_calibration6 import (  # noqa: E402
    _direction,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402
from relational_teacher_v988_rollout_aligned_direction_contract import (  # noqa: E402
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
    ROLES,
    SCHEMA,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    TASK_ORDER,
    V987_SCHEMA,
    canonical_sha256,
    conflict_pairs,
    diagnose_directions,
    rank_eligible_directions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-radius-summary", type=Path, required=True)
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


def _validate_bound_files(value: Mapping[str, object], label: str) -> None:
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError(label + " path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError(label + " bound file changed: " + str(name))


def _validate_v987(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    expected_audit = [
        "bed_pooled_mae_strictly_improves",
        "bed_pooled_recall_strictly_improves",
    ]
    if (
        value.get("schema") != V987_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_radius") is not None
        or value.get("eligible_selection_order") != []
        or value.get("failed_checks") != ["at_least_one_radius_is_admissible"]
        or value.get("authorizes_cross_scene_objective_or_inference_redesign") is not True
        or value.get("authorizes_fresh_two_scene_common_direction_calibration") is not False
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.7 failure authority changed")
    rows = value.get("response_rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Teacher-v9.8.7 radius inventory changed")
    for row in rows:
        if (
            row.get("eligible") is not False
            or row.get("scene_failed_checks", {}).get(SOURCE_SCENE) != []
            or row.get("scene_failed_checks", {}).get(AUDIT_SCENE) != expected_audit
            or row.get("failed_checks") != ["room0102_response_is_admissible"]
        ):
            raise ValueError("Teacher-v9.8.7 Bed-only audit failure changed")
    _validate_bound_files(value, "Teacher-v9.8.7")
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.7 failure contains forbidden model state")
    return value


def _rollout_task_gradients(
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
        model, diffusion, states, kwargs, batch, mean_tensor, std_tensor
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
        raise RuntimeError("Teacher-v9.8.8 rollout-aligned diagnosis requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != MODEL_SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.8 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.8 output")
    configure_reproducibility(args.seed)

    v987_file = args.failed_radius_summary.expanduser().resolve()
    v987 = _validate_v987(v987_file)
    v987_maps_file = Path(str(v987["paths"]["response_maps"])).resolve()
    v987_arrays = _load_npz(v987_maps_file)
    v984_file = Path(str(v987["paths"]["v984_summary"])).resolve()
    v984 = read_json(v984_file)
    if (
        v984.get("schema") != V984_REPORT_SCHEMA
        or v984.get("status") != "PASS"
        or v984.get("policy_id") != V984_POLICY_ID
        or v984.get("shortlisted_steps") != [4, 3]
        or v984.get("failed_checks")
        or v984.get("binding_id") != v987.get("v984_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.4 authority changed")
    v984_arrays = _load_npz(Path(str(v987["paths"]["v984_maps"])).resolve())
    v98_file = Path(str(v987["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_arrays = _load_npz(Path(str(v987["paths"]["v98_maps"])).resolve())

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _same_bound_arg(args.index, v987, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v987, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v987, "stats_file")
    original_checkpoint = _same_bound_arg(args.original_checkpoint, v987, "original_checkpoint")
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v987, "v5_checkpoint")
    evidence_file = _same_bound_arg(args.v5_evidence_report, v987, "v5_evidence_report")
    source_index = Path(str(v987["paths"]["source_dataset_index"])).resolve()
    if dataset_root != index_file.parent or source_root != source_index.parent:
        raise ValueError("dataset roots differ from sealed Teacher-v9.8.7")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "rollout_aligned_direction_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("rollout-aligned direction policy hash changed")

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {SOURCE_SCENE, AUDIT_SCENE}:
        raise ValueError("two train-scene metadata inventory changed")
    bundles = {
        SOURCE_SCENE: load_train_scene_bundle(
            dataset_root, source_root, records[SOURCE_SCENE]
        ),
        AUDIT_SCENE: load_train_scene_bundle(
            dataset_root, source_root, records[AUDIT_SCENE]
        ),
    }
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        if tuple(bundles[scene]["instance_names"]) != tuple(EXPECTED_INSTANCES[scene]):
            raise ValueError(scene + " verified target order changed")

    mean, std = load_stats(stats_file)
    batches = {
        scene: _build_batch(bundles[scene], mean, std, args.device)
        for scene in (SOURCE_SCENE, AUDIT_SCENE)
    }
    kwargs = {scene: _kwargs(batches[scene]) for scene in (SOURCE_SCENE, AUDIT_SCENE)}
    if any(
        tuple(sorted(kwargs[scene])) != tuple(sorted(FORWARD_INPUT_KEYS))
        for scene in (SOURCE_SCENE, AUDIT_SCENE)
    ):
        raise AssertionError("Teacher forward input changed")
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    metric_policy = read_json(Path(str(v987["paths"]["metric_policy"])).resolve())
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
    base_normalized = np.empty((2, 3, 2, 8192, 6), np.float32)
    rollout_states: Dict[str, Dict[int, torch.Tensor]] = {
        SOURCE_SCENE: {},
        AUDIT_SCENE: {},
    }
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        for generation in range(3):
            seed_pair = stable_rollout_seeds(generation)
            state_pair = []
            for prompt_index, text in enumerate(prompt_texts):
                final, states, rng_states = _capture_trajectory(
                    model,
                    diffusion,
                    bundles[scene],
                    text,
                    seed_pair[0],
                    seed_pair[1],
                    args.device,
                    not args.no_progress,
                )
                repeated = _resume_trajectory(
                    model,
                    diffusion,
                    bundles[scene],
                    text,
                    states[SELECTED_TIMESTEP],
                    rng_states[SELECTED_TIMESTEP],
                    SELECTED_TIMESTEP,
                    args.device,
                )
                if not torch.equal(final, repeated):
                    raise AssertionError(scene + " rollout-aligned Base resume is not exact")
                base_normalized[scene_index, generation, prompt_index] = final.numpy()
                state_pair.append(states[SELECTED_TIMESTEP])
                print(
                    "[ROLLOUT-STATE] scene={} generation={} prompt={}".format(
                        scene, generation, PROMPT_IDS[prompt_index]
                    ),
                    flush=True,
                )
                del final, repeated, states, rng_states
            rollout_states[scene][generation] = torch.cat(state_pair, dim=0).cpu()
    if not np.array_equal(base_normalized, v987_arrays["base_normalized"]):
        raise AssertionError("exact Teacher-v9.8.7 Base K=3 maps did not reproduce")

    reconstruction_cache: Dict[int, list[torch.Tensor]] = {}
    for step in range(2, SELECTED_V984_STEP + 1):
        seed_pair = calibration_design_seeds(step)
        reconstruction_cache[step] = []
        for prompt_index, text in enumerate(prompt_texts):
            final, states, rng_states = _capture_trajectory(
                model,
                diffusion,
                bundles[SOURCE_SCENE],
                text,
                seed_pair[0],
                seed_pair[1],
                args.device,
                not args.no_progress,
            )
            repeated = _resume_trajectory(
                model,
                diffusion,
                bundles[SOURCE_SCENE],
                text,
                states[SELECTED_TIMESTEP],
                rng_states[SELECTED_TIMESTEP],
                SELECTED_TIMESTEP,
                args.device,
            )
            if not torch.equal(final, repeated):
                raise AssertionError("reconstruction Base resume is not exact")
            reconstruction_cache[step].append(states[SELECTED_TIMESTEP])
            print(
                "[RECONSTRUCTION-DESIGN] update={} prompt={}".format(
                    step, PROMPT_IDS[prompt_index]
                ),
                flush=True,
            )
            del final, repeated, states, rng_states

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
                kwargs[SOURCE_SCENE],
                batches[SOURCE_SCENE],
                mean_tensor,
                std_tensor,
            )
            direction, direction_row = _direction(tasks, parameters, 4096)
            design_seed_row = list(v98["design_seeds"])
        else:
            task_order = list(V984_TASK_ORDER)
            design_states = torch.cat(reconstruction_cache[step], dim=0).to(args.device)
            sit_tasks, design_prediction, design_frozen = _sit_tasks(
                model,
                diffusion,
                design_states,
                kwargs[SOURCE_SCENE],
                batches[SOURCE_SCENE],
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
            negative_mask = batches[SOURCE_SCENE]["explicit_negative_mask"].bool()
            negative_tasks = torch.stack(
                [
                    design_physical[prompt][negative_mask[prompt]].mean()
                    for prompt in range(2)
                ]
            )
            tasks = torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)
            direction, direction_row = _direction(tasks, parameters, 8192)
            design_seed_row = list(calibration_design_seeds(step))
        apply_flat_direction(parameters, direction, 0.003)
        post_state_sha256 = _state_sha256(named_lora)
        direction_row.update(
            {
                "step": step,
                "task_order": task_order,
                "pre_state_sha256": pre_state_sha256,
                "post_state_sha256": post_state_sha256,
                "design_seeds": design_seed_row,
            }
        )
        _direction_exact(direction_row, v984["direction_rows"][step - 1], step)
        reconstruction_rows.append(direction_row)
        print(
            "[RECONSTRUCTION_PASS] exact Teacher-v9.8.4 update={}".format(step),
            flush=True,
        )
        del tasks, design_prediction, design_frozen, direction, design_states
        if step > 1:
            del sit_tasks, live_v5_prediction, v5_tasks, design_physical, negative_tasks
        gc.collect()
        torch.cuda.empty_cache()

    selected_state_sha256 = _state_sha256(named_lora)
    if selected_state_sha256 != v987["selected_state_sha256"]:
        raise AssertionError("reconstructed step-4 state differs from v9.8.7")

    gradient_groups = []
    task_losses = []
    actual_task_order = []
    for scene_label, scene in (("source", SOURCE_SCENE), ("audit", AUDIT_SCENE)):
        for generation in range(3):
            group, losses = _rollout_task_gradients(
                model=model,
                diffusion=diffusion,
                states=rollout_states[scene][generation].to(args.device),
                kwargs=kwargs[scene],
                batch=batches[scene],
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                parameters=parameters,
            )
            gradient_groups.append(group)
            task_losses.extend(losses)
            for prompt in ("watch", "write"):
                for role in ROLES:
                    actual_task_order.append(
                        "{}_g{}_{}_{}".format(
                            scene_label, generation, prompt, role
                        )
                    )
            for prompt in ("watch", "write"):
                actual_task_order.append(
                    "{}_g{}_{}_negative".format(scene_label, generation, prompt)
                )
            print(
                "[GRADIENTS] scene={} generation={} tasks=8".format(
                    scene, generation
                ),
                flush=True,
            )

    live_v5_prediction = predict_xstart(
        model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
    )
    v5_tasks = (live_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
    v5_gradients = flattened_task_gradients(v5_tasks, parameters).detach().cpu()
    gradient_groups.append(v5_gradients)
    task_losses.extend(float(value) for value in v5_tasks.detach().cpu().tolist())
    actual_task_order.extend(("v5_chair", "v5_bed", "v5_whiteboard"))
    if tuple(actual_task_order) != TASK_ORDER:
        raise AssertionError("rollout-aligned task order changed")
    gradients = torch.cat(gradient_groups, dim=0)
    if gradients.shape[0] != len(TASK_ORDER):
        raise AssertionError("51-task gradient inventory changed")
    raw_gram = (gradients.double() @ gradients.double().T).numpy()
    raw_asymmetry = float(np.abs(raw_gram - raw_gram.T).max())
    if not np.isfinite(raw_gram).all() or raw_asymmetry > RAW_GRAM_ASYMMETRY_CAP:
        raise RuntimeError("rollout-aligned raw Gram matrix is invalid")
    gram = 0.5 * (raw_gram + raw_gram.T)
    direction_rows = diagnose_directions(gram)
    if [row["name"] for row in direction_rows] != list(CANDIDATE_NAMES):
        raise AssertionError("rollout-aligned direction order changed")
    direction_hashes = []
    for row in direction_rows:
        coefficients = torch.tensor(
            row["effective_coefficients"], dtype=gradients.dtype
        )
        direction = coefficients @ gradients
        direction_l2 = float(torch.linalg.vector_norm(direction).item())
        if not math.isclose(direction_l2, 1.0, rel_tol=2e-5, abs_tol=2e-6):
            raise AssertionError("rollout-aligned direction is not unit length")
        row["direction_sha256"] = tensor_sha256(direction)
        direction_hashes.append(row["direction_sha256"])
        print(
            "[DIRECTION {}] eligible={} audit-bed-min={:.9f} all-min={:.9f}".format(
                row["name"],
                row["eligible"],
                row["minimum_audit_bed_derivative"],
                row["minimum_all_task_derivative"],
            ),
            flush=True,
        )
    if _state_sha256(named_lora) != selected_state_sha256:
        raise AssertionError("diagnosis mutated the reconstructed step-4 state")

    eligible_order = rank_eligible_directions(direction_rows)
    selected_candidate = eligible_order[0] if eligible_order else None
    geometry_file = output_dir / "rollout_aligned_gradient_geometry.npz"
    atomic_savez(
        geometry_file,
        task_order=np.asarray(TASK_ORDER),
        task_losses=np.asarray(task_losses, np.float64),
        gram=np.asarray(gram, np.float64),
        raw_gram_max_asymmetry=np.asarray(raw_asymmetry, np.float64),
        candidate_names=np.asarray(CANDIDATE_NAMES),
        coefficients=np.asarray(
            [row["effective_coefficients"] for row in direction_rows], np.float64
        ),
        directional_derivatives=np.asarray(
            [row["directional_derivatives"] for row in direction_rows], np.float64
        ),
        direction_sha256=np.asarray(direction_hashes),
        base_normalized=base_normalized,
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v988_rollout_aligned_direction.py",
        "contract": PREPARE_ROOT / "relational_teacher_v988_rollout_aligned_direction_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v988_rollout_aligned_direction.py",
        "v987_summary": v987_file,
        "v987_maps": v987_maps_file,
        "v984_summary": v984_file,
        "v984_maps": Path(str(v987["paths"]["v984_maps"])).resolve(),
        "v98_report": v98_file,
        "v98_maps": Path(str(v987["paths"]["v98_maps"])).resolve(),
        "metric_policy": Path(str(v987["paths"]["metric_policy"])).resolve(),
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
        "sealed_v987_room0102_bed_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exact_v987_two_scene_k3_base_states_reproduced": True,
        "exactly_51_live_normalized_rollout_task_gradients": True,
        "at_least_one_rollout_aligned_common_direction_exists": (
            selected_candidate is not None
        ),
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
        "prompt_ids": list(PROMPT_IDS),
        "trajectory_seed_table": [
            list(stable_rollout_seeds(generation)) for generation in range(3)
        ],
        "task_order": list(TASK_ORDER),
        "task_losses": task_losses,
        "gram": gram.tolist(),
        "raw_gram_max_asymmetry": raw_asymmetry,
        "conflict_pairs": conflict_pairs(gram),
        "conflict_pair_total": len(TASK_ORDER) * (len(TASK_ORDER) - 1) // 2,
        "direction_candidates": direction_rows,
        "eligible_selection_order": eligible_order,
        "selected_candidate": selected_candidate,
        "v987_binding_id": v987["binding_id"],
        "v984_binding_id": v984["binding_id"],
        "source_scene_binding": _scene_binding(records[SOURCE_SCENE]),
        "audit_scene_binding": _scene_binding(records[AUDIT_SCENE]),
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["diagnosis_policy"],
        "lora": dict(lora_metadata(model)),
        "zero_state_sha256": zero_state_sha256,
        "selected_state_sha256": selected_state_sha256,
        "reconstruction_rows": reconstruction_rows,
        "checks": top_checks,
        "failed_checks": sorted(
            name for name, passed in top_checks.items() if not passed
        ),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "gradient_geometry_sha256": path_hashes["gradient_geometry"],
        "serialized_model_state": False,
        "authorizes_actual_two_scene_k3_rollout_aligned_radius_grid": (
            status == "PASS"
        ),
        "authorizes_cross_scene_objective_or_inference_redesign": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v987_binding_id": report["v987_binding_id"],
            "policy_id": report["policy_id"],
            "selected_state_sha256": report["selected_state_sha256"],
            "gradient_geometry_sha256": report["gradient_geometry_sha256"],
            "selected_candidate": report["selected_candidate"],
            "status": report["status"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.8 report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.8 unexpectedly saved model state")
    print("[ROLLOUT_ALIGNED_DIRECTION_{}] Teacher-v9.8.8".format(status))
    print("[PASS] exact v9.8.7 K=3 t50 states and step-4 state verified")
    print("[PASS] 51 rollout-aligned task gradients and candidates computed")
    print("[OK] conflicts: {} / {}".format(report["conflict_pairs"], report["conflict_pair_total"]))
    print("[OK] selected candidate:", selected_candidate)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)
    del model, diffusion, gradients, live_v5_prediction, v5_tasks
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

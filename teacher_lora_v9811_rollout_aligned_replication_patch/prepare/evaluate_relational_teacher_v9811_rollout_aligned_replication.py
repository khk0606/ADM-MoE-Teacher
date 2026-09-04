#!/usr/bin/env python3
"""Replicate the selected v9.8.10 state on a disjoint two-scene K=3 panel."""

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

from evaluate_relational_teacher_v987_two_scene_radius_response import (  # noqa: E402
    _close_array,
    _restore_state,
)
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
    _invariance,
    _load_npz,
    _rows,
    _same_bound_arg,
    _sit_tasks,
)
from preflight_relational_teacher_v985_two_scene_step4 import (  # noqa: E402
    _direction_exact,
    _scene_binding,
)
from preflight_relational_teacher_v988_rollout_aligned_direction import (  # noqa: E402
    _rollout_task_gradients,
    _validate_v987,
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
from relational_teacher_v97_early_rollout_k3_contract import stable_rollout_seeds  # noqa: E402
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
    absolute_presence_checks,
    pooled_by_role,
    two_scene_preflight_checks,
)
from relational_teacher_v988_rollout_aligned_direction_contract import (  # noqa: E402
    CANDIDATE_NAMES as V988_CANDIDATE_NAMES,
    POLICY_ID as V988_POLICY_ID,
    RAW_GRAM_ASYMMETRY_CAP,
    TASK_ORDER as V988_TASK_ORDER,
    diagnose_directions,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
)
from run_relational_teacher_v982_rollout_state_calibration6 import _state_sha256  # noqa: E402
from run_relational_teacher_v984_preservation_calibration6 import _direction  # noqa: E402
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402
from relational_teacher_v9811_rollout_aligned_replication_contract import (  # noqa: E402
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    GENERATION_COUNT,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    REPLICATION_SEED_TABLE,
    REPLICATION_TAG,
    RECONSTRUCTION_STEPS,
    SCHEMA,
    SELECTED_DIRECTION,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    V9810_SCHEMA,
    V988_SCHEMA,
    canonical_sha256,
    replication_checks,
    replication_rollout_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-recovery-summary", type=Path, required=True)
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


def _validate_v988(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V988_SCHEMA
        or value.get("status") != "PASS"
        or value.get("policy_id") != V988_POLICY_ID
        or value.get("selected_candidate") != SELECTED_DIRECTION
        or not value.get("eligible_selection_order")
        or value["eligible_selection_order"][0] != SELECTED_DIRECTION
        or value.get("failed_checks")
        or value.get("authorizes_actual_two_scene_k3_rollout_aligned_radius_grid") is not True
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.8 PASS authority changed")
    selected = [
        row
        for row in value.get("direction_candidates", [])
        if row.get("name") == SELECTED_DIRECTION
    ]
    if len(selected) != 1 or selected[0].get("eligible") is not True or selected[0].get(
        "failed_checks"
    ):
        raise ValueError("sealed Teacher-v9.8.8 selected direction changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("sealed Teacher-v9.8.8 path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("sealed Teacher-v9.8.8 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("sealed Teacher-v9.8.8 contains forbidden model state")
    return value


def _validate_v9810(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V9810_SCHEMA
        or value.get("status") != "PASS"
        or float(value.get("selected_radius", -1.0)) != SELECTED_RADIUS
        or value.get("failed_checks")
        or value.get("authorizes_fresh_two_scene_rollout_aligned_calibration") is not True
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.10 PASS authority changed")
    selected = [
        row
        for row in value.get("response_rows", [])
        if float(row.get("radius", -1.0)) == SELECTED_RADIUS
    ]
    if len(selected) != 1 or selected[0].get("eligible") is not True or selected[0].get(
        "failed_checks"
    ):
        raise ValueError("sealed Teacher-v9.8.10 selected response changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("sealed Teacher-v9.8.10 path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("sealed Teacher-v9.8.10 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("sealed Teacher-v9.8.10 contains forbidden model state")
    return value


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8.11 new-seed replication requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != MODEL_SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.11 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.11 output")
    configure_reproducibility(args.seed)

    v9810_file = args.selected_recovery_summary.expanduser().resolve()
    v9810 = _validate_v9810(v9810_file)
    v9810_maps_file = Path(str(v9810["paths"]["response_maps"])).resolve()
    v9810_arrays = _load_npz(v9810_maps_file)
    v988_file = Path(str(v9810["paths"]["v988_report"])).resolve()
    v988 = _validate_v988(v988_file)
    if v9810.get("v988_binding_id") != v988.get("binding_id"):
        raise ValueError("sealed Teacher-v9.8.10 to v9.8.8 binding changed")
    v988_geometry_file = Path(str(v988["paths"]["gradient_geometry"])).resolve()
    v988_geometry = _load_npz(v988_geometry_file)
    v987_file = Path(str(v988["paths"]["v987_summary"])).resolve()
    v987 = _validate_v987(v987_file)
    if v987.get("binding_id") != v988.get("v987_binding_id"):
        raise ValueError("sealed Teacher-v9.8.7 binding changed")
    v987_maps_file = Path(str(v988["paths"]["v987_maps"])).resolve()
    v987_arrays = _load_npz(v987_maps_file)
    v984_file = Path(str(v988["paths"]["v984_summary"])).resolve()
    v984 = read_json(v984_file)
    if (
        v984.get("schema") != V984_REPORT_SCHEMA
        or v984.get("status") != "PASS"
        or v984.get("policy_id") != V984_POLICY_ID
        or v984.get("shortlisted_steps") != [4, 3]
        or v984.get("failed_checks")
        or v984.get("binding_id") != v988.get("v984_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.4 authority changed")
    v984_maps_file = Path(str(v988["paths"]["v984_maps"])).resolve()
    v984_arrays = _load_npz(v984_maps_file)
    v98_file = Path(str(v988["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_maps_file = Path(str(v988["paths"]["v98_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _same_bound_arg(args.index, v988, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v988, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v988, "stats_file")
    original_checkpoint = _same_bound_arg(args.original_checkpoint, v988, "original_checkpoint")
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v988, "v5_checkpoint")
    evidence_file = _same_bound_arg(args.v5_evidence_report, v988, "v5_evidence_report")
    source_index = Path(str(v988["paths"]["source_dataset_index"])).resolve()
    if dataset_root != index_file.parent or source_root != source_index.parent:
        raise ValueError("dataset roots differ from sealed Teacher-v9.8.8")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "rollout_aligned_replication_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("rollout-aligned replication policy hash changed")

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

    metric_policy = read_json(Path(str(v988["paths"]["metric_policy"])).resolve())
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
    selection_base_normalized = np.empty(
        (2, GENERATION_COUNT, 2, 8192, 6), np.float32
    )
    selection_cache: Dict[str, Dict[int, list]] = {
        SOURCE_SCENE: {}, AUDIT_SCENE: {}
    }
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        for generation in range(GENERATION_COUNT):
            seed_pair = stable_rollout_seeds(generation)
            selection_cache[scene][generation] = []
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
                    raise AssertionError(scene + " Base resume is not exact")
                selection_base_normalized[
                    scene_index, generation, prompt_index
                ] = final.numpy()
                selection_cache[scene][generation].append(
                    (
                        states[SELECTED_TIMESTEP].cpu(),
                        rng_states[SELECTED_TIMESTEP],
                    )
                )
                print(
                    "[SELECTION-BASE] scene={} generation={} prompt={}".format(
                        scene, generation, PROMPT_IDS[prompt_index]
                    ),
                    flush=True,
                )
                del final, repeated, states, rng_states
    if not np.array_equal(
        selection_base_normalized, v988_geometry["base_normalized"]
    ):
        raise AssertionError("exact Teacher-v9.8.8 K=3 Base states did not reproduce")
    if not np.array_equal(
        selection_base_normalized, v987_arrays["base_normalized"]
    ):
        raise AssertionError("exact Teacher-v9.8.7 K=3 Base maps changed")
    if not np.array_equal(
        selection_base_normalized, v9810_arrays["base_normalized"]
    ):
        raise AssertionError("exact Teacher-v9.8.10 selection Base maps changed")

    selection_seed_table = tuple(
        stable_rollout_seeds(generation) for generation in range(GENERATION_COUNT)
    )
    replication_seed_table = tuple(
        replication_rollout_seeds(generation) for generation in range(GENERATION_COUNT)
    )
    seed_table_disjoint = not (
        {value for row in selection_seed_table for value in row}
        & {value for row in replication_seed_table for value in row}
    )
    if replication_seed_table != REPLICATION_SEED_TABLE or not seed_table_disjoint:
        raise AssertionError("new-seed replication table is invalid or overlaps selection")

    base_normalized = np.empty((2, GENERATION_COUNT, 2, 8192, 6), np.float32)
    base_repeat_normalized = np.empty_like(base_normalized)
    replication_cache: Dict[str, Dict[int, list]] = {
        SOURCE_SCENE: {}, AUDIT_SCENE: {}
    }
    deterministic_repeats_exact = True
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        for generation in range(GENERATION_COUNT):
            seed_pair = replication_seed_table[generation]
            replication_cache[scene][generation] = []
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
                exact = torch.equal(final, repeated)
                deterministic_repeats_exact = deterministic_repeats_exact and exact
                if not exact:
                    raise AssertionError(scene + " new-seed Base resume is not exact")
                base_normalized[scene_index, generation, prompt_index] = final.numpy()
                base_repeat_normalized[
                    scene_index, generation, prompt_index
                ] = repeated.numpy()
                replication_cache[scene][generation].append(
                    (
                        states[SELECTED_TIMESTEP].cpu(),
                        rng_states[SELECTED_TIMESTEP],
                    )
                )
                print(
                    "[REPLICATION-BASE] scene={} generation={} prompt={}".format(
                        scene, generation, PROMPT_IDS[prompt_index]
                    ),
                    flush=True,
                )
                del final, repeated, states, rng_states

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
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )

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
            design_physical = physical_prediction(design_prediction, mean_tensor, std_tensor)
            negative_mask = batches[SOURCE_SCENE]["explicit_negative_mask"].bool()
            negative_tasks = torch.stack(
                [design_physical[prompt][negative_mask[prompt]].mean() for prompt in range(2)]
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
        print("[RECONSTRUCTION_PASS] exact Teacher-v9.8.4 update={}".format(step), flush=True)
        del tasks, design_prediction, design_frozen, direction, design_states
        if step > 1:
            del sit_tasks, live_v5_prediction, v5_tasks, design_physical, negative_tasks
        gc.collect()
        torch.cuda.empty_cache()

    selected_state_sha256 = _state_sha256(named_lora)
    if selected_state_sha256 != v988["selected_state_sha256"]:
        raise AssertionError("reconstructed step-4 state differs from v9.8.8")
    step4_state = {
        name: parameter.detach().cpu().clone() for name, parameter in named_lora.items()
    }

    gradient_groups = []
    task_losses = []
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        for generation in range(3):
            states = torch.cat(
                [selection_cache[scene][generation][prompt][0] for prompt in range(2)],
                dim=0,
            ).to(args.device)
            group, losses = _rollout_task_gradients(
                model=model,
                diffusion=diffusion,
                states=states,
                kwargs=kwargs[scene],
                batch=batches[scene],
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                parameters=parameters,
            )
            gradient_groups.append(group)
            task_losses.extend(losses)
            del states
    live_v5_prediction = predict_xstart(
        model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
    )
    v5_tasks = (live_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2))
    gradient_groups.append(flattened_task_gradients(v5_tasks, parameters).detach().cpu())
    task_losses.extend(float(value) for value in v5_tasks.detach().cpu().tolist())
    gradients = torch.cat(gradient_groups, dim=0)
    if gradients.shape[0] != len(V988_TASK_ORDER):
        raise AssertionError("51-task gradient inventory changed")
    raw_gram = (gradients.double() @ gradients.double().T).numpy()
    raw_asymmetry = float(np.abs(raw_gram - raw_gram.T).max())
    if not np.isfinite(raw_gram).all() or raw_asymmetry > RAW_GRAM_ASYMMETRY_CAP:
        raise RuntimeError("v9.8.11 raw Gram matrix is invalid")
    gram = 0.5 * (raw_gram + raw_gram.T)
    _close_array(task_losses, v988_geometry["task_losses"], "task losses")
    _close_array(gram, v988_geometry["gram"], "Gram matrix")
    diagnostic_rows = diagnose_directions(gram)
    if [row["name"] for row in diagnostic_rows] != list(V988_CANDIDATE_NAMES):
        raise AssertionError("v9.8.8 candidate order changed")
    selected_rows = [row for row in diagnostic_rows if row["name"] == SELECTED_DIRECTION]
    if len(selected_rows) != 1 or selected_rows[0]["eligible"] is not True:
        raise AssertionError("selected v9.8.8 direction did not reproduce")
    selected_coefficients = torch.tensor(
        selected_rows[0]["effective_coefficients"], dtype=gradients.dtype
    )
    selected_direction = selected_coefficients @ gradients
    selected_direction_l2 = float(torch.linalg.vector_norm(selected_direction).item())
    if not math.isclose(selected_direction_l2, 1.0, rel_tol=2e-5, abs_tol=2e-6):
        raise AssertionError("selected v9.8.8 direction is not unit length")
    selected_direction_sha256 = tensor_sha256(selected_direction)
    sealed_selected = [
        row for row in v988["direction_candidates"] if row["name"] == SELECTED_DIRECTION
    ][0]
    if selected_direction_sha256 != sealed_selected["direction_sha256"]:
        raise AssertionError("selected v9.8.8 direction hash changed")
    if _state_sha256(named_lora) != selected_state_sha256:
        raise AssertionError("direction recomputation mutated the step-4 state")

    base_physical = np.clip(
        base_normalized * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    scene_base_rows = {}
    scene_base_pooled = {}
    scene_base_invariance = {}
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        scene_base_rows[scene] = _rows(bundles[scene], base_physical[scene_index])
        scene_base_pooled[scene] = pooled_by_role(
            scene_base_rows[scene], list(bundles[scene]["instance_names"])
        )
        scene_base_invariance[scene] = _invariance(
            base_physical[scene_index],
            np.asarray(bundles[scene]["verified_positive_mask"], bool),
        )

    _restore_state(named_lora, step4_state)
    if _state_sha256(named_lora) != selected_state_sha256:
        raise AssertionError("selected candidate did not start from exact step 4")
    apply_flat_direction(parameters, selected_direction.to(args.device), SELECTED_RADIUS)
    candidate_state_sha256 = _state_sha256(named_lora)
    sealed_v9810_row = [
        row
        for row in v9810["response_rows"]
        if float(row["radius"]) == SELECTED_RADIUS
    ][0]
    selected_state_exact = (
        candidate_state_sha256 == sealed_v9810_row["candidate_state_sha256"]
    )
    if not selected_state_exact:
        raise AssertionError("selected Teacher-v9.8.10 LoRA state did not reproduce")

    candidate_normalized = np.empty_like(base_normalized)
    model.eval()
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        for generation in range(GENERATION_COUNT):
            for prompt_index, text in enumerate(prompt_texts):
                state, rng_state = replication_cache[scene][generation][prompt_index]
                final = _resume_trajectory(
                    model,
                    diffusion,
                    bundles[scene],
                    text,
                    state,
                    rng_state,
                    SELECTED_TIMESTEP,
                    args.device,
                )
                candidate_normalized[
                    scene_index, generation, prompt_index
                ] = final.numpy()
                print(
                    "[REPLICATION] scene={} generation={} prompt={}".format(
                        scene, generation, PROMPT_IDS[prompt_index]
                    ),
                    flush=True,
                )
                del final

    with torch.no_grad():
        candidate_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    candidate_v5_prediction = (
        candidate_v5.detach().cpu().numpy().astype(np.float32)
    )
    candidate_v5_dense = (
        (candidate_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    selected_index = next(
        index
        for index, row in enumerate(v9810["response_rows"])
        if float(row["radius"]) == SELECTED_RADIUS
    )
    selected_v5_exact = np.array_equal(
        candidate_v5_prediction,
        v9810_arrays["candidate_v5_predictions"][selected_index],
    ) and np.array_equal(
        np.asarray(candidate_v5_dense, np.float64),
        np.asarray(sealed_v9810_row["candidate_v5_dense"], np.float64),
    )
    if not selected_v5_exact:
        raise AssertionError("selected Teacher-v9.8.10 v5 response did not reproduce")

    candidate_physical = np.clip(
        candidate_normalized * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    scene_candidate_rows = {}
    scene_candidate_pooled = {}
    scene_candidate_invariance = {}
    scene_checks = {}
    scene_presence = {}
    maximum_map_delta = {}
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        instance_names = list(bundles[scene]["instance_names"])
        candidate_scene = candidate_physical[scene_index]
        scene_candidate_rows[scene] = _rows(bundles[scene], candidate_scene)
        scene_candidate_pooled[scene] = pooled_by_role(
            scene_candidate_rows[scene], instance_names
        )
        scene_candidate_invariance[scene] = _invariance(
            candidate_scene,
            np.asarray(bundles[scene]["verified_positive_mask"], bool),
        )
        maximum_map_delta[scene] = float(
            np.abs(candidate_scene - base_physical[scene_index]).max()
        )
        scene_checks[scene] = two_scene_preflight_checks(
            base_rows=scene_base_rows[scene],
            candidate_rows=scene_candidate_rows[scene],
            instance_names=instance_names,
            base_prompt_invariance=scene_base_invariance[scene],
            candidate_prompt_invariance=scene_candidate_invariance[scene],
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=candidate_v5_dense,
            maximum_map_delta=maximum_map_delta[scene],
        )
        scene_presence[scene] = [
            [
                absolute_presence_checks(
                    scene_candidate_rows[scene][generation][prompt], instance_names
                )
                for prompt in range(2)
            ]
            for generation in range(GENERATION_COUNT)
        ]
    response_checks = replication_checks(
        source_checks=scene_checks[SOURCE_SCENE],
        audit_checks=scene_checks[AUDIT_SCENE],
        selected_state_exact=selected_state_exact,
        selected_v5_exact=selected_v5_exact,
        deterministic_repeats_exact=deterministic_repeats_exact,
        seed_table_disjoint=seed_table_disjoint,
    )
    response_failed_checks = sorted(
        name for name, passed in response_checks.items() if not passed
    )
    response_eligible = all(response_checks.values())
    print(
        "[NEW-SEED RESPONSE] eligible={} room0101_failed={} room0102_failed={}".format(
            response_eligible,
            len([name for name, passed in scene_checks[SOURCE_SCENE].items() if not passed]),
            len([name for name, passed in scene_checks[AUDIT_SCENE].items() if not passed]),
        ),
        flush=True,
    )

    arrays_file = output_dir / "rollout_aligned_replication_maps.npz"
    save_arrays = {
        "scene_ids": np.asarray((SOURCE_SCENE, AUDIT_SCENE)),
        "prompt_ids": np.asarray(PROMPT_IDS),
        "selection_seed_table": np.asarray(selection_seed_table, np.int64),
        "replication_seed_table": np.asarray(replication_seed_table, np.int64),
        "selection_base_normalized": selection_base_normalized,
        "base_normalized": base_normalized,
        "base_repeat_normalized": base_repeat_normalized,
        "candidate_normalized": candidate_normalized,
        "base": base_physical,
        "candidate": candidate_physical,
        "v5_target": v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        "base_v5_prediction": base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        "candidate_v5_prediction": candidate_v5_prediction,
    }
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        prefix = "source" if scene == SOURCE_SCENE else "audit"
        bundle = bundles[scene]
        save_arrays.update(
            {
                prefix + "_xyz": np.asarray(bundle["xyz"], np.float32),
                prefix + "_points": np.asarray(bundle["points"], np.float32),
                prefix + "_instance_ids": np.asarray(bundle["instance_ids"], np.int64),
                prefix + "_category_ids": np.asarray(bundle["category_ids"], np.int64),
                prefix + "_instance_names": np.asarray(bundle["instance_names"]),
                prefix + "_verified_object_mask": np.asarray(bundle["verified_object_mask"], bool),
                prefix + "_verified_positive_mask": np.asarray(bundle["verified_positive_mask"], bool),
                prefix + "_unknown_sittable_mask": np.asarray(bundle["unknown_sittable_mask"], bool),
                prefix + "_explicit_negative_mask": np.asarray(bundle["explicit_negative_mask"], bool),
                prefix + "_instance_targets": np.asarray(bundle["instance_targets"], np.float32),
                prefix + "_all_sittable_gt": np.asarray(bundle["all_target"], np.float32),
            }
        )
    atomic_savez(arrays_file, **save_arrays)

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v9811_rollout_aligned_replication.py",
        "contract": PREPARE_ROOT / "relational_teacher_v9811_rollout_aligned_replication_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v9811_rollout_aligned_replication.py",
        "v9810_summary": v9810_file,
        "v9810_maps": v9810_maps_file,
        "v988_report": v988_file,
        "v988_geometry": v988_geometry_file,
        "v987_summary": v987_file,
        "v987_maps": v987_maps_file,
        "v984_summary": v984_file,
        "v984_maps": v984_maps_file,
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "metric_policy": Path(str(v988["paths"]["metric_policy"])).resolve(),
        "response_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "response_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    top_checks = {
        "sealed_v9810_pass_and_selected_radius_bound": True,
        "sealed_v988_pass_and_selected_direction_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exact_v988_51_task_direction_recomputed": True,
        "selected_v9810_state_exactly_reproduced": selected_state_exact,
        "selected_v9810_v5_response_exactly_reproduced": selected_v5_exact,
        "selection_and_replication_seed_tables_are_disjoint": seed_table_disjoint,
        "new_seed_base_resumes_are_exact": deterministic_repeats_exact,
        "both_scene_new_seed_k3_two_prompt_response_completed":
        candidate_normalized.shape == (2, GENERATION_COUNT, 2, 8192, 6),
        "new_seed_response_is_admissible": response_eligible,
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
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
        "replication_tag": REPLICATION_TAG,
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
        "selected_direction": SELECTED_DIRECTION,
        "selected_direction_sha256": selected_direction_sha256,
        "selected_radius": SELECTED_RADIUS,
        "prompt_ids": list(PROMPT_IDS),
        "forward_input_keys": sorted(kwargs[SOURCE_SCENE]),
        "selection_seed_table": [list(row) for row in selection_seed_table],
        "replication_seed_table": [list(row) for row in replication_seed_table],
        "trajectory_accounting": {
            "selection_two_scene_base_full_draws": 12,
            "selection_two_scene_base_exact_partial_repeats": 12,
            "replication_two_scene_base_full_draws": 12,
            "replication_two_scene_base_exact_partial_repeats": 12,
            "room0101_reconstruction_full_draws": 6,
            "room0101_reconstruction_exact_partial_repeats": 6,
            "rollout_aligned_gradient_groups": 6,
            "selected_candidate_partial_resumes": 12,
        },
        "v9810_binding_id": v9810["binding_id"],
        "v988_binding_id": v988["binding_id"],
        "v987_binding_id": v987["binding_id"],
        "v984_binding_id": v984["binding_id"],
        "source_scene_binding": _scene_binding(records[SOURCE_SCENE]),
        "audit_scene_binding": _scene_binding(records[AUDIT_SCENE]),
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["response_policy"],
        "lora": dict(lora_metadata(model)),
        "zero_state_sha256": zero_state_sha256,
        "step4_state_sha256": selected_state_sha256,
        "candidate_state_sha256": candidate_state_sha256,
        "reconstruction_rows": reconstruction_rows,
        "direction_task_losses": task_losses,
        "direction_gram": gram.tolist(),
        "direction_raw_gram_max_asymmetry": raw_asymmetry,
        "scene_base_rows": scene_base_rows,
        "scene_base_pooled": scene_base_pooled,
        "scene_base_invariance": scene_base_invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "scene_candidate_rows": scene_candidate_rows,
        "scene_candidate_pooled": scene_candidate_pooled,
        "scene_candidate_invariance": scene_candidate_invariance,
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "maximum_map_delta": maximum_map_delta,
        "presence": scene_presence,
        "scene_checks": scene_checks,
        "scene_failed_checks": {
            scene: sorted(
                name for name, passed in scene_checks[scene].items() if not passed
            )
            for scene in (SOURCE_SCENE, AUDIT_SCENE)
        },
        "response_checks": response_checks,
        "response_failed_checks": response_failed_checks,
        "response_eligible": response_eligible,
        "checks": top_checks,
        "failed_checks": sorted(name for name, passed in top_checks.items() if not passed),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "response_maps_sha256": path_hashes["response_maps"],
        "serialized_model_state": False,
        "authorizes_fresh_two_scene_rollout_aligned_multiupdate_calibration": status
        == "PASS",
        "authorizes_cross_scene_training_objective_redesign": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v9810_binding_id": report["v9810_binding_id"],
            "policy_id": report["policy_id"],
            "candidate_state_sha256": report["candidate_state_sha256"],
            "selected_direction_sha256": report["selected_direction_sha256"],
            "response_maps_sha256": report["response_maps_sha256"],
            "replication_seed_table": report["replication_seed_table"],
            "status": report["status"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.11 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.11 unexpectedly saved model state")
    print("[ROLLOUT_ALIGNED_REPLICATION_{}] Teacher-v9.8.11".format(status))
    print("[PASS] sealed v9.8.10 radius 0.006 state and exact v9.8.8 direction verified")
    print("[PASS] disjoint-seed two scenes x K=3 x two prompts completed")
    print("[OK] selected radius:", SELECTED_RADIUS)
    print("[OK] response failed checks:", response_failed_checks)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    del model, diffusion, gradients, selected_direction
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

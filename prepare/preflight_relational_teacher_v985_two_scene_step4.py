#!/usr/bin/env python3
"""Reconstruct Teacher-v9.8.4 step 4 and audit it on room_0102."""

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
    _invariance,
    _load_npz,
    _rows,
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
)
from relational_teacher_v94_common_descent import apply_flat_direction  # noqa: E402
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
    TASK_ORDER,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
)
from run_relational_teacher_v982_rollout_state_calibration6 import _state_sha256  # noqa: E402
from run_relational_teacher_v984_preservation_calibration6 import _direction  # noqa: E402
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402
from relational_teacher_v985_two_scene_step4_preflight_contract import (  # noqa: E402
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    EXPECTED_INSTANCES,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PREFLIGHT_TAG,
    PROMPT_IDS,
    RECONSTRUCTION_STEPS,
    SCHEMA,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    STEP_RADIUS,
    absolute_presence_checks,
    canonical_sha256,
    pooled_by_role,
    two_scene_preflight_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-summary", type=Path, required=True)
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


def _validate_v984(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V984_REPORT_SCHEMA
        or value.get("status") != "PASS"
        or value.get("policy_id") != V984_POLICY_ID
        or value.get("shortlisted_steps") != [4, 3]
        or value.get("failed_checks")
        or value.get("authorizes_two_scene_preservation_calibration_preflight") is not True
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_room_0102") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.4 PASS authority changed")
    if len(value.get("direction_rows", ())) != 6 or len(value.get("monitor_rows", ())) != 6:
        raise ValueError("sealed Teacher-v9.8.4 row inventory changed")
    if value["monitor_rows"][3].get("step") != 4 or value["monitor_rows"][3].get(
        "eligible"
    ) is not True:
        raise ValueError("sealed Teacher-v9.8.4 selected step changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("sealed Teacher-v9.8.4 path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("sealed Teacher-v9.8.4 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("sealed Teacher-v9.8.4 contains forbidden model state")
    return value


def _direction_exact(actual: Mapping[str, object], expected: Mapping[str, object], step: int) -> None:
    for key in (
        "task_order",
        "task_losses",
        "gram",
        "weights",
        "directional_derivatives",
        "direction_norm_before_unit",
        "direction_sha256",
        "pre_state_sha256",
        "post_state_sha256",
        "design_seeds",
    ):
        left = actual[key]
        right = expected[key]
        if isinstance(left, (list, tuple)) and left and isinstance(left[0], (int, float, list)):
            try:
                if not np.allclose(np.asarray(left, np.float64), np.asarray(right, np.float64), rtol=1e-7, atol=1e-8):
                    raise AssertionError("step {} reconstruction changed: {}".format(step, key))
                continue
            except (TypeError, ValueError):
                pass
        if left != right:
            raise AssertionError("step {} reconstruction changed: {}".format(step, key))


def _scene_binding(record: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "scene_id": record["scene_id"],
            "manifest_sha256": record["manifest_sha256"],
            "verified_target_instances": record["verified_target_instances"],
            "source_scene_record": record["source_scene_record"],
            "rows": record["rows"],
        }
    )


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8.5 two-scene preflight requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != MODEL_SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.5 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.5 output")
    configure_reproducibility(args.seed)

    v984_file = args.calibration_summary.expanduser().resolve()
    v984 = _validate_v984(v984_file)
    v984_maps_file = Path(str(v984["paths"]["calibration_maps"])).resolve()
    v984_arrays = _load_npz(v984_maps_file)
    v98_file = Path(str(v984["paths"]["v98_report"])).resolve()
    v98 = read_json(v98_file)
    v98_maps_file = Path(str(v984["paths"]["v98_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _same_bound_arg(args.index, v984, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v984, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v984, "stats_file")
    original_checkpoint = _same_bound_arg(args.original_checkpoint, v984, "original_checkpoint")
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v984, "v5_checkpoint")
    evidence_file = _same_bound_arg(args.v5_evidence_report, v984, "v5_evidence_report")
    source_index = Path(str(v984["paths"]["source_dataset_index"])).resolve()
    if dataset_root != index_file.parent or source_root != source_index.parent:
        raise ValueError("dataset root differs from sealed Teacher-v9.8.4")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "two_scene_step4_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("two-scene policy hash changed")

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {SOURCE_SCENE, AUDIT_SCENE}:
        raise ValueError("two train-scene metadata inventory changed")
    source_bundle = load_train_scene_bundle(dataset_root, source_root, records[SOURCE_SCENE])
    audit_bundle = load_train_scene_bundle(dataset_root, source_root, records[AUDIT_SCENE])
    if tuple(source_bundle["instance_names"]) != EXPECTED_INSTANCES[SOURCE_SCENE]:
        raise ValueError("source-scene target order changed")
    if tuple(audit_bundle["instance_names"]) != EXPECTED_INSTANCES[AUDIT_SCENE]:
        raise ValueError("audit-scene target order changed")

    mean, std = load_stats(stats_file)
    source_batch = _build_batch(source_bundle, mean, std, args.device)
    source_kwargs = _kwargs(source_batch)
    if tuple(sorted(source_kwargs)) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher forward input changed")
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    metric_policy = read_json(Path(str(v984["paths"]["metric_policy"])).resolve())
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
    audit_cache: Dict[int, list] = {}
    base_normalized = np.empty((3, 2, 8192, 6), np.float32)
    for generation in range(3):
        seed_pair = stable_rollout_seeds(generation)
        audit_cache[generation] = []
        for prompt_index, text in enumerate(prompt_texts):
            final, states, rng_states = _capture_trajectory(
                model,
                diffusion,
                audit_bundle,
                text,
                seed_pair[0],
                seed_pair[1],
                args.device,
                not args.no_progress,
            )
            base_normalized[generation, prompt_index] = final.numpy()
            audit_cache[generation].append(
                (states[SELECTED_TIMESTEP], rng_states[SELECTED_TIMESTEP])
            )
            print(
                f"[AUDIT-BASE] generation={generation} prompt={PROMPT_IDS[prompt_index]}",
                flush=True,
            )

    design_cache: Dict[int, list] = {}
    for step in range(2, SELECTED_V984_STEP + 1):
        seed_pair = calibration_design_seeds(step)
        design_cache[step] = []
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
                raise AssertionError("source-scene Base design resume is not exact")
            design_cache[step].append(states[SELECTED_TIMESTEP])
            print(
                f"[SOURCE-DESIGN] update={step} prompt={PROMPT_IDS[prompt_index]}",
                flush=True,
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
            task_order = list(TASK_ORDER[:6])
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
            task_order = list(TASK_ORDER)
            design_states = torch.cat(design_cache[step], dim=0).to(args.device)
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
        print(f"[RECONSTRUCTION_PASS] exact Teacher-v9.8.4 update={step}", flush=True)
        del tasks, design_prediction, design_frozen, direction
        if step > 1:
            del sit_tasks, live_v5_prediction, v5_tasks, design_physical, negative_tasks
        gc.collect()
        torch.cuda.empty_cache()

    if _state_sha256(named_lora) != v984["direction_rows"][3]["post_state_sha256"]:
        raise AssertionError("reconstructed step-4 state hash changed")
    model.eval()
    candidate_normalized = np.empty_like(base_normalized)
    for generation in range(3):
        for prompt_index, text in enumerate(prompt_texts):
            state, rng_state = audit_cache[generation][prompt_index]
            final = _resume_trajectory(
                model,
                diffusion,
                audit_bundle,
                text,
                state,
                rng_state,
                SELECTED_TIMESTEP,
                args.device,
            )
            candidate_normalized[generation, prompt_index] = final.numpy()
            print(
                f"[AUDIT-CANDIDATE] generation={generation} prompt={PROMPT_IDS[prompt_index]}",
                flush=True,
            )
    with torch.no_grad():
        candidate_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(
        candidate_v5_prediction.detach().cpu().numpy(),
        v984_arrays["candidate_v5_predictions"][SELECTED_V984_STEP - 1],
    ):
        raise AssertionError("reconstructed step-4 v5 prediction differs from v9.8.4")

    mean_k3 = mean.reshape(1, 1, 1, 6)
    std_k3 = std.reshape(1, 1, 1, 6)
    base_physical = np.clip(base_normalized * std_k3 + mean_k3, 0.0, 1.0).astype(np.float32)
    candidate_physical = np.clip(
        candidate_normalized * std_k3 + mean_k3, 0.0, 1.0
    ).astype(np.float32)
    base_rows = _rows(audit_bundle, base_physical)
    candidate_rows = _rows(audit_bundle, candidate_physical)
    instance_names = list(audit_bundle["instance_names"])
    base_pooled = pooled_by_role(base_rows, instance_names)
    candidate_pooled = pooled_by_role(candidate_rows, instance_names)
    verified_mask = np.asarray(audit_bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(base_physical, verified_mask)
    candidate_invariance = _invariance(candidate_physical, verified_mask)
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    candidate_v5_dense = (
        (candidate_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    checks = two_scene_preflight_checks(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        instance_names=instance_names,
        base_prompt_invariance=base_invariance,
        candidate_prompt_invariance=candidate_invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        maximum_map_delta=float(np.abs(candidate_physical - base_physical).max()),
    )
    presence = [
        [
            absolute_presence_checks(candidate_rows[g][p], instance_names)
            for p in range(2)
        ]
        for g in range(3)
    ]
    audit_eligible = all(checks.values())

    arrays_file = output_dir / "two_scene_step4_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(audit_bundle["xyz"], np.float32),
        points=np.asarray(audit_bundle["points"], np.float32),
        instance_ids=np.asarray(audit_bundle["instance_ids"], np.int64),
        category_ids=np.asarray(audit_bundle["category_ids"], np.int64),
        instance_names=np.asarray(instance_names),
        verified_object_mask=np.asarray(audit_bundle["verified_object_mask"], bool),
        verified_positive_mask=verified_mask,
        unknown_sittable_mask=np.asarray(audit_bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(audit_bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(audit_bundle["instance_targets"], np.float32),
        all_sittable_gt=np.asarray(audit_bundle["all_target"], np.float32),
        prompt_ids=np.asarray(PROMPT_IDS),
        audit_seed_table=np.asarray([stable_rollout_seeds(g) for g in range(3)], np.int64),
        base_normalized=base_normalized,
        candidate_normalized=candidate_normalized,
        base=base_physical,
        candidate=candidate_physical,
        v5_target=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_prediction=base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        candidate_v5_prediction=candidate_v5_prediction.detach().cpu().numpy().astype(np.float32),
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v985_two_scene_step4_preflight.py",
        "contract": PREPARE_ROOT / "relational_teacher_v985_two_scene_step4_preflight_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v985_two_scene_step4_preflight.py",
        "v984_summary": v984_file,
        "v984_maps": v984_maps_file,
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "metric_policy": Path(str(v984["paths"]["metric_policy"])).resolve(),
        "preflight_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "audit_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    top_checks = {
        "sealed_v984_pass_and_step4_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "room0101_step4_v5_exactly_reproduced": True,
        "room0102_actual_k3_two_prompt_audit_completed": True,
        "room0102_step4_response_is_admissible": audit_eligible,
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
        "preflight_tag": PREFLIGHT_TAG,
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
        "step_radius": STEP_RADIUS,
        "prompt_ids": list(PROMPT_IDS),
        "prompt_text": {name: PROMPTS[name] for name in PROMPT_IDS},
        "forward_input_keys": sorted(source_kwargs),
        "audit_seed_table": [list(stable_rollout_seeds(g)) for g in range(3)],
        "trajectory_accounting": {
            "room0102_base_full_draws": 6,
            "room0101_design_base_full_draws": 6,
            "room0101_design_exact_partial_repeats": 6,
            "room0102_candidate_partial_resumes": 6,
        },
        "v984_binding_id": v984["binding_id"],
        "source_scene_binding": _scene_binding(records[SOURCE_SCENE]),
        "audit_scene_binding": _scene_binding(records[AUDIT_SCENE]),
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["preflight_policy"],
        "lora": dict(lora_metadata(model)),
        "zero_state_sha256": zero_state_sha256,
        "selected_state_sha256": reconstruction_rows[-1]["post_state_sha256"],
        "reconstruction_rows": reconstruction_rows,
        "instance_names": instance_names,
        "base_rows": base_rows,
        "candidate_rows": candidate_rows,
        "base_pooled": base_pooled,
        "candidate_pooled": candidate_pooled,
        "base_prompt_invariance": base_invariance,
        "candidate_prompt_invariance": candidate_invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "maximum_map_delta": float(np.abs(candidate_physical - base_physical).max()),
        "presence": presence,
        "checks": top_checks,
        "audit_checks": checks,
        "failed_checks": sorted(name for name, passed in top_checks.items() if not passed),
        "audit_failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "audit_maps_sha256": path_hashes["audit_maps"],
        "serialized_model_state": False,
        "authorizes_two_scene_preservation_response_preflight": status == "PASS",
        "authorizes_cross_scene_direction_diagnosis": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "v984_binding_id": report["v984_binding_id"],
            "policy_id": report["policy_id"],
            "selected_state_sha256": report["selected_state_sha256"],
            "audit_maps_sha256": report["audit_maps_sha256"],
            "status": report["status"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.5 report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.5 unexpectedly saved model state")
    print(f"[TWO_SCENE_STEP4_{status}] Teacher-v9.8.5 room_0102 preflight")
    print("[PASS] fresh v5r4 and exact v9.8.4 step-4 reconstruction verified")
    print("[PASS] actual room_0102 K=3 two-prompt maps completed")
    print("[OK] pooled:", candidate_pooled)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] audit failed checks:", report["audit_failed_checks"])
    print("[OK] report:", report_file)
    del model, diffusion
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

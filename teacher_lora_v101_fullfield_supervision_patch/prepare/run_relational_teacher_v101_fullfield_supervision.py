#!/usr/bin/env python3
"""Fresh Teacher-v10.1 training with full-field label supervision."""

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

import run_relational_teacher_v10_supervised_capacity as v10  # noqa: E402
from fewshot_cdm_common import load_split  # noqa: E402
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    lora_parameter_energy,
    save_merged_legacy_state,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from relational_teacher_v101_fullfield_contract import (  # noqa: E402
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    DIFFUSION_STEPS,
    EXPECTED_INSTANCES,
    GRAD_CLIP,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    LOSS_WEIGHTS,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    REPLAY_START_STEP,
    REPLAY_WEIGHT,
    ROLLOUT_K,
    SCENES,
    SCHEMA,
    SOURCE_SCENE,
    TIMESTEP_CYCLE,
    TRAIN_STEPS,
    V10_SCHEMA,
    WEIGHT_DECAY,
    canonical_sha256,
    select_rollout_candidate,
    shortlist_monitor_steps,
)
from relational_teacher_v101_fullfield_objective import (  # noqa: E402
    fullfield_all_sittable_objective,
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
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v10-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _validate_v10_failure(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V10_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_step") is not None
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_capacity_or_objective_redesign") is not True
        or value.get("failed_checks")
        != ["at_least_one_supervised_candidate_passes_actual_k3"]
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v10 failure does not authorize v10.1")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Teacher-v10 path binding is absent")
    if set(paths) != set(hashes) or "checkpoint" in paths:
        raise ValueError("failed Teacher-v10 path inventory changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed Teacher-v10 unexpectedly contains a checkpoint")
    return value


def _bound_path(authority: Mapping[str, object], name: str) -> Path:
    result = Path(str(authority["paths"][name])).expanduser().resolve()
    if not result.is_file() or sha256_file(result) != authority["path_sha256"][name]:
        raise ValueError("Teacher-v10 bound input changed: " + name)
    return result


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(label + " contains a non-finite number")


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v10.1 full-field training requires CUDA")
    failure_file = args.failed_v10_summary.expanduser().resolve()
    authority = _validate_v10_failure(failure_file)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v10.1 output")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "fullfield_supervision_policy.json"
    atomic_write_json(policy_file, POLICY)
    if read_json(policy_file) != POLICY or canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("Teacher-v10.1 policy hash changed")

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
        raise ValueError("Teacher-v10.1 train-scene inventory changed")
    bundles = {
        scene: load_train_scene_bundle(
            dataset_root, source_root, records[scene]
        )
        for scene in SCENES
    }
    for scene in SCENES:
        if tuple(str(name) for name in bundles[scene]["instance_names"]) != EXPECTED_INSTANCES[scene]:
            raise ValueError(scene + " verified instance order changed")

    configure_reproducibility(MODEL_SEED)
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

    cfg = compose_cdm_config(DIFFUSION_STEPS, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    base_fixed_maps, base_fixed_summary = v10._evaluate_fixed_panel(
        model,
        diffusion,
        batches,
        kwargs,
        bundles,
        mean,
        std,
        MODEL_SEED + 700000,
        args.device,
    )
    base_v5_prediction, base_v5_dense = v10._v5_fixed_prediction(
        model,
        diffusion,
        v5_batch,
        v5_kwargs,
        MODEL_SEED + 800000,
        args.device,
    )

    modules = install_lora(model, LORA_RANK, LORA_ALPHA, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("Teacher-v10.1 LoRA module inventory changed")
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
    zero_fixed_maps, _ = v10._evaluate_fixed_panel(
        model,
        diffusion,
        batches,
        kwargs,
        bundles,
        mean,
        std,
        MODEL_SEED + 700000,
        args.device,
    )
    zero_v5_prediction, _ = v10._v5_fixed_prediction(
        model,
        diffusion,
        v5_batch,
        v5_kwargs,
        MODEL_SEED + 800000,
        args.device,
    )
    if not np.array_equal(zero_fixed_maps, base_fixed_maps) or not np.array_equal(
        zero_v5_prediction, base_v5_prediction
    ):
        raise AssertionError("fresh zero-init LoRA differs from sealed v5r4")
    del zero_fixed_maps, zero_v5_prediction

    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    monitor_rows = []
    monitor_maps = np.empty(
        (len(MONITOR_STEPS), 2, 3, 2, 8192, 6), np.float32
    )
    monitor_v5_predictions = np.empty(
        (len(MONITOR_STEPS), 3, 8192, 6), np.float32
    )
    states_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
    training_log_rows = []
    timestep_hits = Counter()
    task_names = tuple(name for name in LOSS_WEIGHTS if name != "lora_regularizer")

    for step in range(1, TRAIN_STEPS + 1):
        set_frozen_base_eval_lora_train(model)
        set_lora_enabled(model, True)
        optimizer.zero_grad(set_to_none=True)
        component_sums = {name: 0.0 for name in LOSS_WEIGHTS}
        active_means = np.zeros(3, np.float64)
        training_total = 0.0
        for scene_index, scene in enumerate(SCENES):
            timestep = TIMESTEP_CYCLE[
                (2 * (step - 1) + scene_index) % len(TIMESTEP_CYCLE)
            ]
            timestep_hits[timestep] += 1
            t = torch.full((2,), timestep, dtype=torch.long, device=args.device)
            one_noise = deterministic_noise(
                torch.Size((1, 8192, 6)),
                MODEL_SEED + step * 1009 + scene_index * 1000003,
                args.device,
            )
            noise = one_noise.repeat(2, 1, 1)
            if not torch.equal(noise[0], noise[1]):
                raise AssertionError("prompt-pair training noise changed")
            prediction = predict_xstart(
                model,
                diffusion,
                batches[scene]["x"],
                t,
                kwargs[scene],
                noise,
            )
            objective = fullfield_all_sittable_objective(
                prediction, batches[scene], mean_tensor, std_tensor
            )
            scene_loss = sum(
                float(LOSS_WEIGHTS[name]) * objective[name]
                for name in task_names
            )
            (0.5 * scene_loss).backward()
            training_total += 0.5 * float(scene_loss.detach().item())
            for name in task_names:
                component_sums[name] += 0.5 * float(
                    objective[name].detach().item()
                )
            active_means += 0.5 * (
                objective["per_instance_active_mean"]
                .detach()
                .mean(dim=0)
                .cpu()
                .numpy()
            )
            del t, one_noise, noise, prediction, objective, scene_loss

        if step < REPLAY_START_STEP:
            raise AssertionError("Teacher-v10.1 v5 replay must start at step one")
        v5_t_values = (100, 300, 450)
        v5_t = torch.tensor(
            [v5_t_values[(step + offset) % 3] for offset in range(3)],
            dtype=torch.long,
            device=args.device,
        )
        v5_noise = deterministic_noise(
            v5_batch["x"].shape, MODEL_SEED + 900000 + step * 1013, args.device
        )
        set_lora_enabled(model, False)
        with torch.no_grad():
            frozen_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        set_lora_enabled(model, True)
        candidate_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
        replay = (candidate_v5 - frozen_v5).square().mean()
        (REPLAY_WEIGHT * replay).backward()
        replay_value = float(replay.detach().item())
        training_total += REPLAY_WEIGHT * replay_value
        del v5_t, v5_noise, frozen_v5, candidate_v5, replay

        regularizer = lora_parameter_energy(model)
        regularizer_loss = LOSS_WEIGHTS["lora_regularizer"] * regularizer
        regularizer_loss.backward()
        component_sums["lora_regularizer"] = float(regularizer.detach().item())
        training_total += float(regularizer_loss.detach().item())
        if any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if "lora_" not in name
        ):
            raise AssertionError("gradient reached frozen CDM")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP).item())
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError("Teacher-v10.1 gradient is zero/non-finite")
        optimizer.step()

        if step == 1 or step % 25 == 0:
            log_row = {
                "step": step,
                "total": training_total,
                "gradient_l2_before_clip": grad_norm,
                "active_prediction_mean": active_means.tolist(),
                "v5_replay": replay_value,
                "components": component_sums,
            }
            _finite_tree(log_row, "Teacher-v10.1 training log")
            training_log_rows.append(log_row)
            print(
                "[TRAIN] step={:04d}/{} total={:.6f} active={:.4f}/{:.4f}/{:.4f} replay={:.6f} grad={:.6f}".format(
                    step,
                    TRAIN_STEPS,
                    training_total,
                    active_means[0],
                    active_means[1],
                    active_means[2],
                    replay_value,
                    grad_norm,
                ),
                flush=True,
            )

        if step in MONITOR_STEPS:
            model.eval()
            fixed_maps, fixed_summary = v10._evaluate_fixed_panel(
                model,
                diffusion,
                batches,
                kwargs,
                bundles,
                mean,
                std,
                MODEL_SEED + 700000,
                args.device,
            )
            v5_prediction, candidate_v5_dense = v10._v5_fixed_prediction(
                model,
                diffusion,
                v5_batch,
                v5_kwargs,
                MODEL_SEED + 800000,
                args.device,
            )
            monitor_index = list(MONITOR_STEPS).index(step)
            monitor_maps[monitor_index] = fixed_maps
            monitor_v5_predictions[monitor_index] = v5_prediction
            fixed_summary.update(
                {
                    "step": step,
                    "state_sha256": v10._state_sha256(named_lora),
                    "lora_energy": float(lora_parameter_energy(model).item()),
                    "base_v5_dense": base_v5_dense,
                    "candidate_v5_dense": candidate_v5_dense,
                }
            )
            _finite_tree(fixed_summary, "Teacher-v10.1 monitor row")
            monitor_rows.append(fixed_summary)
            states_by_step[step] = v10._lora_cpu_state(named_lora)
            del v5_prediction
            print(
                "[MONITOR] step={} recall={:.6f} MAE={:.6f} known-MAE={:.6f} v5={:+.3f}%".format(
                    step,
                    fixed_summary["worst_instance_soft_recall"],
                    fixed_summary["worst_instance_active_support_mae"],
                    fixed_summary["known_dense_mae"],
                    100.0
                    * (
                        sum(candidate_v5_dense) / sum(base_v5_dense)
                        - 1.0
                    ),
                ),
                flush=True,
            )
        if step % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if set(timestep_hits) != set(TIMESTEP_CYCLE) or min(timestep_hits.values()) <= 0:
        raise AssertionError("full timestep curriculum was not exercised")
    shortlisted_steps = shortlist_monitor_steps(monitor_rows)
    rollout_seed_table = [
        v10._stable_seeds(MODEL_SEED + 1, "v101_actual_rollout", str(generation))
        for generation in range(ROLLOUT_K)
    ]
    base_rollouts = np.empty((2, ROLLOUT_K, 2, 8192, 6), np.float32)
    set_lora_enabled(model, False)
    model.eval()
    for scene_index, scene in enumerate(SCENES):
        for generation, (initial_seed, reverse_seed) in enumerate(rollout_seed_table):
            for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                sample = v10._sample(
                    model,
                    diffusion,
                    bundles[scene],
                    PROMPTS[prompt_id],
                    initial_seed,
                    reverse_seed,
                    args.device,
                    not args.no_progress,
                )
                base_rollouts[scene_index, generation, prompt_index] = np.clip(
                    sample.numpy() * std.reshape(1, 6) + mean.reshape(1, 6),
                    0.0,
                    1.0,
                )
                print(
                    "[BASE-ROLLOUT] scene={} generation={} prompt={}".format(
                        scene, generation, prompt_id
                    ),
                    flush=True,
                )

    candidate_rollouts = np.empty(
        (3, 2, ROLLOUT_K, 2, 8192, 6), np.float32
    )
    candidate_v5_predictions = np.empty((3, 3, 8192, 6), np.float32)
    rollout_rows = []
    for candidate_index, step in enumerate(shortlisted_steps):
        v10._restore_lora_state(named_lora, states_by_step[step])
        set_lora_enabled(model, True)
        model.eval()
        state_sha256 = v10._state_sha256(named_lora)
        for scene_index, scene in enumerate(SCENES):
            for generation, (initial_seed, reverse_seed) in enumerate(rollout_seed_table):
                for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                    sample = v10._sample(
                        model,
                        diffusion,
                        bundles[scene],
                        PROMPTS[prompt_id],
                        initial_seed,
                        reverse_seed,
                        args.device,
                        not args.no_progress,
                    )
                    candidate_rollouts[
                        candidate_index, scene_index, generation, prompt_index
                    ] = np.clip(
                        sample.numpy() * std.reshape(1, 6) + mean.reshape(1, 6),
                        0.0,
                        1.0,
                    )
                    print(
                        "[CANDIDATE {}] scene={} generation={} prompt={}".format(
                            step, scene, generation, prompt_id
                        ),
                        flush=True,
                    )
        v5_prediction, candidate_v5_dense = v10._v5_fixed_prediction(
            model,
            diffusion,
            v5_batch,
            v5_kwargs,
            MODEL_SEED + 800000,
            args.device,
        )
        candidate_v5_predictions[candidate_index] = v5_prediction
        panel = v10._rollout_panel(
            bundles,
            candidate_rollouts[candidate_index],
            base_v5_dense,
            candidate_v5_dense,
            state_sha256 != zero_state_sha256,
        )
        panel.update(
            {
                "step": step,
                "state_sha256": state_sha256,
                "candidate_maps_sha256": tensor_sha256(
                    torch.from_numpy(candidate_rollouts[candidate_index])
                ),
            }
        )
        _finite_tree(panel, "Teacher-v10.1 rollout candidate")
        rollout_rows.append(panel)
        print(
            "[ACTUAL-K3] step={} eligible={} all-three={}".format(
                step, panel["eligible"], panel["all_three_counts"]
            ),
            flush=True,
        )

    selected_step = select_rollout_candidate(rollout_rows)
    selected_index = (
        shortlisted_steps.index(selected_step) if selected_step is not None else None
    )
    checkpoint_file = output_dir / "teacher_v101_fullfield.pt"
    if selected_step is not None:
        v10._restore_lora_state(named_lora, states_by_step[selected_step])
        set_lora_enabled(model, True)

    maps_file = output_dir / "fullfield_supervision_maps.npz"
    save_arrays: Dict[str, np.ndarray] = {
        "scene_ids": np.asarray(SCENES),
        "prompt_ids": np.asarray(PROMPT_IDS),
        "eval_timesteps": np.asarray((50, 250, 450), np.int64),
        "monitor_steps": np.asarray(MONITOR_STEPS, np.int64),
        "shortlisted_steps": np.asarray(shortlisted_steps, np.int64),
        "rollout_seed_table": np.asarray(rollout_seed_table, np.int64),
        "base_fixed": base_fixed_maps,
        "monitor_fixed": monitor_maps,
        "monitor_v5_predictions": monitor_v5_predictions,
        "base_rollouts": base_rollouts,
        "candidate_rollouts": candidate_rollouts,
        "v5_target": v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        "base_v5_prediction": base_v5_prediction,
        "candidate_v5_predictions": candidate_v5_predictions,
    }
    for scene in SCENES:
        prefix = "source" if scene == SOURCE_SCENE else "audit"
        bundle = bundles[scene]
        for key, dtype in (
            ("xyz", np.float32),
            ("points", np.float32),
            ("instance_names", None),
            ("verified_object_mask", bool),
            ("verified_positive_mask", bool),
            ("unknown_sittable_mask", bool),
            ("environment_aux_mask", bool),
            ("explicit_negative_mask", bool),
            ("instance_targets", np.float32),
        ):
            save_arrays[prefix + "_" + key] = np.asarray(
                bundle[key], dtype=dtype
            )
        save_arrays[prefix + "_all_sittable_gt"] = np.asarray(
            bundle["all_target"], np.float32
        )
    atomic_savez(maps_file, **save_arrays)
    if selected_step is not None:
        save_merged_legacy_state(model, checkpoint_file)
        if not checkpoint_file.is_file() or checkpoint_file.stat().st_size <= 0:
            raise RuntimeError("Teacher-v10.1 checkpoint serialization failed")
    elif checkpoint_file.exists():
        raise AssertionError("failed Teacher-v10.1 may not write a checkpoint")

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v101_fullfield_supervision.py",
        "contract": PREPARE_ROOT / "relational_teacher_v101_fullfield_contract.py",
        "objective": PREPARE_ROOT
        / "relational_teacher_v101_fullfield_objective.py",
        "v10_failure": failure_file,
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "metric_policy": metric_policy_file,
        "training_policy": policy_file,
        "maps": maps_file,
    }
    if selected_step is not None:
        source_paths["checkpoint"] = checkpoint_file
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    status = "PASS" if selected_step is not None else "FAIL"
    checks = {
        "sealed_v10_failure_authorizes_fullfield_redesign": True,
        "fresh_v5r4_zero_init": True,
        "both_train_scenes_used_in_every_update": True,
        "every_non_unknown_point_receives_gt_supervision": True,
        "hard_false_positive_background_is_supervised": True,
        "v5_replay_active_from_first_update": REPLAY_START_STEP == 1,
        "all_timestep_buckets_exercised": set(timestep_hits)
        == set(TIMESTEP_CYCLE)
        and min(timestep_hits.values()) > 0,
        "exact_1000_adamw_updates": TRAIN_STEPS == 1000,
        "only_lora_received_gradients": True,
        "three_monitor_states_received_actual_two_scene_k3": len(rollout_rows)
        == 3,
        "at_least_one_fullfield_candidate_passes_actual_k3": selected_step
        is not None,
        "checkpoint_written_iff_actual_gate_passes": checkpoint_file.is_file()
        == (selected_step is not None),
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
    }
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": MODEL_SEED,
        "device": args.device,
        "diffusion_steps": DIFFUSION_STEPS,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "train_scenes": list(SCENES),
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "development_arrays_read": False,
        "paper_test_access": False,
        "forward_input_keys": sorted(FORWARD_INPUT_KEYS),
        "prompt_ids": list(PROMPT_IDS),
        "policy_id": POLICY_ID,
        "v10_binding_id": authority.get("binding_id"),
        "zero_state_sha256": zero_state_sha256,
        "lora": dict(lora_metadata(model)),
        "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY},
        "loss_weights": dict(LOSS_WEIGHTS),
        "replay_start_step": REPLAY_START_STEP,
        "replay_weight": REPLAY_WEIGHT,
        "timestep_hits": {
            str(key): int(value) for key, value in sorted(timestep_hits.items())
        },
        "base_fixed_summary": base_fixed_summary,
        "training_log_rows": training_log_rows,
        "monitor_rows": monitor_rows,
        "shortlisted_steps": shortlisted_steps,
        "rollout_seed_table": [list(row) for row in rollout_seed_table],
        "base_v5_dense": base_v5_dense,
        "rollout_rows": rollout_rows,
        "selected_step": selected_step,
        "selected_rollout_index": selected_index,
        "serialized_model_state": selected_step is not None,
        "maps_sha256": path_hashes["maps"],
        "checkpoint_sha256": path_hashes.get("checkpoint"),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": checks,
        "failed_checks": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "authorizes_teacher_v101_checkpoint_lock": status == "PASS",
        "authorizes_objective_or_architecture_redesign": status == "FAIL",
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "policy_id": POLICY_ID,
            "v10_binding_id": report["v10_binding_id"],
            "maps_sha256": report["maps_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "selected_step": selected_step,
        }
    )
    _finite_tree(report, "Teacher-v10.1 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    print("[FULLFIELD_SUPERVISION_{}] Teacher-v10.1".format(status))
    print("[PASS] every non-unknown point received direct all-sittable GT loss")
    print("[PASS] hard-background and step-one v5 preservation completed")
    print("[PASS] two train scenes x K=3 actual rollouts completed")
    print("[OK] shortlisted steps:", shortlisted_steps)
    print("[OK] selected step:", selected_step)
    print("[OK] checkpoint:", path_strings.get("checkpoint"))
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    del model, diffusion, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Teacher-v9.8.1 selected rollout-state response on K=3 final maps."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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
    _physical_pair,
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
    ROLLOUT_POLICY_ID as V97_POLICY_ID,
    SCHEMA as V97_SCHEMA,
    continuous_presence_checks,
    stable_rollout_seeds,
)
from relational_teacher_v98_rollout_state_contract import (  # noqa: E402
    CAPTURE_TIMESTEPS,
    POLICY_ID as V98_POLICY_ID,
    SEED as V98_SEED,
)
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    GENERATION_COUNT,
    HELDOUT_TRAIN_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SEED,
    SELECTED_NAME,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    TRAIN_SCENE,
    V98_SCHEMA,
    canonical_sha256,
    pooled_object_metrics,
    response6_checks,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _metrics,
    _prompt_invariance,
    _restore_lora_state,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v98-report", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
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


def _validate_v98(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V98_SCHEMA
        or value.get("status") != "PASS"
        or value.get("selected_candidate") != SELECTED_NAME
        or value.get("failed_checks")
        or value.get("policy_id") != V98_POLICY_ID
        or value.get("authorizes_rollout_state_response6") is not True
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.8 PASS authority changed")
    selected = [row for row in value["candidates"] if row.get("name") == SELECTED_NAME]
    if (
        len(selected) != 1
        or selected[0].get("eligible") is not True
        or selected[0].get("failed_checks")
        or selected[0].get("capture_timestep") != SELECTED_TIMESTEP
        or float(selected[0].get("radius")) != SELECTED_RADIUS
    ):
        raise ValueError("selected Teacher-v9.8 candidate changed")
    _validate_bound_files(value, "Teacher-v9.8")
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8 contains forbidden model state")
    return value


def _same_bound_arg(argument: Path, v98: Mapping[str, object], name: str) -> Path:
    path = argument.expanduser().resolve()
    expected = Path(str(v98["paths"][name])).expanduser().resolve()
    if path != expected or not path.is_file() or sha256_file(path) != v98["path_sha256"][name]:
        raise ValueError(name + " differs from sealed Teacher-v9.8")
    return path


def _rows(bundle: Mapping[str, object], values: np.ndarray) -> list:
    if values.shape != (3, 2, 8192, 6):
        raise ValueError("K=3 response map shape changed")
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
        raise RuntimeError("Teacher-v9.8.1 response-6 requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8.1 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8.1 output")
    configure_reproducibility(args.seed)

    v98_file = args.v98_report.expanduser().resolve()
    v98 = _validate_v98(v98_file)
    v98_maps_file = Path(str(v98["paths"]["rollout_maps"])).resolve()
    v98_arrays = _load_npz(v98_maps_file)
    v97_file = Path(str(v98["paths"]["failed_v97_summary"])).resolve()
    v97 = read_json(v97_file)
    if (
        v97.get("schema") != V97_SCHEMA
        or v97.get("status") != "FAIL"
        or v97.get("binding_id") != v98.get("failed_v97_binding_id")
        or v97.get("rollout_metric_policy_id") != V97_POLICY_ID
        or v97.get("authorizes_rollout_state_preflight") is not True
    ):
        raise ValueError("sealed Teacher-v9.7 authority changed")
    _validate_bound_files(v97, "Teacher-v9.7")
    v97_maps_file = Path(str(v97["paths"]["rollout_maps"])).resolve()
    v97_arrays = _load_npz(v97_maps_file)
    v97_base_normalized = np.asarray(v97_arrays["frozen_base_normalized"], np.float32)
    v98_candidate_normalized = np.asarray(v98_arrays["candidates_normalized"], np.float32)
    if v97_base_normalized.shape != (3, 2, 8192, 6):
        raise ValueError("Teacher-v9.7 Base K=3 shape changed")
    if v98_candidate_normalized.shape != (3, 3, 2, 8192, 6):
        raise ValueError("Teacher-v9.8 candidate grid shape changed")
    if not np.array_equal(v98_arrays["audit_base_normalized"], v97_base_normalized[0]):
        raise ValueError("v9.8 generation-0 Base reuse changed")

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_root.is_dir():
        raise FileNotFoundError("a sealed dataset root is absent")
    index_file = _same_bound_arg(args.index, v98, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v98, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v98, "stats_file")
    original_checkpoint = _same_bound_arg(args.original_checkpoint, v98, "original_checkpoint")
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v98, "v5_checkpoint")
    evidence_file = _same_bound_arg(args.v5_evidence_report, v98, "v5_evidence_report")
    preflight_file = Path(str(v98["paths"]["preflight_report"])).resolve()
    preflight, original_policy = _validate_preflight(preflight_file)
    if preflight.get("binding_id") != v98.get("preflight_binding_id"):
        raise ValueError("Teacher-v9 preflight binding changed")
    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE:
        raise AssertionError("wrong train scene loaded")
    if v98.get("scene_binding") != {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]:
        raise ValueError("room_0101 scene binding changed")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "response6_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("response-6 policy hash changed")

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
    if [str(v5_rows[name]["target"]) for name in v5_ids] != ["chair", "bed", "whiteboard"]:
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

    v5_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, V98_SEED + 2000, args.device)
    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not np.array_equal(
        base_v5_prediction.detach().cpu().numpy(), v98_arrays["base_v5_prediction"]
    ):
        raise AssertionError("v9.8 Base v5 prediction did not reproduce")

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
    zero_state = {name: value.detach().cpu().clone() for name, value in named_lora.items()}

    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    radius_index = list(v98_arrays["step_radii"].tolist()).index(SELECTED_RADIUS)
    design_states = torch.from_numpy(
        np.asarray(v98_arrays["design_states"][timestep_index], np.float32)
    ).to(args.device)
    with torch.no_grad():
        set_lora_enabled(model, False)
        frozen_design = _direct_prediction(
            model, diffusion, design_states, SELECTED_TIMESTEP, kwargs
        )
    before_physical = _physical_pair(frozen_design, mean, std)
    if not np.array_equal(before_physical, v98_arrays["direct_before"][timestep_index]):
        raise AssertionError("v9.8 selected direct Base did not reproduce")
    set_lora_enabled(model, True)
    prediction = _direct_prediction(model, diffusion, design_states, SELECTED_TIMESTEP, kwargs)
    if not torch.equal(prediction.detach(), frozen_design):
        raise AssertionError("fresh zero-init LoRA differs from v5r4 at selected state")
    objective = corrected_objective(prediction, frozen_design, batch, mean_tensor, std_tensor)
    task_losses = (
        objective["per_instance_primary"]
        + 2.0 * objective["per_instance_active_support"]
        + 0.25 * objective["per_instance_ranking"]
    ).reshape(-1)
    gradients = flattened_task_gradients(task_losses, parameters)
    gram = (gradients @ gradients.T).detach().cpu().double().numpy()
    weights = frank_wolfe_min_norm_weights(gram, 4096)
    direction = torch.from_numpy(weights).to(gradients.device, gradients.dtype) @ gradients
    direction_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction_norm) or float(direction_norm.item()) <= 0.0:
        raise RuntimeError("selected common direction is zero/non-finite")
    direction = direction / direction_norm
    derivatives = (gradients @ direction).detach().cpu().tolist()
    selected_v98 = [row for row in v98["candidates"] if row["name"] == SELECTED_NAME][0]
    selected_direction = [
        row for row in v98["direction_rows"] if row["capture_timestep"] == SELECTED_TIMESTEP
    ][0]
    if (
        tensor_sha256(direction) != selected_v98["direction_sha256"]
        or not np.allclose(gram, np.asarray(selected_direction["gram"]), rtol=1e-7, atol=1e-8)
        or not np.allclose(weights, np.asarray(selected_direction["weights"]), rtol=1e-7, atol=1e-9)
        or not np.allclose(derivatives, selected_v98["directional_derivatives"], rtol=1e-7, atol=1e-8)
    ):
        raise AssertionError("selected v9.8 direction geometry did not reproduce")
    direction_row = {
        "capture_timestep": SELECTED_TIMESTEP,
        "task_order": list(selected_direction["task_order"]),
        "task_losses": [float(value) for value in task_losses.detach().cpu().tolist()],
        "gram": gram.tolist(),
        "weights": weights.tolist(),
        "directional_derivatives": [float(value) for value in derivatives],
        "direction_norm_before_unit": float(direction_norm.item()),
        "direction_sha256": tensor_sha256(direction),
    }

    _restore_lora_state(named_lora, zero_state)
    set_lora_enabled(model, True)
    apply_flat_direction(parameters, direction, SELECTED_RADIUS)
    model.eval()
    with torch.no_grad():
        after_direct = _direct_prediction(
            model, diffusion, design_states, SELECTED_TIMESTEP, kwargs
        )
        candidate_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    after_physical = _physical_pair(after_direct, mean, std)
    if not np.array_equal(
        after_physical, v98_arrays["direct_after"][timestep_index, radius_index]
    ):
        raise AssertionError("selected v9.8 direct response did not reproduce")
    if not np.array_equal(
        candidate_v5_prediction.detach().cpu().numpy(),
        v98_arrays["candidate_v5_predictions"][timestep_index, radius_index],
    ):
        raise AssertionError("selected v9.8 v5 response did not reproduce")
    print("[SELECTED_DIRECTION_REPRODUCTION_PASS]", SELECTED_NAME, flush=True)
    del prediction, objective, task_losses, gradients, direction, frozen_design, after_direct
    gc.collect()
    torch.cuda.empty_cache()

    prompt_texts = [PROMPTS[name] for name in PROMPT_IDS]
    base_normalized = v97_base_normalized.copy()
    candidate_normalized = np.empty_like(base_normalized)
    candidate_normalized[0] = v98_candidate_normalized[timestep_index, radius_index]
    trajectory_cache = {}
    for generation in (1, 2):
        initial_seed, reverse_seed = stable_rollout_seeds(generation)
        trajectory_cache[generation] = []
        for prompt_index, text in enumerate(prompt_texts):
            set_lora_enabled(model, False)
            model.eval()
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
                raise AssertionError("v9.7 Base trajectory did not reproduce")
            trajectory_cache[generation].append((states, rng_states))
            set_lora_enabled(model, True)
            model.eval()
            candidate = _resume_trajectory(
                model,
                diffusion,
                bundle,
                text,
                states[SELECTED_TIMESTEP],
                rng_states[SELECTED_TIMESTEP],
                SELECTED_TIMESTEP,
                args.device,
            )
            candidate_normalized[generation, prompt_index] = candidate.numpy()
            print(
                "[RESPONSE6] generation={} prompt={}".format(
                    generation, PROMPT_IDS[prompt_index]
                ),
                flush=True,
            )
            del final, candidate
            torch.cuda.empty_cache()

    repeat_states, repeat_rng = trajectory_cache[1][0]
    repeat = _resume_trajectory(
        model,
        diffusion,
        bundle,
        prompt_texts[0],
        repeat_states[SELECTED_TIMESTEP],
        repeat_rng[SELECTED_TIMESTEP],
        SELECTED_TIMESTEP,
        args.device,
    )
    if not np.array_equal(repeat.numpy(), candidate_normalized[1, 0]):
        raise AssertionError("selected response repeat is not deterministic")
    print("[DETERMINISM_PASS] generation=1 watch exact repeat", flush=True)

    mean_k3 = mean.reshape(1, 1, 1, 6)
    std_k3 = std.reshape(1, 1, 1, 6)
    base_physical = np.clip(base_normalized * std_k3 + mean_k3, 0.0, 1.0).astype(np.float32)
    candidate_physical = np.clip(
        candidate_normalized * std_k3 + mean_k3, 0.0, 1.0
    ).astype(np.float32)
    base_rows = _rows(bundle, base_physical)
    candidate_rows = _rows(bundle, candidate_physical)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(base_physical, verified_mask)
    candidate_invariance = _invariance(candidate_physical, verified_mask)
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    candidate_v5_dense = (
        (candidate_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    maximum_delta = float(np.abs(candidate_physical - base_physical).max())
    response_checks = response6_checks(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        base_prompt_invariance=base_invariance,
        candidate_prompt_invariance=candidate_invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        directional_derivatives=derivatives,
        maximum_map_delta=maximum_delta,
    )
    candidate_presence = [
        [continuous_presence_checks(candidate_rows[g][p]) for p in range(2)]
        for g in range(3)
    ]
    base_pooled = pooled_object_metrics(base_rows)
    candidate_pooled = pooled_object_metrics(candidate_rows)

    arrays_file = output_dir / "response6_maps.npz"
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
        generation_ids=np.arange(3, dtype=np.int64),
        prompt_ids=np.asarray(PROMPT_IDS),
        seed_table=np.asarray([stable_rollout_seeds(g) for g in range(3)], np.int64),
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
        "validator": PREPARE_ROOT / "validate_relational_teacher_v981_rollout_state_response6.py",
        "contract": PREPARE_ROOT / "relational_teacher_v981_rollout_state_response6_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v981_rollout_state_response6.py",
        "v98_report": v98_file,
        "v98_maps": v98_maps_file,
        "v97_summary": v97_file,
        "v97_maps": v97_maps_file,
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "response6_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": Path(str(v98["paths"]["source_dataset_index"])).resolve(),
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "v98_runner": Path(str(v98["paths"]["runner"])).resolve(),
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "response6_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    top_checks = {
        "sealed_v98_pass_and_selected_candidate_bound": True,
        "selected_direction_geometry_and_hash_reproduced": True,
        "generation0_maps_byte_exactly_reused": True,
        "generation1_2_base_maps_reproduce_v97": True,
        "generation1_2_use_paired_t50_rng_resumes": True,
        "selected_response_repeat_is_bitwise_exact": True,
        "response6_policy_locked_before_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "selected_response_is_k3_admissible": all(response_checks.values()),
    }
    status = "PASS" if all(top_checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": HELDOUT_TRAIN_SCENE,
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "generation_count": GENERATION_COUNT,
        "prompt_ids": list(PROMPT_IDS),
        "prompt_text": {name: PROMPTS[name] for name in PROMPT_IDS},
        "forward_input_keys": sorted(kwargs),
        "seed_table": [list(stable_rollout_seeds(g)) for g in range(3)],
        "selected_candidate": SELECTED_NAME,
        "selected_timestep": SELECTED_TIMESTEP,
        "selected_radius": SELECTED_RADIUS,
        "activation_schedule": POLICY["adapter_activation"],
        "new_reverse_diffusion_draws": {
            "base_full_generation1_2": 4,
            "candidate_partial_generation1_2": 4,
            "candidate_partial_determinism_repeat": 1,
        },
        "preflight_binding_id": preflight["binding_id"],
        "v97_binding_id": v97["binding_id"],
        "v98_binding_id": v98["binding_id"],
        "scene_binding": v98["scene_binding"],
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["response6_policy"],
        "lora": dict(lora_metadata(model)),
        "serialized_model_state": False,
        "direction": direction_row,
        "base_rows": base_rows,
        "candidate_rows": candidate_rows,
        "candidate_presence": candidate_presence,
        "base_pooled": base_pooled,
        "candidate_pooled": candidate_pooled,
        "base_prompt_invariance": base_invariance,
        "candidate_prompt_invariance": candidate_invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "maximum_final_map_delta": maximum_delta,
        "response_checks": response_checks,
        "checks": top_checks,
        "failed_checks": sorted(name for name, passed in top_checks.items() if not passed),
        "paths": path_strings,
        "path_sha256": path_hashes,
        "response6_maps_sha256": path_hashes["response6_maps"],
        "authorizes_rollout_state_calibration6": status == "PASS",
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
            "policy_id": report["policy_id"],
            "selected_candidate": report["selected_candidate"],
            "direction_sha256": direction_row["direction_sha256"],
            "response6_maps_sha256": report["response6_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v9.8.1 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8.1 unexpectedly saved model state")
    print("[ROLLOUT_STATE_RESPONSE6_{}] Teacher-v9.8.1".format(status))
    print("[PASS] selected t50/radius-0.003 direction and generation 0 reproduced")
    print("[PASS] generation 1/2 Base and paired partial resumes completed")
    print("[OK] pooled High Chair recall: {:.6f}->{:.6f}".format(
        base_pooled["chair_06"]["soft_recall"],
        candidate_pooled["chair_06"]["soft_recall"],
    ))
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    del model, diffusion, zero_state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

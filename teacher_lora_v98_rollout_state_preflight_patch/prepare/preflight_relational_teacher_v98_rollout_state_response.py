#!/usr/bin/env python3
"""Teacher-v9.8 rollout-state common-descent response preflight."""

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

from fewshot_cdm_common import load_split  # noqa: E402
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
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
    stable_rollout_seeds as v97_rollout_seeds,
)
from relational_teacher_v98_rollout_state_contract import (  # noqa: E402
    CAPTURE_TIMESTEPS,
    DEVELOPMENT_SCENE,
    FAILED_V97_SCHEMA,
    HELDOUT_TRAIN_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SEED,
    STEP_RADII,
    TRAIN_SCENE,
    canonical_sha256,
    design_seeds,
    pooled,
    rank_candidates,
    response_checks,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
    _metrics,
    _physical,
    _prompt_invariance,
    _require_same_path,
    _restore_lora_state,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v97-summary", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v5-dataset-root", type=Path, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-evidence-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
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
        raise ValueError(label + " contains NaN/Inf")


def _validate_failed_v97(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != FAILED_V97_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_step") is not None
        or value.get("failed_checks") != ["at_least_one_early_step_is_admissible"]
        or value.get("authorizes_rollout_state_preflight") is not True
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("rollout_metric_policy_id") != V97_POLICY_ID
    ):
        raise ValueError("sealed Teacher-v9.7 failure authority changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.7 path binding is absent")
    for name, raw in paths.items():
        source = Path(str(raw)).expanduser().resolve()
        if not source.is_file() or sha256_file(source) != hashes[name]:
            raise ValueError("Teacher-v9.7 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.7 failure contains forbidden model state")
    return value


def _prompt_kwargs(bundle: Mapping[str, object], text: str, device: str) -> Dict[str, object]:
    xyz = torch.from_numpy(np.asarray(bundle["xyz"], np.float32)[None]).to(device).contiguous()
    feat = torch.from_numpy(
        np.asarray(bundle["points"], np.float32)[None, :, 3:6] / 255.0
    ).to(device).contiguous()
    result = {"c_pc_xyz": xyz, "c_pc_feat": feat, "c_text": [text]}
    if tuple(sorted(result)) != tuple(sorted(FORWARD_INPUT_KEYS)):
        raise AssertionError("Teacher forward input changed")
    return result


@torch.no_grad()
def _capture_trajectory(
    model,
    diffusion,
    bundle: Mapping[str, object],
    text: str,
    initial_seed: int,
    reverse_seed: int,
    device: str,
    progress: bool,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, Mapping[str, torch.Tensor]]]:
    if not hasattr(diffusion, "p_sample_loop_progressive"):
        raise RuntimeError("diffusion lacks p_sample_loop_progressive trajectory semantics")
    noise = deterministic_noise(torch.Size((1, 8192, 6)), initial_seed, device)
    kwargs = _prompt_kwargs(bundle, text, device)
    indices: Sequence[int] = list(range(diffusion.num_timesteps))[::-1]
    if tuple(CAPTURE_TIMESTEPS) != tuple(sorted(CAPTURE_TIMESTEPS, reverse=True)):
        raise AssertionError("capture timestep order changed")
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
    states: Dict[int, torch.Tensor] = {}
    rng_states: Dict[int, Mapping[str, torch.Tensor]] = {}
    torch_device = torch.device(device)
    devices = [
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(reverse_seed)
        torch.cuda.manual_seed(reverse_seed)
        image = noise
        for timestep in indices:
            if timestep in CAPTURE_TIMESTEPS:
                states[timestep] = image.detach().cpu().clone()
                rng_states[timestep] = {
                    "cpu": torch.random.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state(torch_device).cpu().clone(),
                }
            t = torch.tensor([timestep], dtype=torch.long, device=device)
            output = diffusion.p_sample(
                model,
                image,
                t,
                clip_denoised=False,
                model_kwargs=kwargs,
            )
            image = output["sample"]
    if set(states) != set(CAPTURE_TIMESTEPS) or set(rng_states) != set(CAPTURE_TIMESTEPS):
        raise AssertionError("trajectory capture inventory changed")
    return image[0].detach().cpu(), states, rng_states


@torch.no_grad()
def _resume_trajectory(
    model,
    diffusion,
    bundle: Mapping[str, object],
    text: str,
    state: torch.Tensor,
    rng_state: Mapping[str, torch.Tensor],
    start_timestep: int,
    device: str,
) -> torch.Tensor:
    kwargs = _prompt_kwargs(bundle, text, device)
    image = state.to(device)
    torch_device = torch.device(device)
    devices = [
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.set_rng_state(rng_state["cpu"].cpu())
        torch.cuda.set_rng_state(rng_state["cuda"].cpu(), torch_device)
        for timestep in range(int(start_timestep), -1, -1):
            t = torch.tensor([timestep], dtype=torch.long, device=device)
            output = diffusion.p_sample(
                model,
                image,
                t,
                clip_denoised=False,
                model_kwargs=kwargs,
            )
            image = output["sample"]
    return image[0].detach().cpu()


def _direct_prediction(model, diffusion, states: torch.Tensor, timestep: int, kwargs):
    wrapped = diffusion._wrap_model(model) if hasattr(diffusion, "_wrap_model") else model
    t = torch.full((2,), int(timestep), dtype=torch.long, device=states.device)
    return wrapped(states, diffusion._scale_timesteps(t), **dict(kwargs))


def _physical_pair(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(
        value.detach().cpu().numpy() * std.reshape(1, 1, 6)
        + mean.reshape(1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)


def _metric_pair(bundle: Mapping[str, object], value: np.ndarray) -> list[Mapping[str, object]]:
    if value.shape != (2, 8192, 6):
        raise ValueError("watch/write map shape changed")
    return [_metrics(bundle, value[index]) for index in range(2)]


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.8 rollout-state preflight requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.seed != SEED
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
    ):
        raise ValueError("Teacher-v9.8 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.8 output")
    configure_reproducibility(args.seed)

    failed_file = args.failed_v97_summary.expanduser().resolve()
    failed = _validate_failed_v97(failed_file)
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, original_policy = _validate_preflight(preflight_file)
    if failed.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("v9.7/preflight binding differs")

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_root.is_dir():
        raise FileNotFoundError("a sealed dataset root is absent")
    index_file = _require_same_path(args.index, preflight, "dataset_index")
    split_file = _require_same_path(args.v5_split, preflight, "v5_split")
    stats_file = _require_same_path(args.stats_file, preflight, "stats_file")
    original_checkpoint = _require_same_path(
        args.original_checkpoint, preflight, "original_checkpoint"
    )
    v5_checkpoint = _require_same_path(args.v5_checkpoint, preflight, "v5_checkpoint")
    evidence_file = _require_same_path(
        args.v5_evidence_report, preflight, "v5_evidence_report"
    )
    if (source_root / str(read_json(index_file)["source_index_file"])).resolve() != Path(
        str(preflight["paths"]["source_dataset_index"])
    ).resolve():
        raise ValueError("source dataset root differs from sealed preflight")
    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE:
        raise AssertionError("wrong scene loaded")
    scene_binding = {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]
    for name in ("instance_names", "instance_roles", "manifest_sha256", "consensus_sha256", "points_sha256"):
        actual = list(bundle[name]) if isinstance(bundle[name], tuple) else bundle[name]
        expected = scene_binding[name]
        if actual != expected:
            raise ValueError("room_0101 scene binding changed: " + name)

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "rollout_state_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("rollout-state policy hash changed")

    mean, std = load_stats(stats_file)
    batch = _build_batch(bundle, mean, std, args.device)
    kwargs = _kwargs(batch)
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
        raise ValueError("v5 probe order changed")
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
    audit_seeds = v97_rollout_seeds(0)
    design_seed_pair = design_seeds()
    trajectories: Dict[str, list[Mapping[str, object]]] = {"design": [], "audit": []}
    for domain, seeds in (("design", design_seed_pair), ("audit", audit_seeds)):
        for prompt_index, text in enumerate(prompt_texts):
            final, states, rng_states = _capture_trajectory(
                model,
                diffusion,
                bundle,
                text,
                seeds[0],
                seeds[1],
                args.device,
                not args.no_progress,
            )
            trajectories[domain].append(
                {"final": final, "states": states, "rng_states": rng_states}
            )
            print("[BASE {}] prompt={}".format(domain, PROMPT_IDS[prompt_index]), flush=True)

    failed_maps_file = Path(str(failed["paths"]["rollout_maps"])).resolve()
    with np.load(failed_maps_file, allow_pickle=False) as payload:
        v97_base_normalized = np.asarray(payload["frozen_base_normalized"], np.float32)
    audit_base_normalized = np.stack(
        [row["final"].numpy() for row in trajectories["audit"]]
    ).astype(np.float32)
    if not np.array_equal(audit_base_normalized, v97_base_normalized[0]):
        raise AssertionError("audit trajectory does not reproduce v9.7 generation-0 Base")

    # Each stored audit x_t plus its RNG state must exactly reproduce the same
    # Base final map before any candidate direction is evaluated.
    for prompt_index, text in enumerate(prompt_texts):
        for timestep in CAPTURE_TIMESTEPS:
            repeated = _resume_trajectory(
                model,
                diffusion,
                bundle,
                text,
                trajectories["audit"][prompt_index]["states"][timestep],
                trajectories["audit"][prompt_index]["rng_states"][timestep],
                timestep,
                args.device,
            ).numpy()
            if not np.array_equal(repeated, audit_base_normalized[prompt_index]):
                raise AssertionError("captured trajectory/RNG resume is not exact")
    print("[TRAJECTORY_REPRODUCTION_PASS] all six audit resumes are bitwise exact", flush=True)

    v5_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, SEED + 2000, args.device)
    with torch.no_grad():
        base_v5_prediction = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    base_v5_dense = (
        (base_v5_prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )

    modules = install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    parameters = list(named_lora.values())
    zero_state = _lora_cpu_state(named_lora)
    with torch.no_grad():
        zero_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not torch.equal(zero_v5, base_v5_prediction):
        raise AssertionError("fresh zero-init LoRA differs from v5r4")

    audit_base_physical = np.clip(
        audit_base_normalized * std.reshape(1, 1, 6) + mean.reshape(1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    base_rows = _metric_pair(bundle, audit_base_physical)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _prompt_invariance(audit_base_physical, verified_mask)
    base_pooled = pooled(base_rows)

    design_states_array = np.empty((len(CAPTURE_TIMESTEPS), 2, 8192, 6), np.float32)
    audit_states_array = np.empty_like(design_states_array)
    candidate_normalized = np.empty(
        (len(CAPTURE_TIMESTEPS), len(STEP_RADII), 2, 8192, 6), np.float32
    )
    direct_before = np.empty((len(CAPTURE_TIMESTEPS), 2, 8192, 6), np.float32)
    direct_after = np.empty_like(candidate_normalized)
    candidate_v5_predictions = np.empty(
        (len(CAPTURE_TIMESTEPS), len(STEP_RADII), 3, 8192, 6), np.float32
    )
    direction_rows = []
    candidate_rows = []

    for timestep_index, timestep in enumerate(CAPTURE_TIMESTEPS):
        _restore_lora_state(named_lora, zero_state)
        set_lora_enabled(model, True)
        set_frozen_base_eval_lora_train(model)
        design_states = torch.cat(
            [
                trajectories["design"][prompt]["states"][timestep]
                for prompt in range(2)
            ],
            dim=0,
        ).to(args.device)
        audit_states = torch.cat(
            [
                trajectories["audit"][prompt]["states"][timestep]
                for prompt in range(2)
            ],
            dim=0,
        ).to(args.device)
        design_states_array[timestep_index] = design_states.detach().cpu().numpy()
        audit_states_array[timestep_index] = audit_states.detach().cpu().numpy()
        with torch.no_grad():
            set_lora_enabled(model, False)
            frozen_design = _direct_prediction(
                model, diffusion, design_states, timestep, kwargs
            )
        set_lora_enabled(model, True)
        prediction = _direct_prediction(model, diffusion, design_states, timestep, kwargs)
        if not torch.equal(prediction.detach(), frozen_design):
            raise AssertionError("zero-init rollout-state prediction differs from v5r4")
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
            raise RuntimeError("rollout-state common direction is zero/non-finite")
        direction = direction / direction_norm
        derivatives = (gradients @ direction).detach().cpu().tolist()
        if min(float(value) for value in derivatives) < float(
            POLICY["selection"]["minimum_directional_derivative"]
        ):
            raise RuntimeError("rollout-state direction is not common descent")
        before_physical = _physical_pair(frozen_design, mean, std)
        direct_before[timestep_index] = before_physical
        direction_row = {
            "capture_timestep": int(timestep),
            "task_order": list(POLICY["object_tasks"]),
            "task_losses": [float(value) for value in task_losses.detach().cpu().tolist()],
            "gram": gram.tolist(),
            "weights": weights.tolist(),
            "directional_derivatives": [float(value) for value in derivatives],
            "direction_norm_before_unit": float(direction_norm.item()),
            "direction_sha256": tensor_sha256(direction),
        }
        direction_rows.append(direction_row)

        for radius_index, radius in enumerate(STEP_RADII):
            _restore_lora_state(named_lora, zero_state)
            set_lora_enabled(model, True)
            apply_flat_direction(parameters, direction, radius)
            model.eval()
            with torch.no_grad():
                after_direct_normal = _direct_prediction(
                    model, diffusion, design_states, timestep, kwargs
                )
                current_v5 = predict_xstart(
                    model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
                )
            after_direct_physical = _physical_pair(after_direct_normal, mean, std)
            direct_after[timestep_index, radius_index] = after_direct_physical
            current_v5_dense = (
                (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
            )
            candidate_v5_predictions[timestep_index, radius_index] = (
                current_v5.detach().cpu().numpy()
            )
            final_rows = []
            for prompt_index, text in enumerate(prompt_texts):
                final = _resume_trajectory(
                    model,
                    diffusion,
                    bundle,
                    text,
                    trajectories["audit"][prompt_index]["states"][timestep],
                    trajectories["audit"][prompt_index]["rng_states"][timestep],
                    timestep,
                    args.device,
                )
                candidate_normalized[
                    timestep_index, radius_index, prompt_index
                ] = final.numpy()
                final_rows.append(final)
            physical = np.clip(
                candidate_normalized[timestep_index, radius_index]
                * std.reshape(1, 1, 6)
                + mean.reshape(1, 1, 6),
                0.0,
                1.0,
            ).astype(np.float32)
            metrics = _metric_pair(bundle, physical)
            invariance = _prompt_invariance(physical, verified_mask)
            maximum_delta = float(np.abs(physical - audit_base_physical).max())
            checks = response_checks(
                base_rows=base_rows,
                candidate_rows=metrics,
                base_prompt_invariance=base_invariance,
                candidate_prompt_invariance=invariance,
                base_v5_dense=base_v5_dense,
                candidate_v5_dense=current_v5_dense,
                directional_derivatives=derivatives,
                maximum_map_delta=maximum_delta,
            )
            name = "t{}_radius_{}".format(
                timestep, str(radius).replace("0.", "0p")
            )
            row = {
                "name": name,
                "capture_timestep": int(timestep),
                "radius": float(radius),
                "direction_sha256": direction_row["direction_sha256"],
                "directional_derivatives": [float(value) for value in derivatives],
                "base_rows": base_rows,
                "candidate_rows": metrics,
                "base_pooled": base_pooled,
                "candidate_pooled": pooled(metrics),
                "base_prompt_invariance": float(base_invariance),
                "candidate_prompt_invariance": float(invariance),
                "base_v5_dense": [float(value) for value in base_v5_dense],
                "candidate_v5_dense": [float(value) for value in current_v5_dense],
                "maximum_final_map_delta": maximum_delta,
                "direct_before_rows": _metric_pair(bundle, before_physical),
                "direct_after_rows": _metric_pair(bundle, after_direct_physical),
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(key for key, passed in checks.items() if not passed),
            }
            _finite_tree(row, "v9.8 candidate")
            candidate_rows.append(row)
            print(
                "[ROLLOUT-STATE t={} R={}] eligible={} high-recall={:.6f}->{:.6f} high-MAE={:.6f}->{:.6f}".format(
                    timestep,
                    radius,
                    row["eligible"],
                    base_pooled["chair_06"]["soft_recall"],
                    row["candidate_pooled"]["chair_06"]["soft_recall"],
                    base_pooled["chair_06"]["active_support_mae"],
                    row["candidate_pooled"]["chair_06"]["active_support_mae"],
                ),
                flush=True,
            )
            del after_direct_normal, current_v5, final_rows
            torch.cuda.empty_cache()
        del prediction, objective, task_losses, gradients, direction

    eligible_order = rank_candidates(candidate_rows)
    selected_candidate = eligible_order[0] if eligible_order else None
    candidate_physical = np.clip(
        candidate_normalized * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    arrays_file = output_dir / "rollout_state_response_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(bundle["xyz"], np.float32),
        points=np.asarray(bundle["points"], np.float32),
        instance_ids=np.asarray(bundle["instance_ids"], np.int64),
        category_ids=np.asarray(bundle["category_ids"], np.int64),
        verified_object_mask=np.asarray(bundle["verified_object_mask"], bool),
        verified_positive_mask=np.asarray(bundle["verified_positive_mask"], bool),
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        all_sittable_gt=np.asarray(bundle["all_target"], np.float32),
        capture_timesteps=np.asarray(CAPTURE_TIMESTEPS, np.int64),
        step_radii=np.asarray(STEP_RADII, np.float64),
        prompt_ids=np.asarray(PROMPT_IDS),
        design_states=design_states_array,
        audit_states=audit_states_array,
        audit_base_normalized=audit_base_normalized,
        audit_base=audit_base_physical,
        candidates_normalized=candidate_normalized,
        candidates=candidate_physical,
        direct_before=direct_before,
        direct_after=direct_after,
        v5_target=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_prediction=base_v5_prediction.detach().cpu().numpy().astype(np.float32),
        candidate_v5_predictions=candidate_v5_predictions,
    )

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v98_rollout_state_response.py",
        "contract": PREPARE_ROOT / "relational_teacher_v98_rollout_state_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v98_rollout_state_response.py",
        "failed_v97_summary": failed_file,
        "failed_v97_maps": failed_maps_file,
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "rollout_state_policy": policy_file,
        "dataset_index": index_file,
        "source_dataset_index": Path(str(preflight["paths"]["source_dataset_index"])).resolve(),
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "rollout_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    checks = {
        "sealed_v97_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "rollout_policy_locked_before_lora": True,
        "new_design_seed_disjoint_from_v97_audit_seed": design_seed_pair != audit_seeds,
        "audit_base_exactly_reproduces_v97_generation0": True,
        "all_captured_rng_resumes_bitwise_exact": True,
        "six_object_prompt_tasks_have_common_descent": all(
            min(row["directional_derivatives"])
            >= float(POLICY["selection"]["minimum_directional_derivative"])
            for row in direction_rows
        ),
        "candidate_decision_uses_resumed_final_maps": True,
        "exact_topk_is_diagnostic_only": POLICY["topk_role"] == "diagnostic_only",
        "teacher_forward_is_text_plus_scene_only": tuple(sorted(kwargs))
        == tuple(sorted(FORWARD_INPUT_KEYS)),
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_rollout_state_response_is_admissible": selected_candidate is not None,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
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
        "prompt_ids": list(PROMPT_IDS),
        "prompt_text": {name: PROMPTS[name] for name in PROMPT_IDS},
        "forward_input_keys": sorted(kwargs),
        "capture_timesteps": list(CAPTURE_TIMESTEPS),
        "step_radii": list(STEP_RADII),
        "design_seeds": list(design_seed_pair),
        "audit_seeds": list(audit_seeds),
        "trajectory_accounting": {
            "base_full_draws": 4,
            "base_exact_partial_resumes": 6,
            "candidate_partial_resumes": 18,
        },
        "preflight_binding_id": preflight["binding_id"],
        "failed_v97_binding_id": failed["binding_id"],
        "scene_binding": scene_binding,
        "policy_id": POLICY_ID,
        "policy_sha256": path_hashes["rollout_state_policy"],
        "lora": dict(lora_metadata(model)),
        "serialized_model_state": False,
        "base_rows": base_rows,
        "base_pooled": base_pooled,
        "base_prompt_invariance": float(base_invariance),
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "direction_rows": direction_rows,
        "candidates": candidate_rows,
        "eligible_selection_order": eligible_order,
        "selected_candidate": selected_candidate,
        "selection_policy": POLICY["selection_order"],
        "paths": path_strings,
        "path_sha256": path_hashes,
        "rollout_maps_sha256": path_hashes["rollout_maps"],
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "authorizes_rollout_state_response6": status == "PASS",
        "authorizes_checkpoint": False,
        "authorizes_room_0102": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "preflight_binding_id": report["preflight_binding_id"],
            "failed_v97_binding_id": report["failed_v97_binding_id"],
            "policy_id": report["policy_id"],
            "design_seeds": report["design_seeds"],
            "audit_seeds": report["audit_seeds"],
            "selected_candidate": report["selected_candidate"],
            "rollout_maps_sha256": report["rollout_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v9.8 report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.8 unexpectedly saved model state")
    print("[ROLLOUT_STATE_PREFLIGHT_{}] Teacher-v9.8".format(status))
    print("[PASS] v9.7 Base generation-0 and all captured RNG resumes reproduced")
    print("[PASS] nine final-map response candidates evaluated from rollout states")
    print("[OK] selected candidate:", selected_candidate)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)
    del model, diffusion, zero_state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

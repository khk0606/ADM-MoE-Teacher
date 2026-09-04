#!/usr/bin/env python3
"""Fresh, direct-label Teacher-v10 supervised capacity training.

Unlike v9.8.x rollout-state calibration, every optimizer update starts from the
sealed all-sittable x0 label, noises it with q_sample at a locked timestep, and
trains a fresh rank-16 LoRA with AdamW.  Both train scenes contribute to every
update.  Only candidates that pass actual two-scene K=3 500-step rollouts may
be serialized.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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
    lora_parameter_energy,
    save_merged_legacy_state,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from relational_teacher_v10_supervised_capacity_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    DIFFUSION_STEPS,
    EVAL_TIMESTEPS,
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
    ROLES,
    ROLLOUT_K,
    SCENES,
    SCHEMA,
    SOURCE_SCENE,
    TIMESTEP_CYCLE,
    TRAIN_STEPS,
    V9812_SCHEMA,
    WEIGHT_DECAY,
    canonical_sha256,
    rollout_gate,
    select_rollout_candidate,
    shortlist_monitor_steps,
)
from relational_teacher_v10_supervised_objective import (  # noqa: E402
    direct_all_sittable_objective,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    PROMPTS,
    TEACHER_FORWARD_INPUTS,
    atomic_savez,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v9_all_sittable_metrics import (  # noqa: E402
    all_instance_metrics,
    simultaneous_presence_checks,
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
    parser.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _stable_seeds(seed: int, domain: str, case_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(
        "{}|{}|{}".format(seed, domain, case_id).encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(label + " contains a non-finite number")


def _sanitize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_sanitize(nested) for nested in value]
    if isinstance(value, tuple):
        return [_sanitize(nested) for nested in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return 1_000_000.0
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_v9812(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V9812_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("shortlisted_steps") != []
        or value.get("failed_checks")
        != ["at_least_one_all_three_state_is_shortlisted"]
        or value.get("update_count") != 6
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_cross_scene_objective_or_model_capacity_redesign")
        is not True
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v9.8.12 failure does not authorize v10 redesign")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Teacher-v9.8.12 path binding is absent")
    if set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.12 path/hash inventory changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v9.8.12 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed Teacher-v9.8.12 unexpectedly contains a checkpoint")
    return value


def _same_bound_arg(path: Path, authority: Mapping[str, object], name: str) -> Path:
    result = path.expanduser().resolve()
    expected = Path(str(authority["paths"][name])).expanduser().resolve()
    if result != expected or not result.is_file():
        raise ValueError(name + " differs from the sealed Teacher-v9.8.12 input")
    if sha256_file(result) != authority["path_sha256"][name]:
        raise ValueError(name + " hash changed")
    return result


def _build_batch(
    bundle: Mapping[str, object], mean: np.ndarray, std: np.ndarray, device: str
) -> Dict[str, object]:
    targets = np.stack([np.asarray(bundle["all_target"], np.float32)] * 2)
    points = np.stack([np.asarray(bundle["points"], np.float32)] * 2)
    batch: Dict[str, object] = {
        "x": torch.from_numpy(
            ((targets - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)).astype(
                np.float32
            )
        ).to(device),
        "xyz": torch.from_numpy(points[:, :, :3]).to(device).contiguous(),
        "feat": torch.from_numpy(points[:, :, 3:6] / 255.0).to(device).contiguous(),
        "text": [PROMPTS[prompt_id] for prompt_id in PROMPT_IDS],
        "prompt_ids": list(PROMPT_IDS),
    }
    for key, dtype in (
        ("instance_targets", np.float32),
        ("verified_object_mask", bool),
        ("verified_positive_mask", bool),
        ("unknown_sittable_mask", bool),
        ("environment_aux_mask", bool),
        ("explicit_negative_mask", bool),
        ("all_target", np.float32),
    ):
        batch[key] = torch.from_numpy(
            np.stack([np.asarray(bundle[key], dtype=dtype)] * 2)
        ).to(device)
    if batch["prompt_ids"] != list(PROMPT_IDS) or not torch.equal(
        batch["x"][0], batch["x"][1]
    ):
        raise AssertionError("watch/write direct-label pair changed")
    return batch


def _kwargs(batch: Mapping[str, object]) -> Dict[str, object]:
    value = {
        "c_pc_xyz": batch["xyz"],
        "c_pc_feat": batch["feat"],
        "c_text": batch["text"],
    }
    if tuple(sorted(value)) != tuple(sorted(TEACHER_FORWARD_INPUTS)):
        raise AssertionError("label metadata entered Teacher forward")
    return value


def _physical(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return np.clip(
        array * std.reshape(1, 1, 6) + mean.reshape(1, 1, 6), 0.0, 1.0
    ).astype(np.float32)


def _metrics(bundle: Mapping[str, object], prediction: np.ndarray) -> Dict[str, object]:
    return _sanitize(
        all_instance_metrics(
            prediction,
            np.asarray(bundle["instance_targets"], np.float32),
            bundle["instance_names"],
            np.asarray(bundle["xyz"], np.float32),
            np.asarray(bundle["verified_object_mask"], bool),
            np.asarray(bundle["explicit_negative_mask"], bool),
            np.asarray(bundle["unknown_sittable_mask"], bool),
        )
    )


def _role_metrics(
    scene: str, bundle: Mapping[str, object], predictions: np.ndarray
) -> tuple[list, Dict[str, Dict[str, float]]]:
    if predictions.ndim != 4 or predictions.shape[1:] != (2, 8192, 6):
        raise ValueError(scene + " prediction panel must be [K,2,8192,6]")
    rows = [
        [_metrics(bundle, predictions[generation, prompt]) for prompt in range(2)]
        for generation in range(predictions.shape[0])
    ]
    names = [str(name) for name in bundle["instance_names"]]
    pooled: Dict[str, Dict[str, float]] = {}
    for role, name in zip(ROLES, names):
        instance_rows = [
            rows[generation][prompt]["instances"][name]
            for generation in range(predictions.shape[0])
            for prompt in range(2)
        ]
        pooled[role] = {
            key: float(np.mean([float(row[key]) for row in instance_rows]))
            for key in (
                "soft_recall",
                "active_support_mae",
                "topk_overlap",
                "hotspot_centroid_distance_xy",
            )
        }
    return rows, pooled


def _known_dense_mae(bundle: Mapping[str, object], predictions: np.ndarray) -> float:
    known = (
        np.asarray(bundle["verified_positive_mask"], bool)
        | np.asarray(bundle["environment_aux_mask"], bool)
        | np.asarray(bundle["explicit_negative_mask"], bool)
    )
    target = np.asarray(bundle["all_target"], np.float32)
    return float(np.abs(predictions[:, :, known] - target[None, None, known]).mean())


def _prompt_invariance(bundle: Mapping[str, object], predictions: np.ndarray) -> list[float]:
    mask = np.asarray(bundle["verified_positive_mask"], bool)
    return [
        float(np.square(predictions[generation, 0, mask] - predictions[generation, 1, mask]).mean())
        for generation in range(predictions.shape[0])
    ]


def _lora_cpu_state(
    named: Mapping[str, torch.nn.Parameter],
) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in named.items()}


def _restore_lora_state(
    named: Mapping[str, torch.nn.Parameter], state: Mapping[str, torch.Tensor]
) -> None:
    if set(named) != set(state):
        raise ValueError("LoRA state inventory changed")
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(state[name].to(parameter.device))


def _state_sha256(named: Mapping[str, torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for name, value in named.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def _sample(
    model,
    diffusion,
    bundle: Mapping[str, object],
    text: str,
    initial_seed: int,
    reverse_seed: int,
    device: str,
    progress: bool,
) -> torch.Tensor:
    noise = deterministic_noise(torch.Size((1, 8192, 6)), initial_seed, device)
    xyz = torch.from_numpy(np.asarray(bundle["xyz"], np.float32)[None]).to(
        device
    ).contiguous()
    feat = torch.from_numpy(
        np.asarray(bundle["points"], np.float32)[None, :, 3:6] / 255.0
    ).to(device).contiguous()
    kwargs = {"c_pc_xyz": xyz, "c_pc_feat": feat, "c_text": [text]}
    torch_device = torch.device(device)
    devices = []
    if torch_device.type == "cuda":
        devices = [
            torch_device.index
            if torch_device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(reverse_seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed(reverse_seed)
        sample = diffusion.p_sample_loop(
            model,
            (1, 8192, 6),
            clip_denoised=False,
            noise=noise,
            model_kwargs=kwargs,
            device=device,
            progress=progress,
        )
    return sample[0].detach().cpu()


def _evaluate_fixed_panel(
    model,
    diffusion,
    batches: Mapping[str, Mapping[str, object]],
    kwargs: Mapping[str, Mapping[str, object]],
    bundles: Mapping[str, Mapping[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
    device: str,
) -> tuple[np.ndarray, Dict[str, object]]:
    maps = np.empty((2, len(EVAL_TIMESTEPS), 2, 8192, 6), np.float32)
    per_scene_role = {}
    dense_values = []
    for scene_index, scene in enumerate(SCENES):
        for timestep_index, timestep in enumerate(EVAL_TIMESTEPS):
            one_noise = deterministic_noise(
                torch.Size((1, 8192, 6)),
                seed + scene_index * 100003 + timestep * 101,
                device,
            )
            noise = one_noise.repeat(2, 1, 1)
            t = torch.full((2,), timestep, dtype=torch.long, device=device)
            with torch.no_grad():
                prediction = predict_xstart(
                    model,
                    diffusion,
                    batches[scene]["x"],
                    t,
                    kwargs[scene],
                    noise,
                )
            maps[scene_index, timestep_index] = _physical(prediction, mean, std)
            del prediction, noise, one_noise, t
        _, pooled = _role_metrics(scene, bundles[scene], maps[scene_index])
        per_scene_role[scene] = pooled
        dense_values.append(_known_dense_mae(bundles[scene], maps[scene_index]))
    worst_recall = min(
        per_scene_role[scene][role]["soft_recall"]
        for scene in SCENES
        for role in ROLES
    )
    worst_mae = max(
        per_scene_role[scene][role]["active_support_mae"]
        for scene in SCENES
        for role in ROLES
    )
    summary = {
        "per_scene_role": per_scene_role,
        "worst_instance_soft_recall": float(worst_recall),
        "worst_instance_active_support_mae": float(worst_mae),
        "known_dense_mae": float(np.mean(dense_values)),
    }
    _finite_tree(summary, "fixed one-step panel")
    return maps, summary


def _v5_fixed_prediction(
    model,
    diffusion,
    v5_batch: Mapping[str, object],
    v5_kwargs: Mapping[str, object],
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[float]]:
    t = torch.tensor((100, 300, 450), dtype=torch.long, device=device)
    noise = deterministic_noise(v5_batch["x"].shape, seed, device)
    with torch.no_grad():
        prediction = predict_xstart(
            model, diffusion, v5_batch["x"], t, v5_kwargs, noise
        )
    dense = (
        (prediction - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    return prediction.detach().cpu().numpy().astype(np.float32), [
        float(value) for value in dense
    ]


def _rollout_panel(
    bundles: Mapping[str, Mapping[str, object]],
    predictions: np.ndarray,
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    state_changed: bool,
) -> Dict[str, object]:
    scene_rows = {}
    pooled_metrics = {}
    presence = {}
    counts = {}
    invariance = {}
    for scene_index, scene in enumerate(SCENES):
        rows, pooled = _role_metrics(scene, bundles[scene], predictions[scene_index])
        scene_rows[scene] = rows
        pooled_metrics[scene] = pooled
        presence[scene] = [
            [
                simultaneous_presence_checks(
                    rows[generation][prompt], **ABSOLUTE_PRESENCE_LIMITS
                )
                for prompt in range(2)
            ]
            for generation in range(ROLLOUT_K)
        ]
        counts[scene] = {
            "watch": sum(all(presence[scene][generation][0].values()) for generation in range(ROLLOUT_K)),
            "write": sum(all(presence[scene][generation][1].values()) for generation in range(ROLLOUT_K)),
        }
        invariance[scene] = _prompt_invariance(
            bundles[scene], predictions[scene_index]
        )
    checks = rollout_gate(
        all_three_counts=counts,
        prompt_invariance=invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        checkpoint_state_changed=state_changed,
    )
    return {
        "scene_rows": scene_rows,
        "pooled_metrics": pooled_metrics,
        "presence": presence,
        "all_three_counts": counts,
        "prompt_invariance": invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "eligible": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v10 supervised capacity training requires CUDA")
    if (
        args.diffusion_steps != DIFFUSION_STEPS
        or args.steps != TRAIN_STEPS
        or float(args.lr) != LEARNING_RATE
        or float(args.weight_decay) != WEIGHT_DECAY
        or float(args.grad_clip) != GRAD_CLIP
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
        or args.seed != MODEL_SEED
    ):
        raise ValueError("Teacher-v10 supervised protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v10 output")
    failed_file = args.failed_calibration_summary.expanduser().resolve()
    v9812 = _validate_v9812(failed_file)

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _same_bound_arg(args.index, v9812, "dataset_index")
    split_file = _same_bound_arg(args.v5_split, v9812, "v5_split")
    stats_file = _same_bound_arg(args.stats_file, v9812, "stats_file")
    original_checkpoint = _same_bound_arg(
        args.original_checkpoint, v9812, "original_checkpoint"
    )
    v5_checkpoint = _same_bound_arg(args.v5_checkpoint, v9812, "v5_checkpoint")
    evidence_file = _same_bound_arg(
        args.v5_evidence_report, v9812, "v5_evidence_report"
    )
    source_index = Path(str(v9812["paths"]["source_dataset_index"])).resolve()
    if dataset_root != index_file.parent or source_root != source_index.parent:
        raise ValueError("dataset roots differ from sealed Teacher-v9.8.12")
    if v5_root != v5_checkpoint.parents[2]:
        raise ValueError("v5 dataset root differs from sealed checkpoint layout")

    output_dir.mkdir(parents=True)
    policy_file = output_dir / "supervised_capacity_policy.json"
    atomic_write_json(policy_file, POLICY)
    if canonical_sha256(read_json(policy_file)) != POLICY_ID:
        raise AssertionError("Teacher-v10 policy hash changed")

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != set(SCENES):
        raise ValueError("Teacher-v10 train-scene inventory changed")
    bundles = {
        scene: load_train_scene_bundle(dataset_root, source_root, records[scene])
        for scene in SCENES
    }
    for scene in SCENES:
        if tuple(str(name) for name in bundles[scene]["instance_names"]) != EXPECTED_INSTANCES[scene]:
            raise ValueError(scene + " verified instance order changed")

    configure_reproducibility(args.seed)
    mean, std = load_stats(stats_file)
    batches = {
        scene: _build_batch(bundles[scene], mean, std, args.device)
        for scene in SCENES
    }
    kwargs = {scene: _kwargs(batches[scene]) for scene in SCENES}
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    metric_policy = read_json(Path(str(v9812["paths"]["metric_policy"])).resolve())
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
        raise ValueError("v5 fixed probe target order changed")
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

    base_fixed_maps, base_fixed_summary = _evaluate_fixed_panel(
        model,
        diffusion,
        batches,
        kwargs,
        bundles,
        mean,
        std,
        args.seed + 700000,
        args.device,
    )
    base_v5_prediction, base_v5_dense = _v5_fixed_prediction(
        model,
        diffusion,
        v5_batch,
        v5_kwargs,
        args.seed + 800000,
        args.device,
    )

    modules = install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("Teacher-v10 LoRA module inventory changed")
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
    zero_fixed_maps, _ = _evaluate_fixed_panel(
        model,
        diffusion,
        batches,
        kwargs,
        bundles,
        mean,
        std,
        args.seed + 700000,
        args.device,
    )
    zero_v5_prediction, _ = _v5_fixed_prediction(
        model,
        diffusion,
        v5_batch,
        v5_kwargs,
        args.seed + 800000,
        args.device,
    )
    if not np.array_equal(zero_fixed_maps, base_fixed_maps) or not np.array_equal(
        zero_v5_prediction, base_v5_prediction
    ):
        raise AssertionError("fresh zero-init LoRA differs from sealed v5r4")
    del zero_fixed_maps, zero_v5_prediction

    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    monitor_rows = []
    monitor_maps = np.empty(
        (len(MONITOR_STEPS), 2, len(EVAL_TIMESTEPS), 2, 8192, 6), np.float32
    )
    states_by_step: Dict[int, Dict[str, torch.Tensor]] = {}
    training_log_rows = []
    timestep_hits = Counter()

    for step in range(1, args.steps + 1):
        set_frozen_base_eval_lora_train(model)
        set_lora_enabled(model, True)
        optimizer.zero_grad(set_to_none=True)
        component_sums = {name: 0.0 for name in LOSS_WEIGHTS}
        active_means = np.zeros(3, np.float64)
        training_total = 0.0
        for scene_index, scene in enumerate(SCENES):
            timestep = TIMESTEP_CYCLE[(2 * (step - 1) + scene_index) % len(TIMESTEP_CYCLE)]
            timestep_hits[timestep] += 1
            t = torch.full((2,), timestep, dtype=torch.long, device=args.device)
            one_noise = deterministic_noise(
                torch.Size((1, 8192, 6)),
                args.seed + step * 1009 + scene_index * 1000003,
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
            objective = direct_all_sittable_objective(
                prediction, batches[scene], mean_tensor, std_tensor
            )
            scene_loss = sum(
                float(LOSS_WEIGHTS[name]) * objective[name]
                for name in (
                    "instance_full",
                    "instance_active",
                    "verified_union",
                    "environment",
                    "explicit_negative_absolute",
                    "prompt_invariance",
                )
            )
            (0.5 * scene_loss).backward()
            training_total += 0.5 * float(scene_loss.detach().item())
            for name in (
                "instance_full",
                "instance_active",
                "verified_union",
                "environment",
                "explicit_negative_absolute",
                "prompt_invariance",
            ):
                component_sums[name] += 0.5 * float(objective[name].detach().item())
            active_means += 0.5 * (
                objective["per_instance_active_mean"].detach().mean(dim=0).cpu().numpy()
            )
            del t, one_noise, noise, prediction, objective, scene_loss

        replay_value = 0.0
        if step >= REPLAY_START_STEP:
            v5_t_values = (100, 300, 450)
            v5_t = torch.tensor(
                [v5_t_values[(step + offset) % 3] for offset in range(3)],
                dtype=torch.long,
                device=args.device,
            )
            v5_noise = deterministic_noise(
                v5_batch["x"].shape, args.seed + 900000 + step * 1013, args.device
            )
            set_lora_enabled(model, False)
            with torch.no_grad():
                frozen_v5 = predict_xstart(
                    model,
                    diffusion,
                    v5_batch["x"],
                    v5_t,
                    v5_kwargs,
                    v5_noise,
                )
            set_lora_enabled(model, True)
            candidate_v5 = predict_xstart(
                model,
                diffusion,
                v5_batch["x"],
                v5_t,
                v5_kwargs,
                v5_noise,
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
            raise AssertionError("gradient reached the frozen CDM")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip).item())
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError("Teacher-v10 LoRA gradient is zero/non-finite")
        optimizer.step()

        if step == 1 or step % 25 == 0:
            log_row = {
                "step": step,
                "total": float(training_total),
                "gradient_l2_before_clip": grad_norm,
                "active_prediction_mean_bed": float(active_means[0]),
                "active_prediction_mean_normal_chair": float(active_means[1]),
                "active_prediction_mean_high_chair": float(active_means[2]),
                "replay": replay_value,
                "components": dict(component_sums),
            }
            _finite_tree(log_row, "training log")
            training_log_rows.append(log_row)
            print(
                "[TRAIN] step={:04d}/{} total={:.6f} active-mean={:.4f}/{:.4f}/{:.4f} replay={:.6f} grad={:.6f}".format(
                    step,
                    args.steps,
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
            set_lora_enabled(model, True)
            model.eval()
            fixed_maps, fixed_summary = _evaluate_fixed_panel(
                model,
                diffusion,
                batches,
                kwargs,
                bundles,
                mean,
                std,
                args.seed + 700000,
                args.device,
            )
            monitor_index = list(MONITOR_STEPS).index(step)
            monitor_maps[monitor_index] = fixed_maps
            fixed_summary.update(
                {
                    "step": step,
                    "state_sha256": _state_sha256(named_lora),
                    "lora_energy": float(lora_parameter_energy(model).item()),
                }
            )
            _finite_tree(fixed_summary, "monitor row")
            monitor_rows.append(fixed_summary)
            states_by_step[step] = _lora_cpu_state(named_lora)
            print(
                "[MONITOR] step={} worst-recall={:.6f} worst-MAE={:.6f} known-MAE={:.6f}".format(
                    step,
                    fixed_summary["worst_instance_soft_recall"],
                    fixed_summary["worst_instance_active_support_mae"],
                    fixed_summary["known_dense_mae"],
                ),
                flush=True,
            )
        if step % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if set(timestep_hits) != set(TIMESTEP_CYCLE) or min(timestep_hits.values()) <= 0:
        raise AssertionError("full timestep curriculum was not exercised")
    shortlisted_steps = shortlist_monitor_steps(monitor_rows)
    zero_state = zero_state_sha256

    rollout_seed_table = [
        _stable_seeds(args.seed, "v10_actual_rollout", str(generation))
        for generation in range(ROLLOUT_K)
    ]
    base_rollouts = np.empty((2, ROLLOUT_K, 2, 8192, 6), np.float32)
    set_lora_enabled(model, False)
    model.eval()
    for scene_index, scene in enumerate(SCENES):
        for generation, (initial_seed, reverse_seed) in enumerate(rollout_seed_table):
            for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                sample = _sample(
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
        (len(shortlisted_steps), 2, ROLLOUT_K, 2, 8192, 6), np.float32
    )
    candidate_v5_predictions = np.empty(
        (len(shortlisted_steps), 3, 8192, 6), np.float32
    )
    rollout_rows = []
    for candidate_index, step in enumerate(shortlisted_steps):
        _restore_lora_state(named_lora, states_by_step[step])
        set_lora_enabled(model, True)
        model.eval()
        state_sha256 = _state_sha256(named_lora)
        for scene_index, scene in enumerate(SCENES):
            for generation, (initial_seed, reverse_seed) in enumerate(rollout_seed_table):
                for prompt_index, prompt_id in enumerate(PROMPT_IDS):
                    sample = _sample(
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
        v5_prediction, candidate_v5_dense = _v5_fixed_prediction(
            model,
            diffusion,
            v5_batch,
            v5_kwargs,
            args.seed + 800000,
            args.device,
        )
        candidate_v5_predictions[candidate_index] = v5_prediction
        panel = _rollout_panel(
            bundles,
            candidate_rollouts[candidate_index],
            base_v5_dense,
            candidate_v5_dense,
            state_sha256 != zero_state,
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
        _finite_tree(panel, "rollout candidate")
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
    checkpoint_file = output_dir / "teacher_v10_supervised.pt"
    if selected_step is not None:
        _restore_lora_state(named_lora, states_by_step[selected_step])
        set_lora_enabled(model, True)
    elif checkpoint_file.exists():
        raise AssertionError("failed Teacher-v10 run may not write a checkpoint")

    maps_file = output_dir / "supervised_capacity_maps.npz"
    save_arrays: Dict[str, np.ndarray] = {
        "scene_ids": np.asarray(SCENES),
        "prompt_ids": np.asarray(PROMPT_IDS),
        "eval_timesteps": np.asarray(EVAL_TIMESTEPS, np.int64),
        "monitor_steps": np.asarray(MONITOR_STEPS, np.int64),
        "shortlisted_steps": np.asarray(shortlisted_steps, np.int64),
        "rollout_seed_table": np.asarray(rollout_seed_table, np.int64),
        "base_fixed": base_fixed_maps,
        "monitor_fixed": monitor_maps,
        "base_rollouts": base_rollouts,
        "candidate_rollouts": candidate_rollouts,
        "v5_target": v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        "base_v5_prediction": base_v5_prediction,
        "candidate_v5_predictions": candidate_v5_predictions,
    }
    for scene in SCENES:
        prefix = "source" if scene == SOURCE_SCENE else "audit"
        bundle = bundles[scene]
        save_arrays.update(
            {
                prefix + "_xyz": np.asarray(bundle["xyz"], np.float32),
                prefix + "_points": np.asarray(bundle["points"], np.float32),
                prefix + "_instance_names": np.asarray(bundle["instance_names"]),
                prefix + "_verified_object_mask": np.asarray(
                    bundle["verified_object_mask"], bool
                ),
                prefix + "_verified_positive_mask": np.asarray(
                    bundle["verified_positive_mask"], bool
                ),
                prefix + "_unknown_sittable_mask": np.asarray(
                    bundle["unknown_sittable_mask"], bool
                ),
                prefix + "_environment_aux_mask": np.asarray(
                    bundle["environment_aux_mask"], bool
                ),
                prefix + "_explicit_negative_mask": np.asarray(
                    bundle["explicit_negative_mask"], bool
                ),
                prefix + "_instance_targets": np.asarray(
                    bundle["instance_targets"], np.float32
                ),
                prefix + "_all_sittable_gt": np.asarray(
                    bundle["all_target"], np.float32
                ),
            }
        )
    atomic_savez(maps_file, **save_arrays)
    if selected_step is not None:
        save_merged_legacy_state(model, checkpoint_file)
        if not checkpoint_file.is_file() or checkpoint_file.stat().st_size <= 0:
            raise RuntimeError("Teacher-v10 checkpoint serialization failed")

    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v10_supervised_capacity.py",
        "contract": PREPARE_ROOT
        / "relational_teacher_v10_supervised_capacity_contract.py",
        "objective": PREPARE_ROOT / "relational_teacher_v10_supervised_objective.py",
        "v9812_failure": failed_file,
        "metric_policy": Path(str(v9812["paths"]["metric_policy"])).resolve(),
        "dataset_index": index_file,
        "source_dataset_index": source_index,
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "training_policy": policy_file,
        "maps": maps_file,
    }
    if selected_step is not None:
        source_paths["checkpoint"] = checkpoint_file
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}

    status = "PASS" if selected_step is not None else "FAIL"
    checks = {
        "sealed_v9812_failure_authorizes_redesign": True,
        "fresh_v5r4_zero_init": True,
        "both_train_scenes_used_in_every_update": True,
        "all_timestep_buckets_exercised": set(timestep_hits) == set(TIMESTEP_CYCLE),
        "exact_1200_adamw_updates": args.steps == TRAIN_STEPS,
        "only_lora_received_gradients": True,
        "three_monitor_states_received_actual_two_scene_k3": len(rollout_rows)
        == len(shortlisted_steps)
        == 3,
        "at_least_one_supervised_candidate_passes_actual_k3": selected_step
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
        "seed": args.seed,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "train_steps": args.steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "train_scenes": list(SCENES),
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "development_arrays_read": False,
        "paper_test_access": False,
        "forward_input_keys": sorted(FORWARD_INPUT_KEYS),
        "prompt_ids": list(PROMPT_IDS),
        "policy_id": POLICY_ID,
        "v9812_binding_id": v9812.get("binding_id"),
        "zero_state_sha256": zero_state_sha256,
        "lora": dict(lora_metadata(model)),
        "optimizer": {"name": "AdamW", "weight_decay": args.weight_decay},
        "loss_weights": dict(LOSS_WEIGHTS),
        "replay_start_step": REPLAY_START_STEP,
        "replay_weight": REPLAY_WEIGHT,
        "timestep_hits": {str(key): int(value) for key, value in sorted(timestep_hits.items())},
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
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "authorizes_teacher_v10_checkpoint_lock": status == "PASS",
        "authorizes_capacity_or_objective_redesign": status == "FAIL",
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "policy_id": POLICY_ID,
            "v9812_binding_id": report["v9812_binding_id"],
            "maps_sha256": report["maps_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "selected_step": selected_step,
        }
    )
    _finite_tree(report, "Teacher-v10 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    print("[SUPERVISED_CAPACITY_{}] Teacher-v10".format(status))
    print("[PASS] direct GT q_sample training across all timesteps completed")
    print("[PASS] two train scenes x K=3 actual 500-step candidate rollouts completed")
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

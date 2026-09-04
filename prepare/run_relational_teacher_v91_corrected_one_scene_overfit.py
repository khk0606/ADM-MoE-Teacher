#!/usr/bin/env python3
"""Fresh corrected one-scene learnability smoke for Teacher-v9.1.

This is deliberately not a checkpoint-selection run.  It starts from sealed
v5r4 plus zero-output LoRA, updates only on room_0101, and never serializes
model state.  A PASS additionally requires an actual reverse-diffusion output
for both object-agnostic prompts to contain Bed, normal Chair and High Chair
simultaneously under the metric policy sealed before the first optimizer.
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
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
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
from relational_teacher_v9_lora_objective import (  # noqa: E402
    physical_prediction,
    preservation_loss,
)
from relational_teacher_v91_active_support_objective import (  # noqa: E402
    corrected_objective,
)
from relational_teacher_v91_loss_response_contract import (  # noqa: E402
    CANDIDATES as LOSS_RESPONSE_CANDIDATES,
)
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    OBJECTIVE_WEIGHTS,
    canonical_sha256,
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
from relational_teacher_v91_corrected_overfit_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    GRAD_CLIP,
    HELDOUT_TRAIN_SCENE,
    LEARNING_RATE,
    LOSS_RESPONSE_SCHEMA,
    MONITOR_STEPS,
    PREFLIGHT_SCHEMA,
    SCHEMA,
    SEED,
    SELECTED_LOSS_CANDIDATE,
    STEPS,
    TRAIN_SCENE,
    evaluate_smoke_panel,
    rank_eligible_monitor_steps,
    sanitize_metric_nonfinite,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402
from validate_relational_teacher_v9_all_sittable_lora_preflight import (  # noqa: E402
    validate_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--loss-response-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _stable_seeds(seed: int, domain: str, case_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}|{domain}|{case_id}".encode("utf-8")).digest()
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
        raise ValueError(label + " contains a non-finite value")


def _validate_preflight(report_file: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    report = read_json(report_file)
    if (
        report.get("schema") != PREFLIGHT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("authorizes_one_scene_overfit_smoke") is not True
        or report.get("authorizes_response3") is not False
        or report.get("authorizes_calibration") is not False
        or report.get("authorizes_long_training") is not False
        or report.get("authorizes_development_evaluation") is not False
        or report.get("authorizes_paper_test") is not False
        or report.get("development_payloads_read") is not False
        or report.get("train_scenes") != [TRAIN_SCENE, HELDOUT_TRAIN_SCENE]
        or report.get("development_scenes_metadata_only") != [DEVELOPMENT_SCENE]
        or report.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or report.get("diffusion_steps") != 500
        or report.get("failed_checks")
    ):
        raise ValueError("Teacher-v9 preflight does not authorize this smoke")
    paths = report.get("paths")
    hashes = report.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("preflight path binding is absent")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes.get(name):
            raise ValueError("preflight-bound file changed: " + str(name))
    policy_file = Path(str(paths["metric_policy"])).resolve()
    policy = validate_policy(
        policy_file,
        str(report["metric_policy_sha256"]),
        str(report["metric_policy_id"]),
    )
    return report, policy


def _validate_loss_response(
    report_file: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    report = read_json(report_file)
    if (
        report.get("schema") != LOSS_RESPONSE_SCHEMA
        or report.get("status") != "PASS"
        or report.get("selected_candidate") != SELECTED_LOSS_CANDIDATE
        or not isinstance(report.get("eligible_selection_order"), list)
        or not report["eligible_selection_order"]
        or report["eligible_selection_order"][0] != SELECTED_LOSS_CANDIDATE
        or report.get("authorizes_corrected_one_scene_overfit") is not True
        or report.get("serialized_model_state") is not False
        or report.get("heldout_train_arrays_read") is not False
        or report.get("development_arrays_read") is not False
        or report.get("paper_test_access") is not False
        or report.get("failed_checks")
    ):
        raise ValueError("Teacher-v9.1 loss response does not authorize this smoke")
    if report.get("candidate_grid") != list(LOSS_RESPONSE_CANDIDATES):
        raise ValueError("Teacher-v9.1 loss-response candidate grid changed")
    paths = report.get("paths")
    hashes = report.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("loss-response path binding is absent")
    if set(paths) != set(hashes):
        raise ValueError("loss-response path/hash inventory differs")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("loss-response-bound file changed: " + str(name))
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("loss-response grid contains forbidden model state")
    rows = report.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("loss-response candidate rows are absent")
    selected = next(
        (row for row in rows if row.get("name") == SELECTED_LOSS_CANDIDATE),
        None,
    )
    expected = next(
        row for row in LOSS_RESPONSE_CANDIDATES
        if row["name"] == SELECTED_LOSS_CANDIDATE
    )
    if (
        not isinstance(selected, Mapping)
        or selected.get("eligible") is not True
        or selected.get("failed_checks")
        or any(selected.get(name) != value for name, value in expected.items())
    ):
        raise ValueError("selected loss-response row changed")
    return report, selected


def _require_same_path(path: Path, report: Mapping[str, object], name: str) -> Path:
    resolved = path.expanduser().resolve()
    expected = Path(str(report["paths"][name])).expanduser().resolve()
    if resolved != expected or sha256_file(resolved) != report["path_sha256"][name]:
        raise ValueError(name + " differs from sealed preflight")
    return resolved


def _build_batch(
    bundle: Mapping[str, object], mean: np.ndarray, std: np.ndarray, device: str
) -> Dict[str, object]:
    prompt_ids = sorted(PROMPTS)
    targets = np.stack([np.asarray(bundle["all_target"], np.float32)] * 2)
    points = np.stack([np.asarray(bundle["points"], np.float32)] * 2)
    batch: Dict[str, object] = {
        "x": torch.from_numpy(
            ((targets - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)).astype(np.float32)
        ).to(device),
        "xyz": torch.from_numpy(points[:, :, :3]).to(device).contiguous(),
        "feat": torch.from_numpy(points[:, :, 3:6] / 255.0).to(device).contiguous(),
        "text": [PROMPTS[name] for name in prompt_ids],
        "prompt_ids": prompt_ids,
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
    if batch["prompt_ids"] != ["sit_watch_v1", "sit_write_v1"]:
        raise AssertionError("prompt order changed")
    if not torch.equal(batch["x"][0], batch["x"][1]):
        raise AssertionError("watch/write GT differs")
    return batch


def _kwargs(batch: Mapping[str, object]) -> Dict[str, object]:
    value = {
        "c_pc_xyz": batch["xyz"],
        "c_pc_feat": batch["feat"],
        "c_text": batch["text"],
    }
    if tuple(sorted(value)) != tuple(sorted(TEACHER_FORWARD_INPUTS)):
        raise AssertionError("metadata entered Teacher forward")
    return value


def _metrics(bundle: Mapping[str, object], prediction: np.ndarray) -> Dict[str, object]:
    result = all_instance_metrics(
        prediction,
        np.asarray(bundle["instance_targets"], np.float32),
        bundle["instance_names"],
        np.asarray(bundle["xyz"], np.float32),
        np.asarray(bundle["verified_object_mask"], bool),
        np.asarray(bundle["explicit_negative_mask"], bool),
        np.asarray(bundle["unknown_sittable_mask"], bool),
    )
    # An entirely absent object has an infinite hotspot-centroid distance.  It
    # is a legitimate fail-closed result, not a runtime error.  Persist a large
    # finite sentinel so JSON remains strict and the absolute gate stays false.
    result = sanitize_metric_nonfinite(result)
    _finite_tree(result, "model metrics")
    return result


def _prompt_invariance(prediction: np.ndarray, mask: np.ndarray) -> float:
    values = prediction[:, mask]
    if values.shape[0] != 2 or values.shape[1] <= 0:
        raise ValueError("prompt-invariance panel changed")
    return float(np.square(values[0] - values[1]).mean())


def _physical(value: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return np.clip(
        array * std.reshape(1, 1, 6) + mean.reshape(1, 1, 6), 0.0, 1.0
    ).astype(np.float32)


def _panel(
    bundle: Mapping[str, object],
    base: np.ndarray,
    candidate: np.ndarray,
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    relative_rules: Mapping[str, object],
) -> Dict[str, object]:
    row_ids = [f"{TRAIN_SCENE}|{name}" for name in sorted(PROMPTS)]
    base_rows = {row_id: _metrics(bundle, base[index]) for index, row_id in enumerate(row_ids)}
    candidate_rows = {
        row_id: _metrics(bundle, candidate[index]) for index, row_id in enumerate(row_ids)
    }
    presence = {
        row_id: simultaneous_presence_checks(
            candidate_rows[row_id], **ABSOLUTE_PRESENCE_LIMITS
        )
        for row_id in row_ids
    }
    verified = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _prompt_invariance(base, verified)
    candidate_invariance = _prompt_invariance(candidate, verified)
    checks = evaluate_smoke_panel(
        candidate_rows=candidate_rows,
        base_rows=base_rows,
        absolute_presence=presence,
        base_prompt_invariance=base_invariance,
        candidate_prompt_invariance=candidate_invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        relative_rules=relative_rules,
    )
    return {
        "base_rows": base_rows,
        "candidate_rows": candidate_rows,
        "absolute_presence": presence,
        "base_prompt_invariance": base_invariance,
        "candidate_prompt_invariance": candidate_invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "checks": checks,
        "eligible": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


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
    xyz = torch.from_numpy(np.asarray(bundle["xyz"], np.float32)[None]).to(device).contiguous()
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


def _lora_cpu_state(named: Mapping[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in named.items()}


def _restore_lora_state(
    named: Mapping[str, torch.nn.Parameter], state: Mapping[str, torch.Tensor]
) -> None:
    if set(named) != set(state):
        raise ValueError("LoRA state inventory changed")
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(state[name].to(parameter.device))


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.1 corrected overfit smoke requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.steps != STEPS
        or float(args.lr) != LEARNING_RATE
        or float(args.grad_clip) != GRAD_CLIP
        or args.lora_rank != 4
        or float(args.lora_alpha) != 8.0
        or args.seed != SEED
    ):
        raise ValueError("Teacher-v9.1 corrected overfit hyperparameters are sealed")
    configure_reproducibility(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.1 corrected smoke")
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, policy = _validate_preflight(preflight_file)
    loss_response_file = args.loss_response_report.expanduser().resolve()
    loss_response, loss_weights = _validate_loss_response(loss_response_file)
    if loss_response.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("loss-response/preflight binding differs")
    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not v5_root.is_dir():
        raise FileNotFoundError("a dataset root is absent")
    index_file = _require_same_path(args.index, preflight, "dataset_index")
    v5_split_file = _require_same_path(args.v5_split, preflight, "v5_split")
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
        raise ValueError("source dataset root differs from preflight")

    top_index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise AssertionError("train-scene metadata inventory changed")
    # The only train-scene array load in this program.  room_0102 is metadata-only.
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE:
        raise AssertionError("overfit scene changed")
    preflight_binding = {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]
    for name in (
        "instance_names",
        "instance_roles",
        "manifest_sha256",
        "consensus_sha256",
        "points_sha256",
    ):
        actual = list(bundle[name]) if isinstance(bundle[name], tuple) else bundle[name]
        expected = (
            list(preflight_binding[name])
            if isinstance(bundle[name], tuple)
            else preflight_binding[name]
        )
        if actual != expected:
            raise ValueError(TRAIN_SCENE + " preflight scene binding changed: " + name)

    mean, std = load_stats(stats_file)
    batch = _build_batch(bundle, mean, std, args.device)
    kwargs = _kwargs(batch)
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    split = load_split(v5_split_file)
    v5_rows = load_v5_rows(v5_root, split, "train", mean, std, 4.0, 16.0, 0.7)
    counts = Counter(str(row["target"]) for row in v5_rows.values())
    if counts != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise ValueError("sealed v5 replay inventory changed")
    v5_ids = list(policy["v5_probe_ids"])
    if [str(v5_rows[name]["target"]) for name in v5_ids] != [
        "chair",
        "bed",
        "whiteboard",
    ]:
        raise ValueError("v5 fixed-probe target order changed")
    v5_batch = stack_v5_batch(v5_rows, v5_ids, args.device)
    v5_kwargs = {
        "c_pc_xyz": v5_batch["xyz"],
        "c_pc_feat": v5_batch["feat"],
        "c_text": v5_batch["text"],
    }
    monitor_t = torch.tensor([125, 125], dtype=torch.long, device=args.device)
    monitor_noise_one = deterministic_noise(
        torch.Size((1, 8192, 6)), args.seed + 1000, args.device
    )
    monitor_noise = monitor_noise_one.repeat(2, 1, 1)
    v5_monitor_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_monitor_noise = deterministic_noise(
        v5_batch["x"].shape, args.seed + 2000, args.device
    )

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    with torch.no_grad():
        base_monitor_normal = predict_xstart(
            model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise
        )
        base_v5_monitor = predict_xstart(
            model,
            diffusion,
            v5_batch["x"],
            v5_monitor_t,
            v5_kwargs,
            v5_monitor_noise,
        )
    base_monitor = _physical(base_monitor_normal, mean, std)
    base_v5_dense = (
        (base_v5_monitor - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )

    modules = install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("Teacher-v9 LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    with torch.no_grad():
        zero = predict_xstart(
            model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise
        )
        zero_v5 = predict_xstart(
            model,
            diffusion,
            v5_batch["x"],
            v5_monitor_t,
            v5_kwargs,
            v5_monitor_noise,
        )
    if not torch.equal(zero, base_monitor_normal) or not torch.equal(
        zero_v5, base_v5_monitor
    ):
        raise AssertionError("fresh zero-init LoRA differs from sealed v5r4")

    optimizer = torch.optim.AdamW(list(named_lora.values()), lr=args.lr, weight_decay=0.0)
    monitor_rows = []
    eligible_states = []

    def monitor(step: int) -> Dict[str, object]:
        set_lora_enabled(model, True)
        model.eval()
        with torch.no_grad():
            current_normal = predict_xstart(
                model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise
            )
            current_v5 = predict_xstart(
                model,
                diffusion,
                v5_batch["x"],
                v5_monitor_t,
                v5_kwargs,
                v5_monitor_noise,
            )
        current = _physical(current_normal, mean, std)
        v5_dense = (
            (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        panel = _panel(
            bundle,
            base_monitor,
            current,
            base_v5_dense,
            v5_dense,
            policy["relative_selection_rules"],
        )
        panel["step"] = int(step)
        panel["prediction_sha256"] = tensor_sha256(current_normal)
        panel["v5_prediction_sha256"] = tensor_sha256(current_v5)
        panel["lora_energy"] = float(lora_parameter_energy(model).item())
        _finite_tree(panel, "monitor panel")
        return panel

    initial_panel = monitor(0)
    monitor_rows.append(initial_panel)
    for step in range(1, args.steps + 1):
        set_frozen_base_eval_lora_train(model)
        set_lora_enabled(model, True)
        optimizer.zero_grad(set_to_none=True)
        timestep_value = (50, 150, 250, 350, 450)[(step - 1) % 5]
        timesteps = torch.full((2,), timestep_value, dtype=torch.long, device=args.device)
        one_noise = deterministic_noise(
            torch.Size((1, 8192, 6)), args.seed + step * 1009, args.device
        )
        noise = one_noise.repeat(2, 1, 1)
        if not torch.equal(noise[0], noise[1]):
            raise AssertionError("training prompt pair noise changed")

        set_lora_enabled(model, False)
        with torch.no_grad():
            frozen = predict_xstart(
                model, diffusion, batch["x"], timesteps, kwargs, noise
            )
        set_lora_enabled(model, True)
        prediction = predict_xstart(
            model, diffusion, batch["x"], timesteps, kwargs, noise
        )
        objective = corrected_objective(
            prediction, frozen, batch, mean_tensor, std_tensor
        )

        v5_t_values = (100, 300, 450)
        v5_t = torch.tensor(
            [v5_t_values[(step - 1 + offset) % 3] for offset in range(3)],
            dtype=torch.long,
            device=args.device,
        )
        v5_noise = deterministic_noise(
            v5_batch["x"].shape, args.seed + 50000 + step * 1013, args.device
        )
        set_lora_enabled(model, False)
        with torch.no_grad():
            frozen_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        set_lora_enabled(model, True)
        current_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
        replay = preservation_loss(current_v5, frozen_v5)
        total = (
            OBJECTIVE_WEIGHTS["instance_macro_primary"]
            * objective["instance_macro_primary"]
            + OBJECTIVE_WEIGHTS["verified_union"] * objective["verified_union"]
            + OBJECTIVE_WEIGHTS["environment_auxiliary"]
            * objective["environment_auxiliary"]
            + float(loss_weights["active_support_weight"])
            * objective["active_support_macro"]
            + float(loss_weights["ranking_weight"])
            * objective["within_object_ranking"]
            + float(loss_weights["negative_weight"])
            * objective["explicit_negative_addition"]
            + float(loss_weights["prompt_weight"])
            * objective["paired_prompt_invariance"]
            + float(loss_weights["v5_preservation_weight"]) * replay
            + OBJECTIVE_WEIGHTS["lora_regularizer"] * lora_parameter_energy(model)
        )
        total.backward()
        if any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if "lora_" not in name
        ):
            raise AssertionError("gradient reached frozen CDM")
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(list(named_lora.values()), args.grad_clip).item()
        )
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError("non-finite or zero LoRA gradient")
        optimizer.step()
        if step == 1 or step % 10 == 0:
            per_instance = (
                objective["per_instance_active_support"].detach().mean(dim=0).cpu()
            )
            total_value = float(total.detach().item())
            ranking_value = float(
                objective["within_object_ranking"].detach().item()
            )
            print(
                f"[CORRECTED] step={step:03d}/{args.steps} "
                f"total={total_value:.6f} "
                f"support(bed/chair/high)={float(per_instance[0]):.6f}/"
                f"{float(per_instance[1]):.6f}/{float(per_instance[2]):.6f} "
                f"rank={ranking_value:.6f} "
                f"grad={grad_norm:.6f}",
                flush=True,
            )
        del frozen, prediction, objective, frozen_v5, current_v5, replay, total
        if step in MONITOR_STEPS:
            row = monitor(step)
            monitor_rows.append(row)
            if row["eligible"]:
                eligible_states.append(
                    {
                        "step": step,
                        "state": _lora_cpu_state(named_lora),
                        "panel": copy.deepcopy(row),
                    }
                )
            chair = row["candidate_rows"][f"{TRAIN_SCENE}|sit_watch_v1"]["instances"]
            print(
                f"[MONITOR] step={step:03d} eligible={row['eligible']} "
                f"MAE(bed/chair/high)="
                f"{chair['bed_01']['active_support_mae']:.6f}/"
                f"{chair['chair_01']['active_support_mae']:.6f}/"
                f"{chair['chair_06']['active_support_mae']:.6f}",
                flush=True,
            )
        if step % 10 == 0:
            torch.cuda.empty_cache()

    # Use a fixed, predeclared train-only ranking: maximize worst-object top-k,
    # then recall, then minimize worst active-support MAE.  State stays in RAM.
    ranked_monitor_steps = rank_eligible_monitor_steps(
        [row["panel"] for row in eligible_states]
    )
    state_by_step = {int(row["step"]): row for row in eligible_states}
    selected = state_by_step[ranked_monitor_steps[0]] if ranked_monitor_steps else {
        "step": args.steps,
        "state": _lora_cpu_state(named_lora),
        "panel": monitor_rows[-1],
    }
    _restore_lora_state(named_lora, selected["state"])
    set_lora_enabled(model, True)
    model.eval()
    initial_seed, reverse_seed = _stable_seeds(
        args.seed, "v91_corrected_one_scene_overfit_rollout", TRAIN_SCENE
    )
    base_rollout_rows = []
    candidate_rollout_rows = []
    for prompt_id in sorted(PROMPTS):
        set_lora_enabled(model, False)
        base_sample = _sample(
            model,
            diffusion,
            bundle,
            PROMPTS[prompt_id],
            initial_seed,
            reverse_seed,
            args.device,
            not args.no_progress,
        )
        set_lora_enabled(model, True)
        candidate_sample = _sample(
            model,
            diffusion,
            bundle,
            PROMPTS[prompt_id],
            initial_seed,
            reverse_seed,
            args.device,
            not args.no_progress,
        )
        base_rollout_rows.append(base_sample.numpy())
        candidate_rollout_rows.append(candidate_sample.numpy())
        print("[ROLLOUT]", TRAIN_SCENE, prompt_id, flush=True)
    set_lora_enabled(model, True)
    base_rollout_normal = np.stack(base_rollout_rows).astype(np.float32)
    candidate_rollout_normal = np.stack(candidate_rollout_rows).astype(np.float32)
    base_rollout = np.clip(
        base_rollout_normal * std.reshape(1, 1, 6) + mean.reshape(1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    candidate_rollout = np.clip(
        candidate_rollout_normal * std.reshape(1, 1, 6) + mean.reshape(1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    rollout_panel = _panel(
        bundle,
        base_rollout,
        candidate_rollout,
        selected["panel"]["base_v5_dense"],
        selected["panel"]["candidate_v5_dense"],
        policy["relative_selection_rules"],
    )

    selected_one_step_ok = bool(selected["panel"]["eligible"])
    rollout_ok = bool(rollout_panel["eligible"])
    changed = any(
        bool(torch.count_nonzero(value).item())
        for name, value in selected["state"].items()
        if name.endswith("lora_B")
    )
    gate_checks = {
        "sealed_loss_response_authority": True,
        "sealed_preflight_authority": True,
        "fresh_v5r4_zero_init": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "exact_120_optimizer_updates": True,
        "only_lora_updated": True,
        "one_step_three_object_panel_passes": selected_one_step_ok,
        "reverse_diffusion_three_object_panel_passes": rollout_ok,
        "lora_changed_from_zero": changed,
        "no_model_checkpoint_saved": True,
        "teacher_forward_is_text_plus_scene_only": tuple(sorted(kwargs))
        == tuple(sorted(FORWARD_INPUT_KEYS)),
    }
    status = "PASS" if all(gate_checks.values()) else "FAIL"
    output_dir.mkdir(parents=True)
    arrays_file = output_dir / "one_scene_rollout_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(bundle["xyz"], np.float32),
        points=np.asarray(bundle["points"], np.float32),
        instance_ids=np.asarray(bundle["instance_ids"], np.int64),
        category_ids=np.asarray(bundle["category_ids"], np.int64),
        verified_object_mask=np.asarray(bundle["verified_object_mask"], bool),
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        all_sittable_gt=np.asarray(bundle["all_target"], np.float32),
        frozen_base=base_rollout,
        candidate=candidate_rollout,
        frozen_base_normalized=base_rollout_normal,
        candidate_normalized=candidate_rollout_normal,
        prompt_ids=np.asarray(sorted(PROMPTS)),
    )
    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v91_corrected_one_scene_overfit.py",
        "contract": PREPARE_ROOT / "relational_teacher_v91_corrected_overfit_contract.py",
        "corrected_objective": PREPARE_ROOT
        / "relational_teacher_v91_active_support_objective.py",
        "loss_response_contract": PREPARE_ROOT
        / "relational_teacher_v91_loss_response_contract.py",
        "loss_response_report": loss_response_file,
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "dataset_index": index_file,
        "source_dataset_index": Path(
            str(preflight["paths"]["source_dataset_index"])
        ).resolve(),
        "stats_file": stats_file,
        "v5_split": v5_split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "rollout_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "steps": args.steps,
        "learning_rate": args.lr,
        "grad_clip": args.grad_clip,
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": HELDOUT_TRAIN_SCENE,
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "prompts": PROMPTS,
        "forward_input_keys": sorted(kwargs),
        "preflight_binding_id": preflight["binding_id"],
        "loss_response_binding_id": loss_response["binding_id"],
        "selected_loss_candidate": SELECTED_LOSS_CANDIDATE,
        "selected_loss_weights": {
            name: loss_weights[name]
            for name in (
                "active_support_weight",
                "ranking_weight",
                "negative_weight",
                "prompt_weight",
                "v5_preservation_weight",
            )
        },
        "metric_policy_id": preflight["metric_policy_id"],
        "metric_policy_sha256": preflight["metric_policy_sha256"],
        "scene_binding": preflight_binding,
        "lora": dict(lora_metadata(model)),
        "optimizer": {"name": "AdamW", "weight_decay": 0.0},
        "monitor_steps": list(MONITOR_STEPS),
        "monitor_panels": monitor_rows,
        "eligible_monitor_steps": [int(row["step"]) for row in eligible_states],
        "ranked_eligible_monitor_steps": ranked_monitor_steps,
        "selected_monitor_step": int(selected["step"]),
        "selection_policy": (
            "maximize_worst_topk_then_recall_then_minimize_worst_mae_"
            "then_earliest_else_final_diagnostic"
        ),
        "rollout": {
            "initial_seed": initial_seed,
            "reverse_seed": reverse_seed,
            "prompt_pair_uses_identical_randomness": True,
            "panel": rollout_panel,
        },
        "rollout_maps_sha256": path_hashes["rollout_maps"],
        "serialized_model_state": False,
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": gate_checks,
        "authorizes_fresh_response3": status == "PASS",
        "authorizes_calibration": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
        "failed_checks": sorted(name for name, passed in gate_checks.items() if not passed),
    }
    report["binding_id"] = canonical_sha256(
        {
            "preflight_binding_id": report["preflight_binding_id"],
            "loss_response_binding_id": report["loss_response_binding_id"],
            "selected_loss_candidate": report["selected_loss_candidate"],
            "selected_loss_weights": report["selected_loss_weights"],
            "metric_policy_id": report["metric_policy_id"],
            "scene_binding": report["scene_binding"],
            "hyperparameters": {
                "steps": args.steps,
                "learning_rate": args.lr,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
            },
            "rollout_maps_sha256": report["rollout_maps_sha256"],
        }
    )
    _finite_tree(report, "overfit report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    print(
        f"[CORRECTED_OVERFIT_{status}] "
        "Teacher-v9.1 active-support/ranking learnability smoke"
    )
    print("[OK] selected monitor step:", report["selected_monitor_step"])
    print("[OK] rollout presence:", rollout_panel["absolute_presence"])
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    # Make accidental checkpoint persistence visibly impossible in this output.
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("overfit smoke unexpectedly persisted model state")
    del model, diffusion, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fresh in-memory step-3/6/12 Teacher-v9.7 reverse-diffusion K=3 audit."""

from __future__ import annotations

import argparse
import copy
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
    lora_parameter_energy,
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
from relational_teacher_v9_lora_objective import preservation_loss  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    OBJECTIVE_WEIGHTS,
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
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    FAILED_V91_SCHEMA,
    GENERATION_COUNT,
    GRAD_CLIP,
    HELDOUT_TRAIN_SCENE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    OBJECTS,
    PROMPT_IDS,
    ROLLOUT_POLICY,
    ROLLOUT_POLICY_ID,
    ROLLOUT_SEED,
    SCHEMA,
    SELECTED_LOSS_CANDIDATE,
    SNAPSHOT_STEPS,
    TRAINING_SEED,
    TRAINING_STEPS,
    TRAIN_SCENE,
    canonical_sha256,
    continuous_presence_checks,
    pooled_object_metrics,
    rank_eligible_steps,
    rollout_step_checks,
    stable_rollout_seeds,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
    _metrics,
    _panel,
    _physical,
    _prompt_invariance,
    _require_same_path,
    _restore_lora_state,
    _sample,
    _validate_loss_response,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


EXPECTED_FAILED_CHECKS = [
    "one_step_three_object_panel_passes",
    "reverse_diffusion_three_object_panel_passes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v91-summary", type=Path, required=True)
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
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--rollout-seed", type=int, default=ROLLOUT_SEED)
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


def _equal_tree(left: object, right: object, label: str) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(label + " keys changed")
        for key in left:
            _equal_tree(left[key], right[key], label + "." + str(key))
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(label + " length changed")
        for index, (one, two) in enumerate(zip(left, right)):
            _equal_tree(one, two, "{}[{}]".format(label, index))
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " changed")


def _validate_failed_v91(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != FAILED_V91_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("seed") != TRAINING_SEED
        or value.get("steps") != 120
        or value.get("learning_rate") != LEARNING_RATE
        or value.get("grad_clip") != GRAD_CLIP
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("selected_loss_candidate") != SELECTED_LOSS_CANDIDATE
        or value.get("selected_monitor_step") != 120
        or value.get("failed_checks") != EXPECTED_FAILED_CHECKS
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("sealed Teacher-v9.1 failure authority changed")
    monitors = value.get("monitor_panels")
    if not isinstance(monitors, list):
        raise ValueError("sealed v9.1 monitor panels are absent")
    rows = {int(row.get("step", -1)): row for row in monitors}
    if not set((0,) + SNAPSHOT_STEPS).issubset(rows):
        raise ValueError("sealed v9.1 early monitor rows are absent")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("sealed v9.1 path binding is absent")
    for name, raw in paths.items():
        source = Path(str(raw)).expanduser().resolve()
        if not source.is_file() or sha256_file(source) != hashes[name]:
            raise ValueError("sealed v9.1 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("sealed v9.1 failure contains forbidden model state")
    return value


def _physical_rollout(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(value, np.float32) * std.reshape(1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)


def _rollout_metrics(
    bundle: Mapping[str, object], values: np.ndarray
) -> list[list[Mapping[str, object]]]:
    if values.shape != (GENERATION_COUNT, 2, 8192, 6):
        raise ValueError("rollout map tensor shape changed")
    return [
        [_metrics(bundle, values[generation, prompt]) for prompt in range(2)]
        for generation in range(GENERATION_COUNT)
    ]


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.7 early rollout K=3 requires CUDA")
    if (
        args.diffusion_steps != 500
        or args.steps != TRAINING_STEPS
        or float(args.lr) != LEARNING_RATE
        or float(args.grad_clip) != GRAD_CLIP
        or args.lora_rank != LORA_RANK
        or float(args.lora_alpha) != LORA_ALPHA
        or args.training_seed != TRAINING_SEED
        or args.rollout_seed != ROLLOUT_SEED
    ):
        raise ValueError("Teacher-v9.7 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.7 output")
    configure_reproducibility(args.training_seed)

    failed_file = args.failed_v91_summary.expanduser().resolve()
    failed = _validate_failed_v91(failed_file)
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, original_policy = _validate_preflight(preflight_file)
    response_file = args.loss_response_report.expanduser().resolve()
    response, loss_weights = _validate_loss_response(response_file)
    if (
        failed.get("preflight_binding_id") != preflight.get("binding_id")
        or failed.get("loss_response_binding_id") != response.get("binding_id")
        or response.get("preflight_binding_id") != preflight.get("binding_id")
    ):
        raise ValueError("v9.7 authorities do not share one binding")

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
        raise ValueError("Teacher-v9.7 train scene metadata changed")
    # The only scene-array load. room_0102 and room_0201 remain unread.
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    if str(bundle["scene_id"]) != TRAIN_SCENE:
        raise AssertionError("Teacher-v9.7 loaded the wrong scene")
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
            if isinstance(preflight_binding[name], list)
            else preflight_binding[name]
        )
        if actual != expected:
            raise ValueError(TRAIN_SCENE + " binding changed: " + name)

    # The rollout policy is materialized and hashed before any model or
    # optimizer exists. It cannot be adapted after observing CUDA outputs.
    output_dir.mkdir(parents=True)
    rollout_policy_file = output_dir / "rollout_metric_policy.json"
    atomic_write_json(rollout_policy_file, ROLLOUT_POLICY)
    if canonical_sha256(read_json(rollout_policy_file)) != ROLLOUT_POLICY_ID:
        raise AssertionError("rollout metric policy hash changed")

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
    if [str(v5_rows[name]["target"]) for name in v5_ids] != [
        "chair",
        "bed",
        "whiteboard",
    ]:
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

    monitor_t = torch.tensor([125, 125], dtype=torch.long, device=args.device)
    monitor_noise = deterministic_noise(
        torch.Size((1, 8192, 6)), TRAINING_SEED + 1000, args.device
    ).repeat(2, 1, 1)
    v5_monitor_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_monitor_noise = deterministic_noise(
        v5_batch["x"].shape, TRAINING_SEED + 2000, args.device
    )
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
        raise AssertionError("Teacher-v9.7 LoRA module inventory changed")
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

    reference_monitors = {
        int(row["step"]): row for row in failed["monitor_panels"]
    }

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
        current_v5_dense = (
            (current_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        result = _panel(
            bundle,
            base_monitor,
            current,
            base_v5_dense,
            current_v5_dense,
            original_policy["relative_selection_rules"],
        )
        result["step"] = int(step)
        result["prediction_sha256"] = tensor_sha256(current_normal)
        result["v5_prediction_sha256"] = tensor_sha256(current_v5)
        result["lora_energy"] = float(lora_parameter_energy(model).item())
        _finite_tree(result, "v9.7 monitor")
        _equal_tree(result, reference_monitors[step], "v9.1 reproduction step {}".format(step))
        return result

    reproduced_monitors = [monitor(0)]
    optimizer = torch.optim.AdamW(
        list(named_lora.values()), lr=args.lr, weight_decay=0.0
    )
    snapshots: Dict[int, Mapping[str, torch.Tensor]] = {}
    for step in range(1, TRAINING_STEPS + 1):
        set_frozen_base_eval_lora_train(model)
        set_lora_enabled(model, True)
        optimizer.zero_grad(set_to_none=True)
        timestep_value = (50, 150, 250, 350, 450)[(step - 1) % 5]
        timesteps = torch.full(
            (2,), timestep_value, dtype=torch.long, device=args.device
        )
        noise = deterministic_noise(
            torch.Size((1, 8192, 6)), TRAINING_SEED + step * 1009, args.device
        ).repeat(2, 1, 1)
        if not torch.equal(noise[0], noise[1]):
            raise AssertionError("training prompt-pair noise changed")
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
            v5_batch["x"].shape, TRAINING_SEED + 50000 + step * 1013, args.device
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
            torch.nn.utils.clip_grad_norm_(
                list(named_lora.values()), args.grad_clip
            ).item()
        )
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError("non-finite or zero LoRA gradient")
        optimizer.step()
        print(
            "[TRAIN] step={:02d}/12 total={:.6f} grad={:.6f}".format(
                step, float(total.detach().item()), grad_norm
            ),
            flush=True,
        )
        del frozen, prediction, objective, frozen_v5, current_v5, replay, total
        if step in SNAPSHOT_STEPS:
            reproduced = monitor(step)
            reproduced_monitors.append(reproduced)
            snapshots[step] = _lora_cpu_state(named_lora)
            print(
                "[REPRODUCTION_PASS] step={} prediction={} v5={}".format(
                    step,
                    reproduced["prediction_sha256"],
                    reproduced["v5_prediction_sha256"],
                ),
                flush=True,
            )
        torch.cuda.empty_cache()
    if tuple(sorted(snapshots)) != SNAPSHOT_STEPS:
        raise AssertionError("in-memory snapshot inventory changed")
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    prompt_texts = [PROMPTS[name] for name in PROMPT_IDS]
    seeds = [stable_rollout_seeds(generation) for generation in range(GENERATION_COUNT)]
    base_normalized = np.empty((GENERATION_COUNT, 2, 8192, 6), dtype=np.float32)
    set_lora_enabled(model, False)
    model.eval()
    for generation, (initial_seed, reverse_seed) in enumerate(seeds):
        for prompt_index, prompt_text in enumerate(prompt_texts):
            sample = _sample(
                model,
                diffusion,
                bundle,
                prompt_text,
                initial_seed,
                reverse_seed,
                args.device,
                not args.no_progress,
            )
            base_normalized[generation, prompt_index] = sample.numpy()
            print(
                "[BASE] generation={} prompt={}".format(
                    generation, PROMPT_IDS[prompt_index]
                ),
                flush=True,
            )
            del sample
            torch.cuda.empty_cache()

    candidate_normalized = np.empty(
        (len(SNAPSHOT_STEPS), GENERATION_COUNT, 2, 8192, 6), dtype=np.float32
    )
    for step_index, step in enumerate(SNAPSHOT_STEPS):
        _restore_lora_state(named_lora, snapshots[step])
        set_lora_enabled(model, True)
        model.eval()
        for generation, (initial_seed, reverse_seed) in enumerate(seeds):
            for prompt_index, prompt_text in enumerate(prompt_texts):
                sample = _sample(
                    model,
                    diffusion,
                    bundle,
                    prompt_text,
                    initial_seed,
                    reverse_seed,
                    args.device,
                    not args.no_progress,
                )
                candidate_normalized[
                    step_index, generation, prompt_index
                ] = sample.numpy()
                print(
                    "[STEP {}] generation={} prompt={}".format(
                        step, generation, PROMPT_IDS[prompt_index]
                    ),
                    flush=True,
                )
                del sample
                torch.cuda.empty_cache()

    # Four exact repeats prove that the seed pairing is effective: one Base and
    # all three in-memory candidates, generation 0/watch only.
    repeat_initial, repeat_reverse = seeds[0]
    set_lora_enabled(model, False)
    repeat = _sample(
        model,
        diffusion,
        bundle,
        prompt_texts[0],
        repeat_initial,
        repeat_reverse,
        args.device,
        not args.no_progress,
    ).numpy()
    if not np.array_equal(repeat, base_normalized[0, 0]):
        raise AssertionError("Base reverse-diffusion determinism changed")
    for step_index, step in enumerate(SNAPSHOT_STEPS):
        _restore_lora_state(named_lora, snapshots[step])
        set_lora_enabled(model, True)
        repeat = _sample(
            model,
            diffusion,
            bundle,
            prompt_texts[0],
            repeat_initial,
            repeat_reverse,
            args.device,
            not args.no_progress,
        ).numpy()
        if not np.array_equal(repeat, candidate_normalized[step_index, 0, 0]):
            raise AssertionError(
                "step-{} reverse-diffusion determinism changed".format(step)
            )
    print("[DETERMINISM_PASS] Base and step 3/6/12 exact repeats", flush=True)

    base_physical = _physical_rollout(base_normalized, mean, std)
    candidate_physical = np.clip(
        candidate_normalized * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    base_rows = _rollout_metrics(bundle, base_physical)
    base_invariance = [
        _prompt_invariance(
            base_physical[generation],
            np.asarray(bundle["verified_positive_mask"], bool),
        )
        for generation in range(GENERATION_COUNT)
    ]
    base_pooled = pooled_object_metrics(base_rows)
    step_rows = []
    monitor_by_step = {int(row["step"]): row for row in reproduced_monitors}
    for step_index, step in enumerate(SNAPSHOT_STEPS):
        rows = _rollout_metrics(bundle, candidate_physical[step_index])
        presence = [
            [continuous_presence_checks(rows[generation][prompt]) for prompt in range(2)]
            for generation in range(GENERATION_COUNT)
        ]
        invariance = [
            _prompt_invariance(
                candidate_physical[step_index, generation],
                np.asarray(bundle["verified_positive_mask"], bool),
            )
            for generation in range(GENERATION_COUNT)
        ]
        checks = rollout_step_checks(
            base_rows=base_rows,
            candidate_rows=rows,
            candidate_presence=presence,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=monitor_by_step[step]["candidate_v5_dense"],
        )
        pooled = pooled_object_metrics(rows)
        row = {
            "step": int(step),
            "base_rows": base_rows,
            "candidate_rows": rows,
            "candidate_presence": presence,
            "base_prompt_invariance": [float(value) for value in base_invariance],
            "candidate_prompt_invariance": [float(value) for value in invariance],
            "base_v5_dense": [float(value) for value in base_v5_dense],
            "candidate_v5_dense": [
                float(value) for value in monitor_by_step[step]["candidate_v5_dense"]
            ],
            "base_pooled": base_pooled,
            "candidate_pooled": pooled,
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        _finite_tree(row, "rollout step row")
        step_rows.append(row)
        print(
            "[ROLLOUT STEP {}] eligible={} pooled={}".format(
                step,
                row["eligible"],
                {
                    name: {
                        "recall": round(float(pooled[name]["soft_recall"]), 6),
                        "mae": round(float(pooled[name]["active_support_mae"]), 6),
                        "topk_diagnostic": round(float(pooled[name]["topk_overlap"]), 6),
                    }
                    for name in OBJECTS
                },
            ),
            flush=True,
        )

    eligible_order = rank_eligible_steps(step_rows)
    selected_step = eligible_order[0] if eligible_order else None
    arrays_file = output_dir / "early_rollout_k3_maps.npz"
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
        snapshot_steps=np.asarray(SNAPSHOT_STEPS, np.int64),
        generation_ids=np.arange(GENERATION_COUNT, dtype=np.int64),
        prompt_ids=np.asarray(PROMPT_IDS),
        seed_table=np.asarray(seeds, np.int64),
        frozen_base=base_physical,
        candidates=candidate_physical,
        frozen_base_normalized=base_normalized,
        candidates_normalized=candidate_normalized,
    )
    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v97_early_rollout_k3.py",
        "contract": PREPARE_ROOT / "relational_teacher_v97_early_rollout_k3_contract.py",
        "summarizer": PREPARE_ROOT / "summarize_relational_teacher_v97_early_rollout_k3.py",
        "failed_v91_summary": failed_file,
        "failed_v91_rollout_maps": Path(str(failed["paths"]["rollout_maps"])).resolve(),
        "preflight_report": preflight_file,
        "loss_response_report": response_file,
        "original_metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "rollout_metric_policy": rollout_policy_file,
        "dataset_index": index_file,
        "source_dataset_index": Path(str(preflight["paths"]["source_dataset_index"])).resolve(),
        "stats_file": stats_file,
        "v5_split": split_file,
        "v5_evidence_report": evidence_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "corrected_objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "v91_runner": PREPARE_ROOT / "run_relational_teacher_v91_corrected_one_scene_overfit.py",
        "rollout_maps": arrays_file,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}
    gate_checks = {
        "sealed_v91_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "exact_first_12_v91_updates_reproduced": True,
        "step_3_6_12_states_kept_in_memory_only": True,
        "paired_base_candidate_initial_and_reverse_noise": True,
        "watch_write_use_identical_randomness": True,
        "base_and_all_candidates_repeat_bitwise": True,
        "rollout_policy_written_before_optimizer": True,
        "exact_topk_is_diagnostic_only": ROLLOUT_POLICY["topk_role"]
        == "diagnostic_only_not_a_gate_due_discrete_one_point_quantization",
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "teacher_forward_is_text_plus_scene_only": tuple(sorted(kwargs))
        == tuple(sorted(FORWARD_INPUT_KEYS)),
        "no_model_checkpoint_saved": True,
        "at_least_one_early_step_is_admissible": selected_step is not None,
    }
    status = "PASS" if all(gate_checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": TRAINING_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "training_steps": TRAINING_STEPS,
        "learning_rate": LEARNING_RATE,
        "grad_clip": GRAD_CLIP,
        "lora": dict(lora_metadata(model)),
        "optimizer": {"name": "AdamW", "weight_decay": 0.0},
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": HELDOUT_TRAIN_SCENE,
        "development_scene_metadata_only": DEVELOPMENT_SCENE,
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "generation_count": GENERATION_COUNT,
        "prompt_ids": list(PROMPT_IDS),
        "prompt_text": {name: PROMPTS[name] for name in PROMPT_IDS},
        "forward_input_keys": sorted(kwargs),
        "seed_table": [list(value) for value in seeds],
        "reverse_diffusion_draw_count": 28,
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
        "preflight_binding_id": preflight["binding_id"],
        "loss_response_binding_id": response["binding_id"],
        "failed_v91_binding_id": failed["binding_id"],
        "scene_binding": preflight_binding,
        "rollout_metric_policy_id": ROLLOUT_POLICY_ID,
        "rollout_metric_policy_sha256": path_hashes["rollout_metric_policy"],
        "reproduced_monitor_panels": reproduced_monitors,
        "base_pooled": base_pooled,
        "step_rows": step_rows,
        "eligible_step_order": eligible_order,
        "selected_step": selected_step,
        "selection_policy": ROLLOUT_POLICY["selection_order"],
        "serialized_model_state": False,
        "paths": path_strings,
        "path_sha256": path_hashes,
        "rollout_maps_sha256": path_hashes["rollout_maps"],
        "checks": gate_checks,
        "failed_checks": sorted(name for name, passed in gate_checks.items() if not passed),
        "authorizes_locked_early_step_confirmation": status == "PASS",
        "authorizes_rollout_state_preflight": status == "FAIL",
        "authorizes_checkpoint": False,
        "authorizes_room_0102": False,
        "authorizes_development_evaluation": False,
        "authorizes_long_training": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "preflight_binding_id": report["preflight_binding_id"],
            "loss_response_binding_id": report["loss_response_binding_id"],
            "failed_v91_binding_id": report["failed_v91_binding_id"],
            "rollout_metric_policy_id": report["rollout_metric_policy_id"],
            "snapshot_steps": report["snapshot_steps"],
            "seed_table": report["seed_table"],
            "selected_step": report["selected_step"],
            "rollout_maps_sha256": report["rollout_maps_sha256"],
        }
    )
    _finite_tree(report, "Teacher-v9.7 report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise AssertionError("Teacher-v9.7 unexpectedly persisted model state")
    print("[EARLY_ROLLOUT_K3_{}] Teacher-v9.7 step 3/6/12".format(status))
    print("[PASS] exact v9.1 first-12 reproduction and in-memory snapshots")
    print("[PASS] 28 paired/repeated 500-step reverse-diffusion draws")
    print("[OK] selected step:", selected_step)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)
    del model, diffusion, snapshots
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

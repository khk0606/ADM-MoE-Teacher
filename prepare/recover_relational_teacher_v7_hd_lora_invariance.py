#!/usr/bin/env python3
"""Fresh invariance-weight recovery for Teacher-v7 High-Desk LoRA.

The run starts from the sealed v5r4 Teacher with a new zero-initialized LoRA.
Every update uses four balanced strata per train scene: Bed, the other Chair,
legacy Chair-06, and a genuine new ``hc_hd`` High-Desk Chair-06 row.  The
development scene remains metadata-only.  It reproduces the sealed failed v7
calibration and changes only the watch/write invariance weight from 0.5 to
0.75; the failed step-12 checkpoint is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
from preflight_relational_teacher_v7_hd_lora import (  # noqa: E402
    build_batch,
    compose_cdm_config,
    configure_reproducibility,
    load_stats,
    predict_xstart,
)
from relational_teacher_v6_contract import (  # noqa: E402
    POINT_COUNT,
    PROMPTS,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v6_semantics import (  # noqa: E402
    all_sittable_semantic_loss,
    frozen_teacher_preservation_loss,
    paired_prompt_invariance_loss,
)
from relational_teacher_v7_hd_preflight_contract import (  # noqa: E402
    DATASET_SCHEMA,
    DENSE_INDEX_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
    SCHEMA as PREFLIGHT_SCHEMA,
)
from relational_teacher_v7_hd_calibration_contract import (  # noqa: E402
    CANDIDATE_WEIGHT,
    DEFAULT_LR,
    LORA_WEIGHT,
    NEGATIVE_WEIGHT,
    NEW_DENSE_WEIGHT,
    PRESERVATION_WEIGHT,
    SCHEMA as CALIBRATION_SCHEMA,
    STEPS,
    TRAINING_STRATA_PER_SCENE,
    V5_REPLAY_WEIGHT,
    classify_training_stratum,
    group_relational_rows,
    select_relational_batch_rows,
    validate_batch_audits,
)
from relational_teacher_v7_hd_recovery_contract import (  # noqa: E402
    EXPECTED_WEIGHTS,
    INVARIANCE_WEIGHT,
    SCHEMA,
    validate_failed_near_miss,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v5-dataset-root", type=Path, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--failed-calibration-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    torch.save(value, temporary)
    os.replace(temporary, path)


def _resolve_dataset_path(dataset_root: Path, raw: object) -> Path:
    path = (dataset_root / str(raw)).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("relational dataset path escapes its root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_relational_rows(
    dataset_root: Path, index: Mapping[str, object]
) -> Dict[str, List[Dict[str, object]]]:
    records = index.get("scenes")
    if not isinstance(records, list):
        raise ValueError("relational dataset scenes are absent")
    train_records = [row for row in records if row.get("split") == "train"]
    development_records = [
        row for row in records if row.get("split") == "development"
    ]
    train_ids = tuple(sorted(str(row["scene_id"]) for row in train_records))
    development_ids = tuple(
        sorted(str(row["scene_id"]) for row in development_records)
    )
    if train_ids != EXPECTED_TRAIN_SCENES:
        raise ValueError("relational training scenes changed")
    if development_ids != EXPECTED_DEVELOPMENT_SCENES:
        raise ValueError("relational development scene changed")

    result: Dict[str, List[Dict[str, object]]] = {}
    for record in sorted(train_records, key=lambda value: str(value["scene_id"])):
        scene_id = str(record["scene_id"])
        points_path = _resolve_dataset_path(dataset_root, record["points_file"])
        sidecar_path = _resolve_dataset_path(dataset_root, record["sidecar_file"])
        if sha256_file(points_path) != str(record["points_sha256"]):
            raise ValueError(f"{scene_id}: points hash changed")
        if sha256_file(sidecar_path) != str(record["sidecar_sha256"]):
            raise ValueError(f"{scene_id}: sidecar hash changed")
        with np.load(points_path, allow_pickle=False) as source:
            points = source["points"].astype(np.float32)
        with np.load(sidecar_path, allow_pickle=False) as source:
            instance_ids = source["instance_ids"].astype(np.int64)
            category_ids = source["category_ids"].astype(np.int64)
            source_indices = source["source_indices"].astype(np.int64)
        if points.shape != (POINT_COUNT, 6):
            raise ValueError(f"{scene_id}: point shape changed")

        dense_index_path = _resolve_dataset_path(
            dataset_root, record["dense_index_file"]
        )
        if sha256_file(dense_index_path) != str(record["dense_index_sha256"]):
            raise ValueError(f"{scene_id}: top-level dense-index binding changed")
        dense_index = read_json(dense_index_path)
        if (
            dense_index.get("schema") != DENSE_INDEX_SCHEMA
            or dense_index.get("status") != "DENSE_CONTACT_V2_PASS"
            or dense_index.get("scene_id") != scene_id
            or dense_index.get("relation_or_distance_used") is not False
            or dense_index.get("motion_count") != 24
            or dense_index.get("prompt_expanded_row_count") != 48
            or dense_index.get("teacher_lora_training_authorized") is not False
        ):
            raise ValueError(f"{scene_id}: dense index is not sealed PASS")
        dense_rows = dense_index.get("rows")
        if not isinstance(dense_rows, list) or len(dense_rows) != 24:
            raise ValueError(f"{scene_id}: expected 24 dense rows")
        scene_rows: List[Dict[str, object]] = []
        for dense_row in dense_rows:
            manifest_path = _resolve_dataset_path(
                dataset_root, dense_row["manifest_file"]
            )
            if sha256_file(manifest_path) != str(dense_row["manifest_sha256"]):
                raise ValueError(f"{scene_id}: dense manifest hash changed")
            manifest = read_json(manifest_path)
            if (
                manifest.get("schema") != DENSE_INDEX_SCHEMA + "_item"
                or manifest.get("status") != "DENSE_CONTACT_ITEM_PASS"
                or manifest.get("relation_or_distance_used") is not False
                or manifest.get("compatible_prompt_ids") != sorted(PROMPTS)
                or manifest.get("motion_id") != dense_row.get("motion_id")
                or manifest.get("target_instance_id")
                != dense_row.get("target_instance_id")
            ):
                raise ValueError(f"{scene_id}: dense item contract changed")
            affordance_path = _resolve_dataset_path(
                dataset_root, manifest["affordance_file"]
            )
            if sha256_file(affordance_path) != str(
                manifest["affordance_sha256"]
            ):
                raise ValueError(f"{scene_id}: affordance hash changed")
            with np.load(affordance_path, allow_pickle=False) as source:
                affordance = source["affordance"].astype(np.float32)
                dense_instances = source["instance_ids"].astype(np.int64)
                dense_sources = source["source_indices"].astype(np.int64)
                sigma = float(source["sigma"])
            if affordance.shape != (POINT_COUNT, 6):
                raise ValueError(f"{scene_id}: affordance shape changed")
            if (
                not np.isfinite(affordance).all()
                or np.any(affordance < 0.0)
                or np.any(affordance > 1.0)
            ):
                raise ValueError(f"{scene_id}: affordance range changed")
            if not np.isclose(sigma, 0.8, rtol=0.0, atol=1e-7):
                raise ValueError(f"{scene_id}: sigma changed")
            if not np.array_equal(instance_ids, dense_instances):
                raise ValueError(f"{scene_id}: instance order changed")
            if not np.array_equal(source_indices, dense_sources):
                raise ValueError(f"{scene_id}: source order changed")
            motion_id = str(dense_row["motion_id"])
            target_instance_id = str(dense_row["target_instance_id"])
            is_high_desk = "_hc_hd_" in motion_id
            training_stratum = classify_training_stratum(
                motion_id, target_instance_id
            )
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "motion_id": motion_id,
                    "target_instance_id": target_instance_id,
                    "training_stratum": training_stratum,
                    "is_high_desk": is_high_desk,
                    "points": points,
                    "instance_ids": instance_ids,
                    "category_ids": category_ids,
                    "affordance": affordance,
                    "affordance_sha256": sha256_file(affordance_path),
                }
            )
        counts = Counter(str(row["target_instance_id"]) for row in scene_rows)
        if len(counts) != 3 or sorted(counts.values()) != [6, 6, 12]:
            raise ValueError(f"{scene_id}: target balance changed: {dict(counts)}")
        strata = Counter(str(row["training_stratum"]) for row in scene_rows)
        if len(strata) != TRAINING_STRATA_PER_SCENE or set(strata.values()) != {6}:
            raise ValueError(f"{scene_id}: four-way v7 strata changed: {dict(strata)}")
        result[scene_id] = sorted(
            scene_rows,
            key=lambda row: (str(row["target_instance_id"]), str(row["motion_id"])),
        )
    return result


def build_selected_relational_batch(
    selected: Sequence[Tuple[Mapping[str, object], str]],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> Dict[str, object]:
    # Reuse the preflight tensor builder one row at a time, then select the
    # requested prompt. This keeps exactly the same XYZ/RGB normalization.
    tensors = []
    for row, prompt_id in selected:
        probe = dict(row)
        pair = build_batch([probe], mean, std, device)
        index = sorted(PROMPTS).index(prompt_id)
        tensors.append((pair, index, row, prompt_id))
    return {
        "x": torch.stack([pair["x"][index] for pair, index, _, _ in tensors]),
        "xyz": torch.stack(
            [pair["xyz"][index] for pair, index, _, _ in tensors]
        ).contiguous(),
        "feat": torch.stack(
            [pair["feat"][index] for pair, index, _, _ in tensors]
        ).contiguous(),
        "text": [PROMPTS[prompt_id] for _, _, _, prompt_id in tensors],
        "instance_ids": torch.stack(
            [pair["instance_ids"][index] for pair, index, _, _ in tensors]
        ),
        "category_ids": torch.stack(
            [pair["category_ids"][index] for pair, index, _, _ in tensors]
        ),
        "scene_ids": [str(row["scene_id"]) for _, _, row, _ in tensors],
        "motion_ids": [str(row["motion_id"]) for _, _, row, _ in tensors],
        "prompt_ids": [prompt_id for _, _, _, prompt_id in tensors],
        "training_strata": [
            str(row["training_stratum"]) for _, _, row, _ in tensors
        ],
        "is_high_desk": torch.tensor(
            [bool(row["is_high_desk"]) for _, _, row, _ in tensors],
            dtype=torch.bool,
            device=device,
        ),
    }


def select_v5_batch_ids(
    rows: Mapping[str, Mapping[str, object]], step: int
) -> List[str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for sample_id, row in rows.items():
        grouped[str(row["target"])].append(sample_id)
    if set(grouped) != {"chair", "bed", "whiteboard"}:
        raise ValueError("v5 replay targets changed")
    selected = []
    for offset, target in enumerate(("chair", "bed", "whiteboard")):
        values = sorted(grouped[target])
        selected.append(values[(step - 1 + offset) % len(values)])
    return selected


def deterministic_noise(shape: torch.Size, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device)


def relational_timesteps(step: int, device: str) -> torch.Tensor:
    base = [50, 175, 325, 450]
    shift = (step - 1) % 4
    rotated = base[shift:] + base[:shift]
    main = rotated + rotated[1:] + rotated[:1]
    pair_a = 100 + 25 * ((step - 1) % 4)
    pair_b = 325 + 25 * ((step - 1) % 4)
    values = main + [pair_a, pair_a, pair_b, pair_b]
    if len(values) != 12:
        raise AssertionError("Teacher-v7 timestep batch changed")
    return torch.tensor(values, dtype=torch.long, device=device)


def relation_objective(
    prediction: torch.Tensor,
    frozen_prediction: torch.Tensor,
    batch: Mapping[str, object],
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    semantic = all_sittable_semantic_loss(
        prediction[:8],
        batch["instance_ids"][:8],
        batch["category_ids"][:8],
        mean_tensor,
        std_tensor,
    )
    candidate = semantic["candidate_any"] + semantic["candidate_pelvis"]
    negative = semantic["negative_any"] + semantic["negative_pelvis"]
    invariance = paired_prompt_invariance_loss(
        prediction[[8, 10]],
        prediction[[9, 11]],
        batch["category_ids"][[8, 10]],
    )
    preservation = frozen_teacher_preservation_loss(
        prediction,
        frozen_prediction,
        batch["category_ids"],
    )
    high_desk_mask = batch["is_high_desk"][:8]
    if int(high_desk_mask.sum().item()) != 2:
        raise ValueError("Teacher-v7 main batch must contain two High-Desk rows")
    dense_per_row = (prediction[:8] - batch["x"][:8]).square().mean(dim=(1, 2))
    return {
        "new_dense": dense_per_row.mean(),
        "high_desk_dense": dense_per_row[high_desk_mask].mean(),
        "semantic_total": semantic["total"],
        "candidate": candidate,
        "negative": negative,
        "invariance": invariance,
        "preservation": preservation,
    }


@torch.no_grad()
def fixed_panel(
    model,
    diffusion,
    grouped_relational,
    v5_rows,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    selected, _ = select_relational_batch_rows(grouped_relational, 1)
    relational = build_selected_relational_batch(selected, mean, std, device)
    timesteps = relational_timesteps(1, device)
    noise = deterministic_noise(relational["x"].shape, seed + 100, device)
    kwargs = {
        "c_pc_xyz": relational["xyz"],
        "c_pc_feat": relational["feat"],
        "c_text": relational["text"],
    }
    prediction = predict_xstart(
        model, diffusion, relational["x"], timesteps, kwargs, noise
    )
    set_lora_enabled(model, False)
    try:
        frozen = predict_xstart(
            model, diffusion, relational["x"], timesteps, kwargs, noise
        )
    finally:
        set_lora_enabled(model, True)
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(device)
    relation = relation_objective(
        prediction, frozen, relational, mean_tensor, std_tensor
    )

    replay_ids = select_v5_batch_ids(v5_rows, 1)
    replay = stack_v5_batch(v5_rows, replay_ids, device)
    replay_timestep = torch.tensor([100, 300, 450], device=device)
    replay_noise = deterministic_noise(replay["x"].shape, seed + 200, device)
    replay_prediction = predict_xstart(
        model,
        diffusion,
        replay["x"],
        replay_timestep,
        {
            "c_pc_xyz": replay["xyz"],
            "c_pc_feat": replay["feat"],
            "c_text": replay["text"],
        },
        replay_noise,
    )
    replay_dense = (replay_prediction - replay["x"]).square().mean()
    dense_combined = (
        NEW_DENSE_WEIGHT * relation["new_dense"]
        + V5_REPLAY_WEIGHT * replay_dense
    )
    proxy = (
        dense_combined
        + CANDIDATE_WEIGHT * relation["candidate"]
        + NEGATIVE_WEIGHT * relation["negative"]
        + INVARIANCE_WEIGHT * relation["invariance"]
        + PRESERVATION_WEIGHT * relation["preservation"]
        + LORA_WEIGHT * lora_parameter_energy(model)
    )
    values = {
        "new_dense": relation["new_dense"],
        "high_desk_dense": relation["high_desk_dense"],
        "v5_replay_dense": replay_dense,
        "dense_combined": dense_combined,
        "semantic_total": relation["semantic_total"],
        "candidate": relation["candidate"],
        "negative": relation["negative"],
        "invariance": relation["invariance"],
        "preservation": relation["preservation"],
        "lora_energy": lora_parameter_energy(model),
        "objective_proxy": proxy,
    }
    return {name: float(value.item()) for name, value in values.items()}


def main() -> None:
    args = parse_args()
    if args.steps != STEPS:
        raise ValueError("Teacher-v7 recovery is sealed to exactly 12 updates")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v7 recovery requires CUDA")
    if args.lr != DEFAULT_LR or args.grad_clip != 1.0:
        raise ValueError("Teacher-v7 recovery is sealed to lr=4e-5 and grad-clip=1")
    if args.lora_rank != 4 or args.lora_alpha != 8.0:
        raise ValueError("Teacher-v7 recovery is sealed to LoRA rank=4 alpha=8")
    configure_reproducibility(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    index_file = args.index.expanduser().resolve()
    v5_dataset_root = args.v5_dataset_root.expanduser().resolve()
    v5_split_file = args.v5_split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    v5_checkpoint = args.v5_checkpoint.expanduser().resolve()
    preflight_file = args.preflight_report.expanduser().resolve()
    failed_file = args.failed_calibration_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite recovery: " + str(output_dir))
    for path in (
        index_file,
        v5_split_file,
        stats_file,
        original_checkpoint,
        v5_checkpoint,
        preflight_file,
        failed_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    failed = read_json(failed_file)
    validate_failed_near_miss(failed, CALIBRATION_SCHEMA)
    if int(failed.get("seed", -1)) != args.seed:
        raise ValueError("Teacher-v7 recovery must reuse the failed calibration seed")
    failed_bound_inputs = {
        "preflight_report": preflight_file,
        "dataset_index": index_file,
        "v5_split": v5_split_file,
        "stats_file": stats_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
    }
    for name, path in failed_bound_inputs.items():
        if failed.get(name + "_sha256") != sha256_file(path):
            raise ValueError("Teacher-v7 recovery A/B input changed: " + name)
    for name in ("lora_state", "merged_checkpoint"):
        failed_artifact = Path(str(failed.get(name, ""))).expanduser().resolve()
        if (
            not failed_artifact.is_file()
            or sha256_file(failed_artifact) != failed.get(name + "_sha256")
        ):
            raise ValueError("sealed failed Teacher-v7 artifact changed: " + name)

    preflight = read_json(preflight_file)
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "PASS"
        or preflight.get("authorizes_12_update_calibration") is not True
        or preflight.get("authorizes_long_training") is not False
        or preflight.get("authorizes_development_evaluation") is not False
        or preflight.get("authorizes_paper_test") is not False
        or preflight.get("development_payloads_read") is not False
    ):
        raise ValueError("Teacher-v7 CUDA preflight is not authorizing")
    preflight_hashes = preflight.get("path_sha256")
    if not isinstance(preflight_hashes, dict):
        raise ValueError("Teacher-v7 preflight path binding is absent")
    for name, path in (
        ("dataset_index", index_file),
        ("stats_file", stats_file),
        ("original_checkpoint", original_checkpoint),
        ("v5_checkpoint", v5_checkpoint),
    ):
        if preflight_hashes.get(name) != sha256_file(path):
            raise ValueError("recovery input differs from preflight: " + name)

    index = read_json(index_file)
    if (
        index.get("schema") != DATASET_SCHEMA
        or index.get("status") != "DENSE_DATASET_PASS"
        or index.get("authorization") != "teacher_lora_v7_cuda_preflight_only"
        or index.get("split_scene_counts") != {"train": 2, "development": 1}
        or index.get("split_row_counts")
        != {"development": 48, "train": 96}
        or index.get("num_dense_motions") != 72
        or index.get("num_rows") != 144
        or index.get("heldout_dense_arrays_read_during_build") is not False
        or index.get("relation_or_distance_used_as_forward_input") is not False
        or index.get("target_instance_gt_is_supervision_only") is not True
    ):
        raise ValueError("Teacher-v7 dataset index changed")
    relational_rows = load_relational_rows(dataset_root, index)
    grouped_relational = group_relational_rows(relational_rows)
    if {scene: len(rows) for scene, rows in relational_rows.items()} != {
        "room_0101": 24,
        "room_0102": 24,
    }:
        raise ValueError("relational train motion counts changed")

    mean, std = load_stats(stats_file)
    v5_split = load_split(v5_split_file)
    v5_rows = load_v5_rows(
        v5_dataset_root,
        v5_split,
        "train",
        mean,
        std,
        4.0,
        16.0,
        0.7,
    )
    v5_counts = Counter(str(row["target"]) for row in v5_rows.values())
    if v5_counts != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise ValueError("sealed v5r4 train replay counts changed")

    cfg = compose_cdm_config(500, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    parity_selected, _ = select_relational_batch_rows(grouped_relational, 1)
    parity_batch = build_selected_relational_batch(
        parity_selected, mean, std, args.device
    )
    parity_timestep = relational_timesteps(1, args.device)
    parity_noise = deterministic_noise(
        parity_batch["x"].shape, args.seed + 300, args.device
    )
    parity_kwargs = {
        "c_pc_xyz": parity_batch["xyz"],
        "c_pc_feat": parity_batch["feat"],
        "c_text": parity_batch["text"],
    }
    with torch.no_grad():
        before = predict_xstart(
            model,
            diffusion,
            parity_batch["x"],
            parity_timestep,
            parity_kwargs,
            parity_noise,
        )
    install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    if len(lora_metadata(model).get("module_names", [])) != 31:
        raise AssertionError("Teacher-v7 expected exactly 31 LoRA modules")
    with torch.no_grad():
        after = predict_xstart(
            model,
            diffusion,
            parity_batch["x"],
            parity_timestep,
            parity_kwargs,
            parity_noise,
        )
    if not torch.equal(before, after):
        raise AssertionError("fresh Teacher-v7 zero-init parity failed")

    optimizer = torch.optim.AdamW(list(named_lora.values()), lr=args.lr)
    initial_panel = fixed_panel(
        model,
        diffusion,
        grouped_relational,
        v5_rows,
        mean,
        std,
        args.device,
        args.seed + 10000,
    )
    failed_initial = failed.get("initial_fixed_panel")
    ab_panel_keys = {
        "new_dense",
        "high_desk_dense",
        "v5_replay_dense",
        "dense_combined",
        "semantic_total",
        "candidate",
        "negative",
        "invariance",
        "preservation",
        "lora_energy",
    }
    if not isinstance(failed_initial, dict) or any(
        initial_panel.get(key) != failed_initial.get(key) for key in ab_panel_keys
    ):
        raise ValueError("Teacher-v7 recovery did not reproduce fresh initialization")
    logs = []
    batch_audits = []
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)
    frozen_gradient_count = 0

    for step in range(1, STEPS + 1):
        set_frozen_base_eval_lora_train(model)
        selected, batch_audit = select_relational_batch_rows(
            grouped_relational, step
        )
        relation_batch = build_selected_relational_batch(
            selected, mean, std, args.device
        )
        relation_timestep = relational_timesteps(step, args.device)
        relation_noise = deterministic_noise(
            relation_batch["x"].shape, args.seed + step * 1009, args.device
        )
        relation_kwargs = {
            "c_pc_xyz": relation_batch["xyz"],
            "c_pc_feat": relation_batch["feat"],
            "c_text": relation_batch["text"],
        }
        prediction = predict_xstart(
            model,
            diffusion,
            relation_batch["x"],
            relation_timestep,
            relation_kwargs,
            relation_noise,
        )
        set_lora_enabled(model, False)
        try:
            with torch.no_grad():
                frozen_prediction = predict_xstart(
                    model,
                    diffusion,
                    relation_batch["x"],
                    relation_timestep,
                    relation_kwargs,
                    relation_noise,
                )
        finally:
            set_lora_enabled(model, True)
        relation = relation_objective(
            prediction,
            frozen_prediction,
            relation_batch,
            mean_tensor,
            std_tensor,
        )

        replay_ids = select_v5_batch_ids(v5_rows, step)
        replay_batch = stack_v5_batch(v5_rows, replay_ids, args.device)
        replay_timestep = torch.tensor(
            [100, 300, 450], dtype=torch.long, device=args.device
        )
        replay_noise = deterministic_noise(
            replay_batch["x"].shape, args.seed + 50000 + step * 1013, args.device
        )
        replay_prediction = predict_xstart(
            model,
            diffusion,
            replay_batch["x"],
            replay_timestep,
            {
                "c_pc_xyz": replay_batch["xyz"],
                "c_pc_feat": replay_batch["feat"],
                "c_text": replay_batch["text"],
            },
            replay_noise,
        )
        replay_dense = (replay_prediction - replay_batch["x"]).square().mean()
        dense_combined = (
            NEW_DENSE_WEIGHT * relation["new_dense"]
            + V5_REPLAY_WEIGHT * replay_dense
        )
        regularizer = lora_parameter_energy(model)
        total = (
            dense_combined
            + CANDIDATE_WEIGHT * relation["candidate"]
            + NEGATIVE_WEIGHT * relation["negative"]
            + INVARIANCE_WEIGHT * relation["invariance"]
            + PRESERVATION_WEIGHT * relation["preservation"]
            + LORA_WEIGHT * regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        frozen_gradients = [
            name
            for name, parameter in model.named_parameters()
            if "lora_" not in name and parameter.grad is not None
        ]
        frozen_gradient_count += len(frozen_gradients)
        if frozen_gradients:
            raise AssertionError("gradient reached frozen CDM during recovery")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(named_lora.values()), args.grad_clip
        )
        gradient_value = float(gradient_norm.item())
        if not math.isfinite(gradient_value) or gradient_value <= 0.0:
            raise RuntimeError("invalid Teacher-v7 recovery gradient")
        optimizer.step()
        log = {
            "step": step,
            "relational_noise_seed": args.seed + step * 1009,
            "v5_replay_noise_seed": args.seed + 50000 + step * 1013,
            "relational_timesteps": relation_timestep.detach().cpu().tolist(),
            "v5_replay_timesteps": replay_timestep.detach().cpu().tolist(),
            "total": float(total.detach().item()),
            "dense_combined": float(dense_combined.detach().item()),
            "new_dense": float(relation["new_dense"].detach().item()),
            "high_desk_dense": float(
                relation["high_desk_dense"].detach().item()
            ),
            "v5_replay_dense": float(replay_dense.detach().item()),
            "candidate": float(relation["candidate"].detach().item()),
            "negative": float(relation["negative"].detach().item()),
            "invariance": float(relation["invariance"].detach().item()),
            "preservation": float(relation["preservation"].detach().item()),
            "lora_energy": float(regularizer.detach().item()),
            "gradient_l2_before_clip": gradient_value,
            "v5_replay_ids": replay_ids,
        }
        if not all(
            math.isfinite(float(value))
            for key, value in log.items()
            if key
            not in {
                "step",
                "relational_noise_seed",
                "v5_replay_noise_seed",
                "relational_timesteps",
                "v5_replay_timesteps",
                "v5_replay_ids",
            }
        ):
            raise RuntimeError("non-finite Teacher-v7 recovery metric")
        logs.append(log)
        batch_audit["v5_replay_ids"] = replay_ids
        batch_audits.append(batch_audit)
        print(
            f"[RECOVERY] step={step:02d}/12 total={log['total']:.6f} "
            f"high_desk={log['high_desk_dense']:.6f} "
            f"semantic={float(relation['semantic_total'].detach().item()):.6f} "
            f"invariance={log['invariance']:.6f} grad={gradient_value:.6f}"
        )

    final_panel = fixed_panel(
        model,
        diffusion,
        grouped_relational,
        v5_rows,
        mean,
        std,
        args.device,
        args.seed + 10000,
    )
    if batch_audits != failed.get("batch_audits"):
        raise ValueError("Teacher-v7 recovery batches differ from sealed A/B source")
    output_dir.mkdir(parents=True, exist_ok=False)
    lora_state_file = output_dir / "lora_state_step12.pt"
    merged_file = output_dir / "merged_teacher_step12.pt"
    atomic_torch_save(
        {
            "schema": SCHEMA + "_diagnostic_lora_state",
            "step": STEPS,
            "seed": args.seed,
            "lora": {
                name: parameter.detach().cpu()
                for name, parameter in named_lora.items()
            },
        },
        lora_state_file,
    )
    temporary_merged = merged_file.with_name(
        merged_file.name + ".tmp." + str(os.getpid())
    )
    save_merged_legacy_state(model, temporary_merged)
    os.replace(temporary_merged, merged_file)

    eps = 1e-12
    checks = {
        "exact_12_updates": len(logs) == STEPS,
        **validate_batch_audits(batch_audits),
        "every_update_uses_v5_chair_bed_whiteboard_replay": all(
            len(row["v5_replay_ids"]) == 3 for row in batch_audits
        ),
        "development_payloads_unread": True,
        "frozen_cdm_has_no_gradient": frozen_gradient_count == 0,
        "all_gradients_finite_and_positive": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in logs
        ),
        "fixed_semantic_total_improves": final_panel["semantic_total"]
        < initial_panel["semantic_total"] - eps,
        "fixed_high_desk_dense_improves": final_panel["high_desk_dense"]
        < initial_panel["high_desk_dense"] - eps,
        "fixed_watch_write_invariance_not_worse": final_panel["invariance"]
        <= initial_panel["invariance"] * 1.02 + eps,
        "fixed_new_dense_not_worse": final_panel["new_dense"]
        <= initial_panel["new_dense"] * 1.02 + eps,
        "fixed_negative_suppression_not_worse": final_panel["negative"]
        <= initial_panel["negative"] * 1.02 + eps,
        "fixed_v5_replay_retained": final_panel["v5_replay_dense"]
        <= initial_panel["v5_replay_dense"] * 1.01 + eps,
        "fixed_preservation_bounded": final_panel["preservation"] <= 1e-3,
        "lora_changed_from_zero": final_panel["lora_energy"] > 0.0,
    }
    status = "RECOVERY_PASS" if all(checks.values()) else "RECOVERY_FAIL"
    source_paths = {
        "recovery": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v7_hd_lora_invariance_recovery.py",
        "preflight_source": PREPARE_ROOT
        / "preflight_relational_teacher_v7_hd_lora.py",
        "preflight_contract_source": PREPARE_ROOT
        / "relational_teacher_v7_hd_preflight_contract.py",
        "calibration_contract_source": PREPARE_ROOT
        / "relational_teacher_v7_hd_calibration_contract.py",
        "recovery_contract_source": PREPARE_ROOT
        / "relational_teacher_v7_hd_recovery_contract.py",
        "lora_source": PREPARE_ROOT / "fewshot_cdm_lora.py",
        "contract_source": PREPARE_ROOT / "relational_teacher_v6_contract.py",
        "semantics_source": PREPARE_ROOT / "relational_teacher_v6_semantics.py",
        "dataset_validator_source": PREPARE_ROOT
        / "validate_relational_teacher_v7_hd_dataset.py",
    }
    payload = {
        "schema": SCHEMA,
        "status": status,
        "steps": STEPS,
        "seed": args.seed,
        "device": args.device,
        "learning_rate": args.lr,
        "gradient_clip": args.grad_clip,
        "train_scenes": list(EXPECTED_TRAIN_SCENES),
        "development_scenes_metadata_only": list(EXPECTED_DEVELOPMENT_SCENES),
        "development_payloads_read": False,
        "relational_motion_counts": {scene: 24 for scene in EXPECTED_TRAIN_SCENES},
        "high_desk_motion_counts": {scene: 6 for scene in EXPECTED_TRAIN_SCENES},
        "training_strata_per_scene": TRAINING_STRATA_PER_SCENE,
        "v5_replay_target_counts": dict(v5_counts),
        "objective_weights": EXPECTED_WEIGHTS,
        "initial_fixed_panel": initial_panel,
        "final_fixed_panel": final_panel,
        "logs": logs,
        "batch_audits": batch_audits,
        "frozen_parameter_gradient_count": frozen_gradient_count,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "fresh_zero_init_from_sealed_v5r4": True,
        "prior_v6_lora_checkpoint_loaded": False,
        "prior_failed_v7_checkpoint_loaded": False,
        "same_seed_batches_noise_timesteps_as_failed_v7": True,
        "single_changed_hyperparameter": {
            "name": "watch_write_invariance",
            "failed_value": 0.5,
            "recovery_value": INVARIANCE_WEIGHT,
        },
        "failed_calibration_summary": str(failed_file),
        "failed_calibration_summary_sha256": sha256_file(failed_file),
        "failed_calibration_lora_state_sha256": failed["lora_state_sha256"],
        "failed_calibration_merged_checkpoint_sha256": failed[
            "merged_checkpoint_sha256"
        ],
        "preflight_report": str(preflight_file),
        "preflight_report_sha256": sha256_file(preflight_file),
        "dataset_index": str(index_file),
        "dataset_index_sha256": sha256_file(index_file),
        "v5_split": str(v5_split_file),
        "v5_split_sha256": sha256_file(v5_split_file),
        "stats_file": str(stats_file),
        "stats_file_sha256": sha256_file(stats_file),
        "original_checkpoint": str(original_checkpoint),
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "v5_checkpoint": str(v5_checkpoint),
        "v5_checkpoint_sha256": sha256_file(v5_checkpoint),
        "lora_metadata": dict(lora_metadata(model)),
        "lora_state": str(lora_state_file),
        "lora_state_sha256": sha256_file(lora_state_file),
        "merged_checkpoint": str(merged_file),
        "merged_checkpoint_sha256": sha256_file(merged_file),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "diagnostic_checkpoint_only": True,
        "authorizes_bounded_train_only_pilot": status == "RECOVERY_PASS",
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, payload)
    print(f"[{status}] relational Teacher-v7 High-Desk invariance recovery")
    print("[PASS] failed v7 source remained immutable; no failed checkpoint loaded")
    print("[PASS] same seed/batches/noise/timesteps; only lambda_invariance=0.75")
    print("[OK] initial fixed panel:", initial_panel)
    print("[OK] final fixed panel:", final_panel)
    print("[OK] failed checks:", payload["failed_checks"])
    print("[OK] summary:", summary_file)


if __name__ == "__main__":
    main()

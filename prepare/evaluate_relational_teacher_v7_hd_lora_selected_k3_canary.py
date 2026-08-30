#!/usr/bin/env python3
"""Train-only K=3 canary for the selected Teacher-v7 step-12 checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_relational_teacher_v7_hd_lora_rollout_canary import SCHEMA as K1_SCHEMA
from fewshot_cdm_common import load_split
from preflight_relational_teacher_v7_hd_lora import (
    compose_cdm_config,
    configure_reproducibility,
    load_stats,
)
from recover_relational_teacher_v7_hd_lora_invariance import load_relational_rows
from relational_teacher_v6_contract import PROMPTS, atomic_write_json, read_json, sha256_file
from relational_teacher_v7_hd_calibration_contract import group_relational_rows
from relational_teacher_v7_hd_k3_common import (
    generation_seeds,
    per_generation_metrics,
    pooled_metrics,
)
from relational_teacher_v7_hd_k3_contract import (
    DIFFUSION_STEPS,
    GENERATIONS,
    RELATIONAL_CASES,
    SCHEMA,
    SEED,
    SEED_POLICY,
    SELECTED_STEP,
    V5_CASES,
    k3_checks,
)
from relational_teacher_v7_hd_rollout_common import (
    create_model,
    denormalize,
    sample_contact,
)
from train_fewshot_cdm import load_rows as load_v5_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS)
    parser.add_argument("--num-generations", type=int, default=GENERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def verify_source_binding(value: dict, label: str) -> None:
    paths, hashes = value.get("source_paths"), value.get("source_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict) or set(paths) != set(hashes):
        raise ValueError(label + " source binding is absent")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(hashes[name]):
            raise ValueError(label + " source changed: " + str(name))


def bound_file(value: dict, key: str) -> Path:
    path = Path(str(value.get(key, ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != str(value.get(key + "_sha256", "")):
        raise ValueError("Teacher-v7 K=1 bound file changed: " + key)
    return path


def main() -> None:
    args = parse_args()
    if args.diffusion_steps != DIFFUSION_STEPS or args.num_generations != GENERATIONS:
        raise ValueError("Teacher-v7 selected canary is sealed to 500-step K=3")
    if args.seed != SEED:
        raise ValueError("Teacher-v7 selected K=3 seed is sealed to 20260911")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v7 selected K=3 requires CUDA")
    configure_reproducibility(args.seed)

    k1_file = args.k1_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v7 selected K=3")
    if not k1_file.is_file():
        raise FileNotFoundError(k1_file)
    k1 = read_json(k1_file)
    if (k1.get("schema") != K1_SCHEMA or k1.get("status") != "CANARY_PASS"
            or k1.get("failed_checks") != []
            or int(k1.get("selected_step", -1)) != SELECTED_STEP
            or int(k1.get("seed", -1)) != SEED
            or k1.get("authorizes_train_only_k3_canary") is not True
            or k1.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 K=1 canary does not authorize K=3")
    verify_source_binding(k1, "Teacher-v7 K=1 canary")
    input_paths = {key: bound_file(k1, key) for key in (
        "dataset_index", "v5_split", "stats_file", "original_checkpoint",
        "v5_checkpoint", "candidate_checkpoint", "predictions",
    )}

    with np.load(input_paths["predictions"], allow_pickle=False) as source:
        reference = {name: source[name] for name in source.files}
    if reference["rel_base"].shape != (RELATIONAL_CASES, 8192, 6):
        raise ValueError("Teacher-v7 K=1 relational map shape changed")
    if reference["v5_base"].shape != (V5_CASES, 8192, 6):
        raise ValueError("Teacher-v7 K=1 v5 map shape changed")

    dataset_root = input_paths["dataset_index"].parent
    v5_dataset_root = input_paths["v5_split"].parent.parent
    mean, std = load_stats(input_paths["stats_file"])
    index = read_json(input_paths["dataset_index"])
    grouped = group_relational_rows(load_relational_rows(dataset_root, index))
    v5_rows = load_v5_rows(
        v5_dataset_root, load_split(input_paths["v5_split"]), "train",
        mean, std, 4.0, 16.0, 0.7,
    )

    rel_scene_ids = [str(value) for value in reference["rel_scene_ids"].tolist()]
    rel_strata = [str(value) for value in reference["rel_strata"].tolist()]
    rel_prompt_ids = [str(value) for value in reference["rel_prompt_ids"].tolist()]
    rel_case_ids = [str(value) for value in reference["rel_case_ids"].tolist()]
    v5_ids = [str(value) for value in reference["v5_ids"].tolist()]
    v5_targets = [str(value) for value in reference["v5_targets"].tolist()]
    row_lookup = {
        (scene_id, stratum): rows[0]
        for scene_id, strata in grouped.items()
        for stratum, rows in strata.items()
    }

    relational_base = np.empty((RELATIONAL_CASES, GENERATIONS, 8192, 6), dtype=np.float32)
    relational_candidate = np.empty_like(relational_base)
    v5_base = np.empty((V5_CASES, GENERATIONS, 8192, 6), dtype=np.float32)
    v5_candidate = np.empty_like(v5_base)
    relational_base[:, 0] = reference["rel_base"]
    relational_candidate[:, 0] = reference["rel_candidate"]
    v5_base[:, 0] = reference["v5_base"]
    v5_candidate[:, 0] = reference["v5_candidate"]
    relational_seeds = np.zeros((RELATIONAL_CASES, GENERATIONS, 2), dtype=np.int64)
    v5_seeds = np.zeros((V5_CASES, GENERATIONS, 2), dtype=np.int64)

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    base_model, base_diffusion = create_model(
        cfg, input_paths["original_checkpoint"], input_paths["v5_checkpoint"],
        None, args.device,
    )
    candidate_model, candidate_diffusion = create_model(
        cfg, input_paths["original_checkpoint"], input_paths["v5_checkpoint"],
        input_paths["candidate_checkpoint"], args.device,
    )
    repeatability = {"base": None, "candidate": None}
    for index_value, (scene_id, stratum, prompt_id, case_id) in enumerate(
        zip(rel_scene_ids, rel_strata, rel_prompt_ids, rel_case_ids)
    ):
        if (scene_id, stratum) not in row_lookup or prompt_id not in PROMPTS:
            raise ValueError("Teacher-v7 K=3 relational lookup changed")
        if case_id != f"{scene_id}|{stratum}|{prompt_id}":
            raise ValueError("Teacher-v7 K=3 relational case order changed")
        row = row_lookup[(scene_id, stratum)]
        points = np.asarray(row["points"], dtype=np.float32)
        pair_id = f"{scene_id}|{stratum}"
        for generation in range(GENERATIONS):
            initial_seed, reverse_seed = generation_seeds(
                args.seed, "relational_pair", pair_id, generation
            )
            relational_seeds[index_value, generation] = (initial_seed, reverse_seed)
            if generation == 0:
                continue
            base_norm = sample_contact(
                base_model, base_diffusion, points[:, :3], points[:, 3:6] / 255.0,
                PROMPTS[prompt_id], initial_seed, reverse_seed, args.device,
                not args.no_progress,
            )
            candidate_norm = sample_contact(
                candidate_model, candidate_diffusion, points[:, :3], points[:, 3:6] / 255.0,
                PROMPTS[prompt_id], initial_seed, reverse_seed, args.device,
                not args.no_progress,
            )
            if repeatability["base"] is None:
                repeated = sample_contact(
                    base_model, base_diffusion, points[:, :3], points[:, 3:6] / 255.0,
                    PROMPTS[prompt_id], initial_seed, reverse_seed, args.device, False,
                )
                repeatability["base"] = bool(np.array_equal(base_norm, repeated))
            if repeatability["candidate"] is None:
                repeated = sample_contact(
                    candidate_model, candidate_diffusion, points[:, :3], points[:, 3:6] / 255.0,
                    PROMPTS[prompt_id], initial_seed, reverse_seed, args.device, False,
                )
                repeatability["candidate"] = bool(np.array_equal(candidate_norm, repeated))
            relational_base[index_value, generation] = denormalize(base_norm, mean, std)
            relational_candidate[index_value, generation] = denormalize(candidate_norm, mean, std)
            print(f"[REL {index_value + 1:02d}/{RELATIONAL_CASES} GEN {generation + 1}/3] {case_id}")

    for index_value, (sample_id, target) in enumerate(zip(v5_ids, v5_targets)):
        if sample_id not in v5_rows or str(v5_rows[sample_id]["target"]) != target:
            raise ValueError("Teacher-v7 K=3 v5 lookup changed")
        row = v5_rows[sample_id]
        for generation in range(GENERATIONS):
            initial_seed, reverse_seed = generation_seeds(
                args.seed, "v5_replay", sample_id, generation
            )
            v5_seeds[index_value, generation] = (initial_seed, reverse_seed)
            if generation == 0:
                continue
            base_norm = sample_contact(
                base_model, base_diffusion, np.asarray(row["xyz"], dtype=np.float32),
                np.asarray(row["feat"], dtype=np.float32), str(row["text"]),
                initial_seed, reverse_seed, args.device, not args.no_progress,
            )
            candidate_norm = sample_contact(
                candidate_model, candidate_diffusion, np.asarray(row["xyz"], dtype=np.float32),
                np.asarray(row["feat"], dtype=np.float32), str(row["text"]),
                initial_seed, reverse_seed, args.device, not args.no_progress,
            )
            v5_base[index_value, generation] = denormalize(base_norm, mean, std)
            v5_candidate[index_value, generation] = denormalize(candidate_norm, mean, std)
            print(f"[V5 {index_value + 1:02d}/{V5_CASES} GEN {generation + 1}/3] {sample_id}")

    if repeatability != {"base": True, "candidate": True}:
        raise RuntimeError("Teacher-v7 K=3 reverse diffusion is not repeatable")
    if not np.array_equal(relational_base[:, 0], reference["rel_base"]):
        raise AssertionError("Teacher-v7 K=1 relational Base maps changed")
    if not np.array_equal(relational_candidate[:, 0], reference["rel_candidate"]):
        raise AssertionError("Teacher-v7 K=1 relational candidate maps changed")
    if not np.array_equal(v5_base[:, 0], reference["v5_base"]):
        raise AssertionError("Teacher-v7 K=1 v5 Base maps changed")
    if not np.array_equal(v5_candidate[:, 0], reference["v5_candidate"]):
        raise AssertionError("Teacher-v7 K=1 v5 candidate maps changed")

    generation_metrics = per_generation_metrics(
        reference, relational_base, relational_candidate, v5_base, v5_candidate
    )
    pooled = pooled_metrics(generation_metrics)
    checks = k3_checks(pooled, generation_metrics, True)
    checks.update({
        "generation_0_exactly_reuses_sealed_k1": True,
        "all_16_relational_and_3_v5_cases_have_three_generations": (
            relational_base.shape[:2] == (RELATIONAL_CASES, GENERATIONS)
            and v5_base.shape[:2] == (V5_CASES, GENERATIONS)
        ),
        "development_payloads_unread": True,
    })
    status = "SELECTED_K3_PASS" if all(checks.values()) else "SELECTED_K3_FAIL"

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_file = output_dir / "k3_predictions.npz"
    temporary = bundle_file.with_name(bundle_file.name + ".tmp." + str(os.getpid()) + ".npz")
    np.savez_compressed(
        temporary, relational_base=relational_base,
        relational_candidate=relational_candidate,
        relational_seeds=relational_seeds, v5_base=v5_base,
        v5_candidate=v5_candidate, v5_seeds=v5_seeds,
    )
    os.replace(temporary, bundle_file)
    prepare_root = Path(__file__).resolve().parent
    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "validator": prepare_root / "validate_relational_teacher_v7_hd_lora_selected_k3_canary.py",
        "k3_common": prepare_root / "relational_teacher_v7_hd_k3_common.py",
        "k3_contract": prepare_root / "relational_teacher_v7_hd_k3_contract.py",
    }
    summary = {
        "schema": SCHEMA,
        "status": status,
        "partition": "train_canary",
        "selected_step": SELECTED_STEP,
        "diffusion_steps": args.diffusion_steps,
        "num_generations": GENERATIONS,
        "seed": args.seed,
        "seed_policy": SEED_POLICY,
        "paired_initial_and_reverse_noise": True,
        "generation_0_reused_from_k1": True,
        "new_generation_indices": [1, 2],
        "relational_case_count": RELATIONAL_CASES,
        "high_desk_case_count_per_generation": 4,
        "v5_case_count": V5_CASES,
        "paired_evaluation_reverse_draws_generated": 76,
        "repeatability_reverse_draws_generated": 2,
        "total_reverse_draws_generated": 78,
        "repeatability": repeatability,
        "per_generation_metrics": generation_metrics,
        "pooled_metrics": pooled,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "development_payloads_read": False,
        "k1_summary": str(k1_file),
        "k1_summary_sha256": sha256_file(k1_file),
        "k1_predictions": str(input_paths["predictions"]),
        "k1_predictions_sha256": sha256_file(input_paths["predictions"]),
        "candidate_checkpoint": str(input_paths["candidate_checkpoint"]),
        "candidate_checkpoint_sha256": sha256_file(input_paths["candidate_checkpoint"]),
        "dataset_index": str(input_paths["dataset_index"]),
        "dataset_index_sha256": sha256_file(input_paths["dataset_index"]),
        "v5_split": str(input_paths["v5_split"]),
        "v5_split_sha256": sha256_file(input_paths["v5_split"]),
        "stats_file": str(input_paths["stats_file"]),
        "stats_file_sha256": sha256_file(input_paths["stats_file"]),
        "original_checkpoint": str(input_paths["original_checkpoint"]),
        "original_checkpoint_sha256": sha256_file(input_paths["original_checkpoint"]),
        "v5_checkpoint": str(input_paths["v5_checkpoint"]),
        "v5_checkpoint_sha256": sha256_file(input_paths["v5_checkpoint"]),
        "predictions": str(bundle_file),
        "predictions_sha256": sha256_file(bundle_file),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "diagnostic_only": True,
        "authorizes_train_only_full_k3": status == "SELECTED_K3_PASS",
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, summary)
    print(f"[{status}] Teacher-v7 selected step-12 train-only K=3 canary")
    for generation, metrics in enumerate(generation_metrics):
        overall, high_desk = metrics["overall"], metrics["high_desk"]
        print(
            f"[GEN {generation}] target={overall['target_mae_relative_change']:.6%} "
            f"high_desk={high_desk['target_mae_relative_change']:.6%} "
            f"invariance={overall['prompt_invariance_mse_candidate']:.9f} "
            f"v5={overall['v5_replay_relative_degradation']:.6%}"
        )
    print("[POOLED]", pooled["overall"])
    print("[POOLED HIGH DESK]", pooled["high_desk"])
    print("[OK] failed checks:", summary["failed_checks"])
    print("[OK] summary:", summary_file)


if __name__ == "__main__":
    main()

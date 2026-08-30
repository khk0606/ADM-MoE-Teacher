#!/usr/bin/env python3
"""Full train-only K=3 evaluation for the selected Teacher-v7 checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from fewshot_cdm_common import load_split
from preflight_relational_teacher_v7_hd_lora import (
    compose_cdm_config,
    configure_reproducibility,
    load_stats,
)
from recover_relational_teacher_v7_hd_lora_invariance import load_relational_rows
from relational_teacher_v6_contract import PROMPTS, atomic_write_json, read_json, sha256_file
from relational_teacher_v7_hd_calibration_contract import group_relational_rows
from relational_teacher_v7_hd_full_k3_common import (
    generation_seeds,
    per_generation_metrics,
    pooled_metrics,
)
from relational_teacher_v7_hd_full_k3_contract import (
    DIFFUSION_STEPS,
    GENERATIONS,
    HIGH_DESK_CASES_PER_GENERATION,
    RELATIONAL_CASES,
    SCHEMA,
    SEED,
    SEED_POLICY,
    SELECTED_STEP,
    TRAIN_MOTIONS,
    V5_CASES,
    full_checks,
)
from relational_teacher_v7_hd_k3_contract import SCHEMA as K3_SCHEMA
from relational_teacher_v7_hd_preflight_contract import EXPECTED_TRAIN_SCENES
from relational_teacher_v7_hd_rollout_common import create_model, denormalize, sample_contact
from train_fewshot_cdm import load_rows as load_v5_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3-canary-summary", type=Path, required=True)
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
        raise ValueError("Teacher-v7 K=3 bound file changed: " + key)
    return path


def build_reference(dataset_root: Path, index: dict, grouped: dict, v5_rows: dict):
    scene_records = {
        str(row["scene_id"]): row for row in index.get("scenes", [])
        if row.get("split") == "train"
    }
    instance_name_to_id = {}
    for scene_id in EXPECTED_TRAIN_SCENES:
        record = scene_records[scene_id]
        instances_file = (dataset_root / str(record["instances_file"])).resolve()
        if sha256_file(instances_file) != str(record["instances_sha256"]):
            raise ValueError(scene_id + ": instances hash changed")
        manifest = read_json(instances_file)
        instance_name_to_id[scene_id] = {
            str(row["name"]).lower(): int(row["instance_id"])
            for row in manifest["objects"]
        }

    output = {
        "rel_case_ids": [], "rel_scene_ids": [], "rel_target_names": [],
        "rel_strata": [], "rel_motion_ids": [], "rel_prompt_ids": [],
        "rel_target_instance_ids": [], "rel_gt": [], "rel_instance_ids": [],
        "rel_category_ids": [], "v5_ids": [], "v5_targets": [], "v5_gt": [],
    }
    rel_rows = []
    for scene_id in EXPECTED_TRAIN_SCENES:
        for stratum, rows in sorted(grouped[scene_id].items()):
            for row in rows:
                motion_id = str(row["motion_id"])
                target_name = str(row["target_instance_id"])
                target_key = target_name.lower()
                if target_key not in instance_name_to_id[scene_id]:
                    raise ValueError(scene_id + ": target instance name is unresolved")
                for prompt_id in sorted(PROMPTS):
                    output["rel_case_ids"].append(
                        f"{scene_id}|{stratum}|{motion_id}|{prompt_id}"
                    )
                    output["rel_scene_ids"].append(scene_id)
                    output["rel_target_names"].append(target_name)
                    output["rel_strata"].append(stratum)
                    output["rel_motion_ids"].append(motion_id)
                    output["rel_prompt_ids"].append(prompt_id)
                    output["rel_target_instance_ids"].append(
                        instance_name_to_id[scene_id][target_key]
                    )
                    output["rel_gt"].append(np.asarray(row["affordance"], dtype=np.float32))
                    output["rel_instance_ids"].append(
                        np.asarray(row["instance_ids"], dtype=np.int64)
                    )
                    output["rel_category_ids"].append(
                        np.asarray(row["category_ids"], dtype=np.int64)
                    )
                    rel_rows.append(row)

    v5_order = sorted(v5_rows)
    for sample_id in v5_order:
        row = v5_rows[sample_id]
        output["v5_ids"].append(sample_id)
        output["v5_targets"].append(str(row["target"]))
        output["v5_gt"].append(np.asarray(row["gt"], dtype=np.float32))

    if len(rel_rows) != RELATIONAL_CASES or len(v5_order) != V5_CASES:
        raise ValueError("Teacher-v7 full train inventory changed")
    if len(set(zip(output["rel_scene_ids"], output["rel_motion_ids"]))) != TRAIN_MOTIONS:
        raise ValueError("Teacher-v7 full train motion count changed")
    reference = {
        key: np.asarray(value) for key, value in output.items()
        if key not in {"rel_gt", "rel_instance_ids", "rel_category_ids", "v5_gt"}
    }
    reference.update({
        "rel_target_instance_ids": np.asarray(
            output["rel_target_instance_ids"], dtype=np.int64
        ),
        "rel_gt": np.stack(output["rel_gt"]).astype(np.float32),
        "rel_instance_ids": np.stack(output["rel_instance_ids"]).astype(np.int64),
        "rel_category_ids": np.stack(output["rel_category_ids"]).astype(np.int64),
        "v5_gt": np.stack(output["v5_gt"]).astype(np.float32),
    })
    return reference, rel_rows, [v5_rows[sample_id] for sample_id in v5_order]


def main() -> None:
    args = parse_args()
    if args.diffusion_steps != DIFFUSION_STEPS or args.num_generations != GENERATIONS:
        raise ValueError("Teacher-v7 full evaluation is sealed to 500-step K=3")
    if args.seed != SEED:
        raise ValueError("Teacher-v7 full K=3 seed is sealed to 20260911")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v7 full K=3 requires CUDA")
    configure_reproducibility(args.seed)

    k3_file = args.k3_canary_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v7 full K=3")
    if not k3_file.is_file():
        raise FileNotFoundError(k3_file)
    k3 = read_json(k3_file)
    if (k3.get("schema") != K3_SCHEMA or k3.get("status") != "SELECTED_K3_PASS"
            or k3.get("failed_checks") != []
            or int(k3.get("selected_step", -1)) != SELECTED_STEP
            or int(k3.get("seed", -1)) != SEED
            or k3.get("authorizes_train_only_full_k3") is not True
            or k3.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 selected K=3 does not authorize full train evaluation")
    verify_source_binding(k3, "Teacher-v7 selected K=3")
    paths = {key: bound_file(k3, key) for key in (
        "k1_summary", "k1_predictions", "candidate_checkpoint", "dataset_index",
        "v5_split", "stats_file", "original_checkpoint", "v5_checkpoint", "predictions",
    )}
    with np.load(paths["k1_predictions"], allow_pickle=False) as source:
        canary_reference = {name: source[name] for name in source.files}
    with np.load(paths["predictions"], allow_pickle=False) as source:
        canary_maps = {name: source[name] for name in source.files}

    dataset_root = paths["dataset_index"].parent
    v5_root = paths["v5_split"].parent.parent
    mean, std = load_stats(paths["stats_file"])
    index = read_json(paths["dataset_index"])
    grouped = group_relational_rows(load_relational_rows(dataset_root, index))
    v5_rows = load_v5_rows(
        v5_root, load_split(paths["v5_split"]), "train", mean, std, 4.0, 16.0, 0.7
    )
    reference, rel_rows, ordered_v5_rows = build_reference(
        dataset_root, index, grouped, v5_rows
    )

    relational_base = np.empty(
        (RELATIONAL_CASES, GENERATIONS, 8192, 6), dtype=np.float32
    )
    relational_candidate = np.empty_like(relational_base)
    relational_seeds = np.empty((RELATIONAL_CASES, GENERATIONS, 2), dtype=np.int64)
    v5_base = np.empty((V5_CASES, GENERATIONS, 8192, 6), dtype=np.float32)
    v5_candidate = np.empty_like(v5_base)
    v5_seeds = np.empty((V5_CASES, GENERATIONS, 2), dtype=np.int64)

    canary_rel_lookup = {
        str(case_id): index_value
        for index_value, case_id in enumerate(canary_reference["rel_case_ids"].tolist())
    }
    canary_v5_lookup = {
        str(sample_id): index_value
        for index_value, sample_id in enumerate(canary_reference["v5_ids"].tolist())
    }
    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    base_model, base_diffusion = create_model(
        cfg, paths["original_checkpoint"], paths["v5_checkpoint"], None, args.device
    )
    candidate_model, candidate_diffusion = create_model(
        cfg, paths["original_checkpoint"], paths["v5_checkpoint"],
        paths["candidate_checkpoint"], args.device,
    )
    if base_diffusion.num_timesteps != candidate_diffusion.num_timesteps:
        raise AssertionError("Teacher-v7 diffusion schedules differ")

    repeatability = {"base": None, "candidate": None}
    reused_relational = 0
    for index_value, row in enumerate(rel_rows):
        scene_id = str(reference["rel_scene_ids"][index_value])
        stratum = str(reference["rel_strata"][index_value])
        motion_id = str(reference["rel_motion_ids"][index_value])
        prompt_id = str(reference["rel_prompt_ids"][index_value])
        canary_id = f"{scene_id}|{stratum}|{prompt_id}"
        first_motion = str(grouped[scene_id][stratum][0]["motion_id"])
        if motion_id == first_motion:
            source_index = canary_rel_lookup[canary_id]
            relational_base[index_value] = canary_maps["relational_base"][source_index]
            relational_candidate[index_value] = canary_maps["relational_candidate"][source_index]
            relational_seeds[index_value] = canary_maps["relational_seeds"][source_index]
            reused_relational += 1
            continue
        points = np.asarray(row["points"], dtype=np.float32)
        pair_id = f"{scene_id}|{stratum}|{motion_id}"
        for generation in range(GENERATIONS):
            initial_seed, reverse_seed = generation_seeds(
                args.seed, "full_relational_pair", pair_id, generation
            )
            relational_seeds[index_value, generation] = (initial_seed, reverse_seed)
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
            relational_candidate[index_value, generation] = denormalize(
                candidate_norm, mean, std
            )
        print(f"[REL {index_value + 1:03d}/{RELATIONAL_CASES}] {reference['rel_case_ids'][index_value]}")

    reused_v5 = 0
    for index_value, row in enumerate(ordered_v5_rows):
        sample_id = str(reference["v5_ids"][index_value])
        if sample_id in canary_v5_lookup:
            source_index = canary_v5_lookup[sample_id]
            v5_base[index_value] = canary_maps["v5_base"][source_index]
            v5_candidate[index_value] = canary_maps["v5_candidate"][source_index]
            v5_seeds[index_value] = canary_maps["v5_seeds"][source_index]
            reused_v5 += 1
            continue
        for generation in range(GENERATIONS):
            initial_seed, reverse_seed = generation_seeds(
                args.seed, "full_v5_replay", sample_id, generation
            )
            v5_seeds[index_value, generation] = (initial_seed, reverse_seed)
            base_norm = sample_contact(
                base_model, base_diffusion, np.asarray(row["xyz"], dtype=np.float32),
                np.asarray(row["feat"], dtype=np.float32), str(row["text"]),
                initial_seed, reverse_seed, args.device, not args.no_progress,
            )
            candidate_norm = sample_contact(
                candidate_model, candidate_diffusion,
                np.asarray(row["xyz"], dtype=np.float32),
                np.asarray(row["feat"], dtype=np.float32), str(row["text"]),
                initial_seed, reverse_seed, args.device, not args.no_progress,
            )
            v5_base[index_value, generation] = denormalize(base_norm, mean, std)
            v5_candidate[index_value, generation] = denormalize(candidate_norm, mean, std)
        print(f"[V5 {index_value + 1:02d}/{V5_CASES}] {sample_id}")

    if repeatability != {"base": True, "candidate": True}:
        raise RuntimeError("Teacher-v7 full K=3 reverse diffusion is not repeatable")
    if reused_relational != 16 or reused_v5 != 3:
        raise AssertionError("Teacher-v7 canary reuse inventory changed")
    generations = per_generation_metrics(
        reference, relational_base, relational_candidate, v5_base, v5_candidate
    )
    pooled = pooled_metrics(generations)
    checks = full_checks(pooled, generations, True)
    checks.update({
        "sealed_16_relational_and_3_v5_cases_reused": (
            reused_relational == 16 and reused_v5 == 3
        ),
        "all_48_train_motions_have_two_prompts_and_three_generations": (
            len(set(zip(
                reference["rel_scene_ids"].tolist(),
                reference["rel_motion_ids"].tolist(),
            ))) == TRAIN_MOTIONS
            and relational_base.shape[:2] == (RELATIONAL_CASES, GENERATIONS)
        ),
        "high_desk_inventory_is_24_cases_per_generation": all(
            metrics["high_desk"]["count"] == HIGH_DESK_CASES_PER_GENERATION
            for metrics in generations
        ),
        "development_payloads_unread": True,
    })
    status = "FULL_TRAIN_K3_PASS" if all(checks.values()) else "FULL_TRAIN_K3_FAIL"

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_file = output_dir / "full_train_k3_predictions.npz"
    temporary = bundle_file.with_name(bundle_file.name + ".tmp." + str(os.getpid()) + ".npz")
    np.savez_compressed(
        temporary, **reference, relational_base=relational_base,
        relational_candidate=relational_candidate, relational_seeds=relational_seeds,
        v5_base=v5_base, v5_candidate=v5_candidate, v5_seeds=v5_seeds,
    )
    os.replace(temporary, bundle_file)
    prepare_root = Path(__file__).resolve().parent
    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "validator": prepare_root / "validate_relational_teacher_v7_hd_lora_full_train_k3.py",
        "full_common": prepare_root / "relational_teacher_v7_hd_full_k3_common.py",
        "full_contract": prepare_root / "relational_teacher_v7_hd_full_k3_contract.py",
    }
    summary = {
        "schema": SCHEMA, "status": status, "partition": "train_full",
        "selected_step": SELECTED_STEP, "diffusion_steps": args.diffusion_steps,
        "num_generations": GENERATIONS, "seed": args.seed, "seed_policy": SEED_POLICY,
        "paired_initial_and_reverse_noise": True,
        "train_motion_count": TRAIN_MOTIONS,
        "relational_case_count": RELATIONAL_CASES,
        "high_desk_case_count_per_generation": HIGH_DESK_CASES_PER_GENERATION,
        "v5_case_count": V5_CASES,
        "reused_canary_relational_case_count": reused_relational,
        "reused_canary_v5_case_count": reused_v5,
        "paired_evaluation_reverse_draws_generated": 612,
        "repeatability_reverse_draws_generated": 2,
        "total_reverse_draws_generated": 614,
        "repeatability": repeatability,
        "per_generation_metrics": generations, "pooled_metrics": pooled,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "development_payloads_read": False,
        "k3_canary_summary": str(k3_file),
        "k3_canary_summary_sha256": sha256_file(k3_file),
        "k3_canary_predictions": str(paths["predictions"]),
        "k3_canary_predictions_sha256": sha256_file(paths["predictions"]),
        "k1_summary": str(paths["k1_summary"]),
        "k1_summary_sha256": sha256_file(paths["k1_summary"]),
        "k1_predictions": str(paths["k1_predictions"]),
        "k1_predictions_sha256": sha256_file(paths["k1_predictions"]),
        "candidate_checkpoint": str(paths["candidate_checkpoint"]),
        "candidate_checkpoint_sha256": sha256_file(paths["candidate_checkpoint"]),
        "dataset_index": str(paths["dataset_index"]),
        "dataset_index_sha256": sha256_file(paths["dataset_index"]),
        "v5_split": str(paths["v5_split"]),
        "v5_split_sha256": sha256_file(paths["v5_split"]),
        "stats_file": str(paths["stats_file"]),
        "stats_file_sha256": sha256_file(paths["stats_file"]),
        "original_checkpoint": str(paths["original_checkpoint"]),
        "original_checkpoint_sha256": sha256_file(paths["original_checkpoint"]),
        "v5_checkpoint": str(paths["v5_checkpoint"]),
        "v5_checkpoint_sha256": sha256_file(paths["v5_checkpoint"]),
        "predictions": str(bundle_file), "predictions_sha256": sha256_file(bundle_file),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "diagnostic_only": status != "FULL_TRAIN_K3_PASS",
        "checkpoint_locked": status == "FULL_TRAIN_K3_PASS",
        "authorizes_internal_development_evaluation": status == "FULL_TRAIN_K3_PASS",
        "authorizes_paper_test": False,
    }
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, summary)
    print(f"[{status}] Teacher-v7 selected step-12 full train-only K=3")
    for generation, metrics in enumerate(generations):
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

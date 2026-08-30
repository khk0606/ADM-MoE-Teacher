#!/usr/bin/env python3
"""Paired K=1 train-only reverse-diffusion canary for Teacher-v7 High-Desk."""

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
from relational_teacher_v7_hd_pilot_contract import SCHEMA as PILOT_SCHEMA
from relational_teacher_v7_hd_preflight_contract import (
    DATASET_SCHEMA,
    EXPECTED_TRAIN_SCENES,
)
from relational_teacher_v7_hd_rollout_common import (
    aggregate_metrics,
    create_model,
    denormalize,
    sample_contact,
    stable_seeds,
)
from relational_teacher_v7_hd_rollout_contract import (
    RELATIONAL_CASES,
    SCHEMA,
    SELECTED_STEP,
    V5_REPLAY_CASES,
    compute_checks,
    compute_scene_checks,
)
from train_fewshot_cdm import load_rows as load_v5_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v5-dataset-root", type=Path, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.diffusion_steps != 500:
        raise ValueError("Teacher-v7 canary is sealed to 500 diffusion steps")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v7 rollout canary requires CUDA")
    configure_reproducibility(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    index_file = args.index.expanduser().resolve()
    v5_dataset_root = args.v5_dataset_root.expanduser().resolve()
    v5_split_file = args.v5_split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    v5_checkpoint = args.v5_checkpoint.expanduser().resolve()
    pilot_file = args.pilot_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v7 rollout canary")
    for path in (index_file, v5_split_file, stats_file, original_checkpoint,
                 v5_checkpoint, pilot_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    pilot = read_json(pilot_file)
    if (pilot.get("schema") != PILOT_SCHEMA
            or pilot.get("status") != "PILOT_PASS"
            or pilot.get("failed_checks") != []
            or pilot.get("shortlisted_steps") != [SELECTED_STEP]
            or pilot.get("development_payloads_read") is not False
            or pilot.get("authorizes_train_only_reverse_diffusion_shortlist") is not True
            or pilot.get("diagnostic_checkpoints_only") is not True
            or pilot.get("fresh_zero_init_from_sealed_v5r4") is not True
            or pilot.get("recovery_checkpoint_loaded") is not False):
        raise ValueError("Teacher-v7 pilot does not authorize rollout canary")
    for key in ("authorizes_long_training", "authorizes_development_evaluation", "authorizes_paper_test"):
        if pilot.get(key) is not False:
            raise ValueError("Teacher-v7 pilot authority changed: " + key)
    source_paths = pilot.get("source_paths")
    source_hashes = pilot.get("source_sha256")
    if not isinstance(source_paths, dict) or not isinstance(source_hashes, dict) or set(source_paths) != set(source_hashes):
        raise ValueError("Teacher-v7 pilot source binding is absent")
    for name, raw in source_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(source_hashes[name]):
            raise ValueError("Teacher-v7 pilot source changed: " + name)
    for key, path in (
        ("dataset_index", index_file), ("v5_split", v5_split_file),
        ("stats_file", stats_file), ("original_checkpoint", original_checkpoint),
        ("v5_checkpoint", v5_checkpoint),
    ):
        if pilot.get(key + "_sha256") != sha256_file(path):
            raise ValueError("rollout input differs from pilot: " + key)
    records = pilot.get("checkpoint_records")
    selected = [row for row in records if int(row.get("step", -1)) == SELECTED_STEP]
    if len(selected) != 1 or selected[0].get("eligible") is not True:
        raise ValueError("Teacher-v7 selected checkpoint record changed")
    candidate_checkpoint = Path(str(selected[0]["merged_checkpoint"])).resolve()
    if (not candidate_checkpoint.is_file()
            or sha256_file(candidate_checkpoint) != str(selected[0]["merged_checkpoint_sha256"])):
        raise ValueError("Teacher-v7 selected checkpoint changed")

    index = read_json(index_file)
    if (index.get("schema") != DATASET_SCHEMA
            or index.get("status") != "DENSE_DATASET_PASS"
            or index.get("split_row_counts") != {"development": 48, "train": 96}
            or index.get("heldout_dense_arrays_read_during_build") is not False):
        raise ValueError("Teacher-v7 dataset index changed")
    relational_rows = load_relational_rows(dataset_root, index)
    grouped = group_relational_rows(relational_rows)
    mean, std = load_stats(stats_file)
    v5_rows = load_v5_rows(
        v5_dataset_root, load_split(v5_split_file), "train", mean, std, 4.0, 16.0, 0.7
    )
    v5_by_target = {}
    for sample_id, row in sorted(v5_rows.items()):
        v5_by_target.setdefault(str(row["target"]), (sample_id, row))
    if set(v5_by_target) != {"chair", "bed", "whiteboard"}:
        raise ValueError("v5 replay targets changed")

    scene_records = {str(row["scene_id"]): row for row in index.get("scenes", [])
                     if row.get("split") == "train"}
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

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    base_model, base_diffusion = create_model(
        cfg, original_checkpoint, v5_checkpoint, None, args.device
    )
    candidate_model, candidate_diffusion = create_model(
        cfg, original_checkpoint, v5_checkpoint, candidate_checkpoint, args.device
    )
    if base_diffusion.num_timesteps != candidate_diffusion.num_timesteps:
        raise AssertionError("Teacher-v7 diffusion schedules differ")

    rel_ids, rel_scene_ids, rel_target_names, rel_strata, rel_prompt_ids = [], [], [], [], []
    rel_target_instance_ids, rel_gt, rel_base, rel_candidate = [], [], [], []
    rel_instance_ids, rel_category_ids = [], []
    repeatability = None
    for scene_id in EXPECTED_TRAIN_SCENES:
        for stratum, rows in sorted(grouped[scene_id].items()):
            row = rows[0]
            target_name = str(row["target_instance_id"])
            target_key = target_name.lower()
            if target_key not in instance_name_to_id[scene_id]:
                raise ValueError(scene_id + ": target instance name is unresolved")
            for prompt_id, text in sorted(PROMPTS.items()):
                case_id = f"{scene_id}|{stratum}|{prompt_id}"
                initial_seed, reverse_seed = stable_seeds(
                    args.seed, "relational_pair", f"{scene_id}|{stratum}"
                )
                points = np.asarray(row["points"], dtype=np.float32)
                xyz, feat = points[:, :3], points[:, 3:6] / 255.0
                base_normalized = sample_contact(
                    base_model, base_diffusion, xyz, feat, text, initial_seed,
                    reverse_seed, args.device, not args.no_progress,
                )
                candidate_normalized = sample_contact(
                    candidate_model, candidate_diffusion, xyz, feat, text,
                    initial_seed, reverse_seed, args.device, not args.no_progress,
                )
                if repeatability is None:
                    repeated = sample_contact(
                        candidate_model, candidate_diffusion, xyz, feat, text,
                        initial_seed, reverse_seed, args.device, False,
                    )
                    repeatability = bool(np.array_equal(candidate_normalized, repeated))
                    if not repeatability:
                        raise RuntimeError("Teacher-v7 reverse diffusion is not repeatable")
                rel_ids.append(case_id)
                rel_scene_ids.append(scene_id)
                rel_target_names.append(target_name)
                rel_strata.append(stratum)
                rel_prompt_ids.append(prompt_id)
                rel_target_instance_ids.append(instance_name_to_id[scene_id][target_key])
                rel_gt.append(np.asarray(row["affordance"], dtype=np.float32))
                rel_base.append(denormalize(base_normalized, mean, std))
                rel_candidate.append(denormalize(candidate_normalized, mean, std))
                rel_instance_ids.append(np.asarray(row["instance_ids"], dtype=np.int64))
                rel_category_ids.append(np.asarray(row["category_ids"], dtype=np.int64))
                print("[RELATIONAL]", case_id)

    v5_ids, v5_targets, v5_gt, v5_base, v5_candidate = [], [], [], [], []
    for target in ("chair", "bed", "whiteboard"):
        sample_id, row = v5_by_target[target]
        initial_seed, reverse_seed = stable_seeds(args.seed, "v5_replay", sample_id)
        base_normalized = sample_contact(
            base_model, base_diffusion, np.asarray(row["xyz"], dtype=np.float32),
            np.asarray(row["feat"], dtype=np.float32), str(row["text"]),
            initial_seed, reverse_seed, args.device, not args.no_progress,
        )
        candidate_normalized = sample_contact(
            candidate_model, candidate_diffusion, np.asarray(row["xyz"], dtype=np.float32),
            np.asarray(row["feat"], dtype=np.float32), str(row["text"]),
            initial_seed, reverse_seed, args.device, not args.no_progress,
        )
        v5_ids.append(sample_id)
        v5_targets.append(target)
        v5_gt.append(np.asarray(row["gt"], dtype=np.float32))
        v5_base.append(denormalize(base_normalized, mean, std))
        v5_candidate.append(denormalize(candidate_normalized, mean, std))
        print("[V5-REPLAY]", sample_id, "target=" + target)

    arrays = {
        "rel_case_ids": np.asarray(rel_ids),
        "rel_scene_ids": np.asarray(rel_scene_ids),
        "rel_target_names": np.asarray(rel_target_names),
        "rel_strata": np.asarray(rel_strata),
        "rel_prompt_ids": np.asarray(rel_prompt_ids),
        "rel_target_instance_ids": np.asarray(rel_target_instance_ids, dtype=np.int64),
        "rel_gt": np.stack(rel_gt).astype(np.float32),
        "rel_base": np.stack(rel_base).astype(np.float32),
        "rel_candidate": np.stack(rel_candidate).astype(np.float32),
        "rel_instance_ids": np.stack(rel_instance_ids).astype(np.int64),
        "rel_category_ids": np.stack(rel_category_ids).astype(np.int64),
        "v5_ids": np.asarray(v5_ids),
        "v5_targets": np.asarray(v5_targets),
        "v5_gt": np.stack(v5_gt).astype(np.float32),
        "v5_base": np.stack(v5_base).astype(np.float32),
        "v5_candidate": np.stack(v5_candidate).astype(np.float32),
    }
    if len(rel_ids) != RELATIONAL_CASES or len(v5_ids) != V5_REPLAY_CASES:
        raise AssertionError("Teacher-v7 canary case count changed")
    metrics = aggregate_metrics(
        arrays["rel_gt"], arrays["rel_base"], arrays["rel_candidate"],
        arrays["rel_instance_ids"], arrays["rel_category_ids"],
        arrays["rel_target_instance_ids"], rel_scene_ids, rel_target_names,
        rel_strata, rel_prompt_ids, arrays["v5_gt"], arrays["v5_base"],
        arrays["v5_candidate"], v5_targets,
    )
    scene_checks = compute_scene_checks(metrics)
    checks = compute_checks(metrics, scene_checks, repeatability is True, True)
    status = "CANARY_PASS" if all(checks.values()) else "CANARY_FAIL"

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_file = output_dir / "predictions.npz"
    temporary = predictions_file.with_name(predictions_file.name + ".tmp." + str(os.getpid()) + ".npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, predictions_file)
    prepare_root = Path(__file__).resolve().parent
    canary_sources = {
        "evaluator": Path(__file__).resolve(),
        "validator": prepare_root / "validate_relational_teacher_v7_hd_lora_rollout_canary.py",
        "rollout_common": prepare_root / "relational_teacher_v7_hd_rollout_common.py",
        "rollout_contract": prepare_root / "relational_teacher_v7_hd_rollout_contract.py",
    }
    summary = {
        "schema": SCHEMA,
        "status": status,
        "partition": "train_canary",
        "selected_step": SELECTED_STEP,
        "diffusion_steps": args.diffusion_steps,
        "num_generations": 1,
        "seed": args.seed,
        "paired_initial_and_reverse_noise": True,
        "relational_case_count": len(rel_ids),
        "high_desk_case_count": 4,
        "v5_replay_case_count": len(v5_ids),
        "metrics": metrics,
        "scene_checks": scene_checks,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "development_payloads_read": False,
        "pilot_summary": str(pilot_file),
        "pilot_summary_sha256": sha256_file(pilot_file),
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
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "predictions": str(predictions_file),
        "predictions_sha256": sha256_file(predictions_file),
        "source_paths": {name: str(path) for name, path in canary_sources.items()},
        "source_sha256": {name: sha256_file(path) for name, path in canary_sources.items()},
        "diagnostic_only": True,
        "authorizes_train_only_k3_canary": status == "CANARY_PASS",
        "authorizes_full_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, summary)
    print(f"[{status}] Teacher-v7 High-Desk paired train-only reverse-diffusion canary")
    print("[OK] overall:", metrics["overall"])
    print("[OK] high desk:", metrics["high_desk"])
    print("[OK] failed checks:", summary["failed_checks"])
    print("[OK] summary:", summary_file)


if __name__ == "__main__":
    main()

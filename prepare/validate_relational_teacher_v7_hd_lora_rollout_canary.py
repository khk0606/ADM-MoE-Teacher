#!/usr/bin/env python3
"""Validate saved Teacher-v7 High-Desk paired K=1 canary maps."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from relational_teacher_v7_hd_rollout_contract import (
    RELATIONAL_CASES,
    SCHEMA,
    SELECTED_STEP,
    V5_REPLAY_CASES,
    compute_checks,
    compute_scene_checks,
)
from fewshot_cdm_common import load_split
from preflight_relational_teacher_v7_hd_lora import load_stats
from recover_relational_teacher_v7_hd_lora_invariance import load_relational_rows
from relational_teacher_v6_contract import PROMPTS, read_json, sha256_file
from relational_teacher_v7_hd_calibration_contract import group_relational_rows
from relational_teacher_v7_hd_pilot_contract import SCHEMA as PILOT_SCHEMA
from relational_teacher_v7_hd_preflight_contract import EXPECTED_TRAIN_SCENES
from relational_teacher_v7_hd_rollout_common import aggregate_metrics
from train_fewshot_cdm import load_rows as load_v5_rows


REQUIRED_SOURCE_KEYS = {"evaluator", "validator", "rollout_common", "rollout_contract"}


def nested_close(left, right, label: str) -> None:
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError(label + " mapping keys changed")
        for key in left:
            nested_close(left[key], right[key], label + "." + str(key))
        return
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError(label + " list length changed")
        for index, (a, b) in enumerate(zip(left, right)):
            nested_close(a, b, label + f"[{index}]")
        return
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or not math.isclose(
            float(left), float(right), rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " value changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v7 rollout canary schema changed")
    if value.get("status") not in {"CANARY_PASS", "CANARY_FAIL"}:
        raise ValueError("Teacher-v7 rollout canary status changed")
    if (value.get("partition") != "train_canary"
            or int(value.get("selected_step", -1)) != SELECTED_STEP
            or int(value.get("diffusion_steps", -1)) != 500
            or int(value.get("num_generations", -1)) != 1
            or value.get("paired_initial_and_reverse_noise") is not True
            or int(value.get("relational_case_count", -1)) != RELATIONAL_CASES
            or int(value.get("high_desk_case_count", -1)) != 4
            or int(value.get("v5_replay_case_count", -1)) != V5_REPLAY_CASES
            or value.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 rollout protocol changed")

    bound = {
        "pilot_summary": "pilot_summary_sha256",
        "dataset_index": "dataset_index_sha256",
        "v5_split": "v5_split_sha256",
        "stats_file": "stats_file_sha256",
        "original_checkpoint": "original_checkpoint_sha256",
        "v5_checkpoint": "v5_checkpoint_sha256",
        "candidate_checkpoint": "candidate_checkpoint_sha256",
        "predictions": "predictions_sha256",
    }
    resolved = {}
    for path_key, hash_key in bound.items():
        path = Path(str(value.get(path_key, ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(value.get(hash_key, "")):
            raise ValueError("Teacher-v7 rollout artifact changed: " + path_key)
        resolved[path_key] = path
    source_paths = value.get("source_paths")
    source_hashes = value.get("source_sha256")
    if (not isinstance(source_paths, dict) or not isinstance(source_hashes, dict)
            or set(source_paths) != set(source_hashes)
            or set(source_paths) != REQUIRED_SOURCE_KEYS):
        raise ValueError("Teacher-v7 rollout source bindings are absent")
    for name, raw in source_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(source_hashes[name]):
            raise ValueError("Teacher-v7 rollout source changed: " + name)

    pilot = read_json(resolved["pilot_summary"])
    if (pilot.get("schema") != PILOT_SCHEMA
            or pilot.get("status") != "PILOT_PASS"
            or pilot.get("failed_checks") != []
            or pilot.get("shortlisted_steps") != [SELECTED_STEP]
            or pilot.get("development_payloads_read") is not False
            or pilot.get("authorizes_train_only_reverse_diffusion_shortlist") is not True):
        raise ValueError("Teacher-v7 pilot authorization changed")
    record = [row for row in pilot.get("checkpoint_records", [])
              if int(row.get("step", -1)) == SELECTED_STEP]
    if (len(record) != 1 or record[0].get("eligible") is not True
            or record[0].get("merged_checkpoint_sha256")
            != value.get("candidate_checkpoint_sha256")):
        raise ValueError("Teacher-v7 selected checkpoint binding changed")
    for key in ("dataset_index", "v5_split", "stats_file", "original_checkpoint", "v5_checkpoint"):
        if pilot.get(key + "_sha256") != value.get(key + "_sha256"):
            raise ValueError("Teacher-v7 rollout input differs from pilot: " + key)
    pilot_sources, pilot_hashes = pilot.get("source_paths"), pilot.get("source_sha256")
    if not isinstance(pilot_sources, dict) or not isinstance(pilot_hashes, dict) or set(pilot_sources) != set(pilot_hashes):
        raise ValueError("Teacher-v7 pilot source bindings changed")
    for name, raw in pilot_sources.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(pilot_hashes[name]):
            raise ValueError("Teacher-v7 pilot source changed: " + name)

    with np.load(resolved["predictions"], allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    expected_keys = {
        "rel_case_ids", "rel_scene_ids", "rel_target_names", "rel_strata",
        "rel_prompt_ids", "rel_target_instance_ids", "rel_gt", "rel_base",
        "rel_candidate", "rel_instance_ids", "rel_category_ids", "v5_ids",
        "v5_targets", "v5_gt", "v5_base", "v5_candidate",
    }
    if set(arrays) != expected_keys:
        raise ValueError("Teacher-v7 saved prediction keys changed")
    for name in ("rel_gt", "rel_base", "rel_candidate"):
        if arrays[name].shape != (RELATIONAL_CASES, 8192, 6):
            raise ValueError(name + " shape changed")
    for name in ("rel_instance_ids", "rel_category_ids"):
        if arrays[name].shape != (RELATIONAL_CASES, 8192):
            raise ValueError(name + " shape changed")
    for name in ("v5_gt", "v5_base", "v5_candidate"):
        if arrays[name].shape != (V5_REPLAY_CASES, 8192, 6):
            raise ValueError(name + " shape changed")
    for name in ("rel_gt", "rel_base", "rel_candidate", "v5_gt", "v5_base", "v5_candidate"):
        if (not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0.0)
                or np.any(arrays[name] > 1.0)):
            raise ValueError(name + " range changed")
    if Counter(str(item) for item in arrays["rel_scene_ids"].tolist()) != Counter(
        {"room_0101": 8, "room_0102": 8}
    ):
        raise ValueError("Teacher-v7 rollout train scene balance changed")
    if Counter(str(item) for item in arrays["rel_strata"].tolist()) != Counter({
        "bed_01": 4, "high_desk_chair_06": 4, "legacy_chair_06": 4,
        "chair_01": 2, "chair_05": 2,
    }):
        raise ValueError("Teacher-v7 rollout stratum balance changed")
    if set(str(item) for item in arrays["rel_prompt_ids"].tolist()) != set(PROMPTS):
        raise ValueError("Teacher-v7 rollout prompts changed")
    if Counter(str(item) for item in arrays["v5_targets"].tolist()) != Counter(
        {"chair": 1, "bed": 1, "whiteboard": 1}
    ):
        raise ValueError("Teacher-v7 rollout v5 target balance changed")

    dataset_root = resolved["dataset_index"].parent
    index = read_json(resolved["dataset_index"])
    grouped = group_relational_rows(load_relational_rows(dataset_root, index))
    scene_records = {str(row["scene_id"]): row for row in index.get("scenes", [])
                     if row.get("split") == "train"}
    expected_ids, expected_scenes, expected_targets, expected_strata, expected_prompts = [], [], [], [], []
    expected_target_ids, expected_gt, expected_instances, expected_categories = [], [], [], []
    for scene_id in EXPECTED_TRAIN_SCENES:
        record_data = scene_records[scene_id]
        instances_file = (dataset_root / str(record_data["instances_file"])).resolve()
        if sha256_file(instances_file) != str(record_data["instances_sha256"]):
            raise ValueError(scene_id + ": instances hash changed")
        objects = read_json(instances_file)["objects"]
        instance_map = {str(row["name"]).lower(): int(row["instance_id"]) for row in objects}
        for stratum, rows in sorted(grouped[scene_id].items()):
            row = rows[0]
            target_name = str(row["target_instance_id"])
            for prompt_id in sorted(PROMPTS):
                expected_ids.append(f"{scene_id}|{stratum}|{prompt_id}")
                expected_scenes.append(scene_id)
                expected_targets.append(target_name)
                expected_strata.append(stratum)
                expected_prompts.append(prompt_id)
                expected_target_ids.append(instance_map[target_name.lower()])
                expected_gt.append(np.asarray(row["affordance"], dtype=np.float32))
                expected_instances.append(np.asarray(row["instance_ids"], dtype=np.int64))
                expected_categories.append(np.asarray(row["category_ids"], dtype=np.int64))
    for name, expected in (
        ("rel_case_ids", expected_ids), ("rel_scene_ids", expected_scenes),
        ("rel_target_names", expected_targets), ("rel_strata", expected_strata),
        ("rel_prompt_ids", expected_prompts),
    ):
        if [str(item) for item in arrays[name].tolist()] != expected:
            raise ValueError(name + " source binding changed")
    if not np.array_equal(arrays["rel_target_instance_ids"], np.asarray(expected_target_ids, dtype=np.int64)):
        raise ValueError("relational target IDs changed")
    if not np.array_equal(arrays["rel_gt"], np.stack(expected_gt)):
        raise ValueError("relational GT changed")
    if not np.array_equal(arrays["rel_instance_ids"], np.stack(expected_instances)):
        raise ValueError("relational instance IDs changed")
    if not np.array_equal(arrays["rel_category_ids"], np.stack(expected_categories)):
        raise ValueError("relational category IDs changed")

    mean, std = load_stats(resolved["stats_file"])
    v5_root = resolved["v5_split"].parent.parent
    v5_rows = load_v5_rows(v5_root, load_split(resolved["v5_split"]), "train", mean, std, 4.0, 16.0, 0.7)
    v5_by_target = {}
    for sample_id, row in sorted(v5_rows.items()):
        v5_by_target.setdefault(str(row["target"]), (sample_id, row))
    expected_v5 = [v5_by_target[target] for target in ("chair", "bed", "whiteboard")]
    if [str(item) for item in arrays["v5_ids"].tolist()] != [row[0] for row in expected_v5]:
        raise ValueError("v5 replay IDs changed")
    if not np.array_equal(arrays["v5_gt"], np.stack([
        np.asarray(row[1]["gt"], dtype=np.float32) for row in expected_v5
    ])):
        raise ValueError("v5 replay GT changed")

    scene_ids = [str(item) for item in arrays["rel_scene_ids"].tolist()]
    target_names = [str(item) for item in arrays["rel_target_names"].tolist()]
    strata = [str(item) for item in arrays["rel_strata"].tolist()]
    prompt_ids = [str(item) for item in arrays["rel_prompt_ids"].tolist()]
    v5_targets = [str(item) for item in arrays["v5_targets"].tolist()]
    metrics = aggregate_metrics(
        arrays["rel_gt"], arrays["rel_base"], arrays["rel_candidate"],
        arrays["rel_instance_ids"], arrays["rel_category_ids"],
        arrays["rel_target_instance_ids"], scene_ids, target_names, strata,
        prompt_ids, arrays["v5_gt"], arrays["v5_base"], arrays["v5_candidate"],
        v5_targets,
    )
    nested_close(metrics, value.get("metrics"), "metrics")
    scene_checks = compute_scene_checks(metrics)
    if value.get("scene_checks") != scene_checks:
        raise ValueError("Teacher-v7 rollout scene checks changed")
    repeatability = value.get("checks", {}).get("paired_reverse_diffusion_repeatable") is True
    checks = compute_checks(metrics, scene_checks, repeatability, value.get("development_payloads_read") is False)
    if value.get("checks") != checks:
        raise ValueError("Teacher-v7 rollout checks changed")
    failed = [name for name, passed in checks.items() if not passed]
    if value.get("failed_checks") != failed:
        raise ValueError("Teacher-v7 rollout failed-check list changed")
    expected_status = "CANARY_PASS" if not failed else "CANARY_FAIL"
    if value.get("status") != expected_status:
        raise ValueError("Teacher-v7 rollout status/checks disagree")
    if value.get("diagnostic_only") is not True:
        raise ValueError("Teacher-v7 rollout overstates authority")
    if value.get("authorizes_train_only_k3_canary") is not (expected_status == "CANARY_PASS"):
        raise ValueError("Teacher-v7 K=3 canary authority changed")
    for key in ("authorizes_full_training", "authorizes_development_evaluation", "authorizes_paper_test"):
        if value.get(key) is not False:
            raise ValueError("Teacher-v7 rollout authority changed: " + key)

    print(f"[{expected_status}] Teacher-v7 High-Desk reverse-diffusion canary integrity")
    print("[PASS] saved 16 relational and 3 v5 maps recomputed")
    print("[PASS] four-stratum GT/source and selected-step bindings verified")
    print("[PASS] held-out room_0201 payloads unread")
    print("[OK] overall:", metrics["overall"])
    print("[OK] high desk:", metrics["high_desk"])
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

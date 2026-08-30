#!/usr/bin/env python3
"""Validate Teacher-v7 selected step-12 full train-only K=3."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import numpy as np

from evaluate_relational_teacher_v7_hd_lora_full_train_k3 import build_reference
from fewshot_cdm_common import load_split
from preflight_relational_teacher_v7_hd_lora import load_stats
from recover_relational_teacher_v7_hd_lora_invariance import load_relational_rows
from relational_teacher_v6_contract import PROMPTS, read_json, sha256_file
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
from train_fewshot_cdm import load_rows as load_v5_rows


REQUIRED_SOURCE_KEYS = {"evaluator", "validator", "full_common", "full_contract"}


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
        for index, (first, second) in enumerate(zip(left, right)):
            nested_close(first, second, label + f"[{index}]")
        return
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or not math.isclose(
            float(left), float(right), rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " value changed")


def bound_file(value: dict, key: str) -> Path:
    path = Path(str(value.get(key, ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != str(value.get(key + "_sha256", "")):
        raise ValueError("Teacher-v7 full K=3 artifact changed: " + key)
    return path


def verify_source_binding(value: dict, label: str, required=None) -> None:
    paths, hashes = value.get("source_paths"), value.get("source_sha256")
    if (not isinstance(paths, dict) or not isinstance(hashes, dict)
            or set(paths) != set(hashes)
            or (required is not None and set(paths) != required)):
        raise ValueError(label + " source binding is absent")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(hashes[name]):
            raise ValueError(label + " source changed: " + str(name))


def assert_array_equal(actual, expected, label: str) -> None:
    if actual.dtype.kind in {"U", "S"}:
        if [str(item) for item in actual.tolist()] != [str(item) for item in expected.tolist()]:
            raise ValueError(label + " changed")
    elif not np.array_equal(actual, expected):
        raise ValueError(label + " changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v7 full K=3 schema changed")
    if value.get("status") not in {"FULL_TRAIN_K3_PASS", "FULL_TRAIN_K3_FAIL"}:
        raise ValueError("Teacher-v7 full K=3 status changed")
    if (value.get("partition") != "train_full"
            or int(value.get("selected_step", -1)) != SELECTED_STEP
            or int(value.get("diffusion_steps", -1)) != DIFFUSION_STEPS
            or int(value.get("num_generations", -1)) != GENERATIONS
            or int(value.get("seed", -1)) != SEED
            or value.get("seed_policy") != SEED_POLICY
            or value.get("paired_initial_and_reverse_noise") is not True
            or int(value.get("train_motion_count", -1)) != TRAIN_MOTIONS
            or int(value.get("relational_case_count", -1)) != RELATIONAL_CASES
            or int(value.get("high_desk_case_count_per_generation", -1))
            != HIGH_DESK_CASES_PER_GENERATION
            or int(value.get("v5_case_count", -1)) != V5_CASES
            or int(value.get("reused_canary_relational_case_count", -1)) != 16
            or int(value.get("reused_canary_v5_case_count", -1)) != 3
            or int(value.get("paired_evaluation_reverse_draws_generated", -1)) != 612
            or int(value.get("repeatability_reverse_draws_generated", -1)) != 2
            or int(value.get("total_reverse_draws_generated", -1)) != 614
            or value.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 full K=3 protocol changed")

    resolved = {key: bound_file(value, key) for key in (
        "k3_canary_summary", "k3_canary_predictions", "k1_summary",
        "k1_predictions", "candidate_checkpoint", "dataset_index", "v5_split",
        "stats_file", "original_checkpoint", "v5_checkpoint", "predictions",
    )}
    verify_source_binding(value, "Teacher-v7 full K=3", REQUIRED_SOURCE_KEYS)
    k3 = read_json(resolved["k3_canary_summary"])
    if (k3.get("schema") != K3_SCHEMA or k3.get("status") != "SELECTED_K3_PASS"
            or k3.get("failed_checks") != []
            or int(k3.get("selected_step", -1)) != SELECTED_STEP
            or int(k3.get("seed", -1)) != SEED
            or k3.get("authorizes_train_only_full_k3") is not True
            or k3.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 K=3 authorization changed")
    verify_source_binding(k3, "Teacher-v7 selected K=3")
    for key in (
        "k1_summary", "k1_predictions", "candidate_checkpoint", "dataset_index",
        "v5_split", "stats_file", "original_checkpoint", "v5_checkpoint",
    ):
        if k3.get(key + "_sha256") != value.get(key + "_sha256"):
            raise ValueError("Teacher-v7 full K=3 input changed: " + key)
    if k3.get("predictions_sha256") != value.get("k3_canary_predictions_sha256"):
        raise ValueError("Teacher-v7 K=3 prediction chain changed")

    with np.load(resolved["k1_predictions"], allow_pickle=False) as source:
        canary_reference = {name: source[name] for name in source.files}
    with np.load(resolved["k3_canary_predictions"], allow_pickle=False) as source:
        canary_maps = {name: source[name] for name in source.files}
    with np.load(resolved["predictions"], allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    expected_keys = {
        "rel_case_ids", "rel_scene_ids", "rel_target_names", "rel_strata",
        "rel_motion_ids", "rel_prompt_ids", "rel_target_instance_ids", "rel_gt",
        "rel_instance_ids", "rel_category_ids", "v5_ids", "v5_targets", "v5_gt",
        "relational_base", "relational_candidate", "relational_seeds",
        "v5_base", "v5_candidate", "v5_seeds",
    }
    if set(arrays) != expected_keys:
        raise ValueError("Teacher-v7 full K=3 prediction keys changed")
    for name in ("rel_gt",):
        if arrays[name].shape != (RELATIONAL_CASES, 8192, 6):
            raise ValueError(name + " shape changed")
    for name in ("rel_instance_ids", "rel_category_ids"):
        if arrays[name].shape != (RELATIONAL_CASES, 8192):
            raise ValueError(name + " shape changed")
    for name in ("relational_base", "relational_candidate"):
        if arrays[name].shape != (RELATIONAL_CASES, GENERATIONS, 8192, 6):
            raise ValueError(name + " shape changed")
    for name in ("v5_gt",):
        if arrays[name].shape != (V5_CASES, 8192, 6):
            raise ValueError(name + " shape changed")
    for name in ("v5_base", "v5_candidate"):
        if arrays[name].shape != (V5_CASES, GENERATIONS, 8192, 6):
            raise ValueError(name + " shape changed")
    if (arrays["relational_seeds"].shape != (RELATIONAL_CASES, GENERATIONS, 2)
            or arrays["v5_seeds"].shape != (V5_CASES, GENERATIONS, 2)):
        raise ValueError("Teacher-v7 full K=3 seed table shape changed")
    for name in ("rel_gt", "relational_base", "relational_candidate", "v5_gt",
                 "v5_base", "v5_candidate"):
        array = arrays[name]
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(name + " range changed")

    dataset_root = resolved["dataset_index"].parent
    index = read_json(resolved["dataset_index"])
    grouped = group_relational_rows(load_relational_rows(dataset_root, index))
    mean, std = load_stats(resolved["stats_file"])
    v5_root = resolved["v5_split"].parent.parent
    v5_rows = load_v5_rows(
        v5_root, load_split(resolved["v5_split"]), "train", mean, std,
        4.0, 16.0, 0.7,
    )
    reference, _, _ = build_reference(dataset_root, index, grouped, v5_rows)
    for name in reference:
        assert_array_equal(arrays[name], reference[name], name + " source binding")
    if Counter(str(item) for item in arrays["rel_scene_ids"].tolist()) != Counter({
        "room_0101": 48, "room_0102": 48,
    }):
        raise ValueError("Teacher-v7 full train scene balance changed")
    if Counter(str(item) for item in arrays["rel_strata"].tolist()) != Counter({
        "bed_01": 24, "high_desk_chair_06": 24, "legacy_chair_06": 24,
        "chair_01": 12, "chair_05": 12,
    }):
        raise ValueError("Teacher-v7 full train stratum balance changed")
    if Counter(str(item) for item in arrays["rel_prompt_ids"].tolist()) != Counter({
        prompt_id: TRAIN_MOTIONS for prompt_id in PROMPTS
    }):
        raise ValueError("Teacher-v7 full train prompt balance changed")
    if Counter(str(item) for item in arrays["v5_targets"].tolist()) != Counter({
        "chair": 18, "bed": 1, "whiteboard": 6,
    }):
        raise ValueError("Teacher-v7 full train v5 target balance changed")

    canary_rel_lookup = {
        str(case_id): index_value
        for index_value, case_id in enumerate(canary_reference["rel_case_ids"].tolist())
    }
    canary_v5_lookup = {
        str(sample_id): index_value
        for index_value, sample_id in enumerate(canary_reference["v5_ids"].tolist())
    }
    reused_relational = 0
    for index_value in range(RELATIONAL_CASES):
        scene_id = str(arrays["rel_scene_ids"][index_value])
        stratum = str(arrays["rel_strata"][index_value])
        motion_id = str(arrays["rel_motion_ids"][index_value])
        prompt_id = str(arrays["rel_prompt_ids"][index_value])
        first_motion = str(grouped[scene_id][stratum][0]["motion_id"])
        if motion_id == first_motion:
            source_index = canary_rel_lookup[f"{scene_id}|{stratum}|{prompt_id}"]
            for name, source_name in (
                ("relational_base", "relational_base"),
                ("relational_candidate", "relational_candidate"),
                ("relational_seeds", "relational_seeds"),
            ):
                if not np.array_equal(arrays[name][index_value], canary_maps[source_name][source_index]):
                    raise ValueError("sealed relational canary map/seed changed")
            reused_relational += 1
            continue
        pair_id = f"{scene_id}|{stratum}|{motion_id}"
        for generation in range(GENERATIONS):
            expected_seed = generation_seeds(
                SEED, "full_relational_pair", pair_id, generation
            )
            actual = tuple(int(raw) for raw in arrays["relational_seeds"][index_value, generation])
            if actual != expected_seed:
                raise ValueError("Teacher-v7 full relational seed table changed")

    reused_v5 = 0
    for index_value, raw_sample_id in enumerate(arrays["v5_ids"].tolist()):
        sample_id = str(raw_sample_id)
        if sample_id in canary_v5_lookup:
            source_index = canary_v5_lookup[sample_id]
            for name, source_name in (
                ("v5_base", "v5_base"), ("v5_candidate", "v5_candidate"),
                ("v5_seeds", "v5_seeds"),
            ):
                if not np.array_equal(arrays[name][index_value], canary_maps[source_name][source_index]):
                    raise ValueError("sealed v5 canary map/seed changed")
            reused_v5 += 1
            continue
        for generation in range(GENERATIONS):
            expected_seed = generation_seeds(
                SEED, "full_v5_replay", sample_id, generation
            )
            actual = tuple(int(raw) for raw in arrays["v5_seeds"][index_value, generation])
            if actual != expected_seed:
                raise ValueError("Teacher-v7 full v5 seed table changed")
    if reused_relational != 16 or reused_v5 != 3:
        raise ValueError("Teacher-v7 full K=3 canary reuse count changed")

    generations = per_generation_metrics(
        reference, arrays["relational_base"], arrays["relational_candidate"],
        arrays["v5_base"], arrays["v5_candidate"],
    )
    pooled = pooled_metrics(generations)
    nested_close(generations, value.get("per_generation_metrics"), "per_generation_metrics")
    nested_close(pooled, value.get("pooled_metrics"), "pooled_metrics")
    if value.get("repeatability") != {"base": True, "candidate": True}:
        raise ValueError("Teacher-v7 full K=3 repeatability changed")
    checks = full_checks(pooled, generations, True)
    checks.update({
        "sealed_16_relational_and_3_v5_cases_reused": True,
        "all_48_train_motions_have_two_prompts_and_three_generations": (
            len(set(zip(
                reference["rel_scene_ids"].tolist(),
                reference["rel_motion_ids"].tolist(),
            ))) == TRAIN_MOTIONS
            and arrays["relational_base"].shape[:2] == (RELATIONAL_CASES, GENERATIONS)
        ),
        "high_desk_inventory_is_24_cases_per_generation": all(
            metrics["high_desk"]["count"] == HIGH_DESK_CASES_PER_GENERATION
            for metrics in generations
        ),
        "development_payloads_unread": value.get("development_payloads_read") is False,
    })
    if value.get("checks") != checks:
        raise ValueError("Teacher-v7 full K=3 checks changed")
    failed = [name for name, passed in checks.items() if not passed]
    if value.get("failed_checks") != failed:
        raise ValueError("Teacher-v7 full K=3 failed-check list changed")
    expected_status = "FULL_TRAIN_K3_PASS" if not failed else "FULL_TRAIN_K3_FAIL"
    if value.get("status") != expected_status:
        raise ValueError("Teacher-v7 full K=3 status/checks disagree")
    passed = expected_status == "FULL_TRAIN_K3_PASS"
    if value.get("diagnostic_only") is not (not passed):
        raise ValueError("Teacher-v7 full K=3 diagnostic authority changed")
    if value.get("checkpoint_locked") is not passed:
        raise ValueError("Teacher-v7 checkpoint lock authority changed")
    if value.get("authorizes_internal_development_evaluation") is not passed:
        raise ValueError("Teacher-v7 development authority changed")
    if value.get("authorizes_paper_test") is not False:
        raise ValueError("Teacher-v7 full K=3 overstates paper-test authority")

    print(f"[{expected_status}] Teacher-v7 selected step-12 full train-only K=3 integrity")
    print("[PASS] 48 motions x 2 prompts x K=3 maps and 25 v5 replay rows recomputed")
    print("[PASS] sealed K=3 canary reuse and all remaining seed tables verified")
    print("[PASS] source/checkpoint bindings and room_0201 unread guard")
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
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

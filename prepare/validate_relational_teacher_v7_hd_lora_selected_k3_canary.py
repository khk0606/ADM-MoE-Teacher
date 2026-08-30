#!/usr/bin/env python3
"""Validate the selected Teacher-v7 step-12 train-only K=3 canary."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from relational_teacher_v6_contract import read_json, sha256_file
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
from relational_teacher_v7_hd_rollout_contract import SCHEMA as K1_SCHEMA


REQUIRED_SOURCE_KEYS = {"evaluator", "validator", "k3_common", "k3_contract"}


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
        raise ValueError("Teacher-v7 selected K=3 artifact changed: " + key)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v7 selected K=3 schema changed")
    if value.get("status") not in {"SELECTED_K3_PASS", "SELECTED_K3_FAIL"}:
        raise ValueError("Teacher-v7 selected K=3 status changed")
    if (value.get("partition") != "train_canary"
            or int(value.get("selected_step", -1)) != SELECTED_STEP
            or int(value.get("diffusion_steps", -1)) != DIFFUSION_STEPS
            or int(value.get("num_generations", -1)) != GENERATIONS
            or int(value.get("seed", -1)) != SEED
            or value.get("seed_policy") != SEED_POLICY
            or value.get("paired_initial_and_reverse_noise") is not True
            or value.get("generation_0_reused_from_k1") is not True
            or value.get("new_generation_indices") != [1, 2]
            or int(value.get("relational_case_count", -1)) != RELATIONAL_CASES
            or int(value.get("high_desk_case_count_per_generation", -1)) != 4
            or int(value.get("v5_case_count", -1)) != V5_CASES
            or int(value.get("paired_evaluation_reverse_draws_generated", -1)) != 76
            or int(value.get("repeatability_reverse_draws_generated", -1)) != 2
            or int(value.get("total_reverse_draws_generated", -1)) != 78
            or value.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 selected K=3 protocol changed")

    resolved = {key: bound_file(value, key) for key in (
        "k1_summary", "k1_predictions", "candidate_checkpoint", "dataset_index",
        "v5_split", "stats_file", "original_checkpoint", "v5_checkpoint",
        "predictions",
    )}
    verify_source_binding(value, "Teacher-v7 selected K=3", REQUIRED_SOURCE_KEYS)
    k1 = read_json(resolved["k1_summary"])
    if (k1.get("schema") != K1_SCHEMA or k1.get("status") != "CANARY_PASS"
            or k1.get("failed_checks") != []
            or int(k1.get("selected_step", -1)) != SELECTED_STEP
            or int(k1.get("seed", -1)) != SEED
            or k1.get("authorizes_train_only_k3_canary") is not True
            or k1.get("development_payloads_read") is not False):
        raise ValueError("Teacher-v7 K=1 authorization changed")
    verify_source_binding(k1, "Teacher-v7 K=1 canary")
    if k1.get("predictions_sha256") != value.get("k1_predictions_sha256"):
        raise ValueError("Teacher-v7 K=1 prediction chain changed")
    for key in ("candidate_checkpoint", "dataset_index", "v5_split", "stats_file",
                "original_checkpoint", "v5_checkpoint"):
        if k1.get(key + "_sha256") != value.get(key + "_sha256"):
            raise ValueError("Teacher-v7 selected K=3 input changed: " + key)

    with np.load(resolved["k1_predictions"], allow_pickle=False) as source:
        reference = {name: source[name] for name in source.files}
    with np.load(resolved["predictions"], allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    expected = {
        "relational_base", "relational_candidate", "relational_seeds",
        "v5_base", "v5_candidate", "v5_seeds",
    }
    if set(arrays) != expected:
        raise ValueError("Teacher-v7 selected K=3 prediction keys changed")
    if (arrays["relational_base"].shape != (RELATIONAL_CASES, GENERATIONS, 8192, 6)
            or arrays["relational_candidate"].shape != (RELATIONAL_CASES, GENERATIONS, 8192, 6)
            or arrays["v5_base"].shape != (V5_CASES, GENERATIONS, 8192, 6)
            or arrays["v5_candidate"].shape != (V5_CASES, GENERATIONS, 8192, 6)
            or arrays["relational_seeds"].shape != (RELATIONAL_CASES, GENERATIONS, 2)
            or arrays["v5_seeds"].shape != (V5_CASES, GENERATIONS, 2)):
        raise ValueError("Teacher-v7 selected K=3 array shape changed")
    for name in ("relational_base", "relational_candidate", "v5_base", "v5_candidate"):
        array = arrays[name]
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("Teacher-v7 selected K=3 range changed: " + name)
    if not np.array_equal(arrays["relational_base"][:, 0], reference["rel_base"]):
        raise ValueError("Teacher-v7 relational generation 0 Base changed")
    if not np.array_equal(arrays["relational_candidate"][:, 0], reference["rel_candidate"]):
        raise ValueError("Teacher-v7 relational generation 0 candidate changed")
    if not np.array_equal(arrays["v5_base"][:, 0], reference["v5_base"]):
        raise ValueError("Teacher-v7 v5 generation 0 Base changed")
    if not np.array_equal(arrays["v5_candidate"][:, 0], reference["v5_candidate"]):
        raise ValueError("Teacher-v7 v5 generation 0 candidate changed")

    scene_ids = [str(value) for value in reference["rel_scene_ids"].tolist()]
    strata = [str(value) for value in reference["rel_strata"].tolist()]
    v5_ids = [str(value) for value in reference["v5_ids"].tolist()]
    for index, (scene_id, stratum) in enumerate(zip(scene_ids, strata)):
        pair_id = f"{scene_id}|{stratum}"
        for generation in range(GENERATIONS):
            expected_seed = generation_seeds(SEED, "relational_pair", pair_id, generation)
            actual = tuple(int(raw) for raw in arrays["relational_seeds"][index, generation])
            if actual != expected_seed:
                raise ValueError("Teacher-v7 relational seed table changed")
    for index, sample_id in enumerate(v5_ids):
        for generation in range(GENERATIONS):
            expected_seed = generation_seeds(SEED, "v5_replay", sample_id, generation)
            actual = tuple(int(raw) for raw in arrays["v5_seeds"][index, generation])
            if actual != expected_seed:
                raise ValueError("Teacher-v7 v5 seed table changed")

    generations = per_generation_metrics(
        reference, arrays["relational_base"], arrays["relational_candidate"],
        arrays["v5_base"], arrays["v5_candidate"],
    )
    pooled = pooled_metrics(generations)
    nested_close(generations, value.get("per_generation_metrics"), "per_generation_metrics")
    nested_close(pooled, value.get("pooled_metrics"), "pooled_metrics")
    nested_close(generations[0], k1.get("metrics"), "generation0_k1_metrics")
    if value.get("repeatability") != {"base": True, "candidate": True}:
        raise ValueError("Teacher-v7 selected K=3 repeatability changed")
    checks = k3_checks(pooled, generations, True)
    checks.update({
        "generation_0_exactly_reuses_sealed_k1": True,
        "all_16_relational_and_3_v5_cases_have_three_generations": (
            arrays["relational_base"].shape[:2] == (RELATIONAL_CASES, GENERATIONS)
            and arrays["v5_base"].shape[:2] == (V5_CASES, GENERATIONS)
        ),
        "development_payloads_unread": value.get("development_payloads_read") is False,
    })
    if value.get("checks") != checks:
        raise ValueError("Teacher-v7 selected K=3 checks changed")
    failed = [name for name, passed in checks.items() if not passed]
    if value.get("failed_checks") != failed:
        raise ValueError("Teacher-v7 selected K=3 failed-check list changed")
    expected_status = "SELECTED_K3_PASS" if not failed else "SELECTED_K3_FAIL"
    if value.get("status") != expected_status:
        raise ValueError("Teacher-v7 selected K=3 status/checks disagree")
    if value.get("diagnostic_only") is not True:
        raise ValueError("Teacher-v7 selected K=3 overstates authority")
    if value.get("authorizes_train_only_full_k3") is not (expected_status == "SELECTED_K3_PASS"):
        raise ValueError("Teacher-v7 full train-only K=3 authority changed")
    for key in ("authorizes_development_evaluation", "authorizes_paper_test"):
        if value.get(key) is not False:
            raise ValueError("Teacher-v7 selected K=3 authority changed: " + key)

    print(f"[{expected_status}] Teacher-v7 selected step-12 K=3 canary integrity")
    print("[PASS] generation 0 exactly reuses sealed K=1 maps")
    print("[PASS] generation 1/2 seeds, maps and pooled metrics recomputed")
    print("[PASS] source/checkpoint bindings and held-out-unread guards")
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

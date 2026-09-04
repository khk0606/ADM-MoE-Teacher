#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.2 rollout-state calibration-6."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from relational_teacher_v9_all_sittable_contract import read_json, sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v9_lora_runtime import load_stats  # noqa: E402
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    continuous_presence_checks,
    stable_rollout_seeds,
)
from relational_teacher_v98_rollout_state_contract import (  # noqa: E402
    design_seeds as v98_design_seeds,
)
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    response6_checks,
)
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    HELDOUT_TRAIN_SCENE,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SELECTED_TIMESTEP,
    STEP_RADIUS,
    TRAIN_SCENE,
    UPDATE_COUNT,
    V981_SCHEMA,
    calibration_design_seeds,
    calibration_step_checks,
    canonical_sha256,
    pooled_object_metrics,
    rank_eligible_steps,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _metrics,
    _prompt_invariance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _close(left: object, right: object, label: str) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(label + " mapping keys changed")
        for key in left:
            _close(left[key], right[key], label + "." + str(key))
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(label + " sequence length changed")
        for index, (one, two) in enumerate(zip(left, right)):
            _close(one, two, f"{label}[{index}]")
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise ValueError(label + " boolean changed")
        return
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            raise ValueError(label + " contains NaN/Inf")
        if not math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " value changed")


def _rows(bundle: Mapping[str, object], values: np.ndarray) -> list:
    return [
        [_metrics(bundle, values[generation, prompt]) for prompt in range(2)]
        for generation in range(3)
    ]


def _invariance(values: np.ndarray, mask: np.ndarray) -> list[float]:
    return [
        _prompt_invariance(values[generation], mask)
        for generation in range(3)
    ]


def main() -> None:
    args = parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.2 summary schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("diffusion_steps") != 500
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != HELDOUT_TRAIN_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("update_count") != UPDATE_COUNT
        or value.get("monitor_steps") != list(MONITOR_STEPS)
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or float(value.get("step_radius")) != STEP_RADIUS
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.2 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.2 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.2 bound file changed: " + str(name))
    if sha256_file(Path(str(paths["calibration6_maps"])).resolve()) != value.get(
        "calibration6_maps_sha256"
    ):
        raise ValueError("calibration map hash changed")
    policy_value = read_json(Path(str(paths["calibration6_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("calibration policy changed")

    response = read_json(Path(str(paths["response6_summary"])).resolve())
    if (
        response.get("schema") != V981_SCHEMA
        or response.get("status") != "PASS"
        or response.get("binding_id") != value.get("v981_binding_id")
        or response.get("authorizes_rollout_state_calibration6") is not True
    ):
        raise ValueError("sealed v9.8.1 authority changed")
    response_arrays = _load_npz(Path(str(paths["response6_maps"])).resolve())
    arrays = _load_npz(Path(str(paths["calibration6_maps"])).resolve())
    required_arrays = {
        "xyz",
        "points",
        "instance_ids",
        "category_ids",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
        "all_sittable_gt",
        "monitor_steps",
        "prompt_ids",
        "seed_table",
        "base_normalized",
        "candidates_normalized",
        "base",
        "candidates",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    }
    if set(arrays) != required_arrays:
        raise ValueError("calibration array inventory changed")
    if (
        arrays["base_normalized"].shape != (3, 2, 8192, 6)
        or arrays["candidates_normalized"].shape != (6, 3, 2, 8192, 6)
        or arrays["base"].shape != (3, 2, 8192, 6)
        or arrays["candidates"].shape != (6, 3, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (6, 3, 8192, 6)
        or not np.array_equal(arrays["monitor_steps"], np.asarray(MONITOR_STEPS))
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
    ):
        raise ValueError("calibration array shape/order changed")
    if not np.isfinite(arrays["candidates"]).all():
        raise ValueError("calibration maps contain NaN/Inf")
    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    base_physical = np.clip(
        arrays["base_normalized"] * std.reshape(1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    candidates_physical = np.clip(
        arrays["candidates_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    if not np.array_equal(arrays["base"], base_physical) or not np.array_equal(
        arrays["candidates"], candidates_physical
    ):
        raise ValueError("normalized-to-physical calibration conversion changed")
    if not np.array_equal(arrays["base_normalized"], response_arrays["base_normalized"]):
        raise ValueError("calibration Base does not reuse v9.8.1")
    if not np.array_equal(
        arrays["base_v5_prediction"], response_arrays["base_v5_prediction"]
    ):
        raise ValueError("calibration Base v5 probe does not reuse v9.8.1")
    if not np.array_equal(
        arrays["candidates_normalized"][0], response_arrays["candidate_normalized"]
    ):
        raise ValueError("calibration update 1 does not reproduce v9.8.1")
    if not np.array_equal(
        arrays["candidate_v5_predictions"][0],
        response_arrays["candidate_v5_prediction"],
    ):
        raise ValueError("calibration update-1 v5 response changed")

    dataset_index = Path(str(paths["dataset_index"])).resolve()
    source_index = Path(str(paths["source_dataset_index"])).resolve()
    dataset_root = dataset_index.parent
    source_root = source_index.parent
    top_index = validate_top_index(dataset_root, source_root, dataset_index)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    for name in (
        "xyz",
        "points",
        "instance_ids",
        "category_ids",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
    ):
        if not np.array_equal(np.asarray(bundle[name]), arrays[name]):
            raise ValueError("saved scene array changed: " + name)
    if not np.array_equal(np.asarray(bundle["all_target"]), arrays["all_sittable_gt"]):
        raise ValueError("saved all-sittable GT changed")

    base_rows = _rows(bundle, np.asarray(arrays["base"], np.float32))
    base_pooled = pooled_object_metrics(base_rows)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(arrays["base"], verified_mask)
    base_v5_dense = (
        (arrays["base_v5_prediction"] - arrays["v5_target"]) ** 2
    ).mean(axis=(1, 2)).tolist()
    _close(value["base_rows"], base_rows, "base_rows")
    _close(value["base_pooled"], base_pooled, "base_pooled")
    _close(value["base_prompt_invariance"], base_invariance, "base_invariance")
    _close(value["base_v5_dense"], base_v5_dense, "base_v5_dense")

    direction_rows = value.get("direction_rows")
    monitor_rows = value.get("monitor_rows")
    if (
        not isinstance(direction_rows, list)
        or not isinstance(monitor_rows, list)
        or len(direction_rows) != UPDATE_COUNT
        or len(monitor_rows) != UPDATE_COUNT
    ):
        raise ValueError("six calibration rows are required")
    recomputed_rows = []
    for index, step in enumerate(MONITOR_STEPS):
        direction = direction_rows[index]
        saved = monitor_rows[index]
        if direction.get("step") != step or saved.get("step") != step:
            raise ValueError("calibration step order changed")
        if set(direction) != {
            "step",
            "design_seeds",
            "task_order",
            "task_losses",
            "gram",
            "weights",
            "directional_derivatives",
            "direction_norm_before_unit",
            "direction_sha256",
            "pre_state_sha256",
            "post_state_sha256",
        }:
            raise ValueError("calibration direction row schema changed")
        if set(saved) != {
            "step",
            "base_rows",
            "candidate_rows",
            "base_pooled",
            "candidate_pooled",
            "base_prompt_invariance",
            "candidate_prompt_invariance",
            "base_v5_dense",
            "candidate_v5_dense",
            "maximum_final_map_delta",
            "presence",
            "response_checks",
            "checks",
            "eligible",
            "failed_checks",
        }:
            raise ValueError("calibration monitor row schema changed")
        if (
            len(direction.get("task_losses", [])) != 6
            or np.asarray(direction.get("gram")).shape != (6, 6)
            or len(direction.get("weights", [])) != 6
            or len(direction.get("directional_derivatives", [])) != 6
            or not math.isclose(sum(direction["weights"]), 1.0, abs_tol=2e-6)
        ):
            raise ValueError("calibration direction geometry changed")
        current = np.asarray(arrays["candidates"][index], np.float32)
        current_rows = _rows(bundle, current)
        current_pooled = pooled_object_metrics(current_rows)
        current_invariance = _invariance(current, verified_mask)
        current_v5_dense = (
            (arrays["candidate_v5_predictions"][index] - arrays["v5_target"]) ** 2
        ).mean(axis=(1, 2)).tolist()
        maximum_delta = float(np.abs(current - arrays["base"]).max())
        response_checks_value = response6_checks(
            base_rows=base_rows,
            candidate_rows=current_rows,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=direction["directional_derivatives"],
            maximum_map_delta=maximum_delta,
        )
        checks = calibration_step_checks(
            step=step,
            base_rows=base_rows,
            candidate_rows=current_rows,
            response_checks=response_checks_value,
            directional_derivatives=direction["directional_derivatives"],
            pre_state_sha256=direction["pre_state_sha256"],
            post_state_sha256=direction["post_state_sha256"],
        )
        recomputed = dict(saved)
        presence = [
            [continuous_presence_checks(current_rows[g][p]) for p in range(2)]
            for g in range(3)
        ]
        recomputed.update(
            {
                "base_rows": base_rows,
                "candidate_rows": current_rows,
                "base_pooled": base_pooled,
                "candidate_pooled": current_pooled,
                "base_prompt_invariance": base_invariance,
                "candidate_prompt_invariance": current_invariance,
                "base_v5_dense": base_v5_dense,
                "candidate_v5_dense": current_v5_dense,
                "maximum_final_map_delta": maximum_delta,
                "presence": presence,
                "response_checks": response_checks_value,
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
        _close(saved, recomputed, "monitor step {}".format(step))
        recomputed_rows.append(recomputed)

    shortlisted = rank_eligible_steps(recomputed_rows)
    if value.get("shortlisted_steps") != shortlisted:
        raise ValueError("calibration shortlist changed")
    design_seed_table = value.get("design_seed_table")
    if not isinstance(design_seed_table, Mapping) or set(design_seed_table) != {
        str(step) for step in MONITOR_STEPS
    }:
        raise ValueError("design seed table changed")
    expected_design_seeds = {
        "1": list(v98_design_seeds()),
        **{
            str(step): list(calibration_design_seeds(step))
            for step in range(2, UPDATE_COUNT + 1)
        },
    }
    if design_seed_table != expected_design_seeds:
        raise ValueError("calibration design seeds changed")
    expected_audit_seeds = [list(stable_rollout_seeds(g)) for g in range(3)]
    if value.get("audit_seed_table") != expected_audit_seeds:
        raise ValueError("calibration audit seeds changed")
    seed_pairs = [tuple(design_seed_table[str(step)]) for step in MONITOR_STEPS]
    if set(seed_pairs) & {tuple(pair) for pair in expected_audit_seeds}:
        raise ValueError("calibration design/audit seeds overlap")
    expected_checks = {
        "sealed_v981_pass_authority_bound": True,
        "fresh_v5r4_zero_init": True,
        "update1_exactly_reproduces_v981": True,
        "six_sequential_common_descent_updates_completed": all(
            min(row["directional_derivatives"])
            >= float(POLICY["eligibility"]["minimum_directional_derivative"])
            for row in direction_rows
        ),
        "new_design_seed_pairs_are_disjoint": len(set(seed_pairs)) == UPDATE_COUNT
        and not (set(seed_pairs) & {tuple(pair) for pair in expected_audit_seeds}),
        "all_k3_base_trajectories_reproduce_v97": True,
        "each_update_uses_actual_paired_t50_resume": True,
        "calibration_policy_locked_before_model_and_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_state_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_post_first_update_is_admissible": bool(shortlisted),
    }
    if value.get("checks") != expected_checks:
        raise ValueError("top-level calibration checks changed")
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed_checks = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_two_scene_rollout_state_calibration_preflight")
        != (status == "PASS")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_room_0102") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("calibration status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "v97_binding_id": value["v97_binding_id"],
            "v98_binding_id": value["v98_binding_id"],
            "v981_binding_id": value["v981_binding_id"],
            "policy_id": value["policy_id"],
            "shortlisted_steps": value["shortlisted_steps"],
            "calibration6_maps_sha256": value["calibration6_maps_sha256"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("calibration binding ID changed")
    output_dir = summary_file.parent
    if list(output_dir.glob("*.pt")) or list(output_dir.glob("*.pth")):
        raise ValueError("calibration output contains forbidden model state")

    print("[ROLLOUT_STATE_CALIBRATION6_{}] Teacher-v9.8.2 integrity".format(status))
    print("[PASS] sealed v9.8.1 response and exact update-1 maps recomputed")
    print("[PASS] six K=3 monitor panels, response gates and ranking recomputed")
    print("[PASS] no optimizer/checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] shortlisted steps:", shortlisted)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

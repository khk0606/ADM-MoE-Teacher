#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.4 preservation-aware calibration-6."""

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
from relational_teacher_v9_lora_runtime import FORWARD_INPUT_KEYS  # noqa: E402
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    continuous_presence_checks,
    stable_rollout_seeds,
)
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    response6_checks,
)
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    calibration_design_seeds,
)
from relational_teacher_v983_preservation_direction_contract import (  # noqa: E402
    POLICY_ID as V983_POLICY_ID,
    SCHEMA as V983_REPORT_SCHEMA,
)
from relational_teacher_v984_preservation_calibration6_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    HELDOUT_TRAIN_SCENE,
    MODEL_SEED,
    MONITOR_STEPS,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SELECTED_TIMESTEP,
    SELECTED_V983_CANDIDATE,
    STEP_RADIUS,
    TASK_ORDER,
    TRAIN_SCENE,
    UPDATE_COUNT,
    calibration_checks,
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
    if values.shape != (3, 2, 8192, 6):
        raise ValueError("calibration map shape changed")
    return [
        [_metrics(bundle, values[generation, prompt]) for prompt in range(2)]
        for generation in range(3)
    ]


def _invariance(values: np.ndarray, mask: np.ndarray) -> list[float]:
    return [
        _prompt_invariance(values[generation], mask)
        for generation in range(3)
    ]


def _validate_direction_geometry(row: Mapping[str, object], expected_count: int) -> None:
    expected_keys = {
        "step",
        "task_order",
        "task_losses",
        "gram",
        "weights",
        "directional_derivatives",
        "direction_norm_before_unit",
        "direction_sha256",
        "pre_state_sha256",
        "post_state_sha256",
        "design_seeds",
    }
    if set(row) != expected_keys:
        raise ValueError("calibration direction schema changed")
    gram = np.asarray(row["gram"], np.float64)
    weights = np.asarray(row["weights"], np.float64)
    derivatives = np.asarray(row["directional_derivatives"], np.float64)
    losses = np.asarray(row["task_losses"], np.float64)
    norm = float(row["direction_norm_before_unit"])
    if (
        losses.shape != (expected_count,)
        or gram.shape != (expected_count, expected_count)
        or weights.shape != (expected_count,)
        or derivatives.shape != (expected_count,)
        or not np.isfinite(losses).all()
        or not np.isfinite(gram).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(derivatives).all()
        or not np.allclose(gram, gram.T, atol=1e-7)
        or float(np.linalg.eigvalsh(gram).min()) < -1e-5
        or float(weights.min()) < -1e-10
        or not math.isclose(float(weights.sum()), 1.0, abs_tol=2e-6)
        or not math.isfinite(norm)
        or norm <= 0.0
        or float(derivatives.min()) < float(POLICY["minimum_directional_derivative"])
    ):
        raise ValueError("calibration direction geometry changed")
    expected_norm = math.sqrt(max(float(weights @ gram @ weights), 0.0))
    expected_derivatives = (gram @ weights) / expected_norm
    if not math.isclose(norm, expected_norm, rel_tol=2e-5, abs_tol=2e-7):
        raise ValueError("calibration direction norm changed")
    if not np.allclose(derivatives, expected_derivatives, rtol=2e-5, atol=2e-7):
        raise ValueError("calibration directional derivatives changed")
    for key in ("direction_sha256", "pre_state_sha256", "post_state_sha256"):
        if not isinstance(row.get(key), str) or len(row[key]) != 64:
            raise ValueError("calibration state/direction hash changed")
    if row["pre_state_sha256"] == row["post_state_sha256"]:
        raise ValueError("calibration direction did not change LoRA state")


def main() -> None:
    args = parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.4 summary schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("calibration_tag") != 20261020
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
        or value.get("task_order") != list(TASK_ORDER)
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("design_seed_table")
        != {
            str(step): list(calibration_design_seeds(step))
            for step in range(2, UPDATE_COUNT + 1)
        }
        or value.get("audit_seed_table")
        != [list(stable_rollout_seeds(generation)) for generation in range(3)]
        or value.get("trajectory_accounting")
        != {
            "audit_base_full_draws": 6,
            "design_base_full_draws": 10,
            "design_exact_partial_repeats": 10,
            "candidate_monitor_partial_resumes": 36,
        }
    ):
        raise ValueError("Teacher-v9.8.4 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.4 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.4 bound file changed: " + str(name))
    maps_file = Path(str(paths["calibration_maps"])).resolve()
    if sha256_file(maps_file) != value.get("calibration_maps_sha256"):
        raise ValueError("calibration map hash changed")
    policy_value = read_json(Path(str(paths["calibration_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("preservation calibration policy changed")

    v983 = read_json(Path(str(paths["v983_report"])).resolve())
    if (
        v983.get("schema") != V983_REPORT_SCHEMA
        or v983.get("status") != "PASS"
        or v983.get("policy_id") != V983_POLICY_ID
        or v983.get("binding_id") != value.get("v983_binding_id")
        or v983.get("selected_candidate") != SELECTED_V983_CANDIDATE
        or v983.get("failed_checks")
        or v983.get("authorizes_preservation_aware_calibration6") is not True
    ):
        raise ValueError("sealed v9.8.3 PASS changed")
    v983_paths = v983.get("paths")
    v983_hashes = v983.get("path_sha256")
    if (
        not isinstance(v983_paths, Mapping)
        or not isinstance(v983_hashes, Mapping)
        or set(v983_paths) != set(v983_hashes)
    ):
        raise ValueError("sealed v9.8.3 path binding changed")
    for name, raw in v983_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != v983_hashes[name]:
            raise ValueError("sealed v9.8.3 bound file changed: " + str(name))
    if list(Path(str(paths["v983_report"])).resolve().parent.glob("*.pt")) or list(
        Path(str(paths["v983_report"])).resolve().parent.glob("*.pth")
    ):
        raise ValueError("sealed v9.8.3 contains forbidden model state")

    preflight = read_json(Path(str(paths["preflight_report"])).resolve())
    v97 = read_json(Path(str(paths["v97_summary"])).resolve())
    v98 = read_json(Path(str(paths["v98_report"])).resolve())
    response = read_json(Path(str(paths["response6_summary"])).resolve())
    if (
        value.get("preflight_binding_id") != preflight.get("binding_id")
        or value.get("v97_binding_id") != v97.get("binding_id")
        or value.get("v98_binding_id") != v98.get("binding_id")
        or value.get("v981_binding_id") != response.get("binding_id")
        or value.get("scene_binding") != v983.get("scene_binding")
    ):
        raise ValueError("nested Teacher-v9 authority binding changed")

    arrays = _load_npz(maps_file)
    v983_arrays = _load_npz(Path(str(paths["v983_maps"])).resolve())
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
        "audit_seed_table",
        "design_seed_table",
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
        or arrays["monitor_steps"].tolist() != list(MONITOR_STEPS)
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["audit_seed_table"].tolist()
        != [list(stable_rollout_seeds(g)) for g in range(3)]
        or arrays["design_seed_table"].tolist()
        != [list(calibration_design_seeds(step)) for step in range(2, 7)]
    ):
        raise ValueError("calibration array shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("calibration array contains NaN/Inf: " + name)

    selected_index = [
        index
        for index, row in enumerate(v983["candidates"])
        if row["name"] == SELECTED_V983_CANDIDATE
    ]
    if len(selected_index) != 1:
        raise ValueError("v9.8.3 selected candidate inventory changed")
    if not np.array_equal(arrays["base_normalized"], v983_arrays["base_normalized"]):
        raise ValueError("calibration Base differs from v9.8.3")
    if not np.array_equal(
        arrays["candidates_normalized"][0], v983_arrays["step1_normalized"]
    ):
        raise ValueError("calibration update 1 differs from v9.8.1")
    if not np.array_equal(
        arrays["candidates_normalized"][1],
        v983_arrays["candidates_normalized"][selected_index[0]],
    ):
        raise ValueError("calibration update 2 differs from selected v9.8.3")
    if not np.array_equal(arrays["base_v5_prediction"], v983_arrays["base_v5_prediction"]):
        raise ValueError("calibration Base v5 changed")
    if not np.array_equal(
        arrays["candidate_v5_predictions"][0], v983_arrays["step1_v5_prediction"]
    ):
        raise ValueError("calibration update-1 v5 changed")
    if not np.array_equal(
        arrays["candidate_v5_predictions"][1],
        v983_arrays["candidate_v5_predictions"][selected_index[0]],
    ):
        raise ValueError("calibration update-2 v5 changed")

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
        raise ValueError("normalized-to-physical conversion changed")

    dataset_index = Path(str(paths["dataset_index"])).resolve()
    source_index = Path(str(paths["source_dataset_index"])).resolve()
    top_index = validate_top_index(dataset_index.parent, source_index.parent, dataset_index)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {TRAIN_SCENE, HELDOUT_TRAIN_SCENE}:
        raise ValueError("train scene metadata changed")
    bundle = load_train_scene_bundle(
        dataset_index.parent, source_index.parent, records[TRAIN_SCENE]
    )
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

    base_rows = _rows(bundle, arrays["base"])
    base_pooled = pooled_object_metrics(base_rows)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(arrays["base"], verified_mask)
    base_v5_dense = (
        (arrays["base_v5_prediction"] - arrays["v5_target"]) ** 2
    ).mean(axis=(1, 2)).tolist()
    for key, actual in (
        ("base_rows", base_rows),
        ("base_pooled", base_pooled),
        ("base_prompt_invariance", base_invariance),
        ("base_v5_dense", base_v5_dense),
    ):
        _close(value[key], actual, key)

    direction_rows = value.get("direction_rows")
    monitor_rows = value.get("monitor_rows")
    if (
        not isinstance(direction_rows, list)
        or len(direction_rows) != UPDATE_COUNT
        or not isinstance(monitor_rows, list)
        or len(monitor_rows) != UPDATE_COUNT
    ):
        raise ValueError("six calibration direction/monitor rows are required")
    if (
        value.get("zero_state_sha256") != direction_rows[0].get("pre_state_sha256")
        or value.get("zero_state_sha256") != v983.get("zero_state_sha256")
        or direction_rows[0].get("post_state_sha256")
        != v983.get("step1_state_sha256")
    ):
        raise ValueError("zero-state chain changed")
    for index in range(1, UPDATE_COUNT):
        if direction_rows[index - 1].get("post_state_sha256") != direction_rows[index].get(
            "pre_state_sha256"
        ):
            raise ValueError("LoRA state chain changed")

    recomputed_rows = []
    previous_rows = base_rows
    previous_v5_dense = base_v5_dense
    direction_keys = list(TASK_ORDER)
    for index, step in enumerate(MONITOR_STEPS):
        direction = direction_rows[index]
        expected_count = 6 if step == 1 else len(TASK_ORDER)
        _validate_direction_geometry(direction, expected_count)
        expected_order = direction_keys[:6] if step == 1 else direction_keys
        expected_seeds = (
            list(v98["design_seeds"])
            if step == 1
            else list(calibration_design_seeds(step))
        )
        if (
            direction.get("step") != step
            or direction.get("task_order") != expected_order
            or direction.get("design_seeds") != expected_seeds
        ):
            raise ValueError("direction step/order/seed changed")
        if step == 1:
            expected_direction = response["direction"]
        elif step == 2:
            expected_direction = v983["direction"]
        else:
            expected_direction = None
        if expected_direction is not None:
            for key in (
                "task_losses",
                "gram",
                "weights",
                "directional_derivatives",
                "direction_norm_before_unit",
                "direction_sha256",
            ):
                _close(direction[key], expected_direction[key], f"direction{step}.{key}")

        current = arrays["candidates"][index]
        current_rows = _rows(bundle, current)
        current_pooled = pooled_object_metrics(current_rows)
        current_invariance = _invariance(current, verified_mask)
        current_v5_dense = (
            (arrays["candidate_v5_predictions"][index] - arrays["v5_target"]) ** 2
        ).mean(axis=(1, 2)).tolist()
        maximum_base_delta = float(np.abs(current - arrays["base"]).max())
        response_checks_value = response6_checks(
            base_rows=base_rows,
            candidate_rows=current_rows,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=direction["directional_derivatives"][:6],
            maximum_map_delta=maximum_base_delta,
        )
        checks = calibration_checks(
            step=step,
            previous_rows=previous_rows,
            candidate_rows=current_rows,
            previous_v5_dense=previous_v5_dense,
            candidate_v5_dense=current_v5_dense,
            response_checks=response_checks_value,
            directional_derivatives=direction["directional_derivatives"],
            pre_state_sha256=direction["pre_state_sha256"],
            post_state_sha256=direction["post_state_sha256"],
        )
        presence = [
            [continuous_presence_checks(current_rows[g][p]) for p in range(2)]
            for g in range(3)
        ]
        recomputed = {
            "step": step,
            "design_seeds": expected_seeds,
            "previous_rows": previous_rows,
            "candidate_rows": current_rows,
            "previous_pooled": pooled_object_metrics(previous_rows),
            "candidate_pooled": current_pooled,
            "base_prompt_invariance": base_invariance,
            "candidate_prompt_invariance": current_invariance,
            "base_v5_dense": base_v5_dense,
            "previous_v5_dense": previous_v5_dense,
            "candidate_v5_dense": current_v5_dense,
            "maximum_base_map_delta": maximum_base_delta,
            "presence": presence,
            "response_checks": response_checks_value,
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        _close(monitor_rows[index], recomputed, "monitor row " + str(step))
        recomputed_rows.append(recomputed)
        previous_rows = current_rows
        previous_v5_dense = current_v5_dense

    shortlist = rank_eligible_steps(recomputed_rows)
    expected_checks = {
        "sealed_v983_pass_and_selected_candidate_bound": True,
        "fresh_v5r4_zero_init": True,
        "update1_exactly_reproduces_v981": True,
        "update2_exactly_reproduces_selected_v983": True,
        "six_updates_and_actual_k3_monitors_completed": len(recomputed_rows) == UPDATE_COUNT,
        "all_directions_are_common_descent": all(
            row["checks"]["direction_is_common_descent"] for row in recomputed_rows
        ),
        "policy_locked_before_model_and_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_post_first_update_is_admissible": bool(shortlist),
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed_checks = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("shortlisted_steps") != shortlist
        or value.get("checks") != expected_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_two_scene_preservation_calibration_preflight")
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
            "v983_binding_id": value["v983_binding_id"],
            "policy_id": value["policy_id"],
            "shortlisted_steps": value["shortlisted_steps"],
            "calibration_maps_sha256": value["calibration_maps_sha256"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("calibration binding ID changed")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("calibration output contains forbidden model state")

    print(f"[PRESERVATION_CALIBRATION6_{status}] Teacher-v9.8.4 integrity")
    print("[PASS] sealed v9.8.3 and exact update-1/update-2 reproduction verified")
    print("[PASS] six directions, K=3 maps, v5 and incremental gates recomputed")
    print("[PASS] no optimizer/checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] shortlisted steps:", shortlist)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

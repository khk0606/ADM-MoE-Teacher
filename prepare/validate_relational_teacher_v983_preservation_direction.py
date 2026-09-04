#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.3 preservation-aware preflight."""

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
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    response6_checks,
)
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    POLICY_ID as V982_POLICY_ID,
    calibration_design_seeds,
)
from relational_teacher_v983_preservation_direction_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    HELDOUT_TRAIN_SCENE,
    MODEL_SEED,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SELECTED_TIMESTEP,
    STEP_RADII,
    TASK_ORDER,
    TRAIN_SCENE,
    V982_SCHEMA,
    canonical_sha256,
    pooled_object_metrics,
    preservation_checks,
    rank_candidates,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _metrics,
    _prompt_invariance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
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


def _validate_v982_nested_failure(value: Mapping[str, object]) -> None:
    expected = {
        1: [],
        2: ["v5_fixed_probe_retained_1pct"],
        3: ["v5_fixed_probe_retained_1pct"],
        4: ["generation_2_negative_mean_retained", "v5_fixed_probe_retained_1pct"],
        5: [
            "generation_0_negative_mean_retained",
            "generation_1_negative_mean_retained",
            "generation_2_negative_mean_retained",
            "v5_fixed_probe_retained_1pct",
        ],
        6: [
            "generation_0_negative_mean_retained",
            "generation_1_negative_mean_retained",
            "generation_2_negative_mean_retained",
            "v5_fixed_probe_retained_1pct",
        ],
    }
    rows = value.get("monitor_rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("sealed v9.8.2 monitor inventory changed")
    if [int(row.get("step", 0)) for row in rows] != list(range(1, 7)):
        raise ValueError("sealed v9.8.2 monitor order changed")
    for row in rows:
        step = int(row["step"])
        failures = sorted(
            name
            for name, passed in row.get("response_checks", {}).items()
            if passed is not True
        )
        if failures != expected[step]:
            raise ValueError("sealed v9.8.2 nested failure diagnosis changed")


def main() -> None:
    args = parse_args()
    report_file = args.report.expanduser().resolve()
    value = read_json(report_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.3 report schema changed")
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
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or value.get("step_radii") != list(STEP_RADII)
        or value.get("task_order") != list(TASK_ORDER)
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.3 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.3 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.3 bound file changed: " + str(name))
    maps_file = Path(str(paths["preservation_maps"])).resolve()
    if sha256_file(maps_file) != value.get("preservation_maps_sha256"):
        raise ValueError("preservation map hash changed")
    policy_value = read_json(Path(str(paths["preservation_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("preservation policy changed")

    failed = read_json(Path(str(paths["failed_calibration_summary"])).resolve())
    if (
        failed.get("schema") != V982_SCHEMA
        or failed.get("status") != "FAIL"
        or failed.get("policy_id") != V982_POLICY_ID
        or failed.get("binding_id") != value.get("v982_binding_id")
        or failed.get("failed_checks")
        != ["at_least_one_post_first_update_is_admissible"]
    ):
        raise ValueError("sealed v9.8.2 failure changed")
    _validate_v982_nested_failure(failed)
    failed_paths = failed.get("paths")
    failed_hashes = failed.get("path_sha256")
    if (
        not isinstance(failed_paths, Mapping)
        or not isinstance(failed_hashes, Mapping)
        or set(failed_paths) != set(failed_hashes)
    ):
        raise ValueError("sealed v9.8.2 path binding changed")
    for name, raw in failed_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != failed_hashes[name]:
            raise ValueError("sealed v9.8.2 bound file changed: " + str(name))
    if list(Path(str(paths["failed_calibration_summary"])).resolve().parent.glob("*.pt")):
        raise ValueError("sealed v9.8.2 failure contains forbidden model state")
    arrays = _load_npz(maps_file)
    response_arrays = _load_npz(Path(str(paths["response6_maps"])).resolve())
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
        "step_radii",
        "prompt_ids",
        "audit_seed_table",
        "update2_design_seeds",
        "base_normalized",
        "step1_normalized",
        "candidates_normalized",
        "base",
        "step1",
        "candidates",
        "v5_target",
        "base_v5_prediction",
        "step1_v5_prediction",
        "candidate_v5_predictions",
    }
    if set(arrays) != required_arrays:
        raise ValueError("preservation array inventory changed")
    if (
        arrays["base_normalized"].shape != (3, 2, 8192, 6)
        or arrays["step1_normalized"].shape != (3, 2, 8192, 6)
        or arrays["candidates_normalized"].shape != (5, 3, 2, 8192, 6)
        or arrays["base"].shape != (3, 2, 8192, 6)
        or arrays["step1"].shape != (3, 2, 8192, 6)
        or arrays["candidates"].shape != (5, 3, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (5, 3, 8192, 6)
        or not np.array_equal(arrays["step_radii"], np.asarray(STEP_RADII))
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["audit_seed_table"].tolist()
        != [list(stable_rollout_seeds(g)) for g in range(3)]
        or arrays["update2_design_seeds"].tolist()
        != list(calibration_design_seeds(2))
    ):
        raise ValueError("preservation array shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("preservation array contains NaN/Inf: " + name)
    if not np.array_equal(arrays["base_normalized"], response_arrays["base_normalized"]):
        raise ValueError("preservation Base does not reuse v9.8.1")
    if not np.array_equal(arrays["step1_normalized"], response_arrays["candidate_normalized"]):
        raise ValueError("preservation step1 does not reproduce v9.8.1")
    if not np.array_equal(arrays["base_v5_prediction"], response_arrays["base_v5_prediction"]):
        raise ValueError("preservation Base v5 probe changed")
    if not np.array_equal(arrays["step1_v5_prediction"], response_arrays["candidate_v5_prediction"]):
        raise ValueError("preservation step1 v5 probe changed")

    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    base_physical = np.clip(
        arrays["base_normalized"] * std.reshape(1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    step1_physical = np.clip(
        arrays["step1_normalized"] * std.reshape(1, 1, 1, 6)
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
    if (
        not np.array_equal(arrays["base"], base_physical)
        or not np.array_equal(arrays["step1"], step1_physical)
        or not np.array_equal(arrays["candidates"], candidates_physical)
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
    step1_rows = _rows(bundle, arrays["step1"])
    base_pooled = pooled_object_metrics(base_rows)
    step1_pooled = pooled_object_metrics(step1_rows)
    verified_mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(arrays["base"], verified_mask)
    step1_invariance = _invariance(arrays["step1"], verified_mask)
    base_v5_dense = (
        (arrays["base_v5_prediction"] - arrays["v5_target"]) ** 2
    ).mean(axis=(1, 2)).tolist()
    step1_v5_dense = (
        (arrays["step1_v5_prediction"] - arrays["v5_target"]) ** 2
    ).mean(axis=(1, 2)).tolist()
    for key, actual in (
        ("base_rows", base_rows),
        ("step1_rows", step1_rows),
        ("base_pooled", base_pooled),
        ("step1_pooled", step1_pooled),
        ("base_prompt_invariance", base_invariance),
        ("step1_prompt_invariance", step1_invariance),
        ("base_v5_dense", base_v5_dense),
        ("step1_v5_dense", step1_v5_dense),
    ):
        _close(value[key], actual, key)

    direction = value.get("direction")
    if not isinstance(direction, Mapping) or set(direction) != {
        "task_order",
        "task_losses",
        "gram",
        "weights",
        "directional_derivatives",
        "direction_norm_before_unit",
        "direction_sha256",
        "step1_state_sha256",
    }:
        raise ValueError("preservation direction schema changed")
    gram = np.asarray(direction["gram"], np.float64)
    weights = np.asarray(direction["weights"], np.float64)
    derivatives = np.asarray(direction["directional_derivatives"], np.float64)
    task_losses = np.asarray(direction["task_losses"], np.float64)
    direction_norm = float(direction["direction_norm_before_unit"])
    if (
        direction["task_order"] != list(TASK_ORDER)
        or task_losses.shape != (len(TASK_ORDER),)
        or gram.shape != (len(TASK_ORDER), len(TASK_ORDER))
        or not np.isfinite(gram).all()
        or not np.allclose(gram, gram.T, atol=1e-7)
        or float(np.linalg.eigvalsh(gram).min()) < -1e-5
        or weights.shape != (len(TASK_ORDER),)
        or not np.isfinite(weights).all()
        or float(weights.min()) < -1e-10
        or not math.isclose(float(weights.sum()), 1.0, abs_tol=2e-6)
        or derivatives.shape != (len(TASK_ORDER),)
        or not np.isfinite(derivatives).all()
        or not np.isfinite(task_losses).all()
        or not math.isfinite(direction_norm)
        or direction_norm <= 0.0
        or float(derivatives.min())
        < float(POLICY["gates"]["minimum_directional_derivative"])
    ):
        raise ValueError("preservation direction geometry changed")
    expected_norm = math.sqrt(max(float(weights @ gram @ weights), 0.0))
    expected_derivatives = (gram @ weights) / expected_norm
    if not math.isclose(direction_norm, expected_norm, rel_tol=2e-5, abs_tol=2e-7):
        raise ValueError("preservation direction norm changed")
    if not np.allclose(derivatives, expected_derivatives, rtol=2e-5, atol=2e-7):
        raise ValueError("preservation directional derivatives changed")
    if value.get("step1_state_sha256") != direction.get("step1_state_sha256"):
        raise ValueError("step-1 state binding changed")
    for key in ("zero_state_sha256", "step1_state_sha256"):
        if not isinstance(value.get(key), str) or len(value[key]) != 64:
            raise ValueError(key + " changed")

    saved_candidates = value.get("candidates")
    if not isinstance(saved_candidates, list) or len(saved_candidates) != len(STEP_RADII):
        raise ValueError("five preservation candidates are required")
    recomputed_candidates = []
    candidate_keys = {
        "name",
        "radius",
        "base_rows",
        "step1_rows",
        "candidate_rows",
        "base_pooled",
        "step1_pooled",
        "candidate_pooled",
        "base_prompt_invariance",
        "step1_prompt_invariance",
        "candidate_prompt_invariance",
        "base_v5_dense",
        "step1_v5_dense",
        "candidate_v5_dense",
        "maximum_base_map_delta",
        "maximum_incremental_map_delta",
        "presence",
        "response_checks",
        "checks",
        "eligible",
        "failed_checks",
    }
    for index, radius in enumerate(STEP_RADII):
        saved = saved_candidates[index]
        expected_name = "preserve11_radius_" + str(radius).replace("0.", "0p")
        if (
            not isinstance(saved, Mapping)
            or set(saved) != candidate_keys
            or saved.get("name") != expected_name
            or float(saved.get("radius")) != radius
        ):
            raise ValueError("preservation radius order changed")
        current = arrays["candidates"][index]
        current_rows = _rows(bundle, current)
        current_pooled = pooled_object_metrics(current_rows)
        current_invariance = _invariance(current, verified_mask)
        current_v5_dense = (
            (arrays["candidate_v5_predictions"][index] - arrays["v5_target"]) ** 2
        ).mean(axis=(1, 2)).tolist()
        maximum_base_delta = float(np.abs(current - arrays["base"]).max())
        maximum_incremental_delta = float(np.abs(current - arrays["step1"]).max())
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
        checks = preservation_checks(
            step1_rows=step1_rows,
            candidate_rows=current_rows,
            step1_v5_dense=step1_v5_dense,
            candidate_v5_dense=current_v5_dense,
            response_checks=response_checks_value,
            directional_derivatives=direction["directional_derivatives"],
            maximum_incremental_map_delta=maximum_incremental_delta,
        )
        presence = [
            [continuous_presence_checks(current_rows[g][p]) for p in range(2)]
            for g in range(3)
        ]
        recomputed = dict(saved)
        recomputed.update(
            {
                "base_rows": base_rows,
                "step1_rows": step1_rows,
                "candidate_rows": current_rows,
                "base_pooled": base_pooled,
                "step1_pooled": step1_pooled,
                "candidate_pooled": current_pooled,
                "base_prompt_invariance": base_invariance,
                "step1_prompt_invariance": step1_invariance,
                "candidate_prompt_invariance": current_invariance,
                "base_v5_dense": base_v5_dense,
                "step1_v5_dense": step1_v5_dense,
                "candidate_v5_dense": current_v5_dense,
                "maximum_base_map_delta": maximum_base_delta,
                "maximum_incremental_map_delta": maximum_incremental_delta,
                "presence": presence,
                "response_checks": response_checks_value,
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
        _close(saved, recomputed, "candidate " + str(radius))
        recomputed_candidates.append(recomputed)

    order = rank_candidates(recomputed_candidates)
    selected = order[0] if order else None
    if value.get("eligible_selection_order") != order or value.get("selected_candidate") != selected:
        raise ValueError("preservation candidate ranking changed")
    expected_checks = {
        "sealed_v982_failure_and_nested_diagnosis_bound": True,
        "fresh_v5r4_zero_init": True,
        "v981_update1_state_and_maps_exactly_reproduced": True,
        "eleven_task_direction_is_common_descent": min(direction["directional_derivatives"])
        >= float(POLICY["gates"]["minimum_directional_derivative"]),
        "five_actual_k3_radius_candidates_evaluated": len(recomputed_candidates)
        == len(STEP_RADII),
        "policy_locked_before_model_and_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_preservation_candidate_is_admissible": selected is not None,
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed_checks = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("checks") != expected_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_preservation_aware_calibration6") != (status == "PASS")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_room_0102") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("preservation status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "v97_binding_id": value["v97_binding_id"],
            "v98_binding_id": value["v98_binding_id"],
            "v981_binding_id": value["v981_binding_id"],
            "v982_binding_id": value["v982_binding_id"],
            "policy_id": value["policy_id"],
            "selected_candidate": value["selected_candidate"],
            "preservation_maps_sha256": value["preservation_maps_sha256"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("preservation binding ID changed")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("preflight output contains forbidden model state")

    print("[PRESERVATION_DIRECTION_PREFLIGHT_{}] Teacher-v9.8.3 integrity".format(status))
    print("[PASS] sealed v9.8.2 failure and exact v9.8.1 update-1 response verified")
    print("[PASS] eleven-task geometry, five K=3 panels and ranking recomputed")
    print("[PASS] no optimizer/checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

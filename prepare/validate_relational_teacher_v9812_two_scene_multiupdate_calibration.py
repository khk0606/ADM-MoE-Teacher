#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.12 two-scene multi-update calibration."""

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

from preflight_relational_teacher_v983_preservation_direction import (  # noqa: E402
    _invariance,
    _load_npz,
    _rows,
)
from preflight_relational_teacher_v985_two_scene_step4 import _scene_binding  # noqa: E402
from relational_teacher_v9_all_sittable_contract import read_json, sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v9_lora_runtime import FORWARD_INPUT_KEYS, load_stats  # noqa: E402
from relational_teacher_v97_early_rollout_k3_contract import stable_rollout_seeds  # noqa: E402
from relational_teacher_v984_preservation_calibration6_contract import (  # noqa: E402
    POLICY_ID as V984_POLICY_ID,
    SCHEMA as V984_REPORT_SCHEMA,
)
from relational_teacher_v985_two_scene_step4_preflight_contract import (  # noqa: E402
    EXPECTED_INSTANCES,
    absolute_presence_checks,
    pooled_by_role,
    two_scene_preflight_checks,
)
from relational_teacher_v988_rollout_aligned_direction_contract import (  # noqa: E402
    POLICY_ID as V988_POLICY_ID,
    RAW_GRAM_ASYMMETRY_CAP,
    SCHEMA as V988_SCHEMA,
    TASK_ORDER as V988_TASK_ORDER,
)
from relational_teacher_v9812_two_scene_multiupdate_calibration_contract import (  # noqa: E402
    AUDIT_SCENE,
    AUDIT_SEED_TABLE,
    CALIBRATION_TAG,
    DESIGN_SEED_TABLE,
    DEVELOPMENT_SCENE,
    GENERATION_COUNT,
    MINIMUM_DIRECTIONAL_DERIVATIVE,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    RECONSTRUCTION_STEPS,
    SCHEMA,
    SELECTED_DIRECTION,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SHORTLIST_LIMIT,
    SOURCE_SCENE,
    STEP_RADIUS,
    UPDATE_COUNT,
    V9810_SCHEMA,
    V9811_SCHEMA,
    all_three_counts,
    calibration_checks,
    canonical_sha256,
    direction_candidates,
    rank_shortlist,
    select_direction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


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
            _close(one, two, "{}[{}]".format(label, index))
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


def _scene_metrics(
    bundles: Mapping[str, Mapping[str, object]],
    maps: np.ndarray,
) -> tuple[dict, dict, dict]:
    rows = {}
    pooled = {}
    invariance = {}
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        rows[scene] = _rows(bundles[scene], maps[scene_index])
        pooled[scene] = pooled_by_role(rows[scene], list(bundles[scene]["instance_names"]))
        invariance[scene] = _invariance(
            maps[scene_index],
            np.asarray(bundles[scene]["verified_positive_mask"], bool),
        )
    return rows, pooled, invariance


def main() -> None:
    summary_file = parse_args().summary.expanduser().resolve()
    value = read_json(summary_file)
    selection_seeds = [
        list(stable_rollout_seeds(generation)) for generation in range(GENERATION_COUNT)
    ]
    design_seeds = [list(row) for row in DESIGN_SEED_TABLE]
    audit_seeds = [list(row) for row in AUDIT_SEED_TABLE]
    accounting = {
        "selection_two_scene_base_full_draws": 12,
        "selection_two_scene_base_exact_partial_repeats": 12,
        "design_two_scene_base_full_draws": 12,
        "design_two_scene_base_exact_partial_repeats": 12,
        "audit_two_scene_base_full_draws": 12,
        "audit_two_scene_base_exact_partial_repeats": 12,
        "room0101_reconstruction_full_draws": 6,
        "room0101_reconstruction_exact_partial_repeats": 6,
        "sealed_direction_gradient_groups": 6,
        "start_state_partial_resumes": 12,
        "calibration_gradient_groups": UPDATE_COUNT * 6,
        "calibration_monitor_partial_resumes": UPDATE_COUNT * 12,
    }
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.12 report schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("calibration_tag") != CALIBRATION_TAG
        or value.get("diffusion_steps") != 500
        or value.get("source_scene") != SOURCE_SCENE
        or value.get("audit_scene") != AUDIT_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("selected_v984_step") != SELECTED_V984_STEP
        or value.get("reconstruction_steps") != list(RECONSTRUCTION_STEPS)
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or value.get("selected_direction") != SELECTED_DIRECTION
        or float(value.get("selected_radius", -1.0)) != SELECTED_RADIUS
        or value.get("update_count") != UPDATE_COUNT
        or value.get("monitor_steps") != list(MONITOR_STEPS)
        or float(value.get("step_radius", -1.0)) != STEP_RADIUS
        or value.get("shortlist_limit") != SHORTLIST_LIMIT
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("selection_seed_table") != selection_seeds
        or value.get("design_seed_table") != design_seeds
        or value.get("audit_seed_table") != audit_seeds
        or value.get("trajectory_accounting") != accounting
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.12 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.12 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.12 bound file changed: " + str(name))
    policy = read_json(Path(str(paths["calibration_policy"])).resolve())
    if policy != POLICY or canonical_sha256(policy) != POLICY_ID:
        raise ValueError("Teacher-v9.8.12 policy changed")
    if value.get("policy_sha256") != hashes["calibration_policy"]:
        raise ValueError("Teacher-v9.8.12 policy file hash changed")

    v9811 = read_json(Path(str(paths["v9811_summary"])).resolve())
    if (
        v9811.get("schema") != V9811_SCHEMA
        or v9811.get("status") != "PASS"
        or v9811.get("failed_checks")
        or v9811.get("response_failed_checks")
        or v9811.get("binding_id") != value.get("v9811_binding_id")
        or v9811.get("authorizes_fresh_two_scene_rollout_aligned_multiupdate_calibration")
        is not True
    ):
        raise ValueError("sealed Teacher-v9.8.11 PASS changed")
    prior_replication_seeds = v9811["replication_seed_table"]
    if value.get("prior_replication_seed_table") != prior_replication_seeds:
        raise ValueError("sealed v9.8.11 seed binding changed")
    all_seed_values = [
        number
        for table in (selection_seeds, prior_replication_seeds, design_seeds, audit_seeds)
        for row in table
        for number in row
    ]
    if len(all_seed_values) != len(set(all_seed_values)):
        raise ValueError("selection/replication/design/audit seeds overlap")

    v9810 = read_json(Path(str(paths["v9810_summary"])).resolve())
    if (
        v9810.get("schema") != V9810_SCHEMA
        or v9810.get("status") != "PASS"
        or float(v9810.get("selected_radius", -1.0)) != SELECTED_RADIUS
        or v9810.get("failed_checks")
        or v9810.get("binding_id") != value.get("v9810_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.10 PASS changed")
    selected_v9810 = [
        (index, row)
        for index, row in enumerate(v9810["response_rows"])
        if float(row["radius"]) == SELECTED_RADIUS
    ]
    if len(selected_v9810) != 1 or selected_v9810[0][1].get("eligible") is not True:
        raise ValueError("sealed radius-0.006 response changed")
    selected_v9810_index, selected_v9810_row = selected_v9810[0]

    v988 = read_json(Path(str(paths["v988_report"])).resolve())
    if (
        v988.get("schema") != V988_SCHEMA
        or v988.get("status") != "PASS"
        or v988.get("policy_id") != V988_POLICY_ID
        or v988.get("selected_candidate") != SELECTED_DIRECTION
        or v988.get("binding_id") != value.get("v988_binding_id")
        or v988.get("failed_checks")
    ):
        raise ValueError("sealed Teacher-v9.8.8 PASS changed")
    v984 = read_json(Path(str(paths["v984_summary"])).resolve())
    if (
        v984.get("schema") != V984_REPORT_SCHEMA
        or v984.get("status") != "PASS"
        or v984.get("policy_id") != V984_POLICY_ID
        or v984.get("shortlisted_steps") != [4, 3]
        or v984.get("failed_checks")
        or v984.get("binding_id") != value.get("v984_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.4 authority changed")
    reconstruction = value.get("reconstruction_rows")
    if not isinstance(reconstruction, list) or len(reconstruction) != 4:
        raise ValueError("four exact reconstruction rows required")
    for index, row in enumerate(reconstruction):
        _close(row, v984["direction_rows"][index], "reconstruction row")
    if (
        value.get("zero_state_sha256") != v984.get("zero_state_sha256")
        or value.get("step4_state_sha256") != v988.get("selected_state_sha256")
        or value.get("start_state_sha256") != v9811.get("candidate_state_sha256")
        or value.get("start_state_sha256")
        != selected_v9810_row.get("candidate_state_sha256")
    ):
        raise ValueError("calibration start-state reconstruction changed")
    selected_v988 = [
        row for row in v988["direction_candidates"] if row["name"] == SELECTED_DIRECTION
    ]
    if (
        len(selected_v988) != 1
        or value.get("selected_direction_sha256")
        != selected_v988[0].get("direction_sha256")
    ):
        raise ValueError("sealed selected-direction binding changed")
    v988_geometry = _load_npz(Path(str(paths["v988_geometry"])).resolve())
    _close(value["direction_task_losses"], v988_geometry["task_losses"].tolist(), "sealed direction losses")
    _close(value["direction_gram"], v988_geometry["gram"].tolist(), "sealed direction Gram")
    if float(value["direction_raw_gram_max_asymmetry"]) > RAW_GRAM_ASYMMETRY_CAP:
        raise ValueError("sealed direction Gram asymmetry changed")

    arrays_file = Path(str(paths["calibration_maps"])).resolve()
    if sha256_file(arrays_file) != value.get("calibration_maps_sha256"):
        raise ValueError("calibration maps hash changed")
    arrays = _load_npz(arrays_file)
    base_required = {
        "scene_ids", "prompt_ids", "selection_seed_table",
        "prior_replication_seed_table", "design_seed_table", "audit_seed_table",
        "selection_base_normalized", "design_base_normalized",
        "design_base_repeat_normalized", "base_normalized", "base_repeat_normalized",
        "start_normalized", "candidates_normalized", "base", "start", "candidates",
        "v5_target", "base_v5_prediction", "start_v5_prediction",
        "candidate_v5_predictions",
    }
    suffixes = {
        "xyz", "points", "instance_ids", "category_ids", "instance_names",
        "verified_object_mask", "verified_positive_mask", "unknown_sittable_mask",
        "explicit_negative_mask", "instance_targets", "all_sittable_gt",
    }
    required = base_required | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in suffixes
    }
    if set(arrays) != required:
        raise ValueError("Teacher-v9.8.12 array inventory changed")
    map_shape = (2, GENERATION_COUNT, 2, 8192, 6)
    if (
        arrays["scene_ids"].tolist() != [SOURCE_SCENE, AUDIT_SCENE]
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["selection_seed_table"].tolist() != selection_seeds
        or arrays["prior_replication_seed_table"].tolist() != prior_replication_seeds
        or arrays["design_seed_table"].tolist() != design_seeds
        or arrays["audit_seed_table"].tolist() != audit_seeds
        or arrays["selection_base_normalized"].shape != map_shape
        or arrays["design_base_normalized"].shape != map_shape
        or arrays["design_base_repeat_normalized"].shape != map_shape
        or arrays["base_normalized"].shape != map_shape
        or arrays["base_repeat_normalized"].shape != map_shape
        or arrays["start_normalized"].shape != map_shape
        or arrays["candidates_normalized"].shape
        != (UPDATE_COUNT, 2, GENERATION_COUNT, 2, 8192, 6)
        or arrays["base"].shape != map_shape
        or arrays["start"].shape != map_shape
        or arrays["candidates"].shape
        != (UPDATE_COUNT, 2, GENERATION_COUNT, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["start_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (UPDATE_COUNT, 3, 8192, 6)
    ):
        raise ValueError("Teacher-v9.8.12 array shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Teacher-v9.8.12 array contains NaN/Inf: " + name)
    if not np.array_equal(arrays["selection_base_normalized"], v988_geometry["base_normalized"]):
        raise ValueError("sealed selection Base maps changed")
    if not np.array_equal(
        arrays["design_base_normalized"], arrays["design_base_repeat_normalized"]
    ) or not np.array_equal(arrays["base_normalized"], arrays["base_repeat_normalized"]):
        raise ValueError("design/audit Base resumes are not bitwise exact")
    v9811_arrays = _load_npz(Path(str(paths["v9811_maps"])).resolve())
    v9810_arrays = _load_npz(Path(str(paths["v9810_maps"])).resolve())
    if not np.array_equal(arrays["start_v5_prediction"], v9811_arrays["candidate_v5_prediction"]):
        raise ValueError("v9.8.11 start v5 response changed")
    if not np.array_equal(
        arrays["start_v5_prediction"],
        v9810_arrays["candidate_v5_predictions"][selected_v9810_index],
    ):
        raise ValueError("v9.8.10 selected v5 response changed")

    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    expected_base = np.clip(
        arrays["base_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6), 0.0, 1.0
    ).astype(np.float32)
    expected_start = np.clip(
        arrays["start_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6), 0.0, 1.0
    ).astype(np.float32)
    expected_candidates = np.clip(
        arrays["candidates_normalized"] * std.reshape(1, 1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 1, 6), 0.0, 1.0
    ).astype(np.float32)
    if (
        not np.array_equal(arrays["base"], expected_base)
        or not np.array_equal(arrays["start"], expected_start)
        or not np.array_equal(arrays["candidates"], expected_candidates)
    ):
        raise ValueError("normalized-to-physical conversion changed")

    dataset_index = Path(str(paths["dataset_index"])).resolve()
    source_index = Path(str(paths["source_dataset_index"])).resolve()
    top_index = validate_top_index(dataset_index.parent, source_index.parent, dataset_index)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {SOURCE_SCENE, AUDIT_SCENE}:
        raise ValueError("two train-scene metadata inventory changed")
    bundles = {
        scene: load_train_scene_bundle(dataset_index.parent, source_index.parent, records[scene])
        for scene in (SOURCE_SCENE, AUDIT_SCENE)
    }
    for scene, prefix in ((SOURCE_SCENE, "source"), (AUDIT_SCENE, "audit")):
        bundle = bundles[scene]
        if tuple(bundle["instance_names"]) != tuple(EXPECTED_INSTANCES[scene]):
            raise ValueError(scene + " instance order changed")
        for suffix in suffixes - {"instance_names", "all_sittable_gt"}:
            if not np.array_equal(np.asarray(bundle[suffix]), arrays[prefix + "_" + suffix]):
                raise ValueError(scene + " saved array changed: " + suffix)
        if arrays[prefix + "_instance_names"].tolist() != list(bundle["instance_names"]):
            raise ValueError(scene + " saved instance names changed")
        if not np.array_equal(np.asarray(bundle["all_target"]), arrays[prefix + "_all_sittable_gt"]):
            raise ValueError(scene + " saved GT changed")
    if (
        value.get("source_scene_binding") != _scene_binding(records[SOURCE_SCENE])
        or value.get("audit_scene_binding") != _scene_binding(records[AUDIT_SCENE])
    ):
        raise ValueError("two-scene metadata binding changed")

    base_v5 = ((arrays["base_v5_prediction"] - arrays["v5_target"]) ** 2).mean(
        axis=(1, 2)
    ).tolist()
    start_v5 = ((arrays["start_v5_prediction"] - arrays["v5_target"]) ** 2).mean(
        axis=(1, 2)
    ).tolist()
    _close(value["base_v5_dense"], base_v5, "base v5")
    _close(value["start_v5_dense"], start_v5, "start v5")
    _close(start_v5, selected_v9810_row["candidate_v5_dense"], "selected start v5")
    base_rows, base_pooled, base_invariance = _scene_metrics(bundles, arrays["base"])
    start_rows, start_pooled, start_invariance = _scene_metrics(bundles, arrays["start"])
    _close(value["scene_base_rows"], base_rows, "base rows")
    _close(value["scene_base_pooled"], base_pooled, "base pooled")
    _close(value["scene_base_invariance"], base_invariance, "base invariance")
    _close(value["scene_start_rows"], start_rows, "start rows")
    _close(value["scene_start_pooled"], start_pooled, "start pooled")
    _close(value["scene_start_invariance"], start_invariance, "start invariance")
    start_presence = {}
    start_checks = {}
    start_delta = {}
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        names = list(bundles[scene]["instance_names"])
        start_delta[scene] = float(np.abs(arrays["start"][scene_index] - arrays["base"][scene_index]).max())
        start_checks[scene] = two_scene_preflight_checks(
            base_rows=base_rows[scene], candidate_rows=start_rows[scene], instance_names=names,
            base_prompt_invariance=base_invariance[scene],
            candidate_prompt_invariance=start_invariance[scene],
            base_v5_dense=base_v5, candidate_v5_dense=start_v5,
            maximum_map_delta=start_delta[scene],
        )
        start_presence[scene] = [
            [absolute_presence_checks(start_rows[scene][g][p], names) for p in range(2)]
            for g in range(GENERATION_COUNT)
        ]
    _close(value["start_maximum_map_delta"], start_delta, "start delta")
    _close(value["start_presence"], start_presence, "start presence")
    _close(value["start_all_three_counts"], all_three_counts(start_presence), "start counts")
    _close(value["scene_start_checks"], start_checks, "start checks")
    start_failed = {
        scene: sorted(name for name, passed in start_checks[scene].items() if not passed)
        for scene in (SOURCE_SCENE, AUDIT_SCENE)
    }
    _close(value["scene_start_failed_checks"], start_failed, "start failed")

    directions = value.get("direction_rows")
    monitors = value.get("monitor_rows")
    if not isinstance(directions, list) or len(directions) != UPDATE_COUNT:
        raise ValueError("six direction rows required")
    if not isinstance(monitors, list) or len(monitors) != UPDATE_COUNT:
        raise ValueError("six monitor rows required")
    previous_state = value["start_state_sha256"]
    recomputed_monitors = []
    for step in MONITOR_STEPS:
        direction = directions[step - 1]
        if (
            direction.get("step") != step
            or direction.get("task_order") != list(V988_TASK_ORDER)
            or direction.get("pre_state_sha256") != previous_state
            or direction.get("pre_state_sha256") == direction.get("post_state_sha256")
            or not isinstance(direction.get("selected_direction_sha256"), str)
            or len(direction["selected_direction_sha256"]) != 64
            or not math.isclose(float(direction.get("selected_direction_l2", 0.0)), 1.0, rel_tol=2e-5, abs_tol=2e-6)
        ):
            raise ValueError("calibration direction/state chain changed")
        gram = np.asarray(direction["gram"], np.float64)
        if float(direction["raw_gram_max_asymmetry"]) > RAW_GRAM_ASYMMETRY_CAP:
            raise ValueError("calibration raw Gram asymmetry changed")
        candidates = direction_candidates(gram, V988_TASK_ORDER)
        _close(direction["candidates"], candidates, "direction candidates")
        selected = select_direction(candidates)
        if direction.get("selected_candidate") != selected["name"]:
            raise ValueError("calibration direction selection changed")
        if min(float(x) for x in selected["directional_derivatives"]) < MINIMUM_DIRECTIONAL_DERIVATIVE:
            raise ValueError("selected direction is not common descent")

        current = arrays["candidates"][step - 1]
        current_v5 = (
            (arrays["candidate_v5_predictions"][step - 1] - arrays["v5_target"]) ** 2
        ).mean(axis=(1, 2)).tolist()
        current_rows, current_pooled, current_invariance = _scene_metrics(bundles, current)
        current_checks = {}
        current_presence = {}
        current_delta = {}
        for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
            names = list(bundles[scene]["instance_names"])
            current_delta[scene] = float(np.abs(current[scene_index] - arrays["base"][scene_index]).max())
            current_checks[scene] = two_scene_preflight_checks(
                base_rows=base_rows[scene], candidate_rows=current_rows[scene],
                instance_names=names, base_prompt_invariance=base_invariance[scene],
                candidate_prompt_invariance=current_invariance[scene],
                base_v5_dense=base_v5, candidate_v5_dense=current_v5,
                maximum_map_delta=current_delta[scene],
            )
            current_presence[scene] = [
                [absolute_presence_checks(current_rows[scene][g][p], names) for p in range(2)]
                for g in range(GENERATION_COUNT)
            ]
        counts = all_three_counts(current_presence)
        checks = calibration_checks(
            step=step, scene_checks=current_checks, start_pooled=start_pooled,
            candidate_pooled=current_pooled, presence=current_presence,
            directional_derivatives=selected["directional_derivatives"],
            pre_state_sha256=direction["pre_state_sha256"],
            post_state_sha256=direction["post_state_sha256"],
        )
        recomputed = {
            "step": step,
            "pre_state_sha256": direction["pre_state_sha256"],
            "post_state_sha256": direction["post_state_sha256"],
            "selected_direction_candidate": selected["name"],
            "selected_direction_sha256": direction["selected_direction_sha256"],
            "scene_candidate_rows": current_rows,
            "scene_candidate_pooled": current_pooled,
            "scene_candidate_invariance": current_invariance,
            "candidate_v5_dense": current_v5,
            "maximum_map_delta": current_delta,
            "presence": current_presence,
            "all_three_counts": counts,
            "scene_checks": current_checks,
            "scene_failed_checks": {
                scene: sorted(name for name, passed in current_checks[scene].items() if not passed)
                for scene in (SOURCE_SCENE, AUDIT_SCENE)
            },
            "checks": checks,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
            "eligible": all(checks.values()),
        }
        _close(monitors[step - 1], recomputed, "monitor row")
        recomputed_monitors.append(recomputed)
        previous_state = direction["post_state_sha256"]
    if value.get("final_state_sha256") != previous_state:
        raise ValueError("final state chain changed")

    shortlist = rank_shortlist(recomputed_monitors)
    expected_shortlist_hashes = {
        str(step): recomputed_monitors[step - 1]["post_state_sha256"] for step in shortlist
    }
    if (
        value.get("shortlisted_steps") != shortlist
        or value.get("shortlisted_state_sha256") != expected_shortlist_hashes
    ):
        raise ValueError("shortlist ranking/binding changed")
    expected_checks = {
        "sealed_v9811_pass_and_selected_state_bound": True,
        "sealed_v9810_pass_and_selected_radius_bound": True,
        "sealed_v988_pass_and_selected_direction_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exact_v988_51_task_direction_recomputed": True,
        "selected_v9811_state_exactly_reproduced": True,
        "selected_v9811_v5_response_exactly_reproduced": True,
        "selection_replication_design_audit_seeds_are_disjoint": True,
        "design_and_audit_base_resumes_are_exact": True,
        "six_rollout_aligned_updates_completed": True,
        "six_actual_two_scene_k3_monitors_completed": True,
        "every_selected_update_direction_is_common_descent": True,
        "at_least_one_all_three_state_is_shortlisted": bool(shortlist),
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "room0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("checks") != expected_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed
        or value.get("authorizes_shortlisted_state_checkpoint_export_gate")
        != (status == "PASS")
        or value.get("authorizes_cross_scene_objective_or_model_capacity_redesign")
        != (status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.8.12 status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "v9811_binding_id": value["v9811_binding_id"],
            "policy_id": value["policy_id"],
            "start_state_sha256": value["start_state_sha256"],
            "shortlisted_steps": value["shortlisted_steps"],
            "selected_direction_sha256": value["selected_direction_sha256"],
            "calibration_maps_sha256": value["calibration_maps_sha256"],
            "audit_seed_table": value["audit_seed_table"],
            "status": value["status"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9.8.12 binding ID changed")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.12 contains forbidden model state")

    print("[TWO_SCENE_MULTIUPDATE_CALIBRATION_{}] Teacher-v9.8.12 integrity".format(status))
    print("[PASS] v9.8.11 start, two-scene maps and six state transitions verified")
    print("[PASS] every K=3 role/negative/v5/all-three gate independently recomputed")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")
    print("[OK] shortlisted steps:", shortlist)
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

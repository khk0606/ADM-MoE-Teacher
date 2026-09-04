#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.11 disjoint-seed rollout replication."""

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
)
from relational_teacher_v9811_rollout_aligned_replication_contract import (  # noqa: E402
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    GENERATION_COUNT,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    RECONSTRUCTION_STEPS,
    REPLICATION_SEED_TABLE,
    REPLICATION_TAG,
    SCHEMA,
    SELECTED_DIRECTION,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    V9810_SCHEMA,
    V988_SCHEMA,
    canonical_sha256,
    replication_checks,
    replication_rollout_seeds,
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


def main() -> None:
    summary_file = parse_args().summary.expanduser().resolve()
    value = read_json(summary_file)
    selection_seed_table = [
        list(stable_rollout_seeds(generation)) for generation in range(GENERATION_COUNT)
    ]
    replication_seed_table = [
        list(replication_rollout_seeds(generation))
        for generation in range(GENERATION_COUNT)
    ]
    accounting = {
        "selection_two_scene_base_full_draws": 12,
        "selection_two_scene_base_exact_partial_repeats": 12,
        "replication_two_scene_base_full_draws": 12,
        "replication_two_scene_base_exact_partial_repeats": 12,
        "room0101_reconstruction_full_draws": 6,
        "room0101_reconstruction_exact_partial_repeats": 6,
        "rollout_aligned_gradient_groups": 6,
        "selected_candidate_partial_resumes": 12,
    }
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.11 report schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("replication_tag") != REPLICATION_TAG
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
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("selection_seed_table") != selection_seed_table
        or value.get("replication_seed_table") != replication_seed_table
        or value.get("trajectory_accounting") != accounting
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.11 sealed protocol changed")
    if tuple(tuple(row) for row in replication_seed_table) != REPLICATION_SEED_TABLE:
        raise ValueError("replication seed contract changed")
    if {x for row in selection_seed_table for x in row} & {
        x for row in replication_seed_table for x in row
    }:
        raise ValueError("selection and replication seeds overlap")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.11 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.11 bound file changed: " + str(name))
    policy_value = read_json(Path(str(paths["response_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("Teacher-v9.8.11 policy changed")
    if value.get("policy_sha256") != hashes["response_policy"]:
        raise ValueError("Teacher-v9.8.11 policy file hash changed")

    v9810 = read_json(Path(str(paths["v9810_summary"])).resolve())
    if (
        v9810.get("schema") != V9810_SCHEMA
        or v9810.get("status") != "PASS"
        or float(v9810.get("selected_radius", -1.0)) != SELECTED_RADIUS
        or v9810.get("failed_checks")
        or v9810.get("binding_id") != value.get("v9810_binding_id")
        or v9810.get("authorizes_fresh_two_scene_rollout_aligned_calibration") is not True
        or v9810.get("serialized_model_state") is not False
        or v9810.get("authorizes_checkpoint") is not False
    ):
        raise ValueError("sealed Teacher-v9.8.10 PASS changed")
    selected_rows = [
        (index, row)
        for index, row in enumerate(v9810.get("response_rows", []))
        if float(row.get("radius", -1.0)) == SELECTED_RADIUS
    ]
    if len(selected_rows) != 1 or selected_rows[0][1].get("eligible") is not True:
        raise ValueError("sealed Teacher-v9.8.10 selected response changed")
    selected_index, selected_v9810 = selected_rows[0]

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
        or value.get("candidate_state_sha256")
        != selected_v9810.get("candidate_state_sha256")
    ):
        raise ValueError("selected LoRA state reconstruction changed")
    selected_v988 = [
        row for row in v988["direction_candidates"] if row["name"] == SELECTED_DIRECTION
    ]
    if (
        len(selected_v988) != 1
        or selected_v988[0].get("eligible") is not True
        or value.get("selected_direction_sha256")
        != selected_v988[0].get("direction_sha256")
    ):
        raise ValueError("selected rollout-aligned direction binding changed")
    v988_geometry = _load_npz(Path(str(paths["v988_geometry"])).resolve())
    _close(value["direction_task_losses"], v988_geometry["task_losses"].tolist(), "direction losses")
    _close(value["direction_gram"], v988_geometry["gram"].tolist(), "direction Gram")
    asymmetry = float(value.get("direction_raw_gram_max_asymmetry", float("nan")))
    if not math.isfinite(asymmetry) or not 0.0 <= asymmetry <= RAW_GRAM_ASYMMETRY_CAP:
        raise ValueError("direction raw Gram asymmetry changed")

    arrays_file = Path(str(paths["response_maps"])).resolve()
    if sha256_file(arrays_file) != value.get("response_maps_sha256"):
        raise ValueError("replication maps hash changed")
    arrays = _load_npz(arrays_file)
    base_required = {
        "scene_ids", "prompt_ids", "selection_seed_table", "replication_seed_table",
        "selection_base_normalized", "base_normalized", "base_repeat_normalized",
        "candidate_normalized",
        "base", "candidate", "v5_target", "base_v5_prediction",
        "candidate_v5_prediction",
    }
    per_scene_suffixes = {
        "xyz", "points", "instance_ids", "category_ids", "instance_names",
        "verified_object_mask", "verified_positive_mask", "unknown_sittable_mask",
        "explicit_negative_mask", "instance_targets", "all_sittable_gt",
    }
    required = base_required | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in per_scene_suffixes
    }
    if set(arrays) != required:
        raise ValueError("Teacher-v9.8.11 array inventory changed")
    map_shape = (2, GENERATION_COUNT, 2, 8192, 6)
    if (
        arrays["scene_ids"].tolist() != [SOURCE_SCENE, AUDIT_SCENE]
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["selection_seed_table"].tolist() != selection_seed_table
        or arrays["replication_seed_table"].tolist() != replication_seed_table
        or arrays["selection_base_normalized"].shape != map_shape
        or arrays["base_normalized"].shape != map_shape
        or arrays["base_repeat_normalized"].shape != map_shape
        or arrays["candidate_normalized"].shape != map_shape
        or arrays["base"].shape != map_shape
        or arrays["candidate"].shape != map_shape
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_prediction"].shape != (3, 8192, 6)
    ):
        raise ValueError("Teacher-v9.8.11 array shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Teacher-v9.8.11 array contains NaN/Inf: " + name)
    if not np.array_equal(arrays["selection_base_normalized"], v988_geometry["base_normalized"]):
        raise ValueError("exact v9.8.8 selection Base maps changed")
    if not np.array_equal(arrays["base_normalized"], arrays["base_repeat_normalized"]):
        raise ValueError("new-seed Base partial resumes are not bitwise exact")
    v9810_arrays = _load_npz(Path(str(paths["v9810_maps"])).resolve())
    if not np.array_equal(
        arrays["selection_base_normalized"], v9810_arrays["base_normalized"]
    ):
        raise ValueError("exact v9.8.10 selection Base maps changed")
    if not np.array_equal(
        arrays["candidate_v5_prediction"],
        v9810_arrays["candidate_v5_predictions"][selected_index],
    ):
        raise ValueError("selected v9.8.10 v5 prediction changed")

    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    expected_base = np.clip(
        arrays["base_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6), 0.0, 1.0
    ).astype(np.float32)
    expected_candidate = np.clip(
        arrays["candidate_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6), 0.0, 1.0
    ).astype(np.float32)
    if not np.array_equal(arrays["base"], expected_base) or not np.array_equal(
        arrays["candidate"], expected_candidate
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
        for suffix in per_scene_suffixes - {"instance_names", "all_sittable_gt"}:
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

    base_v5_dense = ((arrays["base_v5_prediction"] - arrays["v5_target"]) ** 2).mean(
        axis=(1, 2)
    ).tolist()
    candidate_v5_dense = (
        (arrays["candidate_v5_prediction"] - arrays["v5_target"]) ** 2
    ).mean(axis=(1, 2)).tolist()
    _close(value["base_v5_dense"], base_v5_dense, "base v5")
    _close(value["candidate_v5_dense"], candidate_v5_dense, "candidate v5")
    _close(candidate_v5_dense, selected_v9810["candidate_v5_dense"], "selected v9810 v5")

    scene_base_rows = {}
    scene_base_pooled = {}
    scene_base_invariance = {}
    scene_candidate_rows = {}
    scene_candidate_pooled = {}
    scene_candidate_invariance = {}
    scene_checks = {}
    scene_presence = {}
    maximum_map_delta = {}
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        bundle = bundles[scene]
        names = list(bundle["instance_names"])
        positive_mask = np.asarray(bundle["verified_positive_mask"], bool)
        scene_base_rows[scene] = _rows(bundle, arrays["base"][scene_index])
        scene_base_pooled[scene] = pooled_by_role(scene_base_rows[scene], names)
        scene_base_invariance[scene] = _invariance(arrays["base"][scene_index], positive_mask)
        scene_candidate_rows[scene] = _rows(bundle, arrays["candidate"][scene_index])
        scene_candidate_pooled[scene] = pooled_by_role(scene_candidate_rows[scene], names)
        scene_candidate_invariance[scene] = _invariance(
            arrays["candidate"][scene_index], positive_mask
        )
        maximum_map_delta[scene] = float(
            np.abs(arrays["candidate"][scene_index] - arrays["base"][scene_index]).max()
        )
        scene_checks[scene] = two_scene_preflight_checks(
            base_rows=scene_base_rows[scene],
            candidate_rows=scene_candidate_rows[scene],
            instance_names=names,
            base_prompt_invariance=scene_base_invariance[scene],
            candidate_prompt_invariance=scene_candidate_invariance[scene],
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=candidate_v5_dense,
            maximum_map_delta=maximum_map_delta[scene],
        )
        scene_presence[scene] = [
            [absolute_presence_checks(scene_candidate_rows[scene][g][p], names) for p in range(2)]
            for g in range(GENERATION_COUNT)
        ]
    for key, expected in (
        ("scene_base_rows", scene_base_rows),
        ("scene_base_pooled", scene_base_pooled),
        ("scene_base_invariance", scene_base_invariance),
        ("scene_candidate_rows", scene_candidate_rows),
        ("scene_candidate_pooled", scene_candidate_pooled),
        ("scene_candidate_invariance", scene_candidate_invariance),
        ("maximum_map_delta", maximum_map_delta),
        ("presence", scene_presence),
        ("scene_checks", scene_checks),
    ):
        _close(value[key], expected, key)
    scene_failed_checks = {
        scene: sorted(name for name, passed in scene_checks[scene].items() if not passed)
        for scene in (SOURCE_SCENE, AUDIT_SCENE)
    }
    _close(value["scene_failed_checks"], scene_failed_checks, "scene failed checks")
    response_checks = replication_checks(
        source_checks=scene_checks[SOURCE_SCENE],
        audit_checks=scene_checks[AUDIT_SCENE],
        selected_state_exact=(
            value["candidate_state_sha256"] == selected_v9810["candidate_state_sha256"]
        ),
        selected_v5_exact=np.array_equal(
            arrays["candidate_v5_prediction"],
            v9810_arrays["candidate_v5_predictions"][selected_index],
        ),
        deterministic_repeats_exact=True,
        seed_table_disjoint=True,
    )
    response_failed = sorted(name for name, passed in response_checks.items() if not passed)
    _close(value["response_checks"], response_checks, "response checks")
    if (
        value.get("response_failed_checks") != response_failed
        or value.get("response_eligible") is not all(response_checks.values())
    ):
        raise ValueError("new-seed response status changed")

    expected_checks = {
        "sealed_v9810_pass_and_selected_radius_bound": True,
        "sealed_v988_pass_and_selected_direction_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exact_v988_51_task_direction_recomputed": True,
        "selected_v9810_state_exactly_reproduced": True,
        "selected_v9810_v5_response_exactly_reproduced": True,
        "selection_and_replication_seed_tables_are_disjoint": True,
        "new_seed_base_resumes_are_exact": True,
        "both_scene_new_seed_k3_two_prompt_response_completed": True,
        "new_seed_response_is_admissible": all(response_checks.values()),
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "room0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed_checks = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("checks") != expected_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_fresh_two_scene_rollout_aligned_multiupdate_calibration")
        != (status == "PASS")
        or value.get("authorizes_cross_scene_training_objective_redesign")
        != (status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.8.11 status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "v9810_binding_id": value["v9810_binding_id"],
            "policy_id": value["policy_id"],
            "candidate_state_sha256": value["candidate_state_sha256"],
            "selected_direction_sha256": value["selected_direction_sha256"],
            "response_maps_sha256": value["response_maps_sha256"],
            "replication_seed_table": value["replication_seed_table"],
            "status": value["status"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9.8.11 binding ID changed")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.11 contains forbidden model state")

    print("[ROLLOUT_ALIGNED_REPLICATION_{}] Teacher-v9.8.11 integrity".format(status))
    print("[PASS] exact v9.8.10 radius 0.006 state and v5 response verified")
    print("[PASS] disjoint-seed two-scene K=3 maps and strict role gates recomputed")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")
    print("[OK] response failed checks:", response_failed)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

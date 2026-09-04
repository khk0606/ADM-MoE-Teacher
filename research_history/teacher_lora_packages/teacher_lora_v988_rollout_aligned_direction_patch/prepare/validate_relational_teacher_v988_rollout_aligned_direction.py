#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.8 rollout-aligned direction diagnosis."""

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

from preflight_relational_teacher_v983_preservation_direction import _load_npz  # noqa: E402
from preflight_relational_teacher_v985_two_scene_step4 import _scene_binding  # noqa: E402
from relational_teacher_v9_all_sittable_contract import read_json, sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v985_two_scene_step4_preflight_contract import (  # noqa: E402
    EXPECTED_INSTANCES,
)
from relational_teacher_v988_rollout_aligned_direction_contract import (  # noqa: E402
    AUDIT_SCENE,
    CANDIDATE_NAMES,
    DEVELOPMENT_SCENE,
    DIAGNOSIS_TAG,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    RAW_GRAM_ASYMMETRY_CAP,
    RECONSTRUCTION_STEPS,
    SCHEMA,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    TASK_ORDER,
    V987_SCHEMA,
    canonical_sha256,
    conflict_pairs,
    diagnose_directions,
    rank_eligible_directions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
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


def _validate_paths(value: Mapping[str, object], label: str) -> None:
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError(label + " path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError(label + " bound file changed: " + str(name))


def main() -> None:
    args = parse_args()
    report_file = args.report.expanduser().resolve()
    value = read_json(report_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.8 report schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("diagnosis_tag") != DIAGNOSIS_TAG
        or value.get("diffusion_steps") != 500
        or value.get("source_scene") != SOURCE_SCENE
        or value.get("audit_scene") != AUDIT_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("selected_v984_step") != SELECTED_V984_STEP
        or value.get("reconstruction_steps") != list(RECONSTRUCTION_STEPS)
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("task_order") != list(TASK_ORDER)
    ):
        raise ValueError("Teacher-v9.8.8 protocol metadata changed")
    if value.get("policy_id") != POLICY_ID:
        raise ValueError("Teacher-v9.8.8 policy ID changed")
    _validate_paths(value, "Teacher-v9.8.8")
    paths = value["paths"]
    policy = read_json(Path(str(paths["diagnosis_policy"])).resolve())
    if policy != POLICY or canonical_sha256(policy) != POLICY_ID:
        raise ValueError("Teacher-v9.8.8 policy artifact changed")

    v987_file = Path(str(paths["v987_summary"])).resolve()
    v987 = read_json(v987_file)
    expected_audit = [
        "bed_pooled_mae_strictly_improves",
        "bed_pooled_recall_strictly_improves",
    ]
    if (
        v987.get("schema") != V987_SCHEMA
        or v987.get("status") != "FAIL"
        or v987.get("selected_radius") is not None
        or v987.get("failed_checks") != ["at_least_one_radius_is_admissible"]
        or v987.get("binding_id") != value.get("v987_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.7 failure changed")
    v987_rows = v987.get("response_rows")
    if not isinstance(v987_rows, list) or len(v987_rows) != 4:
        raise ValueError("sealed Teacher-v9.8.7 radius inventory changed")
    for row in v987_rows:
        if (
            row.get("scene_failed_checks", {}).get(SOURCE_SCENE) != []
            or row.get("scene_failed_checks", {}).get(AUDIT_SCENE) != expected_audit
        ):
            raise ValueError("sealed Teacher-v9.8.7 Bed regression changed")
    _validate_paths(v987, "Teacher-v9.8.7")

    geometry_file = Path(str(paths["gradient_geometry"])).resolve()
    geometry = _load_npz(geometry_file)
    expected_inventory = {
        "task_order",
        "task_losses",
        "gram",
        "raw_gram_max_asymmetry",
        "candidate_names",
        "coefficients",
        "directional_derivatives",
        "direction_sha256",
        "base_normalized",
    }
    if set(geometry) != expected_inventory:
        raise ValueError("Teacher-v9.8.8 geometry inventory changed")
    if (
        geometry["task_order"].tolist() != list(TASK_ORDER)
        or geometry["candidate_names"].tolist() != list(CANDIDATE_NAMES)
        or geometry["task_losses"].shape != (51,)
        or geometry["gram"].shape != (51, 51)
        or geometry["coefficients"].shape != (len(CANDIDATE_NAMES), 51)
        or geometry["directional_derivatives"].shape
        != (len(CANDIDATE_NAMES), 51)
        or geometry["direction_sha256"].shape != (len(CANDIDATE_NAMES),)
        or geometry["base_normalized"].shape != (2, 3, 2, 8192, 6)
    ):
        raise ValueError("Teacher-v9.8.8 geometry shape/order changed")
    for name, array in geometry.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Teacher-v9.8.8 geometry contains NaN/Inf: " + name)
    if sha256_file(geometry_file) != value.get("gradient_geometry_sha256"):
        raise ValueError("Teacher-v9.8.8 geometry digest changed")

    v987_arrays = _load_npz(Path(str(paths["v987_maps"])).resolve())
    if not np.array_equal(geometry["base_normalized"], v987_arrays["base_normalized"]):
        raise ValueError("Teacher-v9.8.7 exact K=3 Base reproduction changed")
    _close(value["task_losses"], geometry["task_losses"].tolist(), "task losses")
    _close(value["gram"], geometry["gram"].tolist(), "Gram matrix")
    raw_asymmetry = float(geometry["raw_gram_max_asymmetry"])
    if (
        raw_asymmetry > RAW_GRAM_ASYMMETRY_CAP
        or not math.isclose(
            float(value["raw_gram_max_asymmetry"]),
            raw_asymmetry,
            rel_tol=2e-6,
            abs_tol=2e-12,
        )
    ):
        raise ValueError("Teacher-v9.8.8 raw Gram asymmetry changed")

    recomputed_rows = diagnose_directions(geometry["gram"])
    direction_hashes = geometry["direction_sha256"].tolist()
    if len(direction_hashes) != len(CANDIDATE_NAMES) or any(
        not isinstance(digest, str) or len(digest) != 64 for digest in direction_hashes
    ):
        raise ValueError("Teacher-v9.8.8 direction digest inventory changed")
    for index, row in enumerate(recomputed_rows):
        row["direction_sha256"] = direction_hashes[index]
    _close(value["direction_candidates"], recomputed_rows, "direction candidates")
    _close(
        geometry["coefficients"].tolist(),
        [row["effective_coefficients"] for row in recomputed_rows],
        "direction coefficients",
    )
    _close(
        geometry["directional_derivatives"].tolist(),
        [row["directional_derivatives"] for row in recomputed_rows],
        "direction derivatives",
    )
    expected_conflicts = conflict_pairs(geometry["gram"])
    if (
        value.get("conflict_pairs") != expected_conflicts
        or value.get("conflict_pair_total") != 1275
    ):
        raise ValueError("Teacher-v9.8.8 conflict count changed")
    eligible_order = rank_eligible_directions(recomputed_rows)
    selected_candidate = eligible_order[0] if eligible_order else None
    if (
        value.get("eligible_selection_order") != eligible_order
        or value.get("selected_candidate") != selected_candidate
    ):
        raise ValueError("Teacher-v9.8.8 direction ranking changed")

    dataset_index = Path(str(paths["dataset_index"])).resolve()
    source_index = Path(str(paths["source_dataset_index"])).resolve()
    top_index = validate_top_index(dataset_index.parent, source_index.parent, dataset_index)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {SOURCE_SCENE, AUDIT_SCENE}:
        raise ValueError("two train-scene metadata inventory changed")
    bundles = {
        SOURCE_SCENE: load_train_scene_bundle(
            dataset_index.parent, source_index.parent, records[SOURCE_SCENE]
        ),
        AUDIT_SCENE: load_train_scene_bundle(
            dataset_index.parent, source_index.parent, records[AUDIT_SCENE]
        ),
    }
    for scene in (SOURCE_SCENE, AUDIT_SCENE):
        if tuple(bundles[scene]["instance_names"]) != tuple(EXPECTED_INSTANCES[scene]):
            raise ValueError(scene + " verified instance order changed")
    if (
        value.get("source_scene_binding") != _scene_binding(records[SOURCE_SCENE])
        or value.get("audit_scene_binding") != _scene_binding(records[AUDIT_SCENE])
    ):
        raise ValueError("two-scene metadata binding changed")

    expected_checks = {
        "sealed_v987_room0102_bed_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exact_v987_two_scene_k3_base_states_reproduced": True,
        "exactly_51_live_normalized_rollout_task_gradients": True,
        "at_least_one_rollout_aligned_common_direction_exists": (
            selected_candidate is not None
        ),
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "no_candidate_parameter_update_applied": True,
        "room0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed_checks = sorted(
        name for name, passed in expected_checks.items() if not passed
    )
    if (
        value.get("checks") != expected_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_actual_two_scene_k3_rollout_aligned_radius_grid")
        != (status == "PASS")
        or value.get("authorizes_cross_scene_objective_or_inference_redesign")
        != (status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.8.8 status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "v987_binding_id": value["v987_binding_id"],
            "policy_id": value["policy_id"],
            "selected_state_sha256": value["selected_state_sha256"],
            "gradient_geometry_sha256": value["gradient_geometry_sha256"],
            "selected_candidate": value["selected_candidate"],
            "status": value["status"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9.8.8 binding ID changed")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.8 contains forbidden model state")

    print("[ROLLOUT_ALIGNED_DIRECTION_{}] Teacher-v9.8.8 integrity".format(status))
    print("[PASS] sealed v9.8.7 Bed regression and exact K=3 states verified")
    print("[PASS] 51-task Gram, candidates, ranking and bindings recomputed")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")
    print("[OK] conflicts: {} / 1275".format(expected_conflicts))
    print("[OK] selected candidate:", selected_candidate)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

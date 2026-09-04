#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.6 cross-scene direction diagnosis."""

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
from relational_teacher_v9_lora_runtime import FORWARD_INPUT_KEYS  # noqa: E402
from relational_teacher_v984_preservation_calibration6_contract import (  # noqa: E402
    POLICY_ID as V984_POLICY_ID,
    SCHEMA as V984_REPORT_SCHEMA,
)
from relational_teacher_v985_two_scene_step4_preflight_contract import (  # noqa: E402
    EXPECTED_INSTANCES,
    POLICY_ID as V985_POLICY_ID,
)
from relational_teacher_v986_cross_scene_direction_contract import (  # noqa: E402
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
    STEP_RADIUS,
    TASK_ORDER,
    V985_SCHEMA,
    canonical_sha256,
    conflict_pairs,
    cross_scene_design_seeds,
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


def main() -> None:
    args = parse_args()
    report_file = args.report.expanduser().resolve()
    value = read_json(report_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.6 report schema changed")
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
        or float(value.get("step_radius_reconstruction_only")) != STEP_RADIUS
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("task_order") != list(TASK_ORDER)
        or value.get("source_design_seeds") != list(cross_scene_design_seeds(SOURCE_SCENE))
        or value.get("audit_design_seeds") != list(cross_scene_design_seeds(AUDIT_SCENE))
        or value.get("trajectory_accounting")
        != {
            "room0101_reconstruction_full_draws": 6,
            "room0101_reconstruction_exact_partial_repeats": 6,
            "room0101_diagnosis_full_draws": 2,
            "room0101_diagnosis_exact_partial_repeats": 2,
            "room0102_diagnosis_full_draws": 2,
            "room0102_diagnosis_exact_partial_repeats": 2,
        }
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.6 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.6 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.6 bound file changed: " + str(name))
    policy_value = read_json(Path(str(paths["diagnosis_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("Teacher-v9.8.6 policy changed")
    if value.get("policy_sha256") != hashes["diagnosis_policy"]:
        raise ValueError("Teacher-v9.8.6 policy file hash changed")

    v985_file = Path(str(paths["v985_report"])).resolve()
    v985 = read_json(v985_file)
    if (
        v985.get("schema") != V985_SCHEMA
        or v985.get("status") != "FAIL"
        or v985.get("policy_id") != V985_POLICY_ID
        or v985.get("failed_checks") != ["room0102_step4_response_is_admissible"]
        or v985.get("audit_failed_checks")
        != ["bed_pooled_mae_strictly_improves", "bed_pooled_recall_strictly_improves"]
        or v985.get("authorizes_cross_scene_direction_diagnosis") is not True
        or v985.get("binding_id") != value.get("v985_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.5 failure changed")
    v985_paths = v985.get("paths")
    v985_hashes = v985.get("path_sha256")
    if not isinstance(v985_paths, Mapping) or not isinstance(v985_hashes, Mapping):
        raise ValueError("sealed Teacher-v9.8.5 path binding changed")
    for name, raw in v985_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != v985_hashes[name]:
            raise ValueError("sealed Teacher-v9.8.5 bound file changed: " + str(name))
    if list(v985_file.parent.glob("*.pt")) or list(v985_file.parent.glob("*.pth")):
        raise ValueError("sealed Teacher-v9.8.5 contains forbidden model state")

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
        or value.get("selected_state_sha256")
        != v984["direction_rows"][SELECTED_V984_STEP - 1]["post_state_sha256"]
    ):
        raise ValueError("reconstructed LoRA state chain changed")

    geometry_file = Path(str(paths["gradient_geometry"])).resolve()
    if sha256_file(geometry_file) != value.get("gradient_geometry_sha256"):
        raise ValueError("gradient geometry hash changed")
    arrays = _load_npz(geometry_file)
    required = {
        "task_order",
        "task_losses",
        "gram",
        "candidate_names",
        "effective_coefficients",
        "directional_derivatives",
        "source_design_seeds",
        "audit_design_seeds",
    }
    if set(arrays) != required:
        raise ValueError("Teacher-v9.8.6 geometry inventory changed")
    if (
        arrays["task_order"].tolist() != list(TASK_ORDER)
        or arrays["task_losses"].shape != (len(TASK_ORDER),)
        or arrays["gram"].shape != (len(TASK_ORDER), len(TASK_ORDER))
        or arrays["candidate_names"].tolist() != list(CANDIDATE_NAMES)
        or arrays["effective_coefficients"].shape != (len(CANDIDATE_NAMES), len(TASK_ORDER))
        or arrays["directional_derivatives"].shape != (len(CANDIDATE_NAMES), len(TASK_ORDER))
        or arrays["source_design_seeds"].tolist() != list(cross_scene_design_seeds(SOURCE_SCENE))
        or arrays["audit_design_seeds"].tolist() != list(cross_scene_design_seeds(AUDIT_SCENE))
    ):
        raise ValueError("Teacher-v9.8.6 geometry shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Teacher-v9.8.6 geometry contains NaN/Inf: " + name)
    _close(value["task_losses"], arrays["task_losses"].tolist(), "task losses")
    _close(value["gram"], arrays["gram"].tolist(), "Gram matrix")
    raw_asymmetry = float(value.get("raw_gram_max_asymmetry", float("nan")))
    if (
        not math.isfinite(raw_asymmetry)
        or raw_asymmetry < 0.0
        or raw_asymmetry > RAW_GRAM_ASYMMETRY_CAP
    ):
        raise ValueError("raw Gram asymmetry audit changed")

    recomputed_rows = diagnose_directions(arrays["gram"])
    reported_rows = value.get("direction_candidates")
    if not isinstance(reported_rows, list) or len(reported_rows) != len(CANDIDATE_NAMES):
        raise ValueError("direction candidate inventory changed")
    for index, (reported, recomputed) in enumerate(zip(reported_rows, recomputed_rows)):
        for key in (
            "name",
            "effective_coefficients",
            "directional_derivatives",
            "minimum_directional_derivative",
            "minimum_audit_bed_derivative",
            "checks",
            "failed_checks",
            "eligible",
        ):
            _close(reported[key], recomputed[key], "candidate {} {}".format(index, key))
        if (
            not isinstance(reported.get("direction_sha256"), str)
            or len(reported["direction_sha256"]) != 64
            or not math.isclose(float(reported.get("direction_l2")), 1.0, rel_tol=2e-5, abs_tol=2e-6)
        ):
            raise ValueError("candidate direction digest/norm changed")
    _close(
        arrays["effective_coefficients"].tolist(),
        [row["effective_coefficients"] for row in recomputed_rows],
        "saved coefficients",
    )
    _close(
        arrays["directional_derivatives"].tolist(),
        [row["directional_derivatives"] for row in recomputed_rows],
        "saved derivatives",
    )
    expected_conflicts = conflict_pairs(arrays["gram"])
    _close(value.get("conflict_pairs"), expected_conflicts, "conflict pairs")
    if value.get("conflict_pair_count") != len(expected_conflicts):
        raise ValueError("conflict count changed")
    selection_order = rank_eligible_directions(recomputed_rows)
    selected = selection_order[0] if selection_order else None
    if value.get("eligible_selection_order") != selection_order or value.get(
        "selected_candidate"
    ) != selected:
        raise ValueError("cross-scene direction ranking changed")

    dataset_index = Path(str(paths["dataset_index"])).resolve()
    source_index = Path(str(paths["source_dataset_index"])).resolve()
    top_index = validate_top_index(dataset_index.parent, source_index.parent, dataset_index)
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != {SOURCE_SCENE, AUDIT_SCENE}:
        raise ValueError("two train-scene metadata inventory changed")
    source_bundle = load_train_scene_bundle(
        dataset_index.parent, source_index.parent, records[SOURCE_SCENE]
    )
    audit_bundle = load_train_scene_bundle(
        dataset_index.parent, source_index.parent, records[AUDIT_SCENE]
    )
    if (
        tuple(source_bundle["instance_names"]) != tuple(EXPECTED_INSTANCES[SOURCE_SCENE])
        or tuple(audit_bundle["instance_names"]) != tuple(EXPECTED_INSTANCES[AUDIT_SCENE])
        or value.get("source_scene_binding") != _scene_binding(records[SOURCE_SCENE])
        or value.get("audit_scene_binding") != _scene_binding(records[AUDIT_SCENE])
    ):
        raise ValueError("two-scene GT binding changed")

    expected_checks = {
        "sealed_v985_bed_only_cross_scene_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exactly_19_live_normalized_task_gradients": True,
        "at_least_one_strict_cross_scene_direction_exists": selected is not None,
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "no_candidate_parameter_update_applied": True,
        "room0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
    }
    expected_status = "PASS" if all(expected_checks.values()) else "FAIL"
    expected_failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("checks") != expected_checks
        or value.get("status") != expected_status
        or value.get("failed_checks") != expected_failed
        or value.get("authorizes_actual_two_scene_k3_direction_response_grid")
        != (expected_status == "PASS")
        or value.get("authorizes_objective_or_capacity_redesign")
        != (expected_status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.8.6 status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "v985_binding_id": value["v985_binding_id"],
            "policy_id": value["policy_id"],
            "selected_state_sha256": value["selected_state_sha256"],
            "gradient_geometry_sha256": value["gradient_geometry_sha256"],
            "selected_candidate": value["selected_candidate"],
            "status": value["status"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9.8.6 binding ID changed")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.6 contains forbidden model state")

    print("[CROSS_SCENE_DIRECTION_{}] Teacher-v9.8.6 integrity".format(expected_status))
    print("[PASS] sealed room_0102 Bed failure and exact step-4 chain verified")
    print("[PASS] 19-task Gram geometry, candidates and ranking recomputed")
    print("[PASS] no candidate update/checkpoint, room_0201 or paper-test access")
    print("[OK] conflicts: {} / 171".format(len(expected_conflicts)))
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", expected_failed)


if __name__ == "__main__":
    main()

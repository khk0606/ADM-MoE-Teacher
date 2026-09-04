#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.5 two-scene step-4 preflight."""

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
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    EXPECTED_INSTANCES,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PREFLIGHT_TAG,
    PROMPT_IDS,
    RECONSTRUCTION_STEPS,
    SCHEMA,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    STEP_RADIUS,
    absolute_presence_checks,
    canonical_sha256,
    pooled_by_role,
    two_scene_preflight_checks,
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


def _scene_binding(record: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "scene_id": record["scene_id"],
            "manifest_sha256": record["manifest_sha256"],
            "verified_target_instances": record["verified_target_instances"],
            "source_scene_record": record["source_scene_record"],
            "rows": record["rows"],
        }
    )


def main() -> None:
    args = parse_args()
    report_file = args.report.expanduser().resolve()
    value = read_json(report_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.5 report schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("preflight_tag") != PREFLIGHT_TAG
        or value.get("diffusion_steps") != 500
        or value.get("source_scene") != SOURCE_SCENE
        or value.get("audit_scene") != AUDIT_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("selected_v984_step") != SELECTED_V984_STEP
        or value.get("reconstruction_steps") != list(RECONSTRUCTION_STEPS)
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or float(value.get("step_radius")) != STEP_RADIUS
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("audit_seed_table")
        != [list(stable_rollout_seeds(generation)) for generation in range(3)]
        or value.get("trajectory_accounting")
        != {
            "room0102_base_full_draws": 6,
            "room0101_design_base_full_draws": 6,
            "room0101_design_exact_partial_repeats": 6,
            "room0102_candidate_partial_resumes": 6,
        }
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.5 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.5 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.5 bound file changed: " + str(name))
    policy_value = read_json(Path(str(paths["preflight_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("Teacher-v9.8.5 policy changed")
    if value.get("policy_sha256") != hashes["preflight_policy"]:
        raise ValueError("Teacher-v9.8.5 policy file hash changed")

    v984 = read_json(Path(str(paths["v984_summary"])).resolve())
    if (
        v984.get("schema") != V984_REPORT_SCHEMA
        or v984.get("status") != "PASS"
        or v984.get("policy_id") != V984_POLICY_ID
        or v984.get("shortlisted_steps") != [4, 3]
        or v984.get("failed_checks")
        or v984.get("authorizes_two_scene_preservation_calibration_preflight") is not True
        or v984.get("binding_id") != value.get("v984_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.4 PASS changed")
    v984_paths = v984.get("paths")
    v984_hashes = v984.get("path_sha256")
    if not isinstance(v984_paths, Mapping) or not isinstance(v984_hashes, Mapping):
        raise ValueError("sealed Teacher-v9.8.4 path binding changed")
    for name, raw in v984_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != v984_hashes[name]:
            raise ValueError("sealed Teacher-v9.8.4 bound file changed: " + str(name))
    if list(Path(str(paths["v984_summary"])).resolve().parent.glob("*.pt")) or list(
        Path(str(paths["v984_summary"])).resolve().parent.glob("*.pth")
    ):
        raise ValueError("sealed Teacher-v9.8.4 contains forbidden model state")

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

    arrays_file = Path(str(paths["audit_maps"])).resolve()
    if sha256_file(arrays_file) != value.get("audit_maps_sha256"):
        raise ValueError("audit map hash changed")
    arrays = _load_npz(arrays_file)
    required = {
        "xyz",
        "points",
        "instance_ids",
        "category_ids",
        "instance_names",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
        "all_sittable_gt",
        "prompt_ids",
        "audit_seed_table",
        "base_normalized",
        "candidate_normalized",
        "base",
        "candidate",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_prediction",
    }
    if set(arrays) != required:
        raise ValueError("Teacher-v9.8.5 array inventory changed")
    if (
        arrays["base_normalized"].shape != (3, 2, 8192, 6)
        or arrays["candidate_normalized"].shape != (3, 2, 8192, 6)
        or arrays["base"].shape != (3, 2, 8192, 6)
        or arrays["candidate"].shape != (3, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_prediction"].shape != (3, 8192, 6)
        or arrays["instance_names"].tolist() != list(EXPECTED_INSTANCES[AUDIT_SCENE])
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["audit_seed_table"].tolist()
        != [list(stable_rollout_seeds(generation)) for generation in range(3)]
    ):
        raise ValueError("Teacher-v9.8.5 array shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Teacher-v9.8.5 array contains NaN/Inf: " + name)

    v984_arrays = _load_npz(Path(str(paths["v984_maps"])).resolve())
    if not np.array_equal(arrays["base_v5_prediction"], v984_arrays["base_v5_prediction"]):
        raise ValueError("fresh v5r4 fixed probe changed")
    if not np.array_equal(
        arrays["candidate_v5_prediction"],
        v984_arrays["candidate_v5_predictions"][SELECTED_V984_STEP - 1],
    ):
        raise ValueError("reconstructed step-4 v5 prediction changed")

    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    expected_base = np.clip(
        arrays["base_normalized"] * std.reshape(1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    expected_candidate = np.clip(
        arrays["candidate_normalized"] * std.reshape(1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 6),
        0.0,
        1.0,
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
    bundle = load_train_scene_bundle(
        dataset_index.parent, source_index.parent, records[AUDIT_SCENE]
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
            raise ValueError("saved room_0102 scene array changed: " + name)
    if not np.array_equal(np.asarray(bundle["all_target"]), arrays["all_sittable_gt"]):
        raise ValueError("saved room_0102 all-sittable GT changed")
    if (
        value.get("source_scene_binding") != _scene_binding(records[SOURCE_SCENE])
        or value.get("audit_scene_binding") != _scene_binding(records[AUDIT_SCENE])
    ):
        raise ValueError("two-scene metadata binding changed")

    base_rows = _rows(bundle, arrays["base"])
    candidate_rows = _rows(bundle, arrays["candidate"])
    instance_names = list(bundle["instance_names"])
    base_pooled = pooled_by_role(base_rows, instance_names)
    candidate_pooled = pooled_by_role(candidate_rows, instance_names)
    mask = np.asarray(bundle["verified_positive_mask"], bool)
    base_invariance = _invariance(arrays["base"], mask)
    candidate_invariance = _invariance(arrays["candidate"], mask)
    base_v5_dense = ((arrays["base_v5_prediction"] - arrays["v5_target"]) ** 2).mean(
        axis=(1, 2)
    ).tolist()
    candidate_v5_dense = (
        (arrays["candidate_v5_prediction"] - arrays["v5_target"]) ** 2
    ).mean(axis=(1, 2)).tolist()
    maximum_map_delta = float(np.abs(arrays["candidate"] - arrays["base"]).max())
    checks = two_scene_preflight_checks(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        instance_names=instance_names,
        base_prompt_invariance=base_invariance,
        candidate_prompt_invariance=candidate_invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        maximum_map_delta=maximum_map_delta,
    )
    presence = [
        [
            absolute_presence_checks(candidate_rows[g][p], instance_names)
            for p in range(2)
        ]
        for g in range(3)
    ]
    for key, actual in (
        ("instance_names", instance_names),
        ("base_rows", base_rows),
        ("candidate_rows", candidate_rows),
        ("base_pooled", base_pooled),
        ("candidate_pooled", candidate_pooled),
        ("base_prompt_invariance", base_invariance),
        ("candidate_prompt_invariance", candidate_invariance),
        ("base_v5_dense", base_v5_dense),
        ("candidate_v5_dense", candidate_v5_dense),
        ("maximum_map_delta", maximum_map_delta),
        ("presence", presence),
        ("audit_checks", checks),
        ("audit_failed_checks", sorted(name for name, passed in checks.items() if not passed)),
    ):
        _close(value[key], actual, key)

    audit_eligible = all(checks.values())
    top_checks = {
        "sealed_v984_pass_and_step4_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "room0101_step4_v5_exactly_reproduced": True,
        "room0102_actual_k3_two_prompt_audit_completed": True,
        "room0102_step4_response_is_admissible": audit_eligible,
        "policy_locked_before_scene_arrays_and_model": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "room0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
    }
    status = "PASS" if all(top_checks.values()) else "FAIL"
    failed_checks = sorted(name for name, passed in top_checks.items() if not passed)
    if (
        value.get("checks") != top_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_two_scene_preservation_response_preflight")
        != (status == "PASS")
        or value.get("authorizes_cross_scene_direction_diagnosis") != (status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.8.5 status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "v984_binding_id": value["v984_binding_id"],
            "policy_id": value["policy_id"],
            "selected_state_sha256": value["selected_state_sha256"],
            "audit_maps_sha256": value["audit_maps_sha256"],
            "status": value["status"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9.8.5 binding ID changed")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.5 output contains forbidden model state")

    print("[TWO_SCENE_STEP4_{}] Teacher-v9.8.5 integrity".format(status))
    print("[PASS] sealed v9.8.4 and exact update-1..4 state chain verified")
    print("[PASS] room_0102 K=3 maps, three roles, v5 and retention recomputed")
    print("[PASS] no optimizer/checkpoint, room_0201 arrays or paper-test access")
    print("[OK] failed checks:", failed_checks)
    print("[OK] audit failed checks:", value["audit_failed_checks"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deep validator for Teacher-v9.8.10 rollout-aligned recovery bracket."""

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
from relational_teacher_v9810_rollout_aligned_recovery_contract import (  # noqa: E402
    AUDIT_SCENE,
    DEVELOPMENT_SCENE,
    FAILED_RADIUS_GRID,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    RADIUS_GRID,
    RECONSTRUCTION_STEPS,
    RESPONSE_TAG,
    SCHEMA,
    SELECTED_DIRECTION,
    SELECTED_TIMESTEP,
    SELECTED_V984_STEP,
    SOURCE_SCENE,
    V988_SCHEMA,
    V989_SCHEMA,
    canonical_sha256,
    radius_response_checks,
    rank_eligible_radii,
    validate_failed_recovery_signature,
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
    args = parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)
    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v9.8.10 report schema changed")
    if (
        value.get("model_seed") != MODEL_SEED
        or value.get("response_tag") != RESPONSE_TAG
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
        or value.get("radius_grid") != list(RADIUS_GRID)
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("audit_seed_table")
        != [list(stable_rollout_seeds(generation)) for generation in range(3)]
        or value.get("trajectory_accounting")
        != {
            "two_scene_base_full_draws": 12,
            "room0101_reconstruction_full_draws": 6,
            "room0101_reconstruction_exact_partial_repeats": 6,
            "rollout_aligned_gradient_groups": 6,
            "candidate_partial_resumes": len(RADIUS_GRID) * 12,
        }
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.10 sealed protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.8.10 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.10 bound file changed: " + str(name))
    policy_value = read_json(Path(str(paths["response_policy"])).resolve())
    if policy_value != POLICY or canonical_sha256(policy_value) != POLICY_ID:
        raise ValueError("Teacher-v9.8.10 policy changed")
    if value.get("policy_sha256") != hashes["response_policy"]:
        raise ValueError("Teacher-v9.8.10 policy file hash changed")

    v989 = read_json(Path(str(paths["v989_summary"])).resolve())
    if (
        v989.get("schema") != V989_SCHEMA
        or v989.get("status") != "FAIL"
        or v989.get("selected_radius") is not None
        or v989.get("radius_grid") != list(FAILED_RADIUS_GRID)
        or v989.get("failed_checks") != ["at_least_one_radius_is_admissible"]
        or v989.get("authorizes_cross_scene_objective_or_inference_redesign") is not True
        or v989.get("binding_id") != value.get("v989_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.9 recovery authority changed")
    validate_failed_recovery_signature(v989)

    v988 = read_json(Path(str(paths["v988_report"])).resolve())
    if (
        v988.get("schema") != V988_SCHEMA
        or v988.get("status") != "PASS"
        or v988.get("policy_id") != V988_POLICY_ID
        or v988.get("selected_candidate") != SELECTED_DIRECTION
        or not v988.get("eligible_selection_order")
        or v988["eligible_selection_order"][0] != SELECTED_DIRECTION
        or v988.get("failed_checks")
        or v988.get("authorizes_actual_two_scene_k3_rollout_aligned_radius_grid") is not True
        or v988.get("binding_id") != value.get("v988_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.8 PASS changed")
    v987 = read_json(Path(str(paths["v987_summary"])).resolve())
    if (
        v987.get("status") != "FAIL"
        or v987.get("selected_radius") is not None
        or v987.get("binding_id") != value.get("v987_binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.7 failure changed")
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
        or value.get("selected_state_sha256") != v988.get("selected_state_sha256")
    ):
        raise ValueError("reconstructed step-4 state changed")
    selected_v988 = [
        row for row in v988["direction_candidates"] if row["name"] == SELECTED_DIRECTION
    ]
    if (
        len(selected_v988) != 1
        or selected_v988[0].get("eligible") is not True
        or value.get("selected_direction_sha256") != selected_v988[0]["direction_sha256"]
    ):
        raise ValueError("selected rollout-aligned direction binding changed")
    v988_geometry = _load_npz(Path(str(paths["v988_geometry"])).resolve())
    _close(value["direction_task_losses"], v988_geometry["task_losses"].tolist(), "direction losses")
    _close(value["direction_gram"], v988_geometry["gram"].tolist(), "direction Gram")
    raw_asymmetry = float(value.get("direction_raw_gram_max_asymmetry", float("nan")))
    if not math.isfinite(raw_asymmetry) or not 0.0 <= raw_asymmetry <= RAW_GRAM_ASYMMETRY_CAP:
        raise ValueError("direction raw Gram asymmetry changed")

    arrays_file = Path(str(paths["response_maps"])).resolve()
    if sha256_file(arrays_file) != value.get("response_maps_sha256"):
        raise ValueError("response maps hash changed")
    arrays = _load_npz(arrays_file)
    base_required = {
        "scene_ids",
        "prompt_ids",
        "audit_seed_table",
        "radius_grid",
        "base_normalized",
        "candidates_normalized",
        "base",
        "candidates",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    }
    per_scene_suffixes = {
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
    }
    required = base_required | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in per_scene_suffixes
    }
    if set(arrays) != required:
        raise ValueError("Teacher-v9.8.10 array inventory changed")
    if (
        arrays["scene_ids"].tolist() != [SOURCE_SCENE, AUDIT_SCENE]
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["audit_seed_table"].tolist()
        != [list(stable_rollout_seeds(generation)) for generation in range(3)]
        or arrays["radius_grid"].tolist() != list(RADIUS_GRID)
        or arrays["base_normalized"].shape != (2, 3, 2, 8192, 6)
        or arrays["candidates_normalized"].shape
        != (len(RADIUS_GRID), 2, 3, 2, 8192, 6)
        or arrays["base"].shape != (2, 3, 2, 8192, 6)
        or arrays["candidates"].shape != (len(RADIUS_GRID), 2, 3, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape
        != (len(RADIUS_GRID), 3, 8192, 6)
    ):
        raise ValueError("Teacher-v9.8.10 array shape/order changed")
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Teacher-v9.8.10 array contains NaN/Inf: " + name)
    if not np.array_equal(arrays["base_normalized"], v988_geometry["base_normalized"]):
        raise ValueError("exact v9.8.8 Base K=3 maps changed")
    v989_arrays = _load_npz(Path(str(paths["v989_maps"])).resolve())
    if not np.array_equal(arrays["base_normalized"], v989_arrays["base_normalized"]):
        raise ValueError("exact v9.8.9 Base K=3 maps changed")

    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    expected_base = np.clip(
        arrays["base_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    expected_candidates = np.clip(
        arrays["candidates_normalized"] * std.reshape(1, 1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    if not np.array_equal(arrays["base"], expected_base) or not np.array_equal(
        arrays["candidates"], expected_candidates
    ):
        raise ValueError("normalized-to-physical conversion changed")

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
    _close(value["base_v5_dense"], base_v5_dense, "base v5")
    scene_base_rows = {}
    scene_base_pooled = {}
    scene_base_invariance = {}
    for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
        scene_base_rows[scene] = _rows(bundles[scene], arrays["base"][scene_index])
        scene_base_pooled[scene] = pooled_by_role(
            scene_base_rows[scene], list(bundles[scene]["instance_names"])
        )
        scene_base_invariance[scene] = _invariance(
            arrays["base"][scene_index],
            np.asarray(bundles[scene]["verified_positive_mask"], bool),
        )
    _close(value["scene_base_rows"], scene_base_rows, "base rows")
    _close(value["scene_base_pooled"], scene_base_pooled, "base pooled")
    _close(value["scene_base_invariance"], scene_base_invariance, "base invariance")

    response_rows = value.get("response_rows")
    if not isinstance(response_rows, list) or len(response_rows) != len(RADIUS_GRID):
        raise ValueError("radius response row inventory changed")
    recomputed_rows = []
    selected_direction_sha256 = value["selected_direction_sha256"]
    selected_state_sha256 = value["selected_state_sha256"]
    for radius_index, radius in enumerate(RADIUS_GRID):
        reported = response_rows[radius_index]
        candidate_v5_dense = (
            (arrays["candidate_v5_predictions"][radius_index] - arrays["v5_target"]) ** 2
        ).mean(axis=(1, 2)).tolist()
        scene_candidate_rows = {}
        scene_candidate_pooled = {}
        scene_candidate_invariance = {}
        scene_checks = {}
        scene_presence = {}
        maximum_map_delta = {}
        for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):
            instance_names = list(bundles[scene]["instance_names"])
            candidate = arrays["candidates"][radius_index, scene_index]
            scene_candidate_rows[scene] = _rows(bundles[scene], candidate)
            scene_candidate_pooled[scene] = pooled_by_role(
                scene_candidate_rows[scene], instance_names
            )
            scene_candidate_invariance[scene] = _invariance(
                candidate, np.asarray(bundles[scene]["verified_positive_mask"], bool)
            )
            maximum_map_delta[scene] = float(
                np.abs(candidate - arrays["base"][scene_index]).max()
            )
            scene_checks[scene] = two_scene_preflight_checks(
                base_rows=scene_base_rows[scene],
                candidate_rows=scene_candidate_rows[scene],
                instance_names=instance_names,
                base_prompt_invariance=scene_base_invariance[scene],
                candidate_prompt_invariance=scene_candidate_invariance[scene],
                base_v5_dense=base_v5_dense,
                candidate_v5_dense=candidate_v5_dense,
                maximum_map_delta=maximum_map_delta[scene],
            )
            scene_presence[scene] = [
                [
                    absolute_presence_checks(
                        scene_candidate_rows[scene][generation][prompt], instance_names
                    )
                    for prompt in range(2)
                ]
                for generation in range(3)
            ]
        if (
            not isinstance(reported.get("candidate_state_sha256"), str)
            or len(reported["candidate_state_sha256"]) != 64
            or reported["candidate_state_sha256"] == selected_state_sha256
        ):
            raise ValueError("candidate state digest changed")
        checks = radius_response_checks(
            source_checks=scene_checks[SOURCE_SCENE],
            audit_checks=scene_checks[AUDIT_SCENE],
            selected_direction_reproduced=(
                reported.get("selected_direction") == SELECTED_DIRECTION
                and reported.get("selected_direction_sha256") == selected_direction_sha256
            ),
            candidate_state_changed=True,
        )
        recomputed = {
            "radius": radius,
            "pre_state_sha256": selected_state_sha256,
            "candidate_state_sha256": reported["candidate_state_sha256"],
            "selected_direction": SELECTED_DIRECTION,
            "selected_direction_sha256": selected_direction_sha256,
            "scene_base_pooled": scene_base_pooled,
            "scene_candidate_rows": scene_candidate_rows,
            "scene_candidate_pooled": scene_candidate_pooled,
            "scene_candidate_invariance": scene_candidate_invariance,
            "candidate_v5_dense": candidate_v5_dense,
            "maximum_map_delta": maximum_map_delta,
            "presence": scene_presence,
            "scene_checks": scene_checks,
            "scene_failed_checks": {
                scene: sorted(name for name, passed in scene_checks[scene].items() if not passed)
                for scene in (SOURCE_SCENE, AUDIT_SCENE)
            },
            "checks": checks,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
            "eligible": all(checks.values()),
        }
        _close(reported, recomputed, "radius response")
        recomputed_rows.append(recomputed)

    candidate_state_hashes = [row["candidate_state_sha256"] for row in response_rows]
    if len(set(candidate_state_hashes)) != len(RADIUS_GRID):
        raise ValueError("distinct radii did not produce distinct LoRA states")
    selection_order = rank_eligible_radii(recomputed_rows)
    selected_radius = selection_order[0] if selection_order else None
    if (
        value.get("eligible_selection_order") != selection_order
        or value.get("selected_radius") != selected_radius
    ):
        raise ValueError("radius ranking changed")
    expected_checks = {
        "sealed_v989_bed_only_monotone_recovery_failure_bound": True,
        "sealed_v988_pass_and_selected_direction_bound": True,
        "fresh_v5r4_zero_init": True,
        "updates_1_through_4_exactly_reconstructed": True,
        "exact_v988_51_task_direction_recomputed": True,
        "five_recovery_radii_share_exact_step4_start": True,
        "five_recovery_radii_produce_distinct_lora_states": True,
        "both_scene_actual_k3_two_prompt_responses_completed": True,
        "at_least_one_recovery_radius_is_admissible": selected_radius is not None,
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
        or value.get("authorizes_fresh_two_scene_rollout_aligned_calibration")
        != (status == "PASS")
        or value.get("authorizes_cross_scene_training_objective_redesign")
        != (status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.8.10 status/authorization changed")
    expected_binding = canonical_sha256(
        {
            "v989_binding_id": value["v989_binding_id"],
            "policy_id": value["policy_id"],
            "selected_state_sha256": value["selected_state_sha256"],
            "selected_direction_sha256": value["selected_direction_sha256"],
            "response_maps_sha256": value["response_maps_sha256"],
            "selected_radius": value["selected_radius"],
            "status": value["status"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9.8.10 binding ID changed")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.10 contains forbidden model state")

    print("[ROLLOUT_ALIGNED_RECOVERY_{}] Teacher-v9.8.10 integrity".format(status))
    print("[PASS] v9.8.9 monotone recovery and exact v9.8.8 direction verified")
    print("[PASS] both scene K=3 maps, three roles, negatives and v5 recomputed")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")
    print("[OK] selected radius:", selected_radius)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

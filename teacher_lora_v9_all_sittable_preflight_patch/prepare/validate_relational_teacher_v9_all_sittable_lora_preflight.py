#!/usr/bin/env python3
"""Validate the sealed Teacher-v9 all-sittable CUDA preflight report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

from relational_teacher_v9_all_sittable_contract import PROMPTS, sha256_file
from relational_teacher_v9_lora_preflight_contract import (
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TARGETS,
    EXPECTED_TRAIN_SCENES,
    OBJECTIVE_WEIGHTS,
    POLICY_SCHEMA,
    RELATIVE_SELECTION_RULES,
    SCHEMA,
    canonical_sha256,
)


REQUIRED_PATHS = {
    "source_dataset_index",
    "dataset_index",
    "v5_split",
    "stats_file",
    "original_checkpoint",
    "v5_checkpoint",
    "v5_evidence_report",
    "metric_policy",
    "preflight",
    "validator",
    "objective",
    "runtime",
    "preflight_contract",
    "dataset_contract",
    "metrics",
    "dataset_validator",
    "lora",
    "v5_common",
    "v5_loader",
}
REQUIRED_COMPONENTS = {
    "bed_primary",
    "normal_chair_primary",
    "high_chair_primary",
    "instance_macro_primary",
    "verified_union",
    "environment_auxiliary",
    "paired_prompt_invariance",
    "explicit_negative_addition_perturbed",
    "v5_replay_preservation_perturbed",
    "lora_regularizer_perturbed",
}
REQUIRED_CHECKS = {
    "cuda_used",
    "sealed_v9_all_sittable_index",
    "exact_two_train_scenes",
    "development_payloads_unread",
    "one_primary_gt_per_scene",
    "watch_write_share_gt_timestep_noise",
    "bed_normal_high_present_simultaneously",
    "equal_instance_macro_not_point_weighted",
    "unknown_sittable_ignored_not_negative",
    "explicit_negative_is_tv_desk_whiteboard_only",
    "counterexamples_fail_closed",
    "metric_policy_locked_before_optimizer",
    "gate0a_v5_checkpoint_bound",
    "original_checkpoint_bound",
    "fresh_zero_init_v5_output_bitwise_equal",
    "only_lora_parameters_trainable",
    "every_instance_component_gradient_reaches_lora",
    "union_environment_prompt_gradients_reach_lora",
    "negative_replay_regularizer_gradients_reach_lora_when_perturbed",
    "total_objective_gradient_reaches_lora",
    "frozen_cdm_has_no_gradient",
    "teacher_forward_is_text_plus_scene_only",
}


def _positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(label + " must be finite and positive")
    return number


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(label + " contains a non-finite value")


def validate_policy(path: Path, expected_hash: str, expected_id: str) -> Mapping[str, object]:
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError("Teacher-v9 metric policy hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != POLICY_SCHEMA
        or value.get("status") != "LOCKED_BEFORE_OPTIMIZER"
        or value.get("optimizer_constructed") is not False
        or value.get("optimizer_updates") != 0
        or value.get("train_scenes") != list(EXPECTED_TRAIN_SCENES)
        or value.get("development_scenes_unread")
        != list(EXPECTED_DEVELOPMENT_SCENES)
        or value.get("panel_rows")
        != [
            "room_0101|sit_watch_v1",
            "room_0101|sit_write_v1",
            "room_0102|sit_watch_v1",
            "room_0102|sit_write_v1",
        ]
        or value.get("prompt_pairing")
        != "same_scene_same_x0_same_timestep_same_noise"
        or value.get("timesteps") != [125, 125, 375, 375]
        or value.get("v5_timesteps") != [100, 300, 450]
        or value.get("absolute_presence_limits") != ABSOLUTE_PRESENCE_LIMITS
        or value.get("relative_selection_rules") != RELATIVE_SELECTION_RULES
        or value.get("objective_weights") != OBJECTIVE_WEIGHTS
        or value.get("unknown_sittable_policy")
        != "ignored_in_all_v9_task_losses_never_negative"
        or value.get("explicit_negative_policy")
        != "tv_desk_whiteboard_addition_over_frozen_only"
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v9 metric policy contract changed")
    if len(str(value.get("relational_noise_sha256", ""))) != 64:
        raise ValueError("relational noise hash is invalid")
    if len(str(value.get("v5_noise_sha256", ""))) != 64:
        raise ValueError("v5 noise hash is invalid")
    if len(value.get("v5_probe_ids", ())) != 3:
        raise ValueError("v5 fixed panel changed")
    baseline = value.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Teacher-v9 frozen baseline is absent")
    rows = baseline.get("per_scene_prompt_instance_metrics")
    if not isinstance(rows, Mapping) or set(rows) != set(value["panel_rows"]):
        raise ValueError("Teacher-v9 baseline row inventory changed")
    for row in rows.values():
        instances = row.get("instances") if isinstance(row, Mapping) else None
        if not isinstance(instances, Mapping) or len(instances) != 3:
            raise ValueError("a baseline row does not report all three instances")
    if set(baseline.get("v5_replay_dense_mse_per_target", {})) != {
        "chair",
        "bed",
        "whiteboard",
    }:
        raise ValueError("Teacher-v9 v5 baseline target inventory changed")
    counterexamples = value.get("counterexamples")
    if not isinstance(counterexamples, Mapping) or set(counterexamples) != set(
        EXPECTED_TRAIN_SCENES
    ):
        raise ValueError("Teacher-v9 counterexample scene inventory changed")
    expected_names = {
        "valid_all_three",
        "bed_only",
        "missing_normal_chair",
        "missing_high_chair",
        "same_object_displacement",
        "explicit_negative_hotspot",
    }
    for scene_rows in counterexamples.values():
        if not isinstance(scene_rows, Mapping) or set(scene_rows) != expected_names:
            raise ValueError("Teacher-v9 counterexample inventory changed")
        if scene_rows["valid_all_three"].get("all_checks_pass") is not True:
            raise ValueError("exact all-sittable GT did not pass")
        for name in expected_names - {"valid_all_three"}:
            if scene_rows[name].get("all_checks_pass") is not False:
                raise ValueError(name + " was not rejected")
    _finite_tree(value, "Teacher-v9 metric policy")
    calculated_id = canonical_sha256(
        {key: nested for key, nested in value.items() if key not in {"created_utc", "policy_id"}}
    )
    if value.get("policy_id") != expected_id or calculated_id != expected_id:
        raise ValueError("Teacher-v9 metric policy ID arithmetic changed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = json.loads(report_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") != "PASS":
        raise ValueError("Teacher-v9 LoRA preflight schema/status changed")
    if value.get("train_scenes") != list(EXPECTED_TRAIN_SCENES):
        raise ValueError("Teacher-v9 train scenes changed")
    if value.get("development_scenes_metadata_only") != list(
        EXPECTED_DEVELOPMENT_SCENES
    ):
        raise ValueError("Teacher-v9 development scene changed")
    if value.get("development_payloads_read") is not False:
        raise ValueError("Teacher-v9 preflight read development arrays")
    if value.get("forward_input_keys") != sorted(
        ("c_pc_feat", "c_pc_xyz", "c_text")
    ):
        raise ValueError("Teacher-v9 forward keys changed")
    if value.get("prompts") != PROMPTS or value.get("diffusion_steps") != 500:
        raise ValueError("Teacher-v9 prompt/diffusion contract changed")

    scene_bindings = value.get("scene_bindings")
    if not isinstance(scene_bindings, list) or len(scene_bindings) != 2:
        raise ValueError("Teacher-v9 scene bindings changed")
    for row in scene_bindings:
        scene_id = str(row.get("scene_id", ""))
        if scene_id not in EXPECTED_TARGETS:
            raise ValueError("unexpected Teacher-v9 scene binding")
        if tuple(row.get("instance_names", ())) != EXPECTED_TARGETS[scene_id]:
            raise ValueError(scene_id + ": target binding changed")
        if tuple(row.get("instance_roles", ())) != (
            "bed",
            "normal_chair",
            "high_chair",
        ):
            raise ValueError(scene_id + ": target roles changed")
        for key in ("manifest_sha256", "consensus_sha256", "points_sha256"):
            if len(str(row.get(key, ""))) != 64:
                raise ValueError(scene_id + ": invalid source hash")

    panel = value.get("panel")
    if not isinstance(panel, Mapping) or panel.get("row_ids") != [
        "room_0101|sit_watch_v1",
        "room_0101|sit_write_v1",
        "room_0102|sit_watch_v1",
        "room_0102|sit_write_v1",
    ]:
        raise ValueError("Teacher-v9 fixed panel changed")
    if panel.get("timesteps") != [125, 125, 375, 375] or panel.get(
        "v5_timesteps"
    ) != [100, 300, 450]:
        raise ValueError("Teacher-v9 fixed timesteps changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != REQUIRED_PATHS
        or set(hashes) != REQUIRED_PATHS
    ):
        raise ValueError("Teacher-v9 immutable path inventory changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(hashes[name]):
            raise ValueError("Teacher-v9 bound path changed: " + name)
    policy = validate_policy(
        Path(str(paths["metric_policy"])).resolve(),
        str(value.get("metric_policy_sha256", "")),
        str(value.get("metric_policy_id", "")),
    )
    if policy["panel_rows"] != panel["row_ids"]:
        raise ValueError("report/policy panel binding differs")
    for key in (
        "relational_noise_sha256",
        "v5_probe_ids",
        "v5_timesteps",
        "v5_noise_sha256",
    ):
        if panel.get(key) != policy.get(key):
            raise ValueError("report/policy fixed panel differs: " + key)

    for key in (
        "zero_init_parity_max_abs",
        "zero_init_v5_parity_max_abs",
        "zero_init_negative_addition",
        "zero_init_v5_preservation",
        "zero_init_lora_energy",
    ):
        if float(value.get(key, -1.0)) != 0.0:
            raise ValueError(key + " must be exactly zero")
    lora = value.get("lora")
    if (
        not isinstance(lora, Mapping)
        or lora.get("rank") != 4
        or float(lora.get("alpha", 0.0)) != 8.0
        or lora.get("module_count") != 31
        or lora.get("base_parameters_frozen") is not True
        or lora.get("zero_initialized_output_projection") is not True
        or lora.get("export_format") != "merged_legacy_partial_state_dict"
    ):
        raise ValueError("Teacher-v9 LoRA metadata changed")

    components = value.get("component_gradients")
    if not isinstance(components, Mapping) or set(components) != REQUIRED_COMPONENTS:
        raise ValueError("Teacher-v9 component-gradient inventory changed")
    for name, row in components.items():
        if not isinstance(row, Mapping) or row.get("all_finite") is not True:
            raise ValueError(name + ": invalid gradient record")
        _positive(row.get("gradient_l2"), name + " gradient")
        if float(row.get("loss", -1.0)) < 0.0:
            raise ValueError(name + ": negative loss")
        if int(row.get("parameters_with_gradient", 0)) <= 0:
            raise ValueError(name + ": no LoRA parameter received gradient")
    directions = value.get("perturbation_direction_audit")
    if not isinstance(directions, list) or [row.get("amount") for row in directions] != [
        1e-5,
        -1e-5,
    ]:
        raise ValueError("Teacher-v9 perturbation-direction audit changed")
    total = value.get("total_objective")
    if (
        not isinstance(total, Mapping)
        or total.get("weights") != OBJECTIVE_WEIGHTS
    ):
        raise ValueError("Teacher-v9 total objective changed")
    _positive(total.get("loss"), "total objective loss")
    _positive(total.get("gradient_l2"), "total objective gradient")
    if value.get("frozen_parameter_gradient_count") != 0:
        raise ValueError("gradient reached frozen CDM parameters")

    checks = value.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != REQUIRED_CHECKS
        or not all(result is True for result in checks.values())
    ):
        raise ValueError("Teacher-v9 preflight checks are not all PASS")
    if value.get("authorizes_one_scene_overfit_smoke") is not True:
        raise ValueError("Teacher-v9 preflight did not authorize the next smoke gate")
    for key in (
        "authorizes_response3",
        "authorizes_calibration",
        "authorizes_long_training",
        "authorizes_development_evaluation",
        "authorizes_paper_test",
    ):
        if value.get(key) is not False:
            raise ValueError("Teacher-v9 authorization widened: " + key)
    if value.get("failed_checks") != []:
        raise ValueError("Teacher-v9 preflight contains failed checks")
    _finite_tree(value, "Teacher-v9 preflight")
    expected_binding = canonical_sha256(
        {
            "paths": hashes,
            "scene_bindings": scene_bindings,
            "panel": panel,
            "metric_policy_id": value["metric_policy_id"],
            "seed": int(value["seed"]),
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v9 binding-ID arithmetic changed")

    print("[PREFLIGHT_PASS] Teacher-v9 all-sittable LoRA preflight integrity")
    print("[PASS] sealed v5r4 zero-init parity and immutable v9 GT bindings")
    print("[PASS] Bed/normal-Chair/High-Chair equal-macro gradients recomputed")
    print("[PASS] unknown/negative roles and counterexample policy verified")
    print("[PASS] room_0201 and paper-test payloads unread")
    print("[OK] LoRA modules:", lora["module_count"])
    print("[OK] policy ID:", value["metric_policy_id"])
    print("[OK] binding ID:", value["binding_id"])


if __name__ == "__main__":
    main()

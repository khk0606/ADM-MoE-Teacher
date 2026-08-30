#!/usr/bin/env python3
"""Validate the sealed relational Teacher-v7 High-Desk LoRA CUDA preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from relational_teacher_v7_hd_preflight_contract import (
    DATASET_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
    SCHEMA,
    canonical_sha256,
)
from relational_teacher_v6_contract import FORWARD_INPUT_KEYS, PROMPTS, sha256_file


REQUIRED_PATH_KEYS = {
    "dataset_index",
    "stats_file",
    "original_checkpoint",
    "v5_checkpoint",
    "v5_evidence_report",
    "preflight",
    "validator",
    "lora",
    "contract",
    "semantics",
    "preflight_contract",
    "dataset_validator",
}

REQUIRED_CHECK_KEYS = {
    "cuda_used",
    "exact_train_scene_split",
    "sealed_v7_72_motion_144_row_index",
    "development_payloads_unread",
    "train_probes_are_new_high_desk_hc_hd",
    "gate0a_v5_checkpoint_bound",
    "original_checkpoint_bound",
    "zero_init_v5_output_bitwise_equal",
    "only_lora_parameters_trainable",
    "dense_replay_gradient_reaches_lora",
    "semantic_gradient_reaches_lora",
    "invariance_gradient_reaches_lora",
    "perturbed_preservation_gradient_reaches_lora",
    "perturbed_regularizer_gradient_reaches_lora",
    "total_objective_gradient_reaches_lora",
    "frozen_cdm_has_no_gradient",
    "relation_distance_absent_from_forward",
}


def require_finite_positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = json.loads(report_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") != "PASS":
        raise ValueError("Teacher-v7 LoRA preflight schema/status changed")
    if value.get("train_scenes") != list(EXPECTED_TRAIN_SCENES):
        raise ValueError("Teacher-v7 train scenes changed")
    if value.get("development_scenes_metadata_only") != list(
        EXPECTED_DEVELOPMENT_SCENES
    ):
        raise ValueError("Teacher-v7 development scenes changed")
    if value.get("development_payloads_read") is not False:
        raise ValueError("Teacher-v7 preflight read development payloads")
    if value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS):
        raise ValueError("Teacher-v7 forward input keys changed")
    if value.get("prompts") != PROMPTS:
        raise ValueError("Teacher-v7 prompt policy changed")
    if int(value.get("diffusion_steps", -1)) != 500:
        raise ValueError("Teacher-v7 diffusion-step contract changed")

    if value.get("dataset_contract") != {
        "schema": DATASET_SCHEMA,
        "status": "DENSE_DATASET_PASS",
        "dense_motions": 72,
        "prompt_expanded_rows": 144,
        "train_rows": 96,
        "development_rows_metadata_only": 48,
    }:
        raise ValueError("Teacher-v7 dense dataset contract changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError("Teacher-v7 immutable path binding is absent")
    if set(paths) != set(hashes) or set(paths) != REQUIRED_PATH_KEYS:
        raise ValueError("Teacher-v7 path/hash bindings differ")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(hashes[name]):
            raise ValueError("Teacher-v7 bound source changed: " + name)

    probes = value.get("probe_records")
    if not isinstance(probes, list) or len(probes) != 2:
        raise ValueError("Teacher-v7 train probe set changed")
    if sorted(str(row.get("scene_id")) for row in probes) != list(
        EXPECTED_TRAIN_SCENES
    ):
        raise ValueError("Teacher-v7 probe scene IDs changed")
    required_probe_hashes = {
        "points_sha256",
        "sidecar_sha256",
        "dense_index_sha256",
        "affordance_sha256",
    }
    for row in probes:
        if not required_probe_hashes.issubset(row):
            raise ValueError("Teacher-v7 probe hashes are incomplete")
        if any(len(str(row[name])) != 64 for name in required_probe_hashes):
            raise ValueError("Teacher-v7 probe hash length changed")
        if (
            row.get("probe_role") != "new_high_desk_hc_hd_train_motion"
            or "_hc_hd_" not in str(row.get("motion_id", ""))
            or row.get("target_instance_id") != "chair_06"
        ):
            raise ValueError("Teacher-v7 probe is not a new High-Desk train row")

    if float(value.get("zero_init_parity_max_abs", -1.0)) != 0.0:
        raise ValueError("zero-init v7 LoRA does not preserve v5r4 bitwise")
    if float(value.get("zero_init_preservation_loss", -1.0)) != 0.0:
        raise ValueError("zero-init preservation loss must be zero")
    if float(value.get("zero_init_lora_energy", -1.0)) != 0.0:
        raise ValueError("zero-init LoRA energy must be zero")
    lora = value.get("lora")
    if (
        not isinstance(lora, dict)
        or lora.get("base_parameters_frozen") is not True
        or lora.get("zero_initialized_output_projection") is not True
        or lora.get("export_format") != "merged_legacy_partial_state_dict"
        or int(lora.get("module_count", 0)) != 31
    ):
        raise ValueError("Teacher-v7 LoRA metadata changed")

    gradients = value.get("component_gradients")
    required_gradients = {
        "dense_replay",
        "all_sittable_semantic",
        "watch_write_invariance",
        "preservation_perturbed",
        "lora_regularizer_perturbed",
    }
    if not isinstance(gradients, dict) or set(gradients) != required_gradients:
        raise ValueError("Teacher-v7 component-gradient audit changed")
    for name, row in gradients.items():
        if not isinstance(row, dict) or row.get("all_finite") is not True:
            raise ValueError(name + ": non-finite gradient record")
        require_finite_positive(row.get("gradient_l2"), name + " gradient")
        require_finite_positive(row.get("loss"), name + " loss")
        if int(row.get("parameters_with_gradient", 0)) <= 0:
            raise ValueError(name + ": no LoRA parameter received gradient")
    total = value.get("total_objective")
    if not isinstance(total, dict):
        raise ValueError("Teacher-v7 total objective audit is absent")
    require_finite_positive(total.get("loss"), "total loss")
    require_finite_positive(total.get("gradient_l2"), "total gradient")
    if total.get("weights") != {
        "dense_replay": 1.0,
        "all_sittable_semantic": 0.5,
        "watch_write_invariance": 0.25,
        "frozen_v5_preservation": 1.0,
        "lora": 1e-4,
    }:
        raise ValueError("Teacher-v7 preflight objective weights changed")
    if int(value.get("frozen_parameter_gradient_count", -1)) != 0:
        raise ValueError("gradient reached frozen CDM parameters")

    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECK_KEYS or not all(
        result is True for result in checks.values()
    ):
        raise ValueError("Teacher-v7 preflight checks are not all PASS")
    if value.get("authorizes_12_update_calibration") is not True:
        raise ValueError("Teacher-v7 preflight does not authorize calibration")
    for key in (
        "authorizes_long_training",
        "authorizes_development_evaluation",
        "authorizes_paper_test",
    ):
        if value.get(key) is not False:
            raise ValueError("Teacher-v7 authorization guard changed: " + key)
    if len(str(value.get("binding_id", ""))) != 64:
        raise ValueError("Teacher-v7 binding ID is invalid")

    expected_binding = canonical_sha256(
        {
            "paths": hashes,
            "probes": probes,
            "train_scenes": list(EXPECTED_TRAIN_SCENES),
            "development_scenes": list(EXPECTED_DEVELOPMENT_SCENES),
            "seed": int(value["seed"]),
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("Teacher-v7 binding ID arithmetic changed")

    print("[PASS] relational Teacher-v7 High-Desk LoRA CUDA preflight integrity")
    print("[PASS] sealed v5r4 zero-init parity and immutable path bindings")
    print("[PASS] new High-Desk component/total gradients reach LoRA while CDM stays frozen")
    print("[PASS] room_0201 development payloads unread")
    print("[OK] LoRA modules:", lora["module_count"])
    print("[OK] total gradient L2:", total["gradient_l2"])
    print("[OK] binding ID:", value["binding_id"])


if __name__ == "__main__":
    main()

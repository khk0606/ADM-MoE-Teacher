#!/usr/bin/env python3
"""CPU round-trip and tamper test for the Teacher-v7 preflight report."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from relational_teacher_v6_contract import FORWARD_INPUT_KEYS, PROMPTS, sha256_file
from relational_teacher_v7_hd_preflight_contract import (
    DATASET_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
    SCHEMA,
    canonical_sha256,
)
from validate_relational_teacher_v7_hd_lora_preflight import (
    REQUIRED_CHECK_KEYS,
    REQUIRED_PATH_KEYS,
    main as validate_main,
)


def gradient_row() -> dict:
    return {
        "loss": 0.5,
        "gradient_l2": 0.25,
        "lora_A_gradient_l2": 0.1,
        "lora_B_gradient_l2": 0.2,
        "parameters_with_gradient": 31,
        "all_finite": True,
    }


def validate(path: Path) -> None:
    previous = sys.argv
    try:
        sys.argv = ["validate", "--report", str(path)]
        validate_main()
    finally:
        sys.argv = previous


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="teacher_v7_report_test.") as raw:
        root = Path(raw).resolve()
        paths = {}
        hashes = {}
        for number, name in enumerate(sorted(REQUIRED_PATH_KEYS)):
            path = root / (name + ".bin")
            path.write_bytes(f"sealed-{number}-{name}".encode("utf-8"))
            paths[name] = str(path)
            hashes[name] = sha256_file(path)
        probes = [
            {
                "scene_id": scene_id,
                "motion_id": f"{scene_id}_hc_hd_b_1_sit_write",
                "target_instance_id": "chair_06",
                "probe_role": "new_high_desk_hc_hd_train_motion",
                "points_sha256": "1" * 64,
                "sidecar_sha256": "2" * 64,
                "dense_index_sha256": "3" * 64,
                "affordance_sha256": "4" * 64,
            }
            for scene_id in EXPECTED_TRAIN_SCENES
        ]
        seed = 20260908
        value = {
            "schema": SCHEMA,
            "status": "PASS",
            "seed": seed,
            "device": "cuda:0",
            "diffusion_steps": 500,
            "train_scenes": list(EXPECTED_TRAIN_SCENES),
            "development_scenes_metadata_only": list(EXPECTED_DEVELOPMENT_SCENES),
            "development_payloads_read": False,
            "forward_input_keys": sorted(FORWARD_INPUT_KEYS),
            "prompts": PROMPTS,
            "dataset_contract": {
                "schema": DATASET_SCHEMA,
                "status": "DENSE_DATASET_PASS",
                "dense_motions": 72,
                "prompt_expanded_rows": 144,
                "train_rows": 96,
                "development_rows_metadata_only": 48,
            },
            "paths": paths,
            "path_sha256": hashes,
            "probe_records": probes,
            "zero_init_parity_max_abs": 0.0,
            "zero_init_preservation_loss": 0.0,
            "zero_init_lora_energy": 0.0,
            "lora": {
                "base_parameters_frozen": True,
                "zero_initialized_output_projection": True,
                "export_format": "merged_legacy_partial_state_dict",
                "module_count": 31,
            },
            "component_gradients": {
                "dense_replay": gradient_row(),
                "all_sittable_semantic": gradient_row(),
                "watch_write_invariance": gradient_row(),
                "preservation_perturbed": gradient_row(),
                "lora_regularizer_perturbed": gradient_row(),
            },
            "total_objective": {
                "loss": 1.0,
                "gradient_l2": 0.75,
                "weights": {
                    "dense_replay": 1.0,
                    "all_sittable_semantic": 0.5,
                    "watch_write_invariance": 0.25,
                    "frozen_v5_preservation": 1.0,
                    "lora": 1e-4,
                },
            },
            "frozen_parameter_gradient_count": 0,
            "checks": {name: True for name in REQUIRED_CHECK_KEYS},
            "authorizes_12_update_calibration": True,
            "authorizes_long_training": False,
            "authorizes_development_evaluation": False,
            "authorizes_paper_test": False,
        }
        value["binding_id"] = canonical_sha256(
            {
                "paths": hashes,
                "probes": probes,
                "train_scenes": list(EXPECTED_TRAIN_SCENES),
                "development_scenes": list(EXPECTED_DEVELOPMENT_SCENES),
                "seed": seed,
            }
        )
        report = root / "preflight.json"
        report.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        validate(report)

        value["binding_id"] = "0" * 64
        report.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        try:
            validate(report)
        except ValueError:
            pass
        else:
            raise AssertionError("binding-ID tamper was accepted")

    print("[PASS] Teacher-v7 preflight report round trip and tamper guard")


if __name__ == "__main__":
    main()

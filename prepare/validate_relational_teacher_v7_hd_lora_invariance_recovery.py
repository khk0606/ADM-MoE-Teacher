#!/usr/bin/env python3
"""Validate the Teacher-v7 High-Desk invariance recovery."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch

from relational_teacher_v7_hd_preflight_contract import (
    DATASET_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
    SCHEMA as PREFLIGHT_SCHEMA,
)
from relational_teacher_v7_hd_calibration_contract import (
    CANDIDATE_WEIGHT,
    DEFAULT_LR,
    LORA_WEIGHT,
    NEGATIVE_WEIGHT,
    NEW_DENSE_WEIGHT,
    PRESERVATION_WEIGHT,
    SCHEMA as CALIBRATION_SCHEMA,
    STEPS,
    V5_REPLAY_WEIGHT,
    validate_batch_audits,
)
from relational_teacher_v7_hd_recovery_contract import (
    EXPECTED_WEIGHTS,
    INVARIANCE_WEIGHT,
    SCHEMA,
    validate_failed_near_miss,
)
from relational_teacher_v6_contract import read_json, sha256_file


REQUIRED_SOURCE_KEYS = {
    "recovery",
    "validator",
    "preflight_source",
    "preflight_contract_source",
    "calibration_contract_source",
    "recovery_contract_source",
    "lora_source",
    "contract_source",
    "semantics_source",
    "dataset_validator_source",
}

PANEL_KEYS = {
    "new_dense",
    "high_desk_dense",
    "v5_replay_dense",
    "dense_combined",
    "semantic_total",
    "candidate",
    "negative",
    "invariance",
    "preservation",
    "lora_energy",
    "objective_proxy",
}


def torch_load(path: Path):
    return torch.load(path, map_location="cpu")


def finite_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError(label + " is absent")
    for key, raw in value.items():
        if not math.isfinite(float(raw)):
            raise ValueError(f"{label}.{key} is non-finite")
    return value


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-10)


def float32_weighted_sum(
    first: float, terms: Tuple[Tuple[float, float], ...]
) -> float:
    """Reproduce the sequential CUDA float32 objective accumulation."""

    result = torch.tensor(float(first), dtype=torch.float32)
    for weight, raw in terms:
        result = result + float(weight) * torch.tensor(
            float(raw), dtype=torch.float32
        )
    return float(result.item())


def expected_relational_timesteps(step: int) -> list:
    base = [50, 175, 325, 450]
    shift = (step - 1) % 4
    rotated = base[shift:] + base[:shift]
    main = rotated + rotated[1:] + rotated[:1]
    pair_a = 100 + 25 * ((step - 1) % 4)
    pair_b = 325 + 25 * ((step - 1) % 4)
    return main + [pair_a, pair_a, pair_b, pair_b]


def verify_panel_arithmetic(panel: dict, label: str) -> None:
    if set(panel) != PANEL_KEYS:
        raise ValueError(label + " keys changed")
    dense = float32_weighted_sum(
        0.0,
        (
            (NEW_DENSE_WEIGHT, panel["new_dense"]),
            (V5_REPLAY_WEIGHT, panel["v5_replay_dense"]),
        ),
    )
    proxy = float32_weighted_sum(
        panel["dense_combined"],
        (
            (CANDIDATE_WEIGHT, panel["candidate"]),
            (NEGATIVE_WEIGHT, panel["negative"]),
            (INVARIANCE_WEIGHT, panel["invariance"]),
            (PRESERVATION_WEIGHT, panel["preservation"]),
            (LORA_WEIGHT, panel["lora_energy"]),
        ),
    )
    if not close(panel["dense_combined"], dense):
        raise ValueError(label + " dense arithmetic changed")
    semantic = float32_weighted_sum(
        panel["candidate"], ((1.0, panel["negative"]),)
    )
    if not close(panel["semantic_total"], semantic):
        raise ValueError(label + " semantic arithmetic changed")
    if not close(panel["objective_proxy"], proxy):
        raise ValueError(label + " objective arithmetic changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)

    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v7 recovery schema changed")
    if value.get("status") not in {"RECOVERY_PASS", "RECOVERY_FAIL"}:
        raise ValueError("Teacher-v7 recovery status changed")
    if int(value.get("steps", -1)) != STEPS:
        raise ValueError("Teacher-v7 recovery is not exactly 12 updates")
    if float(value.get("learning_rate", float("nan"))) != DEFAULT_LR:
        raise ValueError("Teacher-v7 recovery learning rate changed")
    if value.get("objective_weights") != EXPECTED_WEIGHTS:
        raise ValueError("Teacher-v7 recovery objective weights changed")
    if value.get("train_scenes") != list(EXPECTED_TRAIN_SCENES):
        raise ValueError("Teacher-v7 calibration train scenes changed")
    if value.get("development_scenes_metadata_only") != list(
        EXPECTED_DEVELOPMENT_SCENES
    ):
        raise ValueError("Teacher-v7 calibration development scene changed")
    if value.get("development_payloads_read") is not False:
        raise ValueError("Teacher-v7 calibration read development payloads")
    if value.get("relational_motion_counts") != {
        "room_0101": 24,
        "room_0102": 24,
    }:
        raise ValueError("Teacher-v7 calibration motion counts changed")
    if value.get("high_desk_motion_counts") != {
        "room_0101": 6,
        "room_0102": 6,
    }:
        raise ValueError("Teacher-v7 High-Desk motion counts changed")
    if int(value.get("training_strata_per_scene", -1)) != 4:
        raise ValueError("Teacher-v7 four-way training strata changed")
    if Counter(value.get("v5_replay_target_counts", {})) != Counter(
        {"chair": 18, "bed": 1, "whiteboard": 6}
    ):
        raise ValueError("Teacher-v7 calibration replay inventory changed")

    bound_files = {
        "failed_calibration_summary": "failed_calibration_summary_sha256",
        "preflight_report": "preflight_report_sha256",
        "dataset_index": "dataset_index_sha256",
        "v5_split": "v5_split_sha256",
        "stats_file": "stats_file_sha256",
        "original_checkpoint": "original_checkpoint_sha256",
        "v5_checkpoint": "v5_checkpoint_sha256",
        "lora_state": "lora_state_sha256",
        "merged_checkpoint": "merged_checkpoint_sha256",
    }
    resolved = {}
    for path_key, hash_key in bound_files.items():
        path = Path(str(value.get(path_key, ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(value.get(hash_key, "")):
            raise ValueError("Teacher-v7 recovery artifact changed: " + path_key)
        resolved[path_key] = path

    source_paths = value.get("source_paths")
    source_hashes = value.get("source_sha256")
    if (
        not isinstance(source_paths, dict)
        or not isinstance(source_hashes, dict)
        or set(source_paths) != set(source_hashes)
        or set(source_paths) != REQUIRED_SOURCE_KEYS
    ):
        raise ValueError("Teacher-v7 recovery source bindings are absent")
    for name, raw in source_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(source_hashes[name]):
            raise ValueError("Teacher-v7 recovery source changed: " + name)

    failed = read_json(resolved["failed_calibration_summary"])
    validate_failed_near_miss(failed, CALIBRATION_SCHEMA)
    if int(failed.get("seed", -1)) != int(value.get("seed", -2)):
        raise ValueError("Teacher-v7 recovery seed differs from failed A/B source")
    if value.get("failed_calibration_lora_state_sha256") != failed.get(
        "lora_state_sha256"
    ):
        raise ValueError("Teacher-v7 failed LoRA binding changed")
    if value.get("failed_calibration_merged_checkpoint_sha256") != failed.get(
        "merged_checkpoint_sha256"
    ):
        raise ValueError("Teacher-v7 failed merged binding changed")
    for name in (
        "preflight_report",
        "dataset_index",
        "v5_split",
        "stats_file",
        "original_checkpoint",
        "v5_checkpoint",
    ):
        if value.get(name + "_sha256") != failed.get(name + "_sha256"):
            raise ValueError("Teacher-v7 recovery A/B input changed: " + name)

    logs = value.get("logs")
    audits = value.get("batch_audits")
    if not isinstance(logs, list) or len(logs) != STEPS:
        raise ValueError("Teacher-v7 calibration logs changed")
    if not isinstance(audits, list) or len(audits) != STEPS:
        raise ValueError("Teacher-v7 calibration batch audits changed")
    if [int(row.get("step", -1)) for row in logs] != list(range(1, STEPS + 1)):
        raise ValueError("Teacher-v7 calibration step order changed")

    numeric_keys = {
        "total",
        "dense_combined",
        "new_dense",
        "high_desk_dense",
        "v5_replay_dense",
        "candidate",
        "negative",
        "invariance",
        "preservation",
        "lora_energy",
        "gradient_l2_before_clip",
    }
    for row in logs:
        step = int(row["step"])
        if int(row.get("relational_noise_seed", -1)) != int(value["seed"]) + step * 1009:
            raise ValueError("Teacher-v7 recovery relational noise seed changed")
        if int(row.get("v5_replay_noise_seed", -1)) != int(value["seed"]) + 50000 + step * 1013:
            raise ValueError("Teacher-v7 recovery replay noise seed changed")
        if row.get("relational_timesteps") != expected_relational_timesteps(step):
            raise ValueError("Teacher-v7 recovery relational timesteps changed")
        if row.get("v5_replay_timesteps") != [100, 300, 450]:
            raise ValueError("Teacher-v7 recovery replay timesteps changed")
        for key in numeric_keys:
            if not math.isfinite(float(row.get(key, float("nan")))):
                raise ValueError("Teacher-v7 calibration log is non-finite: " + key)
        if float(row["gradient_l2_before_clip"]) <= 0.0:
            raise ValueError("Teacher-v7 calibration gradient is not positive")
        expected_dense = float32_weighted_sum(
            0.0,
            (
                (NEW_DENSE_WEIGHT, row["new_dense"]),
                (V5_REPLAY_WEIGHT, row["v5_replay_dense"]),
            ),
        )
        if not close(row["dense_combined"], expected_dense):
            raise ValueError("Teacher-v7 calibration log dense arithmetic changed")
        expected_total = float32_weighted_sum(
            row["dense_combined"],
            (
                (CANDIDATE_WEIGHT, row["candidate"]),
                (NEGATIVE_WEIGHT, row["negative"]),
                (INVARIANCE_WEIGHT, row["invariance"]),
                (PRESERVATION_WEIGHT, row["preservation"]),
                (LORA_WEIGHT, row["lora_energy"]),
            ),
        )
        if not close(row["total"], expected_total):
            raise ValueError("Teacher-v7 calibration log objective arithmetic changed")
        if row.get("v5_replay_ids") != audits[step - 1].get(
            "v5_replay_ids"
        ):
            raise ValueError("Teacher-v7 calibration replay audit changed")

    split = read_json(resolved["v5_split"])
    target_by_id = {
        str(row["sample_id"]): str(row["target"])
        for row in split.get("samples", [])
    }
    batch_contract = validate_batch_audits(audits)
    if not all(batch_contract.values()):
        raise ValueError("Teacher-v7 four-stratum/High-Desk batch contract failed")
    if audits != failed.get("batch_audits"):
        raise ValueError("Teacher-v7 recovery batches differ from failed A/B source")
    for step, audit in enumerate(audits, start=1):
        if int(audit.get("step", -1)) != step:
            raise ValueError("Teacher-v7 calibration audit order changed")
        replay_ids = audit.get("v5_replay_ids")
        if Counter(target_by_id.get(str(sample_id)) for sample_id in replay_ids) != Counter(
            {"chair": 1, "bed": 1, "whiteboard": 1}
        ):
            raise ValueError("Teacher-v7 calibration replay is not target-balanced")

    preflight = read_json(resolved["preflight_report"])
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "PASS"
        or preflight.get("authorizes_12_update_calibration") is not True
        or preflight.get("authorizes_long_training") is not False
        or preflight.get("authorizes_development_evaluation") is not False
        or preflight.get("authorizes_paper_test") is not False
        or preflight.get("development_payloads_read") is not False
    ):
        raise ValueError("Teacher-v7 calibration preflight binding changed")
    preflight_hashes = preflight.get("path_sha256")
    if not isinstance(preflight_hashes, dict):
        raise ValueError("Teacher-v7 preflight hashes are absent")
    for name, path_key in (
        ("dataset_index", "dataset_index"),
        ("stats_file", "stats_file"),
        ("original_checkpoint", "original_checkpoint"),
        ("v5_checkpoint", "v5_checkpoint"),
    ):
        if preflight_hashes.get(name) != sha256_file(resolved[path_key]):
            raise ValueError("Teacher-v7 calibration differs from preflight: " + name)

    dataset_index = read_json(resolved["dataset_index"])
    if (
        dataset_index.get("schema") != DATASET_SCHEMA
        or dataset_index.get("status") != "DENSE_DATASET_PASS"
        or dataset_index.get("num_dense_motions") != 72
        or dataset_index.get("num_rows") != 144
        or dataset_index.get("split_row_counts")
        != {"development": 48, "train": 96}
        or dataset_index.get("heldout_dense_arrays_read_during_build") is not False
    ):
        raise ValueError("Teacher-v7 dense dataset binding changed")

    initial = finite_mapping(value.get("initial_fixed_panel"), "initial panel")
    final = finite_mapping(value.get("final_fixed_panel"), "final panel")
    failed_initial = finite_mapping(
        failed.get("initial_fixed_panel"), "failed initial panel"
    )
    ab_panel_keys = PANEL_KEYS - {"objective_proxy"}
    if any(initial[key] != failed_initial[key] for key in ab_panel_keys):
        raise ValueError("Teacher-v7 recovery fresh initialization changed")
    if value.get("single_changed_hyperparameter") != {
        "name": "watch_write_invariance",
        "failed_value": 0.5,
        "recovery_value": INVARIANCE_WEIGHT,
    }:
        raise ValueError("Teacher-v7 recovery changed more than lambda_invariance")
    if value.get("same_seed_batches_noise_timesteps_as_failed_v7") is not True:
        raise ValueError("Teacher-v7 recovery A/B protocol guard changed")
    verify_panel_arithmetic(initial, "initial panel")
    verify_panel_arithmetic(final, "final panel")

    eps = 1e-12
    checks = {
        "exact_12_updates": len(logs) == STEPS,
        **batch_contract,
        "every_update_uses_v5_chair_bed_whiteboard_replay": all(
            len(row["v5_replay_ids"]) == 3 for row in audits
        ),
        "development_payloads_unread": value.get("development_payloads_read") is False,
        "frozen_cdm_has_no_gradient": int(value.get("frozen_parameter_gradient_count", -1)) == 0,
        "all_gradients_finite_and_positive": all(
            float(row["gradient_l2_before_clip"]) > 0.0 for row in logs
        ),
        "fixed_semantic_total_improves": final["semantic_total"] < initial["semantic_total"] - eps,
        "fixed_high_desk_dense_improves": final["high_desk_dense"]
        < initial["high_desk_dense"] - eps,
        "fixed_watch_write_invariance_not_worse": final["invariance"] <= initial["invariance"] * 1.02 + eps,
        "fixed_new_dense_not_worse": final["new_dense"] <= initial["new_dense"] * 1.02 + eps,
        "fixed_negative_suppression_not_worse": final["negative"] <= initial["negative"] * 1.02 + eps,
        "fixed_v5_replay_retained": final["v5_replay_dense"] <= initial["v5_replay_dense"] * 1.01 + eps,
        "fixed_preservation_bounded": final["preservation"] <= 1e-3,
        "lora_changed_from_zero": final["lora_energy"] > 0.0,
    }
    if value.get("checks") != checks:
        raise ValueError("Teacher-v7 recovery checks changed")
    failed_checks = [name for name, passed in checks.items() if not passed]
    if value.get("failed_checks") != failed_checks:
        raise ValueError("Teacher-v7 recovery failed-check list changed")
    expected_status = "RECOVERY_PASS" if not failed_checks else "RECOVERY_FAIL"
    if value.get("status") != expected_status:
        raise ValueError("Teacher-v7 recovery status/checks disagree")

    metadata = value.get("lora_metadata")
    if (
        not isinstance(metadata, dict)
        or int(metadata.get("module_count", 0)) != 31
        or int(metadata.get("rank", 0)) != 4
        or float(metadata.get("alpha", 0.0)) != 8.0
        or float(metadata.get("scale", 0.0)) != 2.0
        or metadata.get("base_parameters_frozen") is not True
        or metadata.get("zero_initialized_output_projection") is not True
        or metadata.get("export_format") != "merged_legacy_partial_state_dict"
        or not isinstance(metadata.get("module_names"), list)
        or len(metadata["module_names"]) != 31
    ):
        raise ValueError("Teacher-v7 recovery LoRA metadata changed")

    lora_state = torch_load(resolved["lora_state"])
    if (
        not isinstance(lora_state, dict)
        or lora_state.get("schema") != SCHEMA + "_diagnostic_lora_state"
        or int(lora_state.get("step", -1)) != STEPS
        or int(lora_state.get("seed", -1)) != int(value.get("seed", -2))
        or not isinstance(lora_state.get("lora"), dict)
        or not lora_state["lora"]
    ):
        raise ValueError("Teacher-v7 recovery LoRA checkpoint changed")
    if not all(
        isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
        for tensor in lora_state["lora"].values()
    ):
        raise ValueError("Teacher-v7 recovery LoRA checkpoint is non-finite")
    expected_lora_keys = {
        f"{name}.{suffix}"
        for name in metadata["module_names"]
        for suffix in ("lora_A", "lora_B")
    }
    if set(lora_state["lora"]) != expected_lora_keys:
        raise ValueError("Teacher-v7 recovery LoRA parameter inventory changed")
    merged = torch_load(resolved["merged_checkpoint"])
    if not isinstance(merged, dict) or not merged or not all(
        isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
        for tensor in merged.values()
    ):
        raise ValueError("Teacher-v7 recovery merged checkpoint changed")
    if any("lora_" in str(name) for name in merged):
        raise ValueError("Teacher-v7 merged checkpoint leaked LoRA wrapper keys")

    if value.get("fresh_zero_init_from_sealed_v5r4") is not True:
        raise ValueError("Teacher-v7 calibration did not start from fresh v5r4")
    if value.get("prior_v6_lora_checkpoint_loaded") is not False:
        raise ValueError("Teacher-v7 recovery improperly loaded a v6 LoRA")
    if value.get("prior_failed_v7_checkpoint_loaded") is not False:
        raise ValueError("Teacher-v7 recovery loaded the failed v7 checkpoint")

    if value.get("diagnostic_checkpoint_only") is not True:
        raise ValueError("Teacher-v7 recovery checkpoint overstates authority")
    if value.get("authorizes_bounded_train_only_pilot") != (
        expected_status == "RECOVERY_PASS"
    ):
        raise ValueError("Teacher-v7 recovery pilot authority changed")
    for key in (
        "authorizes_long_training",
        "authorizes_development_evaluation",
        "authorizes_paper_test",
    ):
        if value.get(key) is not False:
            raise ValueError("Teacher-v7 recovery authority changed: " + key)

    print(f"[{expected_status}] relational Teacher-v7 High-Desk invariance recovery integrity")
    print("[PASS] immutable failed near miss and fresh v5r4 initialization verified")
    print("[PASS] identical batches/noise/timesteps; only lambda_invariance=0.75")
    print("[PASS] checkpoint hashes, objective arithmetic and leakage guards")
    print("[OK] initial fixed panel:", initial)
    print("[OK] final fixed panel:", final)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

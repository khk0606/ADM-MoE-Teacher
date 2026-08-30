#!/usr/bin/env python3
"""Validate the fresh bounded Teacher-v7 High-Desk LoRA pilot."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch

from relational_teacher_v6_contract import read_json, sha256_file
from relational_teacher_v7_hd_calibration_contract import (
    CANDIDATE_WEIGHT,
    DEFAULT_LR,
    LORA_WEIGHT,
    NEGATIVE_WEIGHT,
    NEW_DENSE_WEIGHT,
    PRESERVATION_WEIGHT,
    TRAINING_STRATA_PER_SCENE,
    V5_REPLAY_WEIGHT,
    validate_batch_audits,
)
from relational_teacher_v7_hd_pilot_contract import (
    CHECKPOINT_STEPS,
    SCHEMA,
    STEPS,
    checkpoint_eligibility,
    rank_shortlist,
)
from relational_teacher_v7_hd_preflight_contract import (
    DATASET_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
)
from relational_teacher_v7_hd_recovery_contract import (
    EXPECTED_WEIGHTS,
    INVARIANCE_WEIGHT,
    SCHEMA as RECOVERY_SCHEMA,
)


PANEL_KEYS = {
    "new_dense", "high_desk_dense", "v5_replay_dense", "dense_combined",
    "semantic_total", "candidate", "negative", "invariance", "preservation",
    "lora_energy", "objective_proxy",
}
REQUIRED_SOURCE_KEYS = {
    "pilot", "validator", "recovery_source", "recovery_validator_source",
    "pilot_contract_source", "recovery_contract_source",
    "calibration_contract_source", "preflight_contract_source", "lora_source",
    "contract_source", "semantics_source", "dataset_validator_source",
}


def torch_load(path: Path):
    return torch.load(path, map_location="cpu")


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-10)


def float32_weighted_sum(first: float, terms: Tuple[Tuple[float, float], ...]) -> float:
    result = torch.tensor(float(first), dtype=torch.float32)
    for weight, raw in terms:
        result = result + float(weight) * torch.tensor(float(raw), dtype=torch.float32)
    return float(result.item())


def expected_relational_timesteps(step: int) -> list:
    cycle_step = ((step - 1) % 12) + 1
    base = [50, 175, 325, 450]
    shift = (cycle_step - 1) % 4
    rotated = base[shift:] + base[:shift]
    main = rotated + rotated[1:] + rotated[:1]
    pair_a = 100 + 25 * ((cycle_step - 1) % 4)
    pair_b = 325 + 25 * ((cycle_step - 1) % 4)
    return main + [pair_a, pair_a, pair_b, pair_b]


def verify_panel(panel: object, label: str) -> dict:
    if not isinstance(panel, dict) or set(panel) != PANEL_KEYS:
        raise ValueError(label + " keys changed")
    if not all(math.isfinite(float(raw)) for raw in panel.values()):
        raise ValueError(label + " contains non-finite values")
    dense = float32_weighted_sum(
        0.0,
        ((NEW_DENSE_WEIGHT, panel["new_dense"]),
         (V5_REPLAY_WEIGHT, panel["v5_replay_dense"])),
    )
    semantic = float32_weighted_sum(panel["candidate"], ((1.0, panel["negative"]),))
    proxy = float32_weighted_sum(
        panel["dense_combined"],
        ((CANDIDATE_WEIGHT, panel["candidate"]),
         (NEGATIVE_WEIGHT, panel["negative"]),
         (INVARIANCE_WEIGHT, panel["invariance"]),
         (PRESERVATION_WEIGHT, panel["preservation"]),
         (LORA_WEIGHT, panel["lora_energy"])),
    )
    if not close(panel["dense_combined"], dense):
        raise ValueError(label + " dense arithmetic changed")
    if not close(panel["semantic_total"], semantic):
        raise ValueError(label + " semantic arithmetic changed")
    if not close(panel["objective_proxy"], proxy):
        raise ValueError(label + " objective arithmetic changed")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())

    if value.get("schema") != SCHEMA:
        raise ValueError("Teacher-v7 pilot schema changed")
    if value.get("status") not in {"PILOT_PASS", "PILOT_FAIL"}:
        raise ValueError("Teacher-v7 pilot status changed")
    if int(value.get("steps", -1)) != STEPS:
        raise ValueError("Teacher-v7 pilot is not exactly 60 updates")
    if value.get("checkpoint_steps") != list(CHECKPOINT_STEPS):
        raise ValueError("Teacher-v7 pilot checkpoint schedule changed")
    if float(value.get("learning_rate", float("nan"))) != DEFAULT_LR:
        raise ValueError("Teacher-v7 pilot learning rate changed")
    if value.get("objective_weights") != EXPECTED_WEIGHTS:
        raise ValueError("Teacher-v7 pilot objective weights changed")
    if value.get("train_scenes") != list(EXPECTED_TRAIN_SCENES):
        raise ValueError("Teacher-v7 pilot train scenes changed")
    if value.get("development_scenes_metadata_only") != list(EXPECTED_DEVELOPMENT_SCENES):
        raise ValueError("Teacher-v7 pilot development scene changed")
    if value.get("development_payloads_read") is not False:
        raise ValueError("Teacher-v7 pilot read development payloads")
    if value.get("relational_motion_counts") != {"room_0101": 24, "room_0102": 24}:
        raise ValueError("Teacher-v7 pilot motion inventory changed")
    if value.get("high_desk_motion_counts") != {"room_0101": 6, "room_0102": 6}:
        raise ValueError("Teacher-v7 pilot High-Desk inventory changed")
    if int(value.get("training_strata_per_scene", -1)) != TRAINING_STRATA_PER_SCENE:
        raise ValueError("Teacher-v7 pilot stratum count changed")
    if Counter(value.get("v5_replay_target_counts", {})) != Counter(
        {"chair": 18, "bed": 1, "whiteboard": 6}
    ):
        raise ValueError("Teacher-v7 pilot v5 replay inventory changed")

    bound = {
        "recovery_summary": "recovery_summary_sha256",
        "dataset_index": "dataset_index_sha256",
        "v5_split": "v5_split_sha256",
        "stats_file": "stats_file_sha256",
        "original_checkpoint": "original_checkpoint_sha256",
        "v5_checkpoint": "v5_checkpoint_sha256",
    }
    resolved = {}
    for path_key, hash_key in bound.items():
        path = Path(str(value.get(path_key, ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(value.get(hash_key, "")):
            raise ValueError("Teacher-v7 pilot bound file changed: " + path_key)
        resolved[path_key] = path

    source_paths = value.get("source_paths")
    source_hashes = value.get("source_sha256")
    if (not isinstance(source_paths, dict) or not isinstance(source_hashes, dict)
            or set(source_paths) != set(source_hashes)
            or set(source_paths) != REQUIRED_SOURCE_KEYS):
        raise ValueError("Teacher-v7 pilot source bindings are absent")
    for name, raw in source_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(source_hashes[name]):
            raise ValueError("Teacher-v7 pilot source changed: " + name)

    recovery = read_json(resolved["recovery_summary"])
    if (recovery.get("schema") != RECOVERY_SCHEMA
            or recovery.get("status") != "RECOVERY_PASS"
            or recovery.get("failed_checks") != []
            or int(recovery.get("steps", -1)) != 12
            or int(recovery.get("seed", -1)) != int(value.get("seed", -2))
            or recovery.get("objective_weights") != EXPECTED_WEIGHTS
            or recovery.get("authorizes_bounded_train_only_pilot") is not True
            or recovery.get("development_payloads_read") is not False
            or recovery.get("prior_failed_v7_checkpoint_loaded") is not False):
        raise ValueError("Teacher-v7 recovery authorization changed")
    for key in ("dataset_index", "v5_split", "stats_file", "original_checkpoint", "v5_checkpoint"):
        if value.get(key + "_sha256") != recovery.get(key + "_sha256"):
            raise ValueError("Teacher-v7 pilot input differs from recovery: " + key)

    dataset = read_json(resolved["dataset_index"])
    if (dataset.get("schema") != DATASET_SCHEMA
            or dataset.get("status") != "DENSE_DATASET_PASS"
            or dataset.get("num_dense_motions") != 72
            or dataset.get("num_rows") != 144
            or dataset.get("split_row_counts") != {"development": 48, "train": 96}
            or dataset.get("heldout_dense_arrays_read_during_build") is not False
            or dataset.get("relation_or_distance_used_as_forward_input") is not False):
        raise ValueError("Teacher-v7 pilot dataset binding changed")

    logs = value.get("logs")
    audits = value.get("batch_audits")
    records = value.get("checkpoint_records")
    if not isinstance(logs, list) or len(logs) != STEPS:
        raise ValueError("Teacher-v7 pilot logs changed")
    if not isinstance(audits, list) or len(audits) != STEPS:
        raise ValueError("Teacher-v7 pilot batch audits changed")
    if not isinstance(records, list) or len(records) != len(CHECKPOINT_STEPS):
        raise ValueError("Teacher-v7 pilot checkpoint records changed")
    if [int(row.get("step", -1)) for row in logs] != list(range(1, STEPS + 1)):
        raise ValueError("Teacher-v7 pilot step order changed")
    if [int(row.get("step", -1)) for row in records] != list(CHECKPOINT_STEPS):
        raise ValueError("Teacher-v7 pilot checkpoint order changed")
    if logs[:12] != recovery.get("logs") or audits[:12] != recovery.get("batch_audits"):
        raise ValueError("Teacher-v7 pilot first 12 updates differ from recovery")

    numeric_keys = {
        "total", "dense_combined", "new_dense", "high_desk_dense",
        "v5_replay_dense", "candidate", "negative", "invariance",
        "preservation", "lora_energy", "gradient_l2_before_clip",
    }
    for row in logs:
        step = int(row["step"])
        if int(row.get("relational_noise_seed", -1)) != int(value["seed"]) + step * 1009:
            raise ValueError("Teacher-v7 pilot relational noise seed changed")
        if int(row.get("v5_replay_noise_seed", -1)) != int(value["seed"]) + 50000 + step * 1013:
            raise ValueError("Teacher-v7 pilot replay noise seed changed")
        if row.get("relational_timesteps") != expected_relational_timesteps(step):
            raise ValueError("Teacher-v7 pilot relational timesteps changed")
        if row.get("v5_replay_timesteps") != [100, 300, 450]:
            raise ValueError("Teacher-v7 pilot replay timesteps changed")
        for key in numeric_keys:
            if not math.isfinite(float(row.get(key, float("nan")))):
                raise ValueError("Teacher-v7 pilot log is non-finite: " + key)
        if float(row["gradient_l2_before_clip"]) <= 0.0:
            raise ValueError("Teacher-v7 pilot gradient is not positive")
        dense = float32_weighted_sum(
            0.0,
            ((NEW_DENSE_WEIGHT, row["new_dense"]),
             (V5_REPLAY_WEIGHT, row["v5_replay_dense"])),
        )
        total = float32_weighted_sum(
            row["dense_combined"],
            ((CANDIDATE_WEIGHT, row["candidate"]),
             (NEGATIVE_WEIGHT, row["negative"]),
             (INVARIANCE_WEIGHT, row["invariance"]),
             (PRESERVATION_WEIGHT, row["preservation"]),
             (LORA_WEIGHT, row["lora_energy"])),
        )
        if not close(row["dense_combined"], dense):
            raise ValueError("Teacher-v7 pilot log dense arithmetic changed")
        if not close(row["total"], total):
            raise ValueError("Teacher-v7 pilot log objective arithmetic changed")
        if row.get("v5_replay_ids") != audits[step - 1].get("v5_replay_ids"):
            raise ValueError("Teacher-v7 pilot replay audit changed")

    split = read_json(resolved["v5_split"])
    target_by_id = {str(row["sample_id"]): str(row["target"])
                    for row in split.get("samples", [])}
    batch_contract = validate_batch_audits(audits)
    if not all(batch_contract.values()):
        raise ValueError("Teacher-v7 pilot four-stratum batch contract failed")
    for step, audit in enumerate(audits, start=1):
        if int(audit.get("step", -1)) != step:
            raise ValueError("Teacher-v7 pilot audit order changed")
        replay_ids = audit.get("v5_replay_ids")
        if Counter(target_by_id.get(str(sample_id)) for sample_id in replay_ids) != Counter(
            {"chair": 1, "bed": 1, "whiteboard": 1}
        ):
            raise ValueError("Teacher-v7 pilot replay is not target-balanced")

    initial = verify_panel(value.get("initial_fixed_panel"), "initial panel")
    if initial != recovery.get("initial_fixed_panel"):
        raise ValueError("Teacher-v7 pilot initialization differs from recovery")
    metadata = value.get("lora_metadata")
    if (not isinstance(metadata, dict)
            or int(metadata.get("module_count", 0)) != 31
            or int(metadata.get("rank", 0)) != 4
            or float(metadata.get("alpha", 0.0)) != 8.0
            or float(metadata.get("scale", 0.0)) != 2.0
            or metadata.get("base_parameters_frozen") is not True
            or metadata.get("zero_initialized_output_projection") is not True
            or metadata.get("export_format") != "merged_legacy_partial_state_dict"
            or not isinstance(metadata.get("module_names"), list)
            or len(metadata["module_names"]) != 31):
        raise ValueError("Teacher-v7 pilot LoRA metadata changed")
    expected_lora_keys = {f"{name}.{suffix}" for name in metadata["module_names"]
                          for suffix in ("lora_A", "lora_B")}

    for record in records:
        step = int(record["step"])
        panel = verify_panel(record.get("fixed_panel"), f"step {step} panel")
        eligibility = checkpoint_eligibility(panel, initial)
        if record.get("eligibility_checks") != eligibility:
            raise ValueError(f"Teacher-v7 step {step} eligibility changed")
        if record.get("eligible") is not all(eligibility.values()):
            raise ValueError(f"Teacher-v7 step {step} eligibility status changed")
        lora_path = Path(str(record.get("lora_state", ""))).expanduser().resolve()
        merged_path = Path(str(record.get("merged_checkpoint", ""))).expanduser().resolve()
        if not lora_path.is_file() or sha256_file(lora_path) != str(record.get("lora_state_sha256", "")):
            raise ValueError(f"Teacher-v7 step {step} LoRA checkpoint changed")
        if not merged_path.is_file() or sha256_file(merged_path) != str(record.get("merged_checkpoint_sha256", "")):
            raise ValueError(f"Teacher-v7 step {step} merged checkpoint changed")
        lora = torch_load(lora_path)
        if (not isinstance(lora, dict)
                or lora.get("schema") != SCHEMA + "_diagnostic_lora_state"
                or int(lora.get("step", -1)) != step
                or int(lora.get("seed", -1)) != int(value.get("seed", -2))
                or not isinstance(lora.get("lora"), dict)
                or set(lora["lora"]) != expected_lora_keys
                or not all(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
                           for tensor in lora["lora"].values())):
            raise ValueError(f"Teacher-v7 step {step} LoRA state changed")
        merged = torch_load(merged_path)
        if (not isinstance(merged, dict) or not merged
                or any("lora_" in str(name) for name in merged)
                or not all(isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all()
                           for tensor in merged.values())):
            raise ValueError(f"Teacher-v7 step {step} merged state changed")

    if records[0]["fixed_panel"] != recovery.get("final_fixed_panel"):
        raise ValueError("Teacher-v7 step 12 does not reproduce recovery")
    shortlisted = list(rank_shortlist(records))
    if value.get("shortlisted_steps") != shortlisted:
        raise ValueError("Teacher-v7 pilot shortlist changed")
    if value.get("selection_rank") != [
        "objective_proxy", "high_desk_dense", "semantic_total", "step"
    ]:
        raise ValueError("Teacher-v7 pilot selection rank changed")

    checks = {
        "exact_60_updates": len(logs) == STEPS,
        "five_checkpoint_steps_present": [int(row["step"]) for row in records]
        == list(CHECKPOINT_STEPS),
        "first12_exactly_reproduce_recovery": (
            logs[:12] == recovery.get("logs")
            and audits[:12] == recovery.get("batch_audits")
            and records[0]["fixed_panel"] == recovery.get("final_fixed_panel")
        ),
        **batch_contract,
        "every_update_uses_v5_chair_bed_whiteboard_replay": all(
            len(row["v5_replay_ids"]) == 3 for row in audits
        ),
        "development_payloads_unread": value.get("development_payloads_read") is False,
        "frozen_cdm_has_no_gradient": int(value.get("frozen_parameter_gradient_count", -1)) == 0,
        "all_gradients_finite_and_positive": all(
            float(row["gradient_l2_before_clip"]) > 0.0 for row in logs
        ),
        "at_least_one_checkpoint_is_admissible": len(shortlisted) >= 1,
        "shortlist_size_at_most_two": 1 <= len(shortlisted) <= 2,
    }
    if value.get("checks") != checks:
        raise ValueError("Teacher-v7 pilot checks changed")
    failed_checks = [name for name, passed in checks.items() if not passed]
    if value.get("failed_checks") != failed_checks:
        raise ValueError("Teacher-v7 pilot failed-check list changed")
    expected_status = "PILOT_PASS" if not failed_checks else "PILOT_FAIL"
    if value.get("status") != expected_status:
        raise ValueError("Teacher-v7 pilot status/checks disagree")

    if value.get("fresh_zero_init_from_sealed_v5r4") is not True:
        raise ValueError("Teacher-v7 pilot initialization claim changed")
    if value.get("prior_v6_lora_checkpoint_loaded") is not False:
        raise ValueError("Teacher-v7 pilot loaded a v6 LoRA checkpoint")
    if value.get("prior_failed_v7_checkpoint_loaded") is not False:
        raise ValueError("Teacher-v7 pilot loaded failed calibration weights")
    if value.get("recovery_checkpoint_loaded") is not False:
        raise ValueError("Teacher-v7 pilot loaded recovery weights")
    if value.get("diagnostic_checkpoints_only") is not True:
        raise ValueError("Teacher-v7 pilot checkpoints overstate authority")
    if value.get("authorizes_train_only_reverse_diffusion_shortlist") is not (
        expected_status == "PILOT_PASS"
    ):
        raise ValueError("Teacher-v7 pilot shortlist authority changed")
    for key in ("authorizes_long_training", "authorizes_development_evaluation", "authorizes_paper_test"):
        if value.get(key) is not False:
            raise ValueError("Teacher-v7 pilot authority changed: " + key)

    print(f"[{expected_status}] relational Teacher-v7 High-Desk bounded pilot integrity")
    print("[PASS] fresh 60 updates and exact recovery reproduction verified")
    print("[PASS] five High-Desk/invariance/v5 checkpoint panels recomputed")
    print("[PASS] checkpoint hashes, finite tensors and held-out-unread guards")
    print("[OK] shortlisted steps:", shortlisted)
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

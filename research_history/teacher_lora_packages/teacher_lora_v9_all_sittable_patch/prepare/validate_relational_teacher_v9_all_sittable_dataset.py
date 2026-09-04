#!/usr/bin/env python3
"""Deep validator for Teacher-v9 all-sittable artifacts.

The validator distrusts the generated manifest: it reloads all 48 sealed v7
train dense maps, reapplies target/environment masking, recomputes every train
consensus and replica, and compares the saved arrays bit for bit.  The locked
room_0201 record is checked from metadata and hashes only; none of its array
payloads are opened at this gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from relational_teacher_v9_all_sittable_contract import (
    ACTIVE_THRESHOLD,
    CONTACT_DIM,
    GROUP_ORDER,
    EXPECTED_SCENE_TARGETS,
    OUTPUT_DATASET_SCHEMA,
    POINT_COUNT,
    PROMPTS,
    REPLICA_COUNT,
    SCENE_MANIFEST_SCHEMA,
    SOURCE_DATASET_SCHEMA,
    TEACHER_FORWARD_INPUTS,
    aggregate_all_sittable,
    assert_all_sittable_contract,
    load_source_scene,
    mask_maps_to_target_and_environment,
    read_json,
    resolve_under,
    scene_metrics,
    sha256_file,
    validate_source_metadata_without_arrays,
)


EXPECTED_SOURCE_SCENES = {
    "room_0101": "train",
    "room_0102": "train",
    "room_0201": "development",
}
EXPECTED_OUTPUT_SCENES = {
    "room_0101": "train",
    "room_0102": "train",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    return parser.parse_args()


def load_only_affordance(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {"affordance"}:
            raise ValueError(f"{path}: expected only an affordance array")
        value = source["affordance"].astype(np.float32)
    return value


def exact_array(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError(
            f"{label}: dtype/shape changed: {actual.dtype}/{actual.shape} "
            f"!= {expected.dtype}/{expected.shape}"
        )
    if not np.array_equal(actual, expected):
        maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        raise ValueError(f"{label}: array differs (max abs {maximum})")


def verify_consensus_bundle(
    path: Path,
    aggregate: Mapping[str, object],
    loaded: Mapping[str, object],
) -> None:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "xyz",
            "source_indices",
            "instance_ids",
            "category_ids",
            "eligible_instance_ids",
            "eligible_numeric_instance_ids",
            "verified_object_mask",
            "verified_positive_mask",
            "instance_affordance",
            "instance_active_mask",
            "all_sittable_affordance",
            "unknown_sittable_mask",
            "environment_aux_mask",
            "explicit_negative_mask",
            "winner_instance_slot",
        }
        if set(source.files) != required:
            raise ValueError(f"{path}: consensus bundle keys changed")
        arrays = {key: source[key] for key in source.files}

    targets = sorted(aggregate["instance_consensus"])
    instance_stack = np.stack(
        [aggregate["instance_consensus"][target] for target in targets], axis=0
    ).astype(np.float32)
    numeric = np.asarray(
        [loaded["stable_instances"][target][0] for target in targets], dtype=np.int64
    )
    unknown_numeric = np.asarray(
        [
            loaded["stable_instances"][target][0]
            for target in loaded["partition"]["unknown_sittable_not_negative"]
        ],
        dtype=np.int64,
    )
    expected = {
        "xyz": loaded["xyz"],
        "source_indices": loaded["source_indices"],
        "instance_ids": loaded["instance_ids"],
        "category_ids": loaded["category_ids"],
        "eligible_instance_ids": np.asarray(targets),
        "eligible_numeric_instance_ids": numeric,
        "verified_object_mask": np.stack(
            [loaded["instance_ids"] == value for value in numeric], axis=0
        ),
        "verified_positive_mask": np.isin(loaded["instance_ids"], numeric),
        "instance_affordance": instance_stack,
        "instance_active_mask": instance_stack >= ACTIVE_THRESHOLD,
        "all_sittable_affordance": aggregate["all_consensus"],
        "unknown_sittable_mask": np.isin(loaded["instance_ids"], unknown_numeric),
        "environment_aux_mask": loaded["instance_ids"] == 0,
        "explicit_negative_mask": np.isin(loaded["category_ids"], [3, 4, 5]),
    }
    winner = np.argmax(instance_stack, axis=0).astype(np.int8)
    winner[np.max(instance_stack, axis=0) <= 0.0] = -1
    expected["winner_instance_slot"] = winner
    for key, value in expected.items():
        exact_array(path.name + "/" + key, arrays[key], np.asarray(value))
    if np.any(arrays["unknown_sittable_mask"] & arrays["explicit_negative_mask"]):
        raise ValueError("unknown Sit instances overlap explicit negatives")
    if np.any(
        arrays["unknown_sittable_mask"]
        & arrays["verified_object_mask"].any(axis=0)
    ):
        raise ValueError("unknown Sit instances overlap verified positives")


def verify_scene(
    source_root: Path,
    output_root: Path,
    source_record: Mapping[str, object],
    output_record: Mapping[str, object],
) -> None:
    loaded = load_source_scene(source_root, source_record)
    scene_id = str(loaded["scene_id"])
    expected_group_targets = EXPECTED_SCENE_TARGETS[scene_id]
    expected_verified_targets = sorted(set(expected_group_targets.values()))
    source_keys = (
        "scene_id",
        "split",
        "points_file",
        "points_sha256",
        "sidecar_file",
        "sidecar_sha256",
        "instances_file",
        "instances_sha256",
        "motion_index_file",
        "motion_index_sha256",
        "dense_index_file",
        "dense_index_sha256",
    )
    if output_record.get("source_scene_record") != {
        key: source_record[key] for key in source_keys
    }:
        raise ValueError(f"{scene_id}: top-level source scene binding changed")
    if (
        output_record.get("replica_count") != REPLICA_COUNT
        or output_record.get("row_count") != 2
        or output_record.get("verified_target_instances")
        != expected_verified_targets
        or output_record.get("unknown_sittable_instances_are_negative") is not False
    ):
        raise ValueError(f"{scene_id}: top-level scene counts/roles changed")
    manifest_path = resolve_under(output_root, output_record["manifest_file"])
    if sha256_file(manifest_path) != str(output_record["manifest_sha256"]):
        raise ValueError(f"{scene_id}: output manifest hash changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != SCENE_MANIFEST_SCHEMA
        or manifest.get("status") != "ALL_SITTABLE_SCENE_PASS"
        or manifest.get("scene_id") != scene_id
        or manifest.get("split") != loaded["split"]
        or manifest.get("verified_target_instances")
        != expected_verified_targets
        or manifest.get("relation_or_distance_used") is not False
        or manifest.get("teacher_forward_inputs")
        != list(TEACHER_FORWARD_INPUTS)
        or manifest.get("unknown_sittable_instances_are_negative") is not False
        or manifest.get("primary_loss_policy")
        != "equal_instance_macro_on_verified_object_masks"
        or manifest.get("environment_contact_is_auxiliary_only") is not True
        or manifest.get("development_used_for_parameter_selection") is not False
        or manifest.get("failed_checks")
    ):
        raise ValueError(f"{scene_id}: scene manifest contract changed")
    expected_aggregation = {
        "motion_group_consensus": "nearest_rank_q75_equals_second_largest_of_six",
        "same_instance_multi_group_union": "exact_pointwise_max",
        "scene_instance_union": "exact_pointwise_max",
        "replica_pairing": "six_row_latin_shift_one_source_per_stratum",
        "source_object_filter": "bound_target_instance_plus_environment_only",
        "active_threshold": ACTIVE_THRESHOLD,
    }
    if manifest.get("aggregation") != expected_aggregation:
        raise ValueError(f"{scene_id}: aggregation policy changed")
    if (
        loaded["targets"] != expected_group_targets
        or manifest.get("group_targets") != expected_group_targets
    ):
        raise ValueError(f"{scene_id}: group target binding changed")
    if manifest.get("instance_partition") != loaded["partition"]:
        raise ValueError(f"{scene_id}: positive/unknown partition changed")
    if manifest.get("source_dense_bindings") != loaded["source_bindings"]:
        raise ValueError(f"{scene_id}: source dense binding changed")
    if (
        manifest.get("source_files") != loaded["source_files"]
        or manifest.get("source_hashes") != loaded["source_hashes"]
    ):
        raise ValueError(f"{scene_id}: source scene file binding changed")

    filtered = mask_maps_to_target_and_environment(
        loaded["grouped_rows"],
        loaded["targets"],
        loaded["maps_by_motion"],
        loaded["stable_instances"],
        loaded["instance_ids"],
    )
    aggregate = aggregate_all_sittable(
        loaded["grouped_rows"], loaded["targets"], filtered
    )
    assert_all_sittable_contract(aggregate, loaded["targets"])
    metrics = scene_metrics(
        aggregate,
        loaded["targets"],
        loaded["stable_instances"],
        loaded["instance_ids"],
    )
    if manifest.get("metrics") != metrics:
        raise ValueError(f"{scene_id}: scene metrics changed")

    for group in GROUP_ORDER:
        row = manifest["group_consensus"][group]
        if row.get("target_instance_id") != expected_group_targets[group]:
            raise ValueError(f"{scene_id}/{group}: group target metadata changed")
        path = resolve_under(output_root, row["file"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{scene_id}/{group}: group file hash changed")
        exact_array(
            f"{scene_id}/{group}",
            load_only_affordance(path),
            aggregate["group_consensus"][group],
        )
    for target, expected in aggregate["instance_consensus"].items():
        row = manifest["instance_consensus"][target]
        path = resolve_under(output_root, row["file"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{scene_id}/{target}: instance file hash changed")
        exact_array(f"{scene_id}/{target}", load_only_affordance(path), expected)
    if set(manifest.get("instance_consensus", {})) != set(
        aggregate["instance_consensus"]
    ):
        raise ValueError(f"{scene_id}: instance consensus inventory changed")

    consensus_path = resolve_under(output_root, manifest["all_sittable_consensus_file"])
    if sha256_file(consensus_path) != manifest["all_sittable_consensus_sha256"]:
        raise ValueError(f"{scene_id}: consensus bundle hash changed")
    verify_consensus_bundle(consensus_path, aggregate, loaded)

    replicas = manifest.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != REPLICA_COUNT:
        raise ValueError(f"{scene_id}: replica inventory changed")
    replica_files = {}
    for number, row in enumerate(replicas):
        expected_id = f"replica_{number:02d}"
        if row.get("replica_id") != expected_id:
            raise ValueError(f"{scene_id}: replica order changed")
        if row.get("source_motion_ids") != aggregate["replica_motion_ids"][number]:
            raise ValueError(f"{scene_id}/{expected_id}: source pairing changed")
        path = resolve_under(output_root, row["file"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"{scene_id}/{expected_id}: replica hash changed")
        exact_array(
            f"{scene_id}/{expected_id}",
            load_only_affordance(path),
            aggregate["replicas"][number],
        )
        replica_files[expected_id] = (row["file"], row["sha256"])

    prompt_rows = output_record.get("rows")
    if not isinstance(prompt_rows, list) or len(prompt_rows) != 2:
        raise ValueError(f"{scene_id}: prompt rows changed")
    consensus_file = manifest["all_sittable_consensus_file"]
    consensus_hash = manifest["all_sittable_consensus_sha256"]
    seen_prompts = []
    for row in prompt_rows:
        if row.get("scene_id") != scene_id:
            raise ValueError(f"{scene_id}: prompt row scene changed")
        prompt_id = str(row.get("prompt_id"))
        if row.get("row_id") != f"{scene_id}_{prompt_id}":
            raise ValueError(f"{scene_id}: prompt row ID changed")
        if PROMPTS.get(prompt_id) != row.get("text"):
            raise ValueError(f"{scene_id}: prompt policy changed")
        if row.get("supervision") != "scene_level_all_verified_sittable_consensus":
            raise ValueError(f"{scene_id}: primary supervision changed")
        if (
            row.get("all_sittable_gt_file") != consensus_file
            or row.get("all_sittable_gt_sha256") != consensus_hash
            or row.get("all_sittable_array_key") != "all_sittable_affordance"
            or row.get("auxiliary_replica_ids") != sorted(replica_files)
        ):
            raise ValueError(f"{scene_id}: prompt/primary GT binding changed")
        seen_prompts.append(prompt_id)
    if sorted(seen_prompts) != sorted(PROMPTS):
        raise ValueError(f"{scene_id}: watch/write pairs are incomplete")
    print(
        f"[VERIFY] {scene_id}: three verified instances, six auxiliary unions, "
        "watch/write GT byte-identical"
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_dataset_root.expanduser().resolve()
    output_root = args.dataset_root.expanduser().resolve()
    index_path = args.index.expanduser().resolve()
    try:
        index_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Teacher-v9 index must be inside its dataset root") from exc
    index = read_json(index_path)
    if (
        index.get("schema") != OUTPUT_DATASET_SCHEMA
        or index.get("status") != "ALL_SITTABLE_DATASET_PASS"
        or index.get("authorization") != "teacher_v9_all_sittable_cuda_preflight_only"
        or index.get("teacher_lora_v9_training_authorized") is not False
        or index.get("relation_or_distance_used_as_forward_input") is not False
        or index.get("motion_or_target_id_used_as_forward_input") is not False
        or index.get("unverified_sittable_instances_are_negative") is not False
        or index.get("primary_loss_policy")
        != "equal_instance_macro_on_verified_object_masks"
        or index.get("environment_contact_is_auxiliary_only") is not True
        or index.get("development_used_for_parameter_selection") is not False
        or index.get("development_scene_id") != "room_0201"
        or index.get("development_materialization_deferred_until_checkpoint_lock")
        is not True
        or index.get("development_arrays_read") is not False
        or index.get("teacher_forward_inputs")
        != list(TEACHER_FORWARD_INPUTS)
        or index.get("prompt_policy") != PROMPTS
        or index.get("failed_checks")
    ):
        raise ValueError("Teacher-v9 top-level dataset contract changed")
    source_index_path = resolve_under(source_root, index["source_index_file"])
    if (
        sha256_file(source_index_path) != index["source_index_sha256"]
        or str(source_root) != index["source_dataset_root"]
    ):
        raise ValueError("source Teacher-v7 dataset binding changed")
    source_index = read_json(source_index_path)
    if (
        source_index.get("schema") != SOURCE_DATASET_SCHEMA
        or source_index.get("status") != "DENSE_DATASET_PASS"
        or source_index.get("authorization")
        != "teacher_lora_v7_cuda_preflight_only"
        or source_index.get("prompt_policy_id")
        != "purpose_sit_object_agnostic_v1"
        or source_index.get("point_count") != POINT_COUNT
        or source_index.get("contact_dim") != CONTACT_DIM
        or source_index.get("num_scenes") != 3
        or source_index.get("relation_or_distance_used_as_forward_input") is not False
        or source_index.get("target_instance_gt_is_supervision_only") is not True
        or source_index.get("heldout_dense_arrays_read_during_build") is not False
        or source_index.get("teacher_lora_v7_training_authorized") is not False
        or source_index.get("num_dense_motions") != 72
        or source_index.get("num_rows") != 144
        or source_index.get("split_scene_counts")
        != {"train": 2, "development": 1}
        or source_index.get("split_row_counts")
        != {"train": 96, "development": 48}
        or source_index.get("failed_checks")
    ):
        raise ValueError("source Teacher-v7 top-level contract changed")
    source_scene_rows = source_index.get("scenes")
    if not isinstance(source_scene_rows, list) or len(source_scene_rows) != 3:
        raise ValueError("source Teacher-v7 scene inventory changed")
    source_records = {
        str(row["scene_id"]): row for row in source_scene_rows
    }
    if len(source_records) != 3:
        raise ValueError("source Teacher-v7 contains duplicate scene IDs")
    source_actual = {
        scene_id: str(row["split"]) for scene_id, row in source_records.items()
    }
    if source_actual != EXPECTED_SOURCE_SCENES:
        raise ValueError(f"source Teacher-v7 scene split changed: {source_actual}")
    metadata_audit = validate_source_metadata_without_arrays(
        source_root, list(source_records.values())
    )
    if index.get("source_metadata_audit") != metadata_audit:
        raise ValueError("source metadata audit changed")
    development_keys = (
        "scene_id",
        "split",
        "points_file",
        "points_sha256",
        "sidecar_file",
        "sidecar_sha256",
        "instances_file",
        "instances_sha256",
        "motion_index_file",
        "motion_index_sha256",
        "dense_index_file",
        "dense_index_sha256",
    )
    if index.get("development_source_scene_record") != {
        key: source_records["room_0201"][key] for key in development_keys
    }:
        raise ValueError("deferred room_0201 source binding changed")
    output_records = index.get("scenes")
    if not isinstance(output_records, list) or len(output_records) != 2:
        raise ValueError("Teacher-v9 scene records absent")
    actual = {str(row["scene_id"]): str(row["split"]) for row in output_records}
    if len(actual) != 2:
        raise ValueError("Teacher-v9 contains duplicate scene IDs")
    row_ids = [
        str(row.get("row_id", ""))
        for scene_record in output_records
        for row in scene_record.get("rows", [])
    ]
    if len(row_ids) != 4 or len(set(row_ids)) != 4 or any(not value for value in row_ids):
        raise ValueError("Teacher-v9 primary row IDs must be four globally unique values")
    if actual != EXPECTED_OUTPUT_SCENES:
        raise ValueError(f"Teacher-v9 scene split changed: {actual}")
    for record in sorted(output_records, key=lambda row: str(row["scene_id"])):
        scene_id = str(record["scene_id"])
        verify_scene(source_root, output_root, source_records[scene_id], record)

    split_rows = Counter()
    for row in output_records:
        split_rows[str(row["split"])] += len(row["rows"])
    if (
        index.get("scene_count") != 2
        or index.get("replica_count") != 12
        or index.get("prompt_expanded_row_count") != 4
        or index.get("split_scene_counts") != {"train": 2}
        or index.get("split_row_counts") != dict(sorted(split_rows.items()))
        or split_rows != Counter({"train": 4})
    ):
        raise ValueError("Teacher-v9 aggregate counts changed")
    print("[ALL_SITTABLE_DATASET_PASS] Teacher-v9 dataset integrity")
    print("[PASS] all 48 train source maps and 2x6 auxiliary unions recomputed")
    print("[PASS] Bed, normal Chair and High Chair contribute simultaneously")
    print("[PASS] unknown Sit objects ignored; explicit negatives kept separate")
    print("[PASS] no synthetic distance arrays and no relation/distance forward leakage")
    print("[PASS] room_0201 source is hash-bound but its arrays remain unread")
    print("[OK] next gate: train-only visualization, then CUDA preflight")


if __name__ == "__main__":
    main()

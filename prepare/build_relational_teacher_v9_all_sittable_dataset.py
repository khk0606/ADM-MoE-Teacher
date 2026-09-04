#!/usr/bin/env python3
"""Build immutable scene-level all-sittable targets from sealed Teacher-v7 GT."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from relational_teacher_v9_all_sittable_contract import (
    ACTIVE_THRESHOLD,
    CONTACT_DIM,
    GROUP_ORDER,
    OUTPUT_DATASET_SCHEMA,
    POINT_COUNT,
    PROMPTS,
    REPLICA_COUNT,
    SCENE_MANIFEST_SCHEMA,
    SOURCE_DATASET_SCHEMA,
    TEACHER_FORWARD_INPUTS,
    aggregate_all_sittable,
    assert_all_sittable_contract,
    atomic_savez,
    atomic_write_json,
    load_source_scene,
    mask_maps_to_target_and_environment,
    read_json,
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
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def relative_output(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def source_scene_record(record: Mapping[str, object]) -> Dict[str, object]:
    keys = (
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
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"source scene record is missing {missing}")
    return {key: record[key] for key in keys}


def build_scene(
    source_root: Path,
    record: Mapping[str, object],
    temporary_root: Path,
) -> Dict[str, object]:
    loaded = load_source_scene(source_root, record)
    scene_id = str(loaded["scene_id"])
    scene_dir = temporary_root / "scenes" / scene_id
    grouped = loaded["grouped_rows"]
    targets = loaded["targets"]
    filtered_maps = mask_maps_to_target_and_environment(
        grouped,
        targets,
        loaded["maps_by_motion"],
        loaded["stable_instances"],
        loaded["instance_ids"],
    )
    aggregate = aggregate_all_sittable(grouped, targets, filtered_maps)
    assert_all_sittable_contract(aggregate, targets)
    metrics = scene_metrics(
        aggregate,
        targets,
        loaded["stable_instances"],
        loaded["instance_ids"],
    )

    group_files = {}
    for group in GROUP_ORDER:
        path = scene_dir / "group_consensus" / (group + ".npz")
        atomic_savez(path, affordance=aggregate["group_consensus"][group])
        group_files[group] = {
            "file": relative_output(path, temporary_root),
            "sha256": sha256_file(path),
            "target_instance_id": str(targets[group]),
        }

    instance_files = {}
    for target, value in sorted(aggregate["instance_consensus"].items()):
        path = scene_dir / "instance_consensus" / (target + ".npz")
        atomic_savez(path, affordance=value)
        instance_files[target] = {
            "file": relative_output(path, temporary_root),
            "sha256": sha256_file(path),
        }

    ordered_targets = sorted(aggregate["instance_consensus"])
    instance_stack = np.stack(
        [aggregate["instance_consensus"][target] for target in ordered_targets], axis=0
    ).astype(np.float32)
    eligible_numeric_ids = np.asarray(
        [loaded["stable_instances"][target][0] for target in ordered_targets],
        dtype=np.int64,
    )
    unknown_numeric_ids = np.asarray(
        [
            loaded["stable_instances"][target][0]
            for target in loaded["partition"]["unknown_sittable_not_negative"]
        ],
        dtype=np.int64,
    )
    unknown_mask = np.isin(loaded["instance_ids"], unknown_numeric_ids)
    explicit_negative_mask = np.isin(loaded["category_ids"], [3, 4, 5])
    verified_object_mask = np.stack(
        [loaded["instance_ids"] == value for value in eligible_numeric_ids], axis=0
    )
    if bool(np.any(unknown_mask & explicit_negative_mask)):
        raise ValueError(f"{scene_id}: unknown Sit and explicit-negative roles overlap")
    if bool(np.any(unknown_mask & verified_object_mask.any(axis=0))):
        raise ValueError(f"{scene_id}: unknown and verified Sit roles overlap")
    winner = np.argmax(instance_stack, axis=0).astype(np.int8)
    winner[np.max(instance_stack, axis=0) <= 0.0] = -1

    consensus_path = scene_dir / "all_sittable_consensus.npz"
    atomic_savez(
        consensus_path,
        xyz=loaded["xyz"],
        source_indices=loaded["source_indices"],
        instance_ids=loaded["instance_ids"],
        category_ids=loaded["category_ids"],
        eligible_instance_ids=np.asarray(ordered_targets),
        eligible_numeric_instance_ids=eligible_numeric_ids,
        verified_object_mask=verified_object_mask,
        verified_positive_mask=verified_object_mask.any(axis=0),
        instance_affordance=instance_stack,
        instance_active_mask=instance_stack >= ACTIVE_THRESHOLD,
        all_sittable_affordance=aggregate["all_consensus"],
        unknown_sittable_mask=unknown_mask,
        environment_aux_mask=loaded["instance_ids"] == 0,
        explicit_negative_mask=explicit_negative_mask,
        winner_instance_slot=winner,
    )
    replica_rows = []
    for replica, (value, motion_ids) in enumerate(
        zip(aggregate["replicas"], aggregate["replica_motion_ids"])
    ):
        replica_id = f"replica_{replica:02d}"
        path = scene_dir / "replicas" / (replica_id + ".npz")
        atomic_savez(path, affordance=value)
        replica_rows.append(
            {
                "replica_id": replica_id,
                "file": relative_output(path, temporary_root),
                "sha256": sha256_file(path),
                "source_motion_ids": motion_ids,
            }
        )

    manifest_path = scene_dir / "manifest.json"
    manifest = {
        "schema": SCENE_MANIFEST_SCHEMA,
        "status": "ALL_SITTABLE_SCENE_PASS",
        "scene_id": scene_id,
        "split": str(loaded["split"]),
        "point_count": POINT_COUNT,
        "contact_dim": CONTACT_DIM,
        "replica_count": REPLICA_COUNT,
        "aggregation": {
            "motion_group_consensus": "nearest_rank_q75_equals_second_largest_of_six",
            "same_instance_multi_group_union": "exact_pointwise_max",
            "scene_instance_union": "exact_pointwise_max",
            "replica_pairing": "six_row_latin_shift_one_source_per_stratum",
            "source_object_filter": "bound_target_instance_plus_environment_only",
            "active_threshold": ACTIVE_THRESHOLD,
        },
        "verified_target_instances": sorted(set(targets.values())),
        "unknown_sittable_instances_are_negative": False,
        "primary_loss_policy": "equal_instance_macro_on_verified_object_masks",
        "environment_contact_is_auxiliary_only": True,
        "instance_partition": loaded["partition"],
        "group_targets": dict(targets),
        "group_consensus": group_files,
        "instance_consensus": instance_files,
        "all_sittable_consensus_file": relative_output(
            consensus_path, temporary_root
        ),
        "all_sittable_consensus_sha256": sha256_file(consensus_path),
        "replicas": replica_rows,
        "metrics": metrics,
        "source_files": loaded["source_files"],
        "source_hashes": loaded["source_hashes"],
        "source_dense_bindings": loaded["source_bindings"],
        "relation_or_distance_used": False,
        "teacher_forward_inputs": list(TEACHER_FORWARD_INPUTS),
        "development_used_for_parameter_selection": False,
        "failed_checks": [],
    }
    atomic_write_json(manifest_path, manifest)

    # There is exactly one primary x0 target per scene.  Replicas and original
    # motion maps are auxiliary evidence only; putting replica IDs into rows
    # would recreate the old same-input/different-primary-target conflict.
    prompt_rows = []
    for prompt_id in sorted(PROMPTS):
        prompt_rows.append(
            {
                "row_id": f"{scene_id}_{prompt_id}",
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "text": PROMPTS[prompt_id],
                "all_sittable_gt_file": relative_output(
                    consensus_path, temporary_root
                ),
                "all_sittable_gt_sha256": sha256_file(consensus_path),
                "all_sittable_array_key": "all_sittable_affordance",
                "auxiliary_replica_ids": [row["replica_id"] for row in replica_rows],
                "supervision": "scene_level_all_verified_sittable_consensus",
            }
        )
    print(
        f"[BUILD] {scene_id}: verified={manifest['verified_target_instances']} "
        f"unknown={loaded['partition']['unknown_sittable_not_negative']} "
        f"aux_replicas={len(replica_rows)} primary_rows={len(prompt_rows)}"
    )
    return {
        "scene_id": scene_id,
        "split": str(loaded["split"]),
        "source_scene_record": source_scene_record(record),
        "manifest_file": relative_output(manifest_path, temporary_root),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_target_instances": manifest["verified_target_instances"],
        "unknown_sittable_instances_are_negative": False,
        "replica_count": len(replica_rows),
        "row_count": len(prompt_rows),
        "rows": prompt_rows,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_dataset_root.expanduser().resolve()
    source_index_path = args.source_index.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    try:
        source_index_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("source index must be inside source dataset root") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Teacher-v9 dataset: {output_root}")

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
        raise ValueError("source Teacher-v7 dataset is not the sealed dense dataset")
    source_scenes = source_index.get("scenes")
    if not isinstance(source_scenes, list) or len(source_scenes) != 3:
        raise ValueError("source Teacher-v7 scene inventory changed")
    actual = {str(row["scene_id"]): str(row["split"]) for row in source_scenes}
    if actual != EXPECTED_SOURCE_SCENES:
        raise ValueError(f"source scene split changed: {actual}")
    by_scene = {str(row["scene_id"]): row for row in source_scenes}
    source_metadata_audit = validate_source_metadata_without_arrays(
        source_root, source_scenes
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.parent / ("." + output_root.name + ".tmp." + str(os.getpid()))
    if temporary_root.exists():
        raise FileExistsError(temporary_root)
    temporary_root.mkdir()
    try:
        scene_records = [
            build_scene(source_root, by_scene[scene_id], temporary_root)
            for scene_id in sorted(EXPECTED_OUTPUT_SCENES)
        ]
        split_rows = Counter()
        for row in scene_records:
            split_rows[str(row["split"])] += int(row["row_count"])
        payload = {
            "schema": OUTPUT_DATASET_SCHEMA,
            "status": "ALL_SITTABLE_DATASET_PASS",
            "authorization": "teacher_v9_all_sittable_cuda_preflight_only",
            "source_dataset_root": str(source_root),
            "source_index_file": str(source_index_path.relative_to(source_root)),
            "source_index_sha256": sha256_file(source_index_path),
            "point_count": POINT_COUNT,
            "contact_dim": CONTACT_DIM,
            "scene_count": 2,
            "replica_count": 12,
            "prompt_expanded_row_count": 4,
            "split_scene_counts": {"train": 2},
            "split_row_counts": dict(sorted(split_rows.items())),
            "prompt_policy": PROMPTS,
            "teacher_forward_inputs": list(TEACHER_FORWARD_INPUTS),
            "relation_or_distance_used_as_forward_input": False,
            "motion_or_target_id_used_as_forward_input": False,
            "unverified_sittable_instances_are_negative": False,
            "primary_loss_policy": "equal_instance_macro_on_verified_object_masks",
            "environment_contact_is_auxiliary_only": True,
            "development_scene_id": "room_0201",
            "development_source_scene_record": source_scene_record(
                by_scene["room_0201"]
            ),
            "source_metadata_audit": source_metadata_audit,
            "development_materialization_deferred_until_checkpoint_lock": True,
            "development_arrays_read": False,
            "development_used_for_parameter_selection": False,
            "teacher_lora_v9_training_authorized": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scenes": scene_records,
            "failed_checks": [],
        }
        atomic_write_json(temporary_root / "index.json", payload)
        os.replace(str(temporary_root), str(output_root))
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(str(temporary_root))
        raise

    print("[ALL_SITTABLE_DATASET_PASS] Teacher-v9 scene-level Sit GT")
    print("[PASS] each scene contains Bed + normal Chair + High Chair simultaneously")
    print("[PASS] one primary GT per scene; watch/write share byte-identical GT")
    print("[PASS] six balanced unions retained as auxiliary motion evidence only")
    print("[PASS] unverified Chair/Bed instances remain unknown, never negative")
    print("[PASS] room_0201 arrays remain unread and materialization is deferred")
    print("[OK] train scenes=2 auxiliary replicas=12 primary prompt rows=4")
    print("[OK] Teacher-v9 training authorized: False")
    print(f"[OK] output: {output_root / 'index.json'}")


if __name__ == "__main__":
    main()

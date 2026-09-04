#!/usr/bin/env python3
"""Fail-closed contracts for the Teacher-v9 all-sittable CUDA preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import numpy as np

from relational_teacher_v9_all_sittable_contract import (
    CONTACT_DIM,
    OUTPUT_DATASET_SCHEMA,
    POINT_COUNT,
    PROMPTS,
    SCENE_MANIFEST_SCHEMA,
    TEACHER_FORWARD_INPUTS,
    read_json,
    resolve_under,
    sha256_file,
)


SCHEMA = "relational_teacher_v9_all_sittable_lora_cuda_preflight_v1"
POLICY_SCHEMA = "relational_teacher_v9_all_sittable_metric_policy_v1"
EXPECTED_TRAIN_SCENES = ("room_0101", "room_0102")
EXPECTED_DEVELOPMENT_SCENES = ("room_0201",)
EXPECTED_TARGETS = {
    "room_0101": ("bed_01", "chair_01", "chair_06"),
    "room_0102": ("bed_01", "chair_05", "chair_06"),
}
ROLE_BY_INSTANCE = {
    "room_0101": {
        "bed_01": "bed",
        "chair_01": "normal_chair",
        "chair_06": "high_chair",
    },
    "room_0102": {
        "bed_01": "bed",
        "chair_05": "normal_chair",
        "chair_06": "high_chair",
    },
}

# These numbers are fixed before any optimizer is constructed.  They are used
# only to prove that structural counterexamples fail.  Later candidate gates
# also compare against the measured frozen-v5r4 baseline stored in policy.json.
ABSOLUTE_PRESENCE_LIMITS = {
    "minimum_soft_recall": 0.75,
    "minimum_topk_overlap": 0.50,
    "maximum_active_support_mae": 0.10,
    "maximum_hotspot_centroid_distance_xy": 0.60,
    "maximum_negative_mean": 0.10,
    "maximum_negative_max": 0.80,
}
RELATIVE_SELECTION_RULES = {
    "tie_epsilon": 1e-8,
    "bed_active_support_mae_relative_cap": 1.005,
    "chair_active_support_mae_requires_strict_improvement": True,
    "soft_recall_absolute_tolerance": 0.005,
    "topk_overlap_absolute_tolerance": 0.02,
    "hotspot_centroid_distance_xy_tolerance": 0.05,
    "explicit_negative_mean_addition_cap": 0.002,
    "explicit_negative_max_addition_cap": 0.01,
    "prompt_invariance_relative_cap": 1.02,
    "v5_replay_dense_relative_cap": 1.01,
}
OBJECTIVE_WEIGHTS = {
    "instance_macro_primary": 1.0,
    "verified_union": 0.5,
    "environment_auxiliary": 0.1,
    "explicit_negative_addition": 0.5,
    "paired_prompt_invariance": 1.0,
    "frozen_v5_replay_preservation": 1.0,
    "lora_regularizer": 1e-4,
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_top_index(
    dataset_root: Path,
    source_dataset_root: Path,
    index_file: Path,
) -> MutableMapping[str, object]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    source_dataset_root = Path(source_dataset_root).expanduser().resolve()
    index_file = Path(index_file).expanduser().resolve()
    try:
        index_file.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("Teacher-v9 index must be inside its dataset root") from exc
    value = read_json(index_file)
    if (
        value.get("schema") != OUTPUT_DATASET_SCHEMA
        or value.get("status") != "ALL_SITTABLE_DATASET_PASS"
        or value.get("authorization")
        != "teacher_v9_all_sittable_cuda_preflight_only"
        or value.get("point_count") != POINT_COUNT
        or value.get("contact_dim") != CONTACT_DIM
        or value.get("scene_count") != 2
        or value.get("replica_count") != 12
        or value.get("prompt_expanded_row_count") != 4
        or value.get("split_scene_counts") != {"train": 2}
        or value.get("split_row_counts") != {"train": 4}
        or value.get("prompt_policy") != PROMPTS
        or value.get("teacher_forward_inputs") != list(TEACHER_FORWARD_INPUTS)
        or value.get("relation_or_distance_used_as_forward_input") is not False
        or value.get("motion_or_target_id_used_as_forward_input") is not False
        or value.get("unverified_sittable_instances_are_negative") is not False
        or value.get("primary_loss_policy")
        != "equal_instance_macro_on_verified_object_masks"
        or value.get("environment_contact_is_auxiliary_only") is not True
        or value.get("development_scene_id") != "room_0201"
        or value.get("development_materialization_deferred_until_checkpoint_lock")
        is not True
        or value.get("development_arrays_read") is not False
        or value.get("development_used_for_parameter_selection") is not False
        or value.get("teacher_lora_v9_training_authorized") is not False
        or value.get("failed_checks")
    ):
        raise ValueError("Teacher-v9 all-sittable index is not sealed preflight-only data")
    if Path(str(value.get("source_dataset_root", ""))).resolve() != source_dataset_root:
        raise ValueError("Teacher-v9 source-dataset root binding changed")
    source_index = resolve_under(source_dataset_root, value["source_index_file"])
    if sha256_file(source_index) != str(value["source_index_sha256"]):
        raise ValueError("Teacher-v9 source-index hash changed")
    records = value.get("scenes")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("Teacher-v9 must expose exactly two train scenes")
    if tuple(sorted(str(row.get("scene_id")) for row in records)) != EXPECTED_TRAIN_SCENES:
        raise ValueError("Teacher-v9 train-scene inventory changed")
    if any(str(row.get("split")) != "train" for row in records):
        raise ValueError("non-train row entered the Teacher-v9 preflight")
    return value


def _required_array(source: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in source:
        raise ValueError("all-sittable bundle is missing array: " + name)
    return np.asarray(source[name])


def load_train_scene_bundle(
    dataset_root: Path,
    source_dataset_root: Path,
    scene_record: Mapping[str, object],
) -> Dict[str, object]:
    """Load one materialized train scene without touching development arrays."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    source_dataset_root = Path(source_dataset_root).expanduser().resolve()
    scene_id = str(scene_record.get("scene_id", ""))
    if scene_id not in EXPECTED_TRAIN_SCENES or scene_record.get("split") != "train":
        raise ValueError("preflight scene is not an expected training scene")
    if tuple(scene_record.get("verified_target_instances", ())) != EXPECTED_TARGETS[scene_id]:
        raise ValueError(f"{scene_id}: verified target inventory/order changed")
    if (
        scene_record.get("unknown_sittable_instances_are_negative") is not False
        or scene_record.get("replica_count") != 6
        or scene_record.get("row_count") != 2
    ):
        raise ValueError(f"{scene_id}: scene-level role contract changed")

    manifest_file = resolve_under(dataset_root, scene_record["manifest_file"])
    if sha256_file(manifest_file) != str(scene_record["manifest_sha256"]):
        raise ValueError(f"{scene_id}: manifest hash changed")
    manifest = read_json(manifest_file)
    if (
        manifest.get("schema") != SCENE_MANIFEST_SCHEMA
        or manifest.get("status") != "ALL_SITTABLE_SCENE_PASS"
        or manifest.get("scene_id") != scene_id
        or manifest.get("split") != "train"
        or manifest.get("point_count") != POINT_COUNT
        or manifest.get("contact_dim") != CONTACT_DIM
        or manifest.get("replica_count") != 6
        or tuple(manifest.get("verified_target_instances", ()))
        != EXPECTED_TARGETS[scene_id]
        or manifest.get("unknown_sittable_instances_are_negative") is not False
        or manifest.get("primary_loss_policy")
        != "equal_instance_macro_on_verified_object_masks"
        or manifest.get("environment_contact_is_auxiliary_only") is not True
        or manifest.get("relation_or_distance_used") is not False
        or manifest.get("teacher_forward_inputs") != list(TEACHER_FORWARD_INPUTS)
        or manifest.get("development_used_for_parameter_selection") is not False
        or manifest.get("failed_checks")
    ):
        raise ValueError(f"{scene_id}: all-sittable scene manifest changed")

    rows = scene_record.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError(f"{scene_id}: expected exactly two primary prompt rows")
    by_prompt = {str(row.get("prompt_id")): row for row in rows}
    if set(by_prompt) != set(PROMPTS):
        raise ValueError(f"{scene_id}: prompt IDs changed")
    hashes = {str(row.get("all_sittable_gt_sha256", "")) for row in rows}
    files = {str(row.get("all_sittable_gt_file", "")) for row in rows}
    if len(hashes) != 1 or len(files) != 1:
        raise ValueError(f"{scene_id}: watch/write do not share one exact GT")
    for prompt_id, row in by_prompt.items():
        if (
            row.get("text") != PROMPTS[prompt_id]
            or row.get("all_sittable_array_key") != "all_sittable_affordance"
            or row.get("supervision")
            != "scene_level_all_verified_sittable_consensus"
            or len(row.get("auxiliary_replica_ids", ())) != 6
        ):
            raise ValueError(f"{scene_id}: primary prompt row changed")

    consensus_file = resolve_under(dataset_root, next(iter(files)))
    consensus_hash = sha256_file(consensus_file)
    if consensus_hash != next(iter(hashes)):
        raise ValueError(f"{scene_id}: primary GT hash changed")
    if (
        str(manifest.get("all_sittable_consensus_file")) != next(iter(files))
        or str(manifest.get("all_sittable_consensus_sha256")) != consensus_hash
    ):
        raise ValueError(f"{scene_id}: manifest/row GT binding differs")

    with np.load(consensus_file, allow_pickle=False) as source:
        arrays = {name: _required_array(source, name) for name in (
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
        )}
    names = tuple(str(value) for value in arrays["eligible_instance_ids"].tolist())
    if names != EXPECTED_TARGETS[scene_id]:
        raise ValueError(f"{scene_id}: consensus target order changed")
    if arrays["xyz"].shape != (POINT_COUNT, 3):
        raise ValueError(f"{scene_id}: xyz shape changed")
    if arrays["source_indices"].shape != (POINT_COUNT,):
        raise ValueError(f"{scene_id}: source-index shape changed")
    if arrays["instance_ids"].shape != (POINT_COUNT,) or arrays["category_ids"].shape != (POINT_COUNT,):
        raise ValueError(f"{scene_id}: semantic sidecar shape changed")
    if arrays["verified_object_mask"].shape != (3, POINT_COUNT):
        raise ValueError(f"{scene_id}: verified object masks must be [3,8192]")
    if arrays["instance_affordance"].shape != (3, POINT_COUNT, CONTACT_DIM):
        raise ValueError(f"{scene_id}: instance affordance must be [3,8192,6]")
    if arrays["all_sittable_affordance"].shape != (POINT_COUNT, CONTACT_DIM):
        raise ValueError(f"{scene_id}: all-sittable GT must be [8192,6]")
    for name in (
        "verified_positive_mask",
        "unknown_sittable_mask",
        "environment_aux_mask",
        "explicit_negative_mask",
    ):
        if arrays[name].shape != (POINT_COUNT,):
            raise ValueError(f"{scene_id}: {name} shape changed")
    # Winner is recorded independently for every point and contact channel.
    # instance_stack is [3,N,6], therefore argmax(axis=0) is exactly [N,6].
    if arrays["winner_instance_slot"].shape != (POINT_COUNT, CONTACT_DIM):
        raise ValueError(f"{scene_id}: winner_instance_slot must be [8192,6]")
    if arrays["instance_active_mask"].shape != (3, POINT_COUNT, CONTACT_DIM):
        raise ValueError(f"{scene_id}: instance_active_mask must be [3,8192,6]")
    if arrays["eligible_numeric_instance_ids"].shape != (3,):
        raise ValueError(f"{scene_id}: eligible numeric instance IDs must be [3]")
    instance_targets = arrays["instance_affordance"].astype(np.float32)
    all_target = arrays["all_sittable_affordance"].astype(np.float32)
    if (
        not np.isfinite(instance_targets).all()
        or not np.isfinite(all_target).all()
        or np.any(instance_targets < 0.0)
        or np.any(instance_targets > 1.0)
        or np.any(all_target < 0.0)
        or np.any(all_target > 1.0)
        or not np.array_equal(all_target, instance_targets.max(axis=0))
    ):
        raise ValueError(f"{scene_id}: primary GT is not exact finite max union")
    verified = arrays["verified_object_mask"].astype(bool)
    positive = arrays["verified_positive_mask"].astype(bool)
    unknown = arrays["unknown_sittable_mask"].astype(bool)
    environment = arrays["environment_aux_mask"].astype(bool)
    negative = arrays["explicit_negative_mask"].astype(bool)
    if (
        not np.array_equal(positive, verified.any(axis=0))
        or np.any(verified.sum(axis=1) <= 0)
        or np.any(verified.sum(axis=0) > 1)
        or not np.any(unknown)
        or not np.any(environment)
        or not np.any(negative)
        or np.any(unknown & positive)
        or np.any(negative & (unknown | positive))
        or np.any(environment & (unknown | positive | negative))
    ):
        raise ValueError(f"{scene_id}: point-role masks overlap or are empty")

    source_record = scene_record.get("source_scene_record")
    if not isinstance(source_record, Mapping) or source_record.get("scene_id") != scene_id:
        raise ValueError(f"{scene_id}: source scene binding missing")
    points_file = resolve_under(source_dataset_root, source_record["points_file"])
    if sha256_file(points_file) != str(source_record["points_sha256"]):
        raise ValueError(f"{scene_id}: source point-cloud hash changed")
    with np.load(points_file, allow_pickle=False) as source:
        points = np.asarray(source["points"], dtype=np.float32)
    if points.shape != (POINT_COUNT, 6) or not np.array_equal(points[:, :3], arrays["xyz"]):
        raise ValueError(f"{scene_id}: source point cloud differs from GT xyz")

    return {
        "scene_id": scene_id,
        "points": points,
        "xyz": arrays["xyz"].astype(np.float32),
        "instance_ids": arrays["instance_ids"].astype(np.int64),
        "category_ids": arrays["category_ids"].astype(np.int64),
        "instance_names": names,
        "instance_roles": tuple(ROLE_BY_INSTANCE[scene_id][name] for name in names),
        "instance_targets": instance_targets,
        "all_target": all_target,
        "verified_object_mask": verified,
        "verified_positive_mask": positive,
        "unknown_sittable_mask": unknown,
        "environment_aux_mask": environment,
        "explicit_negative_mask": negative,
        "manifest_file": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "consensus_file": str(consensus_file),
        "consensus_sha256": consensus_hash,
        "points_file": str(points_file),
        "points_sha256": sha256_file(points_file),
    }


__all__ = [
    "ABSOLUTE_PRESENCE_LIMITS",
    "EXPECTED_DEVELOPMENT_SCENES",
    "EXPECTED_TARGETS",
    "EXPECTED_TRAIN_SCENES",
    "OBJECTIVE_WEIGHTS",
    "POLICY_SCHEMA",
    "RELATIVE_SELECTION_RULES",
    "ROLE_BY_INSTANCE",
    "SCHEMA",
    "canonical_sha256",
    "load_train_scene_bundle",
    "validate_top_index",
]

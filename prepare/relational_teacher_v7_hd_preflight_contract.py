#!/usr/bin/env python3
"""CPU-verifiable input contract for the Teacher-v7 High-Desk preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from relational_teacher_v6_contract import (
    POINT_COUNT,
    PROMPTS,
    read_json,
    sha256_file,
)


SCHEMA = "relational_teacher_v7_hd_lora_cuda_preflight_v1"
DATASET_SCHEMA = "relational_teacher_v7_hd_dataset_v1"
DENSE_INDEX_SCHEMA = "relational_teacher_v7_hd_dense_contact_v2"
EXPECTED_TRAIN_SCENES = ("room_0101", "room_0102")
EXPECTED_DEVELOPMENT_SCENES = ("room_0201",)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_dataset_path(dataset_root: Path, raw: object) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    path = (dataset_root / str(raw)).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("dataset record escapes dataset root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def select_train_probe(
    dataset_root: Path, scene_record: Mapping[str, object]
) -> Dict[str, object]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    scene_id = str(scene_record["scene_id"])
    if str(scene_record.get("split")) != "train":
        raise ValueError("preflight probe is not a training scene")
    points_file = resolve_dataset_path(dataset_root, scene_record["points_file"])
    sidecar_file = resolve_dataset_path(dataset_root, scene_record["sidecar_file"])
    if sha256_file(points_file) != str(scene_record["points_sha256"]):
        raise ValueError(f"{scene_id}: points hash changed")
    if sha256_file(sidecar_file) != str(scene_record["sidecar_sha256"]):
        raise ValueError(f"{scene_id}: sidecar hash changed")
    with np.load(points_file, allow_pickle=False) as source:
        points = source["points"].astype(np.float32)
    with np.load(sidecar_file, allow_pickle=False) as source:
        instance_ids = source["instance_ids"].astype(np.int64)
        category_ids = source["category_ids"].astype(np.int64)
        source_indices = source["source_indices"].astype(np.int64)
    if points.shape != (POINT_COUNT, 6):
        raise ValueError(f"{scene_id}: points shape changed")

    dense_index_path = resolve_dataset_path(
        dataset_root, scene_record["dense_index_file"]
    )
    if sha256_file(dense_index_path) != str(scene_record["dense_index_sha256"]):
        raise ValueError(f"{scene_id}: top-level dense-index binding changed")
    dense_index = read_json(dense_index_path)
    if (
        dense_index.get("schema") != DENSE_INDEX_SCHEMA
        or dense_index.get("status") != "DENSE_CONTACT_V2_PASS"
        or dense_index.get("scene_id") != scene_id
        or dense_index.get("relation_or_distance_used") is not False
        or dense_index.get("motion_count") != 24
        or dense_index.get("prompt_expanded_row_count") != 48
        or dense_index.get("teacher_lora_training_authorized") is not False
    ):
        raise ValueError(f"{scene_id}: dense-contact source is not sealed PASS")
    rows = dense_index.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError(f"{scene_id}: dense row inventory changed")
    high_desk_rows = [
        row
        for row in rows
        if "_hc_hd_" in str(row.get("motion_id", ""))
        and str(row.get("target_instance_id")) == "chair_06"
    ]
    if len(high_desk_rows) != 6:
        raise ValueError(
            f"{scene_id}: expected six sealed hc_hd High-Desk rows, "
            f"got {len(high_desk_rows)}"
        )
    row = sorted(high_desk_rows, key=lambda item: str(item["motion_id"]))[0]
    manifest_file = resolve_dataset_path(dataset_root, row["manifest_file"])
    if sha256_file(manifest_file) != str(row["manifest_sha256"]):
        raise ValueError(f"{scene_id}: dense manifest hash changed")
    manifest = read_json(manifest_file)
    if (
        manifest.get("schema") != DENSE_INDEX_SCHEMA + "_item"
        or manifest.get("status") != "DENSE_CONTACT_ITEM_PASS"
        or manifest.get("relation_or_distance_used") is not False
        or manifest.get("compatible_prompt_ids") != sorted(PROMPTS)
        or manifest.get("motion_id") != row.get("motion_id")
        or manifest.get("target_instance_id") != "chair_06"
    ):
        raise ValueError(f"{scene_id}: dense item contract changed")
    affordance_file = resolve_dataset_path(dataset_root, manifest["affordance_file"])
    if sha256_file(affordance_file) != str(manifest["affordance_sha256"]):
        raise ValueError(f"{scene_id}: affordance hash changed")
    with np.load(affordance_file, allow_pickle=False) as source:
        affordance = source["affordance"].astype(np.float32)
        dense_instances = source["instance_ids"].astype(np.int64)
        dense_sources = source["source_indices"].astype(np.int64)
        sigma = float(source["sigma"])
    if affordance.shape != (POINT_COUNT, 6):
        raise ValueError(f"{scene_id}: affordance shape changed")
    if (
        not np.isfinite(affordance).all()
        or np.any(affordance < 0.0)
        or np.any(affordance > 1.0)
    ):
        raise ValueError(f"{scene_id}: affordance range changed")
    if not np.isclose(sigma, 0.8, rtol=0.0, atol=1e-7):
        raise ValueError(f"{scene_id}: sigma changed")
    if not np.array_equal(instance_ids, dense_instances):
        raise ValueError(f"{scene_id}: dense instance order changed")
    if not np.array_equal(source_indices, dense_sources):
        raise ValueError(f"{scene_id}: dense source order changed")
    return {
        "scene_id": scene_id,
        "motion_id": str(row["motion_id"]),
        "target_instance_id": str(row["target_instance_id"]),
        "probe_role": "new_high_desk_hc_hd_train_motion",
        "points": points,
        "instance_ids": instance_ids,
        "category_ids": category_ids,
        "affordance": affordance,
        "points_sha256": sha256_file(points_file),
        "sidecar_sha256": sha256_file(sidecar_file),
        "dense_index_sha256": sha256_file(dense_index_path),
        "affordance_sha256": sha256_file(affordance_file),
    }

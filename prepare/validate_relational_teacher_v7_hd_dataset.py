#!/usr/bin/env python3
"""Validate the top-level Teacher-v7 dense dataset index without reading held-out arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPTS = {
    "sit_watch_v1": "Sit anywhere to watch.",
    "sit_write_v1": "Sit anywhere to write.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    index = read_json(args.index.expanduser().resolve())
    if index.get("schema") != "relational_teacher_v7_hd_dataset_v1":
        raise ValueError("dataset schema changed")
    if index.get("status") != "DENSE_DATASET_PASS" or index.get("failed_checks"):
        raise ValueError("dataset status changed")
    if index.get("authorization") != "teacher_lora_v7_cuda_preflight_only":
        raise ValueError("authorization changed")
    if index.get("teacher_lora_v7_training_authorized") is not False:
        raise ValueError("dataset index prematurely authorized training")
    if index.get("relation_or_distance_used_as_forward_input") is not False:
        raise ValueError("relation/distance leakage guard changed")
    if index.get("target_instance_gt_is_supervision_only") is not True:
        raise ValueError("target GT role changed")
    if index.get("heldout_dense_arrays_read_during_build") is not False:
        raise ValueError("held-out array read guard changed")
    if sha256(root / index["scene_split_file"]) != index["scene_split_sha256"]:
        raise ValueError("scene split hash changed")
    if sha256(root / index["staging_report_file"]) != index["staging_report_sha256"]:
        raise ValueError("staging report hash changed")
    scenes = index.get("scenes", [])
    if len(scenes) != 3 or {row["scene_id"] for row in scenes} != {"room_0101", "room_0102", "room_0201"}:
        raise ValueError("scene inventory changed")
    total_rows = 0
    split_counts = {"train": 0, "development": 0}
    for scene in scenes:
        scene_id = scene["scene_id"]
        expected_split = "development" if scene_id == "room_0201" else "train"
        if scene["split"] != expected_split:
            raise ValueError(f"{scene_id}: split changed")
        for file_key, hash_key in (
            ("points_file", "points_sha256"),
            ("sidecar_file", "sidecar_sha256"),
            ("instances_file", "instances_sha256"),
            ("motion_index_file", "motion_index_sha256"),
            ("dense_index_file", "dense_index_sha256"),
        ):
            if sha256(root / scene[file_key]) != scene[hash_key]:
                raise ValueError(f"{scene_id}: {file_key} hash changed")
        dense = read_json(root / scene["dense_index_file"])
        if dense.get("status") != "DENSE_CONTACT_V2_PASS" or dense.get("motion_count") != 24:
            raise ValueError(f"{scene_id}: dense index changed")
        rows = scene.get("rows", [])
        if len(rows) != 48 or len({row["row_id"] for row in rows}) != 48:
            raise ValueError(f"{scene_id}: expanded rows changed")
        for row in rows:
            if PROMPTS.get(row["prompt_id"]) != row["text"]:
                raise ValueError(f"{scene_id}: prompt contract changed")
            if row["supervision"] != "motion_derived_dense_contact":
                raise ValueError(f"{scene_id}: supervision changed")
            manifest = root / row["dense_manifest_file"]
            if sha256(manifest) != row["dense_manifest_sha256"]:
                raise ValueError(f"{scene_id}: dense manifest hash changed")
        total_rows += len(rows)
        split_counts[expected_split] += len(rows)
        print(f"[VERIFY] {scene_id}: split={expected_split} rows=48 dense=24")
    if total_rows != 144 or index.get("num_rows") != 144 or index.get("num_dense_motions") != 72:
        raise ValueError("dataset aggregate counts changed")
    if index.get("split_row_counts") != split_counts:
        raise ValueError("split row counts changed")
    print("[DENSE_DATASET_PASS] relational Teacher-v7 dataset integrity")
    print("[PASS] 3 scene hashes, 72 dense-motion bindings and 144 prompt rows")
    print("[PASS] train/development split and held-out-array unread guard")
    print("[OK] next gate: Teacher LoRA v7 CUDA preflight")


if __name__ == "__main__":
    main()

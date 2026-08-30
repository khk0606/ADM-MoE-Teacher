#!/usr/bin/env python3
"""Validate the non-destructive Teacher-v7 High-Desk dataset staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


SCENES = ("room_0101", "room_0102", "room_0201")


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


def inspect(path: Path, expected_frames: int) -> None:
    frames = 0
    maximum_error = 0.0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        values = [float(token) for token in raw.replace(",", " ").split()]
        if len(values) != 103 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid row: {path}:{line_number}")
        for column in range(3, 103, 4):
            norm = math.sqrt(sum(value * value for value in values[column : column + 4]))
            maximum_error = max(maximum_error, abs(norm - 1.0))
        frames += 1
    if frames != expected_frames or maximum_error > 5.0e-5:
        raise ValueError(f"motion contract changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    report = read_json(root / "staging_report.json")
    if report.get("schema") != "relational_teacher_v7_hd_dataset_staging_v1":
        raise ValueError("staging schema changed")
    if report.get("status") != "HD_DATASET_STAGING_PASS" or report.get("failed_checks"):
        raise ValueError("staging status changed")
    if report.get("relation_or_distance_used_as_forward_input") is not False:
        raise ValueError("relation/distance leakage guard changed")
    provenance = root / "provenance"
    if sha256(provenance / "teacher_v6_index.json") != report["source_dataset_index_sha256"]:
        raise ValueError("v6 source index provenance changed")
    if sha256(provenance / "hc_hd_train_selection.json") != report["train_selection_sha256"]:
        raise ValueError("train selection provenance changed")
    if sha256(provenance / "hcw_hdw_development_selection.json") != report["development_selection_sha256"]:
        raise ValueError("development selection provenance changed")

    summary_by_scene = {row["scene_id"]: row for row in report.get("scenes", [])}
    if set(summary_by_scene) != set(SCENES):
        raise ValueError("scene staging inventory changed")
    group_sets = {}
    for scene_id in SCENES:
        scene_dir = root / "scenes" / scene_id
        index_path = scene_dir / "motion_index.json"
        index = read_json(index_path)
        if index.get("schema") != "relational_teacher_v7_hd_motion_inventory_v1":
            raise ValueError(f"{scene_id}: motion schema changed")
        if index.get("status") != "HD_MOTION_BINDING_PASS":
            raise ValueError(f"{scene_id}: motion status changed")
        if index.get("relation_or_distance_used_as_forward_input") is not False:
            raise ValueError(f"{scene_id}: relation/distance leakage")
        if index.get("train_development_source_overlap") is not False:
            raise ValueError(f"{scene_id}: source-overlap guard changed")
        rows = index.get("rows", [])
        if len(rows) != int(index["motion_count"]):
            raise ValueError(f"{scene_id}: motion count changed")
        dense = [
            row for row in rows
            if row.get("teacher_v7_usage", row.get("teacher_v6_usage"))
            == "motion_derived_dense_eligible"
        ]
        if len(dense) != 24 or int(index["dense_sit_eligible_count"]) != 24:
            raise ValueError(f"{scene_id}: expected 24 dense motions")
        counts = dict(sorted(Counter(row["target_instance_id"] for row in dense).items()))
        if counts != index.get("dense_sit_target_counts"):
            raise ValueError(f"{scene_id}: target counts changed")
        additions = [row for row in rows if row.get("source_collection_id") in {"hc_hd", "hcw_hdw"}]
        if len(additions) != 6 or len({row["source_group_id"] for row in additions}) != 6:
            raise ValueError(f"{scene_id}: High-Desk six changed")
        expected_role = "development" if scene_id == "room_0201" else "train"
        expected_collection = "hcw_hdw" if scene_id == "room_0201" else "hc_hd"
        expected_target = "chair_02" if scene_id == "room_0201" else "chair_06"
        for row in additions:
            if row["source_split_role"] != expected_role or row["source_collection_id"] != expected_collection:
                raise ValueError(f"{scene_id}: split/source binding changed")
            if row["target_instance_id"] != expected_target or row["purpose_instance_id"] != "desk_01":
                raise ValueError(f"{scene_id}: target/purpose binding changed")
            if row["compatible_prompt_ids"] != ["sit_watch_v1", "sit_write_v1"]:
                raise ValueError(f"{scene_id}: prompt invariance binding changed")
            motion = scene_dir / row["motion_file"]
            if sha256(motion) != row["motion_sha256"]:
                raise ValueError(f"{scene_id}: motion hash changed")
            inspect(motion, int(row["total_frames"]))
            if not 0 <= int(row["frame_start_inclusive"]) <= int(row["frame_end_inclusive"]) < int(row["total_frames"]):
                raise ValueError(f"{scene_id}: motion crop changed")
        if sha256(index_path) != summary_by_scene[scene_id]["motion_index_sha256"]:
            raise ValueError(f"{scene_id}: staging summary hash changed")
        group_sets[scene_id] = {row["source_group_id"] for row in additions}
        print(f"[VERIFY] {scene_id}: dense=24 new=6 targets={counts}")

    if group_sets["room_0101"] != group_sets["room_0102"]:
        raise ValueError("train scenes no longer share exact train sources")
    if group_sets["room_0101"] & group_sets["room_0201"]:
        raise ValueError("train/development source overlap")
    print("[HD_DATASET_STAGING_PASS] Teacher-v7 High-Desk staging integrity")
    print("[PASS] 18 added motion hashes, crops, columns and quaternions recomputed")
    print("[PASS] train source reuse and independent development-source split")
    print("[PASS] purpose relation/distance excluded from Teacher forward")
    print("[OK] next gate: generate dense_contact_v2 for all three scenes")


if __name__ == "__main__":
    main()

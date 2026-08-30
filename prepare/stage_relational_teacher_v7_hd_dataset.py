#!/usr/bin/env python3
"""Create a non-destructive Teacher-v7 dataset with genuine High-Desk motions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


MOTION_INDEX_SCHEMA_V6 = "relational_teacher_v6_motion_inventory_v1"
MOTION_INDEX_SCHEMA_V7 = "relational_teacher_v7_hd_motion_inventory_v1"
PROMPT_IDS = ["sit_watch_v1", "sit_write_v1"]
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
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect_motion(path: Path) -> tuple[int, float]:
    frames = 0
    maximum_error = 0.0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        values = [float(token) for token in raw.replace(",", " ").split()]
        if len(values) != 103 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid motion row: {path}:{line_number}")
        for column in range(3, 103, 4):
            norm = math.sqrt(sum(value * value for value in values[column : column + 4]))
            maximum_error = max(maximum_error, abs(norm - 1.0))
        frames += 1
    if frames == 0 or maximum_error > 5.0e-5:
        raise ValueError(f"motion numerical contract failed: {path}")
    return frames, maximum_error


def find_one(root: Path, candidate_id: str) -> Path:
    matches = sorted(root.glob(candidate_id + "_*.txt"))
    if len(matches) != 1:
        raise ValueError(f"expected one bound motion for {candidate_id} under {root}")
    return matches[0]


def new_motion_row(
    *,
    scene_id: str,
    target: str,
    candidate_id: str,
    source_group_id: str,
    source: Path,
    destination: Path,
    frame_start: int,
    frame_end: int,
    role: str,
    collection_id: str,
) -> dict:
    frames, quaternion_error = inspect_motion(source)
    if not 0 <= frame_start <= frame_end < frames:
        raise ValueError(f"invalid crop for {scene_id}/{candidate_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    if sha256(destination) != source_hash:
        raise ValueError(f"motion copy changed: {scene_id}/{candidate_id}")
    return {
        "motion_id": f"{scene_id}_{candidate_id}_sit_write",
        "source_sample_id": candidate_id,
        "source_group_id": source_group_id,
        "source_collection_id": collection_id,
        "source_split_role": role,
        "action": "sit",
        "target_instance_id": target,
        "purpose_instance_id": "desk_01",
        "text": "Sit anywhere to write.",
        "motion_file": destination.relative_to(destination.parents[3]).as_posix(),
        "motion_sha256": source_hash,
        "total_frames": frames,
        "frame_start_inclusive": frame_start,
        "frame_end_inclusive": frame_end,
        "source_fps": 30,
        "columns": 103,
        "maximum_quaternion_norm_error": quaternion_error,
        "target_contact_passed": True,
        "non_target_collision_passed": True,
        "teacher_v6_usage": "motion_derived_dense_eligible",
        "teacher_v7_usage": "motion_derived_dense_eligible",
        "compatible_prompt_ids": PROMPT_IDS,
        "dense_affordance_gt_generated": False,
        "dense_affordance_gt_next_gate": "Teacher-v7 FK/contact-kernel generation required",
        "purpose_relation_used_as_forward_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--output-dataset-root", type=Path, required=True)
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--train-binding-root", type=Path, required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--development-binding-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_dataset_root.expanduser().resolve()
    output_root = args.output_dataset_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Teacher-v7 dataset: {output_root}")
    train_selection = read_json(args.train_selection)
    development_selection = read_json(args.development_selection)
    if train_selection.get("status") != "TRAIN_SELECTION_PASS_DEVELOPMENT_SOURCE_BLOCKED":
        raise ValueError("sealed hc_hd train selection changed")
    if development_selection.get("status") != "DEVELOPMENT_SELECTION_PASS":
        raise ValueError("sealed hcw_hdw development selection changed")
    if development_selection.get("raw_source_hash_overlap_with_train") != 0:
        raise ValueError("train/development source separation changed")

    output_root.mkdir(parents=True)
    shutil.copytree(source_root / "scenes", output_root / "scenes")
    shutil.copy2(source_root / "scene_split.json", output_root / "scene_split.json")
    provenance = output_root / "provenance"
    provenance.mkdir()
    for name, path in (
        ("teacher_v6_index.json", source_root / "index.json"),
        ("hc_hd_train_selection.json", args.train_selection),
        ("hcw_hdw_development_selection.json", args.development_selection),
    ):
        shutil.copy2(path, provenance / name)

    train_ids = [row["candidate_id"] for row in train_selection["train_selected"]]
    train_reports = {
        scene_id: read_json(args.train_binding_root / (
            "room_0101_audit_report.json" if scene_id == "room_0101" else "room_0102_report.json"
        ))
        for scene_id in ("room_0101", "room_0102")
    }
    development_rows = {row["candidate_id"]: row for row in development_selection["selected"]}
    scene_summaries = []
    all_new_source_groups = {}
    for scene_id in SCENES:
        scene_dir = output_root / "scenes" / scene_id
        index_path = scene_dir / "motion_index.json"
        base_index = read_json(index_path)
        if base_index.get("schema") != MOTION_INDEX_SCHEMA_V6:
            raise ValueError(f"{scene_id}: source motion schema changed")
        base_hash = sha256(index_path)
        rows = list(base_index.get("rows", []))
        dense_before = [
            row for row in rows
            if row.get("teacher_v6_usage") == "motion_derived_dense_eligible"
        ]
        if len(dense_before) != 18:
            raise ValueError(f"{scene_id}: expected 18 sealed v6 dense rows")

        additions = []
        if scene_id in train_reports:
            report_rows = {row["candidate_id"]: row for row in train_reports[scene_id]["candidates"]}
            binding_dir = args.train_binding_root / scene_id / "chair_06"
            for candidate_id in train_ids:
                audit = report_rows[candidate_id]
                source = find_one(binding_dir, candidate_id)
                additions.append(new_motion_row(
                    scene_id=scene_id,
                    target="chair_06",
                    candidate_id=candidate_id,
                    source_group_id=audit["source_group_id"],
                    source=source,
                    destination=scene_dir / "motions" / "relational_high_desk_v1" / "chair_06" / source.name,
                    frame_start=int(audit["frame_start_inclusive"]),
                    frame_end=int(audit["frame_end_inclusive"]),
                    role="train",
                    collection_id="hc_hd",
                ))
        else:
            binding_dir = args.development_binding_root / "room_0201" / "chair_02"
            for candidate_id in [row["candidate_id"] for row in development_selection["selected"]]:
                selected = development_rows[candidate_id]
                source = binding_dir / selected["motion_file"]
                additions.append(new_motion_row(
                    scene_id=scene_id,
                    target="chair_02",
                    candidate_id=candidate_id,
                    source_group_id=selected["source_group_id"],
                    source=source,
                    destination=scene_dir / "motions" / "relational_high_desk_v1" / "chair_02" / source.name,
                    frame_start=int(selected["frame_start_inclusive"]),
                    frame_end=int(selected["frame_end_inclusive"]),
                    role="development",
                    collection_id="hcw_hdw",
                ))

        if len(additions) != 6 or len({row["source_group_id"] for row in additions}) != 6:
            raise ValueError(f"{scene_id}: High-Desk source count changed")
        rows.extend(additions)
        rows.sort(key=lambda row: str(row["motion_id"]))
        if len({row["motion_id"] for row in rows}) != len(rows):
            raise ValueError(f"{scene_id}: duplicate motion IDs")
        dense_rows = [
            row for row in rows
            if row.get("teacher_v7_usage", row.get("teacher_v6_usage"))
            == "motion_derived_dense_eligible"
        ]
        counts = dict(sorted(Counter(row["target_instance_id"] for row in dense_rows).items()))
        base_index.update({
            "schema": MOTION_INDEX_SCHEMA_V7,
            "status": "HD_MOTION_BINDING_PASS",
            "authorization": "dense_contact_v2_generation_not_yet_run",
            "base_v6_motion_index_sha256": base_hash,
            "motion_count": len(rows),
            "dense_sit_eligible_count": len(dense_rows),
            "dense_sit_target_counts": counts,
            "relational_high_desk_motion_count": 6,
            "relational_high_desk_split_role": additions[0]["source_split_role"],
            "relational_high_desk_source_collection": additions[0]["source_collection_id"],
            "relation_or_distance_used_as_forward_input": False,
            "train_development_source_overlap": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        })
        write_json(index_path, base_index)
        all_new_source_groups[scene_id] = sorted(row["source_group_id"] for row in additions)
        scene_summaries.append({
            "scene_id": scene_id,
            "motion_index_sha256": sha256(index_path),
            "motion_count": len(rows),
            "dense_count": len(dense_rows),
            "target_counts": counts,
            "new_high_desk_count": 6,
        })
        print(f"[STAGE] {scene_id}: dense={len(dense_rows)} targets={counts}")

    train_groups = set(all_new_source_groups["room_0101"])
    if train_groups != set(all_new_source_groups["room_0102"]):
        raise ValueError("train scenes do not share the exact train six")
    if train_groups & set(all_new_source_groups["room_0201"]):
        raise ValueError("train/development High-Desk source overlap")
    report = {
        "schema": "relational_teacher_v7_hd_dataset_staging_v1",
        "status": "HD_DATASET_STAGING_PASS",
        "authorization": "dense_contact_v2_generation_required_before_training",
        "source_dataset_root": str(source_root),
        "source_dataset_index_sha256": sha256(source_root / "index.json"),
        "train_selection_sha256": sha256(args.train_selection),
        "development_selection_sha256": sha256(args.development_selection),
        "scene_count": 3,
        "new_motion_count": 18,
        "train_development_source_overlap": False,
        "relation_or_distance_used_as_forward_input": False,
        "scenes": scene_summaries,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failed_checks": [],
    }
    write_json(output_root / "staging_report.json", report)
    print("[HD_DATASET_STAGING_PASS] Teacher-v7 High-Desk motion bindings")
    print("[PASS] train rooms reuse hc_hd six; development uses independent hcw_hdw six")
    print("[PASS] all 18 additions retain physical crops and immutable hashes")
    print("[OK] dense-contact v2 generation authorized: True")
    print("[OK] Teacher LoRA v7 training authorized: False")
    print(f"[OK] output: {output_root}")


if __name__ == "__main__":
    main()

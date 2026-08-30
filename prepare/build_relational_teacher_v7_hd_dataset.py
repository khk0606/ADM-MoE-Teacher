#!/usr/bin/env python3
"""Build the three-scene Teacher-v7 index after dense-contact v2 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Teacher-v7 index: {output}")
    staging = read_json(root / "staging_report.json")
    if staging.get("status") != "HD_DATASET_STAGING_PASS" or staging.get("failed_checks"):
        raise ValueError("Teacher-v7 staging did not pass")
    split = read_json(root / "scene_split.json")
    train = sorted(split.get("train", []))
    development = sorted(split.get("development", []))
    if train != ["room_0101", "room_0102"] or development != ["room_0201"]:
        raise ValueError("sealed scene split changed")

    scenes = []
    total_rows = 0
    split_rows = Counter()
    for split_name, scene_ids in (("train", train), ("development", development)):
        for scene_id in scene_ids:
            relative = Path("scenes") / scene_id
            scene_dir = root / relative
            dense_path = scene_dir / "dense_contact_v2" / "index.json"
            motion_path = scene_dir / "motion_index.json"
            dense = read_json(dense_path)
            motion = read_json(motion_path)
            if dense.get("schema") != "relational_teacher_v7_hd_dense_contact_v2":
                raise ValueError(f"{scene_id}: dense schema changed")
            if dense.get("status") != "DENSE_CONTACT_V2_PASS":
                raise ValueError(f"{scene_id}: dense status changed")
            if dense.get("teacher_lora_training_authorized") is not False:
                raise ValueError(f"{scene_id}: single-scene artifact authorized training")
            if dense.get("relation_or_distance_used") is not False:
                raise ValueError(f"{scene_id}: relation/distance leakage")
            if dense.get("motion_count") != 24 or dense.get("prompt_expanded_row_count") != 48:
                raise ValueError(f"{scene_id}: dense row count changed")
            if dense.get("motion_index_sha256") != sha256(motion_path):
                raise ValueError(f"{scene_id}: dense/motion index binding changed")
            dense_rows = dense.get("rows", [])
            if len(dense_rows) != 24:
                raise ValueError(f"{scene_id}: expected 24 dense rows")
            expanded = []
            for dense_row in dense_rows:
                if dense_row.get("compatible_prompt_ids") != sorted(PROMPTS):
                    raise ValueError(f"{scene_id}: prompt invariance binding changed")
                for prompt_id, text in PROMPTS.items():
                    expanded.append({
                        "row_id": f"{scene_id}_{dense_row['motion_id']}_{prompt_id}",
                        "motion_id": dense_row["motion_id"],
                        "target_instance_id": dense_row["target_instance_id"],
                        "prompt_id": prompt_id,
                        "text": text,
                        "dense_manifest_file": dense_row["manifest_file"],
                        "dense_manifest_sha256": dense_row["manifest_sha256"],
                        "supervision": "motion_derived_dense_contact",
                    })
            if len(expanded) != 48 or len({row["row_id"] for row in expanded}) != 48:
                raise ValueError(f"{scene_id}: prompt expansion changed")
            split_rows[split_name] += len(expanded)
            total_rows += len(expanded)
            scenes.append({
                "scene_id": scene_id,
                "split": split_name,
                "points_file": (relative / "points.npz").as_posix(),
                "points_sha256": sha256(scene_dir / "points.npz"),
                "sidecar_file": (relative / "sidecar.npz").as_posix(),
                "sidecar_sha256": sha256(scene_dir / "sidecar.npz"),
                "instances_file": (relative / "instances.json").as_posix(),
                "instances_sha256": sha256(scene_dir / "instances.json"),
                "motion_index_file": (relative / "motion_index.json").as_posix(),
                "motion_index_sha256": sha256(motion_path),
                "dense_index_file": (relative / "dense_contact_v2/index.json").as_posix(),
                "dense_index_sha256": sha256(dense_path),
                "motion_count": 24,
                "prompt_expanded_row_count": 48,
                "target_counts": dense["target_counts"],
                "rows": expanded,
            })

    payload = {
        "schema": "relational_teacher_v7_hd_dataset_v1",
        "status": "DENSE_DATASET_PASS",
        "authorization": "teacher_lora_v7_cuda_preflight_only",
        "prompt_policy_id": "purpose_sit_object_agnostic_v1",
        "point_count": 8192,
        "contact_dim": 6,
        "num_scenes": 3,
        "num_dense_motions": 72,
        "num_rows": total_rows,
        "split_scene_counts": {"train": 2, "development": 1},
        "split_row_counts": dict(sorted(split_rows.items())),
        "scene_split_file": str((Path("scene_split.json")).as_posix()),
        "scene_split_sha256": sha256(root / "scene_split.json"),
        "staging_report_file": "staging_report.json",
        "staging_report_sha256": sha256(root / "staging_report.json"),
        "relation_or_distance_used_as_forward_input": False,
        "target_instance_gt_is_supervision_only": True,
        "heldout_dense_arrays_read_during_build": False,
        "teacher_lora_v7_training_authorized": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": scenes,
        "failed_checks": [],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[DENSE_DATASET_PASS] relational Teacher-v7 dataset index")
    print("[PASS] train scenes=2 development scenes=1; dense motions=72")
    print("[PASS] 72 motions x 2 object-agnostic prompts = 144 rows")
    print("[PASS] relation/distance excluded; target instance is supervision only")
    print("[OK] Teacher LoRA v7 CUDA preflight authorized: True")
    print("[OK] Teacher LoRA v7 training authorized: False")
    print(f"[OK] output: {output}")


if __name__ == "__main__":
    main()

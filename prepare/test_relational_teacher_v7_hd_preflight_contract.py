#!/usr/bin/env python3
"""CPU contract tests for the Teacher-v7 High-Desk CUDA preflight inputs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from relational_teacher_v7_hd_preflight_contract import (
    DENSE_INDEX_SCHEMA,
    canonical_sha256,
    select_train_probe,
)
from relational_teacher_v6_contract import POINT_COUNT, PROMPTS, sha256_file


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_probe(root: Path) -> tuple[dict, Path]:
    scene_id = "room_0101"
    scene = root / "scenes" / scene_id
    dense = scene / "dense_contact_v2"
    item = dense / "motions" / "room_0101_hc_hd_b_1_sit_write"
    item.mkdir(parents=True)

    points = np.zeros((POINT_COUNT, 6), dtype=np.float32)
    points[:, 0] = np.linspace(-1.0, 1.0, POINT_COUNT, dtype=np.float32)
    instance_ids = np.zeros(POINT_COUNT, dtype=np.int64)
    instance_ids[:256] = 6
    category_ids = np.zeros(POINT_COUNT, dtype=np.int64)
    category_ids[:256] = 1
    source_indices = np.arange(POINT_COUNT, dtype=np.int64)
    points_file = scene / "points.npz"
    sidecar_file = scene / "sidecar.npz"
    np.savez_compressed(points_file, points=points)
    np.savez_compressed(
        sidecar_file,
        instance_ids=instance_ids,
        category_ids=category_ids,
        source_indices=source_indices,
    )

    affordance_file = item / "full_affordance_gt.npz"
    np.savez_compressed(
        affordance_file,
        affordance=np.full((POINT_COUNT, 6), 0.25, dtype=np.float32),
        instance_ids=instance_ids,
        source_indices=source_indices,
        sigma=np.asarray(0.8, dtype=np.float32),
    )
    motion_id = "room_0101_hc_hd_b_1_sit_write"
    manifest_file = item / "manifest.json"
    write_json(
        manifest_file,
        {
            "schema": DENSE_INDEX_SCHEMA + "_item",
            "status": "DENSE_CONTACT_ITEM_PASS",
            "scene_id": scene_id,
            "motion_id": motion_id,
            "target_instance_id": "chair_06",
            "compatible_prompt_ids": sorted(PROMPTS),
            "relation_or_distance_used": False,
            "affordance_file": str(affordance_file.relative_to(root)),
            "affordance_sha256": sha256_file(affordance_file),
        },
    )
    rows = []
    for number in range(1, 7):
        rows.append(
            {
                "motion_id": f"room_0101_hc_hd_b_{number}_sit_write",
                "target_instance_id": "chair_06",
                "manifest_file": str(manifest_file.relative_to(root)),
                "manifest_sha256": sha256_file(manifest_file),
            }
        )
    for number in range(18):
        rows.append(
            {
                "motion_id": f"room_0101_legacy_{number:02d}",
                "target_instance_id": "chair_01",
                "manifest_file": str(manifest_file.relative_to(root)),
                "manifest_sha256": sha256_file(manifest_file),
            }
        )
    dense_index_file = dense / "index.json"
    write_json(
        dense_index_file,
        {
            "schema": DENSE_INDEX_SCHEMA,
            "status": "DENSE_CONTACT_V2_PASS",
            "scene_id": scene_id,
            "motion_count": 24,
            "prompt_expanded_row_count": 48,
            "relation_or_distance_used": False,
            "teacher_lora_training_authorized": False,
            "rows": rows,
        },
    )
    record = {
        "scene_id": scene_id,
        "split": "train",
        "points_file": str(points_file.relative_to(root)),
        "points_sha256": sha256_file(points_file),
        "sidecar_file": str(sidecar_file.relative_to(root)),
        "sidecar_sha256": sha256_file(sidecar_file),
        "dense_index_file": str(dense_index_file.relative_to(root)),
        "dense_index_sha256": sha256_file(dense_index_file),
    }
    return record, dense_index_file


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="teacher_v7_preflight_test.") as raw:
        root = Path(raw).resolve()
        record, dense_index_file = build_probe(root)
        probe = select_train_probe(root, record)
        assert probe["motion_id"] == "room_0101_hc_hd_b_1_sit_write"
        assert probe["target_instance_id"] == "chair_06"
        assert probe["probe_role"] == "new_high_desk_hc_hd_train_motion"

        tampered = json.loads(dense_index_file.read_text(encoding="utf-8"))
        tampered["relation_or_distance_used"] = True
        write_json(dense_index_file, tampered)
        record["dense_index_sha256"] = sha256_file(dense_index_file)
        try:
            select_train_probe(root, record)
        except ValueError:
            pass
        else:
            raise AssertionError("relation/distance tamper was accepted")

    first = canonical_sha256({"b": 2, "a": 1})
    second = canonical_sha256({"a": 1, "b": 2})
    assert first == second and len(first) == 64
    print("[PASS] Teacher-v7 High-Desk probe selection and tamper guard")
    print("[PASS] only sealed hc_hd train rows can enter CUDA preflight")
    print("[PASS] canonical binding arithmetic is order invariant")


if __name__ == "__main__":
    main()

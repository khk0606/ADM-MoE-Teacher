#!/usr/bin/env python3
"""Recompute a 24-clip relational Teacher-v7 dense-contact artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from relational_teacher_v6_contract import PROMPTS, sha256_file
from generate_relational_teacher_v7_hd_dense_contact import (
    CONTACT_JOINT_NAMES,
    CONTACT_JOINTS,
    DENSE_INDEX_SCHEMA,
    YBOT_TO_SMPL22,
    compute_distance_map,
    global_fk,
    load_motion_crop,
    load_scene,
    load_skeleton,
    read_json,
    resolve_v7_target_counts,
    resample_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--raw-scene-manifest", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    index_path = args.index.expanduser().resolve()
    index = read_json(index_path)
    if index.get("schema") != DENSE_INDEX_SCHEMA:
        raise ValueError("unsupported dense-contact index schema")
    if index.get("status") != "DENSE_CONTACT_V2_PASS":
        raise ValueError("dense-contact status is not PASS")
    if index.get("authorization") != "three_scene_validation_required_before_training":
        raise ValueError("single-scene authorization guard changed")
    if index.get("teacher_lora_training_authorized") is not False:
        raise ValueError("single scene must not authorize Teacher training")
    if index.get("relation_or_distance_used") is not False:
        raise ValueError("relation/distance leakage guard changed")
    if index.get("motion_count") != 24 or index.get("prompt_expanded_row_count") != 48:
        raise ValueError("dense-contact counts changed")
    target_counts = index.get("target_counts")
    if not isinstance(target_counts, dict):
        raise ValueError("dense target counts are missing")
    if index.get("contact_joints") != CONTACT_JOINTS.tolist():
        raise ValueError("contact joint order changed")
    if index.get("contact_joint_names") != list(CONTACT_JOINT_NAMES):
        raise ValueError("contact joint names changed")

    dense_root = index_path.parent
    scene_dir = dense_root.parent
    dataset_root = scene_dir.parent.parent
    scene_id = str(index["scene_id"])
    if scene_dir.name != scene_id:
        raise ValueError("dense index path/scene mismatch")
    skeleton_path = args.skeleton.expanduser().resolve()
    if sha256_file(skeleton_path) != str(index["skeleton_sha256"]):
        raise ValueError("skeleton hash changed")
    motion_index_path = scene_dir / "motion_index.json"
    if sha256_file(motion_index_path) != str(index["motion_index_sha256"]):
        raise ValueError("motion inventory hash changed")
    if sha256_file(scene_dir / "points.npz") != str(index["points_sha256"]):
        raise ValueError("point tensor hash changed")
    if sha256_file(scene_dir / "sidecar.npz") != str(index["sidecar_sha256"]):
        raise ValueError("point sidecar hash changed")

    raw_manifest = read_json(args.raw_scene_manifest.expanduser().resolve())
    if raw_manifest.get("scene_id") != scene_id or raw_manifest.get("coordinate_frame") != "ChairCoordinate":
        raise ValueError("raw coordinate contract changed")
    coordinate_scale = np.asarray(raw_manifest["coordinate_world_scale_xyz"], dtype=np.float32)
    np.testing.assert_allclose(
        coordinate_scale,
        np.asarray(index["coordinate_scale_unity_xyz"], dtype=np.float32),
        rtol=0.0,
        atol=1e-8,
    )

    scene_xyz, source_indices, instance_ids, stable_to_numeric = load_scene(scene_dir)
    parents, offsets = load_skeleton(skeleton_path)
    motion_index = read_json(motion_index_path)
    source_rows = {
        str(row["motion_id"]): row
        for row in motion_index.get("rows", [])
        if row.get("teacher_v7_usage", row.get("teacher_v6_usage"))
        == "motion_derived_dense_eligible"
    }
    rows = index.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("expected 24 dense rows")
    recomputed_target_counts = resolve_v7_target_counts(rows)
    if recomputed_target_counts != target_counts:
        raise ValueError("recomputed target balance changed")

    sigma = float(index["sigma"])
    for number, row in enumerate(rows, start=1):
        motion_id = str(row["motion_id"])
        if row.get("compatible_prompt_ids") != sorted(PROMPTS):
            raise ValueError("prompt-invariance binding changed")
        source = source_rows.get(motion_id)
        if source is None:
            raise ValueError(f"missing source motion row: {motion_id}")
        if str(source["target_instance_id"]) != str(row["target_instance_id"]):
            raise ValueError("dense/source target mismatch")
        manifest_path = dataset_root / str(row["manifest_file"])
        if sha256_file(manifest_path) != str(row["manifest_sha256"]):
            raise ValueError(f"dense item manifest hash changed: {motion_id}")
        manifest = read_json(manifest_path)
        if manifest.get("status") != "DENSE_CONTACT_ITEM_PASS":
            raise ValueError(f"dense item is not PASS: {motion_id}")
        if manifest.get("relation_or_distance_used") is not False:
            raise ValueError("relation/distance leaked into dense item")
        if manifest.get("compatible_prompt_ids") != sorted(PROMPTS):
            raise ValueError("dense item prompt binding changed")
        positions_path = dataset_root / str(manifest["positions_file"])
        affordance_path = dataset_root / str(manifest["affordance_file"])
        if sha256_file(positions_path) != str(manifest["positions_sha256"]):
            raise ValueError(f"position artifact hash changed: {motion_id}")
        if sha256_file(affordance_path) != str(manifest["affordance_sha256"]):
            raise ValueError(f"affordance artifact hash changed: {motion_id}")

        with np.load(positions_path, allow_pickle=False) as payload:
            positions = payload["joint_positions22_adm_chair_local_z_up"].astype(np.float32)
            if positions.ndim != 3 or positions.shape[1:] != (22, 3):
                raise ValueError(f"position shape changed: {motion_id}")
        with np.load(affordance_path, allow_pickle=False) as payload:
            required = {
                "xyz", "instance_ids", "source_indices", "distance", "affordance",
                "contact_joints", "contact_joint_names", "sigma", "gt_root_start_xyz",
            }
            if set(payload.files) != required:
                raise ValueError(f"affordance keys changed: {motion_id}")
            np.testing.assert_allclose(payload["xyz"], scene_xyz, rtol=0.0, atol=0.0)
            np.testing.assert_array_equal(payload["instance_ids"], instance_ids)
            np.testing.assert_array_equal(payload["source_indices"], source_indices)
            np.testing.assert_array_equal(payload["contact_joints"], CONTACT_JOINTS)
            np.testing.assert_array_equal(
                payload["contact_joint_names"], np.asarray(CONTACT_JOINT_NAMES)
            )
            if abs(float(payload["sigma"]) - sigma) > 1e-7:
                raise ValueError("sigma changed")
            saved_distance = payload["distance"].astype(np.float32)
            saved_affordance = payload["affordance"].astype(np.float32)

        motion_path = scene_dir / str(source["motion_file"])
        if sha256_file(motion_path) != str(source["motion_sha256"]):
            raise ValueError(f"source motion hash changed: {motion_id}")
        root, quaternions = load_motion_crop(
            motion_path,
            int(source["frame_start_inclusive"]),
            int(source["frame_end_inclusive"]),
        )
        positions25, bone_error = global_fk(
            root, quaternions, parents, offsets, coordinate_scale
        )
        if bone_error > 1e-4:
            raise ValueError(f"recomputed FK bone error: {motion_id}")
        source_positions = np.ascontiguousarray(
            positions25[:, YBOT_TO_SMPL22][..., [0, 2, 1]], dtype=np.float32
        )
        recomputed_positions, _, _ = resample_positions(
            source_positions,
            float(manifest["source_fps"]),
            float(manifest["target_fps"]),
        )
        np.testing.assert_allclose(positions, recomputed_positions, rtol=0.0, atol=1e-6)
        recomputed_distance = compute_distance_map(scene_xyz, positions, args.chunk_size)
        np.testing.assert_allclose(saved_distance, recomputed_distance, rtol=0.0, atol=1e-6)
        recomputed_affordance = np.exp(-0.5 * (recomputed_distance / sigma) ** 2).astype(np.float32)
        np.testing.assert_allclose(saved_affordance, recomputed_affordance, rtol=0.0, atol=1e-7)
        target_numeric = stable_to_numeric[str(row["target_instance_id"])]
        target_max = float(saved_affordance[instance_ids == target_numeric, 0].max())
        if target_max < 0.80 or abs(target_max - float(row["target_pelvis_affordance_max"])) > 1e-7:
            raise ValueError(f"target pelvis response changed: {motion_id}")
        print(f"[VERIFY {number:02d}/24] {motion_id} T={positions.shape[0]}")

    print("[DENSE_CONTACT_V2_PASS] relational Teacher-v7 dense-contact integrity")
    print("[PASS] 24 raw TXT -> YBot FK -> HumanML22 recomputation")
    print("[PASS] 24x8192x6 distance and sigma-0.8 affordance recomputation")
    print(f"[PASS] target counts {target_counts}; two prompts per motion")
    print("[PASS] relation/distance leakage absent; production training unauthorized")


if __name__ == "__main__":
    main()

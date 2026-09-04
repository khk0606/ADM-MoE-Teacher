#!/usr/bin/env python3
"""Synthetic end-to-end build/validate/tamper test for Teacher-v9."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from relational_teacher_v9_all_sittable_contract import (
    PROMPTS,
    atomic_savez,
    atomic_write_json,
    read_json,
    sha256_file,
)
from visualize_relational_teacher_v9_all_sittable_viser import load_viewer_dataset


HERE = Path(__file__).resolve().parent
POINTS = 8192


SCENES = {
    "room_0101": ("train", "chair_01", "chair_06", "hc_hd"),
    "room_0102": ("train", "chair_05", "chair_06", "hc_hd"),
    "room_0201": ("development", "chair_03", "chair_02", "hcw_hdw"),
}
TRAIN_SCENES = {"room_0101", "room_0102"}


def run(command, expect_success=True):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(HERE)
    result = subprocess.run(
        [sys.executable] + [str(value) for value in command],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            "command failed:\n" + result.stdout + "\n" + result.stderr
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError("tampered artifact unexpectedly validated")
    return result


def expect_failure(callable_, message):
    try:
        callable_()
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    raise AssertionError(message)


def scene_arrays():
    xyz = np.zeros((POINTS, 3), dtype=np.float32)
    xyz[:, 0] = np.linspace(-4.0, 4.0, POINTS, dtype=np.float32)
    xyz[:, 1] = np.sin(np.linspace(0.0, 8.0, POINTS, dtype=np.float32))
    points = np.concatenate(
        [xyz, np.full((POINTS, 3), 0.5, dtype=np.float32)], axis=1
    )
    instance_ids = np.zeros(POINTS, dtype=np.int64)
    category_ids = np.zeros(POINTS, dtype=np.int64)
    blocks = (
        (1, 2, 1000, 2100),
        (2, 1, 2100, 2700),
        (3, 1, 2700, 3300),
        (4, 1, 3300, 3800),
        (5, 4, 3800, 4000),
        (6, 5, 4000, 4200),
        (7, 3, 4200, 4400),
    )
    for instance_id, category_id, start, end in blocks:
        instance_ids[start:end] = instance_id
        category_ids[start:end] = category_id
    return points, xyz, instance_ids, category_ids


def make_affordance(instance_ids, target_numeric, source_number):
    value = np.zeros((POINTS, 6), dtype=np.float32)
    environment = instance_ids == 0
    value[environment, 1] = np.float32(0.31 + source_number * 0.01)
    target = instance_ids == target_numeric
    value[target, :] = np.float32(0.82 + source_number * 0.02)
    # Must be removed by the target+environment filter.
    value[instance_ids == 4, :] = 0.97
    value[np.isin(instance_ids, [5, 6, 7]), :] = 0.99
    return value


def build_source(root):
    points, xyz, instance_ids, category_ids = scene_arrays()
    scene_records = []
    for scene_id, (split, normal_name, high_name, collection) in SCENES.items():
        scene = root / "scenes" / scene_id
        points_path = scene / "points.npz"
        sidecar_path = scene / "sidecar.npz"
        instances_path = scene / "instances.json"
        motion_index_path = scene / "motion_index.json"
        dense_index_path = scene / "dense_contact_v2" / "index.json"
        atomic_savez(points_path, points=points)
        atomic_savez(
            sidecar_path,
            xyz_afford_z_up=xyz,
            source_indices=np.arange(POINTS, dtype=np.int64),
            instance_ids=instance_ids,
            category_ids=category_ids,
        )
        atomic_write_json(
            instances_path,
            {
                "schema": "relational_affordance_unity_instances_v1",
                "scene_id": scene_id,
                "objects": [
                    {"name": "bed_01", "instance_id": 1, "category": "bed", "category_id": 2, "anchor_unity_xz": [0.0, 0.0]},
                    {"name": normal_name, "instance_id": 2, "category": "chair", "category_id": 1, "anchor_unity_xz": [1.0, 0.0]},
                    {"name": high_name, "instance_id": 3, "category": "chair", "category_id": 1, "anchor_unity_xz": [2.0, 0.0]},
                    {"name": "chair_99", "instance_id": 4, "category": "chair", "category_id": 1, "anchor_unity_xz": [3.0, 0.0]},
                    {"name": "tv_01", "instance_id": 5, "category": "tv", "category_id": 4, "anchor_unity_xz": [4.0, 0.0]},
                    {"name": "desk_01", "instance_id": 6, "category": "desk", "category_id": 5, "anchor_unity_xz": [5.0, 0.0]},
                    {"name": "whiteboard_01", "instance_id": 7, "category": "whiteboard", "category_id": 3},
                ],
            },
        )

        motion_rows = []
        dense_rows = []
        specifications = (
            ("bed", "bed_01", 1, ""),
            ("normal", normal_name, 2, ""),
            ("high", high_name, 3, collection),
            ("legacy", high_name, 3, ""),
        )
        for group, target_name, target_numeric, source_collection in specifications:
            for number in range(6):
                if group == "high":
                    motion_id = f"{scene_id}_{collection}_recording_{number}_sit_write"
                else:
                    motion_id = f"{scene_id}_{group}_recording_{number}_sit"
                motion_row = {
                    "motion_id": motion_id,
                    "target_instance_id": target_name,
                }
                if source_collection:
                    motion_row.update(
                        {
                            "source_sample_id": f"{source_collection}_{number}",
                            "source_group_id": f"{source_collection}_group_{number}",
                            "source_collection_id": source_collection,
                            "source_split_role": split,
                        }
                    )
                motion_rows.append(motion_row)
                item = scene / "dense_contact_v2" / "motions" / motion_id
                motion_path = scene / "motions" / (motion_id + ".txt")
                motion_path.parent.mkdir(parents=True, exist_ok=True)
                if source_collection == "hc_hd":
                    # The same raw hc_hd recording is placed in two different
                    # scenes.  Its source identity is shared, but the exported
                    # scene-local TXT bytes and hashes must be allowed to differ.
                    marker = 1000 * (tuple(SCENES).index(scene_id) + 1) + 100 + number
                elif source_collection == "hcw_hdw":
                    marker = 1000 * (tuple(SCENES).index(scene_id) + 1) + 200 + number
                else:
                    marker = (
                        1000 * (tuple(SCENES).index(scene_id) + 1)
                        + 100 * tuple(value[0] for value in specifications).index(group)
                        + number
                    )
                motion_path.write_text(
                    " ".join([str(marker)] + ["0"] * 102) + "\n",
                    encoding="utf-8",
                )
                motion_row["motion_file"] = motion_path.relative_to(scene).as_posix()
                motion_row["motion_sha256"] = sha256_file(motion_path)
                positions_path = item / "joint_positions22.npz"
                atomic_savez(
                    positions_path,
                    joint_positions22_adm_chair_local_z_up=np.zeros(
                        (2, 22, 3), dtype=np.float32
                    ),
                    source_times_s=np.asarray((0.0, 1.0 / 30.0), dtype=np.float32),
                    target_times_s=np.asarray((0.0, 1.0 / 20.0), dtype=np.float32),
                    source_frame_start_inclusive=np.asarray(0, dtype=np.int64),
                    source_frame_end_inclusive=np.asarray(1, dtype=np.int64),
                    source_fps=np.asarray(30.0, dtype=np.float32),
                    target_fps=np.asarray(20.0, dtype=np.float32),
                )
                affordance_path = item / "full_affordance_gt.npz"
                affordance = make_affordance(instance_ids, target_numeric, number)
                distance = np.sqrt(
                    np.maximum(-2.0 * np.float32(0.8) ** 2 * np.log(np.maximum(affordance, 1e-12)), 0.0)
                ).astype(np.float32)
                atomic_savez(
                    affordance_path,
                    xyz=xyz,
                    instance_ids=instance_ids,
                    source_indices=np.arange(POINTS, dtype=np.int64),
                    distance=distance,
                    affordance=affordance,
                    contact_joints=np.arange(6, dtype=np.int64),
                    contact_joint_names=np.asarray(
                        ("pelvis", "left_foot", "right_foot", "neck", "left_wrist", "right_wrist")
                    ),
                    sigma=np.asarray(0.8, dtype=np.float32),
                    gt_root_start_xyz=np.zeros(3, dtype=np.float32),
                )
                manifest_path = item / "manifest.json"
                atomic_write_json(
                    manifest_path,
                    {
                        "schema": "relational_teacher_v7_hd_dense_contact_v2_item",
                        "status": "DENSE_CONTACT_ITEM_PASS",
                        "scene_id": scene_id,
                        "motion_id": motion_id,
                        "target_instance_id": target_name,
                        "target_numeric_instance_id": target_numeric,
                        "source_motion_file": motion_path.relative_to(root).as_posix(),
                        "source_motion_sha256": sha256_file(motion_path),
                        "positions_file": positions_path.relative_to(root).as_posix(),
                        "positions_sha256": sha256_file(positions_path),
                        "relation_or_distance_used": False,
                        "compatible_prompt_ids": sorted(PROMPTS),
                        "affordance_file": affordance_path.relative_to(root).as_posix(),
                        "affordance_sha256": sha256_file(affordance_path),
                    },
                )
                dense_rows.append(
                    {
                        "motion_id": motion_id,
                        "target_instance_id": target_name,
                        "manifest_file": manifest_path.relative_to(root).as_posix(),
                        "manifest_sha256": sha256_file(manifest_path),
                    }
                )
        atomic_write_json(
            motion_index_path,
            {
                "schema": "relational_teacher_v7_hd_motion_inventory_v1",
                "status": "HD_MOTION_BINDING_PASS",
                "relation_or_distance_used_as_forward_input": False,
                "rows": motion_rows,
            },
        )
        atomic_write_json(
            dense_index_path,
            {
                "schema": "relational_teacher_v7_hd_dense_contact_v2",
                "status": "DENSE_CONTACT_V2_PASS",
                "scene_id": scene_id,
                "relation_or_distance_used": False,
                "motion_count": 24,
                "prompt_expanded_row_count": 48,
                "rows": sorted(dense_rows, key=lambda row: row["motion_id"]),
            },
        )
        scene_records.append(
            {
                "scene_id": scene_id,
                "split": split,
                "points_file": points_path.relative_to(root).as_posix(),
                "points_sha256": sha256_file(points_path),
                "sidecar_file": sidecar_path.relative_to(root).as_posix(),
                "sidecar_sha256": sha256_file(sidecar_path),
                "instances_file": instances_path.relative_to(root).as_posix(),
                "instances_sha256": sha256_file(instances_path),
                "motion_index_file": motion_index_path.relative_to(root).as_posix(),
                "motion_index_sha256": sha256_file(motion_index_path),
                "dense_index_file": dense_index_path.relative_to(root).as_posix(),
                "dense_index_sha256": sha256_file(dense_index_path),
            }
        )
    index_path = root / "index.json"
    atomic_write_json(
        index_path,
        {
            "schema": "relational_teacher_v7_hd_dataset_v1",
            "status": "DENSE_DATASET_PASS",
            "authorization": "teacher_lora_v7_cuda_preflight_only",
            "prompt_policy_id": "purpose_sit_object_agnostic_v1",
            "point_count": 8192,
            "contact_dim": 6,
            "num_scenes": 3,
            "relation_or_distance_used_as_forward_input": False,
            "target_instance_gt_is_supervision_only": True,
            "heldout_dense_arrays_read_during_build": False,
            "teacher_lora_v7_training_authorized": False,
            "num_dense_motions": 72,
            "num_rows": 144,
            "split_scene_counts": {"train": 2, "development": 1},
            "split_row_counts": {"train": 96, "development": 48},
            "scenes": scene_records,
            "failed_checks": [],
        },
    )
    return index_path


def main():
    with tempfile.TemporaryDirectory(prefix="teacher_v9_roundtrip.") as directory:
        temporary = Path(directory)
        source = temporary / "v7"
        output = temporary / "v9"
        source.mkdir()
        source_index = build_source(source)
        train_high_rows = []
        for scene_id in ("room_0101", "room_0102"):
            motion_index = read_json(source / "scenes" / scene_id / "motion_index.json")
            train_high_rows.append(
                sorted(
                    (
                        str(row["source_group_id"]),
                        str(row["source_sample_id"]),
                        str(row["motion_sha256"]),
                    )
                    for row in motion_index["rows"]
                    if row.get("source_collection_id") == "hc_hd"
                )
            )
        if [row[:2] for row in train_high_rows[0]] != [
            row[:2] for row in train_high_rows[1]
        ]:
            raise AssertionError("synthetic train source identities do not match")
        if [row[2] for row in train_high_rows[0]] == [
            row[2] for row in train_high_rows[1]
        ]:
            raise AssertionError("scene-local train motion hashes unexpectedly match")
        builder = HERE / "build_relational_teacher_v9_all_sittable_dataset.py"
        validator = HERE / "validate_relational_teacher_v9_all_sittable_dataset.py"
        result = run(
            [
                builder,
                "--source-dataset-root",
                source,
                "--source-index",
                source_index,
                "--output-root",
                output,
            ]
        )
        if "[ALL_SITTABLE_DATASET_PASS]" not in result.stdout:
            raise AssertionError("builder PASS marker absent")
        result = run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output / "index.json",
            ]
        )
        if "[ALL_SITTABLE_DATASET_PASS]" not in result.stdout:
            raise AssertionError("validator PASS marker absent")
        viewer_scenes = load_viewer_dataset(source, output, output / "index.json")
        if set(viewer_scenes) != TRAIN_SCENES:
            raise AssertionError("strict viewer loader lost a scene")

        tampered = output / "scenes" / "room_0101" / "replicas" / "replica_00.npz"
        original_replica = tampered.read_bytes()
        with np.load(tampered, allow_pickle=False) as arrays:
            value = arrays["affordance"].copy()
        value[0, 0] = np.float32(min(1.0, float(value[0, 0]) + 0.01))
        np.savez_compressed(tampered, affordance=value)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output / "index.json",
            ],
            expect_success=False,
        )
        tampered.write_bytes(original_replica)

        primary = output / "scenes" / "room_0101" / "all_sittable_consensus.npz"
        original_primary = primary.read_bytes()
        with np.load(primary, allow_pickle=False) as arrays:
            payload = {key: arrays[key].copy() for key in arrays.files}
        payload["all_sittable_affordance"][1000, 0] = np.float32(0.0)
        np.savez_compressed(primary, **payload)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output / "index.json",
            ],
            expect_success=False,
        )
        primary.write_bytes(original_primary)

        output_index_path = output / "index.json"
        original_output_index = output_index_path.read_bytes()
        output_index_value = read_json(output_index_path)
        output_index_value["split_scene_counts"] = {"train": 1}
        atomic_write_json(output_index_path, output_index_value)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output_index_path,
            ],
            expect_success=False,
        )
        output_index_path.write_bytes(original_output_index)

        output_index_value = read_json(output_index_path)
        output_index_value["scenes"][0]["rows"][0]["row_id"] = "duplicate_or_wrong"
        atomic_write_json(output_index_path, output_index_value)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output_index_path,
            ],
            expect_success=False,
        )
        expect_failure(
            lambda: load_viewer_dataset(source, output, output_index_path),
            "viewer accepted a forged primary row ID",
        )
        output_index_path.write_bytes(original_output_index)

        output_index_value = read_json(output_index_path)
        scene_record = next(
            row for row in output_index_value["scenes"] if row["scene_id"] == "room_0101"
        )
        scene_record["verified_target_instances"] = [
            "bed_01",
            "chair_01",
            "chair_99",
        ]
        atomic_write_json(output_index_path, output_index_value)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output_index_path,
            ],
            expect_success=False,
        )
        output_index_path.write_bytes(original_output_index)

        scene_manifest_path = output / "scenes" / "room_0101" / "manifest.json"
        original_scene_manifest = scene_manifest_path.read_bytes()
        scene_manifest = read_json(scene_manifest_path)
        scene_manifest["group_consensus"]["bed"]["target_instance_id"] = "chair_01"
        atomic_write_json(scene_manifest_path, scene_manifest)
        output_index_value = read_json(output_index_path)
        scene_record = next(
            row for row in output_index_value["scenes"] if row["scene_id"] == "room_0101"
        )
        scene_record["manifest_sha256"] = sha256_file(scene_manifest_path)
        atomic_write_json(output_index_path, output_index_value)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output_index_path,
            ],
            expect_success=False,
        )
        scene_manifest_path.write_bytes(original_scene_manifest)
        output_index_path.write_bytes(original_output_index)

        original_source_index = source_index.read_bytes()
        source_index_value = read_json(source_index)
        source_index_value["status"] = "TAMPERED"
        atomic_write_json(source_index, source_index_value)
        output_index_value = read_json(output_index_path)
        output_index_value["source_index_sha256"] = sha256_file(source_index)
        atomic_write_json(output_index_path, output_index_value)
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output_index_path,
            ],
            expect_success=False,
        )
        try:
            load_viewer_dataset(source, output, output_index_path)
        except ValueError:
            pass
        else:
            raise AssertionError("viewer accepted a tampered source top contract")
        source_index.write_bytes(original_source_index)
        output_index_path.write_bytes(original_output_index)

        # Preserve all outer hashes while changing only the motion-inventory
        # target.  The dense row/item still names the original target, so the
        # builder must reject this cross-record target mismatch explicitly.
        motion_index_path = source / "scenes" / "room_0101" / "motion_index.json"
        original_motion_index = motion_index_path.read_bytes()
        original_source_index = source_index.read_bytes()
        motion_index = read_json(motion_index_path)
        high_row = next(
            row
            for row in motion_index["rows"]
            if row.get("source_collection_id") == "hc_hd"
        )
        high_row["target_instance_id"] = "chair_01"
        atomic_write_json(motion_index_path, motion_index)
        source_index_value = read_json(source_index)
        source_scene = next(
            row for row in source_index_value["scenes"] if row["scene_id"] == "room_0101"
        )
        source_scene["motion_index_sha256"] = sha256_file(motion_index_path)
        atomic_write_json(source_index, source_index_value)
        run(
            [
                builder,
                "--source-dataset-root",
                source,
                "--source-index",
                source_index,
                "--output-root",
                temporary / "v9_target_tamper",
            ],
            expect_success=False,
        )
        motion_index_path.write_bytes(original_motion_index)
        source_index.write_bytes(original_source_index)

        pairing_index_path = source / "scenes" / "room_0102" / "motion_index.json"
        original_pairing_index = pairing_index_path.read_bytes()
        original_source_index = source_index.read_bytes()
        pairing_index = read_json(pairing_index_path)
        high_rows = [
            row
            for row in pairing_index["rows"]
            if row.get("source_collection_id") == "hc_hd"
        ]
        high_rows[0]["source_sample_id"], high_rows[1]["source_sample_id"] = (
            high_rows[1]["source_sample_id"],
            high_rows[0]["source_sample_id"],
        )
        atomic_write_json(pairing_index_path, pairing_index)
        source_index_value = read_json(source_index)
        source_scene = next(
            row for row in source_index_value["scenes"] if row["scene_id"] == "room_0102"
        )
        source_scene["motion_index_sha256"] = sha256_file(pairing_index_path)
        atomic_write_json(source_index, source_index_value)
        run(
            [
                builder,
                "--source-dataset-root",
                source,
                "--source-index",
                source_index,
                "--output-root",
                temporary / "v9_identity_pair_tamper",
            ],
            expect_success=False,
        )
        pairing_index_path.write_bytes(original_pairing_index)
        source_index.write_bytes(original_source_index)

        source_motion = sorted((source / "scenes" / "room_0101" / "motions").glob("*.txt"))[0]
        source_motion.write_text(
            source_motion.read_text(encoding="utf-8") + "0\n", encoding="utf-8"
        )
        run(
            [
                validator,
                "--source-dataset-root",
                source,
                "--dataset-root",
                output,
                "--index",
                output / "index.json",
            ],
            expect_success=False,
        )

    print("[PASS] synthetic v7 -> all-sittable v9 build/validate round trip")
    print("[PASS] 2 train scenes, 48 source maps, one primary GT per scene")
    print("[PASS] strict viewer loader accepts both train scene bundles")
    print("[PASS] shared train source IDs allow distinct scene-local TXT hashes")
    print("[PASS] room_0201 metadata is bound while development arrays stay unread")
    print(
        "[PASS] count, row-ID, target, replica, primary and source tampering are rejected"
    )


if __name__ == "__main__":
    main()

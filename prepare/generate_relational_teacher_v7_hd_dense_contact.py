#!/usr/bin/env python3
"""Generate genuine six-channel Teacher-v7 GT for 24 validated Sit clips.

The implementation reproduces the sealed history-affordance contract:

* 103 columns = root XYZ + 25 global Unity-order XYZW quaternions;
* YBot global FK in the canonical ChairCoordinate scale;
* Unity Y-up XYZ -> AMDM Z-up XZY;
* YBot 25 -> HumanML/SMPL 22 joint mapping;
* 30 FPS -> 20 FPS position interpolation, without silent cropping;
* six full-trajectory distance channels;
* ``exp(-0.5 * (distance / sigma) ** 2)`` with sigma 0.8.

Relation labels, purpose-object distances, and nearest-object choices never
enter this computation. Every eligible Sit motion remains compatible with both
object-agnostic relation prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from relational_teacher_v6_contract import PROMPTS, atomic_write_json, sha256_file
from stage_relational_teacher_v7_hd_dataset import MOTION_INDEX_SCHEMA_V7


DENSE_INDEX_SCHEMA = "relational_teacher_v7_hd_dense_contact_v2"
RAW_MANIFEST_SCHEMA = "relational_affordance_scene_export_v1"
CONTACT_JOINTS = np.asarray([0, 10, 11, 12, 20, 21], dtype=np.int64)
CONTACT_JOINT_NAMES = (
    "pelvis",
    "left_foot",
    "right_foot",
    "neck",
    "left_wrist",
    "right_wrist",
)
EXPECTED_PARENTS = np.asarray(
    [
        -1, 0, 1, 2, 3, 4,
        0, 6, 7, 8, 9,
        0, 11, 12,
        13, 14, 15, 16,
        13, 18, 19,
        13, 21, 22, 23,
    ],
    dtype=np.int64,
)
YBOT_TO_SMPL22 = np.asarray(
    [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 18, 14, 21, 19, 15, 22, 16, 23, 17, 24],
    dtype=np.int64,
)


def resolve_v7_target_counts(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    counts = dict(sorted(Counter(str(row["target_instance_id"]) for row in rows).items()))
    chair_targets = sorted(target for target in counts if target.startswith("chair_"))
    if len(rows) != 24 or len(chair_targets) != 2 or counts.get("bed_01") != 6:
        raise ValueError(f"expected 24 rows over two Chairs plus bed_01: {counts}")
    if sorted(counts.values()) != [6, 6, 12] or len(counts) != 3:
        raise ValueError(f"Teacher-v7 target-count contract changed: {counts}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", default="room_0101")
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--raw-scene-manifest", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--min-horizon", type=int, default=24)
    parser.add_argument("--max-horizon", type=int, default=196)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix="." + path.name + ".", suffix=".npz", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_skeleton(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 25:
        raise ValueError(f"expected 25 skeleton rows, got {len(rows)}")
    indices = np.asarray([int(row["index"]) for row in rows], dtype=np.int64)
    parents = np.asarray([int(row["parent_index"]) for row in rows], dtype=np.int64)
    rest_world = np.asarray(
        [[float(row["world_x"]), float(row["world_y"]), float(row["world_z"])] for row in rows],
        dtype=np.float32,
    )
    if not np.array_equal(indices, np.arange(25, dtype=np.int64)):
        raise ValueError("skeleton indices must be exactly 0..24")
    if not np.array_equal(parents, EXPECTED_PARENTS):
        raise ValueError("YBot parent hierarchy changed")
    if not np.isfinite(rest_world).all():
        raise ValueError("skeleton contains NaN/Inf")
    offsets = np.zeros_like(rest_world)
    for joint, parent in enumerate(parents):
        if parent >= 0:
            offsets[joint] = rest_world[joint] - rest_world[parent]
    if float(np.linalg.norm(offsets[1:], axis=-1).min()) < 1e-6:
        raise ValueError("skeleton contains a zero-length non-root bone")
    return parents, offsets


def load_motion_crop(path: Path, frame_start: int, frame_end: int) -> Tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != 103:
        raise ValueError(f"{path}: expected [T,103], got {table.shape}")
    if not np.isfinite(table).all():
        raise ValueError(f"{path}: motion contains NaN/Inf")
    if not 0 <= frame_start <= frame_end < table.shape[0]:
        raise ValueError(f"{path}: invalid crop {frame_start}..{frame_end}")
    crop = np.ascontiguousarray(table[frame_start:frame_end + 1])
    root = crop[:, 0:3]
    quaternions = crop[:, 3:].reshape(crop.shape[0], 25, 4)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if float(norms.min()) < 1e-8:
        raise ValueError(f"{path}: zero-length quaternion")
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    if maximum_error > 1e-4:
        raise ValueError(f"{path}: quaternion norm error {maximum_error}")
    quaternions = np.ascontiguousarray(quaternions / norms, dtype=np.float32)
    return root, quaternions


def rotate_vectors_xyzw(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q_xyz = quaternions[..., 0:3]
    q_w = quaternions[..., 3:4]
    twice_cross = 2.0 * np.cross(q_xyz, vectors)
    return vectors + q_w * twice_cross + np.cross(q_xyz, twice_cross)


def global_fk(
    root: np.ndarray,
    global_quaternions: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
    coordinate_scale: np.ndarray,
) -> Tuple[np.ndarray, float]:
    if coordinate_scale.shape != (3,) or np.any(np.abs(coordinate_scale) < 1e-8):
        raise ValueError("invalid canonical coordinate scale")
    positions = np.zeros((root.shape[0], 25, 3), dtype=np.float32)
    positions[:, 0] = root
    maximum_bone_error = 0.0
    for joint in range(1, 25):
        parent = int(parents[joint])
        offset = np.broadcast_to(offsets[joint], (root.shape[0], 3)).astype(np.float32)
        rotated = rotate_vectors_xyzw(global_quaternions[:, parent], offset)
        scaled = rotated / coordinate_scale[None, :]
        positions[:, joint] = positions[:, parent] + scaled
        restored_lengths = np.linalg.norm(scaled * coordinate_scale[None, :], axis=-1)
        expected_length = float(np.linalg.norm(offsets[joint]))
        maximum_bone_error = max(
            maximum_bone_error,
            float(np.max(np.abs(restored_lengths - expected_length))),
        )
    if not np.isfinite(positions).all():
        raise ValueError("FK produced NaN/Inf")
    return positions, maximum_bone_error


def resample_positions(
    positions: np.ndarray, source_fps: float, target_fps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if positions.shape[0] < 2 or source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("invalid motion/FPS for resampling")
    duration = float((positions.shape[0] - 1) / source_fps)
    target_count = int(round(duration * target_fps)) + 1
    source_times = np.arange(positions.shape[0], dtype=np.float64) / source_fps
    target_times = np.linspace(0.0, duration, target_count, dtype=np.float64)
    source_flat = positions.reshape(positions.shape[0], -1)
    target_flat = np.empty((target_count, source_flat.shape[1]), dtype=np.float32)
    for feature in range(source_flat.shape[1]):
        target_flat[:, feature] = np.interp(target_times, source_times, source_flat[:, feature])
    return (
        np.ascontiguousarray(target_flat.reshape(target_count, 22, 3)),
        source_times,
        target_times,
    )


def compute_distance_map(
    scene_xyz: np.ndarray,
    motion_xyz: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    result = np.empty((scene_xyz.shape[0], 6), dtype=np.float32)
    for channel, joint in enumerate(CONTACT_JOINTS):
        trajectory = motion_xyz[:, int(joint), :]
        for start in range(0, scene_xyz.shape[0], chunk_size):
            end = min(start + chunk_size, scene_xyz.shape[0])
            displacement = scene_xyz[start:end, None, :] - trajectory[None, :, :]
            result[start:end, channel] = np.linalg.norm(displacement, axis=-1).min(axis=1)
    return result


def load_scene(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    with np.load(scene_dir / "points.npz", allow_pickle=False) as source:
        if set(source.files) != {"points"}:
            raise ValueError("points.npz forward contract changed")
        points = source["points"].astype(np.float32)
    with np.load(scene_dir / "sidecar.npz", allow_pickle=False) as source:
        xyz = source["xyz_afford_z_up"].astype(np.float32)
        source_indices = source["source_indices"].astype(np.int64)
        instance_ids = source["instance_ids"].astype(np.int64)
    if points.shape != (8192, 6) or xyz.shape != (8192, 3):
        raise ValueError("prepared scene shape changed")
    if not np.allclose(points[:, :3], xyz, rtol=0.0, atol=1e-6):
        raise ValueError("points/sidecar point order changed")
    if source_indices.shape != (8192,) or len(np.unique(source_indices)) != 8192:
        raise ValueError("source point indices are not unique")
    if instance_ids.shape != (8192,):
        raise ValueError("instance ID shape changed")
    instances = read_json(scene_dir / "instances.json")
    stable_to_numeric = {
        str(row["name"]): int(row["instance_id"])
        for row in instances.get("objects", [])
    }
    return xyz, source_indices, instance_ids, stable_to_numeric


def main() -> None:
    args = parse_args()
    if args.sigma <= 0.0 or args.chunk_size <= 0:
        raise ValueError("sigma and chunk-size must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    scene_id = str(args.scene_id)
    scene_dir = dataset_root / "scenes" / scene_id
    output_root = scene_dir / "dense_contact_v2"
    output_index = output_root / "index.json"
    if output_index.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite dense index: {output_index}")

    motion_index_path = scene_dir / "motion_index.json"
    motion_index = read_json(motion_index_path)
    if motion_index.get("schema") != MOTION_INDEX_SCHEMA_V7:
        raise ValueError("unsupported motion index schema")
    if motion_index.get("status") != "HD_MOTION_BINDING_PASS":
        raise ValueError("motion inventory is not PASS")
    if motion_index.get("relation_or_distance_used_as_forward_input") is not False:
        raise ValueError("relation/distance leakage guard changed")
    eligible = [
        row for row in motion_index.get("rows", [])
        if row.get("teacher_v7_usage", row.get("teacher_v6_usage"))
        == "motion_derived_dense_eligible"
    ]
    if len(eligible) != 24:
        raise ValueError("expected exactly 24 Sit dense-eligible rows")
    target_counts = resolve_v7_target_counts(eligible)
    if motion_index.get("dense_sit_target_counts") != target_counts:
        raise ValueError("motion-index Sit target balance changed")

    preparation_report = read_json(scene_dir / "preparation_report.json")
    if preparation_report.get("authorization") != "single_scene_exporter_canary_only":
        raise ValueError("scene canary authorization changed")
    raw_manifest_path = args.raw_scene_manifest.expanduser().resolve()
    raw_manifest = read_json(raw_manifest_path)
    if raw_manifest.get("schema") != RAW_MANIFEST_SCHEMA or raw_manifest.get("scene_id") != scene_id:
        raise ValueError("raw scene manifest contract mismatch")
    expected_raw_hash = preparation_report.get("raw_checksums", {}).get(
        "scene_manifest_relational_v1.json"
    )
    if sha256_file(raw_manifest_path) != expected_raw_hash:
        raise ValueError("raw scene manifest hash differs from scene preparation")
    if raw_manifest.get("coordinate_frame") != "ChairCoordinate":
        raise ValueError("motion and scene are not in ChairCoordinate")
    coordinate_scale = np.asarray(raw_manifest["coordinate_world_scale_xyz"], dtype=np.float32)

    skeleton_path = args.skeleton.expanduser().resolve()
    parents, offsets = load_skeleton(skeleton_path)
    scene_xyz, source_indices, instance_ids, stable_to_numeric = load_scene(scene_dir)
    rows: List[dict] = []
    for number, row in enumerate(eligible, start=1):
        motion_path = scene_dir / str(row["motion_file"])
        if sha256_file(motion_path) != str(row["motion_sha256"]):
            raise ValueError(f"motion hash changed: {motion_path}")
        root, quaternions = load_motion_crop(
            motion_path,
            int(row["frame_start_inclusive"]),
            int(row["frame_end_inclusive"]),
        )
        positions25, bone_error = global_fk(
            root, quaternions, parents, offsets, coordinate_scale
        )
        if bone_error > 1e-4:
            raise ValueError(f"FK bone error too large: {motion_path}")
        positions22_source = np.ascontiguousarray(
            positions25[:, YBOT_TO_SMPL22][..., [0, 2, 1]], dtype=np.float32
        )
        positions22, source_times, target_times = resample_positions(
            positions22_source, args.source_fps, args.target_fps
        )
        if not args.min_horizon <= positions22.shape[0] <= args.max_horizon:
            raise ValueError(
                f"{motion_path}: resampled frames {positions22.shape[0]} outside "
                f"[{args.min_horizon},{args.max_horizon}]"
            )
        distance = compute_distance_map(scene_xyz, positions22, args.chunk_size)
        affordance = np.exp(-0.5 * (distance / args.sigma) ** 2).astype(np.float32)
        if distance.shape != (8192, 6) or affordance.shape != (8192, 6):
            raise AssertionError("dense contact shape changed")
        if not np.isfinite(distance).all() or not np.isfinite(affordance).all():
            raise ValueError("dense contact contains NaN/Inf")
        if float(distance.min()) < 0.0 or float(affordance.min()) < 0.0 or float(affordance.max()) > 1.0 + 1e-6:
            raise ValueError("dense contact range changed")

        target_stable = str(row["target_instance_id"])
        if target_stable not in stable_to_numeric:
            raise ValueError(f"unknown target instance: {target_stable}")
        target_numeric = stable_to_numeric[target_stable]
        target_mask = instance_ids == target_numeric
        if not np.any(target_mask):
            raise ValueError(f"target has no sampled points: {target_stable}")
        target_pelvis_max = float(affordance[target_mask, 0].max())
        if target_pelvis_max < 0.80:
            raise ValueError(
                f"{row['motion_id']}: target pelvis response too low: {target_pelvis_max}"
            )

        motion_id = str(row["motion_id"])
        item_dir = output_root / "motions" / motion_id
        if item_dir.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite dense item: {item_dir}")
        item_dir.mkdir(parents=True, exist_ok=True)
        positions_file = item_dir / "motion_positions22.npz"
        affordance_file = item_dir / "full_affordance_gt.npz"
        manifest_file = item_dir / "manifest.json"
        atomic_savez(
            positions_file,
            joint_positions22_adm_chair_local_z_up=positions22,
            source_times_s=source_times,
            target_times_s=target_times,
            source_frame_start_inclusive=np.asarray(row["frame_start_inclusive"], dtype=np.int64),
            source_frame_end_inclusive=np.asarray(row["frame_end_inclusive"], dtype=np.int64),
            source_fps=np.asarray(args.source_fps, dtype=np.float32),
            target_fps=np.asarray(args.target_fps, dtype=np.float32),
        )
        atomic_savez(
            affordance_file,
            xyz=scene_xyz,
            instance_ids=instance_ids,
            source_indices=source_indices,
            distance=distance,
            affordance=affordance,
            contact_joints=CONTACT_JOINTS,
            contact_joint_names=np.asarray(CONTACT_JOINT_NAMES),
            sigma=np.asarray(args.sigma, dtype=np.float32),
            gt_root_start_xyz=positions22[0, 0],
        )
        manifest = {
            "schema": DENSE_INDEX_SCHEMA + "_item",
            "status": "DENSE_CONTACT_ITEM_PASS",
            "scene_id": scene_id,
            "motion_id": motion_id,
            "source_motion_file": str(motion_path.relative_to(dataset_root)),
            "source_motion_sha256": str(row["motion_sha256"]),
            "source_crop_inclusive": [int(row["frame_start_inclusive"]), int(row["frame_end_inclusive"])],
            "target_instance_id": target_stable,
            "target_numeric_instance_id": target_numeric,
            "compatible_prompt_ids": sorted(PROMPTS),
            "coordinate_frame": "ChairCoordinate local; AMDM XZY Z-up",
            "source_fps": float(args.source_fps),
            "target_fps": float(args.target_fps),
            "source_crop_frames": int(root.shape[0]),
            "motion_frame_count": int(positions22.shape[0]),
            "point_count": 8192,
            "contact_joints": CONTACT_JOINTS.tolist(),
            "contact_joint_names": list(CONTACT_JOINT_NAMES),
            "sigma": float(args.sigma),
            "max_fk_bone_length_error_m": bone_error,
            "target_pelvis_affordance_max": target_pelvis_max,
            "relation_or_distance_used": False,
            "positions_file": str(positions_file.relative_to(dataset_root)),
            "positions_sha256": sha256_file(positions_file),
            "affordance_file": str(affordance_file.relative_to(dataset_root)),
            "affordance_sha256": sha256_file(affordance_file),
        }
        atomic_write_json(manifest_file, manifest)
        rows.append({
            "motion_id": motion_id,
            "target_instance_id": target_stable,
            "compatible_prompt_ids": sorted(PROMPTS),
            "manifest_file": str(manifest_file.relative_to(dataset_root)),
            "manifest_sha256": sha256_file(manifest_file),
            "motion_frame_count": int(positions22.shape[0]),
            "target_pelvis_affordance_max": target_pelvis_max,
        })
        print(
            f"[DENSE {number:02d}/24] {motion_id} target={target_stable} "
            f"T={positions22.shape[0]} pelvis_max={target_pelvis_max:.6f}"
        )

    rows.sort(key=lambda value: str(value["motion_id"]))
    index = {
        "schema": DENSE_INDEX_SCHEMA,
        "status": "DENSE_CONTACT_V2_PASS",
        "authorization": "three_scene_validation_required_before_training",
        "scene_id": scene_id,
        "motion_count": 24,
        "prompt_expanded_row_count": 48,
        "target_counts": target_counts,
        "sigma": float(args.sigma),
        "contact_joints": CONTACT_JOINTS.tolist(),
        "contact_joint_names": list(CONTACT_JOINT_NAMES),
        "coordinate_scale_unity_xyz": coordinate_scale.tolist(),
        "skeleton_file": str(skeleton_path),
        "skeleton_sha256": sha256_file(skeleton_path),
        "motion_index_sha256": sha256_file(motion_index_path),
        "points_sha256": sha256_file(scene_dir / "points.npz"),
        "sidecar_sha256": sha256_file(scene_dir / "sidecar.npz"),
        "relation_or_distance_used": False,
        "teacher_lora_training_authorized": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_index, index)
    print("[DENSE_CONTACT_V2_PASS] 24 relational Teacher-v7 Sit targets")
    print("[OK] prompt-expanded rows: 48")
    print("[OK] Teacher LoRA training authorized: False (one scene only)")
    print(f"[OK] index: {output_index}")


if __name__ == "__main__":
    main()

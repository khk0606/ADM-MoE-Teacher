#!/usr/bin/env python3
"""Strict source-export preflight for LC17 + HC6 production training.

The two rooms use different chairs and therefore different point ordering.
This checker refuses missing/misaligned scene exports before any ADM/IIW work.
It uses only the Python standard library so it can run before activating CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DATASET_VERSION = "history_affordance_v1"
EXPECTED_PROMPT = "Sit somewhere with a clear view of the TV"
ROOM_COUNTS = {"room_0001": 17, "room_0002": 6}
EXPECTED_INSTANCES = {
    0: "environment",
    1: "chair",
    2: "bed",
    3: "whiteboard",
    4: "tv",
}


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(str(path) + ": expected a JSON object")
    return value


def finite_vector(value: object, length: int, name: str) -> List[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(name + ": expected length " + str(length))
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(name + ": contains NaN/Inf")
    return result


def assert_close(first: Sequence[float], second: Sequence[float], name: str) -> None:
    if len(first) != len(second):
        raise ValueError(name + ": length mismatch")
    maximum = max(abs(float(a) - float(b)) for a, b in zip(first, second))
    if maximum > 1e-5:
        raise ValueError(name + ": mismatch, max_abs_diff=" + str(maximum))


def read_motion_shape(path: Path) -> Sequence[int]:
    rows = 0
    columns = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                raise ValueError(f"{path}:{line_number}: blank row")
            if columns is None:
                columns = len(fields)
            elif len(fields) != columns:
                raise ValueError(f"{path}:{line_number}: ragged columns")
            for value in fields:
                if not math.isfinite(float(value)):
                    raise ValueError(f"{path}:{line_number}: NaN/Inf")
            rows += 1
    if rows <= 0 or columns is None:
        raise ValueError(str(path) + ": empty motion")
    return rows, columns


def validate_scene(unity_root: Path, room_id: str) -> Dict:
    scene_dir = unity_root / "scenes" / room_id
    manifest_file = scene_dir / "scene_manifest.json"
    tsv_file = scene_dir / "scene_points.tsv"
    ply_file = scene_dir / "scene_points.ply"
    for path in (manifest_file, tsv_file, ply_file):
        if not path.is_file():
            raise FileNotFoundError(
                str(path)
                + "\nExport the " + room_id
                + " point cloud from its original saved Unity scene first."
            )
    manifest = load_json(manifest_file)
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(room_id + ": dataset_version mismatch")
    if str(manifest.get("scene_id")) != room_id:
        raise ValueError(room_id + ": scene manifest ID mismatch")
    if str(manifest.get("coordinate_frame")) != "ChairCoordinate":
        raise ValueError(room_id + ": coordinate frame must be ChairCoordinate")
    total_points = int(manifest.get("total_points", -1))
    if total_points <= 0:
        raise ValueError(room_id + ": invalid total_points")
    with tsv_file.open() as handle:
        tsv_rows = sum(1 for _ in handle)
    if tsv_rows != total_points + 1:
        raise ValueError(
            room_id + ": scene_points.tsv rows do not match manifest "
            + f"({tsv_rows} != {total_points}+1)"
        )
    instances = manifest.get("instances")
    if not isinstance(instances, list):
        raise ValueError(room_id + ": instances must be a list")
    found = {
        int(row["instance_id"]): str(row["name"]).lower()
        for row in instances
    }
    if found != EXPECTED_INSTANCES:
        raise ValueError(room_id + ": instance ID/name contract mismatch: " + str(found))
    position = finite_vector(
        manifest.get("chair_coordinate_world_position"), 3,
        room_id + ": scene ChairCoordinate position",
    )
    rotation = finite_vector(
        manifest.get("chair_coordinate_world_rotation_xyzw"), 4,
        room_id + ": scene ChairCoordinate rotation",
    )
    scale = finite_vector(
        manifest.get("chair_coordinate_world_scale"), 3,
        room_id + ": scene ChairCoordinate scale",
    )
    return {
        "manifest": manifest,
        "position": position,
        "rotation": rotation,
        "scale": scale,
        "total_points": total_points,
    }


def validate_samples(unity_root: Path, room_id: str, scene: Dict) -> List[Dict]:
    expected_count = ROOM_COUNTS[room_id]
    sample_root = unity_root / "samples"
    expected_ids = [
        f"{room_id}_sit_chair_{index:04d}"
        for index in range(1, expected_count + 1)
    ]
    actual_ids = sorted(
        path.name for path in sample_root.glob(room_id + "_sit_chair_*")
        if path.is_dir()
    )
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise ValueError(
            room_id + ": sample IDs are not the exact production set; "
            + "missing=" + str(missing) + ", extra=" + str(extra)
        )

    rows = []
    for sample_id in expected_ids:
        sample_dir = sample_root / sample_id
        manifest = load_json(sample_dir / "sample_manifest.json")
        for key, expected in (
            ("dataset_version", DATASET_VERSION),
            ("scene_id", room_id),
            ("sample_id", sample_id),
            ("text", EXPECTED_PROMPT),
            ("target_name", "chair"),
            ("target_instance_id", 1),
            ("target_index", 0),
            ("coordinate_frame", "ChairCoordinate"),
        ):
            if manifest.get(key) != expected:
                raise ValueError(
                    sample_id + ": " + key + " mismatch: "
                    + repr(manifest.get(key)) + " != " + repr(expected)
                )
        if not bool(manifest.get("seat_contact_validated", False)):
            raise ValueError(sample_id + ": seat contact is not validated")
        assert_close(
            finite_vector(manifest.get("coordinate_frame_world_position_xyz"), 3,
                          sample_id + ": coordinate position"),
            scene["position"], sample_id + ": scene/sample coordinate position",
        )
        assert_close(
            finite_vector(manifest.get("coordinate_frame_world_quaternion_xyzw"), 4,
                          sample_id + ": coordinate rotation"),
            scene["rotation"], sample_id + ": scene/sample coordinate rotation",
        )
        assert_close(
            finite_vector(manifest.get("coordinate_frame_local_scale_xyz"), 3,
                          sample_id + ": coordinate scale"),
            scene["scale"], sample_id + ": scene/sample coordinate scale",
        )
        direction = finite_vector(
            manifest.get("history_direction_coordinate_local_xz"), 2,
            sample_id + ": history direction",
        )
        direction_norm = math.sqrt(sum(value * value for value in direction))
        if abs(direction_norm - 1.0) > 1e-4:
            raise ValueError(sample_id + ": history direction is not unit length")
        motion_relative = str(manifest.get("source_motion_file", ""))
        motion_file = sample_dir / motion_relative
        if not motion_file.is_file():
            raise FileNotFoundError(motion_file)
        motion_rows, motion_columns = read_motion_shape(motion_file)
        if motion_rows != int(manifest.get("source_total_frames", -1)):
            raise ValueError(sample_id + ": source motion row count mismatch")
        if motion_columns != int(manifest.get("source_columns", -1)):
            raise ValueError(sample_id + ": source motion column count mismatch")
        if motion_columns != 103:
            raise ValueError(sample_id + ": expected 103 source columns")
        frame_start = int(manifest.get("frame_start_inclusive", -1))
        frame_end = int(manifest.get("frame_end_inclusive", -1))
        if not (0 <= frame_start <= frame_end < motion_rows):
            raise ValueError(sample_id + ": frame crop is outside source motion")
        rows.append(manifest)
    return rows


def validate_prepared(prepared_root: Path) -> None:
    total = 0
    for room_id, expected_count in ROOM_COUNTS.items():
        index_file = prepared_root / ("index_" + room_id + "_iiw.json")
        index = load_json(index_file)
        if str(index.get("scene_id")) != room_id:
            raise ValueError(room_id + ": prepared index scene mismatch")
        if int(index.get("num_samples", -1)) != expected_count:
            raise ValueError(room_id + ": prepared index count mismatch")
        if not bool(index.get("iiw_gt_ready", False)):
            raise ValueError(room_id + ": IIW GT is not ready")
        base_file = (
            prepared_root / "scenes" / room_id
            / "base_affordance" / "base_affordance.npz"
        )
        if not base_file.is_file():
            raise FileNotFoundError(base_file)
        total += expected_count
    if total != 23:
        raise AssertionError("LC+HC total must be exactly 23")
    print("[PASS] prepared LC17+HC6 indices/base caches are present")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=None)
    args = parser.parse_args()
    unity_root = args.unity_root.expanduser().resolve()
    if not unity_root.is_dir():
        raise FileNotFoundError(unity_root)

    total = 0
    for room_id, expected_count in ROOM_COUNTS.items():
        scene = validate_scene(unity_root, room_id)
        samples = validate_samples(unity_root, room_id, scene)
        total += len(samples)
        print(
            f"[PASS] {room_id}: scene_points={scene['total_points']}, "
            f"samples={len(samples)}/{expected_count}"
        )
    if total != 23:
        raise AssertionError("LC17+HC6 total is not 23")
    print("[PASS] LC17+HC6 Unity source contract: total=23")
    if args.prepared_root is not None:
        validate_prepared(args.prepared_root.expanduser().resolve())


if __name__ == "__main__":
    main()

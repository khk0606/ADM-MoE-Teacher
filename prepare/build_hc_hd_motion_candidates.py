#!/usr/bin/env python3
"""Build the 18 immutable hc_hd go/back YBot motion candidates.

The source rows contain a pelvis XYZ followed by 25 world-space Unity
quaternions (XYZW).  Every pair is converted into the recorded chair-local
coordinate system, joined with a 15-frame cross-fade, and written with a
provenance manifest.  This script deliberately does not select the best six;
selection is authorized only after the Unity contact/collision audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


SCHEMA = "relational_affordance_hc_hd_candidates_v1"
DIRECTIONS = ("b", "f", "r")
TAKES = tuple(range(1, 7))
COLUMNS = 103
BONES = 25
FPS = 30

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Frame = List[float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> List[Frame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Frame] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = [float(value) for value in line.replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric row") from exc
        if len(row) != COLUMNS:
            raise ValueError(
                f"{path}:{line_number}: expected {COLUMNS} columns, found {len(row)}"
            )
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{path}:{line_number}: non-finite value")
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two frames")
    return rows


def normalize_quaternion(values: Sequence[float]) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-12:
        raise ValueError("zero-length quaternion")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    return (-q[0], -q[1], -q[2], q[3])


def quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_dot(a: Quaternion, b: Quaternion) -> float:
    return sum(left * right for left, right in zip(a, b))


def quaternion_angle_degrees(a: Quaternion, b: Quaternion) -> float:
    dot = min(1.0, max(-1.0, abs(quaternion_dot(a, b))))
    return math.degrees(2.0 * math.acos(dot))


def quaternion_slerp(a: Quaternion, b: Quaternion, amount: float) -> Quaternion:
    dot = quaternion_dot(a, b)
    if dot < 0.0:
        b = tuple(-value for value in b)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(
            tuple((1.0 - amount) * left + amount * right for left, right in zip(a, b))
        )
    theta = math.acos(dot)
    denominator = math.sin(theta)
    left_weight = math.sin((1.0 - amount) * theta) / denominator
    right_weight = math.sin(amount * theta) / denominator
    return normalize_quaternion(
        tuple(left_weight * left + right_weight * right for left, right in zip(a, b))
    )


def rotate_vector(q: Quaternion, vector: Vector3) -> Vector3:
    x, y, z = vector
    rotated = quaternion_multiply(
        quaternion_multiply(q, (x, y, z, 0.0)), quaternion_conjugate(q)
    )
    return rotated[0], rotated[1], rotated[2]


def root_norm(row: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in row[:3]))


def drop_exporter_dummy(rows: List[Frame]) -> Tuple[List[Frame], bool, float, float]:
    """Drop Unity's initialization row, including small non-zero variants."""
    first_norm = root_norm(rows[0])
    second_norm = root_norm(rows[1])
    is_dummy = first_norm < 0.01 and second_norm > 0.1
    return (rows[1:] if is_dummy else rows), is_dummy, first_norm, second_norm


def read_chair_matrix(path: Path) -> Tuple[Vector3, Quaternion, Vector3]:
    rows = read_matrix_rows(path)
    if len(rows) != 1 or len(rows[0]) != 10:
        raise ValueError(f"{path}: expected one 10-value chair TRS row")
    row = rows[0]
    position = tuple(row[:3])
    rotation = normalize_quaternion(row[3:7])
    scale = tuple(row[7:10])
    if any(abs(value) < 1.0e-12 for value in scale):
        raise ValueError(f"{path}: zero chair scale")
    return position, rotation, scale  # type: ignore[return-value]


def read_matrix_rows(path: Path) -> List[List[float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append([float(value) for value in line.replace(",", " ").split()])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid matrix row") from exc
    return rows


def to_chair_space(
    rows: Sequence[Frame],
    chair_position: Vector3,
    chair_rotation: Quaternion,
    chair_scale: Vector3,
) -> List[Frame]:
    inverse = quaternion_conjugate(chair_rotation)
    converted: List[Frame] = []
    for source in rows:
        shifted = (
            source[0] - chair_position[0],
            source[1] - chair_position[1],
            source[2] - chair_position[2],
        )
        unrotated = rotate_vector(inverse, shifted)
        output: Frame = [
            unrotated[0] / chair_scale[0],
            unrotated[1] / chair_scale[1],
            unrotated[2] / chair_scale[2],
        ]
        for bone_index in range(BONES):
            offset = 3 + 4 * bone_index
            world = normalize_quaternion(source[offset : offset + 4])
            output.extend(normalize_quaternion(quaternion_multiply(inverse, world)))
        converted.append(output)
    return converted


def blend_frames(first: Frame, second: Frame, amount: float) -> Frame:
    output = [
        (1.0 - amount) * first[index] + amount * second[index]
        for index in range(3)
    ]
    for bone_index in range(BONES):
        offset = 3 + 4 * bone_index
        output.extend(
            quaternion_slerp(
                normalize_quaternion(first[offset : offset + 4]),
                normalize_quaternion(second[offset : offset + 4]),
                amount,
            )
        )
    return output


def crossfade(first: Sequence[Frame], second: Sequence[Frame], count: int) -> List[Frame]:
    if not 2 <= count <= min(len(first), len(second)):
        raise ValueError(f"invalid cross-fade length: {count}")
    middle = [
        blend_frames(first[-count + index], second[index], index / (count - 1))
        for index in range(count)
    ]
    return [list(row) for row in first[:-count]] + middle + [list(row) for row in second[count:]]


def enforce_quaternion_continuity(rows: List[Frame]) -> None:
    for bone_index in range(BONES):
        offset = 3 + 4 * bone_index
        previous = normalize_quaternion(rows[0][offset : offset + 4])
        rows[0][offset : offset + 4] = previous
        for row in rows[1:]:
            current = normalize_quaternion(row[offset : offset + 4])
            if quaternion_dot(previous, current) < 0.0:
                current = tuple(-value for value in current)  # type: ignore[assignment]
            row[offset : offset + 4] = current
            previous = current


def maximum_quaternion_error(rows: Sequence[Frame]) -> float:
    maximum = 0.0
    for row in rows:
        for bone_index in range(BONES):
            offset = 3 + 4 * bone_index
            norm = math.sqrt(sum(value * value for value in row[offset : offset + 4]))
            maximum = max(maximum, abs(norm - 1.0))
    return maximum


def maximum_root_step(rows: Sequence[Frame]) -> float:
    return max(
        math.dist(previous[:3], current[:3])
        for previous, current in zip(rows, rows[1:])
    )


def write_rows(path: Path, rows: Iterable[Frame]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(" ".join(f"{value:.9g}" for value in row) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def source_path(
    root: Path,
    direction: str,
    take: int,
    phase: str,
    source_prefix: str = "hc_hd",
) -> Path:
    return root / (
        f"{source_prefix}_{direction}_{take}_{phase}_ret_MIXAMO_rest.bvh_"
        "RetargetedYbot_motion.txt"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-id", default="hc_hd")
    parser.add_argument("--source-prefix")
    parser.add_argument("--blend-frames", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    collection_id = str(args.collection_id).strip()
    source_prefix = str(args.source_prefix or collection_id).strip()
    if not collection_id or not source_prefix:
        raise ValueError("collection ID and source prefix must be non-empty")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in collection_id):
        raise ValueError("collection ID must contain only letters, digits and underscores")
    chair_matrix = source_root / "root_mat_chair.txt"
    desk_matrix = source_root / "root_mat_desk.txt"
    if not desk_matrix.is_file():
        raise FileNotFoundError(desk_matrix)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite candidate set: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    chair_position, chair_rotation, chair_scale = read_chair_matrix(chair_matrix)

    candidates = []
    for direction in DIRECTIONS:
        for take in TAKES:
            go_path = source_path(source_root, direction, take, "go", source_prefix)
            back_path = source_path(source_root, direction, take, "back", source_prefix)
            go_raw = read_rows(go_path)
            back_raw = read_rows(back_path)
            go, go_dummy, go_first, go_second = drop_exporter_dummy(go_raw)
            back, back_dummy, back_first, back_second = drop_exporter_dummy(back_raw)
            go_local = to_chair_space(go, chair_position, chair_rotation, chair_scale)
            back_local = to_chair_space(back, chair_position, chair_rotation, chair_scale)
            join_gap = math.dist(go_local[-1][:3], back_local[0][:3])
            join_root_rotation = quaternion_angle_degrees(
                normalize_quaternion(go_local[-1][3:7]),
                normalize_quaternion(back_local[0][3:7]),
            )
            combined = crossfade(go_local, back_local, int(args.blend_frames))
            enforce_quaternion_continuity(combined)
            name = f"{collection_id}_{direction}_{take}_combined_chair_local.txt"
            output_path = output_dir / name
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite motion: {output_path}")
            write_rows(output_path, combined)
            row = {
                "candidate_id": f"{collection_id}_{direction}_{take}",
                "direction": direction,
                "take": take,
                "source_group_id": f"{collection_id}_recording_{direction}_{take}",
                "go_file": str(go_path),
                "go_sha256": sha256_file(go_path),
                "go_raw_frames": len(go_raw),
                "go_dummy_removed": go_dummy,
                "go_first_root_norm_m": go_first,
                "go_second_root_norm_m": go_second,
                "back_file": str(back_path),
                "back_sha256": sha256_file(back_path),
                "back_raw_frames": len(back_raw),
                "back_dummy_removed": back_dummy,
                "back_first_root_norm_m": back_first,
                "back_second_root_norm_m": back_second,
                "join_root_gap_m_before_crossfade": join_gap,
                "join_root_rotation_deg_before_crossfade": join_root_rotation,
                "output_file": name,
                "output_sha256": sha256_file(output_path),
                "output_frames": len(combined),
                "maximum_root_step_m": maximum_root_step(combined),
                "maximum_quaternion_norm_error": maximum_quaternion_error(combined),
                "unity_audit_status": "PENDING",
                "selection_status": "PENDING",
            }
            candidates.append(row)
            print(
                f"[BUILD {len(candidates):02d}/18] {row['candidate_id']} "
                f"frames={len(combined)} gap={join_gap:.6f}m "
                f"root_rot={join_root_rotation:.3f}deg"
            )

    payload = {
        "schema": f"relational_affordance_{collection_id}_candidates_v1",
        "status": "CANDIDATE_BUILD_PASS",
        "authorization": "unity_contact_collision_audit_required_before_selection",
        "source_root": str(source_root),
        "source_collection_id": collection_id,
        "source_filename_prefix": source_prefix,
        "source_recording_count": 18,
        "go_back_file_count": 36,
        "candidate_count": len(candidates),
        "columns": COLUMNS,
        "bone_count": BONES,
        "source_fps": FPS,
        "blend_frames": int(args.blend_frames),
        "chair_matrix_file": str(chair_matrix),
        "chair_matrix_sha256": sha256_file(chair_matrix),
        "desk_matrix_file": str(desk_matrix),
        "desk_matrix_sha256": sha256_file(desk_matrix),
        "coordinate_space": "recorded_high_desk_chair_local",
        "purpose_relation_contract": (
            "desk matrix is provenance only; purpose distance is not a Teacher forward input"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
    }
    atomic_write_json(manifest_path, payload)
    print(f"[CANDIDATE_BUILD_PASS] {collection_id} 18 go/back pairs")
    print("[PASS] robust exporter-dummy removal and 15-frame cross-fade")
    print("[PASS] 103 columns, 25 normalized quaternion channels, chair-local output")
    print("[OK] selection authorized: False (Unity audit pending)")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

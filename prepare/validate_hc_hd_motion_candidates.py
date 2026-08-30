#!/usr/bin/env python3
"""Validate and exactly reconstruct the hc_hd candidate-set artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build_hc_hd_motion_candidates import (
    BONES,
    COLUMNS,
    DIRECTIONS,
    TAKES,
    crossfade,
    drop_exporter_dummy,
    enforce_quaternion_continuity,
    maximum_quaternion_error,
    maximum_root_step,
    normalize_quaternion,
    quaternion_angle_degrees,
    read_chair_matrix,
    read_rows,
    sha256_file,
    source_path,
    to_chair_space,
)


def close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--require-raw-sources",
        action="store_true",
        help="fail instead of using portable artifact validation",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="validate only self-contained packaged artifacts",
    )
    return parser.parse_args()


def validate_portable_artifact(payload: dict, manifest_path: Path) -> None:
    """Validate sealed outputs when the original Mac recordings are absent."""
    output_dir = manifest_path.parent
    collection_id = str(payload["source_collection_id"])
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 18:
        raise ValueError("candidate rows changed")
    by_id = {str(row.get("candidate_id")): row for row in candidates}
    expected_ids = [
        f"{collection_id}_{direction}_{take}"
        for direction in DIRECTIONS
        for take in TAKES
    ]
    if sorted(by_id) != sorted(expected_ids):
        raise ValueError("candidate ID inventory changed")
    if len({str(row.get("source_group_id")) for row in candidates}) != 18:
        raise ValueError("source groups are not unique")
    for key in ("chair_matrix_sha256", "desk_matrix_sha256"):
        value = str(payload.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid embedded provenance hash: {key}")

    blend_frames = int(payload["blend_frames"])
    for index, candidate_id in enumerate(expected_ids, 1):
        row = by_id[candidate_id]
        direction = str(row["direction"])
        take = int(row["take"])
        if candidate_id != f"{collection_id}_{direction}_{take}":
            raise ValueError(f"candidate identity mismatch: {candidate_id}")
        if row.get("source_group_id") != f"{collection_id}_recording_{direction}_{take}":
            raise ValueError(f"source-group identity mismatch: {candidate_id}")
        for key in ("go_sha256", "back_sha256", "output_sha256"):
            value = str(row.get(key, ""))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{candidate_id}: invalid {key}")
        clean_go = int(row["go_raw_frames"]) - int(bool(row["go_dummy_removed"]))
        clean_back = int(row["back_raw_frames"]) - int(bool(row["back_dummy_removed"]))
        expected_frames = clean_go + clean_back - blend_frames
        if int(row["output_frames"]) != expected_frames:
            raise ValueError(f"{candidate_id}: output frame arithmetic changed")
        output_path = output_dir / str(row["output_file"])
        saved = read_rows(output_path)
        if len(saved) != expected_frames:
            raise ValueError(f"{candidate_id}: packaged frame count changed")
        if sha256_file(output_path) != row["output_sha256"]:
            raise ValueError(f"{candidate_id}: packaged output hash changed")
        if not close(float(row["maximum_root_step_m"]), maximum_root_step(saved), 1.0e-8):
            raise ValueError(f"{candidate_id}: root-step metric changed")
        if not close(
            float(row["maximum_quaternion_norm_error"]),
            maximum_quaternion_error(saved),
            1.0e-8,
        ):
            raise ValueError(f"{candidate_id}: quaternion metric changed")
        if maximum_quaternion_error(saved) > 2.0e-6:
            raise ValueError(f"{candidate_id}: quaternion normalization failed")
        if row.get("unity_audit_status") != "PENDING" or row.get("selection_status") != "PENDING":
            raise ValueError(f"{candidate_id}: candidate manifest was retrospectively changed")
        print(f"[VERIFY {index:02d}/18] {candidate_id} T={len(saved)}")

    print(f"[PORTABLE_CANDIDATE_ARTIFACT_PASS] {collection_id} packaged candidates")
    print("[PASS] 18 output hashes, frame arithmetic and root-step metrics recomputed")
    print("[PASS] finite 103-column rows and 25 normalized quaternion channels")
    print("[PASS] 36 raw-source hashes retained as sealed provenance metadata")
    print("[INFO] raw Mac go/back files were not reread on this host")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection_id = str(payload.get("source_collection_id", ""))
    expected_schema = f"relational_affordance_{collection_id}_candidates_v1"
    if not collection_id or payload.get("schema") != expected_schema:
        raise ValueError("unsupported motion candidate manifest schema")
    if payload.get("status") != "CANDIDATE_BUILD_PASS":
        raise ValueError("candidate build status is not PASS")
    if payload.get("authorization") != "unity_contact_collision_audit_required_before_selection":
        raise ValueError("selection authorization changed")
    if int(payload.get("columns", -1)) != COLUMNS or int(payload.get("bone_count", -1)) != BONES:
        raise ValueError("motion layout changed")
    if int(payload.get("candidate_count", -1)) != 18:
        raise ValueError("expected exactly 18 candidates")
    if int(payload.get("source_recording_count", -1)) != 18:
        raise ValueError("source recording count changed")
    if int(payload.get("go_back_file_count", -1)) != 36:
        raise ValueError("go/back inventory changed")

    source_root = Path(str(payload["source_root"])).resolve()
    chair_matrix = Path(str(payload["chair_matrix_file"])).resolve()
    desk_matrix = Path(str(payload["desk_matrix_file"])).resolve()
    raw_sources_available = not args.portable and (
        source_root.is_dir() and chair_matrix.is_file() and desk_matrix.is_file()
    )
    if not raw_sources_available:
        if args.require_raw_sources:
            raise FileNotFoundError(
                f"raw {collection_id} source collection is unavailable on this host"
            )
        validate_portable_artifact(payload, manifest_path)
        return
    if sha256_file(chair_matrix) != payload.get("chair_matrix_sha256"):
        raise ValueError("chair matrix hash changed")
    if sha256_file(desk_matrix) != payload.get("desk_matrix_sha256"):
        raise ValueError("desk matrix hash changed")
    chair_position, chair_rotation, chair_scale = read_chair_matrix(chair_matrix)
    blend_frames = int(payload["blend_frames"])
    output_dir = manifest_path.parent

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 18:
        raise ValueError("candidate rows changed")
    by_id = {str(row.get("candidate_id")): row for row in candidates}
    expected_ids = [
        f"{collection_id}_{direction}_{take}"
        for direction in DIRECTIONS
        for take in TAKES
    ]
    if sorted(by_id) != sorted(expected_ids):
        raise ValueError("candidate ID inventory changed")

    for index, candidate_id in enumerate(expected_ids, 1):
        row = by_id[candidate_id]
        direction = str(row["direction"])
        take = int(row["take"])
        if candidate_id != f"{collection_id}_{direction}_{take}":
            raise ValueError(f"candidate identity mismatch: {candidate_id}")
        if row.get("source_group_id") != f"{collection_id}_recording_{direction}_{take}":
            raise ValueError(f"source-group identity mismatch: {candidate_id}")
        source_prefix = str(payload.get("source_filename_prefix", collection_id))
        go_path = source_path(source_root, direction, take, "go", source_prefix)
        back_path = source_path(source_root, direction, take, "back", source_prefix)
        if str(go_path) != row.get("go_file") or str(back_path) != row.get("back_file"):
            raise ValueError(f"source path changed: {candidate_id}")
        if sha256_file(go_path) != row.get("go_sha256"):
            raise ValueError(f"go source hash changed: {candidate_id}")
        if sha256_file(back_path) != row.get("back_sha256"):
            raise ValueError(f"back source hash changed: {candidate_id}")

        go_raw = read_rows(go_path)
        back_raw = read_rows(back_path)
        go, go_dummy, go_first, go_second = drop_exporter_dummy(go_raw)
        back, back_dummy, back_first, back_second = drop_exporter_dummy(back_raw)
        scalar_checks = {
            "go_raw_frames": len(go_raw),
            "back_raw_frames": len(back_raw),
            "go_dummy_removed": go_dummy,
            "back_dummy_removed": back_dummy,
        }
        for key, expected in scalar_checks.items():
            if row.get(key) != expected:
                raise ValueError(f"{candidate_id}: {key} changed")
        for key, expected in {
            "go_first_root_norm_m": go_first,
            "go_second_root_norm_m": go_second,
            "back_first_root_norm_m": back_first,
            "back_second_root_norm_m": back_second,
        }.items():
            if not close(float(row[key]), expected):
                raise ValueError(f"{candidate_id}: {key} changed")

        go_local = to_chair_space(go, chair_position, chair_rotation, chair_scale)
        back_local = to_chair_space(back, chair_position, chair_rotation, chair_scale)
        gap = math.dist(go_local[-1][:3], back_local[0][:3])
        rotation = quaternion_angle_degrees(
            normalize_quaternion(go_local[-1][3:7]),
            normalize_quaternion(back_local[0][3:7]),
        )
        combined = crossfade(go_local, back_local, blend_frames)
        enforce_quaternion_continuity(combined)
        output_path = output_dir / str(row["output_file"])
        saved = read_rows(output_path)
        if len(saved) != len(combined) or int(row["output_frames"]) != len(combined):
            raise ValueError(f"{candidate_id}: frame count changed")
        if sha256_file(output_path) != row.get("output_sha256"):
            raise ValueError(f"{candidate_id}: output hash changed")
        if not close(float(row["join_root_gap_m_before_crossfade"]), gap):
            raise ValueError(f"{candidate_id}: join gap changed")
        if not close(float(row["join_root_rotation_deg_before_crossfade"]), rotation):
            raise ValueError(f"{candidate_id}: join rotation changed")
        if not close(float(row["maximum_root_step_m"]), maximum_root_step(saved), 1.0e-8):
            raise ValueError(f"{candidate_id}: root-step metric changed")
        if not close(
            float(row["maximum_quaternion_norm_error"]),
            maximum_quaternion_error(saved),
            1.0e-8,
        ):
            raise ValueError(f"{candidate_id}: quaternion metric changed")
        if maximum_quaternion_error(saved) > 2.0e-6:
            raise ValueError(f"{candidate_id}: quaternion normalization failed")
        if row.get("unity_audit_status") != "PENDING" or row.get("selection_status") != "PENDING":
            raise ValueError(f"{candidate_id}: premature Unity/selection claim")
        print(f"[VERIFY {index:02d}/18] {candidate_id} T={len(saved)}")

    print(f"[CANDIDATE_BUILD_PASS] {collection_id} candidate artifact integrity")
    print("[PASS] 36 source hashes and robust dummy-row decisions recomputed")
    print("[PASS] 18 chair-local cross-fades, frame counts and output hashes verified")
    print("[PASS] finite 103-column rows and 25 normalized quaternion channels")
    print("[OK] Unity contact/collision selection still required")


if __name__ == "__main__":
    main()

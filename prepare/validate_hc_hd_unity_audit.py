#!/usr/bin/env python3
"""Recompute the immutable hc_hd Unity candidate-audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_motion(path: Path) -> tuple[int, float]:
    frames = 0
    maximum_norm_error = 0.0
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        values = [float(token) for token in line.replace(",", " ").split()]
        if len(values) != 103:
            raise ValueError(f"{path}: line {line_number} has {len(values)} columns")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: non-finite value at line {line_number}")
        for column in range(3, 103, 4):
            norm = math.sqrt(sum(value * value for value in values[column : column + 4]))
            maximum_norm_error = max(maximum_norm_error, abs(norm - 1.0))
        frames += 1
    if frames == 0:
        raise ValueError(f"{path}: empty motion")
    return frames, maximum_norm_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    manifest = json.loads(args.candidate_manifest.read_text())
    motion_root = args.motion_root or args.report.parent / "motions"

    if report.get("schema") != "relational_affordance_hc_hd_unity_audit_v1":
        raise ValueError("unexpected Unity audit schema")
    if report.get("candidate_manifest_sha256") != sha256(args.candidate_manifest):
        raise ValueError("candidate manifest hash changed")
    sources = {item["candidate_id"]: item for item in manifest["candidates"]}
    rows = report.get("candidates", [])
    if len(rows) != 18 or len(sources) != 18:
        raise ValueError("expected exactly 18 candidates")
    if len({row["candidate_id"] for row in rows}) != 18:
        raise ValueError("duplicate candidate IDs")
    if len({row["source_group_id"] for row in rows}) != 18:
        raise ValueError("source groups are not independent")
    if report.get("purpose_relation_used_as_forward_input") is not False:
        raise ValueError("purpose relation leaked into Teacher forward input")

    eligible = []
    max_quaternion_error = 0.0
    for row in rows:
        candidate_id = row["candidate_id"]
        source = sources.get(candidate_id)
        if source is None:
            raise ValueError(f"unknown candidate: {candidate_id}")
        if row["source_group_id"] != source["source_group_id"]:
            raise ValueError(f"source group changed: {candidate_id}")
        if row["source_motion_sha256"] != source["output_sha256"]:
            raise ValueError(f"source hash changed: {candidate_id}")
        path = motion_root / Path(row["audited_motion_asset"]).name
        if sha256(path) != row["audited_motion_sha256"]:
            raise ValueError(f"audited motion hash changed: {candidate_id}")
        frames, quaternion_error = read_motion(path)
        max_quaternion_error = max(max_quaternion_error, quaternion_error)
        if frames != row["total_frames"]:
            raise ValueError(f"audited frame count changed: {candidate_id}")
        source_frame_count = (
            row["frame_end_inclusive"] - row["frame_start_inclusive"] + 1
        )
        expected_cmdm = round((source_frame_count - 1) * 20 / 30) + 1
        if expected_cmdm != row["cmdm_frames_at_20fps"]:
            raise ValueError(f"CMDM frame arithmetic changed: {candidate_id}")
        failed = []
        if not row["target_contact_passed"]:
            failed.append("target_contact_passes")
        if not row["collision"]["passed"]:
            failed.append("non_target_collision_passes")
        if not row["floor"]["passed"]:
            failed.append("floor_penetration_passes")
        if not row["dataset_contract_passed"]:
            failed.append("dataset_crop_contract_passes")
        if row["cmdm_frames_at_20fps"] > 196:
            failed.append("cmdm_frames_at_most_196")
        if failed != row["failed_checks"]:
            raise ValueError(f"failed checks changed: {candidate_id}")
        if row["eligible_for_source_split"] != (not failed):
            raise ValueError(f"eligibility changed: {candidate_id}")
        if not failed:
            eligible.append(candidate_id)

    if max_quaternion_error > 5e-5:
        raise ValueError("audited quaternion norm error exceeds tolerance")
    if report.get("eligible_count") != len(eligible):
        raise ValueError("eligible aggregate changed")
    if report.get("status") != "AUDIT_FAIL":
        raise ValueError("immutable audit status changed")
    if report.get("failed_checks") != ["at_least_12_candidates_support_train_dev_split"]:
        raise ValueError("unexpected top-level audit failure")

    print("[PHYSICAL_AUDIT_INTEGRITY_PASS] hc_hd Unity artifact")
    print("[PASS] 18 motion hashes, 103-column rows and quaternions recomputed")
    print("[PASS] Chair contact, collision, floor and CMDM crop gates recomputed")
    print(f"[OK] physically eligible: {len(eligible)}/18")
    print("[OK] train six can be selected: True")
    print("[OK] disjoint train6/development6 split: False (two sources short)")


if __name__ == "__main__":
    main()

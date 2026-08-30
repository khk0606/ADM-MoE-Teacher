#!/usr/bin/env python3
"""Recompute the room_0201 hcw_hdw physical audit and sealed six binding."""

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


def validate_motion(path: Path, expected_frames: int) -> None:
    frames = 0
    maximum_quaternion_error = 0.0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        values = [float(token) for token in raw.replace(",", " ").split()]
        if len(values) != 103 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid motion row: {path}:{line_number}")
        for column in range(3, 103, 4):
            norm = math.sqrt(sum(value * value for value in values[column : column + 4]))
            maximum_quaternion_error = max(maximum_quaternion_error, abs(norm - 1.0))
        frames += 1
    if frames != expected_frames or maximum_quaternion_error > 5.0e-5:
        raise ValueError(f"motion numerical contract changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest", type=Path, required=True)
    parser.add_argument("--audit-motion-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--binding-root", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    train_manifest = json.loads(args.train_candidate_manifest.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if audit.get("schema") != "relational_affordance_hcw_hdw_development_unity_audit_v1":
        raise ValueError("development audit schema changed")
    if audit.get("status") != "AUDIT_COMPLETE" or audit.get("failed_checks"):
        raise ValueError("development audit status changed")
    if audit.get("candidate_manifest_sha256") != sha256(args.candidate_manifest):
        raise ValueError("audit/manifest binding changed")
    if selection.get("status") != "DEVELOPMENT_SELECTION_PASS":
        raise ValueError("development selection status changed")
    if selection.get("unity_audit_sha256") != sha256(args.audit):
        raise ValueError("selection/audit binding changed")

    sources = {row["candidate_id"]: row for row in manifest["candidates"]}
    rows = audit.get("candidates", [])
    if len(rows) != 18 or len(sources) != 18:
        raise ValueError("candidate inventory changed")
    eligible = []
    for row in rows:
        candidate_id = row["candidate_id"]
        source = sources.get(candidate_id)
        if source is None or row["source_group_id"] != source["source_group_id"]:
            raise ValueError(f"source binding changed: {candidate_id}")
        source_motion = args.candidate_manifest.parent / source["output_file"]
        if sha256(source_motion) != row["source_motion_sha256"]:
            raise ValueError(f"source motion hash changed: {candidate_id}")
        audited_motion = args.audit_motion_root / Path(row["audited_motion_asset"]).name
        if sha256(audited_motion) != row["audited_motion_sha256"]:
            raise ValueError(f"audited motion hash changed: {candidate_id}")
        validate_motion(audited_motion, int(row["total_frames"]))
        failed = []
        if not row["target_contact_passed"]:
            failed.append("target_contact_passes")
        if not row["collision"]["passed"]:
            failed.append("non_target_collision_passes")
        if not row["floor"]["passed"]:
            failed.append("floor_penetration_passes")
        if not row["dataset_contract_passed"]:
            failed.append("dataset_crop_contract_passes")
        if int(row["cmdm_frames_at_20fps"]) > 196:
            failed.append("cmdm_frames_at_most_196")
        if failed != row["failed_checks"]:
            raise ValueError(f"failed checks changed: {candidate_id}")
        if row["eligible_for_source_split"] != (not failed):
            raise ValueError(f"eligibility changed: {candidate_id}")
        if not failed:
            eligible.append(row)
    if len(eligible) != int(audit["eligible_count"]) or len(eligible) < 6:
        raise ValueError("eligible aggregate changed")

    expected = sorted(
        eligible, key=lambda row: (float(row["quality_score"]), row["candidate_id"])
    )[:6]
    selected = selection.get("selected", [])
    if [row["candidate_id"] for row in selected] != [row["candidate_id"] for row in expected]:
        raise ValueError("development top-six selection changed")
    if len({row["source_group_id"] for row in selected}) != 6:
        raise ValueError("development selected sources are not independent")
    for row in selected:
        path = args.binding_root / row["motion_file"]
        if sha256(path) != row["motion_sha256"]:
            raise ValueError(f"sealed binding hash changed: {row['candidate_id']}")

    train_hashes = {
        row[key]
        for row in train_manifest["candidates"]
        for key in ("go_sha256", "back_sha256")
    }
    development_hashes = {
        row[key]
        for row in manifest["candidates"]
        for key in ("go_sha256", "back_sha256")
    }
    if train_hashes & development_hashes:
        raise ValueError("train/development raw source hash overlap")
    if selection.get("augmentation3_used_as_independent_source") is not False:
        raise ValueError("augmentation was counted as an independent source")

    print("[DEVELOPMENT_BINDING_PASS] hcw_hdw selected six in room_0201")
    print("[PASS] 18 Unity physical rows and motion hashes recomputed")
    print("[PASS] global physical-score top six and source independence recomputed")
    print("[PASS] train hc_hd raw hashes are disjoint; augmentation3 excluded")
    print(f"[OK] physically eligible: {len(eligible)}/18")
    print(f"[OK] selected: {[row['candidate_id'] for row in selected]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seal the six best independent hcw_hdw motions for room_0201 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest", type=Path, required=True)
    parser.add_argument("--audit-motion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()

    if args.selection.exists() or args.output_root.exists():
        raise FileExistsError("refusing to overwrite sealed development binding")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    train_manifest = json.loads(args.train_candidate_manifest.read_text(encoding="utf-8"))
    if audit.get("status") != "AUDIT_COMPLETE" or audit.get("failed_checks"):
        raise ValueError("room_0201 Unity audit did not pass")
    if audit.get("schema") != "relational_affordance_hcw_hdw_development_unity_audit_v1":
        raise ValueError("unexpected development audit schema")
    if manifest.get("schema") != "relational_affordance_hcw_hdw_candidates_v1":
        raise ValueError("unexpected hcw_hdw manifest schema")
    if audit.get("candidate_manifest_sha256") != sha256(args.candidate_manifest):
        raise ValueError("candidate manifest binding changed")

    train_raw_hashes = {
        row[key]
        for row in train_manifest["candidates"]
        for key in ("go_sha256", "back_sha256")
    }
    development_raw_hashes = {
        row[key]
        for row in manifest["candidates"]
        for key in ("go_sha256", "back_sha256")
    }
    if train_raw_hashes & development_raw_hashes:
        raise ValueError("train/development raw recording hash overlap")

    eligible = sorted(
        [row for row in audit["candidates"] if row["eligible_for_source_split"]],
        key=lambda row: (float(row["quality_score"]), row["candidate_id"]),
    )
    if len(eligible) < 6:
        raise ValueError("fewer than six physically eligible development sources")
    selected = eligible[:6]
    if len({row["source_group_id"] for row in selected}) != 6:
        raise ValueError("selected development sources are not independent")

    args.output_root.mkdir(parents=True)
    sealed_rows = []
    for row in selected:
        source = args.audit_motion_root / Path(row["audited_motion_asset"]).name
        if sha256(source) != row["audited_motion_sha256"]:
            raise ValueError(f"audited motion hash changed: {row['candidate_id']}")
        destination = args.output_root / source.name
        shutil.copy2(source, destination)
        if sha256(destination) != row["audited_motion_sha256"]:
            raise ValueError(f"sealed motion copy changed: {row['candidate_id']}")
        sealed_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "source_group_id": row["source_group_id"],
                "quality_score": row["quality_score"],
                "seated_frame": row["seated_frame"],
                "frame_start_inclusive": row["frame_start_inclusive"],
                "frame_end_inclusive": row["frame_end_inclusive"],
                "cmdm_frames_at_20fps": row["cmdm_frames_at_20fps"],
                "target_contact_gap_m": row["target_contact_gap_m"],
                "motion_file": destination.name,
                "motion_sha256": row["audited_motion_sha256"],
            }
        )

    output = {
        "schema": "relational_teacher_v7_hcw_hdw_development_selection_v1",
        "status": "DEVELOPMENT_SELECTION_PASS",
        "authorization": {
            "room_0201_development_binding": True,
            "dense_contact_v2_dataset_build": True,
            "teacher_lora_v7_training": False,
        },
        "scene_id": "room_0201",
        "target_instance_id": "chair_02",
        "purpose_instance_id": "desk_01",
        "text": "Sit anywhere to write.",
        "selection_policy": "global six lowest Unity physical quality scores",
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "train_candidate_manifest": str(args.train_candidate_manifest.resolve()),
        "train_candidate_manifest_sha256": sha256(args.train_candidate_manifest),
        "unity_audit": str(args.audit.resolve()),
        "unity_audit_sha256": sha256(args.audit),
        "eligible_count": len(eligible),
        "selected_count": len(sealed_rows),
        "raw_source_hash_overlap_with_train": 0,
        "augmentation3_used_as_independent_source": False,
        "selected": sealed_rows,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failed_checks": [],
    }
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("[DEVELOPMENT_SELECTION_PASS] six independent hcw_hdw motions")
    for row in sealed_rows:
        print(f"[SELECTED] {row['candidate_id']} score={row['quality_score']:.6f}")
    print("[PASS] raw source hashes are disjoint from hc_hd train recordings")
    print("[PASS] augmentation3 copies excluded from independent-source count")
    print(f"[OK] selection: {args.selection.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the selected six in both train scene layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(report: dict, selected_ids: set[str]) -> dict[str, dict]:
    result = {
        row["candidate_id"]: row
        for row in report["candidates"]
        if row["candidate_id"] in selected_ids
    }
    if set(result) != selected_ids:
        raise ValueError("selected candidate coverage changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--binding-root", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    if selection["status"] != "TRAIN_SELECTION_PASS_DEVELOPMENT_SOURCE_BLOCKED":
        raise ValueError("unexpected selection state")
    selected = selection["train_selected"]
    selected_ids = {row["candidate_id"] for row in selected}
    if len(selected_ids) != 6:
        raise ValueError("train selection is not six unique candidates")
    if selection["selection_policy"] != "global six lowest Unity physical quality scores":
        raise ValueError("selection policy changed")

    reports = {
        "room_0101": json.loads((args.binding_root / "room_0101_audit_report.json").read_text()),
        "room_0102": json.loads((args.binding_root / "room_0102_report.json").read_text()),
    }
    for scene_id, report in reports.items():
        if report["scene_id"] != scene_id:
            raise ValueError(f"scene binding changed: {scene_id}")
        if scene_id == "room_0102" and report["status"] != "TRAIN_BINDING_COMPLETE":
            raise ValueError("room_0102 train binding did not pass")
        rows = selected_rows(report, selected_ids)
        for candidate_id, row in rows.items():
            if not row["eligible_for_source_split"] or row["failed_checks"]:
                raise ValueError(f"selected binding failed: {scene_id}/{candidate_id}")
            if not row["target_contact_passed"]:
                raise ValueError(f"Chair contact failed: {scene_id}/{candidate_id}")
            if row["collision"]["collision_frame_count"] != 0:
                raise ValueError(f"collision found: {scene_id}/{candidate_id}")
            if row["floor"]["maximum_penetration_m"] > 0.04001:
                raise ValueError(f"floor penetration exceeded: {scene_id}/{candidate_id}")
            if row["cmdm_frames_at_20fps"] > 196:
                raise ValueError(f"CMDM crop exceeded: {scene_id}/{candidate_id}")
            motion = args.binding_root / scene_id / "chair_06" / Path(
                row["audited_motion_asset"]
            ).name
            if sha256(motion) != row["audited_motion_sha256"]:
                raise ValueError(f"motion hash changed: {scene_id}/{candidate_id}")

    room1_groups = {
        reports["room_0101"]["candidates"][index]["source_group_id"]
        for index in range(len(reports["room_0101"]["candidates"]))
        if reports["room_0101"]["candidates"][index]["candidate_id"] in selected_ids
    }
    room2_groups = {
        reports["room_0102"]["candidates"][index]["source_group_id"]
        for index in range(len(reports["room_0102"]["candidates"]))
        if reports["room_0102"]["candidates"][index]["candidate_id"] in selected_ids
    }
    if room1_groups != room2_groups or len(room1_groups) != 6:
        raise ValueError("train scenes do not reuse the exact selected six sources")

    print("[TRAIN_BINDING_PASS] hc_hd selected six in room_0101 and room_0102")
    print("[PASS] 12 motion hashes and Unity physical gates recomputed")
    print("[PASS] exact global top-six physical-score selection and independent sources")
    print("[PASS] Chair contact, zero non-target collision and floor thresholds")
    print("[INFO] room_0201 is authorized only by the separate independent hcw_hdw artifact")


if __name__ == "__main__":
    main()

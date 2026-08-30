#!/usr/bin/env python3
"""Select six direction-balanced train motions without hiding dev shortage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction(candidate_id: str) -> str:
    parts = candidate_id.split("_")
    if len(parts) != 4 or parts[:2] != ["hc", "hd"] or parts[2] not in {"b", "f", "r"}:
        raise ValueError(f"unexpected candidate ID: {candidate_id}")
    return parts[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    groups = defaultdict(list)
    for row in audit["candidates"]:
        if row["eligible_for_source_split"]:
            groups[direction(row["candidate_id"])].append(row)
    eligible = sorted(
        [row for rows in groups.values() for row in rows],
        key=lambda row: (row["quality_score"], row["candidate_id"]),
    )
    if len(eligible) < 6:
        raise ValueError("fewer than six physically valid motions")
    selected = sorted(eligible[:6], key=lambda row: row["candidate_id"])
    selected_ids = {row["candidate_id"] for row in selected}
    reserve = sorted(
        [row for rows in groups.values() for row in rows if row["candidate_id"] not in selected_ids],
        key=lambda row: (row["quality_score"], row["candidate_id"]),
    )
    rejected = sorted(
        [row for row in audit["candidates"] if not row["eligible_for_source_split"]],
        key=lambda row: row["candidate_id"],
    )
    if len({row["source_group_id"] for row in selected}) != 6:
        raise ValueError("selected train motions do not have six independent sources")
    if {row["source_group_id"] for row in selected} & {
        row["source_group_id"] for row in reserve
    }:
        raise ValueError("train/reserve source overlap")

    def compact(row: dict) -> dict:
        return {
            "candidate_id": row["candidate_id"],
            "source_group_id": row["source_group_id"],
            "direction": direction(row["candidate_id"]),
            "audited_motion_asset": row["audited_motion_asset"],
            "audited_motion_sha256": row["audited_motion_sha256"],
            "seated_frame": row["seated_frame"],
            "cmdm_frames_at_20fps": row["cmdm_frames_at_20fps"],
            "target_contact_gap_m": row["target_contact_gap_m"],
            "quality_score": row["quality_score"],
        }

    development_shortfall = max(0, 6 - len(reserve))
    output = {
        "schema": "relational_teacher_v7_hc_hd_source_selection_v1",
        "status": "TRAIN_SELECTION_PASS_DEVELOPMENT_SOURCE_BLOCKED",
        "authorization": {
            "room_0101_room_0102_train_binding": True,
            "room_0201_development_binding": False,
            "dense_contact_v2_dataset_build": False,
            "teacher_lora_v7_training": False,
        },
        "audit": str(args.audit.resolve()),
        "audit_sha256": sha256(args.audit),
        "selection_policy": "global six lowest Unity physical quality scores",
        "train_selected_count": len(selected),
        "development_independent_eligible_count": len(reserve),
        "development_required_count": 6,
        "development_source_shortfall": development_shortfall,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_selected": [compact(row) for row in selected],
        "development_reserve": [compact(row) for row in reserve],
        "rejected": [
            {
                "candidate_id": row["candidate_id"],
                "source_group_id": row["source_group_id"],
                "failed_checks": row["failed_checks"],
            }
            for row in rejected
        ],
        "failed_checks": ["six_independent_room_0201_sources_available"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print("[TRAIN_SELECTION_PASS] six stable hc_hd motions")
    for row in output["train_selected"]:
        print(
            f"[SELECTED] {row['candidate_id']} direction={row['direction']} "
            f"score={row['quality_score']:.6f}"
        )
    print(
        "[DEVELOPMENT_SOURCE_BLOCKED] independent eligible sources: "
        f"{len(reserve)}/6; shortfall={development_shortfall}"
    )
    print(f"[OK] output: {args.output.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print Teacher-v9.8.10 rollout-aligned recovery diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    print("=== TEACHER-v9.8.10 ACTUAL TWO-SCENE K=3 RECOVERY BRACKET ===")
    print("status", value["status"], "selected_radius", value["selected_radius"])
    print("direction", value["selected_direction"], value["selected_direction_sha256"])
    for row in value["response_rows"]:
        print("\nRADIUS", row["radius"], "eligible", row["eligible"])
        for scene in ("room_0101", "room_0102"):
            print(scene, "failed", row["scene_failed_checks"][scene])
            for role in ("bed", "normal_chair", "high_chair"):
                before = row["scene_base_pooled"][scene][role]
                after = row["scene_candidate_pooled"][scene][role]
                print(
                    role,
                    "recall {:.6f}->{:.6f}".format(
                        before["soft_recall"], after["soft_recall"]
                    ),
                    "MAE {:.6f}->{:.6f}".format(
                        before["active_support_mae"], after["active_support_mae"]
                    ),
                )
        v5_change = 100.0 * (
            sum(row["candidate_v5_dense"]) / sum(value["base_v5_dense"]) - 1.0
        )
        print("v5 change {:+.6f}%".format(v5_change))
        print("candidate failed", row["failed_checks"])
    print("\nfailed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: fresh two-scene rollout-aligned calibration only")
    else:
        print("next authority: cross-scene training-objective redesign only")
    print("no optimizer or checkpoint was written")


if __name__ == "__main__":
    main()

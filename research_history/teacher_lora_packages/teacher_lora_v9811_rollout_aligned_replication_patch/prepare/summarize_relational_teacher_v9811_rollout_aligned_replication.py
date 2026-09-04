#!/usr/bin/env python3
"""Print Teacher-v9.8.11 new-seed rollout replication diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    print("=== TEACHER-v9.8.11 DISJOINT-SEED TWO-SCENE K=3 REPLICATION ===")
    print("status", value["status"], "fixed_radius", value["selected_radius"])
    print("direction", value["selected_direction"], value["selected_direction_sha256"])
    print("replication seeds", value["replication_seed_table"])
    for scene in ("room_0101", "room_0102"):
        print("\nSCENE", scene, "failed", value["scene_failed_checks"][scene])
        for role in ("bed", "normal_chair", "high_chair"):
            before = value["scene_base_pooled"][scene][role]
            after = value["scene_candidate_pooled"][scene][role]
            print(
                role,
                "recall {:.6f}->{:.6f}".format(
                    before["soft_recall"], after["soft_recall"]
                ),
                "MAE {:.6f}->{:.6f}".format(
                    before["active_support_mae"], after["active_support_mae"]
                ),
            )
        all_three = {"watch": 0, "write": 0}
        for generation in value["presence"][scene]:
            for prompt_index, prompt in enumerate(("watch", "write")):
                if all(generation[prompt_index].values()):
                    all_three[prompt] += 1
        print("absolute all-three diagnostic", all_three)
    v5_change = 100.0 * (
        sum(value["candidate_v5_dense"]) / sum(value["base_v5_dense"]) - 1.0
    )
    print("\nv5 change {:+.6f}%".format(v5_change))
    print("response failed", value["response_failed_checks"])
    print("failed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: fresh two-scene rollout-aligned multi-update calibration only")
    else:
        print("next authority: cross-scene training-objective redesign only")
    print("no optimizer or checkpoint was written")


if __name__ == "__main__":
    main()

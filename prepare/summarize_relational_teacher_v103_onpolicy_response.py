#!/usr/bin/env python3
"""Concise Teacher-v10.3 response summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from relational_teacher_v9_all_sittable_contract import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.report.expanduser().resolve())
    print("=== TEACHER-v10.3 TWO-SCENE ON-POLICY RESPONSE ===")
    print("status", value["status"], "selected", value["selected_candidate"])
    print("absolute all-three is diagnostic only in this response gate")
    for row in value["candidates"]:
        print("\nCANDIDATE", row["name"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        for scene in value["train_scenes"]:
            print(
                scene,
                "all-three base->candidate",
                row["base_all_three_diagnostic"][scene],
                "->",
                row["candidate_all_three_diagnostic"][scene],
            )
        for role in ("bed", "normal_chair", "high_chair"):
            print(
                role,
                "recall {:.6f}->{:.6f} MAE {:.6f}->{:.6f}".format(
                    row["base_pooled"][role]["soft_recall"],
                    row["candidate_pooled"][role]["soft_recall"],
                    row["base_pooled"][role]["active_support_mae"],
                    row["candidate_pooled"][role]["active_support_mae"],
                ),
            )
        base = sum(row["base_v5_dense"])
        candidate = sum(row["candidate_v5_dense"])
        print("v5 change {:+.3f}%".format(100.0 * (candidate / base - 1.0)))
    print("\nfailed checks", value["failed_checks"])
    print(
        "next authority:",
        "two-scene on-policy multi-update calibration"
        if value["status"] == "PASS"
        else "LoRA capacity/placement diagnosis",
    )


if __name__ == "__main__":
    main()

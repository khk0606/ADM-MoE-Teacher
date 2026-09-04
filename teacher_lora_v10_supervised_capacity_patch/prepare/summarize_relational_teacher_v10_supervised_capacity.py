#!/usr/bin/env python3
"""Concise terminal summary for Teacher-v10 supervised capacity training."""

from __future__ import annotations

import argparse
from pathlib import Path

from relational_teacher_v9_all_sittable_contract import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    print("=== TEACHER-v10 FRESH SUPERVISED ALL-SITTABLE ===")
    print("status", value["status"], "selected_step", value["selected_step"])
    print("training: GT x0 q_sample, AdamW, rank-16, two scenes per update")
    print("shortlisted_steps", value["shortlisted_steps"])
    for row in value["rollout_rows"]:
        print("\nCANDIDATE", row["step"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        print("all-three", row["all_three_counts"])
        for scene, pooled in row["pooled_metrics"].items():
            print(
                scene,
                "recall bed/normal/high",
                "{:.4f}/{:.4f}/{:.4f}".format(
                    pooled["bed"]["soft_recall"],
                    pooled["normal_chair"]["soft_recall"],
                    pooled["high_chair"]["soft_recall"],
                ),
                "MAE",
                "{:.4f}/{:.4f}/{:.4f}".format(
                    pooled["bed"]["active_support_mae"],
                    pooled["normal_chair"]["active_support_mae"],
                    pooled["high_chair"]["active_support_mae"],
                ),
            )
        base = sum(row["base_v5_dense"]) / 3.0
        candidate = sum(row["candidate_v5_dense"]) / 3.0
        print("v5 change {:+.3f}%".format(100.0 * (candidate / base - 1.0)))
    print("\ncheckpoint", value["paths"].get("checkpoint"))
    print("failed checks", value["failed_checks"])
    print(
        "next authority:",
        "checkpoint lock" if value["status"] == "PASS" else "capacity/objective redesign",
    )


if __name__ == "__main__":
    main()

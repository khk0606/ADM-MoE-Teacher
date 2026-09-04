#!/usr/bin/env python3
"""Print Teacher-v9.8.8 rollout-aligned direction diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))
    print("=== TEACHER-v9.8.8 ROLLOUT-ALIGNED DIRECTION DIAGNOSIS ===")
    print("status", value["status"], "selected", value["selected_candidate"])
    print(
        "sealed cause: every v9.8.7 radius regressed room_0102 Bed recall and MAE"
    )
    print("tasks: 36 object + 12 explicit-negative + 3 v5 = 51")
    print("conflicts", value["conflict_pairs"], "/", value["conflict_pair_total"])
    for row in value["direction_candidates"]:
        print(
            "\nCANDIDATE",
            row["name"],
            "eligible",
            row["eligible"],
            "audit-Bed-min",
            "{:.9f}".format(row["minimum_audit_bed_derivative"]),
            "all-task-min",
            "{:.9f}".format(row["minimum_all_task_derivative"]),
        )
        print("failed", row["failed_checks"])
    print("\nfailed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: actual two-scene K=3 rollout-aligned radius grid only")
    else:
        print("next authority: cross-scene objective/inference redesign only")
    print("no parameter update, optimizer or checkpoint was written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print the decision-relevant Teacher-v9.8.6 direction diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))
    print("=== TEACHER-v9.8.6 CROSS-SCENE DIRECTION DIAGNOSIS ===")
    print("status", value["status"], "selected", value["selected_candidate"])
    print("sealed cause: room_0102 Bed recall and MAE regressed at exact step 4")
    print("tasks: 12 Sit + 4 explicit-negative + 3 v5 = 19")
    print("conflicts", value["conflict_pair_count"], "/ 171")
    for row in value["direction_candidates"]:
        print(
            "\nCANDIDATE",
            row["name"],
            "eligible",
            row["eligible"],
            "audit-Bed-min",
            "{:.9f}".format(row["minimum_audit_bed_derivative"]),
            "all-task-min",
            "{:.9f}".format(row["minimum_directional_derivative"]),
        )
        print("failed", row["failed_checks"])
    print("\nfailed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: actual room_0101+0102 K=3 radius response grid only")
    else:
        print("next authority: objective/model-capacity redesign only")
    print("no candidate update or checkpoint was written")


if __name__ == "__main__":
    main()

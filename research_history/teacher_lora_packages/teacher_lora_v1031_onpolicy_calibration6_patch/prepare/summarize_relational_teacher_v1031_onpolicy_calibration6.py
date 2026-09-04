#!/usr/bin/env python3
"""Concise Teacher-v10.3.1 calibration summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from relational_teacher_v102_dense_instance_contract import EXPECTED_INSTANCES
from relational_teacher_v103_onpolicy_response_contract import PROMPT_IDS, ROLES, SCENES
from relational_teacher_v9_all_sittable_contract import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    print("=== TEACHER-v10.3.1 TWO-SCENE ON-POLICY K=3 CALIBRATION ===")
    print("status", value["status"], "shortlisted", value["shortlisted_steps"])
    print("absolute all-three remains diagnostic during this short calibration")
    for row in value["monitor_rows"]:
        print("\nSTEP", row["step"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        print("all-three", row["all_three_counts"])
        for scene in SCENES:
            print(scene)
            for prompt in PROMPT_IDS:
                values = []
                for role_index, role in enumerate(ROLES):
                    name = EXPECTED_INSTANCES[scene][role_index]
                    base = row["start_pooled"][scene][prompt]["instances"][name]
                    current = row["candidate_pooled"][scene][prompt]["instances"][name]
                    values.append(
                        "{} recall {:.5f}->{:.5f} MAE {:.5f}->{:.5f}".format(
                            role,
                            base["soft_recall"],
                            current["soft_recall"],
                            base["active_support_mae"],
                            current["active_support_mae"],
                        )
                    )
                print(prompt, "; ".join(values))
        print(
            "v5 change {:+.3f}%".format(
                100.0
                * (sum(row["candidate_v5_dense"]) / sum(row["start_v5_dense"]) - 1.0)
            )
        )
    print("\nfailed checks", value["failed_checks"])
    print(
        "next authority:",
        "extended on-policy training with actual K=3 monitoring"
        if value["status"] == "PASS"
        else "LoRA capacity/placement diagnosis",
    )
    print("checkpoint authorization: False")


if __name__ == "__main__":
    main()

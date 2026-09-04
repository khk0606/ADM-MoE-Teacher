#!/usr/bin/env python3
"""Readable diagnosis for Teacher-v9.8.2 rollout-state calibration-6."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from relational_teacher_v9_all_sittable_contract import read_json  # noqa: E402
from relational_teacher_v982_rollout_state_calibration6_contract import (  # noqa: E402
    OBJECTS,
    SCHEMA,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("not a Teacher-v9.8.2 calibration summary")

    labels = {"bed_01": "bed", "chair_01": "normal-chair", "chair_06": "high-chair"}
    print("=== TEACHER-v9.8.2 FRESH ROLLOUT-STATE CALIBRATION6 ===")
    print("status", value["status"], "shortlisted_steps", value["shortlisted_steps"])
    print("update rule: fresh/common-descent radius", value["step_radius"], "at t50")
    print("exact Top-k remains diagnostic only")
    for row in value["monitor_rows"]:
        print("\nSTEP", row["step"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        for name in OBJECTS:
            base = row["base_pooled"][name]
            candidate = row["candidate_pooled"][name]
            print(
                labels[name],
                "recall {:.6f}->{:.6f}".format(
                    base["soft_recall"], candidate["soft_recall"]
                ),
                "MAE {:.6f}->{:.6f}".format(
                    base["active_support_mae"], candidate["active_support_mae"]
                ),
                "topk(diag) {:.6f}->{:.6f}".format(
                    base["topk_overlap"], candidate["topk_overlap"]
                ),
            )
        base_v5 = sum(row["base_v5_dense"]) / 3.0
        candidate_v5 = sum(row["candidate_v5_dense"]) / 3.0
        print(
            "v5 {:.6f}->{:.6f} ({:+.3f}%)".format(
                base_v5, candidate_v5, 100.0 * (candidate_v5 / base_v5 - 1.0)
            )
        )
        for prompt_index, prompt in enumerate(("watch", "write")):
            all_three = sum(
                int(all(panel.values()))
                for panel in (row["presence"][generation][prompt_index] for generation in range(3))
            )
            print(prompt, "absolute all-three generations {}/3".format(all_three))
    print("\nartifact failed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: two-train-scene rollout-state calibration preflight only")
    else:
        print("next authority: calibration artifact audit only")


if __name__ == "__main__":
    main()

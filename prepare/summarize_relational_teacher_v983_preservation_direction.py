#!/usr/bin/env python3
"""Readable diagnosis for Teacher-v9.8.3 preservation direction preflight."""

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
from relational_teacher_v983_preservation_direction_contract import (  # noqa: E402
    OBJECTS,
    SCHEMA,
    TASK_ORDER,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.report.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("not a Teacher-v9.8.3 preservation report")

    labels = {"bed_01": "bed", "chair_01": "normal-chair", "chair_06": "high-chair"}
    print("=== TEACHER-v9.8.3 PRESERVATION-AWARE DIRECTION ===")
    print("status", value["status"], "selected", value["selected_candidate"])
    print("design: exact v9.8.1 update 1 plus one hypothetical update at t50")
    print("tasks: 6 Sit + 3 v5 replay + 2 explicit-negative = 11")
    print("exact Top-k remains diagnostic only")
    derivatives = value["direction"]["directional_derivatives"]
    print("minimum directional derivative", min(derivatives))
    print(
        "directional derivatives",
        {name: round(float(score), 8) for name, score in zip(TASK_ORDER, derivatives)},
    )
    base_v5 = value["base_v5_dense"]
    step1_v5 = value["step1_v5_dense"]
    targets = ("chair", "bed", "whiteboard")
    for row in value["candidates"]:
        print("\nCANDIDATE", row["name"], "radius", row["radius"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        for name in OBJECTS:
            first = row["step1_pooled"][name]
            candidate = row["candidate_pooled"][name]
            print(
                labels[name],
                "recall {:.6f}->{:.6f}".format(
                    first["soft_recall"], candidate["soft_recall"]
                ),
                "MAE {:.6f}->{:.6f}".format(
                    first["active_support_mae"], candidate["active_support_mae"]
                ),
                "topk(diag) {:.6f}->{:.6f}".format(
                    first["topk_overlap"], candidate["topk_overlap"]
                ),
            )
        for index, target in enumerate(targets):
            print(
                "v5-{} Base {:.6f} -> step1 {:.6f} -> candidate {:.6f}".format(
                    target,
                    base_v5[index],
                    step1_v5[index],
                    row["candidate_v5_dense"][index],
                )
            )
        print(
            "v5 mean change from Base {:+.3f}% and from step1 {:+.3f}%".format(
                100.0
                * (sum(row["candidate_v5_dense"]) / sum(base_v5) - 1.0),
                100.0
                * (sum(row["candidate_v5_dense"]) / sum(step1_v5) - 1.0),
            )
        )
        for generation in range(3):
            first_negative = max(
                row["step1_rows"][generation][prompt]["explicit_negative_mean"]
                for prompt in range(2)
            )
            candidate_negative = max(
                row["candidate_rows"][generation][prompt]["explicit_negative_mean"]
                for prompt in range(2)
            )
            print(
                "generation {} negative mean {:.6f}->{:.6f} ({:+.6f})".format(
                    generation,
                    first_negative,
                    candidate_negative,
                    candidate_negative - first_negative,
                )
            )
        for prompt_index, prompt in enumerate(("watch", "write")):
            all_three = sum(
                int(all(row["presence"][generation][prompt_index].values()))
                for generation in range(3)
            )
            print(prompt, "absolute all-three generations {}/3 (diagnostic)".format(all_three))

    print("\nartifact failed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: fresh preservation-aware rollout-state calibration6 only")
    else:
        print("next authority: this preservation preflight artifact audit only")


if __name__ == "__main__":
    main()

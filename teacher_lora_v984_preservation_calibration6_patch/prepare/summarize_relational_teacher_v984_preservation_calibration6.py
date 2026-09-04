#!/usr/bin/env python3
"""Readable diagnosis for Teacher-v9.8.4 preservation calibration-6."""

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
from relational_teacher_v984_preservation_calibration6_contract import (  # noqa: E402
    OBJECTS,
    SCHEMA,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.summary.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("not a Teacher-v9.8.4 preservation calibration summary")

    labels = {"bed_01": "bed", "chair_01": "normal-chair", "chair_06": "high-chair"}
    print("=== TEACHER-v9.8.4 FRESH PRESERVATION CALIBRATION6 ===")
    print("status", value["status"], "shortlisted_steps", value["shortlisted_steps"])
    print("update 1: exact v9.8.1 six-Sit direction")
    print("updates 2-6: 6 Sit + 3 v5 + 2 negative common descent")
    print("actual monitor: K=3 x watch/write at t50-to-t0 after every update")
    print("exact Top-k and absolute all-three remain diagnostic only")

    for direction, row in zip(value["direction_rows"], value["monitor_rows"]):
        print("\nSTEP", row["step"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        print("minimum directional derivative", min(direction["directional_derivatives"]))
        for name in OBJECTS:
            previous = row["previous_pooled"][name]
            candidate = row["candidate_pooled"][name]
            print(
                labels[name],
                "recall {:.6f}->{:.6f}".format(
                    previous["soft_recall"], candidate["soft_recall"]
                ),
                "MAE {:.6f}->{:.6f}".format(
                    previous["active_support_mae"], candidate["active_support_mae"]
                ),
                "topk(diag) {:.6f}->{:.6f}".format(
                    previous["topk_overlap"], candidate["topk_overlap"]
                ),
            )
        for index, target in enumerate(("chair", "bed", "whiteboard")):
            print(
                "v5-{} Base {:.6f} previous {:.6f} candidate {:.6f}".format(
                    target,
                    row["base_v5_dense"][index],
                    row["previous_v5_dense"][index],
                    row["candidate_v5_dense"][index],
                )
            )
        print(
            "v5 mean change from Base {:+.3f}% and previous {:+.3f}%".format(
                100.0
                * (sum(row["candidate_v5_dense"]) / sum(row["base_v5_dense"]) - 1.0),
                100.0
                * (
                    sum(row["candidate_v5_dense"])
                    / sum(row["previous_v5_dense"])
                    - 1.0
                ),
            )
        )
        for generation in range(3):
            previous_negative = max(
                row["previous_rows"][generation][prompt]["explicit_negative_mean"]
                for prompt in range(2)
            )
            candidate_negative = max(
                row["candidate_rows"][generation][prompt]["explicit_negative_mean"]
                for prompt in range(2)
            )
            print(
                "generation {} negative mean {:.6f}->{:.6f} ({:+.6f})".format(
                    generation,
                    previous_negative,
                    candidate_negative,
                    candidate_negative - previous_negative,
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
        print("next authority: two-train-scene preservation calibration preflight only")
    else:
        print("next authority: this calibration artifact audit only")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Readable diagnosis for a Teacher-v9.8 rollout-state response report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OBJECTS = ("bed_01", "chair_01", "chair_06")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))

    print("=== TEACHER-v9.8 ROLLOUT-STATE RESPONSE ===")
    print("status", value["status"], "selected_candidate", value["selected_candidate"])
    print("decision unit: resumed final 500-step maps; exact Top-k is diagnostic only")
    for row in value["candidates"]:
        print(
            "\nCANDIDATE",
            row["name"],
            "timestep",
            row["capture_timestep"],
            "radius",
            row["radius"],
            "eligible",
            row["eligible"],
        )
        print("failed", row["failed_checks"])
        for name in OBJECTS:
            base = row["base_pooled"][name]
            candidate = row["candidate_pooled"][name]
            print(
                name,
                "recall {:.6f}->{:.6f}".format(
                    base["soft_recall"], candidate["soft_recall"]
                ),
                "MAE {:.6f}->{:.6f}".format(
                    base["active_support_mae"],
                    candidate["active_support_mae"],
                ),
                "topk(diag) {:.6f}->{:.6f}".format(
                    base["topk_overlap"], candidate["topk_overlap"]
                ),
            )
        print(
            "prompt invariance {:.9f}->{:.9f}".format(
                row["base_prompt_invariance"],
                row["candidate_prompt_invariance"],
            )
        )
        base_v5 = sum(row["base_v5_dense"]) / 3.0
        candidate_v5 = sum(row["candidate_v5_dense"]) / 3.0
        relative = candidate_v5 / base_v5 - 1.0
        print(
            "v5 {:.6f}->{:.6f} ({:+.3f}%)".format(
                base_v5, candidate_v5, relative * 100.0
            )
        )
        print(
            "direction min {:.9g}; final max |delta| {:.9g}".format(
                min(row["directional_derivatives"]),
                row["maximum_final_map_delta"],
            )
        )

    print("\nfailed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: selected rollout-state response-6 confirmation only")
    else:
        print("next authority: failure audit only; no checkpoint or held-out scene")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Human-readable summary of the sealed Teacher-v9.7 K=3 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OBJECTS = ("bed_01", "chair_01", "chair_06")
PROMPTS = ("watch", "write")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    print("=== TEACHER-v9.7 ACTUAL 500-STEP K=3 ===")
    print("status", value["status"], "selected_step", value["selected_step"])
    print("exact_topk", "diagnostic only")
    for row in value["step_rows"]:
        print("\nSTEP", row["step"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        for name in OBJECTS:
            base = row["base_pooled"][name]
            candidate = row["candidate_pooled"][name]
            print(
                name,
                "recall {:.6f}->{:.6f}".format(base["soft_recall"], candidate["soft_recall"]),
                "MAE {:.6f}->{:.6f}".format(
                    base["active_support_mae"], candidate["active_support_mae"]
                ),
                "topk(diag) {:.6f}->{:.6f}".format(
                    base["topk_overlap"], candidate["topk_overlap"]
                ),
            )
        object_keys = [
            key
            for key in row["candidate_presence"][0][0]
            if not key.startswith("explicit_negative_")
        ]
        for prompt_index, prompt in enumerate(PROMPTS):
            passes = sum(
                all(panel[prompt_index][key] for key in object_keys)
                for panel in row["candidate_presence"]
            )
            print(prompt, "all-three generations", "{}/3".format(passes))
        print(
            "prompt invariance",
            [round(number, 9) for number in row["base_prompt_invariance"]],
            "->",
            [round(number, 9) for number in row["candidate_prompt_invariance"]],
        )
        print(
            "v5 fixed probe mean {:.6f}->{:.6f}".format(
                sum(row["base_v5_dense"]) / 3.0,
                sum(row["candidate_v5_dense"]) / 3.0,
            )
        )
    print("\nfailed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: locked selected-step confirmation only")
    else:
        print("next authority: rollout-state/on-policy preflight only")


if __name__ == "__main__":
    main()

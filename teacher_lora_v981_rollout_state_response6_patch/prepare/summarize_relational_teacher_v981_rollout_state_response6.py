#!/usr/bin/env python3
"""Readable summary for Teacher-v9.8.1 rollout-state response-6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OBJECTS = ("bed_01", "chair_01", "chair_06")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    print("=== TEACHER-v9.8.1 SELECTED ROLLOUT-STATE K=3 RESPONSE ===")
    print("status", value["status"], "candidate", value["selected_candidate"])
    print("activation: frozen Base t499..51, selected LoRA t50..0")

    for generation in range(3):
        print("\nGENERATION", generation)
        for name in OBJECTS:
            base_recall = sum(
                value["base_rows"][generation][p]["instances"][name]["soft_recall"]
                for p in range(2)
            ) / 2.0
            candidate_recall = sum(
                value["candidate_rows"][generation][p]["instances"][name]["soft_recall"]
                for p in range(2)
            ) / 2.0
            base_mae = sum(
                value["base_rows"][generation][p]["instances"][name]["active_support_mae"]
                for p in range(2)
            ) / 2.0
            candidate_mae = sum(
                value["candidate_rows"][generation][p]["instances"][name]["active_support_mae"]
                for p in range(2)
            ) / 2.0
            print(
                name,
                "recall {:.6f}->{:.6f}".format(base_recall, candidate_recall),
                "MAE {:.6f}->{:.6f}".format(base_mae, candidate_mae),
            )
        print(
            "prompt invariance {:.9f}->{:.9f}".format(
                value["base_prompt_invariance"][generation],
                value["candidate_prompt_invariance"][generation],
            )
        )

    print("\nPOOLED")
    for name in OBJECTS:
        base = value["base_pooled"][name]
        candidate = value["candidate_pooled"][name]
        print(
            name,
            "recall {:.6f}->{:.6f}".format(base["soft_recall"], candidate["soft_recall"]),
            "MAE {:.6f}->{:.6f}".format(
                base["active_support_mae"], candidate["active_support_mae"]
            ),
            "topk(diag) {:.6f}->{:.6f}".format(base["topk_overlap"], candidate["topk_overlap"]),
        )
    base_v5 = sum(value["base_v5_dense"]) / 3.0
    candidate_v5 = sum(value["candidate_v5_dense"]) / 3.0
    print("v5 {:.6f}->{:.6f} ({:+.3f}%)".format(
        base_v5, candidate_v5, (candidate_v5 / base_v5 - 1.0) * 100.0
    ))
    for prompt, name in enumerate(("watch", "write")):
        count = 0
        for generation in range(3):
            row = value["candidate_presence"][generation][prompt]
            object_checks = [passed for key, passed in row.items() if not key.startswith("explicit_negative_")]
            count += int(all(object_checks))
        print(name, "absolute all-three generations", "{}/3".format(count))
    failed_response = sorted(
        name for name, passed in value["response_checks"].items() if not passed
    )
    print("response failed checks", failed_response)
    print("artifact failed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: fresh rollout-state calibration6 only")
    else:
        print("next authority: response-6 failure audit only")


if __name__ == "__main__":
    main()

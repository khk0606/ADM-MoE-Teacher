#!/usr/bin/env python3
"""Readable diagnosis for Teacher-v9.8.5 two-scene step-4 preflight."""

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
from relational_teacher_v985_two_scene_step4_preflight_contract import (  # noqa: E402
    ROLES,
    SCHEMA,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.report.expanduser().resolve())
    if value.get("schema") != SCHEMA:
        raise ValueError("not a Teacher-v9.8.5 two-scene preflight report")

    print("=== TEACHER-v9.8.5 STEP-4 CROSS-SCENE AUDIT ===")
    print("status", value["status"], "audit_scene", value["audit_scene"])
    print("state: fresh v5r4 -> exact v9.8.4 updates 1,2,3,4")
    print("actual audit: room_0102 K=3 x watch/write, frozen Base vs step 4")
    print("exact Top-k and absolute all-three are diagnostic only")
    for role in ROLES:
        before = value["base_pooled"][role]
        after = value["candidate_pooled"][role]
        print(
            role,
            "recall {:.6f}->{:.6f}".format(before["soft_recall"], after["soft_recall"]),
            "MAE {:.6f}->{:.6f}".format(
                before["active_support_mae"], after["active_support_mae"]
            ),
            "topk(diag) {:.6f}->{:.6f}".format(
                before["topk_overlap"], after["topk_overlap"]
            ),
        )
    print(
        "v5 mean change {:+.3f}%".format(
            100.0
            * (sum(value["candidate_v5_dense"]) / sum(value["base_v5_dense"]) - 1.0)
        )
    )
    print(
        "prompt invariance ratios",
        [
            round(candidate / base, 6) if base > 0.0 else 0.0
            for base, candidate in zip(
                value["base_prompt_invariance"], value["candidate_prompt_invariance"]
            )
        ],
    )
    for generation in range(3):
        before_mean = max(
            value["base_rows"][generation][prompt]["explicit_negative_mean"]
            for prompt in range(2)
        )
        after_mean = max(
            value["candidate_rows"][generation][prompt]["explicit_negative_mean"]
            for prompt in range(2)
        )
        print(
            "generation {} negative mean {:.6f}->{:.6f} ({:+.6f})".format(
                generation, before_mean, after_mean, after_mean - before_mean
            )
        )
    for prompt_index, prompt in enumerate(("watch", "write")):
        all_three = sum(
            int(all(value["presence"][generation][prompt_index].values()))
            for generation in range(3)
        )
        print(prompt, "absolute all-three generations {}/3 (diagnostic)".format(all_three))
    print("audit failed checks", value["audit_failed_checks"])
    print("artifact failed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: fresh two-scene preservation response preflight only")
    else:
        print("next authority: cross-scene direction diagnosis only")


if __name__ == "__main__":
    main()

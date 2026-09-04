#!/usr/bin/env python3
"""Print the decision-relevant Teacher-v9.8.12 calibration results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PREPARE_ROOT = Path(__file__).resolve().parent
if str(PREPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPARE_ROOT))

from relational_teacher_v9_all_sittable_contract import read_json  # noqa: E402
from relational_teacher_v9812_two_scene_multiupdate_calibration_contract import (  # noqa: E402
    AUDIT_SCENE,
    SOURCE_SCENE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _role_line(label: str, before: dict, after: dict) -> str:
    return (
        "{} recall {:.6f}->{:.6f} MAE {:.6f}->{:.6f} topk(diag) {:.6f}->{:.6f}"
    ).format(
        label,
        float(before["soft_recall"]),
        float(after["soft_recall"]),
        float(before["active_support_mae"]),
        float(after["active_support_mae"]),
        float(before["topk_overlap"]),
        float(after["topk_overlap"]),
    )


def main() -> None:
    value = read_json(parse_args().summary.expanduser().resolve())
    print("=== TEACHER-v9.8.12 TWO-SCENE MULTI-UPDATE CALIBRATION ===")
    print("status", value["status"], "shortlisted_steps", value["shortlisted_steps"])
    print(
        "start fixed radius",
        value["selected_radius"],
        "update radius",
        value["step_radius"],
    )
    print("absolute all-three is a hard shortlist gate; exact Top-k is diagnostic")
    start = value["scene_start_pooled"]
    base_v5 = sum(float(x) for x in value["base_v5_dense"]) / 3.0
    for row in value["monitor_rows"]:
        print("\nUPDATE", row["step"], "eligible", row["eligible"])
        print("direction", row["selected_direction_candidate"])
        print("failed", row["failed_checks"])
        for scene in (SOURCE_SCENE, AUDIT_SCENE):
            print("SCENE", scene, "strict_failed", row["scene_failed_checks"][scene])
            for role in ("bed", "normal_chair", "high_chair"):
                print(
                    _role_line(
                        role,
                        start[scene][role],
                        row["scene_candidate_pooled"][scene][role],
                    )
                )
            print("all-three", row["all_three_counts"][scene])
        candidate_v5 = sum(float(x) for x in row["candidate_v5_dense"]) / 3.0
        print("v5 change {:+.6f}%".format(100.0 * (candidate_v5 / base_v5 - 1.0)))
    print("\nshortlisted state hashes", value["shortlisted_state_sha256"])
    print("failed checks", value["failed_checks"])
    if value["status"] == "PASS":
        print("next authority: shortlisted-state reconstruction/checkpoint-export gate only")
    else:
        print("next authority: cross-scene objective or model-capacity redesign only")
    print("no checkpoint was written")


if __name__ == "__main__":
    main()

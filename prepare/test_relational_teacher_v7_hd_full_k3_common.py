#!/usr/bin/env python3
"""CPU arithmetic test for Teacher-v7 full K=3 metric composition."""

from __future__ import annotations

import numpy as np

from relational_teacher_v6_contract import (
    PROMPTS,
    SITTABLE_CATEGORY_IDS,
    SIT_NEGATIVE_OBJECT_CATEGORY_IDS,
)
from relational_teacher_v7_hd_full_k3_common import (
    per_generation_metrics,
    pooled_metrics,
)


def main() -> None:
    scene_ids, target_names, strata, motion_ids, prompt_ids = [], [], [], [], []
    all_strata = ("bed_01", "high_desk_chair_06", "legacy_chair_06", "other")
    for scene_id in ("room_0101", "room_0102"):
        for stratum in all_strata:
            for motion in range(6):
                for prompt_id in sorted(PROMPTS):
                    scene_ids.append(scene_id)
                    target_names.append("chair_06")
                    strata.append(stratum)
                    motion_ids.append(f"{scene_id}_{stratum}_{motion}")
                    prompt_ids.append(prompt_id)
    count = len(scene_ids)
    assert count == 96
    point_count = 4
    sittable = sorted(SITTABLE_CATEGORY_IDS)[0]
    negative = sorted(SIT_NEGATIVE_OBJECT_CATEGORY_IDS)[0]
    reference = {
        "rel_scene_ids": np.asarray(scene_ids),
        "rel_target_names": np.asarray(target_names),
        "rel_strata": np.asarray(strata),
        "rel_motion_ids": np.asarray(motion_ids),
        "rel_prompt_ids": np.asarray(prompt_ids),
        "rel_target_instance_ids": np.ones(count, dtype=np.int64),
        "rel_gt": np.zeros((count, point_count, 6), dtype=np.float32),
        "rel_instance_ids": np.tile(
            np.asarray([[1, 1, 2, 2]], dtype=np.int64), (count, 1)
        ),
        "rel_category_ids": np.tile(
            np.asarray([[sittable, sittable, negative, negative]], dtype=np.int64),
            (count, 1),
        ),
        "v5_ids": np.asarray([f"v5_{index:02d}" for index in range(25)]),
        "v5_targets": np.asarray(["chair"] * 18 + ["bed"] + ["whiteboard"] * 6),
        "v5_gt": np.zeros((25, point_count, 6), dtype=np.float32),
    }
    rel_base = np.full((count, 3, point_count, 6), 0.20, dtype=np.float32)
    rel_candidate = np.full((count, 3, point_count, 6), 0.19, dtype=np.float32)
    v5_base = np.full((25, 3, point_count, 6), 0.20, dtype=np.float32)
    v5_candidate = np.full((25, 3, point_count, 6), 0.19, dtype=np.float32)
    generations = per_generation_metrics(
        reference, rel_base, rel_candidate, v5_base, v5_candidate
    )
    assert len(generations) == 3
    assert all(row["overall"]["count"] == 96 for row in generations)
    assert all(row["high_desk"]["count"] == 24 for row in generations)
    assert all(len(row["prompt_pairs"]) == 48 for row in generations)
    assert all(len(row["v5_replay_cases"]) == 25 for row in generations)
    assert all(len(row["target_groups"]) == 8 for row in generations)
    pooled = pooled_metrics(generations)
    assert pooled["overall"]["count"] == 288
    assert pooled["high_desk"]["count"] == 72
    assert len(pooled["prompt_pairs"]) == 144
    assert len(pooled["v5_replay_cases"]) == 75
    assert set(pooled["v5_target_groups"]) == {"chair", "bed", "whiteboard"}
    print("[PASS] Teacher-v7 48-motion K=3 metric inventory arithmetic")
    print("[PASS] eight strata, High-Desk, prompt-pair and v5 pooled counts")


if __name__ == "__main__":
    main()

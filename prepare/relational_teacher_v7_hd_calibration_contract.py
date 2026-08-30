#!/usr/bin/env python3
"""CPU-verifiable batching and objective constants for Teacher-v7 calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

from relational_teacher_v6_contract import PROMPTS
from relational_teacher_v7_hd_preflight_contract import EXPECTED_TRAIN_SCENES


SCHEMA = "relational_teacher_v7_hd_lora_calibration_v1"
STEPS = 12
TRAINING_STRATA_PER_SCENE = 4

NEW_DENSE_WEIGHT = 0.35
V5_REPLAY_WEIGHT = 0.65
CANDIDATE_WEIGHT = 0.5
NEGATIVE_WEIGHT = 0.35
INVARIANCE_WEIGHT = 0.5
PRESERVATION_WEIGHT = 1.25
LORA_WEIGHT = 1e-4
DEFAULT_LR = 4e-5

EXPECTED_WEIGHTS = {
    "new_dense_within_combined": NEW_DENSE_WEIGHT,
    "v5_replay_dense_within_combined": V5_REPLAY_WEIGHT,
    "combined_dense": 1.0,
    "candidate_semantic": CANDIDATE_WEIGHT,
    "negative_object_suppression": NEGATIVE_WEIGHT,
    "watch_write_invariance": INVARIANCE_WEIGHT,
    "frozen_v5_preservation": PRESERVATION_WEIGHT,
    "lora": LORA_WEIGHT,
}


def classify_training_stratum(motion_id: str, target_instance_id: str) -> str:
    if "_hc_hd_" in motion_id:
        if target_instance_id != "chair_06":
            raise ValueError("hc_hd row does not target chair_06")
        return "high_desk_chair_06"
    if target_instance_id == "chair_06":
        return "legacy_chair_06"
    return target_instance_id


def group_relational_rows(
    rows: Mapping[str, Sequence[Mapping[str, object]]]
) -> Dict[str, Dict[str, List[Mapping[str, object]]]]:
    grouped: Dict[str, Dict[str, List[Mapping[str, object]]]] = {}
    for scene_id, scene_rows in rows.items():
        by_stratum: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
        for row in scene_rows:
            by_stratum[str(row["training_stratum"])].append(row)
        other_chair = "chair_01" if scene_id == "room_0101" else "chair_05"
        expected = {
            "bed_01",
            "high_desk_chair_06",
            "legacy_chair_06",
            other_chair,
        }
        if set(by_stratum) != expected:
            raise ValueError(f"{scene_id}: Teacher-v7 stratum identity changed")
        if Counter(len(values) for values in by_stratum.values()) != Counter({6: 4}):
            raise ValueError(f"{scene_id}: Teacher-v7 stratum counts changed")
        grouped[scene_id] = {
            stratum: sorted(values, key=lambda row: str(row["motion_id"]))
            for stratum, values in sorted(by_stratum.items())
        }
    if set(grouped) != set(EXPECTED_TRAIN_SCENES):
        raise ValueError("Teacher-v7 train scene inventory changed")
    return grouped


def select_relational_batch_rows(
    grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    step: int,
) -> Tuple[List[Tuple[Mapping[str, object], str]], Dict[str, object]]:
    if not 1 <= step <= STEPS:
        raise ValueError("Teacher-v7 calibration step is outside 1..12")
    prompt_ids = sorted(PROMPTS)
    selected: List[Tuple[Mapping[str, object], str]] = []
    main_rows: List[Mapping[str, object]] = []
    audit: Dict[str, object] = {"step": step, "main": [], "pairs": []}
    for scene_offset, scene_id in enumerate(EXPECTED_TRAIN_SCENES):
        strata = sorted(grouped[scene_id])
        for stratum_offset, stratum in enumerate(strata):
            values = grouped[scene_id][stratum]
            row = values[(step - 1 + scene_offset + stratum_offset) % len(values)]
            prompt_id = prompt_ids[
                (step - 1 + scene_offset + stratum_offset) % len(prompt_ids)
            ]
            selected.append((row, prompt_id))
            main_rows.append(row)
            audit["main"].append(
                {
                    "scene_id": scene_id,
                    "motion_id": str(row["motion_id"]),
                    "target_instance_id": str(row["target_instance_id"]),
                    "training_stratum": stratum,
                    "is_high_desk": bool(row["is_high_desk"]),
                    "prompt_id": prompt_id,
                }
            )
    for scene_offset, scene_id in enumerate(EXPECTED_TRAIN_SCENES):
        stratum = "high_desk_chair_06"
        values = grouped[scene_id][stratum]
        row = values[(step - 1 + scene_offset) % len(values)]
        for prompt_id in prompt_ids:
            selected.append((row, prompt_id))
        audit["pairs"].append(
            {
                "scene_id": scene_id,
                "motion_id": str(row["motion_id"]),
                "target_instance_id": str(row["target_instance_id"]),
                "training_stratum": stratum,
                "prompt_ids": prompt_ids,
            }
        )
    if len(selected) != 12 or len(main_rows) != 8:
        raise AssertionError("Teacher-v7 relational calibration batch size changed")
    return selected, audit


def validate_batch_audits(audits: Sequence[Mapping[str, object]]) -> Dict[str, bool]:
    return {
        "every_update_uses_two_train_scenes_and_four_strata": all(
            len(row["main"]) == 8
            and len(row["pairs"]) == 2
            and Counter(item["scene_id"] for item in row["main"])
            == Counter({"room_0101": 4, "room_0102": 4})
            and all(
                Counter(
                    item["training_stratum"]
                    for item in row["main"]
                    if item["scene_id"] == scene_id
                )
                == Counter(
                    {
                        "bed_01": 1,
                        "high_desk_chair_06": 1,
                        "legacy_chair_06": 1,
                        "chair_01" if scene_id == "room_0101" else "chair_05": 1,
                    }
                )
                for scene_id in EXPECTED_TRAIN_SCENES
            )
            for row in audits
        ),
        "every_update_has_high_desk_main_and_prompt_pairs": all(
            sum(bool(item["is_high_desk"]) for item in row["main"]) == 2
            and all(
                item["training_stratum"] == "high_desk_chair_06"
                and item["prompt_ids"] == ["sit_watch_v1", "sit_write_v1"]
                for item in row["pairs"]
            )
            for row in audits
        ),
        "all_12_train_high_desk_motions_covered": len(
            {
                (item["scene_id"], item["motion_id"])
                for row in audits
                for item in row["main"]
                if item["is_high_desk"]
            }
        )
        == 12,
    }

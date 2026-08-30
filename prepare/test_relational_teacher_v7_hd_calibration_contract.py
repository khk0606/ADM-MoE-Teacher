#!/usr/bin/env python3
"""CPU-only tests for Teacher-v7 High-Desk calibration batching."""

from __future__ import annotations

from copy import deepcopy

from relational_teacher_v7_hd_calibration_contract import (
    STEPS,
    classify_training_stratum,
    group_relational_rows,
    select_relational_batch_rows,
    validate_batch_audits,
)


def synthetic_rows():
    result = {}
    for scene_id, other_chair in (
        ("room_0101", "chair_01"),
        ("room_0102", "chair_05"),
    ):
        rows = []
        for target, prefix in (
            ("bed_01", "legacy_bed"),
            (other_chair, "legacy_other_chair"),
            ("chair_06", "legacy_chair06"),
            ("chair_06", "hc_hd_new"),
        ):
            for index in range(1, 7):
                motion_id = f"{scene_id}_{prefix}_{index:02d}"
                if prefix == "hc_hd_new":
                    motion_id = f"{scene_id}_hc_hd_source_{index:02d}"
                rows.append(
                    {
                        "scene_id": scene_id,
                        "motion_id": motion_id,
                        "target_instance_id": target,
                        "training_stratum": classify_training_stratum(
                            motion_id, target
                        ),
                        "is_high_desk": "_hc_hd_" in motion_id,
                    }
                )
        result[scene_id] = rows
    return result


def test_exact_four_strata_and_high_desk_coverage() -> None:
    grouped = group_relational_rows(synthetic_rows())
    audits = []
    for step in range(1, STEPS + 1):
        selected, audit = select_relational_batch_rows(grouped, step)
        assert len(selected) == 12
        assert len(audit["main"]) == 8
        assert len(audit["pairs"]) == 2
        audit["v5_replay_ids"] = ["chair", "bed", "whiteboard"]
        audits.append(audit)
    checks = validate_batch_audits(audits)
    assert checks and all(checks.values())
    print("[PASS] v7 four-stratum B=12 schedule and all High-Desk rows covered")


def test_tamper_guards() -> None:
    rows = synthetic_rows()
    rows["room_0101"][0]["training_stratum"] = "forged"
    try:
        group_relational_rows(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("forged stratum was accepted")

    grouped = group_relational_rows(synthetic_rows())
    audits = []
    for step in range(1, STEPS + 1):
        _, audit = select_relational_batch_rows(grouped, step)
        audits.append(audit)
    tampered = deepcopy(audits)
    tampered[0]["main"][0]["is_high_desk"] = True
    assert not all(validate_batch_audits(tampered).values())
    print("[PASS] v7 stratum and High-Desk audit tamper guards")


if __name__ == "__main__":
    test_exact_four_strata_and_high_desk_coverage()
    test_tamper_guards()

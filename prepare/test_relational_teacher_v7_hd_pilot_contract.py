#!/usr/bin/env python3
"""CPU-only schedule, eligibility and ranking tests for the v7 pilot."""

from __future__ import annotations

from copy import deepcopy

from relational_teacher_v7_hd_calibration_contract import (
    group_relational_rows,
    validate_batch_audits,
)
from relational_teacher_v7_hd_pilot_contract import (
    CHECKPOINT_STEPS,
    STEPS,
    checkpoint_eligibility,
    rank_shortlist,
    select_pilot_batch_rows,
)
from test_relational_teacher_v7_hd_calibration_contract import synthetic_rows


def test_60_update_schedule() -> None:
    grouped = group_relational_rows(synthetic_rows())
    audits = []
    for step in range(1, STEPS + 1):
        selected, audit = select_pilot_batch_rows(grouped, step)
        assert len(selected) == 12
        assert audit["step"] == step
        audit["v5_replay_ids"] = ["chair", "bed", "whiteboard"]
        audits.append(audit)
    assert all(validate_batch_audits(audits).values())
    assert CHECKPOINT_STEPS == (12, 24, 36, 48, 60)
    assert audits[0]["main"] == audits[12]["main"]
    print("[PASS] v7 pilot 60-update cyclic four-stratum schedule")


def test_eligibility_and_ranking() -> None:
    initial = {
        "semantic_total": 1.0,
        "high_desk_dense": 2.0,
        "invariance": 0.1,
        "new_dense": 1.0,
        "negative": 0.4,
        "v5_replay_dense": 0.2,
        "preservation": 0.0,
        "lora_energy": 0.0,
    }
    good = {
        "semantic_total": 0.9,
        "high_desk_dense": 1.8,
        "invariance": 0.101,
        "new_dense": 0.99,
        "negative": 0.39,
        "v5_replay_dense": 0.201,
        "preservation": 0.0005,
        "lora_energy": 1e-8,
        "objective_proxy": 0.8,
    }
    assert all(checkpoint_eligibility(good, initial).values())
    bad = deepcopy(good)
    bad["invariance"] = 0.103
    assert not checkpoint_eligibility(bad, initial)["watch_write_invariance_not_worse"]
    records = [
        {"step": 12, "eligible": True, "fixed_panel": dict(good)},
        {"step": 24, "eligible": False, "fixed_panel": dict(bad)},
        {"step": 36, "eligible": True, "fixed_panel": {**good, "objective_proxy": 0.7}},
        {"step": 48, "eligible": True, "fixed_panel": {**good, "objective_proxy": 0.9}},
    ]
    assert rank_shortlist(records) == (36, 12)
    print("[PASS] v7 pilot joint High-Desk/invariance/v5 gates and top-two rank")


if __name__ == "__main__":
    test_60_update_schedule()
    test_eligibility_and_ranking()

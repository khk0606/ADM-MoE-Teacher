#!/usr/bin/env python3
"""CPU-verifiable schedule and eligibility for the Teacher-v7 pilot."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

from relational_teacher_v7_hd_calibration_contract import (
    STEPS as CALIBRATION_STEPS,
    select_relational_batch_rows,
)


SCHEMA = "relational_teacher_v7_hd_lora_pilot_v1"
STEPS = 60
CHECKPOINT_STEPS = (12, 24, 36, 48, 60)


def select_pilot_batch_rows(grouped, step: int):
    if not 1 <= step <= STEPS:
        raise ValueError("Teacher-v7 pilot step is outside 1..60")
    cycle_step = ((step - 1) % CALIBRATION_STEPS) + 1
    selected, audit = select_relational_batch_rows(grouped, cycle_step)
    audit = dict(audit)
    audit["step"] = step
    return selected, audit


def checkpoint_eligibility(
    panel: Mapping[str, float], initial: Mapping[str, float]
) -> Mapping[str, bool]:
    eps = 1e-12
    return {
        "semantic_total_improves": panel["semantic_total"]
        < initial["semantic_total"] - eps,
        "high_desk_dense_improves": panel["high_desk_dense"]
        < initial["high_desk_dense"] - eps,
        "watch_write_invariance_not_worse": panel["invariance"]
        <= initial["invariance"] * 1.02 + eps,
        "new_dense_not_worse": panel["new_dense"]
        <= initial["new_dense"] * 1.02 + eps,
        "negative_suppression_not_worse": panel["negative"]
        <= initial["negative"] * 1.02 + eps,
        "v5_replay_retained": panel["v5_replay_dense"]
        <= initial["v5_replay_dense"] * 1.01 + eps,
        "preservation_bounded": panel["preservation"] <= 1e-3,
        "lora_changed_from_zero": panel["lora_energy"] > 0.0,
    }


def rank_shortlist(records: Sequence[Mapping[str, object]]) -> Tuple[int, ...]:
    eligible = [row for row in records if row["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            float(row["fixed_panel"]["objective_proxy"]),
            float(row["fixed_panel"]["high_desk_dense"]),
            float(row["fixed_panel"]["semantic_total"]),
            int(row["step"]),
        ),
    )
    return tuple(int(row["step"]) for row in ranked[:2])

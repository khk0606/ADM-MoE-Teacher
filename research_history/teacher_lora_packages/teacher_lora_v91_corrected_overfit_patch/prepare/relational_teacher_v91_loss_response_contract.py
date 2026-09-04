#!/usr/bin/env python3
"""Pure selection contract for the Teacher-v9.1 loss-response grid."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v91_active_support_response_grid_v1"
FAILED_SCHEMA = "relational_teacher_v9_all_sittable_one_scene_overfit_v1"
TRAIN_SCENE = "room_0101"
SEED = 20261007
STEPS_PER_CANDIDATE = 3
LEARNING_RATE = 4e-5
GRAD_CLIP = 1.0
CANDIDATES = (
    {
        "name": "support2_rank025_preserve2",
        "active_support_weight": 2.0,
        "ranking_weight": 0.25,
        "negative_weight": 1.0,
        "prompt_weight": 2.0,
        "v5_preservation_weight": 2.0,
    },
    {
        "name": "support4_rank050_preserve4",
        "active_support_weight": 4.0,
        "ranking_weight": 0.50,
        "negative_weight": 2.0,
        "prompt_weight": 2.0,
        "v5_preservation_weight": 4.0,
    },
    {
        "name": "support8_rank100_preserve8",
        "active_support_weight": 8.0,
        "ranking_weight": 1.00,
        "negative_weight": 4.0,
        "prompt_weight": 4.0,
        "v5_preservation_weight": 8.0,
    },
)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def response_checks(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
) -> Dict[str, bool]:
    """Gate differentiable response without reusing impossible absolute top-k."""

    checks = {
        "active_support_macro_decreases": _finite(
            after["active_support_macro"], "after active support"
        )
        < _finite(before["active_support_macro"], "before active support"),
        "ranking_surrogate_decreases": _finite(
            after["within_object_ranking"], "after ranking"
        )
        < _finite(before["within_object_ranking"], "before ranking"),
        "bed_active_support_decreases": _finite(
            after["per_instance_active_support"][0], "after Bed support"
        )
        < _finite(before["per_instance_active_support"][0], "before Bed support"),
        "normal_chair_active_support_decreases": _finite(
            after["per_instance_active_support"][1], "after normal Chair support"
        )
        < _finite(before["per_instance_active_support"][1], "before normal Chair support"),
        "high_chair_active_support_decreases": _finite(
            after["per_instance_active_support"][2], "after High Chair support"
        )
        < _finite(before["per_instance_active_support"][2], "before High Chair support"),
        "prompt_invariance_retained_2pct": _finite(
            after["prompt_invariance"], "after prompt"
        )
        <= _finite(before["prompt_invariance"], "before prompt") * 1.02 + 1e-8,
        "negative_mean_addition_bounded": _finite(
            after["negative_mean"], "after negative"
        )
        <= _finite(before["negative_mean"], "before negative") + 0.002,
    }
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 response panel must contain three targets")
    base_v5 = sum(_finite(value, "base v5") for value in base_v5_dense) / 3.0
    candidate_v5 = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) / 3.0
    checks["v5_replay_retained_1pct"] = candidate_v5 <= base_v5 * 1.01 + 1e-8
    return checks


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    expected_names = [str(row["name"]) for row in CANDIDATES]
    by_name = {str(row.get("name")): row for row in rows}
    if list(by_name) != expected_names or len(by_name) != len(CANDIDATES):
        raise ValueError("loss-response candidate grid/order changed")
    eligible = [row for row in rows if row.get("eligible") is True]
    eligible.sort(
        key=lambda row: (
            _finite(row["after"]["active_support_macro"], "active response"),
            _finite(row["after"]["within_object_ranking"], "ranking response"),
            sum(_finite(value, "candidate v5") for value in row["candidate_v5_dense"])
            / 3.0,
            expected_names.index(str(row["name"])),
        )
    )
    return [str(row["name"]) for row in eligible]


__all__ = [
    "CANDIDATES",
    "FAILED_SCHEMA",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "SCHEMA",
    "SEED",
    "STEPS_PER_CANDIDATE",
    "TRAIN_SCENE",
    "rank_candidates",
    "response_checks",
]

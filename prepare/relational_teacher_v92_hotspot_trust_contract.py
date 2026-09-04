#!/usr/bin/env python3
"""Pure grid and gate contract for Teacher-v9.2 hotspot/trust response."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v92_hotspot_trust_response_v1"
V91_RESPONSE_SCHEMA = "relational_teacher_v91_active_support_response_grid_v1"
V91_FAILURE_SCHEMA = "relational_teacher_v91_corrected_one_scene_overfit_v1"
TRAIN_SCENE = "room_0101"
SEED = 20261009
STEPS_PER_CANDIDATE = 6
LEARNING_RATE = 1e-5
GRAD_CLIP = 1.0
CANDIDATES = (
    {
        "name": "rank2_list010_trust4",
        "active_support_weight": 0.50,
        "hotspot_margin_weight": 2.0,
        "hotspot_listwise_weight": 0.10,
        "absolute_negative_weight": 2.0,
        "background_trust_weight": 4.0,
        "prompt_weight": 8.0,
        "v5_preservation_weight": 8.0,
    },
    {
        "name": "rank4_list050_trust8",
        "active_support_weight": 0.25,
        "hotspot_margin_weight": 4.0,
        "hotspot_listwise_weight": 0.50,
        "absolute_negative_weight": 4.0,
        "background_trust_weight": 8.0,
        "prompt_weight": 12.0,
        "v5_preservation_weight": 12.0,
    },
    {
        "name": "rank8_list200_trust24",
        "active_support_weight": 0.10,
        "hotspot_margin_weight": 8.0,
        "hotspot_listwise_weight": 2.00,
        "absolute_negative_weight": 8.0,
        "background_trust_weight": 24.0,
        "prompt_weight": 24.0,
        "v5_preservation_weight": 24.0,
    },
)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _instance_means(values: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    prompts = values.get("per_prompt_metrics")
    if not isinstance(prompts, list) or len(prompts) != 2:
        raise ValueError("response must contain two prompt metrics")
    result: Dict[str, Dict[str, float]] = {}
    for name in ("bed_01", "chair_01", "chair_06"):
        rows = [prompt["instances"][name] for prompt in prompts]
        result[name] = {
            key: sum(_finite(row[key], name + " " + key) for row in rows) / 2.0
            for key in ("topk_overlap", "soft_recall", "active_support_mae")
        }
    return result


def response_checks(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
) -> Dict[str, bool]:
    base = _instance_means(before)
    candidate = _instance_means(after)
    checks: Dict[str, bool] = {}
    for name, label in (
        ("bed_01", "bed"),
        ("chair_01", "normal_chair"),
        ("chair_06", "high_chair"),
    ):
        checks[label + "_mean_topk_improves"] = (
            candidate[name]["topk_overlap"] > base[name]["topk_overlap"] + 1e-8
        )
        checks[label + "_recall_retained"] = (
            candidate[name]["soft_recall"]
            >= base[name]["soft_recall"] - 0.005
        )
        checks[label + "_active_mae_not_worse"] = (
            candidate[name]["active_support_mae"]
            <= base[name]["active_support_mae"] + 0.002
        )
    checks["hotspot_margin_decreases"] = _finite(
        after["hotspot_margin_macro"], "after margin"
    ) < _finite(before["hotspot_margin_macro"], "before margin")
    checks["hotspot_listwise_decreases"] = _finite(
        after["hotspot_listwise_macro"], "after listwise"
    ) < _finite(before["hotspot_listwise_macro"], "before listwise")
    checks["prompt_invariance_retained_2pct"] = _finite(
        after["prompt_invariance"], "after prompt"
    ) <= _finite(before["prompt_invariance"], "before prompt") * 1.02 + 1e-8
    checks["negative_mean_addition_bounded"] = _finite(
        after["negative_mean"], "after negative mean"
    ) <= _finite(before["negative_mean"], "before negative mean") + 0.002
    checks["negative_max_addition_bounded"] = _finite(
        after["negative_max"], "after negative max"
    ) <= _finite(before["negative_max"], "before negative max") + 0.01
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 panel must contain three cases")
    base_v5 = sum(_finite(value, "base v5") for value in base_v5_dense) / 3.0
    candidate_v5 = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) / 3.0
    checks["v5_replay_retained_1pct"] = candidate_v5 <= base_v5 * 1.01 + 1e-8
    return checks


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    expected = [str(row["name"]) for row in CANDIDATES]
    if [str(row.get("name")) for row in rows] != expected:
        raise ValueError("hotspot/trust candidate grid order changed")
    eligible = [row for row in rows if row.get("eligible") is True]
    def key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
        before = _instance_means(row["before"])
        after = _instance_means(row["after"])
        gains = [
            after[name]["topk_overlap"] - before[name]["topk_overlap"]
            for name in ("bed_01", "chair_01", "chair_06")
        ]
        return (
            -min(gains),
            -sum(gains),
            _finite(row["after"]["background_trust"], "background trust"),
            expected.index(str(row["name"])),
        )
    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


__all__ = [
    "CANDIDATES",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "SCHEMA",
    "SEED",
    "STEPS_PER_CANDIDATE",
    "TRAIN_SCENE",
    "V91_FAILURE_SCHEMA",
    "V91_RESPONSE_SCHEMA",
    "rank_candidates",
    "response_checks",
]

#!/usr/bin/env python3
"""Pure selection contract for Teacher-v9.4.1 metric-first replication."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v941_metric_replication_v1"
V94_SCHEMA = "relational_teacher_v94_common_descent_preflight_v1"
TRAIN_SCENE = "room_0101"
DIRECTION_SEED = 20261011
SEED = 20261012
SELECTED_RADIUS = 0.01
REPLICATION_PANELS = (
    {"name": "replica_0", "timestep": 50, "noise_seed": 20262013},
    {"name": "replica_1", "timestep": 275, "noise_seed": 20263022},
    {"name": "replica_2", "timestep": 425, "noise_seed": 20264031},
)
OBJECTS = ("bed_01", "chair_01", "chair_06")


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _metric(rows: Sequence[Mapping[str, object]], replica: int, prompt: int, name: str, key: str) -> float:
    return _finite(rows[replica]["per_prompt_metrics"][prompt]["instances"][name][key], key)


def replication_checks(
    *,
    base_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
) -> Dict[str, bool]:
    if len(base_rows) != 3 or len(candidate_rows) != 3:
        raise ValueError("metric replication requires three panels")
    checks: Dict[str, bool] = {}
    for name, label in zip(OBJECTS, ("bed", "normal_chair", "high_chair")):
        base_topk = [_metric(base_rows, r, p, name, "topk_overlap") for r in range(3) for p in range(2)]
        candidate_topk = [
            _metric(candidate_rows, r, p, name, "topk_overlap")
            for r in range(3)
            for p in range(2)
        ]
        checks[label + "_pooled_topk_improves"] = sum(candidate_topk) / 6.0 > sum(base_topk) / 6.0 + 1e-8
        checks[label + "_every_case_topk_not_worse"] = all(
            candidate >= base for base, candidate in zip(base_topk, candidate_topk)
        )
        checks[label + "_at_least_two_cases_improve"] = sum(
            candidate > base + 1e-8 for base, candidate in zip(base_topk, candidate_topk)
        ) >= 2
        for prompt, prompt_label in ((0, "watch"), (1, "write")):
            base_prompt = [_metric(base_rows, r, prompt, name, "topk_overlap") for r in range(3)]
            candidate_prompt = [
                _metric(candidate_rows, r, prompt, name, "topk_overlap")
                for r in range(3)
            ]
            checks[label + "_" + prompt_label + "_pooled_topk_not_worse"] = (
                sum(candidate_prompt) / 3.0 >= sum(base_prompt) / 3.0
            )
        base_recall = sum(
            _metric(base_rows, r, p, name, "soft_recall") for r in range(3) for p in range(2)
        ) / 6.0
        candidate_recall = sum(
            _metric(candidate_rows, r, p, name, "soft_recall") for r in range(3) for p in range(2)
        ) / 6.0
        checks[label + "_recall_retained"] = candidate_recall >= base_recall - 0.005
        base_mae = sum(
            _metric(base_rows, r, p, name, "active_support_mae") for r in range(3) for p in range(2)
        ) / 6.0
        candidate_mae = sum(
            _metric(candidate_rows, r, p, name, "active_support_mae") for r in range(3) for p in range(2)
        ) / 6.0
        checks[label + "_active_mae_retained"] = candidate_mae <= base_mae + 0.002
    checks["each_replica_prompt_invariance_retained_2pct"] = all(
        _finite(candidate["prompt_invariance"], "candidate prompt")
        <= _finite(base["prompt_invariance"], "base prompt") * 1.02 + 1e-8
        for base, candidate in zip(base_rows, candidate_rows)
    )
    checks["each_replica_background_trust_bounded"] = all(
        _finite(row["background_trust"], "background trust") <= 5e-6
        for row in candidate_rows
    )
    checks["each_replica_negative_mean_addition_bounded"] = all(
        _finite(candidate["negative_mean"], "candidate negative mean")
        <= _finite(base["negative_mean"], "base negative mean") + 0.002
        for base, candidate in zip(base_rows, candidate_rows)
    )
    checks["each_replica_negative_max_addition_bounded"] = all(
        _finite(candidate["negative_max"], "candidate negative max")
        <= _finite(base["negative_max"], "base negative max") + 0.01
        for base, candidate in zip(base_rows, candidate_rows)
    )
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 panel must contain three cases")
    base_v5 = sum(_finite(value, "base v5") for value in base_v5_dense) / 3.0
    candidate_v5 = sum(_finite(value, "candidate v5") for value in candidate_v5_dense) / 3.0
    checks["v5_replay_retained_1pct"] = candidate_v5 <= base_v5 * 1.01 + 1e-8
    return checks


__all__ = [
    "DIRECTION_SEED",
    "OBJECTS",
    "REPLICATION_PANELS",
    "SCHEMA",
    "SEED",
    "SELECTED_RADIUS",
    "TRAIN_SCENE",
    "V94_SCHEMA",
    "replication_checks",
]

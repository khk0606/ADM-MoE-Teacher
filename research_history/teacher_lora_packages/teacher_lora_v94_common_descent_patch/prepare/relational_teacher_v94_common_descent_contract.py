#!/usr/bin/env python3
"""Pure protocol contract for the Teacher-v9.4 common-descent audit."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v94_common_descent_preflight_v1"
V93_SCHEMA = "relational_teacher_v93_exact_topk_response_v1"
TRAIN_SCENE = "room_0101"
SEED = 20261011
PROBE_TIMESTEP = 125
TASK_ORDER = (
    "watch_bed",
    "watch_normal_chair",
    "watch_high_chair",
    "write_bed",
    "write_normal_chair",
    "write_high_chair",
)
STEP_RADII = (0.0003, 0.001, 0.003, 0.01)
FW_ITERATIONS = 512
MIN_DIRECTIONAL_DERIVATIVE = 1e-7


EXPECTED_V93_FAILURES = {
    "exact_swap_lr010": [
        "high_chair_each_prompt_topk_not_worse",
        "high_chair_mean_topk_improves",
        "high_chair_swap_loss_decreases",
        "normal_chair_mean_topk_improves",
        "normal_chair_swap_loss_decreases",
    ],
    "exact_swap_lr020": [
        "high_chair_each_prompt_topk_not_worse",
        "high_chair_mean_topk_improves",
        "high_chair_swap_loss_decreases",
        "normal_chair_swap_loss_decreases",
    ],
    "exact_swap_lr040": [
        "bed_active_mae_not_worse",
        "high_chair_each_prompt_topk_not_worse",
        "high_chair_mean_topk_improves",
        "high_chair_swap_loss_decreases",
        "normal_chair_swap_loss_decreases",
    ],
}


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def candidate_checks(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    common_direction_valid: bool,
) -> Dict[str, bool]:
    before_task = list(before["per_task_exact_topk_swap"])
    after_task = list(after["per_task_exact_topk_swap"])
    before_prompts = list(before["per_prompt_metrics"])
    after_prompts = list(after["per_prompt_metrics"])
    if len(before_task) != 2 or len(after_task) != 2:
        raise ValueError("common-descent task panel must contain two prompts")
    if any(len(row) != 3 for row in before_task + after_task):
        raise ValueError("common-descent task panel must contain three objects")
    checks: Dict[str, bool] = {
        "common_direction_exists": bool(common_direction_valid),
        "all_six_same_panel_swap_losses_decrease": all(
            _finite(after_task[prompt][slot], "after task")
            < _finite(before_task[prompt][slot], "before task") - 1e-10
            for prompt in range(2)
            for slot in range(3)
        ),
    }
    names = ("bed_01", "chair_01", "chair_06")
    checks["all_six_topk_overlaps_not_worse"] = all(
        _finite(
            after_prompts[prompt]["instances"][name]["topk_overlap"],
            "after topk",
        )
        >= _finite(
            before_prompts[prompt]["instances"][name]["topk_overlap"],
            "before topk",
        )
        for prompt in range(2)
        for name in names
    )
    checks["all_object_recall_retained"] = all(
        sum(
            _finite(after_prompts[p]["instances"][name]["soft_recall"], "after recall")
            for p in range(2)
        )
        / 2.0
        >= sum(
            _finite(before_prompts[p]["instances"][name]["soft_recall"], "before recall")
            for p in range(2)
        )
        / 2.0
        - 0.005
        for name in names
    )
    checks["all_object_active_mae_retained"] = all(
        sum(
            _finite(
                after_prompts[p]["instances"][name]["active_support_mae"],
                "after active mae",
            )
            for p in range(2)
        )
        / 2.0
        <= sum(
            _finite(
                before_prompts[p]["instances"][name]["active_support_mae"],
                "before active mae",
            )
            for p in range(2)
        )
        / 2.0
        + 0.002
        for name in names
    )
    checks["background_trust_bounded"] = _finite(
        after["background_trust"], "background trust"
    ) <= 5e-6
    checks["prompt_invariance_retained_2pct"] = _finite(
        after["prompt_invariance"], "after prompt invariance"
    ) <= _finite(before["prompt_invariance"], "before prompt invariance") * 1.02 + 1e-8
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
    if [float(row.get("step_radius")) for row in rows] != list(STEP_RADII):
        raise ValueError("common-descent radius grid/order changed")
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float]:
        before = list(row["before"]["per_task_exact_topk_swap"])
        after = list(row["after"]["per_task_exact_topk_swap"])
        relative = [
            (_finite(before[p][s], "before") - _finite(after[p][s], "after"))
            / max(_finite(before[p][s], "before"), 1e-12)
            for p in range(2)
            for s in range(3)
        ]
        return (-min(relative), float(row["step_radius"]))

    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


__all__ = [
    "EXPECTED_V93_FAILURES",
    "FW_ITERATIONS",
    "MIN_DIRECTIONAL_DERIVATIVE",
    "PROBE_TIMESTEP",
    "SCHEMA",
    "SEED",
    "STEP_RADII",
    "TASK_ORDER",
    "TRAIN_SCENE",
    "V93_SCHEMA",
    "candidate_checks",
    "rank_candidates",
]

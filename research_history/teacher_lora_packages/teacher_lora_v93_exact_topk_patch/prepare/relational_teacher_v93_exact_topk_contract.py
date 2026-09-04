#!/usr/bin/env python3
"""Pure protocol and selection contract for Teacher-v9.3 exact top-k."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v93_exact_topk_response_v1"
V92_SCHEMA = "relational_teacher_v92_hotspot_trust_response_v1"
TRAIN_SCENE = "room_0101"
SEED = 20261010
STEPS_PER_CANDIDATE = 6
GRAD_CLIP = 1.0
TRAINING_TIMESTEPS = (50, 125, 200, 275, 350, 425)
COMMON_WEIGHTS = {
    "instance_primary_weight": 0.25,
    "verified_union_weight": 0.10,
    "environment_weight": 0.05,
    "active_support_weight": 0.10,
    "exact_topk_swap_weight": 12.0,
    "absolute_negative_weight": 4.0,
    "background_trust_weight": 8.0,
    "prompt_weight": 12.0,
    "v5_preservation_weight": 12.0,
}
CANDIDATES = (
    {"name": "exact_swap_lr010", "learning_rate": 1e-5, **COMMON_WEIGHTS},
    {"name": "exact_swap_lr020", "learning_rate": 2e-5, **COMMON_WEIGHTS},
    {"name": "exact_swap_lr040", "learning_rate": 4e-5, **COMMON_WEIGHTS},
)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _prompts(panel: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = panel.get("per_prompt_metrics")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("exact top-k panel must contain two prompts")
    return value


def _instance_means(panel: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    prompts = _prompts(panel)
    result = {}
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
    before_prompts = _prompts(before)
    after_prompts = _prompts(after)
    checks: Dict[str, bool] = {}
    names = (("bed_01", "bed"), ("chair_01", "normal_chair"), ("chair_06", "high_chair"))
    for slot, (name, label) in enumerate(names):
        checks[label + "_mean_topk_improves"] = (
            candidate[name]["topk_overlap"] > base[name]["topk_overlap"] + 1e-8
        )
        checks[label + "_each_prompt_topk_not_worse"] = all(
            _finite(after_prompts[index]["instances"][name]["topk_overlap"], "after topk")
            >= _finite(before_prompts[index]["instances"][name]["topk_overlap"], "before topk")
            for index in range(2)
        )
        checks[label + "_recall_retained"] = (
            candidate[name]["soft_recall"] >= base[name]["soft_recall"] - 0.005
        )
        checks[label + "_active_mae_not_worse"] = (
            candidate[name]["active_support_mae"] <= base[name]["active_support_mae"] + 0.002
        )
        checks[label + "_swap_loss_decreases"] = _finite(
            after["per_instance_exact_topk_swap"][slot], "after swap"
        ) < _finite(before["per_instance_exact_topk_swap"][slot], "before swap")
    checks["background_trust_bounded"] = _finite(
        after["background_trust"], "background trust"
    ) <= 5e-6
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
    candidate_v5 = sum(_finite(value, "candidate v5") for value in candidate_v5_dense) / 3.0
    checks["v5_replay_retained_1pct"] = candidate_v5 <= base_v5 * 1.01 + 1e-8
    return checks


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    expected = [str(row["name"]) for row in CANDIDATES]
    if [str(row.get("name")) for row in rows] != expected:
        raise ValueError("exact top-k LR grid/order changed")
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
    "COMMON_WEIGHTS",
    "GRAD_CLIP",
    "SCHEMA",
    "SEED",
    "STEPS_PER_CANDIDATE",
    "TRAINING_TIMESTEPS",
    "TRAIN_SCENE",
    "V92_SCHEMA",
    "rank_candidates",
    "response_checks",
]

#!/usr/bin/env python3
"""Pure contract for Teacher-v9.5 multi-timestep gradient consensus."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v95_multitimestep_consensus_v1"
V941_SCHEMA = "relational_teacher_v941_metric_replication_v1"
TRAIN_SCENE = "room_0101"
INITIALIZATION_SEED = 20261011
SEED = 20261013
DESIGN_PANELS = (
    {"name": "design_t125", "timestep": 125, "noise_seed": 20262011},
    {"name": "replica_0", "timestep": 50, "noise_seed": 20262013},
    {"name": "replica_1", "timestep": 275, "noise_seed": 20263022},
    {"name": "replica_2", "timestep": 425, "noise_seed": 20264031},
)
AUDIT_PANELS = (
    {"name": "audit_t025", "timestep": 25, "noise_seed": 20265040},
    {"name": "audit_t175", "timestep": 175, "noise_seed": 20266049},
    {"name": "audit_t350", "timestep": 350, "noise_seed": 20267058},
    {"name": "audit_t475", "timestep": 475, "noise_seed": 20268067},
)
STEP_RADII = (0.001, 0.003, 0.01)
V5_TIMESTEPS = (75, 250, 475)
V5_NOISE_SEED = SEED + 5000
FW_ITERATIONS = 1024
MIN_DIRECTIONAL_DERIVATIVE = 1e-7
OBJECTS = ("bed_01", "chair_01", "chair_06")


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _metric(rows: Sequence[Mapping[str, object]], panel: int, prompt: int, name: str, key: str) -> float:
    return _finite(rows[panel]["per_prompt_metrics"][prompt]["instances"][name][key], key)


def consensus_candidate_checks(
    *,
    base_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    common_direction_valid: bool,
) -> Dict[str, bool]:
    if len(base_rows) != len(AUDIT_PANELS) or len(candidate_rows) != len(AUDIT_PANELS):
        raise ValueError("multi-timestep audit panel count changed")
    checks: Dict[str, bool] = {"common_direction_exists": bool(common_direction_valid)}
    case_count = len(AUDIT_PANELS) * 2
    for name, label in zip(OBJECTS, ("bed", "normal_chair", "high_chair")):
        base_topk = [
            _metric(base_rows, panel, prompt, name, "topk_overlap")
            for panel in range(len(AUDIT_PANELS))
            for prompt in range(2)
        ]
        candidate_topk = [
            _metric(candidate_rows, panel, prompt, name, "topk_overlap")
            for panel in range(len(AUDIT_PANELS))
            for prompt in range(2)
        ]
        checks[label + "_pooled_topk_improves"] = (
            sum(candidate_topk) / case_count > sum(base_topk) / case_count + 1e-8
        )
        checks[label + "_every_case_topk_not_worse"] = all(
            candidate >= base for base, candidate in zip(base_topk, candidate_topk)
        )
        checks[label + "_at_least_two_cases_improve"] = sum(
            candidate > base + 1e-8 for base, candidate in zip(base_topk, candidate_topk)
        ) >= 2
        for prompt, prompt_label in ((0, "watch"), (1, "write")):
            base_prompt = [
                _metric(base_rows, panel, prompt, name, "topk_overlap")
                for panel in range(len(AUDIT_PANELS))
            ]
            candidate_prompt = [
                _metric(candidate_rows, panel, prompt, name, "topk_overlap")
                for panel in range(len(AUDIT_PANELS))
            ]
            checks[label + "_" + prompt_label + "_pooled_topk_not_worse"] = (
                sum(candidate_prompt) >= sum(base_prompt)
            )
        base_recall = sum(
            _metric(base_rows, panel, prompt, name, "soft_recall")
            for panel in range(len(AUDIT_PANELS))
            for prompt in range(2)
        ) / case_count
        candidate_recall = sum(
            _metric(candidate_rows, panel, prompt, name, "soft_recall")
            for panel in range(len(AUDIT_PANELS))
            for prompt in range(2)
        ) / case_count
        checks[label + "_recall_retained"] = candidate_recall >= base_recall - 0.005
        base_mae = sum(
            _metric(base_rows, panel, prompt, name, "active_support_mae")
            for panel in range(len(AUDIT_PANELS))
            for prompt in range(2)
        ) / case_count
        candidate_mae = sum(
            _metric(candidate_rows, panel, prompt, name, "active_support_mae")
            for panel in range(len(AUDIT_PANELS))
            for prompt in range(2)
        ) / case_count
        checks[label + "_active_mae_retained"] = candidate_mae <= base_mae + 0.002
    checks["each_panel_prompt_invariance_retained_2pct"] = all(
        _finite(candidate["prompt_invariance"], "candidate prompt")
        <= _finite(base["prompt_invariance"], "base prompt") * 1.02 + 1e-8
        for base, candidate in zip(base_rows, candidate_rows)
    )
    checks["each_panel_background_trust_bounded"] = all(
        _finite(row["background_trust"], "background trust") <= 5e-6
        for row in candidate_rows
    )
    checks["each_panel_negative_mean_addition_bounded"] = all(
        _finite(candidate["negative_mean"], "candidate negative mean")
        <= _finite(base["negative_mean"], "base negative mean") + 0.002
        for base, candidate in zip(base_rows, candidate_rows)
    )
    checks["each_panel_negative_max_addition_bounded"] = all(
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


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if [float(row.get("step_radius")) for row in rows] != list(STEP_RADII):
        raise ValueError("multi-timestep radius grid/order changed")
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float, float]:
        base_rows = row["base_rows"]
        candidate_rows = row["candidate_rows"]
        gains = []
        for name in OBJECTS:
            base = sum(
                _metric(base_rows, panel, prompt, name, "topk_overlap")
                for panel in range(len(AUDIT_PANELS))
                for prompt in range(2)
            )
            candidate = sum(
                _metric(candidate_rows, panel, prompt, name, "topk_overlap")
                for panel in range(len(AUDIT_PANELS))
                for prompt in range(2)
            )
            gains.append(candidate - base)
        return (-min(gains), -sum(gains), float(row["step_radius"]))

    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


__all__ = [
    "AUDIT_PANELS",
    "DESIGN_PANELS",
    "FW_ITERATIONS",
    "INITIALIZATION_SEED",
    "MIN_DIRECTIONAL_DERIVATIVE",
    "OBJECTS",
    "SCHEMA",
    "SEED",
    "STEP_RADII",
    "TRAIN_SCENE",
    "V941_SCHEMA",
    "V5_NOISE_SEED",
    "V5_TIMESTEPS",
    "consensus_candidate_checks",
    "rank_candidates",
]

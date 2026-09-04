#!/usr/bin/env python3
"""Pure contract for Teacher-v9.6 object-balanced consensus directions."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

from relational_teacher_v94_common_descent import frank_wolfe_min_norm_weights


SCHEMA = "relational_teacher_v96_object_balanced_consensus_v1"
V95_SCHEMA = "relational_teacher_v95_multitimestep_consensus_v1"
TRAIN_SCENE = "room_0101"
INITIALIZATION_SEED = 20261011
SEED = 20261014
OBJECTS = ("bed_01", "chair_01", "chair_06")
DESIGN_PANELS = (
    {"name": "design_t025", "timestep": 25, "noise_seed": 20265040},
    {"name": "design_t050", "timestep": 50, "noise_seed": 20262013},
    {"name": "design_t125", "timestep": 125, "noise_seed": 20262011},
    {"name": "design_t175", "timestep": 175, "noise_seed": 20266049},
    {"name": "design_t275", "timestep": 275, "noise_seed": 20263022},
    {"name": "design_t350", "timestep": 350, "noise_seed": 20267058},
    {"name": "design_t425", "timestep": 425, "noise_seed": 20264031},
    {"name": "design_t475", "timestep": 475, "noise_seed": 20268067},
)
AUDIT_PANELS = (
    {"name": "audit_t010", "timestep": 10, "noise_seed": 20273012},
    {"name": "audit_t100", "timestep": 100, "noise_seed": 20274021},
    {"name": "audit_t225", "timestep": 225, "noise_seed": 20275030},
    {"name": "audit_t490", "timestep": 490, "noise_seed": 20276039},
)
DIRECTION_NAMES = (
    "task_common",
    "object_balanced",
    "chair_pair_priority",
    "high_chair_priority",
)
STEP_RADII = (0.0005, 0.001, 0.0015, 0.002)
FW_ITERATIONS = 1536
MIN_DIRECTIONAL_DERIVATIVE = 1e-7
SAFE_BLEND_FRACTION = 0.9
V5_TIMESTEPS = (40, 220, 490)
V5_NOISE_SEED = SEED + 5000


EXPECTED_V95_FAILURES = {
    "multitimestep_radius_0p001": [
        "bed_every_case_topk_not_worse",
        "bed_watch_pooled_topk_not_worse",
        "high_chair_at_least_two_cases_improve",
        "high_chair_pooled_topk_improves",
        "normal_chair_every_case_topk_not_worse",
        "normal_chair_watch_pooled_topk_not_worse",
    ],
    "multitimestep_radius_0p003": [
        "high_chair_at_least_two_cases_improve",
        "high_chair_every_case_topk_not_worse",
        "high_chair_pooled_topk_improves",
        "high_chair_write_pooled_topk_not_worse",
        "normal_chair_every_case_topk_not_worse",
        "normal_chair_watch_pooled_topk_not_worse",
    ],
    "multitimestep_radius_0p01": [
        "each_panel_background_trust_bounded",
        "high_chair_at_least_two_cases_improve",
        "high_chair_every_case_topk_not_worse",
        "high_chair_pooled_topk_improves",
        "high_chair_watch_pooled_topk_not_worse",
        "high_chair_write_pooled_topk_not_worse",
        "normal_chair_every_case_topk_not_worse",
    ],
}


def _unit_coefficients(coefficients: np.ndarray, gram: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(coefficients, dtype=np.float64)
    norm_squared = float(values @ gram @ values)
    if not math.isfinite(norm_squared) or norm_squared <= 1e-20:
        raise ValueError("direction coefficient vector has zero/non-finite norm")
    norm = math.sqrt(norm_squared)
    return values / norm, norm


def _group_unit(gram: np.ndarray, labels: Sequence[str], object_name: str) -> np.ndarray:
    indices = [index for index, label in enumerate(labels) if label.endswith("|" + object_name)]
    if len(indices) != len(DESIGN_PANELS) * 2:
        raise ValueError("object gradient group inventory changed: " + object_name)
    coefficients = np.zeros(len(labels), dtype=np.float64)
    coefficients[indices] = 1.0 / len(indices)
    return _unit_coefficients(coefficients, gram)[0]


def _safe_blend(
    gram: np.ndarray,
    common: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    common_dot = gram @ common
    target_dot = gram @ target
    cap = 1.0
    for before, after in zip(common_dot, target_dot):
        if after < MIN_DIRECTIONAL_DERIVATIVE and after < before:
            cap = min(
                cap,
                float((before - MIN_DIRECTIONAL_DERIVATIVE) / (before - after)),
            )
    alpha = max(0.0, min(1.0, cap)) * SAFE_BLEND_FRACTION
    blended = (1.0 - alpha) * common + alpha * target
    return _unit_coefficients(blended, gram)[0], alpha


def build_direction_specs(
    gram: np.ndarray,
    labels: Sequence[str],
) -> list[Dict[str, object]]:
    matrix = np.asarray(gram, dtype=np.float64)
    expected = len(DESIGN_PANELS) * 2 * len(OBJECTS)
    if matrix.shape != (expected, expected) or len(labels) != expected:
        raise ValueError("object-balanced gradient inventory changed")
    if len(set(labels)) != expected or not np.isfinite(matrix).all():
        raise ValueError("object-balanced gradient labels/Gram are invalid")
    if not np.allclose(matrix, matrix.T, atol=2e-6):
        raise ValueError("object-balanced Gram is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=2e-5):
        raise ValueError("object-balanced gradients are not normalized")

    common_weights = frank_wolfe_min_norm_weights(matrix, FW_ITERATIONS)
    common, common_norm = _unit_coefficients(common_weights, matrix)
    bed = _group_unit(matrix, labels, "bed_01")
    chair = _group_unit(matrix, labels, "chair_01")
    high = _group_unit(matrix, labels, "chair_06")
    object_target = _unit_coefficients(bed + chair + high, matrix)[0]
    chair_target = _unit_coefficients(chair + high, matrix)[0]
    high_target = high
    direction_rows = [("task_common", common, 0.0)]
    for name, target in (
        ("object_balanced", object_target),
        ("chair_pair_priority", chair_target),
        ("high_chair_priority", high_target),
    ):
        coefficients, alpha = _safe_blend(matrix, common, target)
        direction_rows.append((name, coefficients, alpha))

    result = []
    for name, coefficients, alpha in direction_rows:
        derivatives = matrix @ coefficients
        valid = bool(
            np.isfinite(derivatives).all()
            and float(derivatives.min()) >= MIN_DIRECTIONAL_DERIVATIVE
        )
        result.append(
            {
                "name": name,
                "blend_alpha": float(alpha),
                "task_coefficients": coefficients.astype(float).tolist(),
                "task_directional_derivatives": derivatives.astype(float).tolist(),
                "minimum_task_directional_derivative": float(derivatives.min()),
                "valid": valid,
                "common_minimum_norm": float(common_norm),
            }
        )
    if tuple(row["name"] for row in result) != DIRECTION_NAMES:
        raise AssertionError("direction order changed")
    return result


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(label + " must be finite")
    return number


def _metric(rows: Sequence[Mapping[str, object]], panel: int, prompt: int, name: str, key: str) -> float:
    return _finite(rows[panel]["per_prompt_metrics"][prompt]["instances"][name][key], key)


def candidate_checks(
    *,
    base_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    direction_valid: bool,
) -> Dict[str, bool]:
    if len(base_rows) != len(AUDIT_PANELS) or len(candidate_rows) != len(AUDIT_PANELS):
        raise ValueError("object-balanced audit panel count changed")
    checks: Dict[str, bool] = {"design_direction_valid": bool(direction_valid)}
    case_count = len(AUDIT_PANELS) * 2
    for name, label in zip(OBJECTS, ("bed", "normal_chair", "high_chair")):
        base_topk = [
            _metric(base_rows, panel, prompt, name, "topk_overlap")
            for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
        ]
        candidate_topk = [
            _metric(candidate_rows, panel, prompt, name, "topk_overlap")
            for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
        ]
        checks[label + "_pooled_topk_improves"] = sum(candidate_topk) > sum(base_topk) + 1e-8
        checks[label + "_every_case_topk_not_worse"] = all(
            candidate >= base for base, candidate in zip(base_topk, candidate_topk)
        )
        checks[label + "_at_least_two_cases_improve"] = sum(
            candidate > base + 1e-8 for base, candidate in zip(base_topk, candidate_topk)
        ) >= 2
        for prompt, prompt_label in ((0, "watch"), (1, "write")):
            base_prompt = [_metric(base_rows, panel, prompt, name, "topk_overlap") for panel in range(len(AUDIT_PANELS))]
            candidate_prompt = [_metric(candidate_rows, panel, prompt, name, "topk_overlap") for panel in range(len(AUDIT_PANELS))]
            checks[label + "_" + prompt_label + "_pooled_topk_not_worse"] = sum(candidate_prompt) >= sum(base_prompt)
        base_recall = sum(
            _metric(base_rows, panel, prompt, name, "soft_recall")
            for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
        ) / case_count
        candidate_recall = sum(
            _metric(candidate_rows, panel, prompt, name, "soft_recall")
            for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
        ) / case_count
        checks[label + "_recall_retained"] = candidate_recall >= base_recall - 0.005
        base_mae = sum(
            _metric(base_rows, panel, prompt, name, "active_support_mae")
            for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
        ) / case_count
        candidate_mae = sum(
            _metric(candidate_rows, panel, prompt, name, "active_support_mae")
            for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
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
    expected = [(direction, radius) for direction in DIRECTION_NAMES for radius in STEP_RADII]
    observed = [(str(row.get("direction_name")), float(row.get("step_radius"))) for row in rows]
    if observed != expected:
        raise ValueError("object-balanced candidate grid/order changed")
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
        gains = []
        for name in OBJECTS:
            base = sum(
                _metric(row["base_rows"], panel, prompt, name, "topk_overlap")
                for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
            )
            candidate = sum(
                _metric(row["candidate_rows"], panel, prompt, name, "topk_overlap")
                for panel in range(len(AUDIT_PANELS)) for prompt in range(2)
            )
            gains.append(candidate - base)
        return (-min(gains), -sum(gains), float(row["step_radius"]), DIRECTION_NAMES.index(str(row["direction_name"])))

    eligible.sort(key=key)
    return [str(row["name"]) for row in eligible]


__all__ = [
    "AUDIT_PANELS", "DESIGN_PANELS", "DIRECTION_NAMES", "EXPECTED_V95_FAILURES",
    "FW_ITERATIONS", "INITIALIZATION_SEED", "MIN_DIRECTIONAL_DERIVATIVE",
    "OBJECTS", "SAFE_BLEND_FRACTION", "SCHEMA", "SEED", "STEP_RADII",
    "TRAIN_SCENE", "V5_NOISE_SEED", "V5_TIMESTEPS", "V95_SCHEMA",
    "build_direction_specs", "candidate_checks", "rank_candidates",
]

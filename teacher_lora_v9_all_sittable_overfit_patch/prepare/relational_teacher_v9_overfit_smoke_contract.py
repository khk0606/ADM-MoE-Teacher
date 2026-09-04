#!/usr/bin/env python3
"""Pure contracts for the Teacher-v9 one-scene overfit smoke."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v9_all_sittable_one_scene_overfit_v1"
PREFLIGHT_SCHEMA = "relational_teacher_v9_all_sittable_lora_cuda_preflight_v1"
POLICY_SCHEMA = "relational_teacher_v9_all_sittable_metric_policy_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
STEPS = 120
MONITOR_STEPS = (0, 10, 20, 40, 60, 80, 100, 120)
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
SEED = 20261006


def sanitize_metric_nonfinite(value: object, path: str = "metrics") -> object:
    """Map only the expected absent-hotspot +inf to a finite fail sentinel."""

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_metric_nonfinite(nested, path + "." + str(key))
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [sanitize_metric_nonfinite(nested, path) for nested in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            if math.isinf(number) and number > 0.0 and "centroid" in path:
                return 1e30
            raise ValueError("invalid non-finite metric: " + path)
    return value


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def evaluate_smoke_panel(
    *,
    candidate_rows: Mapping[str, Mapping[str, object]],
    base_rows: Mapping[str, Mapping[str, object]],
    absolute_presence: Mapping[str, Mapping[str, bool]],
    base_prompt_invariance: float,
    candidate_prompt_invariance: float,
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    relative_rules: Mapping[str, object],
) -> Dict[str, bool]:
    """Apply the already locked policy without choosing new cutoffs."""

    expected_rows = {
        "room_0101|sit_watch_v1",
        "room_0101|sit_write_v1",
    }
    if set(candidate_rows) != expected_rows or set(base_rows) != expected_rows:
        raise ValueError("one-scene smoke panel row IDs changed")
    if set(absolute_presence) != expected_rows:
        raise ValueError("presence-check row IDs changed")
    expected_presence_checks = {
        "every_verified_instance_has_soft_recall",
        "every_verified_instance_has_topk_overlap",
        "every_verified_instance_has_bounded_active_support_mae",
        "every_verified_instance_has_bounded_hotspot_centroid",
        "explicit_negative_mean_bounded",
        "explicit_negative_max_bounded",
    }
    epsilon = _finite(relative_rules["tie_epsilon"], "tie epsilon")
    if epsilon < 0.0:
        raise ValueError("tie epsilon cannot be negative")
    checks: Dict[str, bool] = {}
    for row_id in sorted(expected_rows):
        candidate = candidate_rows[row_id]
        base = base_rows[row_id]
        candidate_instances = candidate.get("instances")
        base_instances = base.get("instances")
        if not isinstance(candidate_instances, Mapping) or not isinstance(
            base_instances, Mapping
        ):
            raise ValueError(row_id + ": per-instance metrics missing")
        expected_instances = {"bed_01", "chair_01", "chair_06"}
        if set(candidate_instances) != expected_instances or set(base_instances) != expected_instances:
            raise ValueError(row_id + ": verified instance inventory changed")
        if set(absolute_presence[row_id]) != expected_presence_checks:
            raise ValueError(row_id + ": absolute-presence inventory changed")
        checks[row_id + "|absolute_presence"] = all(
            bool(value) for value in absolute_presence[row_id].values()
        )
        for instance, role in (
            ("bed_01", "bed"),
            ("chair_01", "normal_chair"),
            ("chair_06", "high_chair"),
        ):
            cand = candidate_instances[instance]
            frozen = base_instances[instance]
            cand_mae = _finite(cand["active_support_mae"], "candidate MAE")
            base_mae = _finite(frozen["active_support_mae"], "base MAE")
            if role == "bed":
                cap = _finite(
                    relative_rules["bed_active_support_mae_relative_cap"],
                    "Bed MAE cap",
                )
                checks[row_id + "|bed_retained"] = cand_mae <= base_mae * cap + epsilon
            else:
                if relative_rules[
                    "chair_active_support_mae_requires_strict_improvement"
                ] is not True:
                    raise ValueError("Chair strict-improvement rule changed")
                checks[row_id + "|" + role + "_mae_improves"] = cand_mae < base_mae - epsilon
            recall_tolerance = _finite(
                relative_rules["soft_recall_absolute_tolerance"],
                "recall tolerance",
            )
            overlap_tolerance = _finite(
                relative_rules["topk_overlap_absolute_tolerance"],
                "overlap tolerance",
            )
            centroid_tolerance = _finite(
                relative_rules["hotspot_centroid_distance_xy_tolerance"],
                "centroid tolerance",
            )
            checks[row_id + "|" + role + "_recall_retained"] = _finite(
                cand["soft_recall"], "candidate recall"
            ) >= _finite(frozen["soft_recall"], "base recall") - recall_tolerance
            checks[row_id + "|" + role + "_topk_retained"] = _finite(
                cand["topk_overlap"], "candidate overlap"
            ) >= _finite(frozen["topk_overlap"], "base overlap") - overlap_tolerance
            checks[row_id + "|" + role + "_centroid_retained"] = _finite(
                cand["hotspot_centroid_distance_xy"], "candidate centroid"
            ) <= _finite(
                frozen["hotspot_centroid_distance_xy"], "base centroid"
            ) + centroid_tolerance
        checks[row_id + "|negative_mean_bounded"] = _finite(
            candidate["explicit_negative_mean"], "candidate negative mean"
        ) <= _finite(base["explicit_negative_mean"], "base negative mean") + _finite(
            relative_rules["explicit_negative_mean_addition_cap"],
            "negative mean cap",
        )
        checks[row_id + "|negative_max_bounded"] = _finite(
            candidate["explicit_negative_max"], "candidate negative max"
        ) <= _finite(base["explicit_negative_max"], "base negative max") + _finite(
            relative_rules["explicit_negative_max_addition_cap"],
            "negative max cap",
        )
    checks["prompt_invariance_retained"] = _finite(
        candidate_prompt_invariance, "candidate prompt invariance"
    ) <= _finite(base_prompt_invariance, "base prompt invariance") * _finite(
        relative_rules["prompt_invariance_relative_cap"],
        "prompt invariance cap",
    ) + epsilon
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 smoke panel must contain chair/bed/whiteboard")
    base_v5 = sum(_finite(value, "base v5 dense") for value in base_v5_dense) / 3.0
    candidate_v5 = sum(
        _finite(value, "candidate v5 dense") for value in candidate_v5_dense
    ) / 3.0
    checks["v5_replay_dense_retained"] = candidate_v5 <= base_v5 * _finite(
        relative_rules["v5_replay_dense_relative_cap"], "v5 cap"
    ) + epsilon
    return checks


__all__ = [
    "DEVELOPMENT_SCENE",
    "GRAD_CLIP",
    "HELDOUT_TRAIN_SCENE",
    "LEARNING_RATE",
    "MONITOR_STEPS",
    "POLICY_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "SCHEMA",
    "SEED",
    "STEPS",
    "TRAIN_SCENE",
    "evaluate_smoke_panel",
    "sanitize_metric_nonfinite",
]

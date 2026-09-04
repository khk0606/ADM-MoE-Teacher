#!/usr/bin/env python3
"""Pure policy and selection contract for Teacher-v9.7 early rollout K=3."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Mapping, Sequence


SCHEMA = "relational_teacher_v97_early_rollout_k3_v1"
POLICY_SCHEMA = "relational_teacher_v97_rollout_metric_policy_v1"
FAILED_V91_SCHEMA = "relational_teacher_v91_corrected_one_scene_overfit_v1"
PREFLIGHT_SCHEMA = "relational_teacher_v9_all_sittable_lora_cuda_preflight_v1"
LOSS_RESPONSE_SCHEMA = "relational_teacher_v91_active_support_response_grid_v1"
TRAIN_SCENE = "room_0101"
HELDOUT_TRAIN_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
TRAINING_SEED = 20261008
ROLLOUT_SEED = 20261015
TRAINING_STEPS = 12
SNAPSHOT_STEPS = (3, 6, 12)
GENERATION_COUNT = 3
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
OBJECTS = ("bed_01", "chair_01", "chair_06")
OBJECT_ROLES = {
    "bed_01": "bed",
    "chair_01": "normal_chair",
    "chair_06": "high_chair",
}
LEARNING_RATE = 4e-5
GRAD_CLIP = 1.0
LORA_RANK = 4
LORA_ALPHA = 8.0
SELECTED_LOSS_CANDIDATE = "support2_rank025_preserve2"


# Locked before any v9.7 model execution. Exact Top-k remains in every metric
# row for diagnosis but is deliberately not a decision threshold: on these
# objects one membership swap changes overlap by 1/138 or 1/272.
ROLLOUT_POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "final_500_step_reverse_diffusion_physical_affordance_map",
    "train_scene": TRAIN_SCENE,
    "snapshot_steps": list(SNAPSHOT_STEPS),
    "generation_count": GENERATION_COUNT,
    "prompt_ids": list(PROMPT_IDS),
    "objects": list(OBJECTS),
    "topk_role": "diagnostic_only_not_a_gate_due_discrete_one_point_quantization",
    "absolute_continuous_presence": {
        "minimum_soft_recall": 0.75,
        "maximum_active_support_mae": 0.10,
        "maximum_hotspot_centroid_distance_xy": 0.60,
        "maximum_negative_mean": 0.10,
        "maximum_negative_max": 0.80,
    },
    "relative_retention": {
        "soft_recall_absolute_tolerance": 0.005,
        "active_support_mae_absolute_tolerance": 0.002,
        "explicit_negative_mean_addition_cap": 0.002,
        "explicit_negative_max_addition_cap": 0.01,
        "prompt_invariance_relative_cap": 1.02,
        "v5_replay_dense_relative_cap": 1.01,
        "tie_epsilon": 1e-8,
    },
    "selection_requirements": {
        "minimum_passing_generations_per_prompt": 2,
        "pooled_each_object_meets_absolute_continuous_presence": True,
        "pooled_each_object_retains_base_continuous_metrics": True,
        "pooled_macro_active_support_mae_strictly_improves": True,
        "pooled_worst_object_soft_recall_strictly_improves": True,
        "every_generation_prompt_invariance_retained": True,
        "every_generation_explicit_negative_retained": True,
        "v5_fixed_probe_retained": True,
    },
    "selection_order": (
        "maximize_minimum_pooled_object_soft_recall_then_"
        "minimize_maximum_pooled_object_active_support_mae_then_"
        "minimize_macro_active_support_mae_then_earliest_step"
    ),
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ROLLOUT_POLICY_ID = canonical_sha256(ROLLOUT_POLICY)


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " must be finite")
    return result


def continuous_presence_checks(metrics: Mapping[str, object]) -> Dict[str, bool]:
    instances = metrics.get("instances")
    if not isinstance(instances, Mapping) or set(instances) != set(OBJECTS):
        raise ValueError("continuous presence instance inventory changed")
    limits = ROLLOUT_POLICY["absolute_continuous_presence"]
    checks: Dict[str, bool] = {}
    for name in OBJECTS:
        role = OBJECT_ROLES[name]
        row = instances[name]
        checks[role + "_soft_recall_at_least_075"] = _finite(
            row["soft_recall"], role + " recall"
        ) >= float(limits["minimum_soft_recall"])
        checks[role + "_active_support_mae_at_most_010"] = _finite(
            row["active_support_mae"], role + " MAE"
        ) <= float(limits["maximum_active_support_mae"])
        checks[role + "_hotspot_centroid_at_most_060m"] = _finite(
            row["hotspot_centroid_distance_xy"], role + " centroid"
        ) <= float(limits["maximum_hotspot_centroid_distance_xy"])
        # Ensure exact Top-k remains present as a disclosed diagnostic even
        # though no Top-k cutoff participates in the decision.
        topk = _finite(row["topk_overlap"], role + " top-k")
        if not 0.0 <= topk <= 1.0:
            raise ValueError(role + " top-k diagnostic is outside [0,1]")
    checks["explicit_negative_mean_at_most_010"] = _finite(
        metrics["explicit_negative_mean"], "negative mean"
    ) <= float(limits["maximum_negative_mean"])
    checks["explicit_negative_max_at_most_080"] = _finite(
        metrics["explicit_negative_max"], "negative max"
    ) <= float(limits["maximum_negative_max"])
    return checks


def _metric(
    rows: Sequence[Sequence[Mapping[str, object]]],
    generation: int,
    prompt: int,
    name: str,
    key: str,
) -> float:
    return _finite(rows[generation][prompt]["instances"][name][key], key)


def rollout_step_checks(
    *,
    base_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Sequence[Mapping[str, object]]],
    candidate_presence: Sequence[Sequence[Mapping[str, bool]]],
    base_prompt_invariance: Sequence[float],
    candidate_prompt_invariance: Sequence[float],
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
) -> Dict[str, bool]:
    if len(base_rows) != GENERATION_COUNT or len(candidate_rows) != GENERATION_COUNT:
        raise ValueError("rollout rows must contain exactly K=3 generations")
    if len(candidate_presence) != GENERATION_COUNT:
        raise ValueError("rollout presence inventory changed")
    for generation in range(GENERATION_COUNT):
        if len(base_rows[generation]) != 2 or len(candidate_rows[generation]) != 2:
            raise ValueError("rollout prompt inventory changed")
        if len(candidate_presence[generation]) != 2:
            raise ValueError("rollout presence prompt inventory changed")
    if len(base_prompt_invariance) != GENERATION_COUNT or len(
        candidate_prompt_invariance
    ) != GENERATION_COUNT:
        raise ValueError("prompt-invariance generation inventory changed")
    relative = ROLLOUT_POLICY["relative_retention"]
    absolute = ROLLOUT_POLICY["absolute_continuous_presence"]
    epsilon = float(relative["tie_epsilon"])
    checks: Dict[str, bool] = {}

    required_presence_keys = set(continuous_presence_checks(candidate_rows[0][0]))
    object_presence_keys = {
        key
        for key in required_presence_keys
        if not key.startswith("explicit_negative_")
    }
    if len(object_presence_keys) != len(OBJECTS) * 3:
        raise ValueError("three-object continuous presence inventory changed")
    minimum_passes = int(
        ROLLOUT_POLICY["selection_requirements"][
            "minimum_passing_generations_per_prompt"
        ]
    )
    for prompt, prompt_name in enumerate(("watch", "write")):
        passing = 0
        for generation in range(GENERATION_COUNT):
            row = candidate_presence[generation][prompt]
            if set(row) != required_presence_keys:
                raise ValueError("continuous presence check inventory changed")
            # "All three objects are present" is deliberately an object-only
            # decision. Environment leakage is retained as an independent
            # safety gate below, so a tiny background value cannot disguise
            # whether Bed, normal Chair and High Chair were learned.
            passing += int(all(bool(row[key]) for key in object_presence_keys))
        checks[prompt_name + "_at_least_two_generations_have_all_three"] = (
            passing >= minimum_passes
        )

    base_pooled: Dict[str, Dict[str, float]] = {}
    candidate_pooled: Dict[str, Dict[str, float]] = {}
    for name in OBJECTS:
        base_pooled[name] = {}
        candidate_pooled[name] = {}
        for key in (
            "soft_recall",
            "active_support_mae",
            "hotspot_centroid_distance_xy",
            "topk_overlap",
        ):
            base_pooled[name][key] = sum(
                _metric(base_rows, generation, prompt, name, key)
                for generation in range(GENERATION_COUNT)
                for prompt in range(2)
            ) / float(GENERATION_COUNT * 2)
            candidate_pooled[name][key] = sum(
                _metric(candidate_rows, generation, prompt, name, key)
                for generation in range(GENERATION_COUNT)
                for prompt in range(2)
            ) / float(GENERATION_COUNT * 2)
        role = OBJECT_ROLES[name]
        checks[role + "_pooled_soft_recall_at_least_075"] = (
            candidate_pooled[name]["soft_recall"]
            >= float(absolute["minimum_soft_recall"])
        )
        checks[role + "_pooled_active_support_mae_at_most_010"] = (
            candidate_pooled[name]["active_support_mae"]
            <= float(absolute["maximum_active_support_mae"])
        )
        checks[role + "_pooled_hotspot_centroid_at_most_060m"] = (
            candidate_pooled[name]["hotspot_centroid_distance_xy"]
            <= float(absolute["maximum_hotspot_centroid_distance_xy"])
        )
        checks[role + "_pooled_soft_recall_retained"] = (
            candidate_pooled[name]["soft_recall"]
            >= base_pooled[name]["soft_recall"]
            - float(relative["soft_recall_absolute_tolerance"])
        )
        checks[role + "_pooled_active_support_mae_retained"] = (
            candidate_pooled[name]["active_support_mae"]
            <= base_pooled[name]["active_support_mae"]
            + float(relative["active_support_mae_absolute_tolerance"])
        )

    base_macro_mae = sum(
        base_pooled[name]["active_support_mae"] for name in OBJECTS
    ) / 3.0
    candidate_macro_mae = sum(
        candidate_pooled[name]["active_support_mae"] for name in OBJECTS
    ) / 3.0
    checks["pooled_macro_active_support_mae_strictly_improves"] = (
        candidate_macro_mae < base_macro_mae - epsilon
    )
    checks["pooled_worst_object_soft_recall_strictly_improves"] = min(
        candidate_pooled[name]["soft_recall"] for name in OBJECTS
    ) > min(base_pooled[name]["soft_recall"] for name in OBJECTS) + epsilon

    checks["every_generation_prompt_invariance_retained_2pct"] = all(
        _finite(candidate_prompt_invariance[generation], "candidate invariance")
        <= _finite(base_prompt_invariance[generation], "base invariance")
        * float(relative["prompt_invariance_relative_cap"])
        + epsilon
        for generation in range(GENERATION_COUNT)
    )
    checks["every_generation_negative_mean_retained"] = all(
        max(
            _finite(candidate_rows[generation][prompt]["explicit_negative_mean"], "candidate negative mean")
            for prompt in range(2)
        )
        <= max(
            _finite(base_rows[generation][prompt]["explicit_negative_mean"], "base negative mean")
            for prompt in range(2)
        )
        + float(relative["explicit_negative_mean_addition_cap"])
        for generation in range(GENERATION_COUNT)
    )
    checks["every_generation_negative_max_retained"] = all(
        max(
            _finite(candidate_rows[generation][prompt]["explicit_negative_max"], "candidate negative max")
            for prompt in range(2)
        )
        <= max(
            _finite(base_rows[generation][prompt]["explicit_negative_max"], "base negative max")
            for prompt in range(2)
        )
        + float(relative["explicit_negative_max_addition_cap"])
        for generation in range(GENERATION_COUNT)
    )
    if len(base_v5_dense) != 3 or len(candidate_v5_dense) != 3:
        raise ValueError("v5 probe must contain chair/bed/whiteboard")
    base_v5 = sum(_finite(value, "base v5") for value in base_v5_dense) / 3.0
    candidate_v5 = sum(
        _finite(value, "candidate v5") for value in candidate_v5_dense
    ) / 3.0
    checks["v5_fixed_probe_retained_1pct"] = candidate_v5 <= (
        base_v5 * float(relative["v5_replay_dense_relative_cap"]) + epsilon
    )
    return checks


def pooled_object_metrics(
    rows: Sequence[Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[str, float]]:
    if len(rows) != GENERATION_COUNT or any(len(row) != 2 for row in rows):
        raise ValueError("pooled rollout row inventory changed")
    result: Dict[str, Dict[str, float]] = {}
    for name in OBJECTS:
        result[name] = {
            key: sum(
                _metric(rows, generation, prompt, name, key)
                for generation in range(GENERATION_COUNT)
                for prompt in range(2)
            )
            / float(GENERATION_COUNT * 2)
            for key in (
                "soft_recall",
                "active_support_mae",
                "hotspot_centroid_distance_xy",
                "topk_overlap",
            )
        }
    return result


def rank_eligible_steps(rows: Sequence[Mapping[str, object]]) -> list[int]:
    if [int(row.get("step", -1)) for row in rows] != list(SNAPSHOT_STEPS):
        raise ValueError("rollout step order changed")
    eligible = [row for row in rows if row.get("eligible") is True]

    def key(row: Mapping[str, object]) -> tuple[float, float, float, int]:
        pooled = row.get("candidate_pooled")
        if not isinstance(pooled, Mapping) or set(pooled) != set(OBJECTS):
            raise ValueError("eligible rollout pooled metrics changed")
        minimum_recall = min(_finite(pooled[name]["soft_recall"], "recall") for name in OBJECTS)
        maximum_mae = max(_finite(pooled[name]["active_support_mae"], "MAE") for name in OBJECTS)
        macro_mae = sum(_finite(pooled[name]["active_support_mae"], "MAE") for name in OBJECTS) / 3.0
        return (-minimum_recall, maximum_mae, macro_mae, int(row["step"]))

    eligible.sort(key=key)
    return [int(row["step"]) for row in eligible]


def stable_rollout_seeds(generation: int) -> tuple[int, int]:
    generation = int(generation)
    if not 0 <= generation < GENERATION_COUNT:
        raise ValueError("rollout generation is outside K=3")
    digest = hashlib.sha256(
        f"{ROLLOUT_SEED}|v97_early_rollout|generation_{generation}".encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


__all__ = [
    "DEVELOPMENT_SCENE",
    "FAILED_V91_SCHEMA",
    "GENERATION_COUNT",
    "GRAD_CLIP",
    "HELDOUT_TRAIN_SCENE",
    "LEARNING_RATE",
    "LORA_ALPHA",
    "LORA_RANK",
    "LOSS_RESPONSE_SCHEMA",
    "OBJECTS",
    "OBJECT_ROLES",
    "POLICY_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "PROMPT_IDS",
    "ROLLOUT_POLICY",
    "ROLLOUT_POLICY_ID",
    "ROLLOUT_SEED",
    "SCHEMA",
    "SELECTED_LOSS_CANDIDATE",
    "SNAPSHOT_STEPS",
    "TRAINING_SEED",
    "TRAINING_STEPS",
    "TRAIN_SCENE",
    "canonical_sha256",
    "continuous_presence_checks",
    "pooled_object_metrics",
    "rank_eligible_steps",
    "rollout_step_checks",
    "stable_rollout_seeds",
]

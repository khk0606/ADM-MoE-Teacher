#!/usr/bin/env python3
"""Pure contract for Teacher-v9.8.11 new-seed rollout replication."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


SCHEMA = "relational_teacher_v9811_rollout_aligned_replication_v1"
POLICY_SCHEMA = "relational_teacher_v9811_rollout_aligned_replication_policy_v1"
V9810_SCHEMA = "relational_teacher_v9810_rollout_aligned_recovery_v1"
V988_SCHEMA = "relational_teacher_v988_rollout_aligned_direction_v1"
SOURCE_SCENE = "room_0101"
AUDIT_SCENE = "room_0102"
DEVELOPMENT_SCENE = "room_0201"
PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
SELECTED_V984_STEP = 4
RECONSTRUCTION_STEPS = (1, 2, 3, 4)
SELECTED_TIMESTEP = 50
SELECTED_DIRECTION = "audit_bed_guard_0p10"
SELECTED_RADIUS = 0.006
MODEL_SEED = 20261016
REPLICATION_TAG = 20261027
GENERATION_COUNT = 3
LORA_RANK = 4
LORA_ALPHA = 8.0


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def replication_rollout_seeds(generation: int) -> tuple[int, int]:
    if generation < 0 or generation >= GENERATION_COUNT:
        raise ValueError("replication generation must be 0, 1 or 2")
    digest = hashlib.sha256(
        "{}|v9811|new-seed|{}".format(MODEL_SEED, generation).encode("utf-8")
    ).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


REPLICATION_SEED_TABLE = tuple(
    replication_rollout_seeds(generation) for generation in range(GENERATION_COUNT)
)


POLICY = {
    "schema": POLICY_SCHEMA,
    "decision_unit": "new_seed_two_scene_t50_to_t0_k3_selected_state_replication",
    "authority": "sealed_v9810_pass_selected_radius_0p006_only",
    "initialization": "fresh_v5r4_zero_output_lora",
    "reconstruction": "exact_v984_updates_1_through_4_on_room0101",
    "selected_direction": SELECTED_DIRECTION,
    "selected_radius": SELECTED_RADIUS,
    "candidate_state": "exact_v9810_selected_lora_state_reproduced_in_memory",
    "replication_seed_table": [list(row) for row in REPLICATION_SEED_TABLE],
    "seed_exclusion": "disjoint_from_v97_v988_v989_v9810_selection_seed_table",
    "inference_schedule": "frozen_base_t499_through_t51_then_selected_lora_t50_through_t0",
    "scenes": [SOURCE_SCENE, AUDIT_SCENE],
    "generation_count": GENERATION_COUNT,
    "prompt_ids": list(PROMPT_IDS),
    "scene_gate": "exact_v985_strict_three_role_response_policy_applied_independently",
    "topk_role": "diagnostic_only",
    "absolute_three_object_presence": "reported_diagnostic_not_a_replication_gate",
    "checkpoint_policy": "no_optimizer_and_no_model_state_serialization",
    "pass_authority": "fresh_two_scene_rollout_aligned_multiupdate_calibration_only",
    "fail_authority": "cross_scene_training_objective_redesign_only",
}


POLICY_ID = canonical_sha256(POLICY)


def replication_checks(
    *,
    source_checks: Mapping[str, object],
    audit_checks: Mapping[str, object],
    selected_state_exact: bool,
    selected_v5_exact: bool,
    deterministic_repeats_exact: bool,
    seed_table_disjoint: bool,
) -> dict[str, bool]:
    if not source_checks or not audit_checks:
        raise ValueError("both scene check mappings are required")
    if not all(isinstance(value, bool) for value in source_checks.values()):
        raise ValueError("source scene checks are not booleans")
    if not all(isinstance(value, bool) for value in audit_checks.values()):
        raise ValueError("audit scene checks are not booleans")
    return {
        "selected_v9810_state_exactly_reproduced": selected_state_exact,
        "selected_v9810_v5_response_exactly_reproduced": selected_v5_exact,
        "new_seed_base_resumes_are_exact": deterministic_repeats_exact,
        "replication_seed_table_is_disjoint": seed_table_disjoint,
        "room0101_new_seed_response_is_admissible": all(source_checks.values()),
        "room0102_new_seed_response_is_admissible": all(audit_checks.values()),
    }


__all__ = [
    "AUDIT_SCENE",
    "DEVELOPMENT_SCENE",
    "GENERATION_COUNT",
    "LORA_ALPHA",
    "LORA_RANK",
    "MODEL_SEED",
    "POLICY",
    "POLICY_ID",
    "PROMPT_IDS",
    "RECONSTRUCTION_STEPS",
    "REPLICATION_SEED_TABLE",
    "REPLICATION_TAG",
    "SCHEMA",
    "SELECTED_DIRECTION",
    "SELECTED_RADIUS",
    "SELECTED_TIMESTEP",
    "SELECTED_V984_STEP",
    "SOURCE_SCENE",
    "V9810_SCHEMA",
    "V988_SCHEMA",
    "canonical_sha256",
    "replication_checks",
    "replication_rollout_seeds",
]

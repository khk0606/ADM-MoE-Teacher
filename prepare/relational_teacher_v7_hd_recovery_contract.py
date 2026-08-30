#!/usr/bin/env python3
"""Sealed constants for the Teacher-v7 High-Desk invariance recovery."""

from __future__ import annotations

from relational_teacher_v7_hd_calibration_contract import (
    CANDIDATE_WEIGHT,
    DEFAULT_LR,
    EXPECTED_WEIGHTS as FAILED_EXPECTED_WEIGHTS,
    LORA_WEIGHT,
    NEGATIVE_WEIGHT,
    NEW_DENSE_WEIGHT,
    PRESERVATION_WEIGHT,
    STEPS,
    V5_REPLAY_WEIGHT,
)


SCHEMA = "relational_teacher_v7_hd_lora_invariance_recovery_v1"
INVARIANCE_WEIGHT = 0.75
SEALED_FAILED_CHECKS = ["fixed_watch_write_invariance_not_worse"]

EXPECTED_WEIGHTS = {
    "new_dense_within_combined": NEW_DENSE_WEIGHT,
    "v5_replay_dense_within_combined": V5_REPLAY_WEIGHT,
    "combined_dense": 1.0,
    "candidate_semantic": CANDIDATE_WEIGHT,
    "negative_object_suppression": NEGATIVE_WEIGHT,
    "watch_write_invariance": INVARIANCE_WEIGHT,
    "frozen_v5_preservation": PRESERVATION_WEIGHT,
    "lora": LORA_WEIGHT,
}


def validate_failed_near_miss(value: object, calibration_schema: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("sealed failed Teacher-v7 summary is absent")
    if value.get("schema") != calibration_schema:
        raise ValueError("sealed failed Teacher-v7 schema changed")
    if value.get("status") != "CALIBRATION_FAIL":
        raise ValueError("sealed Teacher-v7 source is not the failed calibration")
    if value.get("failed_checks") != SEALED_FAILED_CHECKS:
        raise ValueError("sealed Teacher-v7 near-miss identity changed")
    if int(value.get("steps", -1)) != STEPS:
        raise ValueError("sealed Teacher-v7 near-miss step count changed")
    if float(value.get("learning_rate", -1.0)) != DEFAULT_LR:
        raise ValueError("sealed Teacher-v7 near-miss learning rate changed")
    if value.get("objective_weights") != FAILED_EXPECTED_WEIGHTS:
        raise ValueError("sealed Teacher-v7 near-miss objective changed")
    if value.get("development_payloads_read") is not False:
        raise ValueError("sealed Teacher-v7 near miss read development payloads")
    if value.get("fresh_zero_init_from_sealed_v5r4") is not True:
        raise ValueError("sealed Teacher-v7 near miss was not fresh")
    if value.get("prior_v6_lora_checkpoint_loaded") is not False:
        raise ValueError("sealed Teacher-v7 near miss loaded prior LoRA")
    if value.get("authorizes_bounded_train_only_pilot") is not False:
        raise ValueError("failed Teacher-v7 near miss overstates authority")
    for key in (
        "authorizes_long_training",
        "authorizes_development_evaluation",
        "authorizes_paper_test",
    ):
        if value.get(key) is not False:
            raise ValueError("failed Teacher-v7 authority changed: " + key)

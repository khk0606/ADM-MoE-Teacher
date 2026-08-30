#!/usr/bin/env python3
"""CPU-only tamper tests for the Teacher-v7 invariance recovery contract."""

from __future__ import annotations

from copy import deepcopy

from relational_teacher_v7_hd_calibration_contract import (
    DEFAULT_LR,
    EXPECTED_WEIGHTS as FAILED_WEIGHTS,
    SCHEMA as CALIBRATION_SCHEMA,
    STEPS,
)
from relational_teacher_v7_hd_recovery_contract import (
    EXPECTED_WEIGHTS,
    INVARIANCE_WEIGHT,
    validate_failed_near_miss,
)


def failed_summary():
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "CALIBRATION_FAIL",
        "failed_checks": ["fixed_watch_write_invariance_not_worse"],
        "steps": STEPS,
        "learning_rate": DEFAULT_LR,
        "objective_weights": FAILED_WEIGHTS,
        "development_payloads_read": False,
        "fresh_zero_init_from_sealed_v5r4": True,
        "prior_v6_lora_checkpoint_loaded": False,
        "authorizes_bounded_train_only_pilot": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }


def main() -> None:
    source = failed_summary()
    validate_failed_near_miss(source, CALIBRATION_SCHEMA)
    assert FAILED_WEIGHTS["watch_write_invariance"] == 0.5
    assert INVARIANCE_WEIGHT == 0.75
    assert EXPECTED_WEIGHTS["watch_write_invariance"] == 0.75
    assert {
        key: value
        for key, value in EXPECTED_WEIGHTS.items()
        if key != "watch_write_invariance"
    } == {
        key: value
        for key, value in FAILED_WEIGHTS.items()
        if key != "watch_write_invariance"
    }

    for key, forged in (
        ("status", "CALIBRATION_PASS"),
        ("failed_checks", []),
        ("development_payloads_read", True),
        ("authorizes_bounded_train_only_pilot", True),
    ):
        tampered = deepcopy(source)
        tampered[key] = forged
        try:
            validate_failed_near_miss(tampered, CALIBRATION_SCHEMA)
        except ValueError:
            continue
        raise AssertionError("recovery accepted tampered failed source: " + key)

    print("[PASS] sealed v7 near-miss identity and authority guards")
    print("[PASS] only watch/write invariance changes from 0.5 to 0.75")


if __name__ == "__main__":
    main()

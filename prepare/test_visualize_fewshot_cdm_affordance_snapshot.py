#!/usr/bin/env python3
"""Lightweight contract tests for the cached affordance snapshot."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize_fewshot_cdm_affordance_snapshot import (  # noqa: E402
    binary_f1,
    deterministic_representatives,
    reduce_affordance,
    validate_contract,
)


def test_channel_reducers() -> None:
    value = np.zeros((8192, 6), dtype=np.float32)
    value[:, 0] = 0.25
    value[:, 5] = 0.75
    assert np.array_equal(reduce_affordance(value, "pelvis"), value[:, 0])
    assert np.array_equal(reduce_affordance(value, "any_joint"), value[:, 5])
    try:
        reduce_affordance(value, "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid channel accepted")
    print("[PASS] pelvis and any-joint visualization reducers")


def test_representative_policy() -> None:
    rows = {
        "chair_b": {"target": "chair"},
        "chair_a": {"target": "chair"},
        "bed_z": {"target": "bed"},
        "whiteboard_c": {"target": "whiteboard"},
    }
    assert deterministic_representatives(rows) == {
        "chair": "chair_a",
        "bed": "bed_z",
        "whiteboard": "whiteboard_c",
    }
    print("[PASS] deterministic non-cherry-picked representative policy")


def test_binary_f1() -> None:
    target = np.asarray([0.0, 0.8, 0.9, 0.0], dtype=np.float32)
    perfect = np.asarray([0.0, 0.7, 1.0, 0.1], dtype=np.float32)
    partial = np.asarray([0.0, 0.8, 0.1, 0.9], dtype=np.float32)
    assert binary_f1(perfect, target, 0.7) == 1.0
    assert abs(binary_f1(partial, target, 0.7) - 0.5) < 1e-12
    print("[PASS] target-region F1 metric")


def test_source_contract() -> None:
    source = Path(__file__).with_name(
        "visualize_fewshot_cdm_affordance_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "load_prediction" in source
    assert "prediction_path" in source
    assert '"train"' in source
    assert "heldout_sample_tensors_read" in source
    assert "cache_only_no_diffusion_sampling" in source
    assert "sample_contact" not in source
    assert "create_model" not in source
    assert "candidate-step\", type=int, default=3750" in source
    assert "k_samples < 5" in source
    contract_source = inspect.getsource(validate_contract)
    assert 'selection.get("partition") != "train_complete"' in contract_source
    assert 'selection.get("heldout_sample_tensors_read") is not False' in contract_source
    assert "split file changed after rollout" in contract_source
    assert "candidate checkpoint changed after rollout" in contract_source
    print("[PASS] cache-bound train-only no-sampling source contract")


def main() -> None:
    test_channel_reducers()
    test_representative_policy()
    test_binary_f1()
    test_source_contract()
    print("[PASS] cached affordance snapshot unit contract")


if __name__ == "__main__":
    main()

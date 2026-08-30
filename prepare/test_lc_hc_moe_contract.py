#!/usr/bin/env python3
"""Regression checks for LC17+HC6 state-mode and source contracts."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def install_torch_stubs() -> None:
    """Allow importing label helpers without requiring local CUDA/PyTorch."""
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", torch.nn)
    sys.modules.setdefault("torch.nn.functional", torch.nn.functional)


def load_labels_function():
    install_torch_stubs()
    train_iiw = types.ModuleType("prepare.train_iiw_planner")
    train_iiw.ACTIVE_THRESHOLD = 0.7
    train_iiw.NUM_BODIES = 6
    train_iiw.NUM_POINTS = 8192
    train_iiw.STATE_DIM = 4
    train_iiw.TEXT_DIM = 512
    train_iiw.IIWPlannerDataset = object
    for name in (
        "assert_finite", "gradient_l2", "sample_retrieval_metrics",
        "mean_within_scene_prediction_delta", "set_seed",
        "sparse_objective", "tensor_metrics",
    ):
        setattr(train_iiw, name, lambda *args, **kwargs: None)
    sys.modules.setdefault("prepare", types.ModuleType("prepare"))
    sys.modules["prepare.train_iiw_planner"] = train_iiw
    planners = types.ModuleType("models.iiw_planner")
    planners.IIWPlanner = object
    planners.MoEIIWPlanner = object
    sys.modules.setdefault("models", types.ModuleType("models"))
    sys.modules["models.iiw_planner"] = planners
    source = REPO_ROOT / "prepare" / "train_moe_iiw_planner.py"
    spec = importlib.util.spec_from_file_location("lc_hc_train_moe", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.balanced_state_mode_labels


def test_labels() -> None:
    labels_fn = load_labels_function()
    rng = np.random.default_rng(20260813)
    states6 = rng.normal(size=(6, 4)).astype(np.float32)
    labels6_a, meta6_a = labels_fn(states6, 2)
    labels6_b, meta6_b = labels_fn(states6, 2)
    assert np.array_equal(labels6_a, labels6_b)
    assert sorted(meta6_a["counts"]) == [3, 3]
    assert meta6_a == meta6_b

    states23 = rng.normal(size=(23, 4)).astype(np.float32)
    labels23_a, meta23_a = labels_fn(states23, 2)
    labels23_b, meta23_b = labels_fn(states23, 2)
    assert np.array_equal(labels23_a, labels23_b)
    assert sorted(meta23_a["counts"]) == [11, 12]
    assert meta23_a == meta23_b
    assert meta23_a["uses_gt_motion"] is False
    assert meta23_a["uses_gt_iiw"] is False
    print("[PASS] deterministic HC6 3/3 and LC17+HC6 12/11 state labels")


def test_source_contracts() -> None:
    moe_file = REPO_ROOT / "prepare" / "train_moe_iiw_planner.py"
    joint_file = REPO_ROOT / "prepare" / "train_moe_iiw_mapstar_joint.py"
    for path in (moe_file, joint_file):
        source = path.read_text()
        ast.parse(source)
        assert "target_leakage_into_planner_inputs" in source or path == joint_file
    joint = joint_file.read_text()
    assert '"c_pc_contact": mapstar.contiguous()' in joint
    assert "IIWAdapter" not in joint
    assert "c_iiw_residual" not in joint
    assert "base_affordance" in joint and "pc_weight" in joint
    assert "num_samples = len(iiw_dataset)" in joint
    print("[PASS] multi-scene MoE and literal map* source contracts")


if __name__ == "__main__":
    test_labels()
    test_source_contracts()

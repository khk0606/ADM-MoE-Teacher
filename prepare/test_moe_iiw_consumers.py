#!/usr/bin/env python3
"""Lightweight CPU/unit checks for MoE consumer quality and routing logic."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "evaluate_moe_iiw_planner.py"


def passing_checkpoint():
    return {
        "model_type": "moe_iiw_planner",
        "num_experts": 2,
        "summary": {
            "status": "PASS",
            "checks": {
                name: True
                for name in (
                    "dense_trunk_bitwise_unchanged",
                    "combined_objective_improved",
                    "iiw_objective_not_degraded",
                    "router_accuracy_perfect",
                    "both_experts_hard_selected",
                    "both_experts_soft_loaded",
                    "router_confident",
                    "experts_diverged",
                    "sparse_f1_preserved",
                    "temporal_order_preserved",
                    "state_outputs_not_collapsed",
                )
            },
            "final": {
                "routing": {
                    "hard_counts": [3, 3],
                    "soft_load": [0.5, 0.5],
                }
            },
        },
    }


def test_static_contract() -> None:
    text = EVALUATOR.read_text()
    required = (
        "build_iiw_planner_from_checkpoint",
        "forward_with_routing",
        "routing_probabilities",
        "selected_expert",
        "MoE checkpoint failed required quality checks",
        "planner_module.IIWPlanner = _FactoryPlanner",
        "planner_module.IIWPlanner = original",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise AssertionError("consumer source contract missing: " + str(missing))
    print("[PASS] factory, routing, quality-gate, and restoration source contract")


def test_syntax() -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(EVALUATOR)], check=True
    )
    print("[PASS] evaluator syntax contract")


if __name__ == "__main__":
    test_static_contract()
    test_syntax()
    print("[PASS] lightweight MoE IIW consumer contract")

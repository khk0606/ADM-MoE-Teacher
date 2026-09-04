#!/usr/bin/env python3
"""CPU and static tests for Teacher-v9.8 rollout-state response."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v98_rollout_state_contract import (
    CAPTURE_TIMESTEPS,
    OBJECTS,
    POLICY,
    POLICY_ID,
    STEP_RADII,
    canonical_sha256,
    design_seeds,
    pooled,
    rank_candidates,
    response_checks,
)


def metric(recall: float, mae: float) -> dict:
    return {
        "instances": {
            name: {
                "soft_recall": recall,
                "active_support_mae": mae,
                "hotspot_centroid_distance_xy": 0.1,
                "topk_overlap": 0.3,
            }
            for name in OBJECTS
        },
        "explicit_negative_mean": 0.03,
        "explicit_negative_max": 0.4,
    }


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("policy ID changed")
    if CAPTURE_TIMESTEPS != (400, 200, 50) or STEP_RADII != (0.001, 0.003, 0.01):
        raise AssertionError("response grid changed")
    if POLICY["topk_role"] != "diagnostic_only":
        raise AssertionError("exact Top-k became a gate")
    base = [metric(0.20, 0.60), metric(0.21, 0.59)]
    candidate = [metric(0.23, 0.56), metric(0.24, 0.55)]
    checks = response_checks(
        base_rows=base,
        candidate_rows=candidate,
        base_prompt_invariance=0.02,
        candidate_prompt_invariance=0.0205,
        base_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.0501, 0.0601, 0.0701),
        directional_derivatives=(0.1,) * 6,
        maximum_map_delta=0.01,
    )
    if not all(checks.values()):
        raise AssertionError("valid rollout-state response was rejected")
    regression = copy.deepcopy(candidate)
    regression[0]["instances"]["chair_06"]["soft_recall"] = 0.1
    failed = response_checks(
        base_rows=base,
        candidate_rows=regression,
        base_prompt_invariance=0.02,
        candidate_prompt_invariance=0.0205,
        base_v5_dense=(0.05, 0.06, 0.07),
        candidate_v5_dense=(0.0501, 0.0601, 0.0701),
        directional_derivatives=(0.1,) * 6,
        maximum_map_delta=0.01,
    )
    if failed["watch_chair_06_recall_retained"] is not False:
        raise AssertionError("High Chair regression passed")
    base_pooled = pooled(base)
    candidate_pooled = pooled(candidate)
    candidates = [
        {
            "name": name,
            "eligible": True,
            "capture_timestep": timestep,
            "radius": radius,
            "base_pooled": base_pooled,
            "candidate_pooled": candidate_pooled,
        }
        for name, timestep, radius in (
            ("large", 400, 0.01),
            ("small", 200, 0.001),
        )
    ]
    if rank_candidates(candidates) != ["small", "large"]:
        raise AssertionError("candidate ranking changed")
    if design_seeds() != design_seeds() or len(set(design_seeds())) != 2:
        raise AssertionError("design seeds are not stable and distinct")

    runner = Path(__file__).with_name(
        "preflight_relational_teacher_v98_rollout_state_response.py"
    )
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(loads) != 1:
        raise AssertionError("runner must contain one room_0101 array load")
    if "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("v9.8 must not save model state")
    if source.index("atomic_write_json(policy_file") > source.index("install_lora(model"):
        raise AssertionError("policy is written after model adaptation exists")
    for literal in ("p_sample_loop_progressive", "torch.cuda.set_rng_state", "CAPTURE_TIMESTEPS"):
        if literal not in source:
            raise AssertionError("runner lacks trajectory-resume guard: " + literal)
    print("[PASS] Teacher-v9.8 rollout-state response CPU contract")
    print("[PASS] High Chair regression, trust leakage and invalid direction fail closed")
    print("[PASS] one-scene load, pre-model policy, trajectory resume and no-checkpoint guards")


if __name__ == "__main__":
    main()

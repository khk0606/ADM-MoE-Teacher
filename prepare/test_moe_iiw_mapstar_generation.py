#!/usr/bin/env python3
"""CPU-only unit tests for paired generation metrics and map audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare.evaluate_moe_iiw_mapstar_generation import (  # noqa: E402
    NUM_BODY_PARTS,
    NUM_POINTS,
    bootstrap_paired_difference,
    mapstar_instance_audit,
    motion_metrics,
    preserve_selected_object_context,
    paired_comparisons,
    phaseweighted_pc_weight,
)
import torch  # noqa: E402


def test_paired_bootstrap_sign() -> None:
    result = bootstrap_paired_difference(
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        seed=7,
        replicates=2000,
    )
    assert result["mean_delta_m"] < 0.0
    assert result["ci95_high_m"] < 0.0
    assert result["sample_win_rate"] == 1.0
    print("[PASS] paired bootstrap uses negative delta as predicted win")


def test_motion_metrics_identity() -> None:
    target = np.zeros((20, 22, 3), dtype=np.float32)
    target[:, 0, 2] = np.linspace(1.0, 0.7, 20)
    metrics = motion_metrics(target.copy(), target, sitting_start=10)
    assert all(value == 0.0 for value in metrics.values())
    print("[PASS] identical generated and GT motions have zero error")


def test_chair_selective_map_audit() -> None:
    ids = np.zeros(NUM_POINTS, dtype=np.int64)
    boundaries = [2934, 2934 + 1503, 2934 + 1503 + 1502,
                  2934 + 1503 + 1502 + 1502]
    ids[boundaries[0]:boundaries[1]] = 1
    ids[boundaries[1]:boundaries[2]] = 2
    ids[boundaries[2]:boundaries[3]] = 3
    ids[boundaries[3]:] = 4
    weight = np.full(NUM_POINTS, 1e-4, dtype=np.float32)
    weight[ids == 1] = 0.8
    base = np.ones((NUM_POINTS, NUM_BODY_PARTS), dtype=np.float32)
    mapstar = base * weight[:, None]
    result = mapstar_instance_audit(weight, mapstar, ids)
    assert result["chair_dominates_pelvis"]
    assert result["chair_dominates_any_joint"]
    assert result["channels"]["pelvis"]["top10_instance_fraction"]["chair"] == 1.0
    print("[PASS] selective pc_weight preserves chair and suppresses other instances")


def test_context_preserves_only_nearby_environment() -> None:
    ids = torch.zeros((1, NUM_POINTS), dtype=torch.long)
    ids[0, 2934:2934 + 1503] = 1
    ids[0, 2934 + 1503:2934 + 1503 + 1502] = 2
    ids[0, 2934 + 1503 + 1502:2934 + 1503 + 1502 + 1502] = 3
    ids[0, 2934 + 1503 + 1502 + 1502:] = 4
    xyz = torch.full((1, NUM_POINTS, 3), 10.0)
    xyz[0, ids[0] == 1] = torch.tensor([0.0, 0.0, 0.0])
    environment = torch.nonzero(ids[0] == 0).flatten()
    xyz[0, environment[:100]] = torch.tensor([0.2, 0.0, 0.0])
    base = torch.ones((1, NUM_POINTS, NUM_BODY_PARTS))
    weight = torch.full((1, NUM_POINTS), 1e-4)
    weight[0, ids[0] == 1] = 0.8
    expanded_map, expanded_weight, selected, count = (
        preserve_selected_object_context(
            base, weight, xyz, ids, radius=0.5, chunk_size=128
        )
    )
    assert selected == 1
    assert count == 100
    assert torch.all(expanded_weight[0, environment[:100]] == 1.0)
    assert torch.all(expanded_weight[0, ids[0] == 2] == 1e-4)
    assert torch.equal(expanded_map, base * expanded_weight.unsqueeze(-1))
    print("[PASS] context dilation restores only nearby environment Base ADM")


def test_context_floor_stays_scalar_and_bounded() -> None:
    ids = torch.zeros((1, NUM_POINTS), dtype=torch.long)
    ids[0, 2934:2934 + 1503] = 1
    ids[0, 2934 + 1503:2934 + 1503 + 1502] = 2
    ids[0, 2934 + 1503 + 1502:2934 + 1503 + 1502 + 1502] = 3
    ids[0, 2934 + 1503 + 1502 + 1502:] = 4
    xyz = torch.full((1, NUM_POINTS, 3), 10.0)
    xyz[0, ids[0] == 1] = 0.0
    environment = torch.nonzero(ids[0] == 0).flatten()
    xyz[0, environment[:64]] = torch.tensor([0.1, 0.0, 0.0])
    base = torch.rand((1, NUM_POINTS, NUM_BODY_PARTS))
    weight = torch.full((1, NUM_POINTS), 0.01)
    weight[0, ids[0] == 1] = 0.8
    mapstar, expanded, selected, count = preserve_selected_object_context(
        base, weight, xyz, ids, radius=1.0, context_floor=0.5,
        chunk_size=128,
    )
    assert selected == 1 and count == 64
    assert torch.all(expanded[0, environment[:64]] == 0.5)
    assert float(expanded.min()) >= 0.0 and float(expanded.max()) <= 1.0
    assert torch.equal(mapstar, base * expanded.unsqueeze(-1))
    print("[PASS] context floor remains one bounded scalar pc_weight per point")


def test_context_mode_can_be_paired_primary() -> None:
    rows = []
    for sample_index in range(3):
        for mode, value in (
            ("predicted_context_r10", 0.1 + sample_index),
            ("zero_weight", 0.2 + sample_index),
        ):
            row = {
                "sample_id": "sample_{}".format(sample_index),
                "mode": mode,
            }
            for metric in (
                "mpjpe_global_m",
                "mpjpe_start_aligned_m",
                "mpjpe_root_relative_pose_m",
                "root_ade_xy_m",
                "root_fde_xy_m",
                "sitting_root_position_mae_m",
                "sitting_pelvis_height_mae_m",
            ):
                row[metric] = value
            rows.append(row)
    result = paired_comparisons(
        rows,
        ("predicted_context_r10", "zero_weight"),
        "predicted_context_r10",
        seed=9,
        replicates=2000,
    )
    key = "predicted_context_r10_vs_zero_weight"
    assert result[key]["mpjpe_global_m"]["mean_delta_m"] < 0.0
    print("[PASS] context map can be the explicit paired primary mode")


def test_alpha_context_modes_have_matched_shuffled_controls() -> None:
    from prepare.evaluate_moe_iiw_mapstar_generation import (
        ALL_MODES, CONTEXT_FLOOR_BY_MODE, CONTEXT_RADIUS_BY_MODE,
    )

    for suffix, floor in (("a025", 0.25), ("a050", 0.50), ("a075", 0.75)):
        primary = "predicted_context_r10_" + suffix
        shuffled = "predicted_shuffled_context_r10_" + suffix
        assert primary in ALL_MODES and shuffled in ALL_MODES
        assert CONTEXT_RADIUS_BY_MODE[primary] == 1.0
        assert CONTEXT_RADIUS_BY_MODE[shuffled] == 1.0
        assert CONTEXT_FLOOR_BY_MODE[primary] == floor
        assert CONTEXT_FLOOR_BY_MODE[shuffled] == floor
    print("[PASS] alpha context modes have matched shuffled-history controls")


def test_phaseweighted_reducer_preserves_history_and_order() -> None:
    base = torch.ones((1, NUM_POINTS, NUM_BODY_PARTS))
    plan = torch.zeros((1, 8, NUM_POINTS, NUM_BODY_PARTS))
    plan[:, 0, 10, 0] = 1.0
    plan[:, 7, 20, 0] = 1.0
    mapstar, weight, phase_weight = phaseweighted_pc_weight(base, plan)
    reversed_map, reversed_weight, _ = phaseweighted_pc_weight(
        base, plan.flip(1)
    )
    assert tuple(phase_weight.shape) == (1, 8, NUM_POINTS)
    assert torch.isclose(weight[0, 10], torch.tensor(0.25))
    assert torch.isclose(weight[0, 20], torch.tensor(1.0))
    assert torch.equal(mapstar, base * weight.unsqueeze(-1))
    assert not torch.equal(weight, reversed_weight)
    assert not torch.equal(mapstar, reversed_map)
    print("[PASS] phaseweighted scalar pc_weight keeps path and temporal order")


def main() -> None:
    test_paired_bootstrap_sign()
    test_motion_metrics_identity()
    test_chair_selective_map_audit()
    test_context_preserves_only_nearby_environment()
    test_context_floor_stays_scalar_and_bounded()
    test_context_mode_can_be_paired_primary()
    test_alpha_context_modes_have_matched_shuffled_controls()
    test_phaseweighted_reducer_preserves_history_and_order()
    print("[PASS] MoE-IIW map* generation evaluation unit contract")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU unit contract for dense -> residual-MoE IIW migration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PATCH_ROOT = Path(__file__).resolve().parents[1]
if str(PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCH_ROOT))

from models.iiw_planner import (  # noqa: E402
    IIWPlanner,
    MoEIIWPlanner,
    build_iiw_planner_from_checkpoint,
)


def config():
    return {
        "scene_dim": 6, "text_dim": 12, "state_dim": 4,
        "num_phases": 8, "num_bodies": 6, "hidden_dim": 24,
        "point_dim": 16, "context_dim": 20,
    }


def inputs():
    torch.manual_seed(20260813)
    scene = torch.randn(3, 17, 6)
    text = torch.randn(3, 12)
    state = torch.randn(3, 4)
    state[:, 2:] = F.normalize(state[:, 2:], dim=-1)
    return scene, text, state


def main() -> None:
    dense = IIWPlanner(**config()).eval()
    checkpoint = {
        "model": dense.state_dict(),
        "model_config": config(),
        "state_mean": torch.zeros(4),
        "state_std": torch.ones(4),
    }
    migrated = build_iiw_planner_from_checkpoint(
        checkpoint,
        migrate_dense_to_moe=True,
        moe_overrides={
            "num_experts": 2,
            "residual_rank": 5,
            "router_hidden_dim": 9,
            "routing_temperature": 1.0,
        },
    ).eval()
    assert isinstance(migrated, MoEIIWPlanner)
    x = inputs()
    with torch.no_grad():
        dense_logits = dense.forward_logits(*x)
        result = migrated.forward_with_routing(*x, return_expert_logits=True)
    assert torch.equal(dense_logits, result["logits"])
    assert result["prediction"].shape == (3, 8, 17, 6)
    assert result["expert_logits"].shape == (3, 2, 8, 17, 6)
    torch.testing.assert_close(
        result["routing_probabilities"].sum(-1), torch.ones(3)
    )
    assert set(dense.state_dict()).issubset(set(migrated.state_dict()))

    # Once experts move, the gate and both residual experts receive gradient.
    with torch.no_grad():
        migrated.expert_query_projection.normal_(0.0, 0.02)
    result = migrated.forward_with_routing(*x, return_expert_logits=True)
    loss = result["logits"].square().mean()
    loss.backward()
    assert migrated.expert_point_projection.grad is not None
    assert migrated.expert_query_projection.grad is not None
    assert migrated.router[-1].weight.grad is not None

    moe_checkpoint = {
        "model": migrated.state_dict(),
        "model_config": migrated.config,
        "planner_type": "moe_iiw_v1",
        "format_version": 2,
        "state_mean": migrated.state_mean,
        "state_std": migrated.state_std,
    }
    restored = build_iiw_planner_from_checkpoint(moe_checkpoint, strict=True)
    assert isinstance(restored, MoEIIWPlanner)
    with torch.no_grad():
        torch.testing.assert_close(restored.forward_logits(*x), migrated.forward_logits(*x))
    print("[PASS] dense checkpoint keys and output remain backward compatible")
    print("[PASS] zero-residual migration has bitwise exact dense logit parity")
    print("[PASS] soft sample-global routing and expert gradients")
    print("[PASS] version-2 MoE checkpoint factory round trip")


if __name__ == "__main__":
    main()

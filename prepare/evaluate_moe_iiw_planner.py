#!/usr/bin/env python3
"""Evaluate either a legacy dense or state-routed MoE IIW planner.

This is the production evaluator with one deliberate extension: checkpoint
construction goes through :func:`build_iiw_planner_from_checkpoint`.  Dense
checkpoints therefore retain their old behaviour, while MoE checkpoints also
emit and validate per-sample routing diagnostics.

The frozen IIWAdapter/CMDM evaluation remains identical to the dense evaluator
implemented in ``prepare/evaluate_iiw_planner.py``.  To keep that substantial
fixed-noise contract single-sourced, this wrapper temporarily installs a
factory-backed compatibility class and executes the canonical evaluator.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.iiw_planner as planner_module  # noqa: E402


REQUIRED_MOE_CHECKS = (
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


def _is_moe_checkpoint(checkpoint: Mapping) -> bool:
    config = checkpoint.get("model_config", {})
    if not isinstance(config, Mapping):
        config = {}
    value = str(
        checkpoint.get(
            "model_type",
            config.get("planner_type", ""),
        )
    ).lower()
    return "moe" in value


def _strict_moe_quality_gate(checkpoint: Mapping) -> None:
    """Reject a MoE checkpoint whose anti-collapse contract did not pass."""
    if not _is_moe_checkpoint(checkpoint):
        return
    summary = checkpoint.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("MoE checkpoint has no summary mapping")
    if str(summary.get("status")) != "PASS":
        raise ValueError("MoE checkpoint status is not PASS")
    checks = summary.get("checks")
    if not isinstance(checks, Mapping):
        raise TypeError("MoE checkpoint has no strict checks mapping")
    failed = [name for name in REQUIRED_MOE_CHECKS if not bool(checks.get(name))]
    if failed:
        raise ValueError(
            "MoE checkpoint failed required quality checks: " + ", ".join(failed)
        )
    final = summary.get("final")
    routing = final.get("routing") if isinstance(final, Mapping) else None
    if not isinstance(routing, Mapping):
        raise TypeError("MoE checkpoint has no final routing metrics")
    num_experts = int(checkpoint.get("num_experts", 0))
    hard_counts = [int(value) for value in routing.get("hard_counts", [])]
    soft_load = [float(value) for value in routing.get("soft_load", [])]
    if num_experts < 2 or len(hard_counts) != num_experts:
        raise ValueError("MoE routing expert/count metadata is inconsistent")
    if len(soft_load) != num_experts or min(soft_load) <= 0.0:
        raise ValueError("MoE checkpoint has a collapsed/invalid soft expert load")


class _FactoryPlanner:
    """Constructor-compatible factory whose instances retain native classes."""

    _checkpoint: Optional[Mapping] = None
    _device: Optional[torch.device] = None
    _routing_rows: list = []

    @classmethod
    def _capture_routing(cls, native):
        original = native.forward_with_routing

        def routed(scene_points, text_features, state, return_expert_logits=False):
            output = original(
                scene_points, text_features, state
            )
            required = {
                "prediction", "routing_probabilities", "selected_expert",
                "routing_entropy", "expert_load",
            }
            missing = sorted(required.difference(output))
            if missing:
                raise KeyError("MoE routing output missing: " + ", ".join(missing))
            probabilities = output["routing_probabilities"]
            selected = output["selected_expert"]
            if probabilities.ndim != 2 or selected.shape != probabilities.shape[:1]:
                raise ValueError("MoE routing tensor shape mismatch")
            if not bool(torch.isfinite(probabilities).all().item()):
                raise ValueError("MoE routing probabilities contain NaN/Inf")
            if not torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities[:, 0]),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("MoE routing probabilities do not sum to one")
            cls._routing_rows.append({
                "probabilities": probabilities.detach().cpu().numpy(),
                "selected_expert": selected.detach().cpu().numpy(),
                "routing_entropy": output["routing_entropy"].detach().cpu().numpy(),
                "expert_load": output["expert_load"].detach().cpu().numpy(),
            })
            return output

        native.forward_with_routing = routed
        def forward(scene_points, text_features, state):
            return native.forward_with_routing(
                scene_points, text_features, state
            )["prediction"]

        native.forward = forward
        return native

    def __new__(cls, **_: object):
        if cls._checkpoint is None:
            raise RuntimeError("factory evaluator checkpoint was not installed")
        native = planner_module.build_iiw_planner_from_checkpoint(
            cls._checkpoint,
            device=cls._device,
            strict=True,
        )
        if hasattr(native, "forward_with_routing"):
            native = cls._capture_routing(native)
        return native


@contextmanager
def _factory_patch(checkpoint: Mapping, device: torch.device):
    original = planner_module.IIWPlanner
    _FactoryPlanner._checkpoint = checkpoint
    _FactoryPlanner._device = device
    _FactoryPlanner._routing_rows = []
    planner_module.IIWPlanner = _FactoryPlanner
    try:
        yield
    finally:
        planner_module.IIWPlanner = original
        _FactoryPlanner._checkpoint = None
        _FactoryPlanner._device = None


def _write_routing_report(output_dir: Path, checkpoint: Mapping) -> None:
    if not _is_moe_checkpoint(checkpoint):
        print("[OK] dense planner: routing report is not applicable")
        return
    rows = _FactoryPlanner._routing_rows
    if not rows:
        raise AssertionError("MoE evaluator collected no routing diagnostics")
    probabilities = np.concatenate([row["probabilities"] for row in rows], axis=0)
    selected = np.concatenate([row["selected_expert"] for row in rows], axis=0)
    entropy = np.concatenate(
        [np.asarray(row["routing_entropy"]).reshape(-1) for row in rows], axis=0
    )
    num_experts = probabilities.shape[1]
    hard_counts = np.bincount(selected.astype(np.int64), minlength=num_experts)
    soft_load = probabilities.mean(axis=0)
    report: Dict[str, object] = {
        "status": "PASS",
        "planner_type": "moe_iiw",
        "num_forward_items": int(probabilities.shape[0]),
        "num_experts": int(num_experts),
        "hard_counts_across_all_forward_calls": hard_counts.tolist(),
        "soft_load_across_all_forward_calls": soft_load.astype(float).tolist(),
        "mean_routing_entropy": float(entropy.mean()),
        "min_max_probability": float(probabilities.max(axis=1).min()),
        "note": (
            "Rows include repeated downstream evaluation calls; use the "
            "checkpoint training summary for six-sample routing accuracy."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "moe_routing_eval.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("[PASS] MoE routing remained finite, normalized, and observable")
    print("[OK] saved: " + str(path))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    known, _ = parser.parse_known_args()
    planner_file = known.planner_checkpoint.expanduser().resolve()
    if not planner_file.is_file():
        raise FileNotFoundError(planner_file)
    checkpoint = torch.load(str(planner_file), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("planner checkpoint must be a mapping")
    _strict_moe_quality_gate(checkpoint)
    device = torch.device(known.device)

    canonical = REPO_ROOT / "prepare" / "evaluate_iiw_planner.py"
    if not canonical.is_file():
        raise FileNotFoundError(
            "install iiw_pipeline_patch/prepare/evaluate_iiw_planner.py as "
            + str(canonical)
        )
    with _factory_patch(checkpoint, device):
        runpy.run_path(str(canonical), run_name="__main__")
        routing_rows = list(_FactoryPlanner._routing_rows)
    _FactoryPlanner._routing_rows = routing_rows
    _write_routing_report(known.output_dir.expanduser().resolve(), checkpoint)


if __name__ == "__main__":
    main()

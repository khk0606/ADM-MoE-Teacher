#!/usr/bin/env python3
"""Dependency-free source contract for the strict map* joint trainer."""

from __future__ import annotations

import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_moe_iiw_mapstar_joint.py"
ROUTER = HERE.parent / "models" / "iiw_map_router.py"


def require(source: str, token: str) -> None:
    if token not in source:
        raise AssertionError("required source contract is absent: " + token)


def forbid(source: str, token: str) -> None:
    if token in source:
        raise AssertionError("forbidden legacy path re-entered trainer: " + token)


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("function is absent: " + name)


def test_sources_compile() -> None:
    for path in (TRAINER, ROUTER):
        source = path.read_text()
        compile(source, str(path), "exec")


def test_only_mapstar_reaches_cmdm_contact() -> None:
    source = TRAINER.read_text()
    tree = ast.parse(source)
    node = function_node(tree, "cmdm_conditioning")
    dictionaries = [value for value in ast.walk(node) if isinstance(value, ast.Dict)]
    if len(dictionaries) != 1:
        raise AssertionError("cmdm_conditioning must return one literal dictionary")
    keys = {
        value.value for value in dictionaries[0].keys
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    expected = {"x_mask", "c_pc_xyz", "c_pc_contact", "c_text"}
    if keys != expected:
        raise AssertionError("unexpected CMDM conditioning keys: " + str(keys))
    require(source, '"c_pc_contact": mapstar.contiguous()')
    require(source, 'map_router = IIWMapRouter(validate_inputs=True)')
    require(source, 'routed["mapstar"]')
    require(source, 'cached_base_affordance * pc_weight[...,None]')
    for forbidden in (
        "IIWAdapter", "IIWConditionedCMDM", "c_iiw_residual",
        "HistoryAffordanceRouter", "HistoryMoE",
    ):
        forbid(source, forbidden)


def test_training_and_freeze_contracts_are_explicit() -> None:
    source = TRAINER.read_text()
    required = (
        'planner_type=moe_iiw_v1',
        'summary.get("status") != "PASS"',
        'summary.get("overfit_quality_pass", False)',
        'straight_through_expert_logits',
        'training=training_route',
        'weights = hard + soft - soft.detach() if training else hard',
        'def backward_motion_only',
        'motion-only loss failed to reach router and every expert',
        'parameter.requires_grad_(False)',
        'cmdm_bitwise_unchanged',
        'dense_iiw_trunk_bitwise_unchanged',
        'iiw_objective_not_degraded',
        'both_experts_hard_selected',
        'motion_grid_improved',
        '"iiw_adapter_used": False',
        '"embedding_residual_used": False',
        '"cmdm_state_sha256": cmdm_hash_after',
    )
    for token in required:
        require(source, token)
    # Old furniture labels must never become expert labels.
    forbid(source, "c_target_index")
    forbid(source, "target_index")
    forbid(source, '"cmdm_state_dict"')


def test_router_has_literal_terminal_scalar_rule() -> None:
    source = ROUTER.read_text()
    for token in (
        "phase_pc_weight = native_iiw.max(dim=-1)[0]",
        "pc_weight = phase_pc_weight[:, -1, :]",
        "frozen_base = base_affordance.detach()",
        "mapstar = frozen_base * pc_weight.unsqueeze(-1)",
    ):
        require(source, token)


def main() -> None:
    test_sources_compile()
    test_only_mapstar_reaches_cmdm_contact()
    test_training_and_freeze_contracts_are_explicit()
    test_router_has_literal_terminal_scalar_rule()
    print("[PASS] trainer and map-router sources compile")
    print("[PASS] CMDM receives only c_pc_contact=mapstar")
    print("[PASS] adapter/residual and furniture-router paths are absent")
    print("[PASS] ST-train/hard-eval and frozen CMDM contracts are explicit")
    print("[PASS] motion-only router/both-expert gradient checks are required")


if __name__ == "__main__":
    main()

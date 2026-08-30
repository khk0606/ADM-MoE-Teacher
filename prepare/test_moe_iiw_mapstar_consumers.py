#!/usr/bin/env python3
"""Lightweight source contract for the literal map* consumer."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "evaluate_moe_iiw_mapstar.py"
CONTRACT = HERE / "iiw_mapstar_eval_contract.py"
AUDIT = HERE / "audit_moe_iiw_mapstar_joint_v2.py"


def require(source: str, marker: str) -> None:
    if marker not in source:
        raise AssertionError("missing map* consumer contract: " + marker)


def reject(source: str, marker: str) -> None:
    if marker in source:
        raise AssertionError("forbidden adapter path in map* consumer: " + marker)


def main() -> None:
    source = EVALUATOR.read_text()
    contract = CONTRACT.read_text()
    audit = AUDIT.read_text()
    for marker in (
        "IIWMapRouter",
        '"c_pc_contact": contact_for(batch, mode)',
        '"identity_weight"',
        '"legacy_base"',
        '"identity_base_tensor_bitwise_equal"',
        "torch.equal(contact, batch[\"base\"])",
        '"quality_status"',
        "REQUIRED_JOINT_CHECKS",
        '"--joint-v2-audit"',
        '"external_noninferiority_v2"',
        "load_external_audit",
        "joint_trainer_state_digest",
        '"uses_iiw_adapter": False',
        '"uses_embedding_residual": False',
        '"pc_weight_definition"',
        '"predicted_beats_zero_weight"',
        "build_iiw_planner_from_checkpoint",
        "REQUIRED_MOE_CHECKS",
        "from prepare.iiw_mapstar_eval_contract import",
    ):
        require(source, marker)
    for marker in (
        "from models.iiw_adapter import",
        "from models.iiw_cmdm import",
        "from prepare.evaluate_iiw_planner import",
        "from prepare.test_iiw_oracle import",
        'kwargs["c_iiw_residual"]',
        "IIWConditionedCMDM(",
        "identity_legacy_diff != 0.0",
    ):
        reject(source, marker)
    for marker in (
        "prepare.evaluate_iiw_planner",
        "prepare.test_iiw_oracle",
        "models.iiw_adapter",
        "models.iiw_cmdm",
    ):
        reject(contract, marker)
    for marker in (
        "IIWMapstarEvalDataset",
        "IIWPlannerDataset",
        "HistoryAffordanceV1ContactMotionDataset",
        "checkpoint_coverage",
        "load_cmdm_config",
        "plan_metrics",
        "aggregate_plan_metrics",
        "joint_trainer_state_digest",
    ):
        require(contract, marker)
    for marker in (
        "hc6_mapstar_noninferiority_v2",
        "MAX_IIW_RELATIVE_DEGRADATION = 0.002",
        "MIN_MOTION_RELATIVE_IMPROVEMENT = 0.02",
        "MAX_F1_ABSOLUTE_DROP = 0.005",
        "MIN_IMPROVED_TIMESTEPS = 5",
        "only_exact_v1_iiw_check_failed",
        "file_sha256(checkpoint_file)",
        "canonical_json_sha256(summary)",
    ):
        require(audit, marker)
    compile(source, str(EVALUATOR), "exec")
    compile(contract, str(CONTRACT), "exec")
    compile(audit, str(AUDIT), "exec")
    print("[PASS] map* consumer uses only CMDM c_pc_contact")
    print("[PASS] IIWAdapter and c_iiw_residual are absent")
    print("[PASS] identity-weight and legacy-base parity gate exists")
    print("[PASS] quality-passed MoE checkpoint is required")
    print("[PASS] neutral map* data/evaluation contract has no legacy dependency")
    print("[PASS] CHECK joint checkpoints require a hash-bound v2 audit")
    print("[PASS] map* evaluator syntax contract")


if __name__ == "__main__":
    main()

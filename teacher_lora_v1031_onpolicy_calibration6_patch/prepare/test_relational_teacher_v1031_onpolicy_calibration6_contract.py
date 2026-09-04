#!/usr/bin/env python3
"""CPU/static contract for Teacher-v10.3.1 calibration."""

from __future__ import annotations

import ast
from pathlib import Path

from relational_teacher_v103_onpolicy_response_contract import (
    EXPECTED_INSTANCES,
    PROMPT_IDS,
    ROLES,
    SCENES,
)
from relational_teacher_v1031_onpolicy_calibration6_contract import (
    GENERATION_COUNT,
    POLICY,
    POLICY_ID,
    all_three_counts,
    calibration_checks,
    canonical_sha256,
    rank_shortlist,
    stable_seed_pair,
)


def _metric(recall: float, mae: float) -> dict:
    return {
        "soft_recall": recall,
        "active_support_mae": mae,
        "topk_overlap": 0.4,
        "hotspot_centroid_distance_xy": 0.1,
    }


def _raw(recall: float, mae: float) -> dict:
    return {
        scene: [
            [
                {
                    "instances": {
                        name: _metric(recall + 0.001 * index, mae - 0.001 * index)
                        for index, name in enumerate(EXPECTED_INSTANCES[scene])
                    },
                    "explicit_negative_mean": 0.02,
                    "explicit_negative_max": 0.2,
                }
                for _ in PROMPT_IDS
            ]
            for _ in range(GENERATION_COUNT)
        ]
        for scene in SCENES
    }


def _pooled(recall: float, mae: float) -> dict:
    return {
        scene: {
            prompt: {
                "instances": {
                    name: _metric(recall + 0.001 * index, mae - 0.001 * index)
                    for index, name in enumerate(EXPECTED_INSTANCES[scene])
                },
                "explicit_negative_mean": 0.02,
                "explicit_negative_max": 0.2,
            }
            for prompt in PROMPT_IDS
        }
        for scene in SCENES
    }


def main() -> None:
    if canonical_sha256(POLICY) != POLICY_ID:
        raise AssertionError("Teacher-v10.3.1 policy hash changed")
    seeds = [
        stable_seed_pair(domain, scene, prompt, generation)
        for domain in ("design", "audit")
        for scene in SCENES
        for generation in range(GENERATION_COUNT)
        for prompt in PROMPT_IDS
    ]
    if len(seeds) != 24 or len(set(seeds)) != 24:
        raise AssertionError("Teacher-v10.3.1 seed domains overlap")
    presence = {
        scene: [
            [
                {"bed": True, "normal": True, "high": generation != 2}
                for _ in PROMPT_IDS
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene in SCENES
    }
    counts = all_three_counts(presence)
    if any(counts[scene][prompt] != 2 for scene in SCENES for prompt in PROMPT_IDS):
        raise AssertionError("Teacher-v10.3.1 all-three counting changed")

    start_raw = _raw(0.4, 0.3)
    candidate_raw = _raw(0.42, 0.28)
    checks = calibration_checks(
        start_rows=start_raw,
        candidate_rows=candidate_raw,
        pooled_response_checks={"all": True},
        pre_state_sha256="1" * 64,
        post_state_sha256="2" * 64,
    )
    if not all(checks.values()):
        raise AssertionError("valid Teacher-v10.3.1 calibration was rejected")
    pooled_failure = calibration_checks(
        start_rows=start_raw,
        candidate_rows=candidate_raw,
        pooled_response_checks={"v5_fixed_probe_retained_1pct": False},
        pre_state_sha256="1" * 64,
        post_state_sha256="2" * 64,
    )
    if pooled_failure["pooled_response_is_admissible"]:
        raise AssertionError("failed pooled response was accepted")
    broken = _raw(0.42, 0.28)
    normal_name = EXPECTED_INSTANCES["room_0102"][ROLES.index("normal_chair")]
    broken["room_0102"][2][1]["instances"][normal_name]["soft_recall"] = 0.2
    rejected = calibration_checks(
        start_rows=start_raw,
        candidate_rows=broken,
        pooled_response_checks={"all": True},
        pre_state_sha256="1" * 64,
        post_state_sha256="2" * 64,
    )
    if rejected["room_0102_g2_sit_write_v1_normal_chair_recall_retained"]:
        raise AssertionError("one-map Normal Chair collapse was accepted")
    rows = [
        {
            "step": 1,
            "eligible": True,
            "start_pooled": _pooled(0.4, 0.3),
            "candidate_pooled": _pooled(0.42, 0.28),
            "start_v5_dense": [0.06] * 3,
            "candidate_v5_dense": [0.0601] * 3,
        },
        {
            "step": 2,
            "eligible": True,
            "start_pooled": _pooled(0.4, 0.3),
            "candidate_pooled": _pooled(0.43, 0.27),
            "start_v5_dense": [0.06] * 3,
            "candidate_v5_dense": [0.0602] * 3,
        },
    ]
    if rank_shortlist(rows) != [2, 1]:
        raise AssertionError("Teacher-v10.3.1 shortlist ranking changed")

    prepare = Path(__file__).resolve().parent
    runner = (
        prepare / "run_relational_teacher_v1031_onpolicy_calibration6.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(runner)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("Adam", "AdamW", "SGD", "save", "save_checkpoint")
        )
        or (
            isinstance(node.func, ast.Name)
            and node.func.id in ("save_merged_legacy_state", "torch_save")
        )
    ]
    if forbidden:
        raise AssertionError("optimizer/checkpoint entered Teacher-v10.3.1")
    for literal in (
        'value.get("selected_candidate") != "t150_radius_0p004"',
        "_reconstruct_v103_direction(",
        "_apply_direction_on_lora_device(parameters, direction, SELECTED_RADIUS)",
        "device=reference.device,",
        "dtype=reference.dtype,",
        "applied_direction = _apply_direction_on_lora_device(",
        'tensor_sha256(applied_direction) != tensor_sha256(direction)',
        "for step in MONITOR_STEPS:",
        "flattened_task_gradients(scene_losses, parameters)",
        "frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)",
        "all_three_counts(current_presence)",
        '"absolute_all_three_is_diagnostic_only": True',
        '"no_optimizer_created": True',
        '"no_model_checkpoint_saved": True',
        '"room_0201_arrays_unread": True',
    ):
        if literal not in runner:
            raise AssertionError("Teacher-v10.3.1 runner guard changed: " + literal)
    direct_apply_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_flat_direction"
    ]
    if len(direct_apply_calls) != 1:
        raise AssertionError("Teacher-v10.3.1 bypasses checked direction transfer")
    validator = (
        prepare / "validate_relational_teacher_v1031_onpolicy_calibration6.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'SEALED_V3_VALIDATOR_SHA256 = (',
        'hashes["validator"] != SEALED_V3_VALIDATOR_SHA256',
        'validator_path != Path(__file__).resolve()',
        'if name == "validator":',
        "expected_weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)",
        'direction["direction_norm_before_unit"]',
        'direction["directional_derivatives"], dtype=np.float64',
        "analytic_derivatives = gram @ weights / analytic_norm",
        "derivatives = actual_derivatives.tolist()",
    ):
        if literal not in validator:
            raise AssertionError("Teacher-v10.3.1 validator guard changed: " + literal)

    print("[PASS] Teacher-v10.3.1 on-policy calibration CPU/static contract")
    print("[PASS] exact selected response, six updates and disjoint K=3 policy")
    print("[PASS] CPU directions transfer to one LoRA device/dtype before updates")
    print("[PASS] per-map Normal-Chair, v5 and no-checkpoint gates fail closed")


if __name__ == "__main__":
    main()

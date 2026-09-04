#!/usr/bin/env python3
"""Validate immutable Teacher-v10.3.1 calibration delivery."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "teacher_lora_v1031_onpolicy_calibration6_package_v1":
        raise ValueError("Teacher-v10.3.1 package schema changed")
    files = manifest.get("files")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if not isinstance(files, dict) or set(files) != actual:
        raise ValueError("Teacher-v10.3.1 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v10.3.1 package file changed: " + name)
    expected_authorization = {
        "sealed_v103_pass_input": True,
        "exact_t150_radius_0p004_reconstruction": True,
        "fresh_v5r4_rank16_lora": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "fresh_design_and_audit_k3": True,
        "six_common_descent_updates": True,
        "actual_k3_monitor_after_each_update": True,
        "shortlist_in_memory_only": True,
        "optimizer": False,
        "checkpoint": False,
        "room_0201_arrays": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v10.3.1 authorization changed")

    runner = (
        root / "prepare/run_relational_teacher_v1031_onpolicy_calibration6.py"
    ).read_text(encoding="utf-8")
    validator = (
        root / "prepare/validate_relational_teacher_v1031_onpolicy_calibration6.py"
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
    policy_writes = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "create_model_and_diffusion"
    ]
    if (
        forbidden
        or not policy_writes
        or len(model_calls) != 1
        or min(node.lineno for node in policy_writes) >= model_calls[0].lineno
    ):
        raise ValueError("Teacher-v10.3.1 execution inventory changed")
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
        '"shortlisted_states_verified_in_memory": bool(shortlisted_steps)',
        '"absolute_all_three_is_diagnostic_only": True',
        '"no_optimizer_created": True',
        '"no_model_checkpoint_saved": True',
        '"room_0201_arrays_unread": True',
    ):
        if literal not in runner:
            raise ValueError("Teacher-v10.3.1 runner guard changed: " + literal)
    direct_apply_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_flat_direction"
    ]
    if len(direct_apply_calls) != 1:
        raise ValueError("Teacher-v10.3.1 bypasses checked direction transfer")
    for literal in (
        'SEALED_V3_VALIDATOR_SHA256 = (',
        '"9ac11f4a0c9aa5bd7440d18e9a342913509d4fe1d918f928c5ff83b0f96eeeea"',
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
            raise ValueError("Teacher-v10.3.1 validator guard changed: " + literal)
    shell = (root / "run_calibration.sh").read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in shell.splitlines()):
        raise ValueError("Teacher-v10.3.1 launcher contains a continuation backslash")
    for literal in (
        "onpolicy_response_s20261101_v1/preflight.json",
        "onpolicy_calibration6_s20261102_v3",
        "[REUSE] completed v3 summary; validating without CUDA rerun",
        "validate_relational_teacher_v103_onpolicy_response.py",
        "run_relational_teacher_v1031_onpolicy_calibration6.py",
        "validate_relational_teacher_v1031_onpolicy_calibration6.py",
        "this terminal remains open",
    ):
        if literal not in shell:
            raise ValueError("Teacher-v10.3.1 launcher changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v10.3.1 on-policy calibration delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] six common-descent updates and actual K=3 monitor guards")
    print("[PASS] CPU directions transfer to the exact LoRA device and dtype")
    print("[PASS] float32 applied geometry and float64 analytic geometry separated")
    print("[PASS] sealed v3 validator identity authorizes only its replacement")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")


if __name__ == "__main__":
    main()

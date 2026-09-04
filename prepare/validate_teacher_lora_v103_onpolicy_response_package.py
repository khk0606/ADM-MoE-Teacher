#!/usr/bin/env python3
"""Validate immutable Teacher-v10.3 on-policy response delivery."""

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
    if manifest.get("schema") != "teacher_lora_v103_onpolicy_response_package_v1":
        raise ValueError("Teacher-v10.3 package schema changed")
    files = manifest.get("files")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if not isinstance(files, dict) or actual != set(files):
        raise ValueError("Teacher-v10.3 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v10.3 package file changed: " + name)
    expected_authorization = {
        "sealed_v102_failure_input": True,
        "fresh_v5r4_zero_output_lora": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "design_audit_seed_disjoint": True,
        "onpolicy_capture_timesteps": [350, 150, 50],
        "dense_instance_role_tasks": 12,
        "negative_prompt_v5_tasks": 9,
        "resumed_final_map_selection": True,
        "optimizer": False,
        "checkpoint": False,
        "room_0201_arrays": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v10.3 authorization changed")

    runner = (
        root / "prepare/preflight_relational_teacher_v103_onpolicy_response.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(runner)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizer_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr in ("Adam", "AdamW", "SGD")
    ]
    checkpoint_calls = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("save_merged_legacy_state", "torch_save")
        )
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("save", "save_checkpoint")
        )
    ]
    policy_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "create_model_and_diffusion"
    ]
    if (
        optimizer_calls
        or checkpoint_calls
        or not policy_calls
        or len(model_calls) != 1
        or min(node.lineno for node in policy_calls) >= model_calls[0].lineno
    ):
        raise ValueError("Teacher-v10.3 execution inventory changed")
    for required in (
        "dense_instance_all_sittable_objective(",
        "_capture_trajectory(",
        "_resume_trajectory(",
        "flattened_task_gradients(task_losses, parameters)",
        "frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)",
        "apply_flat_direction(parameters, direction, radius)",
        "if tuple(observed_order) != TASK_ORDER:",
        '"candidate_decision_uses_resumed_final_maps": True',
        '"absolute_all_three_is_diagnostic_only": True',
        '"no_optimizer_created": True',
        '"no_model_checkpoint_saved": True',
        '"room_0201_arrays_unread": True',
    ):
        if required not in runner:
            raise ValueError("Teacher-v10.3 runner guard changed: " + required)

    shell = (root / "run_preflight.sh").read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in shell.splitlines()):
        raise ValueError("Teacher-v10.3 launcher contains a continuation backslash")
    for required in (
        "onpolicy_response_s20261101_v1",
        "validate_relational_teacher_v102_dense_instance_supervision.py",
        "preflight_relational_teacher_v103_onpolicy_response.py",
        "validate_relational_teacher_v103_onpolicy_response.py",
        "this terminal remains open",
    ):
        if required not in shell:
            raise ValueError("Teacher-v10.3 launcher changed: " + required)

    print("[PACKAGE_PASS] Teacher-v10.3 on-policy response delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] 21-task common descent and resumed-final selection guards")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")


if __name__ == "__main__":
    main()

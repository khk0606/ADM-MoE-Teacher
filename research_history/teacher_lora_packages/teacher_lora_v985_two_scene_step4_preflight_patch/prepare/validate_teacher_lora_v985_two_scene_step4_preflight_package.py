#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.5 two-scene preflight package."""

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
    if manifest.get("schema") != "teacher_lora_v985_two_scene_step4_preflight_package_v1":
        raise ValueError("Teacher-v9.8.5 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.5 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.5 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.5 package file changed: " + name)

    expected_authorization = {
        "sealed_v984_pass_input": True,
        "fresh_v5r4_exact_step4_reconstruction": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "room0102_actual_k3_two_prompt_audit": True,
        "new_parameter_update": False,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.5 authorization policy changed")

    runbook = (
        root / "TEACHER_V985_TWO_SCENE_STEP4_PREFLIGHT_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v985_two_scene_step4_preflight"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/preflight_relational_teacher_v985_two_scene_step4.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
    ]
    policy_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "create_model_and_diffusion"
    ]
    if len(loads) != 2:
        raise ValueError("two-scene runner array-load count changed")
    if (
        len(policy_calls) < 1
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("two-scene policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("two-scene runner contains optimizer/checkpoint code")
    for literal in (
        "for step in RECONSTRUCTION_STEPS:",
        "_direction_exact(direction_row, v984[\"direction_rows\"][step - 1], step)",
        "candidate_v5_predictions\"][SELECTED_V984_STEP - 1]",
        "two_scene_preflight_checks(",
        "room0102_step4_response_is_admissible",
        "authorizes_cross_scene_direction_diagnosis",
        'f"[AUDIT-CANDIDATE] generation={generation} prompt={PROMPT_IDS[prompt_index]}"',
    ):
        if literal not in source:
            raise ValueError("two-scene runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v985_two_scene_step4_preflight_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'AUDIT_SCENE = "room_0102"',
        "SELECTED_V984_STEP = 4",
        "RECONSTRUCTION_STEPS = (1, 2, 3, 4)",
        "SELECTED_TIMESTEP = 50",
        "STEP_RADIUS = 0.003",
        '"each_role_pooled_recall_strictly_improves": True',
        '"each_role_pooled_mae_strictly_improves": True',
        '"topk_role": "diagnostic_only"',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("sealed two-scene contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v985_two_scene_step4_preflight.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "_close(row, v984[\"direction_rows\"][index]",
        "two_scene_preflight_checks(",
        "absolute_presence_checks(",
        "report_file.parent.glob(\"*.pth\")",
        "candidate_v5_predictions\"][SELECTED_V984_STEP - 1]",
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.5 two-scene step-4 preflight delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, exact reconstruction and room_0102 recomputation guards")


if __name__ == "__main__":
    main()

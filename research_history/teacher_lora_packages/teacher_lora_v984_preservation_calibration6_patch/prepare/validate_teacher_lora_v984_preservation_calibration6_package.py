#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.4 preservation calibration package."""

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
    if manifest.get("schema") != "teacher_lora_v984_preservation_calibration6_package_v1":
        raise ValueError("Teacher-v9.8.4 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.4 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.4 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.4 package file changed: " + name)

    expected_authorization = {
        "sealed_v983_pass_input": True,
        "fresh_v5r4_updates": 6,
        "exact_update1_update2_reproduction": True,
        "actual_k3_two_prompt_monitor_after_each_update": True,
        "eleven_task_preservation_from_update2": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.4 authorization policy changed")

    runbook = (
        root / "TEACHER_V984_PRESERVATION_CALIBRATION6_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v984_preservation_calibration6"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/run_relational_teacher_v984_preservation_calibration6.py"
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
        raise ValueError("calibration runner scene-array load count changed")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("calibration runner contains optimizer/checkpoint code")
    if source.index("atomic_write_json(policy_file") > source.index(
        "create_model_and_diffusion"
    ):
        raise ValueError("calibration policy is not locked before model creation")
    for literal in (
        "for step in MONITOR_STEPS:",
        "calibration_design_seeds(step)",
        "torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)",
        "frank_wolfe_min_norm_weights(gram, iterations)",
        "apply_flat_direction(parameters, direction, STEP_RADIUS)",
        "response6_checks(",
        "calibration_checks(",
        "rank_eligible_steps(monitor_rows)",
        "update-1 K=3 maps do not reproduce v9.8.1",
        "update-2 K=3 maps do not reproduce selected v9.8.3",
        'f"[CALIBRATION] step={step}/6 eligible={eligible} "',
    ):
        if literal not in source:
            raise ValueError("calibration runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v984_preservation_calibration6_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "UPDATE_COUNT = 6",
        "MONITOR_STEPS = (1, 2, 3, 4, 5, 6)",
        "STEP_RADIUS = 0.003",
        "MODEL_SEED = 20261016",
        "CALIBRATION_TAG = 20261020",
        '"updates_3_through_6": "fresh_eleven_task_directions_on_disjoint_v982_design_states"',
        '"topk_role": "diagnostic_only"',
        '"each_generation_negative_mean_addition_cap": 0.00025',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("sealed calibration contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v984_preservation_calibration6.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "expected_derivatives = (gram @ weights) / expected_norm",
        "arrays[\"candidates_normalized\"][0]",
        "arrays[\"candidates_normalized\"][1]",
        "calibration_checks(",
        "rank_eligible_steps(recomputed_rows)",
        "summary_file.parent.glob(\"*.pth\")",
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.4 preservation calibration-6 delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, exact update-1/update-2 and deep recomputation guards")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.2 calibration-6 package."""

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
    if manifest.get("schema") != "teacher_lora_v982_rollout_state_calibration6_package_v1":
        raise ValueError("Teacher-v9.8.2 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.2 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.2 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.2 package file changed: " + name)
    expected_authorization = {
        "sealed_v981_pass_input": True,
        "fresh_v5r4_room0101_calibration_updates": 6,
        "actual_k3_monitor_after_each_update": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.2 authorization policy changed")

    runbook = (root / "TEACHER_V982_ROLLOUT_STATE_CALIBRATION6_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if "--device cuda:0 --no-progress" not in runbook or "cuda:0--no-progress" in runbook:
        raise ValueError("runbook CUDA arguments are malformed")

    runner = root / "prepare/run_relational_teacher_v982_rollout_state_calibration6.py"
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
        "for step in range(1, UPDATE_COUNT + 1)",
        "calibration_design_seeds(step)",
        "apply_flat_direction(parameters, direction, STEP_RADIUS)",
        "response6_checks(",
        "rank_eligible_steps(monitor_rows)",
        "update-1 K=3 response does not reproduce v9.8.1",
    ):
        if literal not in source:
            raise ValueError("calibration runner guard changed: " + literal)
    contract = (
        root / "prepare/relational_teacher_v982_rollout_state_calibration6_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "UPDATE_COUNT = 6",
        "MONITOR_STEPS = (1, 2, 3, 4, 5, 6)",
        "SELECTED_TIMESTEP = 50",
        "STEP_RADIUS = 0.003",
        "MODEL_SEED = 20261016",
        '"post_first_update_required": True',
        '"topk_role": "diagnostic_only"',
        '"checkpoint_policy": "no_model_state_is_serialized"',
    ):
        if literal not in contract:
            raise ValueError("sealed calibration contract literal changed: " + literal)
    print("[PACKAGE_PASS] Teacher-v9.8.2 rollout-state calibration-6 delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] six fresh updates, actual K=3 monitors and no-checkpoint guards")


if __name__ == "__main__":
    main()

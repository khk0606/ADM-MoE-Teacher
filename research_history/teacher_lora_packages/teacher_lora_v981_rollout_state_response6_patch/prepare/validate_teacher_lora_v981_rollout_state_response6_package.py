#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.1 response-6 package."""

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
    if manifest.get("schema") != "teacher_lora_v981_rollout_state_response6_package_v1":
        raise ValueError("Teacher-v9.8.1 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.1 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.1 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.1 package file changed: " + name)
    expected_authorization = {
        "sealed_v98_selected_response_input": True,
        "room_0101_selected_t50_k3_response": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.1 authorization policy changed")

    runbook = (root / "TEACHER_V981_ROLLOUT_STATE_RESPONSE6_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive shell")

    runner = root / "prepare/evaluate_relational_teacher_v981_rollout_state_response6.py"
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
        raise ValueError("runner scene-array load count changed")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("runner contains optimizer/checkpoint code")
    if source.index("atomic_write_json(policy_file") > source.index("install_lora(model"):
        raise ValueError("response-6 policy is not locked before LoRA installation")
    contract = (
        root / "prepare/relational_teacher_v981_rollout_state_response6_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'SELECTED_NAME = "t50_radius_0p003"',
        "SELECTED_TIMESTEP = 50",
        "SELECTED_RADIUS = 0.003",
        "SEED = 20261016",
        '"topk_role": "diagnostic_only"',
        '"pass_authority": "fresh_rollout_state_calibration6_only"',
    ):
        if literal not in contract:
            raise ValueError("sealed response-6 contract literal changed: " + literal)
    print("[PACKAGE_PASS] Teacher-v9.8.1 selected rollout-state response-6 delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] no shell continuation, optimizer, checkpoint writer or held-out load")


if __name__ == "__main__":
    main()

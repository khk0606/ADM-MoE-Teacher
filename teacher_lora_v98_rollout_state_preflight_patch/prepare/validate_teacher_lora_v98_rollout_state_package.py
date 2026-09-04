#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8 rollout-state delivery package."""

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
    if manifest.get("schema") != "teacher_lora_v98_rollout_state_preflight_package_v1":
        raise ValueError("Teacher-v9.8 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8 package file changed: " + name)
    expected_authorization = {
        "sealed_v97_failure_input": True,
        "fresh_v5r4_rollout_state_response": True,
        "room_0101_arrays": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8 authorization policy changed")

    runbook = (root / "TEACHER_V98_ROLLOUT_STATE_PREFLIGHT_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive shell")

    runner = root / "prepare/preflight_relational_teacher_v98_rollout_state_response.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    loads = [node for node in calls if node.func.id == "load_train_scene_bundle"]
    if len(loads) != 1:
        raise ValueError("runner scene-array load count changed")
    if any(node.func.id in {"Adam", "AdamW", "SGD"} for node in calls) or "torch.optim" in source:
        raise ValueError("runner unexpectedly creates an optimizer")
    if "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("runner contains a model-state writer")
    if source.index("atomic_write_json(policy_file") > source.index("install_lora(model"):
        raise ValueError("rollout-state policy is not locked before LoRA installation")
    for literal in (
        "CAPTURE_TIMESTEPS = (400, 200, 50)",
        "STEP_RADII = (0.001, 0.003, 0.01)",
        '"decision_unit": "resumed_final_500_step_reverse_diffusion_map"',
        '"topk_role": "diagnostic_only"',
    ):
        contract = (
            root / "prepare/relational_teacher_v98_rollout_state_contract.py"
        ).read_text(encoding="utf-8")
        if literal not in contract:
            raise ValueError("sealed v9.8 contract literal changed: " + literal)
    print("[PACKAGE_PASS] Teacher-v9.8 rollout-state response delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] no shell continuation, optimizer, checkpoint writer or held-out load")


if __name__ == "__main__":
    main()

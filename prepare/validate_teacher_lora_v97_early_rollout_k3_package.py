#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.7 delivery package."""

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
    if manifest.get("schema") != "teacher_lora_v97_early_rollout_k3_package_v1":
        raise ValueError("Teacher-v9.7 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.7 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.7 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.7 package file changed: " + name)
    expected_authorization = {
        "fresh_v91_first_12_reproduction": True,
        "in_memory_step_3_6_12_snapshots": True,
        "room_0101_final_500_step_k3": True,
        "checkpoint_write": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.7 authorization policy changed")

    runbook = (root / "TEACHER_V97_EARLY_ROLLOUT_K3_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the user's interactive shell")

    runner = root / "prepare/evaluate_relational_teacher_v97_early_rollout_k3.py"
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
    if "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("runner contains a model-state writer")
    if source.index("atomic_write_json(rollout_policy_file") > source.index(
        "optimizer = torch.optim.AdamW"
    ):
        raise ValueError("rollout policy is not locked before the optimizer")
    for literal in (
        "SNAPSHOT_STEPS = (3, 6, 12)",
        "GENERATION_COUNT = 3",
        '"topk_role": "diagnostic_only_not_a_gate_due_discrete_one_point_quantization"',
    ):
        contract = (
            root / "prepare/relational_teacher_v97_early_rollout_k3_contract.py"
        ).read_text(encoding="utf-8")
        if literal not in contract:
            raise ValueError("sealed v9.7 contract literal changed: " + literal)
    print("[PACKAGE_PASS] Teacher-v9.7 early actual-rollout K=3 delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] no shell continuation, forced exit, checkpoint writer or held-out load")


if __name__ == "__main__":
    main()

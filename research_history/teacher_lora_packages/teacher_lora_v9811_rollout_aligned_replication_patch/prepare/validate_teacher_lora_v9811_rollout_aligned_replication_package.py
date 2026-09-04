#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.11 replication package."""

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
    if manifest.get("schema") != "teacher_lora_v9811_rollout_aligned_replication_package_v1":
        raise ValueError("Teacher-v9.8.11 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.11 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.11 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.11 package file changed: " + name)

    expected_authorization = {
        "sealed_v9810_pass_selected_radius_0p006_input": True,
        "sealed_v988_pass_input": True,
        "fresh_v5r4_exact_step4_reconstruction": True,
        "selected_direction_exact_recomputation": True,
        "disjoint_replication_seed_table": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "actual_two_scene_k3_new_seed_replication": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.11 authorization policy changed")

    runbook = (
        root / "TEACHER_V9811_ROLLOUT_ALIGNED_REPLICATION_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v9811_rollout_aligned_replication"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/evaluate_relational_teacher_v9811_rollout_aligned_replication.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
    ]
    applies = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_flat_direction"
    ]
    policy_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    model_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "create_model_and_diffusion"
    ]
    if len(loads) != 2 or len(applies) != 2:
        raise ValueError("two-scene/reconstruction-plus-selected-state contract changed")
    if (
        not policy_calls
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("replication policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("replication runner contains optimizer/checkpoint code")
    for literal in (
        "v9810 = _validate_v9810(v9810_file)",
        "for step in RECONSTRUCTION_STEPS:",
        "selection_cache",
        "replication_cache",
        "replication_rollout_seeds(generation)",
        "seed_table_disjoint",
        "selected_state_exact",
        "selected_v5_exact",
        "two_scene_preflight_checks(",
        "replication_checks(",
        '"new_seed_response_is_admissible": response_eligible',
        '"authorizes_fresh_two_scene_rollout_aligned_multiupdate_calibration"',
    ):
        if literal not in source:
            raise ValueError("replication runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v9811_rollout_aligned_replication_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'SELECTED_DIRECTION = "audit_bed_guard_0p10"',
        "SELECTED_RADIUS = 0.006",
        '"seed_exclusion": "disjoint_from_v97_v988_v989_v9810_selection_seed_table"',
        '"scene_gate": "exact_v985_strict_three_role_response_policy_applied_independently"',
        '"topk_role": "diagnostic_only"',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("replication contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v9811_rollout_aligned_replication.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "two_scene_preflight_checks(",
        "absolute_presence_checks(",
        "replication_checks(",
        "load_train_scene_bundle(",
        'summary_file.parent.glob("*.pth")',
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.11 disjoint-seed replication delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, actual K=3, new seeds and no-checkpoint guards")


if __name__ == "__main__":
    main()

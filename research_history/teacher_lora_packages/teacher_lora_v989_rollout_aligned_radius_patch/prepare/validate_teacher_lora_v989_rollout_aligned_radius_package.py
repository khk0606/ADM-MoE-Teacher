#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.9 radius-response package."""

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
    if manifest.get("schema") != "teacher_lora_v989_rollout_aligned_radius_package_v1":
        raise ValueError("Teacher-v9.8.9 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.9 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.9 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.9 package file changed: " + name)

    expected_authorization = {
        "sealed_v988_pass_input": True,
        "fresh_v5r4_exact_step4_reconstruction": True,
        "selected_direction_exact_recomputation": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "actual_two_scene_k3_radius_grid": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.9 authorization policy changed")

    runbook = (
        root / "TEACHER_V989_ROLLOUT_ALIGNED_RADIUS_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v989_rollout_aligned_radius"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/evaluate_relational_teacher_v989_rollout_aligned_radius.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
    ]
    applies = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_flat_direction"
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
    if len(loads) != 2 or len(applies) != 2:
        raise ValueError("two-scene/reconstruction-plus-radius contract changed")
    if (
        len(policy_calls) < 1
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("response policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("response runner contains optimizer/checkpoint code")
    for literal in (
        'value.get("selected_candidate") != SELECTED_DIRECTION',
        "for step in RECONSTRUCTION_STEPS:",
        "for radius_index, radius in enumerate(RADIUS_GRID):",
        "_restore_state(named_lora, step4_state)",
        "distinct radii produced duplicate LoRA states",
        'selected_direction_sha256 != sealed_selected["direction_sha256"]',
        "two_scene_preflight_checks(",
        "radius_response_checks(",
        '"at_least_one_radius_is_admissible": selected_radius is not None',
        '"authorizes_fresh_two_scene_rollout_aligned_calibration": status == "PASS"',
    ):
        if literal not in source:
            raise ValueError("rollout-aligned radius runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v989_rollout_aligned_radius_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'SELECTED_DIRECTION = "audit_bed_guard_0p10"',
        "RADIUS_GRID = (0.00025, 0.0005, 0.001, 0.002, 0.003)",
        '"scene_gate": "exact_v985_strict_three_role_response_policy_applied_independently"',
        '"topk_role": "diagnostic_only"',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("rollout-aligned radius contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v989_rollout_aligned_radius.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "two_scene_preflight_checks(",
        "absolute_presence_checks(",
        "rank_eligible_radii(recomputed_rows)",
        "load_train_scene_bundle(",
        'summary_file.parent.glob("*.pth")',
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.9 rollout-aligned radius delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, exact direction, actual K=3 and no-checkpoint guards")


if __name__ == "__main__":
    main()

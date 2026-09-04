#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.8 rollout-aligned package."""

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
    if manifest.get("schema") != "teacher_lora_v988_rollout_aligned_direction_package_v1":
        raise ValueError("Teacher-v9.8.8 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.8 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.8 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.8 package file changed: " + name)

    expected_authorization = {
        "sealed_v987_failure_input": True,
        "fresh_v5r4_exact_step4_reconstruction": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "actual_k3_t50_states": True,
        "rollout_aligned_51_task_gradients": True,
        "candidate_parameter_update": False,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.8 authorization policy changed")

    runbook = (
        root / "TEACHER_V988_ROLLOUT_ALIGNED_DIRECTION_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v988_rollout_aligned_direction"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook status handling is malformed")

    runner = root / "prepare/preflight_relational_teacher_v988_rollout_aligned_direction.py"
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
    if len(loads) != 2 or len(applies) != 1:
        raise ValueError("two-scene/reconstruction-only mutation contract changed")
    if (
        not policy_calls
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("diagnosis policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("diagnosis runner contains optimizer/checkpoint code")
    for literal in (
        'value.get("status") != "FAIL"',
        'row.get("scene_failed_checks", {}).get(SOURCE_SCENE) != []',
        'row.get("scene_failed_checks", {}).get(AUDIT_SCENE) != expected_audit',
        'np.array_equal(base_normalized, v987_arrays["base_normalized"])',
        'for scene_index, scene in enumerate((SOURCE_SCENE, AUDIT_SCENE)):',
        'for generation in range(3):',
        'flattened_task_gradients(tasks, parameters)',
        'if tuple(actual_task_order) != TASK_ORDER:',
        'if _state_sha256(named_lora) != selected_state_sha256:',
        '"no_candidate_parameter_update_applied": True',
        '"authorizes_actual_two_scene_k3_rollout_aligned_radius_grid"',
    ):
        if literal not in source:
            raise ValueError("rollout-aligned runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v988_rollout_aligned_direction_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        '"total": 51',
        'AUDIT_BED_GUARD_MIXES = (0.05, 0.10, 0.20, 0.30, 0.40)',
        '"diagnostic_only": True',
        '"no_candidate_parameter_update": True',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("rollout-aligned contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v988_rollout_aligned_direction.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "diagnose_directions(geometry[\"gram\"])",
        "rank_eligible_directions(recomputed_rows)",
        "load_train_scene_bundle(",
        "report_file.parent.glob(\"*.pth\")",
        'value.get("conflict_pair_total") != 1275',
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.8 rollout-aligned direction delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, exact K=3 states and no-checkpoint guards")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.12 delivery package."""

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
    if manifest.get("schema") != "teacher_lora_v9812_two_scene_multiupdate_calibration_package_v1":
        raise ValueError("Teacher-v9.8.12 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.12 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.12 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.12 package file changed: " + name)

    expected_authorization = {
        "sealed_v9811_pass_selected_radius_0p006_input": True,
        "exact_selected_state_reconstruction": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "two_scene_51_task_updates": True,
        "six_short_updates": True,
        "actual_k3_after_every_update": True,
        "all_three_hard_shortlist_gate": True,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.12 authorization policy changed")

    runbook = (
        root / "TEACHER_V9812_TWO_SCENE_MULTIUPDATE_CALIBRATION_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v9812_two_scene_multiupdate_calibration"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/run_relational_teacher_v9812_two_scene_multiupdate_calibration.py"
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
    if len(loads) != 2 or len(applies) != 3 or len(model_calls) != 1:
        raise ValueError("two-scene/reconstruction/update call inventory changed")
    if (
        not policy_calls
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("calibration policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("calibration runner contains optimizer/checkpoint code")
    for literal in (
        "v9811 = _validate_v9811(v9811_file)",
        "for step in RECONSTRUCTION_STEPS:",
        "for step in MONITOR_STEPS:",
        "design_cache",
        "audit_cache",
        "direction_candidates(current_gram, V988_TASK_ORDER)",
        "two_scene_preflight_checks(",
        "all_three_counts(scene_presence)",
        "rank_shortlist(monitor_rows)",
        '"at_least_one_all_three_state_is_shortlisted": bool(shortlisted_steps)',
        '"authorizes_shortlisted_state_checkpoint_export_gate"',
    ):
        if literal not in source:
            raise ValueError("calibration runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v9812_two_scene_multiupdate_calibration_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'SELECTED_DIRECTION = "audit_bed_guard_0p10"',
        "SELECTED_RADIUS = 0.006",
        "UPDATE_COUNT = 6",
        "STEP_RADIUS = 0.001",
        '"absolute_three_object_gate": "each_scene_and_prompt_at_least_two_of_three_generations"',
        '"topk_role": "diagnostic_only"',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("calibration contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v9812_two_scene_multiupdate_calibration.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "direction_candidates(gram, V988_TASK_ORDER)",
        "two_scene_preflight_checks(",
        "absolute_presence_checks(",
        "calibration_checks(",
        "rank_shortlist(recomputed_monitors)",
        "load_train_scene_bundle(",
        'summary_file.parent.glob("*.pth")',
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.12 two-scene multi-update delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, per-update K=3, all-three and no-checkpoint guards")


if __name__ == "__main__":
    main()

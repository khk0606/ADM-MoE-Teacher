#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.6 cross-scene diagnosis package."""

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
    if manifest.get("schema") != "teacher_lora_v986_cross_scene_direction_package_v2":
        raise ValueError("Teacher-v9.8.6 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.6 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.6 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.6 package file changed: " + name)

    expected_authorization = {
        "sealed_v985_fail_input": True,
        "fresh_v5r4_exact_step4_reconstruction": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "nineteen_task_gradient_geometry": True,
        "candidate_parameter_update": False,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.6 authorization policy changed")

    runbook = (root / "TEACHER_V986_CROSS_SCENE_DIRECTION_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count(
        "run_v986_cross_scene_direction"
    ) != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "RUN_RC=$?" not in runbook or "this terminal remains open" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/preflight_relational_teacher_v986_cross_scene_direction.py"
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
        raise ValueError("two-scene/reconstruction-only static contract changed")
    if (
        len(policy_calls) < 1
        or len(model_calls) != 1
        or policy_calls[0].lineno >= min(node.lineno for node in loads)
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("diagnosis policy is not locked before arrays/model")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("diagnosis contains optimizer/checkpoint code")
    for literal in (
        'value.get("audit_failed_checks") != expected_audit_failures',
        "for step in RECONSTRUCTION_STEPS:",
        "apply_flat_direction(parameters, direction, STEP_RADIUS)",
        "source_gradients, source_losses = _scene_task_gradients(",
        "audit_gradients, audit_losses = _scene_task_gradients(",
        "v5_gradients = flattened_task_gradients(v5_tasks, parameters)",
        "gradients.shape[0] != len(TASK_ORDER)",
        "gradient64 = gradients.double()",
        "raw_gram_max_asymmetry = float(np.abs(raw_gram - raw_gram.T).max())",
        "gram = 0.5 * (raw_gram + raw_gram.T)",
        '"no_candidate_parameter_update_applied": True',
        '"authorizes_actual_two_scene_k3_direction_response_grid": status == "PASS"',
    ):
        if literal not in source:
            raise ValueError("cross-scene runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v986_cross_scene_direction_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'SOURCE_SCENE = "room_0101"',
        'AUDIT_SCENE = "room_0102"',
        "SELECTED_V984_STEP = 4",
        "SELECTED_TIMESTEP = 50",
        "MINIMUM_DIRECTIONAL_DERIVATIVE = 1e-8",
        "RAW_GRAM_ASYMMETRY_CAP = 1e-8",
        '"audit_bed_watch"',
        '"audit_bed_write"',
        '"no_candidate_parameter_update": True',
        '"pass_authority": "actual_two_scene_k3_direction_response_grid_only"',
    ):
        if literal not in contract:
            raise ValueError("cross-scene contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v986_cross_scene_direction.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "diagnose_directions(arrays[\"gram\"])",
        "conflict_pairs(arrays[\"gram\"])",
        "rank_eligible_directions(recomputed_rows)",
        "load_train_scene_bundle(",
        "report_file.parent.glob(\"*.pth\")",
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.6 cross-scene direction delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, 19-task geometry and no-update guards")


if __name__ == "__main__":
    main()

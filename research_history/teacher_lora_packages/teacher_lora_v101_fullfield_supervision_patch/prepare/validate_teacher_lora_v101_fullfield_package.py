#!/usr/bin/env python3
"""Validate immutable Teacher-v10.1 full-field delivery."""

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
    if manifest.get("schema") != "teacher_lora_v101_fullfield_package_v1":
        raise ValueError("Teacher-v10.1 package schema changed")
    files = manifest.get("files")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if not isinstance(files, dict) or actual != set(files):
        raise ValueError("Teacher-v10.1 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v10.1 package file changed: " + name)
    expected_authorization = {
        "sealed_v10_failure_input": True,
        "fresh_v5r4_zero_output_lora": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "every_non_unknown_point_supervised": True,
        "unknown_sittable_points_ignored": True,
        "hard_false_positive_penalty": True,
        "v5_replay_from_step_one": True,
        "actual_two_scene_k3_selection": True,
        "checkpoint_only_after_actual_pass": True,
        "room_0201_arrays": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v10.1 authorization changed")

    runner = root / "prepare/run_relational_teacher_v101_fullfield_supervision.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizers = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW"
    ]
    checkpoints = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "save_merged_legacy_state"
    ]
    loads = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    models = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "create_model_and_diffusion"
    ]
    policies = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "atomic_write_json"
    ]
    if (
        len(optimizers) != 1
        or len(checkpoints) != 1
        or len(loads) != 1
        or len(models) != 1
        or not policies
        or policies[0].lineno >= loads[0].lineno
        or policies[0].lineno >= models[0].lineno
    ):
        raise ValueError("Teacher-v10.1 execution inventory changed")
    for required in (
        "fullfield_all_sittable_objective(",
        "for step in range(1, TRAIN_STEPS + 1):",
        "for scene_index, scene in enumerate(SCENES):",
        "if step < REPLAY_START_STEP:",
        "shortlisted_steps = shortlist_monitor_steps(monitor_rows)",
        "selected_step = select_rollout_candidate(rollout_rows)",
        "if selected_step is not None:",
        "save_merged_legacy_state(model, checkpoint_file)",
        '"every_non_unknown_point_receives_gt_supervision": True',
        '"hard_false_positive_background_is_supervised": True',
        '"room_0201_arrays_unread": True',
    ):
        if required not in source:
            raise ValueError("Teacher-v10.1 runner guard changed: " + required)
    if "apply_flat_direction" in source:
        raise ValueError("manual calibration entered Teacher-v10.1")

    objective = (
        root / "prepare/relational_teacher_v101_fullfield_objective.py"
    ).read_text(encoding="utf-8")
    ast.parse(objective)
    for required in (
        "known = ~unknown",
        "full_field_known = _point_mse(physical, target, known)",
        "safe_background = known &",
        "torch.topk(values.detach()",
        '"worst_instance_active"',
    ):
        if required not in objective:
            raise ValueError("Teacher-v10.1 objective guard changed: " + required)

    shell = (root / "run_training.sh").read_text(encoding="utf-8")
    viewer_shell = (root / "run_viewer.sh").read_text(encoding="utf-8")
    if any(
        line.rstrip().endswith("\\")
        for text in (shell, viewer_shell)
        for line in text.splitlines()
    ):
        raise ValueError("Teacher-v10.1 launcher contains a continuation backslash")
    for required in (
        "fullfield_supervision_s20261030_v1",
        "run_relational_teacher_v101_fullfield_supervision.py",
        "validate_relational_teacher_v101_fullfield_supervision.py",
        "this terminal remains open",
    ):
        if required not in shell:
            raise ValueError("Teacher-v10.1 training launcher changed: " + required)
    for required in (
        "visualize_relational_teacher_v101_fullfield_viser.py",
        "--host 0.0.0.0 --port 8080",
        "this terminal remains open",
    ):
        if required not in viewer_shell:
            raise ValueError("Teacher-v10.1 viewer launcher changed: " + required)

    print("[PACKAGE_PASS] Teacher-v10.1 full-field supervision delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] full-field/background/v5 replay and actual K=3 guards verified")
    print("[PASS] checkpoint-if-pass and read-only viewer launch guards verified")


if __name__ == "__main__":
    main()

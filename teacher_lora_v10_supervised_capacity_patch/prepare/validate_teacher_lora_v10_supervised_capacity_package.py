#!/usr/bin/env python3
"""Validate the immutable Teacher-v10 supervised-capacity delivery."""

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
    if manifest.get("schema") != "teacher_lora_v10_supervised_capacity_package_v1":
        raise ValueError("Teacher-v10 package schema changed")
    files = manifest.get("files")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if not isinstance(files, dict) or actual != set(files):
        raise ValueError("Teacher-v10 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v10 package file changed: " + name)

    expected_authorization = {
        "sealed_v9812_failure_input": True,
        "fresh_v5r4_zero_output_lora": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "direct_x0_q_sample_supervision": True,
        "full_timestep_curriculum": True,
        "adamw_optimizer": True,
        "actual_two_scene_k3_selection": True,
        "checkpoint_only_after_actual_pass": True,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v10 authorization changed")

    runner = root / "prepare/run_relational_teacher_v10_supervised_capacity.py"
    source = runner.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    optimizer_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW"
    ]
    checkpoint_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "save_merged_legacy_state"
    ]
    load_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "load_train_scene_bundle"
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
    if (
        len(optimizer_calls) != 1
        or len(checkpoint_calls) != 1
        or len(load_calls) != 1
        or len(model_calls) != 1
    ):
        raise ValueError("Teacher-v10 optimizer/checkpoint/data/model inventory changed")
    if (
        not policy_calls
        or policy_calls[0].lineno >= load_calls[0].lineno
        or policy_calls[0].lineno >= model_calls[0].lineno
    ):
        raise ValueError("Teacher-v10 policy is not locked before arrays/model")
    for required in (
        "direct_all_sittable_objective(",
        "prediction = predict_xstart(",
        "for step in range(1, args.steps + 1):",
        "for scene_index, scene in enumerate(SCENES):",
        "if step >= REPLAY_START_STEP:",
        "torch.optim.AdamW(",
        "shortlist_monitor_steps(monitor_rows)",
        "select_rollout_candidate(rollout_rows)",
        "if selected_step is not None:",
        "save_merged_legacy_state(model, checkpoint_file)",
        '"checkpoint_written_iff_actual_gate_passes"',
        '"room_0201_arrays_unread": True',
    ):
        if required not in source:
            raise ValueError("Teacher-v10 runner guard changed: " + required)
    if "apply_flat_direction" in source:
        raise ValueError("old manual calibration entered Teacher-v10")

    shell = (root / "run_training.sh").read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in shell.splitlines()):
        raise ValueError("Teacher-v10 launcher contains a continuation backslash")
    for required in (
        "--steps 1200",
        "--lora-rank 16",
        "--lora-alpha 16",
        "--device cuda:0 --no-progress",
        "this terminal remains open",
    ):
        if required not in shell:
            raise ValueError("Teacher-v10 launcher guard changed: " + required)

    viewer = (
        root / "prepare/visualize_relational_teacher_v10_supervised_capacity_viser.py"
    ).read_text(encoding="utf-8")
    viewer_tree = ast.parse(viewer)
    forbidden_viewer_calls = {
        "create_model_and_diffusion",
        "load_ckpt",
        "save_merged_legacy_state",
    }
    for node in ast.walk(viewer_tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_viewer_calls:
                raise ValueError("Teacher-v10 viewer gained a model/checkpoint call")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "Adam",
                "AdamW",
                "backward",
            }:
                raise ValueError("Teacher-v10 viewer gained an optimizer call")
    for required in (
        'arrays["base_rollouts"]',
        'arrays["candidate_rollouts"]',
        'arrays[prefix + "_all_sittable_gt"]',
        '"Candidate − Base"',
        '"|Candidate − GT|"',
    ):
        if required not in viewer:
            raise ValueError("Teacher-v10 viewer guard changed: " + required)

    viewer_shell = (root / "run_viewer.sh").read_text(encoding="utf-8")
    if any(line.rstrip().endswith("\\") for line in viewer_shell.splitlines()):
        raise ValueError("Teacher-v10 viewer launcher contains a continuation backslash")
    for required in (
        "validate_relational_teacher_v10_supervised_capacity.py",
        "visualize_relational_teacher_v10_supervised_capacity_viser.py",
        "--host 0.0.0.0 --port 8080",
        "this terminal remains open",
    ):
        if required not in viewer_shell:
            raise ValueError("Teacher-v10 viewer launcher guard changed: " + required)

    print("[PACKAGE_PASS] Teacher-v10 fresh supervised delivery")
    print("[PASS] {} files and direct-label authorization verified".format(len(files)))
    print("[PASS] AdamW/full-timestep/K=3/checkpoint-if-pass guards verified")
    print("[PASS] read-only GT/Base/candidate Viser audit guards verified")


if __name__ == "__main__":
    main()

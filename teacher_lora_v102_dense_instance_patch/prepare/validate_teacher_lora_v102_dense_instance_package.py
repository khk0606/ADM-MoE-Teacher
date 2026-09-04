#!/usr/bin/env python3
"""Validate immutable Teacher-v10.2 dense-instance delivery."""

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
    if manifest.get("schema") != "teacher_lora_v102_dense_instance_package_v1":
        raise ValueError("Teacher-v10.2 package schema changed")
    files = manifest.get("files")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if not isinstance(files, dict) or actual != set(files):
        raise ValueError("Teacher-v10.2 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v10.2 package file changed: " + name)
    expected_authorization = {
        "sealed_v101_failure_input": True,
        "fresh_v5r4_zero_output_lora": True,
        "room0101_arrays": True,
        "room0102_arrays": True,
        "full_dense_instance_support_equal_macro": True,
        "all_positive_support_excluded_from_hard_negatives": True,
        "unknown_sittable_points_ignored": True,
        "reduced_background_penalty": True,
        "v5_replay_from_step_one": True,
        "actual_two_scene_k3_selection": True,
        "checkpoint_only_after_actual_pass": True,
        "room_0201_arrays": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v10.2 authorization changed")

    objective = (
        root / "prepare/relational_teacher_v102_dense_instance_objective.py"
    ).read_text(encoding="utf-8")
    ast.parse(objective)
    for required in (
        "instance_targets.max(dim=-1).values >= SUPPORT_THRESHOLD",
        "all_instance_support = instance_support_points.any(dim=1)",
        "instance_active_channels = instance_targets >= ACTIVE_THRESHOLD",
        "& ~all_instance_support",
        '"instance_dense_support"',
        '"instance_dense_active"',
        '"worst_instance_dense_active"',
        "torch.topk(values.detach()",
    ):
        if required not in objective:
            raise ValueError("Teacher-v10.2 objective guard changed: " + required)
    for forbidden in (
        "verified_masks =",
        "& point_mask.unsqueeze(-1)",
        "point_mask = verified_masks",
    ):
        if forbidden in objective:
            raise ValueError("v10.1 surface-only mask re-entered v10.2")

    runner = (
        root / "prepare/run_relational_teacher_v102_dense_instance_supervision.py"
    ).read_text(encoding="utf-8")
    ast.parse(runner)
    for required in (
        '"fullfield_all_sittable_objective": dense_instance_all_sittable_objective',
        '"_validate_v10_failure": _validate_v101_failure',
        '"dense_instance_positive_support_is_equal_macro": True',
        '"all_instance_positive_support_excluded_from_hard_background": True',
        '"at_least_one_dense_instance_candidate_passes_actual_k3"',
        'checkpoint_file = output_dir / "teacher_v102_dense_instance.pt"',
        '"checkpoint_written_iff_actual_gate_passes"',
        '"room_0201_arrays_unread": True',
    ):
        if required not in runner:
            raise ValueError("Teacher-v10.2 runner guard changed: " + required)
    if "apply_flat_direction" in runner:
        raise ValueError("manual calibration entered Teacher-v10.2")

    shell = (root / "run_training.sh").read_text(encoding="utf-8")
    viewer_shell = (root / "run_viewer.sh").read_text(encoding="utf-8")
    if any(
        line.rstrip().endswith("\\")
        for text in (shell, viewer_shell)
        for line in text.splitlines()
    ):
        raise ValueError("Teacher-v10.2 launcher contains a continuation backslash")
    for required in (
        "dense_instance_supervision_s20261031_v1",
        "validate_relational_teacher_v101_fullfield_supervision.py",
        "run_relational_teacher_v102_dense_instance_supervision.py",
        "validate_relational_teacher_v102_dense_instance_supervision.py",
        "this terminal remains open",
    ):
        if required not in shell:
            raise ValueError("Teacher-v10.2 training launcher changed: " + required)
    for required in (
        "visualize_relational_teacher_v102_dense_instance_viser.py",
        "--host 0.0.0.0 --port 8080",
        "this terminal remains open",
    ):
        if required not in viewer_shell:
            raise ValueError("Teacher-v10.2 viewer launcher changed: " + required)

    print("[PACKAGE_PASS] Teacher-v10.2 dense-instance supervision delivery")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] full dense support, hard-negative exclusion and v5 guards verified")
    print("[PASS] actual two-scene K=3 and checkpoint-if-pass guards verified")


if __name__ == "__main__":
    main()

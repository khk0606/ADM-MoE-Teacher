#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.3 preservation preflight package."""

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
    if manifest.get("schema") != "teacher_lora_v983_preservation_direction_preflight_package_v2":
        raise ValueError("Teacher-v9.8.3 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.8.3 package inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.8.3 package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.3 package file changed: " + name)
    expected_authorization = {
        "sealed_v982_failure_input": True,
        "fresh_v5r4_exact_v981_update1_reconstruction": True,
        "room0101_eleven_task_direction_grid": True,
        "actual_k3_partial_rollouts_per_candidate": 6,
        "optimizer_creation": False,
        "checkpoint_write": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "development_evaluation": False,
        "long_training": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected_authorization:
        raise ValueError("Teacher-v9.8.3 authorization policy changed")

    runbook = (root / "TEACHER_V983_PRESERVATION_DIRECTION_PREFLIGHT_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    if any(line.rstrip().endswith("\\") for line in runbook.splitlines()):
        raise ValueError("runbook contains a shell continuation backslash")
    if "set -e" in runbook or "set -u" in runbook or "set -o pipefail" in runbook:
        raise ValueError("runbook can terminate the interactive terminal")
    if runbook.count("```bash") != 1 or runbook.count("run_v983_preservation_preflight") != 3:
        raise ValueError("runbook is not one complete copy-paste block")
    if "exit 1" in runbook or "--device cuda:0 --no-progress" not in runbook:
        raise ValueError("runbook terminal/CUDA safety changed")
    if "cuda:0--no-progress" in runbook or "RUN_RC=$?" not in runbook:
        raise ValueError("runbook CUDA/status handling is malformed")

    runner = root / "prepare/preflight_relational_teacher_v983_preservation_direction.py"
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
        raise ValueError("preservation runner scene-array load count changed")
    if "torch.optim" in source or "torch.save" in source or "save_trainable_state" in source:
        raise ValueError("preservation runner contains optimizer/checkpoint code")
    if source.index("atomic_write_json(policy_file") > source.index(
        "create_model_and_diffusion"
    ):
        raise ValueError("preservation policy is not locked before model creation")
    for literal in (
        "failed.get(\"design_seed_table\", {}).get(\"2\")",
        "flattened_task_gradients(all_tasks, parameters)",
        "torch.cat((sit_tasks, v5_tasks, negative_tasks), dim=0)",
        "frank_wolfe_min_norm_weights(gram, 8192)",
        "for radius_index, radius in enumerate(STEP_RADII)",
        "response6_checks(",
        "preservation_checks(",
        "rank_candidates(candidate_rows)",
        "reconstructed update-1 K=3 maps differ from v9.8.1",
        "eligible = all(checks.values())",
        'f"[PRESERVE R={radius:.5f}] eligible={eligible} "',
    ):
        if literal not in source:
            raise ValueError("preservation runner guard changed: " + literal)

    contract = (
        root / "prepare/relational_teacher_v983_preservation_direction_contract.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "MODEL_SEED = 20261016",
        "SELECTED_TIMESTEP = 50",
        "STEP_RADII = (0.00025, 0.0005, 0.001, 0.002, 0.003)",
        '"direction": "minimum_norm_common_descent_of_eleven_normalized_gradients"',
        '"topk_role": "diagnostic_only"',
        '"each_v5_case_not_worse_than_step1": True',
        '"each_generation_negative_mean_not_worse_than_step1_tolerance": 0.00025',
        '"checkpoint_policy": "no_optimizer_and_no_model_state_serialization"',
    ):
        if literal not in contract:
            raise ValueError("sealed preservation contract literal changed: " + literal)

    validator = (
        root / "prepare/validate_relational_teacher_v983_preservation_direction.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "_validate_v982_nested_failure(failed)",
        "expected_derivatives = (gram @ weights) / expected_norm",
        "set(saved) != candidate_keys",
        "preservation_checks(",
        "rank_candidates(recomputed_candidates)",
    ):
        if literal not in validator:
            raise ValueError("deep-validator guard changed: " + literal)

    print("[PACKAGE_PASS] Teacher-v9.8.3 preservation direction delivery v2")
    print("[PASS] {} files and fail-closed authorization verified".format(len(files)))
    print("[PASS] one-block shell, eleven-task geometry and deep recomputation guards")


if __name__ == "__main__":
    main()

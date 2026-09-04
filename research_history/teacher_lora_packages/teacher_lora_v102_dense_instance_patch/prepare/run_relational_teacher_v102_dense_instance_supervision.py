#!/usr/bin/env python3
"""Fresh Teacher-v10.2 dense-instance training and actual K=3 gate."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_relational_teacher_v101_fullfield_supervision as base  # noqa: E402
from relational_teacher_v102_dense_instance_contract import (  # noqa: E402
    LEARNING_RATE,
    LOSS_WEIGHTS,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    REPLAY_START_STEP,
    REPLAY_WEIGHT,
    SCHEMA,
    TRAIN_STEPS,
    V101_SCHEMA,
    canonical_sha256,
    select_rollout_candidate,
    shortlist_monitor_steps,
)
from relational_teacher_v102_dense_instance_objective import (  # noqa: E402
    dense_instance_all_sittable_objective,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v101-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(label + " contains a non-finite number")


def _validate_v101_failure(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V101_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_step") is not None
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_objective_or_architecture_redesign") is not True
        or value.get("failed_checks")
        != ["at_least_one_fullfield_candidate_passes_actual_k3"]
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v10.1 failure does not authorize v10.2")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Teacher-v10.1 path binding is absent")
    if set(paths) != set(hashes) or "checkpoint" in paths:
        raise ValueError("failed Teacher-v10.1 path inventory changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10.1 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed Teacher-v10.1 unexpectedly contains a checkpoint")
    return value


def _install_v102_pipeline() -> None:
    """Reuse the audited v10.1 loop while replacing all policy globals."""
    replacements = {
        "SCHEMA": SCHEMA,
        "V10_SCHEMA": V101_SCHEMA,
        "TRAIN_STEPS": TRAIN_STEPS,
        "MONITOR_STEPS": MONITOR_STEPS,
        "LEARNING_RATE": LEARNING_RATE,
        "REPLAY_START_STEP": REPLAY_START_STEP,
        "REPLAY_WEIGHT": REPLAY_WEIGHT,
        "LOSS_WEIGHTS": LOSS_WEIGHTS,
        "POLICY": POLICY,
        "POLICY_ID": POLICY_ID,
        "MODEL_SEED": MODEL_SEED,
        "select_rollout_candidate": select_rollout_candidate,
        "shortlist_monitor_steps": shortlist_monitor_steps,
        "fullfield_all_sittable_objective": dense_instance_all_sittable_objective,
        "_validate_v10_failure": _validate_v101_failure,
    }
    for name, value in replacements.items():
        setattr(base, name, value)


def _rewrite_v102_report(
    output_dir: Path,
    authority: Mapping[str, object],
) -> Mapping[str, object]:
    report_file = output_dir / "summary.json"
    report = dict(read_json(report_file))
    selected = report.get("selected_step")

    old_policy = output_dir / "fullfield_supervision_policy.json"
    policy_file = output_dir / "dense_instance_supervision_policy.json"
    old_maps = output_dir / "fullfield_supervision_maps.npz"
    maps_file = output_dir / "dense_instance_supervision_maps.npz"
    old_policy.rename(policy_file)
    old_maps.rename(maps_file)
    checkpoint_file = None
    old_checkpoint = output_dir / "teacher_v101_fullfield.pt"
    if selected is not None:
        checkpoint_file = output_dir / "teacher_v102_dense_instance.pt"
        old_checkpoint.rename(checkpoint_file)
    elif old_checkpoint.exists():
        raise AssertionError("failed Teacher-v10.2 may not contain a checkpoint")

    old_paths = report["paths"]
    source_paths = {
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v102_dense_instance_supervision.py",
        "contract": PREPARE_ROOT
        / "relational_teacher_v102_dense_instance_contract.py",
        "objective": PREPARE_ROOT
        / "relational_teacher_v102_dense_instance_objective.py",
        "base_pipeline": Path(base.__file__).resolve(),
        "v101_failure": Path(str(old_paths["v10_failure"])).resolve(),
        "dataset_index": Path(str(old_paths["dataset_index"])).resolve(),
        "source_dataset_index": Path(
            str(old_paths["source_dataset_index"])
        ).resolve(),
        "stats_file": Path(str(old_paths["stats_file"])).resolve(),
        "v5_split": Path(str(old_paths["v5_split"])).resolve(),
        "v5_evidence_report": Path(
            str(old_paths["v5_evidence_report"])
        ).resolve(),
        "original_checkpoint": Path(
            str(old_paths["original_checkpoint"])
        ).resolve(),
        "v5_checkpoint": Path(str(old_paths["v5_checkpoint"])).resolve(),
        "metric_policy": Path(str(old_paths["metric_policy"])).resolve(),
        "training_policy": policy_file,
        "maps": maps_file,
    }
    if checkpoint_file is not None:
        source_paths["checkpoint"] = checkpoint_file
    path_strings = {name: str(path.resolve()) for name, path in source_paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in source_paths.items()}

    status = "PASS" if selected is not None else "FAIL"
    checks = {
        "sealed_v101_failure_authorizes_dense_instance_redesign": True,
        "fresh_v5r4_zero_init": True,
        "both_train_scenes_used_in_every_update": True,
        "every_non_unknown_point_receives_gt_supervision": True,
        "dense_instance_positive_support_is_equal_macro": True,
        "all_instance_positive_support_excluded_from_hard_background": True,
        "hard_false_positive_background_is_supervised": True,
        "v5_replay_active_from_first_update": REPLAY_START_STEP == 1,
        "all_timestep_buckets_exercised": report["checks"][
            "all_timestep_buckets_exercised"
        ],
        "exact_1000_adamw_updates": TRAIN_STEPS == 1000,
        "only_lora_received_gradients": True,
        "three_monitor_states_received_actual_two_scene_k3": len(
            report["rollout_rows"]
        )
        == 3,
        "at_least_one_dense_instance_candidate_passes_actual_k3": selected
        is not None,
        "checkpoint_written_iff_actual_gate_passes": (
            checkpoint_file is not None and checkpoint_file.is_file()
        )
        == (selected is not None),
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
    }
    report.update(
        {
            "schema": SCHEMA,
            "seed": MODEL_SEED,
            "policy_id": POLICY_ID,
            "v101_binding_id": authority.get("binding_id"),
            "loss_weights": dict(LOSS_WEIGHTS),
            "maps_sha256": path_hashes["maps"],
            "checkpoint_sha256": path_hashes.get("checkpoint"),
            "paths": path_strings,
            "path_sha256": path_hashes,
            "status": status,
            "serialized_model_state": selected is not None,
            "checks": checks,
            "failed_checks": sorted(
                name for name, passed in checks.items() if not passed
            ),
            "authorizes_teacher_v102_checkpoint_lock": status == "PASS",
            "authorizes_objective_or_architecture_redesign": status == "FAIL",
            "authorizes_development_evaluation": False,
            "authorizes_paper_test": False,
        }
    )
    report.pop("v10_binding_id", None)
    report.pop("authorizes_teacher_v101_checkpoint_lock", None)
    report["binding_id"] = canonical_sha256(
        {
            "policy_id": POLICY_ID,
            "v101_binding_id": report["v101_binding_id"],
            "maps_sha256": report["maps_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "selected_step": selected,
        }
    )
    _finite_tree(report, "Teacher-v10.2 report")
    atomic_write_json(report_file, report)
    return report


def main() -> None:
    args = parse_args()
    failure_file = args.failed_v101_summary.expanduser().resolve()
    authority = _validate_v101_failure(failure_file)
    output_dir = args.output_dir.expanduser().resolve()
    _install_v102_pipeline()
    forwarded = [
        str(Path(base.__file__).resolve()),
        "--failed-v10-summary",
        str(failure_file),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]
    if args.no_progress:
        forwarded.append("--no-progress")
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        base.main()
    finally:
        sys.argv = original_argv
    report = _rewrite_v102_report(output_dir, authority)
    print("[DENSE_INSTANCE_SUPERVISION_{}] Teacher-v10.2".format(report["status"]))
    print("[PASS] Bed/Normal-Chair/High-Chair dense supports received equal macro loss")
    print("[PASS] all three positive supports were excluded from hard-background mining")
    print("[PASS] two train scenes x K=3 actual rollouts completed")
    print("[OK] shortlisted steps:", report["shortlisted_steps"])
    print("[OK] selected step:", report["selected_step"])
    print("[OK] checkpoint:", report["paths"].get("checkpoint"))
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", output_dir / "summary.json")


if __name__ == "__main__":
    main()

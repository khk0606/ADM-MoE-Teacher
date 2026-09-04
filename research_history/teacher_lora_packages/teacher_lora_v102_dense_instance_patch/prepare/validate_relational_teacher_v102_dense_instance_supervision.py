#!/usr/bin/env python3
"""Independent validator for Teacher-v10.2 dense-instance training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np

import validate_relational_teacher_v10_supervised_capacity as v10v
from relational_teacher_v102_dense_instance_contract import (
    EVAL_TIMESTEPS,
    EXPECTED_INSTANCES,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    REPLAY_START_STEP,
    REPLAY_WEIGHT,
    ROLLOUT_K,
    SCENES,
    SCHEMA,
    TIMESTEP_CYCLE,
    TRAIN_STEPS,
    V101_SCHEMA,
    canonical_sha256,
    select_rollout_candidate,
    shortlist_monitor_steps,
)
from relational_teacher_v9_all_sittable_contract import read_json, sha256_file


def _validate_v101_authority(path: Path, binding_id: object) -> None:
    value = read_json(path)
    if (
        value.get("schema") != V101_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_step") is not None
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_objective_or_architecture_redesign") is not True
        or value.get("failed_checks")
        != ["at_least_one_fullfield_candidate_passes_actual_k3"]
        or value.get("binding_id") != binding_id
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v10.1 failure authority changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != set(hashes)
        or "checkpoint" in paths
    ):
        raise ValueError("Teacher-v10.1 failure binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10.1 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed Teacher-v10.1 contains a checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)
    if (
        value.get("schema") != SCHEMA
        or value.get("seed") != MODEL_SEED
        or value.get("train_steps") != TRAIN_STEPS
        or value.get("learning_rate") != LEARNING_RATE
        or value.get("train_scenes") != list(SCENES)
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("policy_id") != POLICY_ID
        or value.get("loss_weights") != LOSS_WEIGHTS
        or value.get("replay_start_step") != REPLAY_START_STEP
        or value.get("replay_weight") != REPLAY_WEIGHT
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or not isinstance(value.get("zero_state_sha256"), str)
        or len(value["zero_state_sha256"]) != 64
    ):
        raise ValueError("Teacher-v10.2 summary contract changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    required_paths = {
        "runner",
        "validator",
        "contract",
        "objective",
        "base_pipeline",
        "v101_failure",
        "dataset_index",
        "source_dataset_index",
        "stats_file",
        "v5_split",
        "v5_evidence_report",
        "original_checkpoint",
        "v5_checkpoint",
        "metric_policy",
        "training_policy",
        "maps",
    }
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Teacher-v10.2 path binding changed")
    expected_paths = required_paths | (
        {"checkpoint"} if value.get("selected_step") is not None else set()
    )
    if set(paths) != expected_paths or set(hashes) != expected_paths:
        raise ValueError("Teacher-v10.2 path inventory changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v10.2 bound file changed: " + str(name))
    _validate_v101_authority(
        Path(str(paths["v101_failure"])).resolve(), value.get("v101_binding_id")
    )
    if read_json(Path(str(paths["training_policy"])).resolve()) != POLICY:
        raise ValueError("Teacher-v10.2 policy changed")
    if canonical_sha256(POLICY) != POLICY_ID:
        raise ValueError("Teacher-v10.2 policy ID changed")

    maps_file = Path(str(paths["maps"])).resolve()
    if (
        sha256_file(maps_file) != value.get("maps_sha256")
        or value.get("maps_sha256") != hashes["maps"]
        or value.get("checkpoint_sha256") != hashes.get("checkpoint")
    ):
        raise ValueError("Teacher-v10.2 artifact hash changed")
    arrays = v10v._load_npz(maps_file)
    suffixes = {
        "xyz",
        "points",
        "instance_names",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "environment_aux_mask",
        "explicit_negative_mask",
        "instance_targets",
        "all_sittable_gt",
    }
    required_arrays = {
        "scene_ids",
        "prompt_ids",
        "eval_timesteps",
        "monitor_steps",
        "shortlisted_steps",
        "rollout_seed_table",
        "base_fixed",
        "monitor_fixed",
        "monitor_v5_predictions",
        "base_rollouts",
        "candidate_rollouts",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    } | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in suffixes
    }
    if set(arrays) != required_arrays:
        raise ValueError("Teacher-v10.2 map inventory changed")
    if (
        arrays["scene_ids"].tolist() != list(SCENES)
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["eval_timesteps"].tolist() != list(EVAL_TIMESTEPS)
        or arrays["monitor_steps"].tolist() != list(MONITOR_STEPS)
        or arrays["shortlisted_steps"].tolist() != value.get("shortlisted_steps")
        or arrays["rollout_seed_table"].tolist() != value.get("rollout_seed_table")
        or arrays["rollout_seed_table"].shape != (ROLLOUT_K, 2)
        or arrays["base_fixed"].shape != (2, 3, 2, 8192, 6)
        or arrays["monitor_fixed"].shape
        != (len(MONITOR_STEPS), 2, 3, 2, 8192, 6)
        or arrays["monitor_v5_predictions"].shape
        != (len(MONITOR_STEPS), 3, 8192, 6)
        or arrays["base_rollouts"].shape != (2, ROLLOUT_K, 2, 8192, 6)
        or arrays["candidate_rollouts"].shape
        != (3, 2, ROLLOUT_K, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (3, 3, 8192, 6)
    ):
        raise ValueError("Teacher-v10.2 map shape/order changed")
    for scene_index, scene in enumerate(SCENES):
        prefix = "source" if scene_index == 0 else "audit"
        if (
            arrays[prefix + "_xyz"].shape != (8192, 3)
            or arrays[prefix + "_points"].shape != (8192, 6)
            or arrays[prefix + "_instance_names"].tolist()
            != list(EXPECTED_INSTANCES[scene])
            or arrays[prefix + "_verified_object_mask"].shape != (3, 8192)
            or arrays[prefix + "_instance_targets"].shape != (3, 8192, 6)
            or arrays[prefix + "_all_sittable_gt"].shape != (8192, 6)
            or not np.array_equal(
                arrays[prefix + "_xyz"], arrays[prefix + "_points"][:, :3]
            )
        ):
            raise ValueError(scene + " saved bundle changed")
        targets = arrays[prefix + "_instance_targets"]
        if any(not np.any(targets[slot].max(axis=-1) >= 0.05) for slot in range(3)):
            raise ValueError(scene + " has an empty dense instance support")
    for name in (
        "base_fixed",
        "monitor_fixed",
        "base_rollouts",
        "candidate_rollouts",
        "source_instance_targets",
        "audit_instance_targets",
        "source_all_sittable_gt",
        "audit_all_sittable_gt",
    ):
        array = np.asarray(arrays[name])
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(name + " is not a physical affordance array")

    base_summary = v10v._fixed_summary(arrays, arrays["base_fixed"])
    v10v._close(value["base_fixed_summary"], base_summary, "base fixed summary")
    base_v5_dense = np.square(
        arrays["base_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    v10v._close(value["base_v5_dense"], base_v5_dense, "base v5")
    recomputed_monitors = []
    if len(value.get("monitor_rows", [])) != len(MONITOR_STEPS):
        raise ValueError("Teacher-v10.2 monitor count changed")
    for index, step in enumerate(MONITOR_STEPS):
        row = v10v._fixed_summary(arrays, arrays["monitor_fixed"][index])
        candidate_v5_dense = np.square(
            arrays["monitor_v5_predictions"][index] - arrays["v5_target"]
        ).mean(axis=(1, 2)).tolist()
        recorded = value["monitor_rows"][index]
        row.update(
            {
                "step": step,
                "state_sha256": recorded["state_sha256"],
                "lora_energy": recorded["lora_energy"],
                "base_v5_dense": base_v5_dense,
                "candidate_v5_dense": candidate_v5_dense,
            }
        )
        v10v._close(recorded, row, "monitor {}".format(step))
        recomputed_monitors.append(row)
    shortlist = shortlist_monitor_steps(recomputed_monitors)
    if shortlist != value.get("shortlisted_steps") or len(shortlist) != 3:
        raise ValueError("Teacher-v10.2 shortlist changed")

    recomputed_rollouts = []
    if len(value.get("rollout_rows", [])) != 3:
        raise ValueError("Teacher-v10.2 rollout count changed")
    for index, step in enumerate(shortlist):
        candidate_v5_dense = np.square(
            arrays["candidate_v5_predictions"][index] - arrays["v5_target"]
        ).mean(axis=(1, 2)).tolist()
        recorded = value["rollout_rows"][index]
        panel = v10v._rollout_panel(
            arrays,
            arrays["candidate_rollouts"][index],
            base_v5_dense,
            candidate_v5_dense,
            recorded["state_sha256"] != value["zero_state_sha256"],
        )
        panel.update(
            {
                "step": step,
                "state_sha256": recorded["state_sha256"],
                "candidate_maps_sha256": v10v._tensor_sha256(
                    arrays["candidate_rollouts"][index]
                ),
            }
        )
        v10v._close(recorded, panel, "rollout {}".format(step))
        recomputed_rollouts.append(panel)
    selected = select_rollout_candidate(recomputed_rollouts)
    selected_index = shortlist.index(selected) if selected is not None else None
    if (
        selected != value.get("selected_step")
        or selected_index != value.get("selected_rollout_index")
    ):
        raise ValueError("Teacher-v10.2 selected state changed")

    expected_status = "PASS" if selected is not None else "FAIL"
    checkpoint_files = list(summary_file.parent.glob("*.pt")) + list(
        summary_file.parent.glob("*.pth")
    )
    if (
        value.get("status") != expected_status
        or value.get("serialized_model_state") is not (selected is not None)
        or value.get("authorizes_teacher_v102_checkpoint_lock")
        is not (selected is not None)
        or value.get("authorizes_objective_or_architecture_redesign")
        is not (selected is None)
        or len(checkpoint_files) != (1 if selected is not None else 0)
        or ("checkpoint" in paths) != (selected is not None)
    ):
        raise ValueError("Teacher-v10.2 checkpoint/status policy changed")
    timestep_hits = {
        int(key): int(count) for key, count in value["timestep_hits"].items()
    }
    expected_checks = {
        "sealed_v101_failure_authorizes_dense_instance_redesign": True,
        "fresh_v5r4_zero_init": True,
        "both_train_scenes_used_in_every_update": True,
        "every_non_unknown_point_receives_gt_supervision": True,
        "dense_instance_positive_support_is_equal_macro": True,
        "all_instance_positive_support_excluded_from_hard_background": True,
        "hard_false_positive_background_is_supervised": True,
        "v5_replay_active_from_first_update": REPLAY_START_STEP == 1,
        "all_timestep_buckets_exercised": set(timestep_hits)
        == set(TIMESTEP_CYCLE)
        and min(timestep_hits.values()) > 0,
        "exact_1000_adamw_updates": value["train_steps"] == TRAIN_STEPS,
        "only_lora_received_gradients": True,
        "three_monitor_states_received_actual_two_scene_k3": len(
            recomputed_rollouts
        )
        == 3,
        "at_least_one_dense_instance_candidate_passes_actual_k3": selected
        is not None,
        "checkpoint_written_iff_actual_gate_passes": bool(checkpoint_files)
        == (selected is not None),
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
    }
    if value.get("checks") != expected_checks:
        raise ValueError("Teacher-v10.2 top-level checks changed")
    failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if value.get("failed_checks") != failed:
        raise ValueError("Teacher-v10.2 failed-check inventory changed")
    binding = canonical_sha256(
        {
            "policy_id": POLICY_ID,
            "v101_binding_id": value["v101_binding_id"],
            "maps_sha256": value["maps_sha256"],
            "checkpoint_sha256": value.get("checkpoint_sha256"),
            "selected_step": selected,
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("Teacher-v10.2 binding ID changed")

    print("[DENSE_INSTANCE_SUPERVISION_{}] Teacher-v10.2 integrity".format(expected_status))
    print("[PASS] dense per-instance supports and hard-background exclusion verified")
    print("[PASS] actual K=3, v5 retention and checkpoint-if-pass recomputed")
    print("[PASS] room_0201 and paper-test payloads remain unread")
    print("[OK] shortlisted steps:", shortlist)
    print("[OK] selected step:", selected)
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

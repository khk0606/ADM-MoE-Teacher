#!/usr/bin/env python3
"""Deep, map-level validator for Teacher-v10 supervised capacity training."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from relational_teacher_v10_supervised_capacity_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    EVAL_TIMESTEPS,
    EXPECTED_INSTANCES,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    ROLES,
    ROLLOUT_K,
    SCENES,
    SCHEMA,
    TIMESTEP_CYCLE,
    TRAIN_STEPS,
    canonical_sha256,
    rollout_gate,
    select_rollout_candidate,
    shortlist_monitor_steps,
)
from relational_teacher_v9_all_sittable_contract import read_json, sha256_file  # noqa: E402
from relational_teacher_v9_all_sittable_metrics import (  # noqa: E402
    all_instance_metrics,
    simultaneous_presence_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _close(left: object, right: object, label: str) -> None:
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            raise ValueError(label + " mapping changed")
        for key in left:
            _close(left[key], right[key], label + "." + str(key))
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError(label + " sequence changed")
        for index, (one, two) in enumerate(zip(left, right)):
            _close(one, two, "{}[{}]".format(label, index))
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise ValueError(label + " boolean changed")
        return
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        if not math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " value changed")


def _sanitize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_sanitize(nested) for nested in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return 1_000_000.0
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _bundle(arrays: Mapping[str, np.ndarray], scene_index: int) -> Dict[str, object]:
    prefix = "source" if scene_index == 0 else "audit"
    return {
        "xyz": arrays[prefix + "_xyz"],
        "points": arrays[prefix + "_points"],
        "instance_names": arrays[prefix + "_instance_names"].tolist(),
        "verified_object_mask": arrays[prefix + "_verified_object_mask"],
        "verified_positive_mask": arrays[prefix + "_verified_positive_mask"],
        "unknown_sittable_mask": arrays[prefix + "_unknown_sittable_mask"],
        "environment_aux_mask": arrays[prefix + "_environment_aux_mask"],
        "explicit_negative_mask": arrays[prefix + "_explicit_negative_mask"],
        "instance_targets": arrays[prefix + "_instance_targets"],
        "all_target": arrays[prefix + "_all_sittable_gt"],
    }


def _metrics(bundle: Mapping[str, object], prediction: np.ndarray) -> Dict[str, object]:
    return _sanitize(
        all_instance_metrics(
            prediction,
            np.asarray(bundle["instance_targets"], np.float32),
            bundle["instance_names"],
            np.asarray(bundle["xyz"], np.float32),
            np.asarray(bundle["verified_object_mask"], bool),
            np.asarray(bundle["explicit_negative_mask"], bool),
            np.asarray(bundle["unknown_sittable_mask"], bool),
        )
    )


def _role_metrics(
    scene: str, bundle: Mapping[str, object], predictions: np.ndarray
) -> tuple[list, Dict[str, Dict[str, float]]]:
    rows = [
        [_metrics(bundle, predictions[generation, prompt]) for prompt in range(2)]
        for generation in range(predictions.shape[0])
    ]
    names = [str(name) for name in bundle["instance_names"]]
    pooled: Dict[str, Dict[str, float]] = {}
    for role, name in zip(ROLES, names):
        instance_rows = [
            rows[generation][prompt]["instances"][name]
            for generation in range(predictions.shape[0])
            for prompt in range(2)
        ]
        pooled[role] = {
            key: float(np.mean([float(row[key]) for row in instance_rows]))
            for key in (
                "soft_recall",
                "active_support_mae",
                "topk_overlap",
                "hotspot_centroid_distance_xy",
            )
        }
    return rows, pooled


def _known_dense_mae(bundle: Mapping[str, object], predictions: np.ndarray) -> float:
    known = (
        np.asarray(bundle["verified_positive_mask"], bool)
        | np.asarray(bundle["environment_aux_mask"], bool)
        | np.asarray(bundle["explicit_negative_mask"], bool)
    )
    target = np.asarray(bundle["all_target"], np.float32)
    return float(np.abs(predictions[:, :, known] - target[None, None, known]).mean())


def _fixed_summary(
    arrays: Mapping[str, np.ndarray], predictions: np.ndarray
) -> Dict[str, object]:
    per_scene_role = {}
    dense_values = []
    for scene_index, scene in enumerate(SCENES):
        bundle = _bundle(arrays, scene_index)
        _, pooled = _role_metrics(scene, bundle, predictions[scene_index])
        per_scene_role[scene] = pooled
        dense_values.append(_known_dense_mae(bundle, predictions[scene_index]))
    return {
        "per_scene_role": per_scene_role,
        "worst_instance_soft_recall": min(
            per_scene_role[scene][role]["soft_recall"]
            for scene in SCENES
            for role in ROLES
        ),
        "worst_instance_active_support_mae": max(
            per_scene_role[scene][role]["active_support_mae"]
            for scene in SCENES
            for role in ROLES
        ),
        "known_dense_mae": float(np.mean(dense_values)),
    }


def _prompt_invariance(bundle: Mapping[str, object], predictions: np.ndarray) -> list[float]:
    mask = np.asarray(bundle["verified_positive_mask"], bool)
    return [
        float(np.square(predictions[generation, 0, mask] - predictions[generation, 1, mask]).mean())
        for generation in range(ROLLOUT_K)
    ]


def _rollout_panel(
    arrays: Mapping[str, np.ndarray],
    predictions: np.ndarray,
    base_v5_dense: Sequence[float],
    candidate_v5_dense: Sequence[float],
    state_changed: bool,
) -> Dict[str, object]:
    scene_rows = {}
    pooled_metrics = {}
    presence = {}
    counts = {}
    invariance = {}
    for scene_index, scene in enumerate(SCENES):
        bundle = _bundle(arrays, scene_index)
        rows, pooled = _role_metrics(scene, bundle, predictions[scene_index])
        scene_rows[scene] = rows
        pooled_metrics[scene] = pooled
        presence[scene] = [
            [
                simultaneous_presence_checks(
                    rows[generation][prompt], **ABSOLUTE_PRESENCE_LIMITS
                )
                for prompt in range(2)
            ]
            for generation in range(ROLLOUT_K)
        ]
        counts[scene] = {
            "watch": sum(all(presence[scene][generation][0].values()) for generation in range(ROLLOUT_K)),
            "write": sum(all(presence[scene][generation][1].values()) for generation in range(ROLLOUT_K)),
        }
        invariance[scene] = _prompt_invariance(bundle, predictions[scene_index])
    checks = rollout_gate(
        all_three_counts=counts,
        prompt_invariance=invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        checkpoint_state_changed=state_changed,
    )
    return {
        "scene_rows": scene_rows,
        "pooled_metrics": pooled_metrics,
        "presence": presence,
        "all_three_counts": counts,
        "prompt_invariance": invariance,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "eligible": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)
    if (
        value.get("schema") != SCHEMA
        or value.get("train_steps") != TRAIN_STEPS
        or value.get("train_scenes") != list(SCENES)
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("policy_id") != POLICY_ID
        or not isinstance(value.get("zero_state_sha256"), str)
        or len(value["zero_state_sha256"]) != 64
    ):
        raise ValueError("Teacher-v10 summary contract changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v10 path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v10 bound file changed: " + str(name))
    failure_authority = read_json(Path(str(paths["v9812_failure"])).resolve())
    if (
        failure_authority.get("schema") != "relational_teacher_v9812_two_scene_multiupdate_calibration_v1"
        or failure_authority.get("status") != "FAIL"
        or failure_authority.get("failed_checks")
        != ["at_least_one_all_three_state_is_shortlisted"]
        or failure_authority.get("serialized_model_state") is not False
        or failure_authority.get(
            "authorizes_cross_scene_objective_or_model_capacity_redesign"
        )
        is not True
        or failure_authority.get("development_arrays_read") is not False
        or failure_authority.get("paper_test_access") is not False
        or value.get("v9812_binding_id") != failure_authority.get("binding_id")
    ):
        raise ValueError("sealed Teacher-v9.8.12 failure authority changed")
    policy_file = Path(str(paths["training_policy"])).resolve()
    if canonical_sha256(read_json(policy_file)) != POLICY_ID or read_json(policy_file) != POLICY:
        raise ValueError("Teacher-v10 training policy changed")
    maps_file = Path(str(paths["maps"])).resolve()
    if sha256_file(maps_file) != value.get("maps_sha256"):
        raise ValueError("Teacher-v10 maps hash changed")
    arrays = _load_npz(maps_file)

    base_names = {
        "scene_ids",
        "prompt_ids",
        "eval_timesteps",
        "monitor_steps",
        "shortlisted_steps",
        "rollout_seed_table",
        "base_fixed",
        "monitor_fixed",
        "base_rollouts",
        "candidate_rollouts",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    }
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
    required = base_names | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in suffixes
    }
    if set(arrays) != required:
        raise ValueError("Teacher-v10 map inventory changed")
    if (
        arrays["scene_ids"].tolist() != list(SCENES)
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["eval_timesteps"].tolist() != list(EVAL_TIMESTEPS)
        or arrays["monitor_steps"].tolist() != list(MONITOR_STEPS)
        or arrays["shortlisted_steps"].tolist() != value.get("shortlisted_steps")
        or arrays["rollout_seed_table"].tolist() != value.get("rollout_seed_table")
        or arrays["rollout_seed_table"].shape != (ROLLOUT_K, 2)
        or arrays["base_fixed"].shape != (2, 3, 2, 8192, 6)
        or arrays["monitor_fixed"].shape != (len(MONITOR_STEPS), 2, 3, 2, 8192, 6)
        or arrays["base_rollouts"].shape != (2, ROLLOUT_K, 2, 8192, 6)
        or arrays["candidate_rollouts"].shape != (3, 2, ROLLOUT_K, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (3, 3, 8192, 6)
    ):
        raise ValueError("Teacher-v10 map shape/order changed")
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
        ):
            raise ValueError(scene + " saved bundle changed")
    for name in (
        "base_fixed",
        "monitor_fixed",
        "base_rollouts",
        "candidate_rollouts",
        "source_instance_targets",
        "audit_instance_targets",
    ):
        array = np.asarray(arrays[name])
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(name + " is not finite physical affordance")

    base_summary = _fixed_summary(arrays, arrays["base_fixed"])
    _close(value["base_fixed_summary"], base_summary, "base fixed summary")
    recomputed_monitors = []
    for index, step in enumerate(MONITOR_STEPS):
        summary = _fixed_summary(arrays, arrays["monitor_fixed"][index])
        recorded = value["monitor_rows"][index]
        summary.update(
            {
                "step": step,
                "state_sha256": recorded["state_sha256"],
                "lora_energy": recorded["lora_energy"],
            }
        )
        _close(recorded, summary, "monitor {}".format(step))
        recomputed_monitors.append(summary)
    shortlist = shortlist_monitor_steps(recomputed_monitors)
    if shortlist != value.get("shortlisted_steps") or len(shortlist) != 3:
        raise ValueError("Teacher-v10 shortlist changed")

    base_v5_dense = np.square(
        arrays["base_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    _close(value["base_v5_dense"], base_v5_dense, "base v5")
    recomputed_rollouts = []
    for index, step in enumerate(shortlist):
        candidate_v5_dense = np.square(
            arrays["candidate_v5_predictions"][index] - arrays["v5_target"]
        ).mean(axis=(1, 2)).tolist()
        recorded = value["rollout_rows"][index]
        panel = _rollout_panel(
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
                "candidate_maps_sha256": _tensor_sha256(
                    arrays["candidate_rollouts"][index]
                ),
            }
        )
        _close(recorded, panel, "rollout {}".format(step))
        recomputed_rollouts.append(panel)
    selected = select_rollout_candidate(recomputed_rollouts)
    if selected != value.get("selected_step"):
        raise ValueError("Teacher-v10 selected step changed")
    expected_selected_index = (
        shortlist.index(selected) if selected is not None else None
    )
    if value.get("selected_rollout_index") != expected_selected_index:
        raise ValueError("Teacher-v10 selected rollout index changed")

    expected_status = "PASS" if selected is not None else "FAIL"
    checkpoint_files = list(summary_file.parent.glob("*.pt")) + list(
        summary_file.parent.glob("*.pth")
    )
    if (
        value.get("status") != expected_status
        or value.get("serialized_model_state") is not (selected is not None)
        or value.get("authorizes_teacher_v10_checkpoint_lock")
        is not (selected is not None)
        or value.get("authorizes_capacity_or_objective_redesign")
        is not (selected is None)
        or (selected is not None and len(checkpoint_files) != 1)
        or (selected is None and checkpoint_files)
        or (selected is not None and "checkpoint" not in paths)
        or (selected is None and "checkpoint" in paths)
    ):
        raise ValueError("Teacher-v10 checkpoint/status policy changed")
    timestep_hits = {int(key): int(count) for key, count in value["timestep_hits"].items()}
    expected_checks = {
        "sealed_v9812_failure_authorizes_redesign": True,
        "fresh_v5r4_zero_init": True,
        "both_train_scenes_used_in_every_update": True,
        "all_timestep_buckets_exercised": set(timestep_hits) == set(TIMESTEP_CYCLE)
        and min(timestep_hits.values()) > 0,
        "exact_1200_adamw_updates": value["train_steps"] == TRAIN_STEPS,
        "only_lora_received_gradients": True,
        "three_monitor_states_received_actual_two_scene_k3": len(recomputed_rollouts)
        == 3,
        "at_least_one_supervised_candidate_passes_actual_k3": selected is not None,
        "checkpoint_written_iff_actual_gate_passes": bool(checkpoint_files)
        == (selected is not None),
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
    }
    if value.get("checks") != expected_checks:
        raise ValueError("Teacher-v10 top-level checks changed")
    failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if value.get("failed_checks") != failed:
        raise ValueError("Teacher-v10 failed-check inventory changed")
    expected_binding_id = canonical_sha256(
        {
            "policy_id": POLICY_ID,
            "v9812_binding_id": value["v9812_binding_id"],
            "maps_sha256": value["maps_sha256"],
            "checkpoint_sha256": value.get("checkpoint_sha256"),
            "selected_step": selected,
        }
    )
    if value.get("binding_id") != expected_binding_id:
        raise ValueError("Teacher-v10 report binding ID changed")

    print("[SUPERVISED_CAPACITY_{}] Teacher-v10 integrity".format(expected_status))
    print("[PASS] direct-label two-scene monitor maps and K=3 rollouts recomputed")
    print("[PASS] shortlist, v5 retention and checkpoint-if-pass policy verified")
    print("[PASS] room_0201 and paper-test payloads remain unread")
    print("[OK] shortlisted steps:", shortlist)
    print("[OK] selected step:", selected)
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

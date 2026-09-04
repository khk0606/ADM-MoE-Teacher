#!/usr/bin/env python3
"""Deep validator for the Teacher-v9.1 corrected one-scene overfit smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PREPARE_ROOT = Path(__file__).resolve().parent
if str(PREPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPARE_ROOT))

from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    PROMPTS,
    sha256_file,
)
from relational_teacher_v9_all_sittable_metrics import (  # noqa: E402
    all_instance_metrics,
    simultaneous_presence_checks,
)
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    ABSOLUTE_PRESENCE_LIMITS,
    canonical_sha256,
)
from relational_teacher_v91_corrected_overfit_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    GRAD_CLIP,
    HELDOUT_TRAIN_SCENE,
    LEARNING_RATE,
    MONITOR_STEPS,
    SCHEMA,
    SEED,
    SELECTED_LOSS_CANDIDATE,
    STEPS,
    TRAIN_SCENE,
    evaluate_smoke_panel,
    rank_eligible_monitor_steps,
    sanitize_metric_nonfinite,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _stable_seeds,
    _validate_loss_response,
    _validate_preflight,
)


REQUIRED_PATHS = {
    "runner",
    "validator",
    "contract",
    "corrected_objective",
    "loss_response_contract",
    "loss_response_report",
    "preflight_report",
    "metric_policy",
    "dataset_index",
    "source_dataset_index",
    "stats_file",
    "v5_split",
    "v5_evidence_report",
    "original_checkpoint",
    "v5_checkpoint",
    "rollout_maps",
}


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(label + " contains a non-finite value")


def _metrics(arrays: Mapping[str, np.ndarray], prediction: np.ndarray) -> Mapping[str, object]:
    return sanitize_metric_nonfinite(
        all_instance_metrics(
            prediction,
            arrays["instance_targets"],
            ("bed_01", "chair_01", "chair_06"),
            arrays["xyz"],
            arrays["verified_object_mask"],
            arrays["explicit_negative_mask"],
            arrays["unknown_sittable_mask"],
        )
    )


def _invariance(value: np.ndarray, mask: np.ndarray) -> float:
    return float(np.square(value[0, mask] - value[1, mask]).mean())


def _recompute_panel(
    arrays: Mapping[str, np.ndarray],
    policy: Mapping[str, object],
    base_v5: Sequence[float],
    candidate_v5: Sequence[float],
) -> Mapping[str, object]:
    base = arrays["frozen_base"]
    candidate = arrays["candidate"]
    row_ids = [f"{TRAIN_SCENE}|{name}" for name in sorted(PROMPTS)]
    base_rows = {name: _metrics(arrays, base[index]) for index, name in enumerate(row_ids)}
    candidate_rows = {
        name: _metrics(arrays, candidate[index]) for index, name in enumerate(row_ids)
    }
    presence = {
        name: simultaneous_presence_checks(
            candidate_rows[name], **ABSOLUTE_PRESENCE_LIMITS
        )
        for name in row_ids
    }
    mask = arrays["verified_object_mask"].any(axis=0)
    base_invariance = _invariance(base, mask)
    candidate_invariance = _invariance(candidate, mask)
    checks = evaluate_smoke_panel(
        candidate_rows=candidate_rows,
        base_rows=base_rows,
        absolute_presence=presence,
        base_prompt_invariance=base_invariance,
        candidate_prompt_invariance=candidate_invariance,
        base_v5_dense=base_v5,
        candidate_v5_dense=candidate_v5,
        relative_rules=policy["relative_selection_rules"],
    )
    return {
        "base_rows": base_rows,
        "candidate_rows": candidate_rows,
        "absolute_presence": presence,
        "base_prompt_invariance": base_invariance,
        "candidate_prompt_invariance": candidate_invariance,
        "base_v5_dense": [float(value) for value in base_v5],
        "candidate_v5_dense": [float(value) for value in candidate_v5],
        "checks": checks,
        "eligible": all(checks.values()),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def _equal_tree(left: object, right: object, label: str) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(label + " keys changed")
        for key in left:
            _equal_tree(left[key], right[key], label + "." + str(key))
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(label + " length changed")
        for index, (one, two) in enumerate(zip(left, right)):
            _equal_tree(one, two, f"{label}[{index}]")
        return
    if isinstance(left, (float, int)) and not isinstance(left, bool) and isinstance(
        right, (float, int)
    ) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = json.loads(summary_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.1 corrected smoke schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("diffusion_steps") != 500
        or value.get("steps") != STEPS
        or value.get("learning_rate") != LEARNING_RATE
        or value.get("grad_clip") != GRAD_CLIP
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != HELDOUT_TRAIN_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("prompts") != PROMPTS
        or value.get("forward_input_keys")
        != sorted(("c_pc_feat", "c_pc_xyz", "c_text"))
        or value.get("monitor_steps") != list(MONITOR_STEPS)
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.1 corrected smoke protocol changed")
    if value.get("selection_policy") != (
        "maximize_worst_topk_then_recall_then_minimize_worst_mae_"
        "then_earliest_else_final_diagnostic"
    ):
        raise ValueError("overfit monitor selection policy changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("overfit path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("overfit path-hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("overfit-bound file changed: " + name)
    if summary_file.parent != Path(str(paths["rollout_maps"])).resolve().parent:
        raise ValueError("summary/array output directories differ")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("overfit smoke persisted forbidden model state")

    preflight, policy = _validate_preflight(Path(str(paths["preflight_report"])))
    loss_response, selected_loss = _validate_loss_response(
        Path(str(paths["loss_response_report"]))
    )
    if (
        value.get("preflight_binding_id") != preflight.get("binding_id")
        or value.get("loss_response_binding_id") != loss_response.get("binding_id")
        or loss_response.get("preflight_binding_id") != preflight.get("binding_id")
        or value.get("selected_loss_candidate") != SELECTED_LOSS_CANDIDATE
        or value.get("selected_loss_weights")
        != {
            name: selected_loss[name]
            for name in (
                "active_support_weight",
                "ranking_weight",
                "negative_weight",
                "prompt_weight",
                "v5_preservation_weight",
            )
        }
        or value.get("metric_policy_id") != preflight.get("metric_policy_id")
        or value.get("metric_policy_sha256")
        != preflight.get("metric_policy_sha256")
        or paths["metric_policy"] != preflight["paths"]["metric_policy"]
    ):
        raise ValueError("overfit/preflight authority binding changed")
    expected_scene_binding = {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]
    if value.get("scene_binding") != expected_scene_binding:
        raise ValueError("room_0101 scene binding changed")
    if value.get("rollout_maps_sha256") != hashes["rollout_maps"]:
        raise ValueError("rollout-map hash field changed")
    if value.get("optimizer") != {"name": "AdamW", "weight_decay": 0.0}:
        raise ValueError("overfit optimizer changed")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or lora.get("rank") != 4 or float(
        lora.get("alpha", 0.0)
    ) != 8.0 or int(lora.get("module_count", 0)) != 31:
        raise ValueError("overfit LoRA contract changed")

    maps_file = Path(str(paths["rollout_maps"])).resolve()
    with np.load(maps_file, allow_pickle=False) as source:
        required_arrays = {
            "xyz",
            "points",
            "instance_ids",
            "category_ids",
            "verified_object_mask",
            "unknown_sittable_mask",
            "explicit_negative_mask",
            "instance_targets",
            "all_sittable_gt",
            "frozen_base",
            "candidate",
            "frozen_base_normalized",
            "candidate_normalized",
            "prompt_ids",
        }
        if set(source.files) != required_arrays:
            raise ValueError("rollout array inventory changed")
        arrays = {name: np.asarray(source[name]) for name in source.files}
    expected_shapes = {
        "xyz": (8192, 3),
        "points": (8192, 6),
        "instance_ids": (8192,),
        "category_ids": (8192,),
        "verified_object_mask": (3, 8192),
        "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,),
        "instance_targets": (3, 8192, 6),
        "all_sittable_gt": (8192, 6),
        "frozen_base": (2, 8192, 6),
        "candidate": (2, 8192, 6),
        "frozen_base_normalized": (2, 8192, 6),
        "candidate_normalized": (2, 8192, 6),
        "prompt_ids": (2,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape changed")
    if tuple(str(item) for item in arrays["prompt_ids"].tolist()) != tuple(sorted(PROMPTS)):
        raise ValueError("rollout prompt order changed")
    for name in (
        "xyz",
        "points",
        "instance_targets",
        "all_sittable_gt",
        "frozen_base",
        "candidate",
        "frozen_base_normalized",
        "candidate_normalized",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "all_sittable_gt", "frozen_base", "candidate"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if not np.array_equal(arrays["all_sittable_gt"], arrays["instance_targets"].max(axis=0)):
        raise ValueError("rollout GT is not exact three-instance max union")
    if not np.array_equal(arrays["xyz"], arrays["points"][:, :3]):
        raise ValueError("rollout points/xyz differ")
    if np.any(arrays["verified_object_mask"].sum(axis=1) <= 0):
        raise ValueError("verified object mask is empty")
    if np.any(arrays["verified_object_mask"].sum(axis=0) > 1):
        raise ValueError("verified object masks overlap")
    if np.any(
        arrays["unknown_sittable_mask"] & arrays["verified_object_mask"].any(axis=0)
    ) or np.any(
        arrays["explicit_negative_mask"]
        & (arrays["unknown_sittable_mask"] | arrays["verified_object_mask"].any(axis=0))
    ):
        raise ValueError("rollout semantic masks overlap")

    consensus_file = Path(str(expected_scene_binding["consensus_file"])).resolve()
    points_file = Path(str(expected_scene_binding["points_file"])).resolve()
    if (
        not consensus_file.is_file()
        or sha256_file(consensus_file) != expected_scene_binding["consensus_sha256"]
        or not points_file.is_file()
        or sha256_file(points_file) != expected_scene_binding["points_sha256"]
    ):
        raise ValueError("room_0101 bound source arrays changed")
    with np.load(consensus_file, allow_pickle=False) as source:
        source_arrays = {
            "xyz": np.asarray(source["xyz"]),
            "instance_ids": np.asarray(source["instance_ids"]),
            "category_ids": np.asarray(source["category_ids"]),
            "verified_object_mask": np.asarray(source["verified_object_mask"]),
            "unknown_sittable_mask": np.asarray(source["unknown_sittable_mask"]),
            "explicit_negative_mask": np.asarray(source["explicit_negative_mask"]),
            "instance_targets": np.asarray(source["instance_affordance"]),
            "all_sittable_gt": np.asarray(source["all_sittable_affordance"]),
        }
    with np.load(points_file, allow_pickle=False) as source:
        source_points = np.asarray(source["points"])
    for name, expected in source_arrays.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError("rollout/source array differs: " + name)
    if not np.array_equal(arrays["points"], source_points):
        raise ValueError("rollout/source point cloud differs")
    with np.load(Path(str(paths["stats_file"])), allow_pickle=False) as source:
        mean = np.asarray(source["mean"], dtype=np.float32).reshape(1, 1, 6)
        std = np.asarray(source["std"], dtype=np.float32).reshape(1, 1, 6)
    for normalized_name, physical_name in (
        ("frozen_base_normalized", "frozen_base"),
        ("candidate_normalized", "candidate"),
    ):
        expected = np.clip(arrays[normalized_name] * std + mean, 0.0, 1.0).astype(
            np.float32
        )
        if not np.array_equal(expected, arrays[physical_name]):
            raise ValueError(physical_name + " denormalization changed")

    rollout = value.get("rollout")
    if not isinstance(rollout, Mapping) or rollout.get(
        "prompt_pair_uses_identical_randomness"
    ) is not True:
        raise ValueError("rollout random-pair contract changed")
    expected_initial, expected_reverse = _stable_seeds(
        SEED, "v91_corrected_one_scene_overfit_rollout", TRAIN_SCENE
    )
    if rollout.get("initial_seed") != expected_initial or rollout.get(
        "reverse_seed"
    ) != expected_reverse:
        raise ValueError("rollout seed derivation changed")
    reported_panel = rollout.get("panel")
    if not isinstance(reported_panel, Mapping):
        raise ValueError("rollout panel is absent")
    recomputed = _recompute_panel(
        arrays,
        policy,
        reported_panel["base_v5_dense"],
        reported_panel["candidate_v5_dense"],
    )
    _equal_tree(recomputed, reported_panel, "rollout panel")

    monitors = value.get("monitor_panels")
    if not isinstance(monitors, list) or [row.get("step") for row in monitors] != list(
        MONITOR_STEPS
    ):
        raise ValueError("monitor panels are incomplete or out of order")
    eligible_steps = []
    initial_base_rows = monitors[0].get("base_rows")
    initial_base_v5 = monitors[0].get("base_v5_dense")
    for row in monitors:
        if row.get("base_rows") != initial_base_rows or row.get(
            "base_v5_dense"
        ) != initial_base_v5:
            raise ValueError("frozen monitor baseline changed across updates")
        if len(str(row.get("prediction_sha256", ""))) != 64 or len(
            str(row.get("v5_prediction_sha256", ""))
        ) != 64:
            raise ValueError("monitor tensor hash is invalid")
        recomputed_checks = evaluate_smoke_panel(
            candidate_rows=row["candidate_rows"],
            base_rows=row["base_rows"],
            absolute_presence=row["absolute_presence"],
            base_prompt_invariance=row["base_prompt_invariance"],
            candidate_prompt_invariance=row["candidate_prompt_invariance"],
            base_v5_dense=row["base_v5_dense"],
            candidate_v5_dense=row["candidate_v5_dense"],
            relative_rules=policy["relative_selection_rules"],
        )
        if row.get("checks") != recomputed_checks:
            raise ValueError("monitor check arithmetic changed")
        eligible = all(recomputed_checks.values())
        if row.get("eligible") is not eligible:
            raise ValueError("monitor eligibility changed")
        failed = sorted(name for name, passed in recomputed_checks.items() if not passed)
        if row.get("failed_checks") != failed:
            raise ValueError("monitor failed-check inventory changed")
        if eligible and int(row["step"]) > 0:
            eligible_steps.append(int(row["step"]))
    initial = monitors[0]
    if (
        initial.get("candidate_rows") != initial.get("base_rows")
        or initial.get("candidate_v5_dense") != initial.get("base_v5_dense")
        or initial.get("candidate_prompt_invariance")
        != initial.get("base_prompt_invariance")
        or float(initial.get("lora_energy", -1.0)) != 0.0
        or initial.get("eligible") is not False
    ):
        raise ValueError("zero-init monitor is not exact Base parity")
    if value.get("eligible_monitor_steps") != eligible_steps:
        raise ValueError("eligible monitor-step list changed")
    eligible_rows = [row for row in monitors if row.get("eligible") is True]
    ranked_steps = rank_eligible_monitor_steps(eligible_rows)
    if value.get("ranked_eligible_monitor_steps") != ranked_steps:
        raise ValueError("ranked eligible monitor steps changed")
    expected_selected = ranked_steps[0] if ranked_steps else STEPS
    if value.get("selected_monitor_step") != expected_selected:
        raise ValueError("selected monitor step changed")

    checks = value.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("top-level checks are absent")
    expected_checks = {
        "sealed_loss_response_authority": True,
        "sealed_preflight_authority": True,
        "fresh_v5r4_zero_init": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "exact_120_optimizer_updates": True,
        "only_lora_updated": True,
        "one_step_three_object_panel_passes": bool(eligible_steps),
        "reverse_diffusion_three_object_panel_passes": bool(recomputed["eligible"]),
        "lora_changed_from_zero": checks.get("lora_changed_from_zero"),
        "no_model_checkpoint_saved": True,
        "teacher_forward_is_text_plus_scene_only": True,
    }
    if checks != expected_checks or checks.get("lora_changed_from_zero") is not True:
        raise ValueError("top-level gate checks changed")
    expected_status = "PASS" if all(checks.values()) else "FAIL"
    if value.get("status") != expected_status:
        raise ValueError("overfit status arithmetic changed")
    if value.get("failed_checks") != sorted(
        name for name, passed in checks.items() if not passed
    ):
        raise ValueError("top-level failed checks changed")
    if value.get("authorizes_fresh_response3") != (expected_status == "PASS"):
        raise ValueError("response3 authorization changed")
    if any(
        value.get(name) is not False
        for name in (
            "authorizes_calibration",
            "authorizes_long_training",
            "authorizes_development_evaluation",
            "authorizes_paper_test",
        )
    ):
        raise ValueError("overfit smoke over-authorized later work")

    calculated_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "loss_response_binding_id": value["loss_response_binding_id"],
            "selected_loss_candidate": value["selected_loss_candidate"],
            "selected_loss_weights": value["selected_loss_weights"],
            "metric_policy_id": value["metric_policy_id"],
            "scene_binding": value["scene_binding"],
            "hyperparameters": {
                "steps": value["steps"],
                "learning_rate": value["learning_rate"],
                "grad_clip": value["grad_clip"],
                "seed": value["seed"],
            },
            "rollout_maps_sha256": value["rollout_maps_sha256"],
        }
    )
    if value.get("binding_id") != calculated_binding:
        raise ValueError("overfit binding ID changed")
    _finite_tree(value, "overfit report")
    print(f"[CORRECTED_OVERFIT_{expected_status}] Teacher-v9.1 smoke integrity")
    print("[PASS] selected loss response, v5r4 and room_0101 binding verified")
    print("[PASS] 120 LoRA-only updates and all fixed monitor gates recomputed")
    print("[PASS] paired reverse-diffusion maps and three-instance metrics recomputed")
    print("[PASS] no checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] selected monitor step:", value["selected_monitor_step"])
    print("[OK] rollout presence:", recomputed["absolute_presence"])
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

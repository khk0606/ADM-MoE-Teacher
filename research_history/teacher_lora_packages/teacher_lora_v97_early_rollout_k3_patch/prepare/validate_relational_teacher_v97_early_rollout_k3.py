#!/usr/bin/env python3
"""Deep artifact validator for Teacher-v9.7 early rollout K=3."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np


PREPARE_ROOT = Path(__file__).resolve().parent
if str(PREPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPARE_ROOT))

from relational_teacher_v9_all_sittable_contract import PROMPTS, sha256_file  # noqa: E402
from relational_teacher_v9_all_sittable_metrics import all_instance_metrics  # noqa: E402
from relational_teacher_v91_corrected_overfit_contract import sanitize_metric_nonfinite  # noqa: E402
from relational_teacher_v97_early_rollout_k3_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    FAILED_V91_SCHEMA,
    GENERATION_COUNT,
    GRAD_CLIP,
    HELDOUT_TRAIN_SCENE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    LOSS_RESPONSE_SCHEMA,
    OBJECTS,
    PROMPT_IDS,
    PREFLIGHT_SCHEMA,
    ROLLOUT_POLICY,
    ROLLOUT_POLICY_ID,
    ROLLOUT_SEED,
    SCHEMA,
    SELECTED_LOSS_CANDIDATE,
    SNAPSHOT_STEPS,
    TRAINING_SEED,
    TRAINING_STEPS,
    TRAIN_SCENE,
    canonical_sha256,
    continuous_presence_checks,
    pooled_object_metrics,
    rank_eligible_steps,
    rollout_step_checks,
    stable_rollout_seeds,
)


REQUIRED_PATHS = {
    "runner",
    "validator",
    "contract",
    "summarizer",
    "failed_v91_summary",
    "failed_v91_rollout_maps",
    "preflight_report",
    "loss_response_report",
    "original_metric_policy",
    "rollout_metric_policy",
    "dataset_index",
    "source_dataset_index",
    "stats_file",
    "v5_split",
    "v5_evidence_report",
    "original_checkpoint",
    "v5_checkpoint",
    "corrected_objective",
    "v91_runner",
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
        raise ValueError(label + " contains NaN/Inf")


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
            _equal_tree(one, two, "{}[{}]".format(label, index))
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " changed")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _metrics(arrays: Mapping[str, np.ndarray], prediction: np.ndarray) -> Mapping[str, object]:
    return sanitize_metric_nonfinite(
        all_instance_metrics(
            prediction,
            arrays["instance_targets"],
            OBJECTS,
            arrays["xyz"],
            arrays["verified_object_mask"],
            arrays["explicit_negative_mask"],
            arrays["unknown_sittable_mask"],
        )
    )


def _rows(
    arrays: Mapping[str, np.ndarray], values: np.ndarray
) -> list[list[Mapping[str, object]]]:
    return [
        [_metrics(arrays, values[generation, prompt]) for prompt in range(2)]
        for generation in range(GENERATION_COUNT)
    ]


def _invariance(values: np.ndarray, mask: np.ndarray) -> list[float]:
    return [
        float(np.square(values[generation, 0, mask] - values[generation, 1, mask]).mean())
        for generation in range(GENERATION_COUNT)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = json.loads(summary_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.7 schema/status changed")
    if (
        value.get("training_seed") != TRAINING_SEED
        or value.get("rollout_seed") != ROLLOUT_SEED
        or value.get("diffusion_steps") != 500
        or value.get("training_steps") != TRAINING_STEPS
        or value.get("learning_rate") != LEARNING_RATE
        or value.get("grad_clip") != GRAD_CLIP
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != HELDOUT_TRAIN_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("snapshot_steps") != list(SNAPSHOT_STEPS)
        or value.get("generation_count") != GENERATION_COUNT
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("prompt_text") != {name: PROMPTS[name] for name in PROMPT_IDS}
        or value.get("forward_input_keys") != ["c_pc_feat", "c_pc_xyz", "c_text"]
        or value.get("reverse_diffusion_draw_count") != 28
        or value.get("selected_loss_candidate") != SELECTED_LOSS_CANDIDATE
        or value.get("serialized_model_state") is not False
        or value.get("selection_policy") != ROLLOUT_POLICY["selection_order"]
    ):
        raise ValueError("Teacher-v9.7 protocol changed")
    expected_seeds = [list(stable_rollout_seeds(index)) for index in range(GENERATION_COUNT)]
    if value.get("seed_table") != expected_seeds:
        raise ValueError("Teacher-v9.7 seed table changed")
    if value.get("rollout_metric_policy_id") != ROLLOUT_POLICY_ID:
        raise ValueError("Teacher-v9.7 rollout metric policy ID changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("Teacher-v9.7 path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("Teacher-v9.7 hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.7 bound file changed: " + name)
    if summary_file.parent != Path(str(paths["rollout_maps"])).resolve().parent:
        raise ValueError("Teacher-v9.7 summary/maps directories differ")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.7 persisted forbidden model state")

    policy_file = Path(str(paths["rollout_metric_policy"])).resolve()
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if policy != ROLLOUT_POLICY or canonical_sha256(policy) != ROLLOUT_POLICY_ID:
        raise ValueError("Teacher-v9.7 rollout metric policy changed")
    if value.get("rollout_metric_policy_sha256") != hashes["rollout_metric_policy"]:
        raise ValueError("Teacher-v9.7 policy file hash field changed")
    if value.get("rollout_maps_sha256") != hashes["rollout_maps"]:
        raise ValueError("Teacher-v9.7 map hash field changed")

    preflight = json.loads(Path(str(paths["preflight_report"])).read_text(encoding="utf-8"))
    response = json.loads(Path(str(paths["loss_response_report"])).read_text(encoding="utf-8"))
    failed = json.loads(Path(str(paths["failed_v91_summary"])).read_text(encoding="utf-8"))
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "PASS"
        or preflight.get("binding_id") != value.get("preflight_binding_id")
        or preflight.get("development_payloads_read") is not False
        or preflight.get("failed_checks")
    ):
        raise ValueError("Teacher-v9 preflight authority changed")
    if (
        response.get("schema") != LOSS_RESPONSE_SCHEMA
        or response.get("status") != "PASS"
        or response.get("binding_id") != value.get("loss_response_binding_id")
        or response.get("selected_candidate") != SELECTED_LOSS_CANDIDATE
        or response.get("preflight_binding_id") != preflight.get("binding_id")
        or response.get("failed_checks")
    ):
        raise ValueError("Teacher-v9.1 loss-response authority changed")
    if (
        failed.get("schema") != FAILED_V91_SCHEMA
        or failed.get("status") != "FAIL"
        or failed.get("binding_id") != value.get("failed_v91_binding_id")
        or failed.get("preflight_binding_id") != preflight.get("binding_id")
        or failed.get("loss_response_binding_id") != response.get("binding_id")
        or failed.get("failed_checks")
        != ["one_step_three_object_panel_passes", "reverse_diffusion_three_object_panel_passes"]
    ):
        raise ValueError("Teacher-v9.1 sealed failure authority changed")
    preflight_scene = {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]
    if value.get("scene_binding") != preflight_scene:
        raise ValueError("room_0101 preflight scene binding changed")
    if Path(str(preflight["paths"]["metric_policy"])).resolve() != Path(
        str(paths["original_metric_policy"])
    ).resolve():
        raise ValueError("original metric-policy binding changed")
    if Path(str(failed["paths"]["rollout_maps"])).resolve() != Path(
        str(paths["failed_v91_rollout_maps"])
    ).resolve():
        raise ValueError("sealed v9.1 rollout-map binding changed")
    failed_monitors = {int(row["step"]): row for row in failed["monitor_panels"]}
    reproduced = value.get("reproduced_monitor_panels")
    if not isinstance(reproduced, list) or [int(row.get("step", -1)) for row in reproduced] != [0, 3, 6, 12]:
        raise ValueError("Teacher-v9.7 reproduced monitor inventory changed")
    for row in reproduced:
        _equal_tree(row, failed_monitors[int(row["step"])], "reproduced monitor")

    maps_file = Path(str(paths["rollout_maps"])).resolve()
    arrays = _load_npz(maps_file)
    required_arrays = {
        "xyz",
        "points",
        "instance_ids",
        "category_ids",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
        "all_sittable_gt",
        "snapshot_steps",
        "generation_ids",
        "prompt_ids",
        "seed_table",
        "frozen_base",
        "candidates",
        "frozen_base_normalized",
        "candidates_normalized",
    }
    if set(arrays) != required_arrays:
        raise ValueError("Teacher-v9.7 rollout array inventory changed")
    expected_shapes = {
        "xyz": (8192, 3),
        "points": (8192, 6),
        "instance_ids": (8192,),
        "category_ids": (8192,),
        "verified_object_mask": (3, 8192),
        "verified_positive_mask": (8192,),
        "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,),
        "instance_targets": (3, 8192, 6),
        "all_sittable_gt": (8192, 6),
        "snapshot_steps": (3,),
        "generation_ids": (3,),
        "prompt_ids": (2,),
        "seed_table": (3, 2),
        "frozen_base": (3, 2, 8192, 6),
        "candidates": (3, 3, 2, 8192, 6),
        "frozen_base_normalized": (3, 2, 8192, 6),
        "candidates_normalized": (3, 3, 2, 8192, 6),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError("{} shape changed: {}".format(name, arrays[name].shape))
    if tuple(int(value) for value in arrays["snapshot_steps"].tolist()) != SNAPSHOT_STEPS:
        raise ValueError("snapshot-step array changed")
    if tuple(int(value) for value in arrays["generation_ids"].tolist()) != (0, 1, 2):
        raise ValueError("generation IDs changed")
    if tuple(str(value) for value in arrays["prompt_ids"].tolist()) != PROMPT_IDS:
        raise ValueError("prompt IDs changed")
    if arrays["seed_table"].tolist() != expected_seeds:
        raise ValueError("seed-table array changed")
    for name in (
        "xyz",
        "points",
        "instance_targets",
        "all_sittable_gt",
        "frozen_base",
        "candidates",
        "frozen_base_normalized",
        "candidates_normalized",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "all_sittable_gt", "frozen_base", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if not np.array_equal(arrays["xyz"], arrays["points"][:, :3]):
        raise ValueError("point/XYZ order changed")
    if not np.array_equal(arrays["all_sittable_gt"], arrays["instance_targets"].max(axis=0)):
        raise ValueError("all-sittable GT is not exact three-object max union")
    verified = arrays["verified_object_mask"]
    if verified.dtype != np.bool_ or np.any(verified.sum(axis=1) <= 0) or np.any(verified.sum(axis=0) > 1):
        raise ValueError("verified object masks changed")
    for name in ("verified_positive_mask", "unknown_sittable_mask", "explicit_negative_mask"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(name + " is not boolean")
    if not np.array_equal(arrays["verified_positive_mask"], verified.any(axis=0)):
        raise ValueError("verified-positive union changed")
    if np.any(arrays["unknown_sittable_mask"] & verified.any(axis=0)) or np.any(
        arrays["explicit_negative_mask"]
        & (arrays["unknown_sittable_mask"] | verified.any(axis=0))
    ):
        raise ValueError("semantic masks overlap")

    scene_binding = value.get("scene_binding")
    if not isinstance(scene_binding, Mapping):
        raise ValueError("room_0101 scene binding is absent")
    consensus_file = Path(str(scene_binding["consensus_file"])).resolve()
    points_file = Path(str(scene_binding["points_file"])).resolve()
    if (
        not consensus_file.is_file()
        or sha256_file(consensus_file) != scene_binding["consensus_sha256"]
        or not points_file.is_file()
        or sha256_file(points_file) != scene_binding["points_sha256"]
    ):
        raise ValueError("room_0101 source arrays changed")
    consensus = _load_npz(consensus_file)
    points = _load_npz(points_file)
    source_arrays = {
        "xyz": consensus["xyz"],
        "instance_ids": consensus["instance_ids"],
        "category_ids": consensus["category_ids"],
        "verified_object_mask": consensus["verified_object_mask"],
        "unknown_sittable_mask": consensus["unknown_sittable_mask"],
        "explicit_negative_mask": consensus["explicit_negative_mask"],
        "instance_targets": consensus["instance_affordance"],
        "all_sittable_gt": consensus["all_sittable_affordance"],
        "points": points["points"],
    }
    for name, expected in source_arrays.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError("rollout/source differs: " + name)

    stats = _load_npz(Path(str(paths["stats_file"])).resolve())
    mean = np.asarray(stats["mean"], np.float32).reshape(1, 1, 1, 6)
    std = np.asarray(stats["std"], np.float32).reshape(1, 1, 1, 6)
    expected_base = np.clip(arrays["frozen_base_normalized"] * std + mean, 0.0, 1.0).astype(np.float32)
    if not np.array_equal(expected_base, arrays["frozen_base"]):
        raise ValueError("Base physical denormalization changed")
    candidate_mean = mean.reshape(1, 1, 1, 1, 6)
    candidate_std = std.reshape(1, 1, 1, 1, 6)
    expected_candidates = np.clip(
        arrays["candidates_normalized"] * candidate_std + candidate_mean, 0.0, 1.0
    ).astype(np.float32)
    if not np.array_equal(expected_candidates, arrays["candidates"]):
        raise ValueError("candidate physical denormalization changed")

    base_rows = _rows(arrays, arrays["frozen_base"])
    base_invariance = _invariance(arrays["frozen_base"], arrays["verified_positive_mask"])
    base_pooled = pooled_object_metrics(base_rows)
    _equal_tree(base_pooled, value.get("base_pooled"), "Base pooled metrics")
    monitor_by_step = {int(row["step"]): row for row in reproduced}
    recomputed_rows = []
    for step_index, step in enumerate(SNAPSHOT_STEPS):
        rows = _rows(arrays, arrays["candidates"][step_index])
        presence = [
            [continuous_presence_checks(rows[generation][prompt]) for prompt in range(2)]
            for generation in range(GENERATION_COUNT)
        ]
        invariance = _invariance(
            arrays["candidates"][step_index], arrays["verified_positive_mask"]
        )
        checks = rollout_step_checks(
            base_rows=base_rows,
            candidate_rows=rows,
            candidate_presence=presence,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=invariance,
            base_v5_dense=monitor_by_step[step]["base_v5_dense"],
            candidate_v5_dense=monitor_by_step[step]["candidate_v5_dense"],
        )
        recomputed_rows.append(
            {
                "step": step,
                "base_rows": base_rows,
                "candidate_rows": rows,
                "candidate_presence": presence,
                "base_prompt_invariance": base_invariance,
                "candidate_prompt_invariance": invariance,
                "base_v5_dense": [float(number) for number in monitor_by_step[step]["base_v5_dense"]],
                "candidate_v5_dense": [float(number) for number in monitor_by_step[step]["candidate_v5_dense"]],
                "base_pooled": base_pooled,
                "candidate_pooled": pooled_object_metrics(rows),
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(name for name, passed in checks.items() if not passed),
            }
        )
    _equal_tree(recomputed_rows, value.get("step_rows"), "rollout step rows")
    eligible_order = rank_eligible_steps(recomputed_rows)
    selected = eligible_order[0] if eligible_order else None
    if value.get("eligible_step_order") != eligible_order or value.get("selected_step") != selected:
        raise ValueError("Teacher-v9.7 selected step changed")

    checks = value.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("Teacher-v9.7 top-level checks are absent")
    expected_checks = {
        "sealed_v91_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "exact_first_12_v91_updates_reproduced": True,
        "step_3_6_12_states_kept_in_memory_only": True,
        "paired_base_candidate_initial_and_reverse_noise": True,
        "watch_write_use_identical_randomness": True,
        "base_and_all_candidates_repeat_bitwise": True,
        "rollout_policy_written_before_optimizer": True,
        "exact_topk_is_diagnostic_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_early_step_is_admissible": selected is not None,
    }
    if checks != expected_checks:
        raise ValueError("Teacher-v9.7 top-level checks changed")
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    if value.get("status") != status or value.get("failed_checks") != sorted(
        name for name, passed in expected_checks.items() if not passed
    ):
        raise ValueError("Teacher-v9.7 status arithmetic changed")
    if value.get("authorizes_locked_early_step_confirmation") != (status == "PASS"):
        raise ValueError("Teacher-v9.7 confirmation authorization changed")
    if value.get("authorizes_rollout_state_preflight") != (status == "FAIL"):
        raise ValueError("Teacher-v9.7 rollout-state authorization changed")
    if any(
        value.get(name) is not False
        for name in (
            "authorizes_checkpoint",
            "authorizes_room_0102",
            "authorizes_development_evaluation",
            "authorizes_long_training",
            "authorizes_paper_test",
        )
    ):
        raise ValueError("Teacher-v9.7 over-authorized later work")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or lora.get("rank") != LORA_RANK or float(
        lora.get("alpha", 0.0)
    ) != LORA_ALPHA or int(lora.get("module_count", 0)) != 31:
        raise ValueError("Teacher-v9.7 LoRA contract changed")
    if value.get("optimizer") != {"name": "AdamW", "weight_decay": 0.0}:
        raise ValueError("Teacher-v9.7 optimizer changed")
    binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "loss_response_binding_id": value["loss_response_binding_id"],
            "failed_v91_binding_id": value["failed_v91_binding_id"],
            "rollout_metric_policy_id": value["rollout_metric_policy_id"],
            "snapshot_steps": value["snapshot_steps"],
            "seed_table": value["seed_table"],
            "selected_step": value["selected_step"],
            "rollout_maps_sha256": value["rollout_maps_sha256"],
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("Teacher-v9.7 binding ID changed")
    _finite_tree(value, "Teacher-v9.7 report")
    print("[EARLY_ROLLOUT_K3_{}] Teacher-v9.7 integrity".format(status))
    print("[PASS] exact first-12 v9.1 reproduction and in-memory snapshot contract")
    print("[PASS] 3 steps x K=3 x 2 prompts plus Base and determinism maps recomputed")
    print("[PASS] continuous three-object gates recomputed; exact Top-k remains diagnostic")
    print("[PASS] no checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] selected step:", selected)
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

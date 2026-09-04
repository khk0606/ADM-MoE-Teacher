#!/usr/bin/env python3
"""Deep artifact validator for Teacher-v9.8.1 rollout-state response-6."""

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
    ROLLOUT_POLICY_ID as V97_POLICY_ID,
    SCHEMA as V97_SCHEMA,
    continuous_presence_checks,
    stable_rollout_seeds,
)
from relational_teacher_v98_rollout_state_contract import (  # noqa: E402
    CAPTURE_TIMESTEPS,
    POLICY_ID as V98_POLICY_ID,
)
from relational_teacher_v981_rollout_state_response6_contract import (  # noqa: E402
    DEVELOPMENT_SCENE,
    GENERATION_COUNT,
    HELDOUT_TRAIN_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SEED,
    SELECTED_NAME,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    TRAIN_SCENE,
    V98_SCHEMA,
    canonical_sha256,
    pooled_object_metrics,
    response6_checks,
)


REQUIRED_PATHS = {
    "runner",
    "validator",
    "contract",
    "summarizer",
    "v98_report",
    "v98_maps",
    "v97_summary",
    "v97_maps",
    "preflight_report",
    "metric_policy",
    "response6_policy",
    "dataset_index",
    "source_dataset_index",
    "stats_file",
    "v5_split",
    "v5_evidence_report",
    "original_checkpoint",
    "v5_checkpoint",
    "v98_runner",
    "objective",
    "common_descent",
    "response6_maps",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _validate_bound_tree(value: Mapping[str, object], label: str) -> None:
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError(label + " path binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError(label + " bound file changed: " + str(name))


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(label + " contains NaN/Inf")


def _close(left: object, right: object, label: str) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(label + " keys changed")
        for key in left:
            _close(left[key], right[key], label + "." + str(key))
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(label + " length changed")
        for index, (one, two) in enumerate(zip(left, right)):
            _close(one, two, "{}[{}]".format(label, index))
        return
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " changed")


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


def _rows(arrays: Mapping[str, np.ndarray], values: np.ndarray) -> list:
    return [
        [_metrics(arrays, values[generation, prompt]) for prompt in range(2)]
        for generation in range(3)
    ]


def _invariance(values: np.ndarray, mask: np.ndarray) -> list[float]:
    return [
        float(np.square(values[g, 0, mask] - values[g, 1, mask]).mean())
        for g in range(3)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = json.loads(summary_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.8.1 schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("diffusion_steps") != 500
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != HELDOUT_TRAIN_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("generation_count") != GENERATION_COUNT
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("prompt_text") != {name: PROMPTS[name] for name in PROMPT_IDS}
        or value.get("forward_input_keys") != ["c_pc_feat", "c_pc_xyz", "c_text"]
        or value.get("seed_table") != [list(stable_rollout_seeds(g)) for g in range(3)]
        or value.get("selected_candidate") != SELECTED_NAME
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or float(value.get("selected_radius")) != SELECTED_RADIUS
        or value.get("activation_schedule") != POLICY["adapter_activation"]
        or value.get("new_reverse_diffusion_draws")
        != {
            "base_full_generation1_2": 4,
            "candidate_partial_generation1_2": 4,
            "candidate_partial_determinism_repeat": 1,
        }
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.8.1 protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("Teacher-v9.8.1 path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("Teacher-v9.8.1 hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8.1 bound file changed: " + name)
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8.1 contains forbidden model state")
    if value.get("response6_maps_sha256") != hashes["response6_maps"]:
        raise ValueError("response-6 map hash changed")
    policy = json.loads(Path(str(paths["response6_policy"])).read_text(encoding="utf-8"))
    if policy != POLICY or value.get("policy_id") != POLICY_ID or canonical_sha256(policy) != POLICY_ID:
        raise ValueError("response-6 policy changed")
    if value.get("policy_sha256") != hashes["response6_policy"]:
        raise ValueError("response-6 policy file hash changed")

    v98 = json.loads(Path(str(paths["v98_report"])).read_text(encoding="utf-8"))
    v97 = json.loads(Path(str(paths["v97_summary"])).read_text(encoding="utf-8"))
    preflight = json.loads(Path(str(paths["preflight_report"])).read_text(encoding="utf-8"))
    if (
        v98.get("schema") != V98_SCHEMA
        or v98.get("status") != "PASS"
        or v98.get("selected_candidate") != SELECTED_NAME
        or v98.get("policy_id") != V98_POLICY_ID
        or v98.get("binding_id") != value.get("v98_binding_id")
        or v98.get("authorizes_rollout_state_response6") is not True
        or v98.get("failed_checks")
    ):
        raise ValueError("Teacher-v9.8 authority changed")
    _validate_bound_tree(v98, "Teacher-v9.8")
    if (
        v97.get("schema") != V97_SCHEMA
        or v97.get("status") != "FAIL"
        or v97.get("binding_id") != value.get("v97_binding_id")
        or v97.get("rollout_metric_policy_id") != V97_POLICY_ID
        or v97.get("authorizes_rollout_state_preflight") is not True
    ):
        raise ValueError("Teacher-v9.7 authority changed")
    _validate_bound_tree(v97, "Teacher-v9.7")
    if (
        preflight.get("status") != "PASS"
        or preflight.get("binding_id") != value.get("preflight_binding_id")
        or value.get("scene_binding")
        != {str(row["scene_id"]): row for row in preflight["scene_bindings"]}[TRAIN_SCENE]
    ):
        raise ValueError("Teacher-v9 preflight/scene authority changed")
    if Path(str(v98["paths"]["rollout_maps"])).resolve() != Path(str(paths["v98_maps"])).resolve():
        raise ValueError("Teacher-v9.8 maps path changed")
    if Path(str(v97["paths"]["rollout_maps"])).resolve() != Path(str(paths["v97_maps"])).resolve():
        raise ValueError("Teacher-v9.7 maps path changed")

    arrays = _load_npz(Path(str(paths["response6_maps"])).resolve())
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
        "generation_ids",
        "prompt_ids",
        "seed_table",
        "base_normalized",
        "candidate_normalized",
        "base",
        "candidate",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_prediction",
    }
    if set(arrays) != required_arrays:
        raise ValueError("response-6 array inventory changed")
    shapes = {
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
        "generation_ids": (3,),
        "prompt_ids": (2,),
        "seed_table": (3, 2),
        "base_normalized": (3, 2, 8192, 6),
        "candidate_normalized": (3, 2, 8192, 6),
        "base": (3, 2, 8192, 6),
        "candidate": (3, 2, 8192, 6),
        "v5_target": (3, 8192, 6),
        "base_v5_prediction": (3, 8192, 6),
        "candidate_v5_prediction": (3, 8192, 6),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError("{} shape changed: {}".format(name, arrays[name].shape))
    if arrays["generation_ids"].tolist() != [0, 1, 2]:
        raise ValueError("generation IDs changed")
    if tuple(str(item) for item in arrays["prompt_ids"].tolist()) != PROMPT_IDS:
        raise ValueError("prompt IDs changed")
    if arrays["seed_table"].tolist() != [list(stable_rollout_seeds(g)) for g in range(3)]:
        raise ValueError("seed table changed")
    for name in (
        "xyz",
        "points",
        "instance_targets",
        "all_sittable_gt",
        "base_normalized",
        "candidate_normalized",
        "base",
        "candidate",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_prediction",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "all_sittable_gt", "base", "candidate"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " outside [0,1]")
    if not np.array_equal(arrays["xyz"], arrays["points"][:, :3]):
        raise ValueError("point/XYZ order changed")
    if not np.array_equal(arrays["all_sittable_gt"], arrays["instance_targets"].max(axis=0)):
        raise ValueError("all-sittable GT union changed")
    if not np.array_equal(arrays["verified_positive_mask"], arrays["verified_object_mask"].any(axis=0)):
        raise ValueError("verified-positive union changed")
    if arrays["verified_object_mask"].dtype != np.bool_:
        raise ValueError("verified object mask is not boolean")
    for name in ("verified_positive_mask", "unknown_sittable_mask", "explicit_negative_mask"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(name + " is not boolean")
    if np.any(arrays["unknown_sittable_mask"] & arrays["verified_positive_mask"]) or np.any(
        arrays["explicit_negative_mask"]
        & (arrays["unknown_sittable_mask"] | arrays["verified_positive_mask"])
    ):
        raise ValueError("semantic masks overlap")

    scene_binding = value["scene_binding"]
    consensus_file = Path(str(scene_binding["consensus_file"])).resolve()
    points_file = Path(str(scene_binding["points_file"])).resolve()
    if (
        not consensus_file.is_file()
        or sha256_file(consensus_file) != scene_binding["consensus_sha256"]
        or not points_file.is_file()
        or sha256_file(points_file) != scene_binding["points_sha256"]
    ):
        raise ValueError("room_0101 source files changed")
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
            raise ValueError("response/source differs: " + name)

    v97_arrays = _load_npz(Path(str(paths["v97_maps"])).resolve())
    v98_arrays = _load_npz(Path(str(paths["v98_maps"])).resolve())
    v97_base = np.asarray(v97_arrays["frozen_base_normalized"], np.float32)
    timestep_index = list(CAPTURE_TIMESTEPS).index(SELECTED_TIMESTEP)
    radius_index = list(v98_arrays["step_radii"].tolist()).index(SELECTED_RADIUS)
    if not np.array_equal(arrays["base_normalized"], v97_base):
        raise ValueError("response Base does not exactly reuse v9.7 K=3")
    if not np.array_equal(
        arrays["candidate_normalized"][0],
        v98_arrays["candidates_normalized"][timestep_index, radius_index],
    ):
        raise ValueError("generation-0 candidate does not exactly reuse v9.8")
    if not np.array_equal(arrays["v5_target"], v98_arrays["v5_target"]):
        raise ValueError("v5 target changed")
    if not np.array_equal(arrays["base_v5_prediction"], v98_arrays["base_v5_prediction"]):
        raise ValueError("Base v5 prediction changed")
    if not np.array_equal(
        arrays["candidate_v5_prediction"],
        v98_arrays["candidate_v5_predictions"][timestep_index, radius_index],
    ):
        raise ValueError("candidate v5 prediction changed")

    stats = _load_npz(Path(str(paths["stats_file"])).resolve())
    mean = np.asarray(stats["mean"], np.float32).reshape(1, 1, 1, 6)
    std = np.asarray(stats["std"], np.float32).reshape(1, 1, 1, 6)
    expected_base = np.clip(arrays["base_normalized"] * std + mean, 0.0, 1.0).astype(np.float32)
    expected_candidate = np.clip(
        arrays["candidate_normalized"] * std + mean, 0.0, 1.0
    ).astype(np.float32)
    if not np.array_equal(arrays["base"], expected_base):
        raise ValueError("Base physical conversion changed")
    if not np.array_equal(arrays["candidate"], expected_candidate):
        raise ValueError("candidate physical conversion changed")

    base_rows = _rows(arrays, arrays["base"])
    candidate_rows = _rows(arrays, arrays["candidate"])
    base_pooled = pooled_object_metrics(base_rows)
    candidate_pooled = pooled_object_metrics(candidate_rows)
    base_invariance = _invariance(arrays["base"], arrays["verified_positive_mask"])
    candidate_invariance = _invariance(arrays["candidate"], arrays["verified_positive_mask"])
    base_v5_dense = np.square(
        arrays["base_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    candidate_v5_dense = np.square(
        arrays["candidate_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    maximum_delta = float(np.abs(arrays["candidate"] - arrays["base"]).max())
    direction = value.get("direction")
    selected_v98_direction = [
        row for row in v98["direction_rows"] if row["capture_timestep"] == SELECTED_TIMESTEP
    ][0]
    _close(direction, selected_v98_direction, "selected direction")
    checks = response6_checks(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        base_prompt_invariance=base_invariance,
        candidate_prompt_invariance=candidate_invariance,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
        directional_derivatives=direction["directional_derivatives"],
        maximum_map_delta=maximum_delta,
    )
    presence = [
        [continuous_presence_checks(candidate_rows[g][p]) for p in range(2)]
        for g in range(3)
    ]
    for left, right, label in (
        (base_rows, value.get("base_rows"), "Base rows"),
        (candidate_rows, value.get("candidate_rows"), "candidate rows"),
        (presence, value.get("candidate_presence"), "presence"),
        (base_pooled, value.get("base_pooled"), "Base pooled"),
        (candidate_pooled, value.get("candidate_pooled"), "candidate pooled"),
        (base_invariance, value.get("base_prompt_invariance"), "Base invariance"),
        (candidate_invariance, value.get("candidate_prompt_invariance"), "candidate invariance"),
        (base_v5_dense, value.get("base_v5_dense"), "Base v5"),
        (candidate_v5_dense, value.get("candidate_v5_dense"), "candidate v5"),
        (maximum_delta, value.get("maximum_final_map_delta"), "map delta"),
        (checks, value.get("response_checks"), "response checks"),
    ):
        _close(left, right, label)

    expected_top_checks = {
        "sealed_v98_pass_and_selected_candidate_bound": True,
        "selected_direction_geometry_and_hash_reproduced": True,
        "generation0_maps_byte_exactly_reused": True,
        "generation1_2_base_maps_reproduce_v97": True,
        "generation1_2_use_paired_t50_rng_resumes": True,
        "selected_response_repeat_is_bitwise_exact": True,
        "response6_policy_locked_before_lora": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "selected_response_is_k3_admissible": all(checks.values()),
    }
    if value.get("checks") != expected_top_checks:
        raise ValueError("top-level check arithmetic changed")
    status = "PASS" if all(expected_top_checks.values()) else "FAIL"
    if value.get("status") != status or value.get("failed_checks") != sorted(
        name for name, passed in expected_top_checks.items() if not passed
    ):
        raise ValueError("Teacher-v9.8.1 status changed")
    if value.get("authorizes_rollout_state_calibration6") != (status == "PASS"):
        raise ValueError("calibration authorization changed")
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
        raise ValueError("Teacher-v9.8.1 over-authorized later work")
    lora = value.get("lora")
    if (
        not isinstance(lora, Mapping)
        or lora.get("rank") != LORA_RANK
        or float(lora.get("alpha", 0.0)) != LORA_ALPHA
        or int(lora.get("module_count", 0)) != 31
    ):
        raise ValueError("LoRA metadata changed")
    binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "v97_binding_id": value["v97_binding_id"],
            "v98_binding_id": value["v98_binding_id"],
            "policy_id": value["policy_id"],
            "selected_candidate": value["selected_candidate"],
            "direction_sha256": direction["direction_sha256"],
            "response6_maps_sha256": value["response6_maps_sha256"],
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("Teacher-v9.8.1 binding ID changed")
    _finite_tree(value, "Teacher-v9.8.1 report")
    print("[ROLLOUT_STATE_RESPONSE6_{}] Teacher-v9.8.1 integrity".format(status))
    print("[PASS] selected direction, generation-0 and v5 responses exactly reuse v9.8")
    print("[PASS] v9.7 K=3 Base, response maps and continuous metrics recomputed")
    print("[PASS] response policy, retention gates and authorization recomputed")
    print("[PASS] no optimizer/checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

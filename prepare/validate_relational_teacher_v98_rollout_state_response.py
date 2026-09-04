#!/usr/bin/env python3
"""Deep artifact validator for Teacher-v9.8 rollout-state response."""

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
from relational_teacher_v97_early_rollout_k3_contract import ROLLOUT_POLICY_ID as V97_POLICY_ID  # noqa: E402
from relational_teacher_v97_early_rollout_k3_contract import stable_rollout_seeds as v97_rollout_seeds  # noqa: E402
from relational_teacher_v98_rollout_state_contract import (  # noqa: E402
    CAPTURE_TIMESTEPS,
    DEVELOPMENT_SCENE,
    FAILED_V97_SCHEMA,
    HELDOUT_TRAIN_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    OBJECTS,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    SCHEMA,
    SEED,
    STEP_RADII,
    TRAIN_SCENE,
    canonical_sha256,
    design_seeds,
    pooled,
    rank_candidates,
    response_checks,
)


REQUIRED_PATHS = {
    "runner",
    "validator",
    "contract",
    "summarizer",
    "failed_v97_summary",
    "failed_v97_maps",
    "preflight_report",
    "metric_policy",
    "rollout_state_policy",
    "dataset_index",
    "source_dataset_index",
    "stats_file",
    "v5_split",
    "v5_evidence_report",
    "original_checkpoint",
    "v5_checkpoint",
    "objective",
    "common_descent",
    "rollout_maps",
}


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


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


def _pair_metrics(arrays: Mapping[str, np.ndarray], values: np.ndarray) -> list:
    if values.shape != (2, 8192, 6):
        raise ValueError("watch/write tensor shape changed")
    return [_metrics(arrays, values[index]) for index in range(2)]


def _invariance(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.square(values[0, mask] - values[1, mask]).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = json.loads(report_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.8 schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("diffusion_steps") != 500
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != HELDOUT_TRAIN_SCENE
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("prompt_text") != {name: PROMPTS[name] for name in PROMPT_IDS}
        or value.get("forward_input_keys") != ["c_pc_feat", "c_pc_xyz", "c_text"]
        or value.get("capture_timesteps") != list(CAPTURE_TIMESTEPS)
        or value.get("step_radii") != list(STEP_RADII)
        or value.get("design_seeds") != list(design_seeds())
        or value.get("audit_seeds") != list(v97_rollout_seeds(0))
        or value.get("trajectory_accounting")
        != {"base_full_draws": 4, "base_exact_partial_resumes": 6, "candidate_partial_resumes": 18}
        or value.get("serialized_model_state") is not False
        or value.get("selection_policy") != POLICY["selection_order"]
    ):
        raise ValueError("Teacher-v9.8 protocol changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("Teacher-v9.8 path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("Teacher-v9.8 hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v9.8 bound file changed: " + name)
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.8 contains forbidden model state")
    if value.get("rollout_maps_sha256") != hashes["rollout_maps"]:
        raise ValueError("rollout-map hash field changed")
    policy = json.loads(Path(str(paths["rollout_state_policy"])).read_text(encoding="utf-8"))
    if policy != POLICY or value.get("policy_id") != POLICY_ID or canonical_sha256(policy) != POLICY_ID:
        raise ValueError("rollout-state policy changed")
    if value.get("policy_sha256") != hashes["rollout_state_policy"]:
        raise ValueError("rollout-state policy file hash changed")

    failed = json.loads(Path(str(paths["failed_v97_summary"])).read_text(encoding="utf-8"))
    preflight = json.loads(Path(str(paths["preflight_report"])).read_text(encoding="utf-8"))
    if (
        failed.get("schema") != FAILED_V97_SCHEMA
        or failed.get("status") != "FAIL"
        or failed.get("selected_step") is not None
        or failed.get("failed_checks") != ["at_least_one_early_step_is_admissible"]
        or failed.get("authorizes_rollout_state_preflight") is not True
        or failed.get("rollout_metric_policy_id") != V97_POLICY_ID
        or failed.get("binding_id") != value.get("failed_v97_binding_id")
        or failed.get("preflight_binding_id") != value.get("preflight_binding_id")
        or preflight.get("binding_id") != value.get("preflight_binding_id")
        or preflight.get("status") != "PASS"
    ):
        raise ValueError("Teacher-v9.7/preflight authority changed")
    if Path(str(failed["paths"]["rollout_maps"])).resolve() != Path(
        str(paths["failed_v97_maps"])
    ).resolve():
        raise ValueError("Teacher-v9.7 map authority changed")
    scene_binding = {
        str(row["scene_id"]): row for row in preflight["scene_bindings"]
    }[TRAIN_SCENE]
    if value.get("scene_binding") != scene_binding:
        raise ValueError("room_0101 scene binding changed")

    arrays = _load_npz(Path(str(paths["rollout_maps"])).resolve())
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
        "capture_timesteps",
        "step_radii",
        "prompt_ids",
        "design_states",
        "audit_states",
        "audit_base_normalized",
        "audit_base",
        "candidates_normalized",
        "candidates",
        "direct_before",
        "direct_after",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    }
    if set(arrays) != required_arrays:
        raise ValueError("Teacher-v9.8 array inventory changed")
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
        "capture_timesteps": (3,),
        "step_radii": (3,),
        "prompt_ids": (2,),
        "design_states": (3, 2, 8192, 6),
        "audit_states": (3, 2, 8192, 6),
        "audit_base_normalized": (2, 8192, 6),
        "audit_base": (2, 8192, 6),
        "candidates_normalized": (3, 3, 2, 8192, 6),
        "candidates": (3, 3, 2, 8192, 6),
        "direct_before": (3, 2, 8192, 6),
        "direct_after": (3, 3, 2, 8192, 6),
        "v5_target": (3, 8192, 6),
        "base_v5_prediction": (3, 8192, 6),
        "candidate_v5_predictions": (3, 3, 3, 8192, 6),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError("{} shape changed: {}".format(name, arrays[name].shape))
    if tuple(arrays["capture_timesteps"].tolist()) != CAPTURE_TIMESTEPS:
        raise ValueError("capture-timestep array changed")
    if not np.array_equal(arrays["step_radii"], np.asarray(STEP_RADII, np.float64)):
        raise ValueError("step-radius array changed")
    if tuple(str(item) for item in arrays["prompt_ids"].tolist()) != PROMPT_IDS:
        raise ValueError("prompt-ID array changed")
    for name in (
        "xyz",
        "points",
        "instance_targets",
        "all_sittable_gt",
        "design_states",
        "audit_states",
        "audit_base_normalized",
        "audit_base",
        "candidates_normalized",
        "candidates",
        "direct_before",
        "direct_after",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "all_sittable_gt", "audit_base", "candidates", "direct_before", "direct_after"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if not np.array_equal(arrays["xyz"], arrays["points"][:, :3]):
        raise ValueError("point/XYZ order changed")
    if not np.array_equal(arrays["all_sittable_gt"], arrays["instance_targets"].max(axis=0)):
        raise ValueError("all-sittable GT union changed")

    consensus_file = Path(str(scene_binding["consensus_file"])).resolve()
    points_file = Path(str(scene_binding["points_file"])).resolve()
    if sha256_file(consensus_file) != scene_binding["consensus_sha256"] or sha256_file(
        points_file
    ) != scene_binding["points_sha256"]:
        raise ValueError("room_0101 source file changed")
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
    if not np.array_equal(
        arrays["verified_positive_mask"], arrays["verified_object_mask"].any(axis=0)
    ):
        raise ValueError("verified-positive mask changed")

    with np.load(Path(str(paths["failed_v97_maps"])).resolve(), allow_pickle=False) as payload:
        v97_base = np.asarray(payload["frozen_base_normalized"], np.float32)
    if not np.array_equal(arrays["audit_base_normalized"], v97_base[0]):
        raise ValueError("audit Base does not exactly reproduce v9.7 generation 0")

    stats = _load_npz(Path(str(paths["stats_file"])).resolve())
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    expected_base = np.clip(
        arrays["audit_base_normalized"] * std.reshape(1, 1, 6)
        + mean.reshape(1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    if not np.array_equal(expected_base, arrays["audit_base"]):
        raise ValueError("audit Base physical conversion changed")
    expected_candidates = np.clip(
        arrays["candidates_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    if not np.array_equal(expected_candidates, arrays["candidates"]):
        raise ValueError("candidate physical conversion changed")

    base_rows = _pair_metrics(arrays, arrays["audit_base"])
    base_pooled = pooled(base_rows)
    base_invariance = _invariance(arrays["audit_base"], arrays["verified_positive_mask"])
    _close(base_rows, value.get("base_rows"), "Base rows")
    _close(base_pooled, value.get("base_pooled"), "Base pooled")
    _close(base_invariance, value.get("base_prompt_invariance"), "Base invariance")
    base_v5_dense = np.square(
        arrays["base_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    _close(base_v5_dense, value.get("base_v5_dense"), "Base v5 dense")
    directions = value.get("direction_rows")
    if not isinstance(directions, list) or len(directions) != 3:
        raise ValueError("direction inventory changed")
    direction_by_timestep = {}
    for expected_timestep, row in zip(CAPTURE_TIMESTEPS, directions):
        if row.get("capture_timestep") != expected_timestep or row.get("task_order") != POLICY["object_tasks"]:
            raise ValueError("direction task/timestep changed")
        gram = np.asarray(row["gram"], np.float64)
        weights = np.asarray(row["weights"], np.float64)
        if gram.shape != (6, 6) or weights.shape != (6,):
            raise ValueError("direction Gram/weight shape changed")
        if not np.isfinite(gram).all() or not np.allclose(gram, gram.T, atol=1e-8):
            raise ValueError("direction Gram matrix is invalid")
        if float(np.linalg.eigvalsh(gram).min()) < -1e-6:
            raise ValueError("direction Gram matrix is not positive semidefinite")
        if np.any(weights < 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
            raise ValueError("common-descent weights are invalid")
        norm = math.sqrt(max(float(weights @ gram @ weights), 0.0))
        if norm <= 0.0 or not math.isclose(
            norm, float(row["direction_norm_before_unit"]), rel_tol=1e-6, abs_tol=1e-8
        ):
            raise ValueError("common-descent direction norm changed")
        derivatives = (gram @ weights / norm).tolist()
        _close(derivatives, row["directional_derivatives"], "direction derivatives")
        if min(derivatives) < float(POLICY["selection"]["minimum_directional_derivative"]):
            raise ValueError("direction is not common descent")
        direction_by_timestep[expected_timestep] = row

    expected_candidate_order = [
        (timestep, radius)
        for timestep in CAPTURE_TIMESTEPS
        for radius in STEP_RADII
    ]
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 9:
        raise ValueError("candidate inventory changed")
    recomputed = []
    for index, (timestep, radius) in enumerate(expected_candidate_order):
        timestep_index = CAPTURE_TIMESTEPS.index(timestep)
        radius_index = STEP_RADII.index(radius)
        source = candidates[index]
        expected_name = "t{}_radius_{}".format(timestep, str(radius).replace("0.", "0p"))
        if (
            source.get("name") != expected_name
            or source.get("capture_timestep") != timestep
            or float(source.get("radius")) != radius
            or source.get("direction_sha256")
            != direction_by_timestep[timestep]["direction_sha256"]
        ):
            raise ValueError("candidate grid/order changed")
        rows = _pair_metrics(arrays, arrays["candidates"][timestep_index, radius_index])
        invariance = _invariance(
            arrays["candidates"][timestep_index, radius_index],
            arrays["verified_positive_mask"],
        )
        maximum_delta = float(
            np.abs(
                arrays["candidates"][timestep_index, radius_index]
                - arrays["audit_base"]
            ).max()
        )
        checks = response_checks(
            base_rows=base_rows,
            candidate_rows=rows,
            base_prompt_invariance=base_invariance,
            candidate_prompt_invariance=invariance,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=np.square(
                arrays["candidate_v5_predictions"][timestep_index, radius_index]
                - arrays["v5_target"]
            ).mean(axis=(1, 2)).tolist(),
            directional_derivatives=direction_by_timestep[timestep]["directional_derivatives"],
            maximum_map_delta=maximum_delta,
        )
        row = {
            "name": expected_name,
            "capture_timestep": timestep,
            "radius": radius,
            "direction_sha256": source["direction_sha256"],
            "directional_derivatives": direction_by_timestep[timestep]["directional_derivatives"],
            "base_rows": base_rows,
            "candidate_rows": rows,
            "base_pooled": base_pooled,
            "candidate_pooled": pooled(rows),
            "base_prompt_invariance": base_invariance,
            "candidate_prompt_invariance": invariance,
            "base_v5_dense": base_v5_dense,
            "candidate_v5_dense": np.square(
                arrays["candidate_v5_predictions"][timestep_index, radius_index]
                - arrays["v5_target"]
            ).mean(axis=(1, 2)).tolist(),
            "maximum_final_map_delta": maximum_delta,
            "direct_before_rows": _pair_metrics(arrays, arrays["direct_before"][timestep_index]),
            "direct_after_rows": _pair_metrics(
                arrays, arrays["direct_after"][timestep_index, radius_index]
            ),
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        recomputed.append(row)
    _close(recomputed, candidates, "candidate rows")
    eligible_order = rank_candidates(recomputed)
    selected = eligible_order[0] if eligible_order else None
    if value.get("eligible_selection_order") != eligible_order or value.get("selected_candidate") != selected:
        raise ValueError("Teacher-v9.8 selection changed")

    expected_checks = {
        "sealed_v97_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "rollout_policy_locked_before_lora": True,
        "new_design_seed_disjoint_from_v97_audit_seed": design_seeds() != v97_rollout_seeds(0),
        "audit_base_exactly_reproduces_v97_generation0": True,
        "all_captured_rng_resumes_bitwise_exact": True,
        "six_object_prompt_tasks_have_common_descent": True,
        "candidate_decision_uses_resumed_final_maps": True,
        "exact_topk_is_diagnostic_only": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_room_0101_arrays_loaded": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_rollout_state_response_is_admissible": selected is not None,
    }
    if value.get("checks") != expected_checks:
        raise ValueError("Teacher-v9.8 top-level checks changed")
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    if value.get("status") != status or value.get("failed_checks") != sorted(
        name for name, passed in expected_checks.items() if not passed
    ):
        raise ValueError("Teacher-v9.8 status arithmetic changed")
    if value.get("authorizes_rollout_state_response6") != (status == "PASS"):
        raise ValueError("Teacher-v9.8 response authorization changed")
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
        raise ValueError("Teacher-v9.8 over-authorized later work")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or lora.get("rank") != LORA_RANK or float(
        lora.get("alpha", 0.0)
    ) != LORA_ALPHA or int(lora.get("module_count", 0)) != 31:
        raise ValueError("Teacher-v9.8 LoRA contract changed")
    binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "failed_v97_binding_id": value["failed_v97_binding_id"],
            "policy_id": value["policy_id"],
            "design_seeds": value["design_seeds"],
            "audit_seeds": value["audit_seeds"],
            "selected_candidate": value["selected_candidate"],
            "rollout_maps_sha256": value["rollout_maps_sha256"],
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("Teacher-v9.8 binding ID changed")
    _finite_tree(value, "Teacher-v9.8 report")
    print("[ROLLOUT_STATE_PREFLIGHT_{}] Teacher-v9.8 integrity".format(status))
    print("[PASS] v9.7 failure, generation-0 Base and source bindings recomputed")
    print("[PASS] three rollout timesteps x three radii final maps recomputed")
    print("[PASS] common-descent geometry, retention and selection recomputed")
    print("[PASS] no checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

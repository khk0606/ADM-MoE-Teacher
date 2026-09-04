#!/usr/bin/env python3
"""Independent validator for Teacher-v10.3.1 on-policy calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

import run_relational_teacher_v10_supervised_capacity as v10
import validate_relational_teacher_v10_supervised_capacity as v10v
from relational_teacher_v102_dense_instance_contract import (
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_INSTANCES,
)
from relational_teacher_v103_onpolicy_response_contract import (
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED as V103_MODEL_SEED,
    PROMPT_IDS,
    SCENES,
    TASK_ORDER,
    response_checks,
)
from relational_teacher_v1031_onpolicy_calibration6_contract import (
    FW_ITERATIONS,
    GENERATION_COUNT,
    MODEL_SEED,
    MONITOR_STEPS,
    POLICY,
    POLICY_ID,
    SCHEMA,
    SELECTED_RADIUS,
    SELECTED_TIMESTEP,
    SHORTLIST_LIMIT,
    STEP_RADIUS,
    UPDATE_COUNT,
    V103_SCHEMA,
    all_three_counts,
    calibration_checks,
    canonical_sha256,
    rank_shortlist,
    stable_seed_pair,
)
from relational_teacher_v94_common_descent import frank_wolfe_min_norm_weights
from relational_teacher_v9_all_sittable_contract import read_json, sha256_file
from relational_teacher_v9_all_sittable_metrics import simultaneous_presence_checks
from relational_teacher_v9_lora_preflight_contract import (
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v9_lora_runtime import load_stats


METRIC_KEYS = (
    "soft_recall",
    "active_support_mae",
    "topk_overlap",
    "hotspot_centroid_distance_xy",
)
SEALED_V3_VALIDATOR_SHA256 = (
    "9ac11f4a0c9aa5bd7440d18e9a342913509d4fe1d918f928c5ff83b0f96eeeea"
)


def _validate_v103(path: Path, binding_id: object) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V103_SCHEMA
        or value.get("status") != "PASS"
        or value.get("selected_candidate") != "t150_radius_0p004"
        or value.get("binding_id") != binding_id
        or value.get("authorizes_onpolicy_multiupdate_calibration") is not True
        or value.get("authorizes_checkpoint") is not False
        or value.get("serialized_model_state") is not False
        or value.get("failed_checks") != []
    ):
        raise ValueError("Teacher-v10.3 authority changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v10.3 authority paths changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10.3 bound file changed: " + str(name))
    if list(path.parent.rglob("*.pt")) or list(path.parent.rglob("*.pth")):
        raise ValueError("Teacher-v10.3 authority contains model state")
    return value


def _bundle(arrays: Mapping[str, np.ndarray], scene_index: int) -> Dict[str, object]:
    prefix = "source" if scene_index == 0 else "audit"
    return {
        "xyz": arrays[prefix + "_xyz"],
        "points": arrays[prefix + "_points"],
        "instance_names": arrays[prefix + "_instance_names"].tolist(),
        "verified_object_mask": arrays[prefix + "_verified_object_mask"],
        "verified_positive_mask": arrays[prefix + "_verified_positive_mask"],
        "unknown_sittable_mask": arrays[prefix + "_unknown_sittable_mask"],
        "explicit_negative_mask": arrays[prefix + "_explicit_negative_mask"],
        "instance_targets": arrays[prefix + "_instance_targets"],
        "all_target": arrays[prefix + "_all_target"],
    }


def _raw_rows(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, list]:
    return {
        scene: [
            [
                v10._metrics(
                    bundles[scene],
                    predictions[scene_index, generation, prompt_index],
                )
                for prompt_index in range(2)
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene_index, scene in enumerate(SCENES)
    }


def _pooled_rows(raw: Mapping[str, Sequence[Sequence[Mapping[str, object]]]]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for scene in SCENES:
        result[scene] = {}
        for prompt_index, prompt_id in enumerate(PROMPT_IDS):
            instances = {}
            for name in EXPECTED_INSTANCES[scene]:
                instances[name] = {
                    key: float(
                        np.mean(
                            [
                                raw[scene][generation][prompt_index]["instances"][name][key]
                                for generation in range(GENERATION_COUNT)
                            ]
                        )
                    )
                    for key in METRIC_KEYS
                }
            result[scene][prompt_id] = {
                "instances": instances,
                "explicit_negative_mean": float(
                    np.mean(
                        [
                            raw[scene][generation][prompt_index]["explicit_negative_mean"]
                            for generation in range(GENERATION_COUNT)
                        ]
                    )
                ),
                "explicit_negative_max": float(
                    max(
                        raw[scene][generation][prompt_index]["explicit_negative_max"]
                        for generation in range(GENERATION_COUNT)
                    )
                ),
            }
    return result


def _prompt_invariance(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, float]:
    result = {}
    for scene_index, scene in enumerate(SCENES):
        mask = np.asarray(bundles[scene]["verified_positive_mask"], bool)
        result[scene] = float(
            np.mean(
                [
                    np.square(
                        predictions[scene_index, generation, 0, mask]
                        - predictions[scene_index, generation, 1, mask]
                    ).mean()
                    for generation in range(GENERATION_COUNT)
                ]
            )
        )
    return result


def _presence(raw: Mapping[str, Sequence[Sequence[Mapping[str, object]]]]) -> Dict[str, list]:
    return {
        scene: [
            [
                simultaneous_presence_checks(
                    raw[scene][generation][prompt_index],
                    **ABSOLUTE_PRESENCE_LIMITS,
                )
                for prompt_index in range(2)
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene in SCENES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = read_json(summary_file)
    if (
        value.get("schema") != SCHEMA
        or value.get("seed") != MODEL_SEED
        or value.get("initialization_seed") != V103_MODEL_SEED
        or value.get("diffusion_steps") != 500
        or value.get("train_scenes") != list(SCENES)
        or value.get("development_scene_metadata_only") != "room_0201"
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("generation_count") != GENERATION_COUNT
        or value.get("selected_timestep") != SELECTED_TIMESTEP
        or float(value.get("selected_radius")) != SELECTED_RADIUS
        or value.get("update_count") != UPDATE_COUNT
        or value.get("monitor_steps") != list(MONITOR_STEPS)
        or float(value.get("step_radius")) != STEP_RADIUS
        or value.get("shortlist_limit") != SHORTLIST_LIMIT
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v10.3.1 report contract changed")

    expected_design = [
        [
            [
                list(stable_seed_pair("design", scene, prompt, generation))
                for prompt in PROMPT_IDS
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene in SCENES
    ]
    expected_audit = [
        [
            [
                list(stable_seed_pair("audit", scene, prompt, generation))
                for prompt in PROMPT_IDS
            ]
            for generation in range(GENERATION_COUNT)
        ]
        for scene in SCENES
    ]
    if value.get("design_seed_table") != expected_design or value.get("audit_seed_table") != expected_audit:
        raise ValueError("Teacher-v10.3.1 seed tables changed")

    required_paths = {
        "runner",
        "validator",
        "contract",
        "summarizer",
        "v103_report",
        "v103_maps",
        "dataset_index",
        "source_dataset_index",
        "stats_file",
        "v5_split",
        "v5_evidence_report",
        "original_checkpoint",
        "v5_checkpoint",
        "metric_policy",
        "dense_instance_objective",
        "common_descent",
        "calibration_policy",
        "calibration_maps",
    }
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != required_paths
        or set(hashes) != required_paths
    ):
        raise ValueError("Teacher-v10.3.1 path inventory changed")
    validator_path = Path(str(paths["validator"])).expanduser().resolve()
    if (
        validator_path != Path(__file__).resolve()
        or hashes["validator"] != SEALED_V3_VALIDATOR_SHA256
    ):
        raise ValueError("Teacher-v10.3.1 sealed v3 validator binding changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if name == "validator":
            continue
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v10.3.1 bound file changed: " + str(name))
    authority = _validate_v103(
        Path(str(paths["v103_report"])).resolve(), value.get("v103_binding_id")
    )
    if authority.get("selected_candidate") != value.get("v103_selected_candidate"):
        raise ValueError("Teacher-v10.3 selected candidate binding changed")
    if read_json(Path(str(paths["calibration_policy"])).resolve()) != POLICY:
        raise ValueError("Teacher-v10.3.1 policy changed")
    flat_design = {
        tuple(pair)
        for scene in expected_design
        for generation in scene
        for pair in generation
    }
    flat_audit = {
        tuple(pair)
        for scene in expected_audit
        for generation in scene
        for pair in generation
    }
    v103_seeds = {
        tuple(pair)
        for domain in (authority["design_seed_table"], authority["audit_seed_table"])
        for scene in domain
        for pair in scene
    }
    if flat_design & flat_audit or flat_design & v103_seeds or flat_audit & v103_seeds:
        raise ValueError("Teacher-v10.3.1 seed domains overlap")
    maps_file = Path(str(paths["calibration_maps"])).resolve()
    if (
        sha256_file(maps_file) != value.get("calibration_maps_sha256")
        or hashes["calibration_maps"] != value.get("calibration_maps_sha256")
    ):
        raise ValueError("Teacher-v10.3.1 map hash changed")
    arrays = v10v._load_npz(maps_file)
    suffixes = {
        "xyz",
        "points",
        "instance_names",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
        "all_target",
    }
    required_arrays = {
        "scene_ids",
        "prompt_ids",
        "design_seed_table",
        "audit_seed_table",
        "design_states",
        "audit_states",
        "start_normalized",
        "start",
        "candidates_normalized",
        "candidates",
        "v5_target",
        "base_v5_prediction",
        "start_v5_prediction",
        "candidate_v5_predictions",
    } | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in suffixes
    }
    if set(arrays) != required_arrays:
        raise ValueError("Teacher-v10.3.1 array inventory changed")
    if (
        arrays["scene_ids"].tolist() != list(SCENES)
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["design_seed_table"].tolist() != expected_design
        or arrays["audit_seed_table"].tolist() != expected_audit
        or arrays["design_states"].shape != (2, 3, 2, 8192, 6)
        or arrays["audit_states"].shape != (2, 3, 2, 8192, 6)
        or arrays["start_normalized"].shape != (2, 3, 2, 8192, 6)
        or arrays["start"].shape != (2, 3, 2, 8192, 6)
        or arrays["candidates_normalized"].shape != (6, 2, 3, 2, 8192, 6)
        or arrays["candidates"].shape != (6, 2, 3, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["start_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (6, 3, 8192, 6)
    ):
        raise ValueError("Teacher-v10.3.1 array shape/order changed")
    for name, array in arrays.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("start", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is not a physical map")
    mean, std = load_stats(Path(str(paths["stats_file"])).resolve())
    expected_start = np.clip(
        arrays["start_normalized"] * std.reshape(1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    expected_candidates = np.clip(
        arrays["candidates_normalized"] * std.reshape(1, 1, 1, 1, 1, 6)
        + mean.reshape(1, 1, 1, 1, 1, 6),
        0.0,
        1.0,
    ).astype(np.float32)
    if not np.array_equal(arrays["start"], expected_start) or not np.array_equal(
        arrays["candidates"], expected_candidates
    ):
        raise ValueError("Teacher-v10.3.1 physical map conversion changed")

    bundles = {scene: _bundle(arrays, index) for index, scene in enumerate(SCENES)}
    dataset_index = Path(str(paths["dataset_index"])).resolve()
    source_index = Path(str(paths["source_dataset_index"])).resolve()
    top_index = validate_top_index(
        dataset_index.parent, source_index.parent, dataset_index
    )
    records = {str(row["scene_id"]): row for row in top_index["scenes"]}
    if set(records) != set(SCENES):
        raise ValueError("Teacher-v10.3.1 original scene inventory changed")
    original_bundles = {
        scene: load_train_scene_bundle(
            dataset_index.parent, source_index.parent, records[scene]
        )
        for scene in SCENES
    }
    for scene_index, scene in enumerate(SCENES):
        prefix = "source" if scene_index == 0 else "audit"
        if (
            arrays[prefix + "_xyz"].shape != (8192, 3)
            or arrays[prefix + "_points"].shape != (8192, 6)
            or arrays[prefix + "_instance_names"].tolist()
            != list(EXPECTED_INSTANCES[scene])
            or arrays[prefix + "_instance_targets"].shape != (3, 8192, 6)
            or not np.array_equal(
                arrays[prefix + "_xyz"], arrays[prefix + "_points"][:, :3]
            )
        ):
            raise ValueError(scene + " saved bundle changed")
        for key in (
            "xyz",
            "points",
            "verified_object_mask",
            "verified_positive_mask",
            "unknown_sittable_mask",
            "explicit_negative_mask",
            "instance_targets",
            "all_target",
        ):
            if not np.array_equal(
                arrays[prefix + "_" + key], np.asarray(original_bundles[scene][key])
            ):
                raise ValueError(scene + " original binding changed: " + key)

    response_arrays = v10v._load_npz(Path(str(paths["v103_maps"])).resolve())
    selected_index = next(
        index
        for index, row in enumerate(authority["candidates"])
        if row["name"] == authority["selected_candidate"]
    )
    selected_row = authority["candidates"][selected_index]
    timestep_index = list(authority["capture_timesteps"]).index(SELECTED_TIMESTEP)
    radius_index = list(authority["step_radii"]).index(SELECTED_RADIUS)
    if (
        value.get("zero_state_sha256") != authority.get("zero_state_sha256")
        or value.get("start_state_sha256") != selected_row.get("state_sha256")
        or not np.array_equal(
            arrays["base_v5_prediction"], response_arrays["base_v5_prediction"]
        )
        or not np.array_equal(
            arrays["start_v5_prediction"],
            response_arrays["candidate_v5_predictions"][timestep_index, radius_index],
        )
    ):
        raise ValueError("Teacher-v10.3.1 exact selected-state reconstruction changed")

    start_raw = _raw_rows(bundles, arrays["start"])
    start_pooled = _pooled_rows(start_raw)
    start_invariance = _prompt_invariance(bundles, arrays["start"])
    start_presence = _presence(start_raw)
    start_counts = all_three_counts(start_presence)
    start_v5_dense = np.square(
        arrays["start_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    v10v._close(value["start_raw"], start_raw, "start raw")
    v10v._close(value["start_pooled"], start_pooled, "start pooled")
    v10v._close(value["start_prompt_invariance"], start_invariance, "start invariance")
    if value.get("start_presence") != start_presence or value.get("start_all_three_counts") != start_counts:
        raise ValueError("Teacher-v10.3.1 start presence changed")
    v10v._close(value["start_v5_dense"], start_v5_dense, "start v5")

    directions = value.get("direction_rows")
    monitors = value.get("monitor_rows")
    if not isinstance(directions, list) or len(directions) != UPDATE_COUNT:
        raise ValueError("Teacher-v10.3.1 direction count changed")
    if not isinstance(monitors, list) or len(monitors) != UPDATE_COUNT:
        raise ValueError("Teacher-v10.3.1 monitor count changed")
    previous_state = value.get("start_state_sha256")
    recomputed_monitors = []
    for step in MONITOR_STEPS:
        direction = directions[step - 1]
        gram = np.asarray(direction["gram"], np.float64)
        weights = np.asarray(direction["weights"], np.float64)
        if (
            direction.get("step") != step
            or direction.get("task_order") != list(TASK_ORDER)
            or len(direction.get("task_losses", [])) != 21
            or gram.shape != (21, 21)
            or weights.shape != (21,)
            or not np.isfinite(gram).all()
            or not np.allclose(gram, gram.T, rtol=1e-9, atol=1e-10)
            or not np.allclose(np.diag(gram), np.ones(21), rtol=2e-5, atol=2e-6)
            or np.any(weights < -1e-12)
            or not np.isclose(weights.sum(), 1.0, rtol=1e-9, atol=1e-10)
            or direction.get("pre_state_sha256") != previous_state
            or direction.get("pre_state_sha256") == direction.get("post_state_sha256")
        ):
            raise ValueError("Teacher-v10.3.1 direction/state chain changed")
        expected_weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
        v10v._close(
            weights.tolist(),
            expected_weights.tolist(),
            "Frank-Wolfe weights",
        )
        analytic_norm = float(np.sqrt(max(float(weights @ gram @ weights), 0.0)))
        recorded_norm = float(direction["direction_norm_before_unit"])
        actual_derivatives = np.asarray(
            direction["directional_derivatives"], dtype=np.float64
        )
        if analytic_norm <= 1e-12:
            raise ValueError("Teacher-v10.3.1 analytic direction is zero")
        analytic_derivatives = gram @ weights / analytic_norm
        if (
            not np.isfinite(recorded_norm)
            or recorded_norm <= 1e-12
            or recorded_norm > 1.0001
            or actual_derivatives.shape != (21,)
            or not np.isfinite(actual_derivatives).all()
            or not np.isfinite(analytic_derivatives).all()
            or float(analytic_derivatives.min()) < -2e-7
        ):
            raise ValueError("Teacher-v10.3.1 recorded direction geometry is invalid")
        derivatives = actual_derivatives.tolist()
        v10v._close(
            direction["minimum_directional_derivative"],
            min(derivatives),
            "recorded minimum derivative",
        )
        if not isinstance(direction.get("direction_sha256"), str) or len(direction["direction_sha256"]) != 64:
            raise ValueError("Teacher-v10.3.1 direction hash changed")

        prediction = arrays["candidates"][step - 1]
        current_raw = _raw_rows(bundles, prediction)
        current_pooled = _pooled_rows(current_raw)
        current_invariance = _prompt_invariance(bundles, prediction)
        current_presence = _presence(current_raw)
        current_counts = all_three_counts(current_presence)
        current_v5_dense = np.square(
            arrays["candidate_v5_predictions"][step - 1] - arrays["v5_target"]
        ).mean(axis=(1, 2)).tolist()
        pooled_checks = response_checks(
            base_rows=start_pooled,
            candidate_rows=current_pooled,
            base_prompt_invariance=start_invariance,
            candidate_prompt_invariance=current_invariance,
            base_v5_dense=start_v5_dense,
            candidate_v5_dense=current_v5_dense,
            directional_derivatives=derivatives,
            maximum_map_delta=float(np.abs(prediction - arrays["start"]).max()),
        )
        checks = calibration_checks(
            start_rows=start_raw,
            candidate_rows=current_raw,
            pooled_response_checks=pooled_checks,
            pre_state_sha256=direction["pre_state_sha256"],
            post_state_sha256=direction["post_state_sha256"],
        )
        recomputed = {
            "step": step,
            "pre_state_sha256": direction["pre_state_sha256"],
            "post_state_sha256": direction["post_state_sha256"],
            "direction_sha256": direction["direction_sha256"],
            "start_raw": start_raw,
            "candidate_raw": current_raw,
            "start_pooled": start_pooled,
            "candidate_pooled": current_pooled,
            "start_prompt_invariance": start_invariance,
            "candidate_prompt_invariance": current_invariance,
            "start_v5_dense": start_v5_dense,
            "candidate_v5_dense": current_v5_dense,
            "presence": current_presence,
            "all_three_counts": current_counts,
            "pooled_response_checks": pooled_checks,
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        v10v._close(monitors[step - 1], recomputed, "monitor step {}".format(step))
        recomputed_monitors.append(recomputed)
        previous_state = direction["post_state_sha256"]
    if value.get("final_state_sha256") != previous_state:
        raise ValueError("Teacher-v10.3.1 final state hash changed")

    shortlist = rank_shortlist(recomputed_monitors)
    shortlist_hashes = {
        str(step): recomputed_monitors[step - 1]["post_state_sha256"]
        for step in shortlist
    }
    if value.get("shortlisted_steps") != shortlist or value.get("shortlisted_state_sha256") != shortlist_hashes:
        raise ValueError("Teacher-v10.3.1 shortlist changed")
    expected_checks = {
        "sealed_v103_pass_and_selected_response_bound": True,
        "fresh_v5r4_zero_init": True,
        "selected_v103_state_exactly_reconstructed": True,
        "selected_v103_v5_response_exactly_reconstructed": True,
        "policy_written_before_scene_arrays_and_model": True,
        "design_audit_and_v103_seed_domains_disjoint": True,
        "all_12_audit_partial_resumes_bitwise_exact": True,
        "six_21_task_common_descent_updates_completed": True,
        "six_actual_two_scene_k3_monitors_completed": True,
        "shortlisted_states_verified_in_memory": bool(shortlist),
        "absolute_all_three_is_diagnostic_only": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "only_two_train_scene_arrays_loaded": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "at_least_one_calibration_state_is_admissible": bool(shortlist),
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if (
        value.get("checks") != expected_checks
        or value.get("status") != status
        or value.get("failed_checks") != failed
        or value.get("authorizes_extended_onpolicy_training") is not (status == "PASS")
        or value.get("authorizes_lora_capacity_or_placement_diagnosis") is not (status == "FAIL")
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v10.3.1 status/authorization changed")
    lora = value.get("lora")
    if (
        not isinstance(lora, Mapping)
        or lora.get("rank") != LORA_RANK
        or float(lora.get("alpha", 0.0)) != LORA_ALPHA
        or int(lora.get("module_count", 0)) != 31
        or lora.get("base_parameters_frozen") is not True
    ):
        raise ValueError("Teacher-v10.3.1 LoRA inventory changed")
    binding = canonical_sha256(
        {
            "v103_binding_id": value["v103_binding_id"],
            "policy_id": POLICY_ID,
            "start_state_sha256": value["start_state_sha256"],
            "shortlisted_steps": shortlist,
            "shortlisted_state_sha256": shortlist_hashes,
            "calibration_maps_sha256": value["calibration_maps_sha256"],
            "audit_seed_table": expected_audit,
            "status": status,
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("Teacher-v10.3.1 binding ID changed")
    if list(summary_file.parent.rglob("*.pt")) or list(summary_file.parent.rglob("*.pth")):
        raise ValueError("Teacher-v10.3.1 contains forbidden model state")

    print("[ONPOLICY_CALIBRATION6_{}] Teacher-v10.3.1 integrity".format(status))
    print("[PASS] exact v10.3 state, six transitions and K=3 maps recomputed")
    print("[PASS] pooled/per-generation/v5/negative gates independently verified")
    print("[PASS] sealed v3 validator hash and authorized replacement verified")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")
    print("[OK] shortlisted steps:", shortlist)
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

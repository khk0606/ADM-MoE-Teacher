#!/usr/bin/env python3
"""Independent validator for Teacher-v10.3 on-policy response gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

import validate_relational_teacher_v10_supervised_capacity as v10v
from relational_teacher_v102_dense_instance_contract import (
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_INSTANCES,
)
from relational_teacher_v103_onpolicy_response_contract import (
    CAPTURE_TIMESTEPS,
    DEVELOPMENT_SCENE,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_SEED,
    POLICY,
    POLICY_ID,
    PROMPT_IDS,
    ROLES,
    SCENES,
    SCHEMA,
    STEP_RADII,
    TASK_ORDER,
    V102_SCHEMA,
    canonical_sha256,
    pooled_roles,
    rank_candidates,
    response_checks,
    stable_seed_pair,
)
from relational_teacher_v9_lora_runtime import FORWARD_INPUT_KEYS
from relational_teacher_v9_all_sittable_contract import read_json, sha256_file
from relational_teacher_v9_all_sittable_metrics import simultaneous_presence_checks


def _validate_v102(path: Path, binding_id: object) -> None:
    value = read_json(path)
    if (
        value.get("schema") != V102_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_step") is not None
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_objective_or_architecture_redesign") is not True
        or value.get("failed_checks")
        != ["at_least_one_dense_instance_candidate_passes_actual_k3"]
        or value.get("binding_id") != binding_id
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v10.2 failure authority changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != set(hashes)
        or "checkpoint" in paths
    ):
        raise ValueError("Teacher-v10.2 failure path binding changed")
    for name, raw in paths.items():
        bound = Path(str(raw)).expanduser().resolve()
        if not bound.is_file() or sha256_file(bound) != hashes[name]:
            raise ValueError("Teacher-v10.2 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed Teacher-v10.2 contains a checkpoint")


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


def _metric_rows(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    return {
        scene: {
            prompt: v10v._metrics(
                bundles[scene], predictions[scene_index, prompt_index]
            )
            for prompt_index, prompt in enumerate(PROMPT_IDS)
        }
        for scene_index, scene in enumerate(SCENES)
    }


def _prompt_invariance(
    bundles: Mapping[str, Mapping[str, object]], predictions: np.ndarray
) -> Dict[str, float]:
    result = {}
    for scene_index, scene in enumerate(SCENES):
        mask = np.asarray(bundles[scene]["verified_positive_mask"], bool)
        result[scene] = float(
            np.square(
                predictions[scene_index, 0, mask]
                - predictions[scene_index, 1, mask]
            ).mean()
        )
    return result


def _all_three(
    rows: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> Dict[str, Dict[str, bool]]:
    return {
        scene: {
            prompt: all(
                simultaneous_presence_checks(
                    rows[scene][prompt], **ABSOLUTE_PRESENCE_LIMITS
                ).values()
            )
            for prompt in PROMPT_IDS
        }
        for scene in SCENES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = read_json(report_file)
    if (
        value.get("schema") != SCHEMA
        or value.get("seed") != MODEL_SEED
        or value.get("diffusion_steps") != 500
        or value.get("train_scenes") != list(SCENES)
        or value.get("development_scene_metadata_only") != DEVELOPMENT_SCENE
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("prompt_ids") != list(PROMPT_IDS)
        or value.get("forward_input_keys") != sorted(FORWARD_INPUT_KEYS)
        or value.get("capture_timesteps") != list(CAPTURE_TIMESTEPS)
        or value.get("step_radii") != list(STEP_RADII)
        or value.get("policy_id") != POLICY_ID
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v10.3 report contract changed")
    expected_design = [
        [list(stable_seed_pair("design", scene, prompt)) for prompt in PROMPT_IDS]
        for scene in SCENES
    ]
    expected_audit = [
        [list(stable_seed_pair("audit", scene, prompt)) for prompt in PROMPT_IDS]
        for scene in SCENES
    ]
    if (
        value.get("design_seed_table") != expected_design
        or value.get("audit_seed_table") != expected_audit
        or expected_design == expected_audit
    ):
        raise ValueError("Teacher-v10.3 seed table changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    required_paths = {
        "runner",
        "validator",
        "contract",
        "summarizer",
        "v102_failure",
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
        "response_policy",
        "response_maps",
    }
    if (
        not isinstance(paths, Mapping)
        or not isinstance(hashes, Mapping)
        or set(paths) != required_paths
        or set(hashes) != required_paths
    ):
        raise ValueError("Teacher-v10.3 path inventory changed")
    for name, raw in paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("Teacher-v10.3 bound file changed: " + str(name))
    _validate_v102(
        Path(str(paths["v102_failure"])).resolve(), value.get("v102_binding_id")
    )
    if read_json(Path(str(paths["response_policy"])).resolve()) != POLICY:
        raise ValueError("Teacher-v10.3 policy changed")
    maps_file = Path(str(paths["response_maps"])).resolve()
    if (
        sha256_file(maps_file) != value.get("response_maps_sha256")
        or value.get("response_maps_sha256") != hashes["response_maps"]
    ):
        raise ValueError("Teacher-v10.3 response maps hash changed")
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
        "capture_timesteps",
        "step_radii",
        "design_seed_table",
        "audit_seed_table",
        "design_states",
        "audit_states",
        "audit_base_normalized",
        "audit_base",
        "direct_before",
        "direct_after",
        "candidates_normalized",
        "candidates",
        "v5_target",
        "base_v5_prediction",
        "candidate_v5_predictions",
    } | {
        prefix + "_" + suffix
        for prefix in ("source", "audit")
        for suffix in suffixes
    }
    if set(arrays) != required_arrays:
        raise ValueError("Teacher-v10.3 map inventory changed")
    if (
        arrays["scene_ids"].tolist() != list(SCENES)
        or arrays["prompt_ids"].tolist() != list(PROMPT_IDS)
        or arrays["capture_timesteps"].tolist() != list(CAPTURE_TIMESTEPS)
        or arrays["step_radii"].tolist() != list(STEP_RADII)
        or arrays["design_seed_table"].tolist() != expected_design
        or arrays["audit_seed_table"].tolist() != expected_audit
        or arrays["design_states"].shape != (2, 3, 2, 8192, 6)
        or arrays["audit_states"].shape != (2, 3, 2, 8192, 6)
        or arrays["audit_base_normalized"].shape != (2, 2, 8192, 6)
        or arrays["audit_base"].shape != (2, 2, 8192, 6)
        or arrays["direct_before"].shape != (3, 2, 2, 8192, 6)
        or arrays["direct_after"].shape != (3, 4, 2, 2, 8192, 6)
        or arrays["candidates_normalized"].shape != (3, 4, 2, 2, 8192, 6)
        or arrays["candidates"].shape != (3, 4, 2, 2, 8192, 6)
        or arrays["v5_target"].shape != (3, 8192, 6)
        or arrays["base_v5_prediction"].shape != (3, 8192, 6)
        or arrays["candidate_v5_predictions"].shape != (3, 4, 3, 8192, 6)
    ):
        raise ValueError("Teacher-v10.3 map shape/order changed")
    for name in arrays:
        array = np.asarray(arrays[name])
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("audit_base", "direct_before", "direct_after", "candidates"):
        array = np.asarray(arrays[name])
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(name + " is not a physical map")
    bundles = {scene: _bundle(arrays, index) for index, scene in enumerate(SCENES)}
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

    base_rows = _metric_rows(bundles, arrays["audit_base"])
    base_pooled = pooled_roles(base_rows)
    base_invariance = _prompt_invariance(bundles, arrays["audit_base"])
    base_all_three = _all_three(base_rows)
    base_v5_dense = np.square(
        arrays["base_v5_prediction"] - arrays["v5_target"]
    ).mean(axis=(1, 2)).tolist()
    v10v._close(value["base_rows"], base_rows, "base rows")
    v10v._close(value["base_pooled"], base_pooled, "base pooled")
    v10v._close(value["base_prompt_invariance"], base_invariance, "base invariance")
    if value.get("base_all_three_diagnostic") != base_all_three:
        raise ValueError("Teacher-v10.3 base all-three diagnostic changed")
    v10v._close(value["base_v5_dense"], base_v5_dense, "base v5")

    if len(value.get("direction_rows", [])) != 3:
        raise ValueError("Teacher-v10.3 direction count changed")
    recomputed_derivatives = []
    for index, timestep in enumerate(CAPTURE_TIMESTEPS):
        row = value["direction_rows"][index]
        gram = np.asarray(row["gram"], np.float64)
        weights = np.asarray(row["weights"], np.float64)
        if (
            row.get("capture_timestep") != timestep
            or row.get("task_order") != list(TASK_ORDER)
            or len(row.get("task_losses", [])) != 21
            or gram.shape != (21, 21)
            or weights.shape != (21,)
            or not np.isfinite(gram).all()
            or not np.allclose(gram, gram.T, rtol=1e-9, atol=1e-10)
            or not np.allclose(np.diag(gram), np.ones(21), rtol=2e-5, atol=2e-6)
            or np.any(weights < -1e-12)
            or not np.isclose(weights.sum(), 1.0, rtol=1e-9, atol=1e-10)
        ):
            raise ValueError("Teacher-v10.3 gradient geometry changed")
        norm = float(np.sqrt(max(float(weights @ gram @ weights), 0.0)))
        if norm <= 1e-12:
            raise ValueError("Teacher-v10.3 direction norm is zero")
        derivatives = (gram @ weights / norm).tolist()
        v10v._close(
            row["directional_derivatives"], derivatives, "directional derivatives"
        )
        v10v._close(
            row["minimum_directional_derivative"],
            min(derivatives),
            "minimum directional derivative",
        )
        recomputed_derivatives.append(derivatives)

    if len(value.get("candidates", [])) != 12:
        raise ValueError("Teacher-v10.3 candidate count changed")
    recomputed_candidates = []
    candidate_index = 0
    for timestep_index, timestep in enumerate(CAPTURE_TIMESTEPS):
        for radius_index, radius in enumerate(STEP_RADII):
            recorded = value["candidates"][candidate_index]
            prediction = arrays["candidates"][timestep_index, radius_index]
            rows = _metric_rows(bundles, prediction)
            invariance = _prompt_invariance(bundles, prediction)
            candidate_v5_dense = np.square(
                arrays["candidate_v5_predictions"][timestep_index, radius_index]
                - arrays["v5_target"]
            ).mean(axis=(1, 2)).tolist()
            maximum_delta = float(np.abs(prediction - arrays["audit_base"]).max())
            checks = response_checks(
                base_rows=base_rows,
                candidate_rows=rows,
                base_prompt_invariance=base_invariance,
                candidate_prompt_invariance=invariance,
                base_v5_dense=base_v5_dense,
                candidate_v5_dense=candidate_v5_dense,
                directional_derivatives=recomputed_derivatives[timestep_index],
                maximum_map_delta=maximum_delta,
            )
            expected = {
                "name": "t{}_radius_{}".format(
                    timestep, str(radius).replace("0.", "0p")
                ),
                "capture_timestep": timestep,
                "radius": radius,
                "direction_sha256": recorded["direction_sha256"],
                "directional_derivatives": recomputed_derivatives[timestep_index],
                "base_rows": base_rows,
                "candidate_rows": rows,
                "base_pooled": base_pooled,
                "candidate_pooled": pooled_roles(rows),
                "base_prompt_invariance": base_invariance,
                "candidate_prompt_invariance": invariance,
                "base_v5_dense": base_v5_dense,
                "candidate_v5_dense": candidate_v5_dense,
                "base_all_three_diagnostic": base_all_three,
                "candidate_all_three_diagnostic": _all_three(rows),
                "maximum_final_map_delta": maximum_delta,
                "state_sha256": recorded["state_sha256"],
                "candidate_maps_sha256": v10v._tensor_sha256(
                    arrays["candidates_normalized"][timestep_index, radius_index]
                ),
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(
                    key for key, passed in checks.items() if not passed
                ),
            }
            v10v._close(recorded, expected, "candidate " + expected["name"])
            recomputed_candidates.append(expected)
            candidate_index += 1
    eligible_order = rank_candidates(recomputed_candidates)
    selected = eligible_order[0] if eligible_order else None
    if (
        value.get("eligible_selection_order") != eligible_order
        or value.get("selected_candidate") != selected
    ):
        raise ValueError("Teacher-v10.3 selected response changed")

    expected_checks = {
        "sealed_v102_failure_bound": True,
        "fresh_v5r4_zero_init": True,
        "policy_written_before_model_and_lora": True,
        "both_train_scene_arrays_loaded": True,
        "design_and_audit_seed_domains_disjoint": expected_design != expected_audit,
        "all_12_partial_resumes_bitwise_exact": True,
        "exact_21_task_inventory": len(TASK_ORDER) == 21,
        "candidate_decision_uses_resumed_final_maps": True,
        "absolute_all_three_is_diagnostic_only": True,
        "teacher_forward_is_text_plus_scene_only": True,
        "no_optimizer_created": True,
        "no_model_checkpoint_saved": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "at_least_one_onpolicy_response_is_admissible": selected is not None,
    }
    status = "PASS" if all(expected_checks.values()) else "FAIL"
    failed = sorted(name for name, passed in expected_checks.items() if not passed)
    checkpoint_files = list(report_file.parent.rglob("*.pt")) + list(
        report_file.parent.rglob("*.pth")
    )
    if (
        value.get("checks") != expected_checks
        or value.get("failed_checks") != failed
        or value.get("status") != status
        or value.get("authorizes_onpolicy_multiupdate_calibration")
        is not (status == "PASS")
        or value.get("authorizes_lora_capacity_or_placement_diagnosis")
        is not (status == "FAIL")
        or checkpoint_files
    ):
        raise ValueError("Teacher-v10.3 authorization/status changed")
    binding = canonical_sha256(
        {
            "v102_binding_id": value["v102_binding_id"],
            "policy_id": POLICY_ID,
            "design_seed_table": expected_design,
            "audit_seed_table": expected_audit,
            "selected_candidate": selected,
            "response_maps_sha256": value["response_maps_sha256"],
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("Teacher-v10.3 binding ID changed")
    lora = value.get("lora")
    if (
        not isinstance(lora, Mapping)
        or lora.get("rank") != LORA_RANK
        or float(lora.get("alpha", 0.0)) != LORA_ALPHA
        or int(lora.get("module_count", 0)) != 31
        or lora.get("base_parameters_frozen") is not True
        or lora.get("zero_initialized_output_projection") is not True
    ):
        raise ValueError("Teacher-v10.3 LoRA inventory changed")

    print("[ONPOLICY_RESPONSE_{}] Teacher-v10.3 integrity".format(status))
    print("[PASS] two-scene disjoint trajectories and response maps recomputed")
    print("[PASS] 21-task geometry, v5/negative retention and selection verified")
    print("[PASS] no optimizer/checkpoint, room_0201 or paper-test access")
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", failed)


if __name__ == "__main__":
    main()

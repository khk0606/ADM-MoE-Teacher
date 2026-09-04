#!/usr/bin/env python3
"""Deep validator for Teacher-v9.5 multi-timestep consensus."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


PREPARE_ROOT = Path(__file__).resolve().parent
if str(PREPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPARE_ROOT))

from preflight_relational_teacher_v95_multitimestep_consensus import (  # noqa: E402
    _objective_values,
    _validate_failed_v941,
)
from relational_teacher_v9_all_sittable_contract import sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import canonical_sha256  # noqa: E402
from relational_teacher_v9_lora_runtime import tensor_sha256  # noqa: E402
from relational_teacher_v93_exact_topk_objective import exact_topk_swap_macro_loss  # noqa: E402
from relational_teacher_v94_common_descent import frank_wolfe_min_norm_weights  # noqa: E402
from relational_teacher_v95_multitimestep_consensus_contract import (  # noqa: E402
    AUDIT_PANELS,
    DESIGN_PANELS,
    FW_ITERATIONS,
    INITIALIZATION_SEED,
    MIN_DIRECTIONAL_DERIVATIVE,
    SCHEMA,
    SEED,
    STEP_RADII,
    TRAIN_SCENE,
    V5_NOISE_SEED,
    V5_TIMESTEPS,
    consensus_candidate_checks,
    rank_candidates,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import _validate_preflight  # noqa: E402


REQUIRED_PATHS = {
    "failed_v941_summary", "failed_v941_maps", "failed_v94_report",
    "preflight_report", "metric_policy", "dataset_index", "source_dataset_index",
    "v5_split", "stats_file", "original_checkpoint", "v5_checkpoint",
    "v5_evidence_report", "v93_objective", "v94_common_descent", "contract",
    "runner", "validator", "response_maps",
}


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
            _close(one, two, f"{label}[{index}]")
        return
    if (
        isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
    ):
        if not math.isclose(float(left), float(right), rel_tol=2e-6, abs_tol=2e-5):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " changed")


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(label + " contains NaN/Inf")


def _values(arrays: Mapping[str, np.ndarray], prediction: np.ndarray, frozen: np.ndarray) -> Mapping[str, object]:
    physical = torch.from_numpy(prediction.astype(np.float32))
    targets = torch.from_numpy(np.stack([arrays["instance_targets"]] * 2).astype(np.float32))
    masks = torch.from_numpy(np.stack([arrays["verified_object_mask"]] * 2))
    _, swap_rows, overlap_rows = exact_topk_swap_macro_loss(physical, targets, masks)
    objective = {"per_instance_exact_topk_swap": swap_rows, "per_instance_exact_topk_overlap": overlap_rows}
    bundle = {
        "instance_targets": arrays["instance_targets"],
        "instance_names": ("bed_01", "chair_01", "chair_06"),
        "xyz": arrays["xyz"],
        "verified_object_mask": arrays["verified_object_mask"],
        "verified_positive_mask": arrays["verified_positive_mask"],
        "explicit_negative_mask": arrays["explicit_negative_mask"],
        "unknown_sittable_mask": arrays["unknown_sittable_mask"],
    }
    return _objective_values(objective, prediction, frozen, bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = json.loads(report_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.5 schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("initialization_seed") != INITIALIZATION_SEED
        or value.get("diffusion_steps") != 500
        or value.get("design_panels") != list(DESIGN_PANELS)
        or value.get("audit_panels") != list(AUDIT_PANELS)
        or value.get("step_radius_grid") != list(STEP_RADII)
        or value.get("v5_timesteps") != list(V5_TIMESTEPS)
        or value.get("v5_noise_seed") != V5_NOISE_SEED
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != "room_0102"
        or value.get("development_scene_metadata_only") != "room_0201"
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("serialized_model_state") is not False
        or value.get("failed_checkpoint_loaded") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_rollout") is not False
        or value.get("authorizes_overfit120") is not False
        or value.get("authorizes_calibration") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.5 protocol changed")
    design_keys = {(row["timestep"], row["noise_seed"]) for row in DESIGN_PANELS}
    audit_keys = {(row["timestep"], row["noise_seed"]) for row in AUDIT_PANELS}
    if design_keys & audit_keys:
        raise ValueError("design/audit panels overlap")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or int(lora.get("module_count", 0)) != 31:
        raise ValueError("LoRA inventory changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("consensus path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("consensus path-hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("consensus bound file changed: " + name)
    if report_file.parent != Path(str(paths["response_maps"])).resolve().parent:
        raise ValueError("report/map output directories differ")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("consensus preflight persisted model state")
    failure = _validate_failed_v941(Path(str(paths["failed_v941_summary"])))
    preflight, _ = _validate_preflight(Path(str(paths["preflight_report"])))
    if value.get("failed_v941_binding_id") != failure.get("binding_id") or value.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("consensus authority binding changed")

    audit = value.get("gradient_audit")
    if not isinstance(audit, Mapping) or audit.get("task_count") != 24:
        raise ValueError("consensus gradient audit inventory changed")
    labels = audit.get("task_labels")
    if not isinstance(labels, list) or len(labels) != 24 or len(set(labels)) != 24:
        raise ValueError("consensus task labels changed")
    gram = np.asarray(audit.get("normalized_gram"), dtype=np.float64)
    if gram.shape != (24, 24) or not np.isfinite(gram).all():
        raise ValueError("consensus Gram matrix changed")
    if not np.allclose(gram, gram.T, atol=2e-6) or not np.allclose(np.diag(gram), 1.0, atol=2e-5):
        raise ValueError("normalized consensus Gram matrix is invalid")
    pair_count = 24 * 23 // 2
    conflicts = sum(1 for left in range(24) for right in range(left + 1, 24) if gram[left, right] < -1e-8)
    weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
    common_norm = float(math.sqrt(max(float(weights @ gram @ weights), 0.0)))
    directional = gram @ weights / common_norm if common_norm > 0.0 else np.zeros(24)
    valid = bool(np.isfinite(directional).all() and float(directional.min()) >= MIN_DIRECTIONAL_DERIVATIVE)
    if audit.get("pair_count") != pair_count or audit.get("pairwise_conflict_count") != conflicts or audit.get("frank_wolfe_iterations") != FW_ITERATIONS:
        raise ValueError("consensus gradient count audit changed")
    _close(audit.get("minimum_norm_weights"), weights.tolist(), "consensus weights")
    _close(audit.get("common_vector_norm"), common_norm, "consensus norm")
    _close(audit.get("task_directional_derivatives"), directional.tolist(), "consensus derivatives")
    _close(audit.get("minimum_task_directional_derivative"), float(directional.min()), "minimum derivative")
    if audit.get("common_direction_valid") is not valid:
        raise ValueError("consensus direction validity changed")

    with np.load(Path(str(paths["response_maps"])), allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required_arrays = {
        "xyz", "verified_object_mask", "verified_positive_mask", "unknown_sittable_mask",
        "explicit_negative_mask", "instance_targets", "base", "candidates",
        "base_normalized", "candidates_normalized", "candidate_names",
        "v5_gt_normalized", "base_v5_normalized", "candidate_v5_normalized",
    }
    if set(arrays) != required_arrays:
        raise ValueError("consensus array inventory changed")
    panel_count = len(AUDIT_PANELS)
    candidate_count = len(STEP_RADII)
    expected_shapes = {
        "xyz": (8192, 3), "verified_object_mask": (3, 8192),
        "verified_positive_mask": (8192,), "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,), "instance_targets": (3, 8192, 6),
        "base": (panel_count, 2, 8192, 6),
        "candidates": (candidate_count, panel_count, 2, 8192, 6),
        "base_normalized": (panel_count, 2, 8192, 6),
        "candidates_normalized": (candidate_count, panel_count, 2, 8192, 6),
        "candidate_names": (candidate_count,), "v5_gt_normalized": (3, 8192, 6),
        "base_v5_normalized": (3, 8192, 6),
        "candidate_v5_normalized": (candidate_count, 3, 8192, 6),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape changed")
    expected_names = tuple("multitimestep_radius_" + str(radius).replace(".", "p") for radius in STEP_RADII)
    if tuple(str(item) for item in arrays["candidate_names"].tolist()) != expected_names:
        raise ValueError("consensus candidate order changed")
    for name, array in arrays.items():
        if name != "candidate_names" and not np.isfinite(array).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "base", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if not np.array_equal(arrays["verified_positive_mask"], arrays["verified_object_mask"].any(axis=0)):
        raise ValueError("verified positive mask changed")
    with np.load(Path(str(paths["failed_v941_maps"])), allow_pickle=False) as source:
        for name in ("xyz", "verified_object_mask", "unknown_sittable_mask", "explicit_negative_mask", "instance_targets"):
            if not np.array_equal(arrays[name], source[name]):
                raise ValueError("consensus/v9.4.1 " + name + " differs")

    base_rows = [_values(arrays, arrays["base"][index], arrays["base"][index]) for index in range(panel_count)]
    v5_gt = arrays["v5_gt_normalized"]
    base_v5_dense = np.square(arrays["base_v5_normalized"] - v5_gt).mean(axis=(1, 2)).tolist()
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != candidate_count:
        raise ValueError("consensus candidate rows changed")
    recomputed_rows = []
    for candidate_index, (radius, row) in enumerate(zip(STEP_RADII, rows)):
        if row.get("name") != expected_names[candidate_index] or row.get("step_radius") != radius:
            raise ValueError("consensus candidate specification changed")
        candidate_rows = [
            _values(arrays, arrays["candidates"][candidate_index, panel], arrays["base"][panel])
            for panel in range(panel_count)
        ]
        candidate_v5_dense = np.square(arrays["candidate_v5_normalized"][candidate_index] - v5_gt).mean(axis=(1, 2)).tolist()
        checks = consensus_candidate_checks(
            base_rows=base_rows,
            candidate_rows=candidate_rows,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=candidate_v5_dense,
            common_direction_valid=valid,
        )
        _close(row.get("base_rows"), base_rows, "candidate base rows")
        _close(row.get("candidate_rows"), candidate_rows, "candidate rows")
        _close(row.get("base_v5_dense"), base_v5_dense, "base v5")
        _close(row.get("candidate_v5_dense"), candidate_v5_dense, "candidate v5")
        if row.get("checks") != checks:
            raise ValueError("consensus candidate checks changed")
        eligible = all(checks.values())
        if row.get("eligible") is not eligible or row.get("failed_checks") != sorted(name for name, passed in checks.items() if not passed):
            raise ValueError("consensus candidate eligibility changed")
        if row.get("candidate_v5_prediction_sha256") != tensor_sha256(torch.from_numpy(arrays["candidate_v5_normalized"][candidate_index])):
            raise ValueError("consensus candidate v5 hash changed")
        recomputed_rows.append({**row, "base_rows": base_rows, "candidate_rows": candidate_rows, "checks": checks})

    order = rank_candidates(recomputed_rows)
    selected = order[0] if order else None
    status = "PASS" if selected is not None else "FAIL"
    expected_checks = {
        "failed_v941_timestep_diagnosis_bound": True,
        "fresh_v5r4_zero_init": True,
        "twenty_four_design_task_gradients_measured": True,
        "common_direction_exists": valid,
        "new_audit_panels_disjoint": True,
        "exact_radius_grid": True,
        "only_lora_perturbed": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_checkpoint_saved": True,
        "at_least_one_consensus_candidate_admissible": selected is not None,
    }
    if (
        value.get("eligible_selection_order") != order
        or value.get("selected_candidate") != selected
        or value.get("status") != status
        or value.get("checks") != expected_checks
        or value.get("authorizes_consensus_response6") is not (status == "PASS")
    ):
        raise ValueError("consensus selection/status changed")
    expected_failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if value.get("failed_checks") != expected_failed:
        raise ValueError("consensus top-level failed checks changed")
    expected_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "failed_v941_binding_id": value["failed_v941_binding_id"],
            "gradient_audit": audit,
            "design_panels": value["design_panels"],
            "audit_panels": value["audit_panels"],
            "step_radius_grid": value["step_radius_grid"],
            "v5_timesteps": value["v5_timesteps"],
            "v5_noise_seed": value["v5_noise_seed"],
            "selected_candidate": selected,
            "response_maps_sha256": hashes["response_maps"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("consensus binding ID changed")
    _finite_tree(value, "consensus report")
    print(f"[MULTITIMESTEP_CONSENSUS_{status}] Teacher-v9.5 integrity")
    print("[PASS] sealed timestep failure and 24-task gradient geometry verified")
    print("[PASS] four new audit panels, maps and retention gates recomputed")
    print("[PASS] no checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] conflicts:", conflicts, "/", pair_count)
    print("[OK] minimum directional derivative:", float(directional.min()))
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", expected_failed)


if __name__ == "__main__":
    main()

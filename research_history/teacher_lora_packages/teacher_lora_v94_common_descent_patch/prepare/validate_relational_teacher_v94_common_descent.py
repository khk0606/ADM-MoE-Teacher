#!/usr/bin/env python3
"""Deep validator for the Teacher-v9.4 common-descent preflight."""

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

from preflight_relational_teacher_v94_common_descent import (  # noqa: E402
    _objective_values,
    _validate_failed_v93,
)
from relational_teacher_v9_all_sittable_contract import sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import canonical_sha256  # noqa: E402
from relational_teacher_v9_lora_runtime import tensor_sha256  # noqa: E402
from relational_teacher_v93_exact_topk_objective import (  # noqa: E402
    exact_topk_swap_macro_loss,
)
from relational_teacher_v94_common_descent import (  # noqa: E402
    frank_wolfe_min_norm_weights,
)
from relational_teacher_v94_common_descent_contract import (  # noqa: E402
    FW_ITERATIONS,
    MIN_DIRECTIONAL_DERIVATIVE,
    PROBE_TIMESTEP,
    SCHEMA,
    SEED,
    STEP_RADII,
    TASK_ORDER,
    TRAIN_SCENE,
    candidate_checks,
    rank_candidates,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _validate_preflight,
)


REQUIRED_PATHS = {
    "failed_v93_report",
    "failed_v93_maps",
    "preflight_report",
    "metric_policy",
    "dataset_index",
    "source_dataset_index",
    "v5_split",
    "stats_file",
    "original_checkpoint",
    "v5_checkpoint",
    "v5_evidence_report",
    "v93_objective",
    "common_descent",
    "contract",
    "runner",
    "validator",
    "response_maps",
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
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
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


def _values(arrays: Mapping[str, np.ndarray], prediction: np.ndarray) -> Mapping[str, object]:
    physical = torch.from_numpy(prediction.astype(np.float32))
    targets = torch.from_numpy(np.stack([arrays["instance_targets"]] * 2).astype(np.float32))
    masks = torch.from_numpy(np.stack([arrays["verified_object_mask"]] * 2))
    _, swap_rows, overlap_rows = exact_topk_swap_macro_loss(physical, targets, masks)
    objective = {
        "per_instance_exact_topk_swap": swap_rows,
        "per_instance_exact_topk_overlap": overlap_rows,
    }
    bundle = {
        "instance_targets": arrays["instance_targets"],
        "instance_names": ("bed_01", "chair_01", "chair_06"),
        "xyz": arrays["xyz"],
        "verified_object_mask": arrays["verified_object_mask"],
        "verified_positive_mask": arrays["verified_positive_mask"],
        "explicit_negative_mask": arrays["explicit_negative_mask"],
        "unknown_sittable_mask": arrays["unknown_sittable_mask"],
    }
    return _objective_values(objective, prediction, arrays["base"], bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = json.loads(report_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.4 schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("diffusion_steps") != 500
        or value.get("probe_timestep") != PROBE_TIMESTEP
        or value.get("step_radius_grid") != list(STEP_RADII)
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != "room_0102"
        or value.get("development_scene_metadata_only") != "room_0201"
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("fresh_zero_lora_state_restored_per_candidate") is not True
        or value.get("failed_v93_checkpoint_loaded") is not False
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_rollout") is not False
        or value.get("authorizes_overfit120") is not False
        or value.get("authorizes_calibration") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.4 protocol changed")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or int(lora.get("module_count", 0)) != 31:
        raise ValueError("LoRA inventory changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("common-descent path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("common-descent path-hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("common-descent bound file changed: " + name)
    if report_file.parent != Path(str(paths["response_maps"])).resolve().parent:
        raise ValueError("report/map output directories differ")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("common-descent preflight persisted model state")

    failure = _validate_failed_v93(Path(str(paths["failed_v93_report"])))
    preflight, _ = _validate_preflight(Path(str(paths["preflight_report"])))
    if (
        value.get("failed_v93_binding_id") != failure.get("binding_id")
        or value.get("preflight_binding_id") != preflight.get("binding_id")
    ):
        raise ValueError("common-descent authority binding changed")

    audit = value.get("gradient_audit")
    if not isinstance(audit, Mapping) or audit.get("task_order") != list(TASK_ORDER):
        raise ValueError("gradient audit task order changed")
    gram = np.asarray(audit.get("normalized_gram"), dtype=np.float64)
    if gram.shape != (6, 6) or not np.isfinite(gram).all():
        raise ValueError("gradient Gram matrix shape/value changed")
    if not np.allclose(gram, gram.T, atol=2e-6) or not np.allclose(np.diag(gram), 1.0, atol=2e-5):
        raise ValueError("normalized gradient Gram matrix is invalid")
    conflict_pairs = sum(
        1 for left in range(6) for right in range(left + 1, 6) if gram[left, right] < -1e-8
    )
    weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
    common_norm = float(math.sqrt(max(float(weights @ gram @ weights), 0.0)))
    directional = gram @ weights / common_norm if common_norm > 0.0 else np.zeros(6)
    valid = bool(np.isfinite(directional).all() and float(directional.min()) >= MIN_DIRECTIONAL_DERIVATIVE)
    if audit.get("pairwise_conflict_count") != conflict_pairs or audit.get("frank_wolfe_iterations") != FW_ITERATIONS:
        raise ValueError("gradient conflict/FW audit changed")
    _close(audit.get("minimum_norm_weights"), weights.tolist(), "gradient weights")
    _close(audit.get("common_vector_norm"), common_norm, "common norm")
    _close(audit.get("task_directional_derivatives"), directional.tolist(), "directional derivatives")
    _close(audit.get("minimum_task_directional_derivative"), float(directional.min()), "minimum derivative")
    if audit.get("common_direction_valid") is not valid:
        raise ValueError("common direction validity changed")

    maps_file = Path(str(paths["response_maps"])).resolve()
    with np.load(maps_file, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required_arrays = {
        "xyz", "verified_object_mask", "verified_positive_mask",
        "unknown_sittable_mask", "explicit_negative_mask", "instance_targets",
        "base", "candidates", "base_normalized", "candidates_normalized",
        "candidate_names", "v5_gt_normalized", "base_v5_normalized",
        "candidate_v5_normalized",
    }
    if set(arrays) != required_arrays:
        raise ValueError("common-descent array inventory changed")
    candidate_count = len(STEP_RADII)
    expected_shapes = {
        "xyz": (8192, 3), "verified_object_mask": (3, 8192),
        "verified_positive_mask": (8192,), "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,), "instance_targets": (3, 8192, 6),
        "base": (2, 8192, 6), "candidates": (candidate_count, 2, 8192, 6),
        "base_normalized": (2, 8192, 6),
        "candidates_normalized": (candidate_count, 2, 8192, 6),
        "candidate_names": (candidate_count,), "v5_gt_normalized": (3, 8192, 6),
        "base_v5_normalized": (3, 8192, 6),
        "candidate_v5_normalized": (candidate_count, 3, 8192, 6),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape changed")
    expected_names = tuple("common_descent_radius_" + str(r).replace(".", "p") for r in STEP_RADII)
    if tuple(str(item) for item in arrays["candidate_names"].tolist()) != expected_names:
        raise ValueError("candidate array order changed")
    for name, array in arrays.items():
        if name != "candidate_names" and not np.isfinite(array).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "base", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if not np.array_equal(arrays["verified_positive_mask"], arrays["verified_object_mask"].any(axis=0)):
        raise ValueError("verified positive mask changed")
    with np.load(Path(str(paths["failed_v93_maps"])), allow_pickle=False) as source:
        for name in ("xyz", "verified_object_mask", "unknown_sittable_mask", "explicit_negative_mask", "instance_targets"):
            if not np.array_equal(arrays[name], source[name]):
                raise ValueError("common-descent/v9.3 " + name + " differs")

    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != candidate_count:
        raise ValueError("common-descent candidate rows changed")
    base_values = _values(arrays, arrays["base"])
    v5_gt = arrays["v5_gt_normalized"]
    base_v5_dense = np.square(arrays["base_v5_normalized"] - v5_gt).mean(axis=(1, 2))
    recomputed_rows = []
    for index, (radius, row) in enumerate(zip(STEP_RADII, rows)):
        if row.get("name") != expected_names[index] or row.get("step_radius") != radius:
            raise ValueError("candidate specification changed")
        after = _values(arrays, arrays["candidates"][index])
        candidate_v5_dense = np.square(arrays["candidate_v5_normalized"][index] - v5_gt).mean(axis=(1, 2))
        checks = candidate_checks(
            before=base_values,
            after=after,
            base_v5_dense=base_v5_dense.tolist(),
            candidate_v5_dense=candidate_v5_dense.tolist(),
            common_direction_valid=valid,
        )
        _close(row["before"], base_values, "candidate.before")
        _close(row["after"], after, "candidate.after")
        _close(row["base_v5_dense"], base_v5_dense.tolist(), "candidate.base_v5")
        _close(row["candidate_v5_dense"], candidate_v5_dense.tolist(), "candidate.v5")
        if row.get("checks") != checks:
            raise ValueError("candidate checks changed")
        eligible = all(checks.values())
        if row.get("eligible") is not eligible or row.get("failed_checks") != sorted(
            name for name, passed in checks.items() if not passed
        ):
            raise ValueError("candidate eligibility changed")
        if row.get("prediction_sha256") != tensor_sha256(torch.from_numpy(arrays["candidates_normalized"][index])):
            raise ValueError("candidate prediction hash changed")
        if row.get("v5_prediction_sha256") != tensor_sha256(torch.from_numpy(arrays["candidate_v5_normalized"][index])):
            raise ValueError("candidate v5 prediction hash changed")
        recomputed_rows.append({**row, "before": base_values, "after": after, "checks": checks})

    order = rank_candidates(recomputed_rows)
    selected = order[0] if order else None
    status = "PASS" if selected is not None else "FAIL"
    expected_checks = {
        "failed_v93_diagnosis_bound": True,
        "fresh_v5r4_zero_init": True,
        "six_exact_task_gradients_measured": True,
        "common_direction_exists": valid,
        "exact_radius_grid": True,
        "only_lora_perturbed": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_checkpoint_saved": True,
        "at_least_one_trust_region_step_admissible": selected is not None,
    }
    if (
        value.get("eligible_selection_order") != order
        or value.get("selected_candidate") != selected
        or value.get("status") != status
        or value.get("checks") != expected_checks
        or value.get("authorizes_common_descent_response6") is not (status == "PASS")
    ):
        raise ValueError("common-descent selection/status changed")
    expected_failed = sorted(name for name, passed in expected_checks.items() if not passed)
    if value.get("failed_checks") != expected_failed:
        raise ValueError("top-level failed checks changed")
    expected_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "failed_v93_binding_id": value["failed_v93_binding_id"],
            "gradient_audit": audit,
            "step_radius_grid": value["step_radius_grid"],
            "selected_candidate": selected,
            "response_maps_sha256": hashes["response_maps"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("common-descent binding ID changed")
    _finite_tree(value, "common-descent report")
    print(f"[COMMON_DESCENT_{status}] Teacher-v9.4 common-descent integrity")
    print("[PASS] sealed v9.3 failure and six-task gradient geometry verified")
    print("[PASS] trust-region maps, exact top-k losses and retention gates recomputed")
    print("[PASS] no checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] conflict pairs:", conflict_pairs, "/ 15")
    print("[OK] minimum directional derivative:", float(directional.min()))
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

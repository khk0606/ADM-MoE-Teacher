#!/usr/bin/env python3
"""Deep validator for Teacher-v9.4.1 metric-first replication."""

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

from evaluate_relational_teacher_v941_metric_replication import (  # noqa: E402
    _objective_values,
    _validate_failed_v94,
)
from relational_teacher_v9_all_sittable_contract import sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import canonical_sha256  # noqa: E402
from relational_teacher_v9_lora_runtime import tensor_sha256  # noqa: E402
from relational_teacher_v93_exact_topk_objective import exact_topk_swap_macro_loss  # noqa: E402
from relational_teacher_v941_metric_replication_contract import (  # noqa: E402
    DIRECTION_SEED,
    REPLICATION_PANELS,
    SCHEMA,
    SEED,
    SELECTED_RADIUS,
    TRAIN_SCENE,
    replication_checks,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import _validate_preflight  # noqa: E402


REQUIRED_PATHS = {
    "failed_v94_report", "failed_v94_maps", "preflight_report", "metric_policy",
    "dataset_index", "source_dataset_index", "v5_split", "stats_file",
    "original_checkpoint", "v5_checkpoint", "v5_evidence_report", "v93_objective",
    "v94_common_descent", "contract", "runner", "validator", "response_maps",
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
    return _objective_values(objective, prediction, frozen, bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary_file = args.summary.expanduser().resolve()
    value = json.loads(summary_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.4.1 schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("direction_seed") != DIRECTION_SEED
        or value.get("diffusion_steps") != 500
        or value.get("selected_radius") != SELECTED_RADIUS
        or value.get("replication_panels") != list(REPLICATION_PANELS)
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != "room_0102"
        or value.get("development_scene_metadata_only") != "room_0201"
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("serialized_model_state") is not False
        or value.get("failed_v94_checkpoint_loaded") is not False
        or value.get("authorizes_checkpoint") is not False
        or value.get("authorizes_rollout") is not False
        or value.get("authorizes_overfit120") is not False
        or value.get("authorizes_calibration") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.4.1 protocol changed")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or int(lora.get("module_count", 0)) != 31:
        raise ValueError("LoRA inventory changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("metric replication path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("metric replication path-hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("metric replication bound file changed: " + name)
    if summary_file.parent != Path(str(paths["response_maps"])).resolve().parent:
        raise ValueError("summary/map output directories differ")
    if list(summary_file.parent.glob("*.pt")) or list(summary_file.parent.glob("*.pth")):
        raise ValueError("metric replication persisted model state")
    failure = _validate_failed_v94(Path(str(paths["failed_v94_report"])))
    preflight, _ = _validate_preflight(Path(str(paths["preflight_report"])))
    if value.get("failed_v94_binding_id") != failure.get("binding_id") or value.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("metric replication authority binding changed")

    with np.load(Path(str(paths["response_maps"])), allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required_arrays = {
        "xyz", "verified_object_mask", "verified_positive_mask", "unknown_sittable_mask",
        "explicit_negative_mask", "instance_targets", "base", "candidate",
        "base_normalized", "candidate_normalized", "v5_gt_normalized",
        "base_v5_normalized", "candidate_v5_normalized",
    }
    if set(arrays) != required_arrays:
        raise ValueError("metric replication array inventory changed")
    expected_shapes = {
        "xyz": (8192, 3), "verified_object_mask": (3, 8192),
        "verified_positive_mask": (8192,), "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,), "instance_targets": (3, 8192, 6),
        "base": (3, 2, 8192, 6), "candidate": (3, 2, 8192, 6),
        "base_normalized": (3, 2, 8192, 6), "candidate_normalized": (3, 2, 8192, 6),
        "v5_gt_normalized": (3, 8192, 6), "base_v5_normalized": (3, 8192, 6),
        "candidate_v5_normalized": (3, 8192, 6),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape changed")
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise ValueError("metric replication arrays contain NaN/Inf")
    for name in ("instance_targets", "base", "candidate"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if not np.array_equal(arrays["verified_positive_mask"], arrays["verified_object_mask"].any(axis=0)):
        raise ValueError("verified positive mask changed")
    with np.load(Path(str(paths["failed_v94_maps"])), allow_pickle=False) as source:
        for name in ("xyz", "verified_object_mask", "unknown_sittable_mask", "explicit_negative_mask", "instance_targets"):
            if not np.array_equal(arrays[name], source[name]):
                raise ValueError("metric replication/v9.4 " + name + " differs")

    base_rows = [_values(arrays, arrays["base"][index], arrays["base"][index]) for index in range(3)]
    candidate_rows = [
        _values(arrays, arrays["candidate"][index], arrays["base"][index])
        for index in range(3)
    ]
    v5_gt = arrays["v5_gt_normalized"]
    base_v5_dense = np.square(arrays["base_v5_normalized"] - v5_gt).mean(axis=(1, 2)).tolist()
    candidate_v5_dense = np.square(arrays["candidate_v5_normalized"] - v5_gt).mean(axis=(1, 2)).tolist()
    checks = replication_checks(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
    )
    _close(value.get("base_rows"), base_rows, "base rows")
    _close(value.get("candidate_rows"), candidate_rows, "candidate rows")
    _close(value.get("base_v5_dense"), base_v5_dense, "base v5")
    _close(value.get("candidate_v5_dense"), candidate_v5_dense, "candidate v5")
    if value.get("base_v5_prediction_sha256") != tensor_sha256(torch.from_numpy(arrays["base_v5_normalized"])):
        raise ValueError("base v5 prediction hash changed")
    if value.get("candidate_v5_prediction_sha256") != tensor_sha256(torch.from_numpy(arrays["candidate_v5_normalized"])):
        raise ValueError("candidate v5 prediction hash changed")
    status = "PASS" if all(checks.values()) else "FAIL"
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    if (
        value.get("checks") != checks
        or value.get("status") != status
        or value.get("failed_checks") != failed_checks
        or value.get("authorizes_common_descent_response6") is not (status == "PASS")
    ):
        raise ValueError("metric replication status changed")
    expected_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "failed_v94_binding_id": value["failed_v94_binding_id"],
            "selected_radius": SELECTED_RADIUS,
            "replication_panels": value["replication_panels"],
            "response_maps_sha256": hashes["response_maps"],
            "checks": checks,
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("metric replication binding ID changed")
    _finite_tree(value, "metric replication report")
    print(f"[METRIC_REPLICATION_{status}] Teacher-v9.4.1 metric replication integrity")
    print("[PASS] sealed v9.4 direction/radius and fresh v5r4 binding verified")
    print("[PASS] three new paired noise/timestep panels and metrics recomputed")
    print("[PASS] no checkpoint, room_0102/0201 arrays or paper-test access")
    print("[OK] failed checks:", failed_checks)


if __name__ == "__main__":
    main()

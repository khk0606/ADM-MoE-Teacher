#!/usr/bin/env python3
"""Deep validator for the Teacher-v9.2 hotspot/trust response grid."""

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

from relational_teacher_v9_all_sittable_contract import sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import canonical_sha256  # noqa: E402
from relational_teacher_v9_lora_runtime import tensor_sha256  # noqa: E402
from relational_teacher_v91_active_support_objective import (  # noqa: E402
    active_support_macro_loss,
)
from relational_teacher_v92_hotspot_trust_contract import (  # noqa: E402
    CANDIDATES,
    GRAD_CLIP,
    LEARNING_RATE,
    SCHEMA,
    SEED,
    STEPS_PER_CANDIDATE,
    TRAIN_SCENE,
    rank_candidates,
    response_checks,
)
from relational_teacher_v92_hotspot_trust_objective import (  # noqa: E402
    hotspot_listwise_macro_loss,
    hotspot_margin_macro_loss,
)
from preflight_relational_teacher_v92_hotspot_trust import (  # noqa: E402
    _objective_values,
    _validate_failed_corrected,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _validate_loss_response,
    _validate_preflight,
)


REQUIRED_PATHS = {
    "failed_corrected_overfit_summary",
    "failed_corrected_overfit_maps",
    "selected_response_report",
    "preflight_report",
    "metric_policy",
    "dataset_index",
    "source_dataset_index",
    "v5_split",
    "stats_file",
    "original_checkpoint",
    "v5_checkpoint",
    "v5_evidence_report",
    "objective",
    "contract",
    "runner",
    "validator",
    "response_maps",
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
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(label + " numeric value changed")
        return
    if left != right:
        raise ValueError(label + " changed")


def _values(arrays: Mapping[str, np.ndarray], prediction: np.ndarray) -> Mapping[str, object]:
    physical = torch.from_numpy(prediction.astype(np.float32))
    targets = torch.from_numpy(
        np.stack([arrays["instance_targets"]] * 2).astype(np.float32)
    )
    masks = torch.from_numpy(
        np.stack([arrays["verified_object_mask"]] * 2)
    )
    active, active_rows = active_support_macro_loss(physical, targets, masks)
    margin, margin_rows = hotspot_margin_macro_loss(physical, targets, masks)
    listwise, listwise_rows = hotspot_listwise_macro_loss(physical, targets, masks)
    # Reuse the exact metric implementation used by the runner.  The small
    # tensor object only supplies the already recomputed loss values.
    objective = {
        "active_support_macro": active,
        "per_instance_active_support": active_rows,
        "hotspot_margin_macro": margin,
        "per_instance_hotspot_margin": margin_rows,
        "hotspot_listwise_macro": listwise,
        "per_instance_hotspot_listwise": listwise_rows,
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
        raise ValueError("Teacher-v9.2 hotspot/trust schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("diffusion_steps") != 500
        or value.get("steps_per_candidate") != STEPS_PER_CANDIDATE
        or value.get("training_timesteps") != [50, 125, 200, 275, 350, 425]
        or value.get("monitor_timestep") != 125
        or value.get("learning_rate") != LEARNING_RATE
        or value.get("grad_clip") != GRAD_CLIP
        or value.get("candidate_grid") != list(CANDIDATES)
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("heldout_train_scene_metadata_only") != "room_0102"
        or value.get("development_scene_metadata_only") != "room_0201"
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("fresh_zero_lora_state_restored_per_candidate") is not True
        or value.get("failed_corrected_checkpoint_loaded") is not False
        or value.get("serialized_model_state") is not False
        or value.get("authorizes_overfit120") is not False
        or value.get("authorizes_calibration") is not False
        or value.get("authorizes_long_training") is not False
        or value.get("authorizes_development_evaluation") is not False
        or value.get("authorizes_paper_test") is not False
    ):
        raise ValueError("Teacher-v9.2 hotspot/trust protocol changed")
    lora = value.get("lora")
    if not isinstance(lora, Mapping) or int(lora.get("module_count", 0)) != 31:
        raise ValueError("LoRA inventory changed")

    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("hotspot/trust path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("hotspot/trust path-hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("hotspot/trust-bound file changed: " + name)
    if report_file.parent != Path(str(paths["response_maps"])).resolve().parent:
        raise ValueError("report/map output directories differ")
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("response grid persisted forbidden model state")

    failure = _validate_failed_corrected(
        Path(str(paths["failed_corrected_overfit_summary"]))
    )
    response, _ = _validate_loss_response(Path(str(paths["selected_response_report"])))
    preflight, _ = _validate_preflight(Path(str(paths["preflight_report"])))
    if (
        value.get("failed_corrected_binding_id") != failure.get("binding_id")
        or value.get("selected_response_binding_id") != response.get("binding_id")
        or value.get("preflight_binding_id") != preflight.get("binding_id")
    ):
        raise ValueError("hotspot/trust authority binding changed")

    maps_file = Path(str(paths["response_maps"])).resolve()
    with np.load(maps_file, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required_arrays = {
        "xyz",
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
        "base",
        "candidates",
        "base_normalized",
        "candidates_normalized",
        "candidate_names",
        "v5_gt_normalized",
        "base_v5_normalized",
        "candidate_v5_normalized",
    }
    if set(arrays) != required_arrays:
        raise ValueError("response array inventory changed")
    expected_shapes = {
        "xyz": (8192, 3),
        "verified_object_mask": (3, 8192),
        "verified_positive_mask": (8192,),
        "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,),
        "instance_targets": (3, 8192, 6),
        "base": (2, 8192, 6),
        "candidates": (3, 2, 8192, 6),
        "base_normalized": (2, 8192, 6),
        "candidates_normalized": (3, 2, 8192, 6),
        "candidate_names": (3,),
        "v5_gt_normalized": (3, 8192, 6),
        "base_v5_normalized": (3, 8192, 6),
        "candidate_v5_normalized": (3, 3, 8192, 6),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape changed")
    if tuple(str(item) for item in arrays["candidate_names"].tolist()) != tuple(
        row["name"] for row in CANDIDATES
    ):
        raise ValueError("candidate array order changed")
    for name, array in arrays.items():
        if name != "candidate_names" and not np.isfinite(array).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "base", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    if np.any(arrays["verified_object_mask"].sum(axis=0) > 1):
        raise ValueError("verified object masks overlap")
    if not np.array_equal(
        arrays["verified_positive_mask"],
        arrays["verified_object_mask"].any(axis=0),
    ):
        raise ValueError("verified positive mask changed")
    with np.load(Path(str(paths["failed_corrected_overfit_maps"])), allow_pickle=False) as source:
        for name in (
            "xyz",
            "verified_object_mask",
            "unknown_sittable_mask",
            "explicit_negative_mask",
            "instance_targets",
        ):
            if not np.array_equal(arrays[name], source[name]):
                raise ValueError("response/failure " + name + " differs")

    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(CANDIDATES):
        raise ValueError("candidate rows changed")
    base_values = _values(arrays, arrays["base"])
    v5_gt = arrays["v5_gt_normalized"]
    base_v5_dense = np.square(arrays["base_v5_normalized"] - v5_gt).mean(axis=(1, 2))
    recomputed_rows = []
    for index, (specification, row) in enumerate(zip(CANDIDATES, rows)):
        if any(row.get(name) != expected for name, expected in specification.items()):
            raise ValueError("candidate specification changed")
        if row.get("steps") != STEPS_PER_CANDIDATE or row.get("learning_rate") != LEARNING_RATE:
            raise ValueError("candidate update protocol changed")
        logs = row.get("update_log")
        if not isinstance(logs, list) or len(logs) != STEPS_PER_CANDIDATE:
            raise ValueError("candidate update log changed")
        if [item.get("step") for item in logs] != list(range(1, STEPS_PER_CANDIDATE + 1)):
            raise ValueError("candidate update steps changed")
        if [item.get("timestep") for item in logs] != [50, 125, 200, 275, 350, 425]:
            raise ValueError("candidate update timesteps changed")
        after = _values(arrays, arrays["candidates"][index])
        candidate_v5_dense = np.square(
            arrays["candidate_v5_normalized"][index] - v5_gt
        ).mean(axis=(1, 2))
        checks = response_checks(
            before=base_values,
            after=after,
            base_v5_dense=base_v5_dense.tolist(),
            candidate_v5_dense=candidate_v5_dense.tolist(),
        )
        _close(row["before"], base_values, "candidate.before")
        _close(row["after"], after, "candidate.after")
        _close(row["base_v5_dense"], base_v5_dense.tolist(), "candidate.base_v5")
        _close(
            row["candidate_v5_dense"],
            candidate_v5_dense.tolist(),
            "candidate.candidate_v5",
        )
        if row.get("checks") != checks:
            raise ValueError("candidate checks changed")
        eligible = all(checks.values())
        if row.get("eligible") is not eligible or row.get("failed_checks") != sorted(
            name for name, passed in checks.items() if not passed
        ):
            raise ValueError("candidate eligibility changed")
        if row.get("prediction_sha256") != tensor_sha256(
            torch.from_numpy(arrays["candidates_normalized"][index])
        ) or row.get("v5_prediction_sha256") != tensor_sha256(
            torch.from_numpy(arrays["candidate_v5_normalized"][index])
        ):
            raise ValueError("candidate prediction hash changed")
        recomputed_rows.append({**row, "before": base_values, "after": after, "checks": checks})

    order = rank_candidates(recomputed_rows)
    selected = order[0] if order else None
    status = "PASS" if selected is not None else "FAIL"
    if (
        value.get("eligible_selection_order") != order
        or value.get("selected_candidate") != selected
        or value.get("status") != status
        or value.get("authorizes_early_rollout_canary") is not (status == "PASS")
    ):
        raise ValueError("response selection/status changed")
    expected_failed = [] if status == "PASS" else ["at_least_one_candidate_admissible"]
    if value.get("failed_checks") != expected_failed:
        raise ValueError("top-level failed checks changed")
    expected_binding = canonical_sha256(
        {
            "preflight_binding_id": value["preflight_binding_id"],
            "selected_response_binding_id": value["selected_response_binding_id"],
            "failed_corrected_binding_id": value["failed_corrected_binding_id"],
            "candidate_grid": value["candidate_grid"],
            "selected_candidate": selected,
            "response_maps_sha256": hashes["response_maps"],
        }
    )
    if value.get("binding_id") != expected_binding:
        raise ValueError("response binding ID changed")
    _finite_tree(value, "hotspot/trust report")
    print(f"[HOTSPOT_TRUST_{status}] Teacher-v9.2 response integrity")
    print("[PASS] sealed v9.1 failure and fresh v5r4 authority verified")
    print("[PASS] all three per-object top-k responses and trust gates recomputed")
    print("[PASS] exact 3x6 train-only updates; no checkpoint or held-out access")
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

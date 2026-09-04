#!/usr/bin/env python3
"""Deep validator for the Teacher-v9.1 active-support response grid."""

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

from relational_teacher_v9_all_sittable_metrics import all_instance_metrics  # noqa: E402
from relational_teacher_v9_all_sittable_contract import sha256_file  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import canonical_sha256  # noqa: E402
from relational_teacher_v9_overfit_smoke_contract import sanitize_metric_nonfinite  # noqa: E402
from relational_teacher_v91_active_support_objective import (  # noqa: E402
    active_support_macro_loss,
    within_object_ranking_loss,
)
from relational_teacher_v91_loss_response_contract import (  # noqa: E402
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
from preflight_relational_teacher_v91_loss_response import (  # noqa: E402
    _validate_failure,
)
from run_relational_teacher_v9_one_scene_overfit_smoke import (  # noqa: E402
    _validate_preflight,
)


REQUIRED_PATHS = {
    "failure_summary",
    "failure_maps",
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
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(label + " length changed")
        for index, (one, two) in enumerate(zip(left, right)):
            _close(one, two, f"{label}[{index}]")
    elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
        right, (int, float)
    ) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(label + " numeric value changed")
    elif left != right:
        raise ValueError(label + " changed")


def _values(
    prediction: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    xyz: np.ndarray,
    negative: np.ndarray,
    unknown: np.ndarray,
) -> Mapping[str, object]:
    physical = torch.from_numpy(prediction.astype(np.float32))
    target_tensor = torch.from_numpy(np.stack([targets] * 2).astype(np.float32))
    mask_tensor = torch.from_numpy(np.stack([masks] * 2))
    active, active_rows = active_support_macro_loss(
        physical, target_tensor, mask_tensor
    )
    ranking, ranking_rows = within_object_ranking_loss(
        physical, target_tensor, mask_tensor
    )
    metrics = [
        sanitize_metric_nonfinite(
            all_instance_metrics(
                prediction[index],
                targets,
                ("bed_01", "chair_01", "chair_06"),
                xyz,
                masks,
                negative,
                unknown,
            )
        )
        for index in range(2)
    ]
    return {
        "active_support_macro": float(active.item()),
        "within_object_ranking": float(ranking.item()),
        "per_instance_active_support": [
            float(value) for value in active_rows.mean(dim=0).tolist()
        ],
        "per_instance_ranking": [
            float(value) for value in ranking_rows.mean(dim=0).tolist()
        ],
        "prompt_invariance": float(
            np.square(prediction[0, masks.any(axis=0)] - prediction[1, masks.any(axis=0)]).mean()
        ),
        "negative_mean": float(
            np.mean([row["explicit_negative_mean"] for row in metrics])
        ),
        "per_prompt_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_file = args.report.expanduser().resolve()
    value = json.loads(report_file.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Teacher-v9.1 response schema/status changed")
    if (
        value.get("seed") != SEED
        or value.get("diffusion_steps") != 500
        or value.get("steps_per_candidate") != STEPS_PER_CANDIDATE
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
        or value.get("failed_smoke_checkpoint_loaded") is not False
        or value.get("serialized_model_state") is not False
    ):
        raise ValueError("Teacher-v9.1 response protocol changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or set(paths) != REQUIRED_PATHS:
        raise ValueError("response path inventory changed")
    if not isinstance(hashes, Mapping) or set(hashes) != REQUIRED_PATHS:
        raise ValueError("response path-hash inventory changed")
    for name in REQUIRED_PATHS:
        path = Path(str(paths[name])).resolve()
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError("response-bound file changed: " + name)
    if list(report_file.parent.glob("*.pt")) or list(report_file.parent.glob("*.pth")):
        raise ValueError("response grid persisted model state")
    failed = _validate_failure(Path(str(paths["failure_summary"])))
    preflight, _ = _validate_preflight(Path(str(paths["preflight_report"])))
    if (
        value.get("failed_smoke_binding_id") != failed.get("binding_id")
        or value.get("preflight_binding_id") != preflight.get("binding_id")
    ):
        raise ValueError("response authority binding changed")

    with np.load(Path(str(paths["response_maps"])), allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required_arrays = {
        "xyz",
        "verified_object_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
        "instance_targets",
        "base",
        "candidates",
        "candidate_names",
        "v5_gt_normalized",
        "base_v5_normalized",
        "candidate_v5_normalized",
    }
    if set(arrays) != required_arrays:
        raise ValueError("response array inventory changed")
    shapes = {
        "xyz": (8192, 3),
        "verified_object_mask": (3, 8192),
        "unknown_sittable_mask": (8192,),
        "explicit_negative_mask": (8192,),
        "instance_targets": (3, 8192, 6),
        "base": (2, 8192, 6),
        "candidates": (3, 2, 8192, 6),
        "candidate_names": (3,),
        "v5_gt_normalized": (3, 8192, 6),
        "base_v5_normalized": (3, 8192, 6),
        "candidate_v5_normalized": (3, 3, 8192, 6),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape changed")
    if tuple(str(item) for item in arrays["candidate_names"].tolist()) != tuple(
        row["name"] for row in CANDIDATES
    ):
        raise ValueError("response candidate array order changed")
    for name, array in arrays.items():
        if name != "candidate_names" and not np.isfinite(array).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "base", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    with np.load(Path(str(paths["failure_maps"])), allow_pickle=False) as source:
        if not np.array_equal(arrays["xyz"], source["xyz"]):
            raise ValueError("response/failure XYZ differs")
        if not np.array_equal(arrays["instance_targets"], source["instance_targets"]):
            raise ValueError("response/failure target differs")
        if not np.array_equal(
            arrays["verified_object_mask"], source["verified_object_mask"]
        ):
            raise ValueError("response/failure object masks differ")
        if not np.array_equal(
            arrays["unknown_sittable_mask"], source["unknown_sittable_mask"]
        ) or not np.array_equal(
            arrays["explicit_negative_mask"], source["explicit_negative_mask"]
        ):
            raise ValueError("response/failure semantic masks differ")

    base_values = _values(
        arrays["base"],
        arrays["instance_targets"],
        arrays["verified_object_mask"],
        arrays["xyz"],
        arrays["explicit_negative_mask"],
        arrays["unknown_sittable_mask"],
    )
    base_v5_dense = np.square(
        arrays["base_v5_normalized"] - arrays["v5_gt_normalized"]
    ).mean(axis=(1, 2))
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(CANDIDATES):
        raise ValueError("response candidate rows changed")
    for index, (expected, row) in enumerate(zip(CANDIDATES, rows)):
        for key, expected_value in expected.items():
            if row.get(key) != expected_value:
                raise ValueError("response candidate definition changed")
        if row.get("steps") != STEPS_PER_CANDIDATE or row.get(
            "learning_rate"
        ) != LEARNING_RATE:
            raise ValueError("response update count/LR changed")
        if row.get("before") != rows[0].get("before"):
            raise ValueError("candidate baselines differ")
        _close(base_values, row["before"], "candidate before")
        after = _values(
            arrays["candidates"][index],
            arrays["instance_targets"],
            arrays["verified_object_mask"],
            arrays["xyz"],
            arrays["explicit_negative_mask"],
            arrays["unknown_sittable_mask"],
        )
        _close(after, row["after"], "candidate after")
        candidate_v5_dense = np.square(
            arrays["candidate_v5_normalized"][index] - arrays["v5_gt_normalized"]
        ).mean(axis=(1, 2))
        _close(base_v5_dense.tolist(), row["base_v5_dense"], "base v5")
        _close(candidate_v5_dense.tolist(), row["candidate_v5_dense"], "candidate v5")
        checks = response_checks(
            before=base_values,
            after=after,
            base_v5_dense=base_v5_dense.tolist(),
            candidate_v5_dense=candidate_v5_dense.tolist(),
        )
        if row.get("checks") != checks or row.get("eligible") is not all(checks.values()):
            raise ValueError("response eligibility arithmetic changed")
        if row.get("failed_checks") != sorted(
            name for name, passed in checks.items() if not passed
        ):
            raise ValueError("response failed checks changed")
        logs = row.get("update_log")
        if not isinstance(logs, list) or [log.get("step") for log in logs] != [1, 2, 3]:
            raise ValueError("response update log changed")
        if any(float(log.get("gradient_l2_before_clip", 0.0)) <= 0.0 for log in logs):
            raise ValueError("response gradient log is invalid")

    order = rank_candidates(rows)
    selected = order[0] if order else None
    expected_status = "PASS" if selected is not None else "FAIL"
    if (
        value.get("eligible_selection_order") != order
        or value.get("selected_candidate") != selected
        or value.get("status") != expected_status
    ):
        raise ValueError("response ranking/status changed")
    checks = value.get("checks")
    expected_checks = {
        "failed_smoke_diagnosis_bound": True,
        "fresh_v5r4_zero_init": True,
        "exact_candidate_grid": True,
        "three_updates_per_candidate": True,
        "only_lora_updated": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_checkpoint_saved": True,
        "at_least_one_candidate_admissible": selected is not None,
    }
    if checks != expected_checks:
        raise ValueError("response top-level checks changed")
    if value.get("failed_checks") != sorted(
        name for name, passed in checks.items() if not passed
    ):
        raise ValueError("response top-level failed checks changed")
    if value.get("authorizes_corrected_one_scene_overfit") != (
        expected_status == "PASS"
    ):
        raise ValueError("corrected-smoke authorization changed")
    if any(
        value.get(name) is not False
        for name in (
            "authorizes_response3",
            "authorizes_calibration",
            "authorizes_long_training",
            "authorizes_development_evaluation",
            "authorizes_paper_test",
        )
    ):
        raise ValueError("response grid over-authorized later stages")
    binding = canonical_sha256(
        {
            "failed_smoke_binding_id": value["failed_smoke_binding_id"],
            "preflight_binding_id": value["preflight_binding_id"],
            "candidate_grid": value["candidate_grid"],
            "selected_candidate": selected,
            "response_maps_sha256": hashes["response_maps"],
        }
    )
    if value.get("binding_id") != binding:
        raise ValueError("response binding ID changed")
    _finite_tree(value, "response report")
    print(f"[LOSS_RESPONSE_{expected_status}] Teacher-v9.1 response integrity")
    print("[PASS] sealed failure diagnosis and fresh v5r4 binding verified")
    print("[PASS] three-object support/ranking responses and v5 retention recomputed")
    print("[PASS] exact 3x3 updates; no checkpoint or held-out array access")
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", value["failed_checks"])


if __name__ == "__main__":
    main()

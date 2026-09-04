#!/usr/bin/env python3
"""CPU-only regression tests for the Teacher-v9 overfit-smoke contract."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from relational_teacher_v9_lora_preflight_contract import RELATIVE_SELECTION_RULES
from relational_teacher_v9_overfit_smoke_contract import (
    evaluate_smoke_panel,
    sanitize_metric_nonfinite,
)


def _instance(mae: float) -> dict:
    return {
        "active_support_mae": mae,
        "soft_recall": 0.90,
        "topk_overlap": 0.80,
        "hotspot_centroid_distance_xy": 0.10,
    }


def _row(bed: float, chair: float, high: float) -> dict:
    return {
        "instances": {
            "bed_01": _instance(bed),
            "chair_01": _instance(chair),
            "chair_06": _instance(high),
        },
        "explicit_negative_mean": 0.02,
        "explicit_negative_max": 0.20,
    }


def main() -> None:
    row_ids = ("room_0101|sit_watch_v1", "room_0101|sit_write_v1")
    base = {name: _row(0.08, 0.09, 0.09) for name in row_ids}
    candidate = {name: _row(0.0802, 0.08, 0.08) for name in row_ids}
    presence = {
        name: {
            "every_verified_instance_has_soft_recall": True,
            "every_verified_instance_has_topk_overlap": True,
            "every_verified_instance_has_bounded_active_support_mae": True,
            "every_verified_instance_has_bounded_hotspot_centroid": True,
            "explicit_negative_mean_bounded": True,
            "explicit_negative_max_bounded": True,
        }
        for name in row_ids
    }
    checks = evaluate_smoke_panel(
        candidate_rows=candidate,
        base_rows=base,
        absolute_presence=presence,
        base_prompt_invariance=0.01,
        candidate_prompt_invariance=0.0101,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
        relative_rules=RELATIVE_SELECTION_RULES,
    )
    if not all(checks.values()):
        raise AssertionError("valid three-object smoke panel was rejected")

    missing_chair = copy.deepcopy(candidate)
    missing_chair[row_ids[0]]["instances"]["chair_01"]["active_support_mae"] = 0.09
    failed = evaluate_smoke_panel(
        candidate_rows=missing_chair,
        base_rows=base,
        absolute_presence=presence,
        base_prompt_invariance=0.01,
        candidate_prompt_invariance=0.0101,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
        relative_rules=RELATIVE_SELECTION_RULES,
    )
    if failed[row_ids[0] + "|normal_chair_mae_improves"] is not False:
        raise AssertionError("non-improving normal Chair was accepted")

    absent = copy.deepcopy(presence)
    absent[row_ids[1]]["every_verified_instance_has_soft_recall"] = False
    failed = evaluate_smoke_panel(
        candidate_rows=candidate,
        base_rows=base,
        absolute_presence=absent,
        base_prompt_invariance=0.01,
        candidate_prompt_invariance=0.0101,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
        relative_rules=RELATIVE_SELECTION_RULES,
    )
    if failed[row_ids[1] + "|absolute_presence"] is not False:
        raise AssertionError("missing verified object was accepted")
    sanitized = sanitize_metric_nonfinite(
        {
            "instances": {
                "chair_01": {"hotspot_centroid_distance_xy": float("inf")}
            }
        }
    )
    if sanitized["instances"]["chair_01"]["hotspot_centroid_distance_xy"] != 1e30:
        raise AssertionError("absent-object centroid was not converted fail-closed")
    try:
        sanitize_metric_nonfinite({"active_support_mae": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("NaN metric was accepted")

    source = Path(__file__).with_name(
        "run_relational_teacher_v9_one_scene_overfit_smoke.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(calls) != 1:
        raise AssertionError("runner must contain exactly one train-scene array load")
    expression = calls[0].args[-1]
    slice_value = expression.slice if isinstance(expression, ast.Subscript) else None
    if isinstance(slice_value, ast.Index):  # Python 3.8 compatibility.
        slice_value = slice_value.value
    if not (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "records"
        and isinstance(slice_value, ast.Name)
        and slice_value.id == "TRAIN_SCENE"
    ):
        raise AssertionError("runner array load is not statically bound to room_0101")
    if "torch.save" in source or "save_trainable_state" in source:
        raise AssertionError("overfit smoke must not persist candidate weights")

    print("[PASS] Teacher-v9 one-scene overfit smoke CPU contract")
    print("[PASS] all-three presence, strict Chair gain and retention gates")
    print("[PASS] exactly one room_0101 array load and no checkpoint writer")


if __name__ == "__main__":
    main()

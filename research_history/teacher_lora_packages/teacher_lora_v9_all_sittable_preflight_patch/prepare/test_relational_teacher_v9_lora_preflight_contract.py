#!/usr/bin/env python3
"""CPU regression tests for the Teacher-v9 LoRA preflight contract."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np

from relational_teacher_v9_all_sittable_metrics import (
    all_instance_metrics,
    simultaneous_presence_checks,
)
from relational_teacher_v9_all_sittable_contract import (
    PROMPTS,
    TEACHER_FORWARD_INPUTS,
    atomic_write_json,
    sha256_file,
)
from relational_teacher_v9_lora_preflight_contract import (
    ABSOLUTE_PRESENCE_LIMITS,
    EXPECTED_TARGETS,
    OBJECTIVE_WEIGHTS,
    RELATIVE_SELECTION_RULES,
    load_train_scene_bundle,
)


def numpy_equal_instance_macro(prediction, targets, masks):
    rows = []
    for instance in range(3):
        difference = (prediction - targets[instance]) ** 2
        rows.append(float(difference[masks[instance]].mean()))
    return float(np.mean(rows)), rows


def check_real_bundle_shapes() -> None:
    """Exercise the exact [8192,6] winner tensor written by the v9 builder."""

    with tempfile.TemporaryDirectory(prefix="teacher_v9_preflight_contract_") as raw:
        root = Path(raw)
        source_root = root / "source"
        dataset_root = root / "dataset"
        source_root.mkdir()
        scene_dir = dataset_root / "scenes" / "room_0101"
        scene_dir.mkdir(parents=True)
        points = np.zeros((8192, 6), dtype=np.float32)
        points[:, 0] = np.linspace(0.0, 8.0, 8192, dtype=np.float32)
        points_file = source_root / "room_0101_points.npz"
        np.savez_compressed(points_file, points=points)

        instance_ids = np.zeros(8192, dtype=np.int64)
        category_ids = np.zeros(8192, dtype=np.int64)
        verified = np.zeros((3, 8192), dtype=bool)
        for slot, (start, end, numeric, category) in enumerate(
            ((0, 10, 1, 2), (10, 20, 2, 1), (20, 30, 3, 1))
        ):
            verified[slot, start:end] = True
            instance_ids[start:end] = numeric
            category_ids[start:end] = category
        instance_ids[30:40] = 4
        category_ids[30:40] = 1
        instance_ids[40:43] = np.asarray([5, 6, 7], dtype=np.int64)
        category_ids[40:43] = np.asarray([3, 4, 5], dtype=np.int64)
        targets = np.zeros((3, 8192, 6), dtype=np.float32)
        for slot in range(3):
            targets[slot, verified[slot]] = np.float32(0.9)
        all_target = targets.max(axis=0)
        winner = np.argmax(targets, axis=0).astype(np.int8)
        winner[all_target <= 0.0] = -1
        consensus_file = scene_dir / "all_sittable_consensus.npz"
        np.savez_compressed(
            consensus_file,
            xyz=points[:, :3],
            source_indices=np.arange(8192, dtype=np.int64),
            instance_ids=instance_ids,
            category_ids=category_ids,
            eligible_instance_ids=np.asarray(("bed_01", "chair_01", "chair_06")),
            eligible_numeric_instance_ids=np.asarray((1, 2, 3), dtype=np.int64),
            verified_object_mask=verified,
            verified_positive_mask=verified.any(axis=0),
            instance_affordance=targets,
            instance_active_mask=targets >= 0.30,
            all_sittable_affordance=all_target,
            unknown_sittable_mask=instance_ids == 4,
            environment_aux_mask=instance_ids == 0,
            explicit_negative_mask=np.isin(category_ids, (3, 4, 5)),
            winner_instance_slot=winner,
        )
        relative = "scenes/room_0101/all_sittable_consensus.npz"
        manifest = {
            "schema": "relational_teacher_v9_all_sittable_scene_v1",
            "status": "ALL_SITTABLE_SCENE_PASS",
            "scene_id": "room_0101",
            "split": "train",
            "point_count": 8192,
            "contact_dim": 6,
            "replica_count": 6,
            "verified_target_instances": ["bed_01", "chair_01", "chair_06"],
            "unknown_sittable_instances_are_negative": False,
            "primary_loss_policy": "equal_instance_macro_on_verified_object_masks",
            "environment_contact_is_auxiliary_only": True,
            "relation_or_distance_used": False,
            "teacher_forward_inputs": list(TEACHER_FORWARD_INPUTS),
            "development_used_for_parameter_selection": False,
            "all_sittable_consensus_file": relative,
            "all_sittable_consensus_sha256": sha256_file(consensus_file),
            "failed_checks": [],
        }
        manifest_file = scene_dir / "manifest.json"
        atomic_write_json(manifest_file, manifest)
        rows = [
            {
                "prompt_id": prompt_id,
                "text": PROMPTS[prompt_id],
                "all_sittable_gt_file": relative,
                "all_sittable_gt_sha256": sha256_file(consensus_file),
                "all_sittable_array_key": "all_sittable_affordance",
                "supervision": "scene_level_all_verified_sittable_consensus",
                "auxiliary_replica_ids": [f"replica_{index:02d}" for index in range(6)],
            }
            for prompt_id in sorted(PROMPTS)
        ]
        record = {
            "scene_id": "room_0101",
            "split": "train",
            "source_scene_record": {
                "scene_id": "room_0101",
                "points_file": points_file.name,
                "points_sha256": sha256_file(points_file),
            },
            "manifest_file": "scenes/room_0101/manifest.json",
            "manifest_sha256": sha256_file(manifest_file),
            "verified_target_instances": ["bed_01", "chair_01", "chair_06"],
            "unknown_sittable_instances_are_negative": False,
            "replica_count": 6,
            "row_count": 2,
            "rows": rows,
        }
        loaded = load_train_scene_bundle(dataset_root, source_root, record)
        if loaded["all_target"].shape != (8192, 6):
            raise AssertionError("real-shape all-sittable bundle did not load")


def main() -> None:
    check_real_bundle_shapes()
    if EXPECTED_TARGETS != {
        "room_0101": ("bed_01", "chair_01", "chair_06"),
        "room_0102": ("bed_01", "chair_05", "chair_06"),
    }:
        raise AssertionError("Teacher-v9 verified targets changed")
    if OBJECTIVE_WEIGHTS["instance_macro_primary"] != 1.0:
        raise AssertionError("instance macro is no longer the primary loss")
    if RELATIVE_SELECTION_RULES[
        "chair_active_support_mae_requires_strict_improvement"
    ] is not True:
        raise AssertionError("both Chair roles must require strict improvement")

    point_count = 14
    target_stack = np.zeros((3, point_count, 6), dtype=np.float32)
    masks = np.zeros((3, point_count), dtype=bool)
    masks[0, 0:8] = True
    masks[1, 8:11] = True
    masks[2, 11:13] = True
    target_stack[0, masks[0]] = 0.2
    target_stack[1, masks[1]] = 1.0
    target_stack[2, masks[2]] = 0.5
    prediction = np.zeros((point_count, 6), dtype=np.float32)
    macro, rows = numpy_equal_instance_macro(prediction, target_stack, masks)
    expected = float(np.mean([0.04, 1.0, 0.25]))
    if not np.isclose(macro, expected, atol=1e-7, rtol=0.0):
        raise AssertionError("Bed/normal-Chair/High-Chair are not equally weighted")
    point_weighted = float(
        sum(rows[index] * int(masks[index].sum()) for index in range(3))
        / int(masks.sum())
    )
    if np.isclose(macro, point_weighted, atol=1e-4, rtol=0.0):
        raise AssertionError("test failed to distinguish macro from point weighting")

    xyz = np.stack(
        [
            np.linspace(0.0, 6.0, point_count, dtype=np.float32),
            np.zeros(point_count, dtype=np.float32),
            np.zeros(point_count, dtype=np.float32),
        ],
        axis=-1,
    )
    target_stack.fill(0.0)
    target_stack[0, 0:3] = 0.9
    target_stack[1, 4:7] = 0.9
    target_stack[2, 8:11] = 0.9
    masks.fill(False)
    masks[0, 0:4] = True
    masks[1, 4:8] = True
    masks[2, 8:12] = True
    all_target = target_stack.max(axis=0)
    negative = np.zeros(point_count, dtype=bool)
    negative[12] = True
    unknown = np.zeros(point_count, dtype=bool)
    unknown[13] = True
    names = ("bed_01", "chair_01", "chair_06")

    def checks(value):
        metrics = all_instance_metrics(
            value, target_stack, names, xyz, masks, negative, unknown
        )
        return simultaneous_presence_checks(
            metrics, **ABSOLUTE_PRESENCE_LIMITS
        )

    if not all(checks(all_target).values()):
        raise AssertionError("exact all-three GT must pass")
    bed_only_metrics = all_instance_metrics(
        target_stack[0], target_stack, names, xyz, masks, negative, unknown
    )
    if not np.isinf(
        bed_only_metrics["instances"]["chair_01"][
            "hotspot_centroid_distance_xy"
        ]
    ):
        raise AssertionError("missing Chair must expose an infinite centroid distance")
    if all(checks(target_stack[0]).values()):
        raise AssertionError("Bed-only prediction must fail")
    if all(checks(np.maximum(target_stack[0], target_stack[2])).values()):
        raise AssertionError("missing normal Chair must fail")
    if all(checks(np.maximum(target_stack[0], target_stack[1])).values()):
        raise AssertionError("missing High Chair must fail")
    hotspot = all_target.copy()
    hotspot[negative] = 1.0
    if all(checks(hotspot).values()):
        raise AssertionError("explicit-negative hotspot must fail")

    # Unknown Sit points cannot affect any v9 task term.  This NumPy reference
    # checks the role partition independently of the CUDA implementation.
    verified = masks.any(axis=0)
    environment = ~(verified | unknown | negative)
    if np.any(unknown & (verified | environment | negative)):
        raise AssertionError("synthetic role masks overlap")
    shifted = all_target.copy()
    shifted[unknown] = 1.0
    for mask in (verified, environment, negative):
        if not np.array_equal(shifted[mask], all_target[mask]):
            raise AssertionError("unknown Sit perturbation leaked into a task role")

    objective_file = Path(__file__).with_name(
        "relational_teacher_v9_lora_objective.py"
    )
    source = objective_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required = {
        "physical_prediction",
        "equal_instance_macro_loss",
        "paired_prompt_invariance_loss",
        "explicit_negative_addition_loss",
        "preservation_loss",
        "all_sittable_objective",
    }
    if not required.issubset(functions):
        raise AssertionError("Teacher-v9 objective API is incomplete")
    forbidden = ("outside =", "~high_desk", "target_instance_id", "distance")
    if any(token in source for token in forbidden):
        raise AssertionError("old target-specific suppression leaked into v9 objective")

    print("[PASS] Teacher-v9 all-sittable LoRA preflight CPU contract")
    print("[PASS] Bed/normal-Chair/High-Chair are exact equal-macro losses")
    print("[PASS] Bed-only and either missing-Chair counterexamples fail")
    print("[PASS] unknown Sit points are ignored; explicit-negative hotspot fails")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU regression tests for the Teacher-v9 all-sittable data contract."""

from __future__ import annotations

import copy

import numpy as np

from relational_teacher_v9_all_sittable_contract import (
    GROUP_ORDER,
    aggregate_all_sittable,
    assert_all_sittable_contract,
    assert_teacher_forward_inputs,
    classify_dense_rows,
    mask_maps_to_target_and_environment,
    nearest_rank_quantile,
    partition_sittable_instances,
    replica_schedule,
)
from relational_teacher_v9_all_sittable_metrics import (
    all_instance_metrics,
    simultaneous_presence_checks,
)


def make_rows():
    rows = []
    for number in range(6):
        rows.append(
            {
                "motion_id": f"room_sit_bed_{number}",
                "target_instance_id": "bed_01",
            }
        )
        rows.append(
            {
                "motion_id": f"room_sit_normal_{number}",
                "target_instance_id": "chair_01",
            }
        )
        rows.append(
            {
                "motion_id": f"room_hc_hd_source_{number}",
                "target_instance_id": "chair_06",
                "source_collection_id": "hc_hd",
            }
        )
        rows.append(
            {
                "motion_id": f"room_sit_legacy_high_{number}",
                "target_instance_id": "chair_06",
            }
        )
    return rows


def make_maps(grouped):
    target_points = {
        "bed": (2, 3),
        "normal_chair": (4, 5),
        "high_desk_motion": (6, 7),
        "high_chair_legacy_motion": (6, 7),
    }
    maps = {}
    for group_index, group in enumerate(GROUP_ORDER):
        for row_index, row in enumerate(grouped[group]):
            value = np.zeros((13, 6), dtype=np.float32)
            value[0, 1] = np.float32(0.40 + 0.01 * row_index)
            start, end = target_points[group]
            value[start:end, :] = np.float32(0.80 + 0.02 * row_index)
            # Deliberate cross-object contamination.  The target/environment
            # filter must remove both the unverified Chair and the TV point.
            value[8:10, :] = np.float32(0.95)
            value[10:13, :] = np.float32(0.99)
            maps[str(row["motion_id"])] = value
    return maps


def expect_failure(callable_, message):
    try:
        callable_()
    except (AssertionError, KeyError, ValueError):
        return
    raise AssertionError(message)


def main():
    rows = make_rows()
    grouped, targets = classify_dense_rows(rows)
    if tuple(grouped) != GROUP_ORDER:
        raise AssertionError("four all-sit source strata changed")
    if targets != {
        "bed": "bed_01",
        "normal_chair": "chair_01",
        "high_desk_motion": "chair_06",
        "high_chair_legacy_motion": "chair_06",
    }:
        raise AssertionError("verified physical target binding changed")
    schedule = replica_schedule(grouped)
    for group in GROUP_ORDER:
        used = {row[group]["motion_id"] for row in schedule}
        if used != {row["motion_id"] for row in grouped[group]}:
            raise AssertionError("Latin replica schedule lost a source motion")

    quantile_sources = [
        np.full((2, 6), float(index) / 10.0, dtype=np.float32)
        for index in range(1, 7)
    ]
    quantile = nearest_rank_quantile(quantile_sources)
    if not np.array_equal(quantile, np.full((2, 6), 0.5, dtype=np.float32)):
        raise AssertionError("q75 must be exactly the second-largest of six")

    stable = {
        "bed_01": (1, 2),
        "chair_01": (2, 1),
        "chair_06": (3, 1),
        "chair_99": (4, 1),
        "tv_01": (5, 4),
        "desk_01": (6, 3),
        "whiteboard_01": (7, 5),
    }
    instance_ids = np.asarray(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7], dtype=np.int64
    )
    category_ids = np.asarray(
        [0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 4, 3, 5], dtype=np.int64
    )
    partition = partition_sittable_instances(stable, set(targets.values()))
    if partition["unknown_sittable_not_negative"] != ["chair_99"]:
        raise AssertionError("unverified Chair must remain unknown/ignored")

    raw_maps = make_maps(grouped)
    filtered = mask_maps_to_target_and_environment(
        grouped, targets, raw_maps, stable, instance_ids
    )
    if any(np.any(value[8:]) for value in filtered.values()):
        raise AssertionError("cross-object/unknown/TV heat survived target masking")
    aggregate = aggregate_all_sittable(grouped, targets, filtered)
    assert_all_sittable_contract(aggregate, targets)
    if set(aggregate["instance_consensus"]) != {"bed_01", "chair_01", "chair_06"}:
        raise AssertionError("High-Desk and legacy sources were counted as two objects")
    for target, expected_points in {
        "bed_01": (2, 3),
        "chair_01": (4, 5),
        "chair_06": (6, 7),
    }.items():
        value = aggregate["instance_consensus"][target]
        if not np.all(value[slice(*expected_points)] > 0.0):
            raise AssertionError(f"{target}: robust contact support disappeared")
    all_gt = aggregate["all_consensus"]
    if not all(np.any(all_gt[start:end] > 0.0) for start, end in ((2, 3), (4, 5), (6, 7))):
        raise AssertionError("all-sit GT does not contain all three verified targets")

    names = sorted(aggregate["instance_consensus"])
    targets_stack = np.stack(
        [aggregate["instance_consensus"][name] for name in names], axis=0
    )
    verified_masks = np.stack(
        [instance_ids == stable[name][0] for name in names], axis=0
    )
    xyz = np.stack(
        [np.arange(13, dtype=np.float32), np.zeros(13), np.zeros(13)], axis=-1
    )
    negative = category_ids == 4
    unknown = instance_ids == 4
    bed_only = aggregate["instance_consensus"]["bed_01"]
    bed_metrics = all_instance_metrics(
        bed_only, targets_stack, names, xyz, verified_masks, negative, unknown
    )
    bed_checks = simultaneous_presence_checks(
        bed_metrics,
        minimum_soft_recall=0.75,
        minimum_topk_overlap=0.50,
        maximum_active_support_mae=0.05,
        maximum_hotspot_centroid_distance_xy=0.60,
        maximum_negative_mean=0.05,
        maximum_negative_max=0.80,
    )
    if all(bed_checks.values()):
        raise AssertionError("Bed-only prediction was incorrectly accepted")
    bed_and_high = np.maximum(
        aggregate["instance_consensus"]["bed_01"],
        aggregate["instance_consensus"]["chair_06"],
    )
    bed_high_checks = simultaneous_presence_checks(
        all_instance_metrics(
            bed_and_high,
            targets_stack,
            names,
            xyz,
            verified_masks,
            negative,
            unknown,
        ),
        minimum_soft_recall=0.75,
        minimum_topk_overlap=0.50,
        maximum_active_support_mae=0.05,
        maximum_hotspot_centroid_distance_xy=0.60,
        maximum_negative_mean=0.05,
        maximum_negative_max=0.80,
    )
    if all(bed_high_checks.values()):
        raise AssertionError("missing normal Chair was incorrectly accepted")
    full_metrics = all_instance_metrics(
        all_gt,
        targets_stack,
        np.asarray(names),
        xyz,
        verified_masks,
        negative,
        unknown,
    )
    expect_failure(
        lambda: all_instance_metrics(
            all_gt,
            targets_stack[:2],
            names[:2],
            xyz,
            verified_masks[:2],
            negative,
            unknown,
        ),
        "two-instance metric call bypassed the all-three contract",
    )
    expect_failure(
        lambda: all_instance_metrics(
            all_gt,
            targets_stack,
            names,
            xyz,
            verified_masks,
            np.zeros_like(negative),
            unknown,
        ),
        "empty explicit-negative mask was treated as a passing zero",
    )
    expect_failure(
        lambda: all_instance_metrics(
            all_gt,
            targets_stack,
            [1, "1", "chair_06"],
            xyz,
            verified_masks,
            negative,
            unknown,
        ),
        "non-string instance names collapsed after canonicalization",
    )
    full_checks = simultaneous_presence_checks(
        full_metrics,
        minimum_soft_recall=0.75,
        minimum_topk_overlap=0.50,
        maximum_active_support_mae=0.05,
        maximum_hotspot_centroid_distance_xy=0.60,
        maximum_negative_mean=0.05,
        maximum_negative_max=0.80,
    )
    if not all(full_checks.values()):
        raise AssertionError(f"complete all-sit prediction was rejected: {full_checks}")
    expect_failure(
        lambda: simultaneous_presence_checks(
            {
                "instances": {},
                "explicit_negative_mean": 0.0,
                "explicit_negative_max": 0.0,
            },
            minimum_soft_recall=0.75,
            minimum_topk_overlap=0.50,
            maximum_active_support_mae=0.05,
            maximum_hotspot_centroid_distance_xy=0.60,
            maximum_negative_mean=0.05,
            maximum_negative_max=0.80,
        ),
        "empty instance metric set passed through all([])",
    )

    misplaced = all_gt.copy()
    misplaced[5, :] = 1.0
    misplaced_metrics = all_instance_metrics(
        misplaced, targets_stack, names, xyz, verified_masks, negative, unknown
    )
    normal_metrics = misplaced_metrics["instances"]["chair_01"]
    if (
        float(normal_metrics["topk_overlap"]) != 0.0
        or float(normal_metrics["hotspot_centroid_distance_xy"]) <= 0.40
    ):
        raise AssertionError("same-Chair false hotspot escaped spatial diagnostics")
    misplaced_checks = simultaneous_presence_checks(
        misplaced_metrics,
        minimum_soft_recall=0.75,
        minimum_topk_overlap=0.50,
        maximum_active_support_mae=0.05,
        maximum_hotspot_centroid_distance_xy=0.60,
        maximum_negative_mean=0.05,
        maximum_negative_max=0.80,
    )
    if misplaced_checks["every_verified_instance_has_topk_overlap"]:
        raise AssertionError("same-Chair false hotspot was accepted")

    unknown_hot = all_gt.copy()
    unknown_hot[unknown] = 1.0
    unknown_metrics = all_instance_metrics(
        unknown_hot,
        targets_stack,
        names,
        xyz,
        verified_masks,
        negative,
        unknown,
    )
    if unknown_metrics["explicit_negative_mean"] != full_metrics["explicit_negative_mean"]:
        raise AssertionError("unknown Chair was incorrectly scored as a negative")
    tv_hot = all_gt.copy()
    tv_hot[negative] = 1.0
    tv_checks = simultaneous_presence_checks(
        all_instance_metrics(
            tv_hot,
            targets_stack,
            names,
            xyz,
            verified_masks,
            negative,
            unknown,
        ),
        minimum_soft_recall=0.75,
        minimum_topk_overlap=0.50,
        maximum_active_support_mae=0.05,
        maximum_hotspot_centroid_distance_xy=0.60,
        maximum_negative_mean=0.05,
        maximum_negative_max=0.80,
    )
    if tv_checks["explicit_negative_mean_bounded"]:
        raise AssertionError("explicit-negative TV hotspot was accepted")

    sparse_negative = category_ids == 3
    sparse_negative |= category_ids == 4
    sparse_negative |= category_ids == 5
    sparse_hot = all_gt.copy()
    sparse_hot[10, 0] = 1.0
    sparse_checks = simultaneous_presence_checks(
        all_instance_metrics(
            sparse_hot,
            targets_stack,
            names,
            xyz,
            verified_masks,
            sparse_negative,
            unknown,
        ),
        minimum_soft_recall=0.75,
        minimum_topk_overlap=0.50,
        maximum_active_support_mae=0.05,
        maximum_hotspot_centroid_distance_xy=0.60,
        maximum_negative_mean=0.10,
        maximum_negative_max=0.80,
    )
    if (
        not sparse_checks["explicit_negative_mean_bounded"]
        or sparse_checks["explicit_negative_max_bounded"]
    ):
        raise AssertionError("single strong explicit-negative hotspot was not isolated")

    assert_teacher_forward_inputs(("c_pc_feat", "c_pc_xyz", "c_text"))
    expect_failure(
        lambda: assert_teacher_forward_inputs(
            ("c_pc_feat", "c_pc_xyz", "c_text", "target_instance_id")
        ),
        "target instance leaked into Teacher forward",
    )
    expect_failure(
        lambda: assert_teacher_forward_inputs(
            ("c_pc_feat", "c_pc_xyz", "c_text", "distance")
        ),
        "relation distance leaked into Teacher forward",
    )

    tampered = copy.deepcopy(rows)
    tampered[2]["target_instance_id"] = "chair_01"
    expect_failure(
        lambda: classify_dense_rows(tampered),
        "mixed High-Desk target tamper was accepted",
    )
    missing = rows[:-1]
    expect_failure(
        lambda: classify_dense_rows(missing),
        "23-row source inventory was accepted",
    )

    print("[PASS] Teacher-v9 all-sittable grouping and q75 consensus contract")
    print("[PASS] Bed + normal Chair + High Chair coexist in one primary GT")
    print("[PASS] Bed-only and missing-normal-Chair predictions are rejected")
    print("[PASS] unknown Chair is ignored while an explicit-negative hotspot fails")
    print("[PASS] same-object displacement and sparse negative peaks fail closed")
    print("[PASS] target/relation metadata cannot enter Teacher forward")
    print("[PASS] cross-object and provenance tampering fail closed")


if __name__ == "__main__":
    main()

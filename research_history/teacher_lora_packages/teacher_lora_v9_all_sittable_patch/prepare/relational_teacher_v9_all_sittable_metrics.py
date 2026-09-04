#!/usr/bin/env python3
"""Instance-balanced diagnostics for one scene-level all-sittable map."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np


def _finite_unit(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{label} must be finite in [0,1]")
    return result


def weighted_centroid_xy(xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.asarray([math.nan, math.nan], dtype=np.float64)
    return (xyz[:, :2] * weights[:, None]).sum(axis=0) / total


def one_instance_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    xyz: np.ndarray,
    *,
    active_threshold: float = 0.30,
    evaluation_point_mask: np.ndarray = None,
    ignore_point_mask: np.ndarray = None,
) -> Dict[str, float]:
    prediction = _finite_unit(prediction, "prediction")
    target = _finite_unit(target, "target")
    xyz = np.asarray(xyz, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError("prediction/target must be matching [N,6] arrays")
    if xyz.shape != (prediction.shape[0], 3):
        raise ValueError("xyz must be [N,3]")
    if evaluation_point_mask is None:
        evaluation = np.ones(prediction.shape[0], dtype=bool)
    else:
        evaluation = np.asarray(evaluation_point_mask, dtype=bool)
        if evaluation.shape != (prediction.shape[0],):
            raise ValueError("evaluation point mask must be [N]")
        if not np.any(evaluation):
            raise ValueError("evaluation point mask is empty")
    if ignore_point_mask is None:
        ignore = np.zeros(prediction.shape[0], dtype=bool)
    else:
        ignore = np.asarray(ignore_point_mask, dtype=bool)
        if ignore.shape != (prediction.shape[0],):
            raise ValueError("ignore point mask must be [N]")
    support = (
        (target >= float(active_threshold))
        & evaluation[:, None]
        & (~ignore[:, None])
    )
    support_count = int(support.sum())
    if support_count <= 0:
        raise ValueError("instance GT has no active support")
    target_support = target[support]
    prediction_support = prediction[support]
    soft_recall = float(
        np.minimum(prediction_support, target_support).sum()
        / max(float(target_support.sum()), 1e-12)
    )
    mae = float(np.abs(prediction_support - target_support).mean())

    valid_points = evaluation & (~ignore)
    target_any = target.max(axis=-1)
    prediction_any = prediction.max(axis=-1)
    active_points = valid_points & (target_any >= float(active_threshold))
    active_indices = np.flatnonzero(active_points)
    candidate_indices = np.flatnonzero(valid_points)
    if active_indices.size == 0:
        raise ValueError("instance GT support was entirely ignored")
    k = max(1, int(math.ceil(float(active_indices.size) * 0.25)))
    target_order = np.lexsort((active_indices, -target_any[active_indices]))
    prediction_order = np.lexsort(
        (candidate_indices, -prediction_any[candidate_indices])
    )
    target_top = active_indices[target_order[:k]]
    prediction_top = candidate_indices[prediction_order[:k]]
    topk_overlap = float(
        np.intersect1d(target_top, prediction_top, assume_unique=False).size / float(k)
    )

    gt_centroid = weighted_centroid_xy(
        xyz[active_points], target_any[active_points]
    )
    pred_centroid = weighted_centroid_xy(
        xyz[valid_points], prediction_any[valid_points]
    )
    centroid_distance = (
        float(np.linalg.norm(gt_centroid - pred_centroid))
        if np.isfinite(pred_centroid).all()
        else math.inf
    )
    return {
        "active_value_count": float(support_count),
        "soft_recall": soft_recall,
        "active_support_mae": mae,
        "topk_overlap": topk_overlap,
        "hotspot_centroid_distance_xy": centroid_distance,
        "prediction_peak_on_gt_support": float(prediction_support.max()),
    }


def all_instance_metrics(
    prediction: np.ndarray,
    instance_targets: np.ndarray,
    instance_names: Sequence[str],
    xyz: np.ndarray,
    verified_object_masks: np.ndarray,
    explicit_negative_mask: np.ndarray,
    unknown_sittable_mask: np.ndarray,
    *,
    active_threshold: float = 0.30,
) -> Dict[str, object]:
    prediction = _finite_unit(prediction, "prediction")
    targets = _finite_unit(instance_targets, "instance targets")
    if (
        targets.ndim != 3
        or targets.shape[0] != 3
        or targets.shape[1:] != prediction.shape
    ):
        raise ValueError("instance targets must be exactly [3,N,6]")
    if len(instance_names) != 3 or any(
        not isinstance(name, (str, np.str_)) for name in instance_names
    ):
        raise ValueError("instance names must be three unique non-empty strings")
    canonical_names = tuple(str(name) for name in instance_names)
    if any(not name for name in canonical_names) or len(set(canonical_names)) != 3:
        raise ValueError("instance names must be three unique non-empty strings")
    verified = np.asarray(verified_object_masks, dtype=bool)
    if verified.shape != (targets.shape[0], prediction.shape[0]):
        raise ValueError("verified object masks must be [K,N]")
    if np.any(verified.sum(axis=1) <= 0) or np.any(verified.sum(axis=0) > 1):
        raise ValueError("verified object masks must be nonempty and disjoint")
    negative = np.asarray(explicit_negative_mask, dtype=bool)
    unknown = np.asarray(unknown_sittable_mask, dtype=bool)
    if negative.shape != (prediction.shape[0],) or unknown.shape != negative.shape:
        raise ValueError("negative/unknown masks must be [N]")
    if not np.any(negative):
        raise ValueError("explicit-negative object mask must not be empty")
    verified_union = verified.any(axis=0)
    if np.any(negative & unknown) or np.any(negative & verified_union):
        raise ValueError("explicit negatives cannot overlap Sit instances")
    if np.any(unknown & verified_union):
        raise ValueError("unknown and verified Sit instances overlap")

    rows = {
        name: one_instance_metrics(
            prediction,
            targets[index],
            xyz,
            active_threshold=active_threshold,
            evaluation_point_mask=verified[index],
            ignore_point_mask=unknown,
        )
        for index, name in enumerate(canonical_names)
    }
    recalls = [float(row["soft_recall"]) for row in rows.values()]
    maes = [float(row["active_support_mae"]) for row in rows.values()]
    overlaps = [float(row["topk_overlap"]) for row in rows.values()]
    centroids = [float(row["hotspot_centroid_distance_xy"]) for row in rows.values()]
    negative_values = prediction[negative]
    negative_mean = float(negative_values.mean())
    negative_max = float(negative_values.max())
    return {
        "instances": rows,
        "instance_macro_soft_recall": float(np.mean(recalls)),
        "worst_instance_soft_recall": float(np.min(recalls)),
        "instance_macro_active_support_mae": float(np.mean(maes)),
        "worst_instance_active_support_mae": float(np.max(maes)),
        "instance_macro_topk_overlap": float(np.mean(overlaps)),
        "worst_instance_topk_overlap": float(np.min(overlaps)),
        "instance_macro_hotspot_centroid_distance_xy": float(np.mean(centroids)),
        "worst_instance_hotspot_centroid_distance_xy": float(np.max(centroids)),
        "explicit_negative_mean": negative_mean,
        "explicit_negative_max": negative_max,
        "unknown_sittable_point_count_ignored": int(unknown.sum()),
    }


def simultaneous_presence_checks(
    metrics: Dict[str, object],
    *,
    minimum_soft_recall: float,
    minimum_topk_overlap: float,
    maximum_active_support_mae: float,
    maximum_hotspot_centroid_distance_xy: float,
    maximum_negative_mean: float,
    maximum_negative_max: float,
) -> Dict[str, bool]:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    rows = metrics.get("instances")
    if not isinstance(rows, Mapping) or len(rows) != 3:
        raise ValueError("metrics must contain exactly three instance rows")
    required = {
        "soft_recall",
        "topk_overlap",
        "active_support_mae",
        "hotspot_centroid_distance_xy",
    }
    for name, row in rows.items():
        if not isinstance(name, (str, np.str_)) or not str(name):
            raise ValueError("metric instance names must be non-empty strings")
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("instance metric row is incomplete")
        recall = float(row["soft_recall"])
        overlap = float(row["topk_overlap"])
        mae = float(row["active_support_mae"])
        centroid = float(row["hotspot_centroid_distance_xy"])
        if (
            not math.isfinite(recall)
            or not 0.0 <= recall <= 1.0
            or not math.isfinite(overlap)
            or not 0.0 <= overlap <= 1.0
            or not math.isfinite(mae)
            or not 0.0 <= mae <= 1.0
            or math.isnan(centroid)
            or centroid < 0.0
        ):
            raise ValueError("instance metric row contains invalid values")
    negative_mean = float(metrics.get("explicit_negative_mean", math.nan))
    negative_max = float(metrics.get("explicit_negative_max", math.nan))
    if (
        not math.isfinite(negative_mean)
        or not 0.0 <= negative_mean <= 1.0
        or not math.isfinite(negative_max)
        or not 0.0 <= negative_max <= 1.0
        or negative_max < negative_mean
    ):
        raise ValueError("explicit-negative metrics are invalid")
    return {
        "every_verified_instance_has_soft_recall": all(
            float(row["soft_recall"]) >= minimum_soft_recall
            for row in rows.values()
        ),
        "every_verified_instance_has_topk_overlap": all(
            float(row["topk_overlap"]) >= minimum_topk_overlap
            for row in rows.values()
        ),
        "every_verified_instance_has_bounded_active_support_mae": all(
            float(row["active_support_mae"]) <= maximum_active_support_mae
            for row in rows.values()
        ),
        "every_verified_instance_has_bounded_hotspot_centroid": all(
            float(row["hotspot_centroid_distance_xy"])
            <= maximum_hotspot_centroid_distance_xy
            for row in rows.values()
        ),
        "explicit_negative_mean_bounded": negative_mean <= maximum_negative_mean,
        "explicit_negative_max_bounded": negative_max <= maximum_negative_max,
    }


__all__ = [
    "all_instance_metrics",
    "one_instance_metrics",
    "simultaneous_presence_checks",
    "weighted_centroid_xy",
]

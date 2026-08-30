#!/usr/bin/env python3
"""CPU-only synthetic tests for the strict v2 Base-teacher contract.

The tests intentionally create complete 25-train/12-development v5r4 evidence
chains in temporary directories.  No checkpoint tensors, GPU, network access,
or real v5r4 outputs are required.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np


PREPARE_DIR = Path(__file__).resolve().parent
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from base_teacher_contract import (  # noqa: E402
    ARTIFACT_SCHEMA,
    AUDIT_QUALITY_THRESHOLDS,
    AUDIT_SCHEMA,
    CHANNEL_JOINT_INDICES,
    CHANNEL_ORDER,
    DEVELOPMENT_QUALITY_THRESHOLDS,
    DEVELOPMENT_SCHEMA,
    NUM_CHANNELS,
    NUM_POINTS,
    PROMPT_BY_TARGET,
    PROMPT_POLICY_ID,
    REQUIRED_AUDIT_CHECKS,
    REQUIRED_DEVELOPMENT_CHECKS,
    REQUIRED_ROLLOUT_SELECTION_CHECKS,
    ROLLOUT_PROTOCOL_FILES,
    ROLLOUT_SELECTION_SCHEMA,
    SHORTLIST_SCHEMA,
    TRAIN_SCHEMA,
    _aggregate_metric_rows,
    _aggregate_per_draw_metric_rows,
    _audit_row_projection,
    _development_row_projection,
    _dominance_rates,
    _recompute_rollout_rows,
    _sit_rates,
    atomic_write_json,
    build_cache_key_payload,
    canonical_json_sha256,
    convert_prediction_draws,
    derive_draw_seeds,
    load_contact_stats,
    load_json,
    load_scene_contract,
    rollout_draw_seeds,
    rollout_sha256_array,
    rollout_sha256_json,
    sha256_array,
    sha256_file,
    sha256_text,
    validate_base_teacher_artifact,
    validate_prompt,
    validate_v5r4_quality_chain,
    write_base_teacher_artifact,
)


VALID_TEXT = "Sit somewhere."
VALID_PROMPT_ID = "sit.generic.v5"
SYNTHETIC_STATS_SHA256 = "8" * 64
SYNTHETIC_ROLLOUT_PROTOCOL_SHA256 = {
    name: sha256_text("synthetic rollout protocol: " + name)
    for name in ROLLOUT_PROTOCOL_FILES
}
SYNTHETIC_RUNTIME_FILE_SHA256 = {
    **SYNTHETIC_ROLLOUT_PROTOCOL_SHA256,
    "utils/registry.py": sha256_text("synthetic runtime: utils/registry.py"),
    "binary/pointops_cuda": sha256_text("synthetic pointops_cuda binary"),
}
SYNTHETIC_RESOLVED_CONFIG = {"model": {"name": "synthetic-cdm"}}
SYNTHETIC_EFFECTIVE_MODEL_STATE = {
    "full_state_sha256": "1" * 64,
    "cdm_core_state_sha256": "2" * 64,
    "scene_model_state_sha256": "3" * 64,
    "text_model_state_sha256": "4" * 64,
}
SYNTHETIC_SAMPLING_ENVIRONMENT = {
    "python_version": "3.synthetic",
    "numpy_version": np.__version__,
    "torch_version": "synthetic",
    "torch_build_config_sha256": "9" * 64,
    "cuda_version": None,
    "cudnn_version": None,
    "device": "cpu",
    "device_name": "synthetic-cpu",
    "device_capability": None,
}
SYNTHETIC_RUNTIME_IDENTITY = {
    "schema": "history_affordance_v2_teacher_runtime_identity_v1",
    "fewshot_checkpoint_sha256": "a" * 64,
    "contact_stats_sha256": SYNTHETIC_STATS_SHA256,
    "effective_model_state": dict(SYNTHETIC_EFFECTIVE_MODEL_STATE),
    "partial_checkpoint_coverage": {"status": "synthetic-complete"},
    "scene_model_pretrained_weight_sha256": "7" * 64,
    "text_model_name": "synthetic-text-model",
    "resolved_config_sha256": canonical_json_sha256(SYNTHETIC_RESOLVED_CONFIG),
    "runtime_file_set_sha256": canonical_json_sha256(
        SYNTHETIC_RUNTIME_FILE_SHA256
    ),
    "sampling_environment": dict(SYNTHETIC_SAMPLING_ENVIRONMENT),
}
SYNTHETIC_RUNTIME_SHA256 = canonical_json_sha256(SYNTHETIC_RUNTIME_IDENTITY)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_prediction_bundle(
    path: Path,
    sample_ids: Tuple[str, ...],
    k_draws: int,
    *,
    targets: Tuple[str, ...] | None = None,
    seed_partition: str = "train_audit",
    offset: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Write a genuine PASS bundle whose quality follows the v5r4 thresholds."""

    if targets is None:
        targets = tuple("chair" for _ in sample_ids)
    if len(targets) != len(sample_ids):
        raise ValueError("synthetic prediction targets/sample IDs differ")
    instance_ids = _quality_instance_ids()
    gt_rows = []
    fewshot_rows = []
    original_rows = []
    for target in targets:
        gt = np.zeros((NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
        gt[instance_ids == _target_instance_id(target), :] = np.float32(0.90)
        fewshot = np.full_like(gt, np.float32(0.05))
        fewshot[instance_ids == _target_instance_id(target), :] = np.float32(0.90)
        if target == "chair":
            bed_mask = instance_ids == _target_instance_id("bed")
            fewshot[bed_mask, :] = np.float32(0.20)
            fewshot[bed_mask, 0] = np.float32(0.15)
            original = fewshot.copy()
        else:
            original = np.zeros_like(gt)
        if offset:
            fewshot = np.clip(fewshot + np.float32(offset), 0.0, 1.0)
            original = np.clip(original + np.float32(offset), 0.0, 1.0)
        gt_rows.append(gt)
        fewshot_rows.append(fewshot)
        original_rows.append(original)
    gt = np.stack(gt_rows).astype(np.float32)
    fewshot_draws = np.repeat(
        np.stack(fewshot_rows)[:, None, :, :], k_draws, axis=1
    ).astype(np.float32)
    original_draws = np.repeat(
        np.stack(original_rows)[:, None, :, :], k_draws, axis=1
    ).astype(np.float32)
    fewshot = fewshot_draws.mean(axis=1).astype(np.float32)
    original = original_draws.mean(axis=1).astype(np.float32)
    seed_pairs = np.asarray(
        [
            [
                rollout_draw_seeds(
                    base_seed=20260815,
                    partition=seed_partition,
                    sample_id=sample_id,
                    draw_index=draw_index,
                )
                for draw_index in range(k_draws)
            ]
            for sample_id in sample_ids
        ],
        dtype=np.int64,
    )
    arrays = {
        "sample_ids": np.asarray(sample_ids),
        "gt": gt,
        "original": original,
        "fewshot": fewshot,
        "original_draws": original_draws,
        "fewshot_draws": fewshot_draws,
        "initial_noise_seeds": seed_pairs[:, :, 0],
        "reverse_noise_seeds": seed_pairs[:, :, 1],
    }
    np.savez_compressed(
        path,
        **arrays,
    )
    return arrays


def _target_instance_id(target: str) -> int:
    return {"chair": 1, "bed": 2, "whiteboard": 3}[target]


def _quality_instance_ids() -> np.ndarray:
    return (np.arange(NUM_POINTS, dtype=np.int64) % 3 + 1).astype(np.int64)


def _write_quality_partition_dataset(
    *,
    dataset_root: Path,
    scene_id: str,
    sample_ids: Tuple[str, ...],
    targets: Tuple[str, ...],
    gt: np.ndarray,
) -> Tuple[Dict[str, Mapping[str, object]], Dict[str, np.ndarray]]:
    """Create the actual index/scene/sample-manifest/GT files read by validation."""

    scene_dir = dataset_root / "scenes" / scene_id / "adm_input"
    scene_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.zeros((NUM_POINTS, 3), dtype=np.float32)
    xyz[:, 0] = np.arange(NUM_POINTS, dtype=np.float32) / np.float32(NUM_POINTS)
    xyz[:, 1] = np.float32(0.25)
    xyz[:, 2] = np.float32(1.0)
    rgb255 = np.full((NUM_POINTS, 3), np.float32(128.0), dtype=np.float32)
    points = np.concatenate((xyz, rgb255), axis=1).astype(np.float32)
    instance_ids = _quality_instance_ids()
    source_indices = np.arange(NUM_POINTS, dtype=np.int64)
    np.savez_compressed(scene_dir / "points.npz", points=points)
    np.savez_compressed(
        scene_dir / "sidecar.npz",
        instance_ids=instance_ids,
        source_indices=source_indices,
    )

    index_rows = []
    fingerprints: Dict[str, Mapping[str, object]] = {}
    instance_ids_by_sample: Dict[str, np.ndarray] = {}
    for index, (sample_id, target) in enumerate(zip(sample_ids, targets)):
        sample_dir = dataset_root / "samples" / sample_id
        manifest_file = sample_dir / "manifest.json"
        _write_json(manifest_file, {"sample_id": sample_id})
        gt_dir = sample_dir / "affordance_gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            gt_dir / "full_affordance_gt.npz",
            affordance=gt[index],
            instance_ids=instance_ids,
            source_indices=source_indices,
        )
        index_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "target_instance_id": _target_instance_id(target),
                "sample_manifest": str(manifest_file.relative_to(dataset_root)),
            }
        )
        fingerprints[sample_id] = {
            "scene_id": scene_id,
            "target": target,
            "target_instance_id": _target_instance_id(target),
            "text": PROMPT_BY_TARGET[target],
            "gt_sha256": rollout_sha256_array(gt[index]),
            "xyz_sha256": rollout_sha256_array(xyz),
            "feat_sha256": rollout_sha256_array(
                (rgb255 / np.float32(255.0)).astype(np.float32)
            ),
            "instance_ids_sha256": rollout_sha256_array(instance_ids),
            "source_indices_sha256": rollout_sha256_array(source_indices),
        }
        instance_ids_by_sample[sample_id] = instance_ids.copy()
    _write_json(
        dataset_root / ("index_" + scene_id + ".json"),
        {
            "scene_id": scene_id,
            "scene_adm_input": str(scene_dir.relative_to(dataset_root)),
            "samples": index_rows,
        },
    )
    return fingerprints, instance_ids_by_sample


def _make_dataset_snapshot(
    *,
    partition: str,
    sample_ids: Tuple[str, ...],
    fingerprints: Mapping[str, Mapping[str, object]],
    split_file: Path,
    stats_file: Path,
) -> Dict[str, object]:
    if list(fingerprints) != list(sample_ids):
        raise ValueError("synthetic snapshot fingerprint order mismatch")
    payload: Dict[str, object] = {
        "schema": "history_affordance_v1_rollout_dataset_snapshot_v1",
        "partition": partition,
        "split_sha256": sha256_file(split_file),
        "stats_file_sha256": sha256_file(stats_file),
        "sample_fingerprints": dict(fingerprints),
    }
    return {**payload, "sha256": rollout_sha256_json(payload)}


def _chair_relative_degradation(per_draw: Mapping[str, object]) -> float:
    original = float(per_draw["chair"]["original"]["mae"])
    fewshot = float(per_draw["chair"]["fewshot"]["mae"])
    return (fewshot - original) / max(original, 1e-12)


def _audit_metric_fields(
    *,
    arrays: Mapping[str, np.ndarray],
    fingerprints: Mapping[str, Mapping[str, object]],
    instance_ids_by_sample: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    rows = _recompute_rollout_rows(
        arrays=arrays,
        fingerprints=fingerprints,
        instance_ids_by_sample=instance_ids_by_sample,
    )
    per_draw = _aggregate_per_draw_metric_rows(rows)
    sit_rates = _sit_rates(rows)
    return {
        "quality_thresholds": copy.deepcopy(AUDIT_QUALITY_THRESHOLDS),
        "per_sample": [_audit_row_projection(row) for row in rows],
        "aggregate_target_region": _aggregate_metric_rows(
            rows, "target_region_metrics"
        ),
        "per_draw_aggregate_target_region": per_draw,
        "dominance_rate": _dominance_rates(rows, include_ensemble=True),
        "sit_contract": {
            "chair_primary_min_rate": 0.90,
            "bed_candidate_min_rate": 0.80,
            "ensemble_chair_primary_rate": sit_rates[
                "ensemble_chair_primary_rate"
            ],
            "ensemble_bed_candidate_rate": sit_rates[
                "ensemble_bed_candidate_rate"
            ],
            "per_draw_chair_primary_rate": sit_rates[
                "per_draw_chair_primary_rate"
            ],
            "per_draw_bed_candidate_rate": sit_rates[
                "per_draw_bed_candidate_rate"
            ],
        },
        "chair_mae_relative_degradation": _chair_relative_degradation(per_draw),
        "dominance_margin_required": 0.01,
    }


def _development_metric_fields(
    *,
    arrays: Mapping[str, np.ndarray],
    fingerprints: Mapping[str, Mapping[str, object]],
    instance_ids_by_sample: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    rows = _recompute_rollout_rows(
        arrays=arrays,
        fingerprints=fingerprints,
        instance_ids_by_sample=instance_ids_by_sample,
    )
    per_draw = _aggregate_per_draw_metric_rows(rows)
    dominance = _dominance_rates(rows, include_ensemble=False)
    sit_rates = _sit_rates(rows)
    return {
        "quality_thresholds": copy.deepcopy(DEVELOPMENT_QUALITY_THRESHOLDS),
        "per_sample": [_development_row_projection(row) for row in rows],
        "aggregate": _aggregate_metric_rows(rows, "metrics"),
        "aggregate_target_region": _aggregate_metric_rows(
            rows, "target_region_metrics"
        ),
        "per_draw_aggregate_target_region": per_draw,
        "per_draw_dominance_rate": dominance,
        "sit_per_draw_contract": {
            "chair_primary_rate": sit_rates["per_draw_chair_primary_rate"],
            "bed_candidate_visible_rate": sit_rates[
                "per_draw_bed_candidate_rate"
            ],
            "minimum_rate": 0.80,
        },
        "chair_mae_relative_degradation": _chair_relative_degradation(per_draw),
        "dominance_margin_required": 0.01,
    }


def _make_scene_files(root: Path, scene_id: str = "room_test") -> Dict[str, object]:
    dataset_root = root / "dataset"
    scene_dir = dataset_root / "scenes" / scene_id / "adm_input"
    scene_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.empty((NUM_POINTS, 3), dtype=np.float32)
    xyz[:, 0] = np.arange(NUM_POINTS, dtype=np.float32) / np.float32(NUM_POINTS)
    xyz[:, 1] = np.arange(NUM_POINTS, dtype=np.float32) % np.float32(17.0)
    xyz[:, 2] = np.float32(1.25)
    rgb = np.empty((NUM_POINTS, 3), dtype=np.float32)
    rgb[:, 0] = np.float32(10.0)
    rgb[:, 1] = np.float32(128.0)
    rgb[:, 2] = np.float32(255.0)
    points = np.concatenate((xyz, rgb), axis=1).astype(np.float32)
    instance_ids = (np.arange(NUM_POINTS, dtype=np.int64) % 5).astype(np.int64)
    source_indices = np.arange(NUM_POINTS, dtype=np.int64)
    points_file = scene_dir / "points.npz"
    sidecar_file = scene_dir / "sidecar.npz"
    np.savez_compressed(points_file, points=points)
    np.savez_compressed(
        sidecar_file,
        xyz_afford_z_up=xyz,
        instance_ids=instance_ids,
        source_indices=source_indices,
    )
    index_file = dataset_root / ("index_" + scene_id + ".json")
    _write_json(
        index_file,
        {
            "scene_id": scene_id,
            "scene_adm_input": str(scene_dir.relative_to(dataset_root)),
        },
    )
    return {
        "dataset_root": dataset_root,
        "scene_id": scene_id,
        "scene_dir": scene_dir,
        "points_file": points_file,
        "sidecar_file": sidecar_file,
        "index_file": index_file,
        "points": points,
        "xyz": xyz,
        "instance_ids": instance_ids,
        "source_indices": source_indices,
    }


def _make_quality_chain(root: Path, k_draws: int = 5, steps: int = 500) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    fewshot_checkpoint = root / "fewshot_cdm.pt"
    original_checkpoint = root / "model300000.pt"
    fewshot_checkpoint.write_bytes(b"synthetic fewshot checkpoint\n")
    original_checkpoint.write_bytes(b"synthetic original checkpoint\n")

    train_ids = tuple("train_%02d" % index for index in range(25))
    development_ids = tuple("development_%02d" % index for index in range(12))
    train_targets = tuple(["chair"] * 18 + ["whiteboard"] * 6 + ["bed"])
    development_targets = tuple(["chair"] * 5 + ["whiteboard"] * 6 + ["bed"])
    split_file = root / "split.json"
    _write_json(
        split_file,
        {
            "schema": "affordance_source_disjoint_split_v1",
            "split_unit": "connected raw-source component",
            "checks": {
                "source_components_disjoint": True,
                "original_motion_ids_disjoint": True,
                "multistart_excluded_from_cdm": True,
                "object_names_absent_from_new_prompts": True,
            },
            "cdm_fewshot": {
                "train": list(train_ids),
                "test": list(development_ids),
            },
            "samples": [
                {
                    "sample_id": sample_id,
                    "stage": "base",
                    "split": partition,
                    "target": target,
                    "scene_id": "synthetic_" + partition + "_scene",
                }
                for partition, sample_ids, targets in (
                    ("train", train_ids, train_targets),
                    ("test", development_ids, development_targets),
                )
                for sample_id, target in zip(sample_ids, targets)
            ],
            "groups": [
                {
                    "split": "train",
                    "source_components": ["synthetic_train_component"],
                    "sample_ids": list(train_ids),
                },
                {
                    "split": "test",
                    "source_components": ["synthetic_development_component"],
                    "sample_ids": list(development_ids),
                },
            ],
        },
    )
    stats_file = root / "stats.npz"
    np.savez_compressed(
        stats_file,
        mean=np.full((1, NUM_CHANNELS), 0.2, dtype=np.float32),
        std=np.full((1, NUM_CHANNELS), 0.3, dtype=np.float32),
    )

    train_predictions = root / "train_rollout_predictions.npz"
    development_predictions = root / "development_predictions.npz"
    train_arrays = _write_prediction_bundle(
        train_predictions,
        train_ids,
        k_draws,
        targets=train_targets,
    )
    development_arrays = _write_prediction_bundle(
        development_predictions,
        development_ids,
        k_draws,
        targets=development_targets,
        seed_partition="development",
    )
    dataset_root = root / "dataset"
    train_fingerprints, train_instance_ids = _write_quality_partition_dataset(
        dataset_root=dataset_root,
        scene_id="synthetic_train_scene",
        sample_ids=train_ids,
        targets=train_targets,
        gt=train_arrays["gt"],
    )
    (
        development_fingerprints,
        development_instance_ids,
    ) = _write_quality_partition_dataset(
        dataset_root=dataset_root,
        scene_id="synthetic_test_scene",
        sample_ids=development_ids,
        targets=development_targets,
        gt=development_arrays["gt"],
    )
    train_snapshot = _make_dataset_snapshot(
        partition="train",
        sample_ids=train_ids,
        fingerprints=train_fingerprints,
        split_file=split_file,
        stats_file=stats_file,
    )
    development_snapshot = _make_dataset_snapshot(
        partition="development",
        sample_ids=development_ids,
        fingerprints=development_fingerprints,
        split_file=split_file,
        stats_file=stats_file,
    )
    audit_metric_fields = _audit_metric_fields(
        arrays=train_arrays,
        fingerprints=train_fingerprints,
        instance_ids_by_sample=train_instance_ids,
    )
    development_metric_fields = _development_metric_fields(
        arrays=development_arrays,
        fingerprints=development_fingerprints,
        instance_ids_by_sample=development_instance_ids,
    )

    shortlist_file = root / "shortlist_status.json"
    _write_json(
        shortlist_file,
        {
            "schema": SHORTLIST_SCHEMA,
            "status": "PASS",
            "partition": "train_only",
            "heldout_sample_tensors_read": False,
            "candidate_count": 3,
            "prompt_policy_id": PROMPT_POLICY_ID,
            "best_one_step": {
                "selection": {
                    "loss_gate": {"passed": True},
                    "semantic_gate": {"passed": True},
                }
            },
            "checks": {"synthetic_shortlist_check": True},
        },
    )
    rollout_selected = {
        "passed": True,
        "checkpoint_sha256": sha256_file(fewshot_checkpoint),
        "checks": {
            name: True for name in sorted(REQUIRED_ROLLOUT_SELECTION_CHECKS)
        },
    }
    rollout_selection_file = root / "rollout_candidate_selection.json"
    _write_json(
        rollout_selection_file,
        {
            "schema": ROLLOUT_SELECTION_SCHEMA,
            "status": "PASS",
            "partition": "train_complete",
            "heldout_sample_tensors_read": False,
            "complete_train_partition": True,
            "k_samples": k_draws,
            "seed_partition": "train_audit",
            "probe_sample_ids": list(train_ids),
            "diffusion_steps": steps,
            "shortlisted_candidate_count": 3,
            "candidate_count": 1,
            "evaluated": [rollout_selected],
            "selected": rollout_selected,
        },
    )

    train_summary_file = root / "train_summary.json"
    _write_json(
        train_summary_file,
        {
            "schema": TRAIN_SCHEMA,
            "status": "PASS",
            "checkpoint_sha256": sha256_file(fewshot_checkpoint),
            "initialization": {
                "sha256": sha256_file(original_checkpoint),
                "standard_original_path": True,
                "one_sample_diagnostic_checkpoint_used": False,
            },
            "split": {
                "sha256": sha256_file(split_file),
                "train_target_counts": {"chair": 18, "whiteboard": 6, "bed": 1},
            },
            "selection_data": "train_only",
            "test_partition_read_during_training": False,
            "test_sample_data_read_during_training": False,
            "test_ids_used_only_for_exclusion_assertion": True,
            "diffusion_steps": steps,
            "sampling": {
                "method": "chair3_bed1_whiteboard1_shuffled_cycles",
                "batch_size": 5,
                "chair_replay_per_step": 3,
                "bed_per_step": 1,
                "whiteboard_per_step": 1,
            },
            "regularization": {
                "method": (
                    "frozen_original_cdm_plus_zero_init_lora_and_"
                    "chair_region_multinoise_teacher_v5r4"
                ),
                "zero_init_bitwise_original": True,
                "original_checkpoint_sha256": sha256_file(original_checkpoint),
                "chair_teacher_region": "target_chair_instance_only",
                "bed_candidate_region_excluded_from_chair_teacher": True,
                "chair_teacher_weight": 1.0,
                "chair_high_timestep_weight": 1.0,
                "chair_high_teacher_weight": 1.0,
                "lora": {
                    "base_parameters_frozen": True,
                    "zero_initialized_output_projection": True,
                    "export_format": "merged_legacy_partial_state_dict",
                },
            },
            "data_objective": {
                "prompt_policy_id": PROMPT_POLICY_ID,
                "prompt_by_target": dict(PROMPT_BY_TARGET),
                "method": (
                    "rollout_aligned_sparse_semantic_multinoise_start_x_v5r4"
                ),
                "diffusion_prediction_target": "START_X",
                "chair_uses_legacy_uniform_weights": True,
                "background_remains_supervised": True,
                "novel_high_timestep_extra_forward": True,
                "chair_high_timestep_extra_forward": True,
                "whiteboard_semantic_channel": "right_wrist_native_index_5",
                "active_threshold": 0.7,
                "base_weight": 1.0,
                "target_instance_additive_weight": 1.0,
                "target_foreground_additive_weight": 1.0,
                "semantic_weight": 1.0,
                "foreground_bce_weight": 1.0,
                "dice_weight": 1.0,
                "ranking_weight": 1.0,
                "high_timestep_weight": 1.0,
                "chair_uniform_legacy_parity_max_abs_diff": 0.0,
                "sit_multicandidate": {
                    "supervision_type": "weak_semantic_prior_only",
                    "motion_gt_relabelled_or_copied": False,
                    "weight": 0.50,
                    "bed_any_joint_band": [0.12, 0.40],
                    "bed_pelvis_band": [0.06, 0.25],
                    "chair_primary_margin": 0.15,
                },
            },
            "checkpoint_selection": {
                "method": "one_step_shortlist_then_train_full_rollout",
                "sit_bed_candidate_min_rate": 0.80,
                "gate_passed": True,
                "best": {
                    "loss_gate": {"passed": True},
                    "semantic_gate": {"passed": True},
                },
                "rollout_gate_passed": True,
                "rollout_selection": {
                    "selection_file_sha256": sha256_file(rollout_selection_file),
                    "selected": rollout_selected,
                    "partition": "train_complete",
                    "complete_train_partition": True,
                    "k_samples": k_draws,
                    "seed_partition": "train_audit",
                    "probe_sample_ids": list(train_ids),
                    "evaluated_candidate_count": 1,
                },
            },
            "checks": {"synthetic_training_check": True},
        },
    )

    rollout_protocol_hashes = {
        name: sha256_text("synthetic protocol: " + name)
        for name in ROLLOUT_PROTOCOL_FILES
    }
    audit_file = root / "train_rollout_audit.json"
    _write_json(
        audit_file,
        {
            "schema": AUDIT_SCHEMA,
            "status": "PASS",
            "decision": "TRAIN_ROLLOUT_VALID",
            "partition": "train",
            "test_sample_tensors_read": False,
            "checkpoint_sha256": sha256_file(fewshot_checkpoint),
            "original_checkpoint_sha256": sha256_file(original_checkpoint),
            "train_summary_sha256": sha256_file(train_summary_file),
            "split_sha256": sha256_file(split_file),
            "stats_file_sha256": sha256_file(stats_file),
            "rollout_protocol_hashes": rollout_protocol_hashes,
            "paired_noise": True,
            "sampling_provenance": {
                "base_seed": 20260815,
                "partition": "train_audit",
                "seed_derivation": "sha256(base|partition|sample_id|draw)",
                "seed_table_storage": "prediction_npz_int64_matrices",
            },
            "pairing_protocol": {
                "initial_xT_and_all_reverse_step_noise_paired": True,
                "caller_rng_state_restored": True,
                "repeatability_canary_bitwise_equal": True,
                "metrics_computed_per_draw": True,
            },
            "k_samples": k_draws,
            "diffusion_steps": steps,
            "train_count": 25,
            "train_sample_ids": list(train_ids),
            "train_target_counts": {"chair": 18, "whiteboard": 6, "bed": 1},
            "dataset_snapshot": train_snapshot,
            **audit_metric_fields,
            "predictions": {
                "file": train_predictions.name,
                "sha256": sha256_file(train_predictions),
                "format": "paired_rollout_affordance_npz_v2",
            },
            "checks": {name: True for name in sorted(REQUIRED_AUDIT_CHECKS)},
        },
    )

    development_summary_file = root / "development_summary.json"
    _write_json(
        development_summary_file,
        {
            "schema": DEVELOPMENT_SCHEMA,
            "status": "PASS",
            "may_proceed_to_moe_iiw": True,
            "final_paper_test_requires_new_unseen_sources": True,
            "pairing_protocol": {
                "initial_xT_and_all_reverse_step_noise_paired": True,
                "caller_rng_state_restored": True,
            },
            "sampling_provenance": {
                "base_seed": 20260815,
                "partition": "development",
                "seed_derivation": "sha256(base|partition|sample_id|draw)",
                "seed_table_storage": "prediction_npz_int64_matrices",
            },
            "k_samples": k_draws,
            "diffusion_steps": steps,
            "dataset_snapshot": development_snapshot,
            **development_metric_fields,
            "train_rollout_audit": {
                "sha256": sha256_file(audit_file),
                "schema": AUDIT_SCHEMA,
                "status": "PASS",
                "k_samples": k_draws,
                "test_sample_tensors_read": False,
            },
            "predictions": {
                "file": development_predictions.name,
                "sha256": sha256_file(development_predictions),
                "format": "paired_rollout_affordance_npz_v2",
            },
            "checks": {
                name: True for name in sorted(REQUIRED_DEVELOPMENT_CHECKS)
            },
        },
    )

    return {
        "fewshot_checkpoint": fewshot_checkpoint,
        "original_checkpoint": original_checkpoint,
        "train_summary_file": train_summary_file,
        "shortlist_status_file": shortlist_file,
        "rollout_selection_file": rollout_selection_file,
        "train_rollout_audit_file": audit_file,
        "train_rollout_predictions_file": train_predictions,
        "development_summary_file": development_summary_file,
        "development_predictions_file": development_predictions,
        "split_file": split_file,
        "stats_file": stats_file,
        "dataset_root": dataset_root,
    }


def _valid_runtime() -> Dict[str, object]:
    return {
        "status": "PASS",
        **SYNTHETIC_SAMPLING_ENVIRONMENT,
        "resolved_config": copy.deepcopy(SYNTHETIC_RESOLVED_CONFIG),
        "resolved_config_sha256": canonical_json_sha256(
            SYNTHETIC_RESOLVED_CONFIG
        ),
        "scene_model_pretrained_weight_sha256": "7" * 64,
        "text_model_name": "synthetic-text-model",
        "partial_checkpoint_coverage": {"status": "synthetic-complete"},
        "teacher_runtime_identity": dict(SYNTHETIC_RUNTIME_IDENTITY),
        "teacher_runtime_sha256": SYNTHETIC_RUNTIME_SHA256,
        "runtime_file_sha256": dict(SYNTHETIC_RUNTIME_FILE_SHA256),
        "model_eval_mode": True,
        "all_parameters_frozen": True,
        "state_unchanged_during_sampling": True,
        "effective_model_state": dict(SYNTHETIC_EFFECTIVE_MODEL_STATE),
    }


def _valid_draws(k_draws: int = 5) -> np.ndarray:
    draws = np.empty((k_draws, NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
    point_ramp = np.arange(NUM_POINTS, dtype=np.float32) / np.float32(NUM_POINTS)
    for draw_index in range(k_draws):
        for channel_index in range(NUM_CHANNELS):
            draws[draw_index, :, channel_index] = np.clip(
                np.float32(0.05 + 0.10 * draw_index + 0.02 * channel_index)
                + point_ramp * np.float32(0.20),
                0.0,
                1.0,
            )
    return draws


def _write_valid_artifact(
    root: Path,
    *,
    source_representation: str = "affordance",
    transform: Mapping[str, object] | None = None,
    scene_override: Mapping[str, object] | None = None,
    k_draws: int = 5,
    diffusion_steps: int = 500,
    base_seed: int = 20260815,
) -> Tuple[Path, Path, Dict[str, object], np.ndarray]:
    scene_files = _make_scene_files(root)
    scene = load_scene_contract(
        scene_files["dataset_root"], str(scene_files["scene_id"])
    )
    if scene_override is not None:
        scene = {**scene, **scene_override}
    draws = _valid_draws(k_draws)
    payload = build_cache_key_payload(
        scene_sha256=str(scene["scene_sha256"]),
        prompt_id=VALID_PROMPT_ID,
        text=VALID_TEXT,
        checkpoint_sha256="a" * 64,
        stats_sha256=SYNTHETIC_STATS_SHA256,
        teacher_runtime_sha256=SYNTHETIC_RUNTIME_SHA256,
        diffusion_steps=diffusion_steps,
        k_draws=draws.shape[0],
        base_seed=base_seed,
    )
    cache_key = canonical_json_sha256(payload)
    seeds = [
        derive_draw_seeds(
            cache_key=cache_key, base_seed=base_seed, draw_index=draw_index
        )
        for draw_index in range(draws.shape[0])
    ]
    if transform is None:
        transform = {
            "name": "identity",
            "source_model_output": "affordance",
            "canonical_output": "affordance",
            "gaussian_applied": False,
            "denormalization_applied": False,
            "distance_kernel_reapplied": False,
        }
    manifest, artifact = write_base_teacher_artifact(
        output_dir=root / "artifact",
        scene=scene,
        prompt_id=VALID_PROMPT_ID,
        text=VALID_TEXT,
        affordance_draws=draws,
        transform=transform,
        quality_gate={
            "status": "PASS",
            "evidence_set_sha256": "b" * 64,
            "rollout_protocol_sha256": dict(
                SYNTHETIC_ROLLOUT_PROTOCOL_SHA256
            ),
            "sha256": {
                "fewshot_checkpoint": "a" * 64,
                "stats": SYNTHETIC_STATS_SHA256,
                "synthetic": "5" * 64,
            },
        },
        runtime_provenance=_valid_runtime(),
        cache_key_payload=payload,
        draw_seeds=seeds,
        source_representation=source_representation,
    )
    return manifest, artifact, scene, draws


class HashPromptAndSeedTests(unittest.TestCase):
    def test_channel_contract_uses_exact_v5r4_native_wrist_names(self) -> None:
        self.assertEqual(
            CHANNEL_ORDER,
            (
                "pelvis",
                "left_foot",
                "right_foot",
                "neck",
                "left_wrist",
                "right_wrist",
            ),
        )
        self.assertEqual(CHANNEL_JOINT_INDICES, (0, 10, 11, 12, 20, 21))

    def test_array_hash_binds_dtype_shape_and_bytes(self) -> None:
        value = np.arange(12, dtype=np.float32).reshape(2, 6)
        self.assertEqual(sha256_array(value), sha256_array(value.copy()))
        self.assertNotEqual(sha256_array(value), sha256_array(value.astype(np.float64)))
        self.assertNotEqual(sha256_array(value), sha256_array(value.reshape(3, 4)))
        changed = value.copy()
        changed[0, 0] += np.float32(1.0)
        self.assertNotEqual(sha256_array(value), sha256_array(changed))

    def test_canonical_json_hash_is_mapping_order_independent(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": 2}),
            canonical_json_sha256({"b": 2, "a": 1}),
        )

    def test_prompt_contract_accepts_only_clean_ids_and_text(self) -> None:
        validate_prompt(VALID_PROMPT_ID, VALID_TEXT)
        for prompt_id in ("", "Bad", "-bad", "bad space", "bad/"):
            with self.subTest(prompt_id=prompt_id):
                with self.assertRaises(ValueError):
                    validate_prompt(prompt_id, VALID_TEXT)
        for text in ("", " ", " Sit somewhere.", "Sit somewhere. "):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    validate_prompt(VALID_PROMPT_ID, text)

    def test_cache_payload_binds_every_sampling_condition(self) -> None:
        base = build_cache_key_payload(
            scene_sha256="1" * 64,
            prompt_id=VALID_PROMPT_ID,
            text=VALID_TEXT,
            checkpoint_sha256="2" * 64,
            stats_sha256="3" * 64,
            teacher_runtime_sha256="4" * 64,
            diffusion_steps=500,
            k_draws=5,
            base_seed=7,
        )
        self.assertEqual(base["prompt_policy_id"], PROMPT_POLICY_ID)
        variants = (
            {"scene_sha256": "3" * 64},
            {"prompt_id": "sit.alternate.v5"},
            {"text": "Lie down somewhere."},
            {"checkpoint_sha256": "5" * 64},
            {"stats_sha256": "6" * 64},
            {"teacher_runtime_sha256": "7" * 64},
            {"diffusion_steps": 499},
            {"k_draws": 6},
            {"base_seed": 8},
        )
        defaults = {
            "scene_sha256": "1" * 64,
            "prompt_id": VALID_PROMPT_ID,
            "text": VALID_TEXT,
            "checkpoint_sha256": "2" * 64,
            "stats_sha256": "3" * 64,
            "teacher_runtime_sha256": "4" * 64,
            "diffusion_steps": 500,
            "k_draws": 5,
            "base_seed": 7,
        }
        for delta in variants:
            with self.subTest(delta=delta):
                other = build_cache_key_payload(**{**defaults, **delta})
                self.assertNotEqual(
                    canonical_json_sha256(base), canonical_json_sha256(other)
                )

    def test_cache_payload_rejects_non_v5_prompt_or_policy(self) -> None:
        common = {
            "scene_sha256": "1" * 64,
            "prompt_id": VALID_PROMPT_ID,
            "checkpoint_sha256": "2" * 64,
            "stats_sha256": "3" * 64,
            "teacher_runtime_sha256": "4" * 64,
            "diffusion_steps": 500,
            "k_draws": 5,
            "base_seed": 7,
        }
        with self.assertRaises(ValueError):
            build_cache_key_payload(text="Sit on the chair.", **common)
        with self.assertRaises(ValueError):
            build_cache_key_payload(
                text=VALID_TEXT, prompt_policy_id="untrusted_policy", **common
            )

    def test_draw_seeds_are_deterministic_distinct_and_63_bit(self) -> None:
        first = derive_draw_seeds(cache_key="a" * 64, base_seed=9, draw_index=0)
        self.assertEqual(
            first,
            derive_draw_seeds(cache_key="a" * 64, base_seed=9, draw_index=0),
        )
        second = derive_draw_seeds(cache_key="a" * 64, base_seed=9, draw_index=1)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first[0], first[1])
        for seed in first + second:
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 1 << 63)
        with self.assertRaises(ValueError):
            derive_draw_seeds(cache_key="a" * 64, base_seed=-1, draw_index=0)
        with self.assertRaises(ValueError):
            derive_draw_seeds(cache_key="a" * 64, base_seed=1, draw_index=-1)

    def test_cache_payload_rejects_non_hex_hashes(self) -> None:
        with self.assertRaises(ValueError):
            build_cache_key_payload(
                scene_sha256="g" * 64,
                prompt_id=VALID_PROMPT_ID,
                text=VALID_TEXT,
                checkpoint_sha256="z" * 64,
                stats_sha256="3" * 64,
                teacher_runtime_sha256="4" * 64,
                diffusion_steps=500,
                k_draws=5,
                base_seed=7,
            )

    def test_cache_payload_rejects_fractional_integer_fields(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            build_cache_key_payload(
                scene_sha256="1" * 64,
                prompt_id=VALID_PROMPT_ID,
                text=VALID_TEXT,
                checkpoint_sha256="2" * 64,
                stats_sha256="3" * 64,
                teacher_runtime_sha256="4" * 64,
                diffusion_steps=500.75,
                k_draws=5.5,
                base_seed=7.25,
            )


class ConversionAndStatsTests(unittest.TestCase):
    def test_affordance_conversion_is_exact_identity(self) -> None:
        value = np.zeros((NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
        value[:, 0] = np.float32(0.0)
        value[:, 1] = np.float32(0.1)
        value[:, 2] = np.float32(0.5)
        value[:, 3] = np.float32(0.7)
        value[:, 4] = np.float32(0.9)
        value[:, 5] = np.float32(1.0)
        converted, transform = convert_prediction_draws(
            value, representation="affordance"
        )
        self.assertEqual(converted.shape, (1, NUM_POINTS, NUM_CHANNELS))
        self.assertTrue(np.array_equal(converted[0], value))
        self.assertEqual(converted.tobytes(), value.tobytes())
        self.assertEqual(transform["name"], "identity")
        self.assertIs(transform["gaussian_applied"], False)
        self.assertIs(transform["denormalization_applied"], False)

    def test_affordance_zero_does_not_become_one(self) -> None:
        value = np.zeros((1, NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
        converted, _ = convert_prediction_draws(value, representation="affordance")
        self.assertEqual(float(converted.max()), 0.0)

    def test_normalized_contact_denormalizes_and_clips(self) -> None:
        normalized = np.empty((2, NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
        normalized[0].fill(-10.0)
        normalized[1].fill(10.0)
        mean = np.full((1, NUM_CHANNELS), 0.2, dtype=np.float32)
        std = np.full((1, NUM_CHANNELS), 0.1, dtype=np.float32)
        converted, transform = convert_prediction_draws(
            normalized,
            representation="normalized_contact",
            mean=mean,
            std=std,
        )
        self.assertTrue(np.array_equal(converted[0], np.full_like(converted[0], 1e-20)))
        self.assertTrue(np.array_equal(converted[1], np.ones_like(converted[1])))
        self.assertEqual(transform["name"], "contact_denormalize_then_clip")
        self.assertIs(transform["gaussian_applied"], False)
        self.assertIs(transform["denormalization_applied"], True)
        self.assertEqual(transform["mean_sha256"], sha256_array(mean))
        self.assertEqual(transform["std_sha256"], sha256_array(std))

    def test_distance_conversion_applies_one_known_gaussian(self) -> None:
        distance = np.zeros((1, NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
        distance[:, :, 1] = np.float32(0.8)
        converted, transform = convert_prediction_draws(
            distance, representation="distance", sigma=0.8
        )
        self.assertTrue(np.array_equal(converted[:, :, 0], np.ones((1, NUM_POINTS))))
        expected = np.float32(np.exp(-0.5))
        self.assertTrue(np.allclose(converted[:, :, 1], expected, atol=1e-7))
        self.assertEqual(transform["name"], "gaussian_distance_to_affordance")
        self.assertIs(transform["gaussian_applied"], True)

    def test_conversion_rejects_ambiguous_or_invalid_inputs(self) -> None:
        valid = np.zeros((1, NUM_POINTS, NUM_CHANNELS), dtype=np.float32)
        cases = []
        cases.append((lambda: convert_prediction_draws(valid, representation="unknown")))
        cases.append(
            lambda: convert_prediction_draws(
                np.zeros((NUM_CHANNELS, NUM_POINTS), dtype=np.float32),
                representation="affordance",
            )
        )
        out_of_range = valid.copy()
        out_of_range[0, 0, 0] = np.float32(1.01)
        cases.append(
            lambda: convert_prediction_draws(out_of_range, representation="affordance")
        )
        nonfinite = valid.copy()
        nonfinite[0, 0, 0] = np.nan
        cases.append(
            lambda: convert_prediction_draws(nonfinite, representation="affordance")
        )
        cases.append(
            lambda: convert_prediction_draws(valid, representation="normalized_contact")
        )
        cases.append(
            lambda: convert_prediction_draws(valid, representation="distance", sigma=None)
        )
        negative_distance = valid.copy()
        negative_distance[0, 0, 0] = np.float32(-0.1)
        cases.append(
            lambda: convert_prediction_draws(
                negative_distance, representation="distance", sigma=0.8
            )
        )
        for index, call in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    call()

    def test_contact_stats_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_stats_") as temp:
            root = Path(temp)
            valid = root / "valid.npz"
            mean = np.zeros((1, NUM_CHANNELS), dtype=np.float32)
            std = np.ones((1, NUM_CHANNELS), dtype=np.float32)
            np.savez_compressed(valid, mean=mean, std=std)
            loaded_mean, loaded_std = load_contact_stats(valid)
            self.assertTrue(np.array_equal(loaded_mean, mean))
            self.assertTrue(np.array_equal(loaded_std, std))

            missing = root / "missing.npz"
            np.savez_compressed(missing, mean=mean)
            with self.assertRaises(KeyError):
                load_contact_stats(missing)
            wrong_shape = root / "wrong_shape.npz"
            np.savez_compressed(
                wrong_shape,
                mean=np.zeros((NUM_CHANNELS,), dtype=np.float32),
                std=std,
            )
            with self.assertRaises(ValueError):
                load_contact_stats(wrong_shape)
            bad_std = root / "bad_std.npz"
            np.savez_compressed(bad_std, mean=mean, std=np.zeros_like(std))
            with self.assertRaises(ValueError):
                load_contact_stats(bad_std)


class SceneContractTests(unittest.TestCase):
    def test_scene_contract_binds_point_order_rgb_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_scene_") as temp:
            fixture = _make_scene_files(Path(temp))
            scene = load_scene_contract(fixture["dataset_root"], fixture["scene_id"])
            self.assertEqual(scene["points"].shape, (NUM_POINTS, 6))
            self.assertTrue(np.array_equal(scene["xyz"], fixture["xyz"]))
            self.assertTrue(
                np.array_equal(
                    scene["rgb01"], fixture["points"][:, 3:6] / np.float32(255.0)
                )
            )
            self.assertEqual(
                scene["hash_payload"]["source_indices_sha256"],
                sha256_array(fixture["source_indices"]),
            )
            self.assertEqual(
                scene["scene_sha256"], canonical_json_sha256(scene["hash_payload"])
            )

    def test_scene_contract_rejects_point_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_scene_") as temp:
            fixture = _make_scene_files(Path(temp))
            xyz = fixture["xyz"].copy()
            xyz[[0, 1]] = xyz[[1, 0]]
            np.savez_compressed(
                fixture["sidecar_file"],
                xyz_afford_z_up=xyz,
                instance_ids=fixture["instance_ids"],
                source_indices=fixture["source_indices"],
            )
            with self.assertRaises(ValueError):
                load_scene_contract(fixture["dataset_root"], fixture["scene_id"])

    def test_scene_contract_rejects_duplicate_source_indices(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_scene_") as temp:
            fixture = _make_scene_files(Path(temp))
            indices = fixture["source_indices"].copy()
            indices[1] = indices[0]
            np.savez_compressed(
                fixture["sidecar_file"],
                xyz_afford_z_up=fixture["xyz"],
                instance_ids=fixture["instance_ids"],
                source_indices=indices,
            )
            with self.assertRaises(ValueError):
                load_scene_contract(fixture["dataset_root"], fixture["scene_id"])

    def test_scene_contract_rejects_invalid_rgb_nan_and_shape(self) -> None:
        mutations = ("rgb", "nan", "shape")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory(prefix="base_teacher_scene_") as temp:
                    fixture = _make_scene_files(Path(temp))
                    points = fixture["points"].copy()
                    if mutation == "rgb":
                        points[0, 3] = np.float32(256.0)
                    elif mutation == "nan":
                        points[0, 0] = np.nan
                    else:
                        points = points[:-1]
                    np.savez_compressed(fixture["points_file"], points=points)
                    with self.assertRaises(ValueError):
                        load_scene_contract(
                            fixture["dataset_root"], fixture["scene_id"]
                        )


class QualityChainTests(unittest.TestCase):
    def test_complete_synthetic_v5r4_chain_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            result = validate_v5r4_quality_chain(
                **paths, minimum_k_draws=5, expected_diffusion_steps=500
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["train_audit_k_draws"], 5)
            self.assertEqual(result["development_k_draws"], 5)
            self.assertEqual(result["prediction_contracts"]["train_rollout"]["sample_count"], 25)
            self.assertEqual(result["prediction_contracts"]["development"]["sample_count"], 12)
            replay_plan = result["quality_replay_plan"]
            self.assertEqual(replay_plan["roles"], ["fewshot", "original"])
            self.assertEqual(replay_plan["total_replayed_draws"], 370)
            self.assertEqual(
                replay_plan["partitions"]["train_audit"][
                    "total_replayed_draws_per_role"
                ],
                125,
            )
            self.assertEqual(
                replay_plan["partitions"]["development"][
                    "total_replayed_draws_per_role"
                ],
                60,
            )
            expected_seal = canonical_json_sha256(
                {
                    "schemas": result["schemas"],
                    "file_sha256": result["sha256"],
                    "diffusion_steps": result["diffusion_steps"],
                    "train_prediction_contract": result["prediction_contracts"][
                        "train_rollout"
                    ],
                    "development_prediction_contract": result[
                        "prediction_contracts"
                    ]["development"],
                    "independent_train_quality": result["independent_quality"]
                    ["train_rollout"],
                    "independent_development_quality": result[
                        "independent_quality"
                    ]["development"],
                    "quality_replay_plan": result["quality_replay_plan"],
                    "rollout_protocol_hashes": result[
                        "rollout_protocol_sha256"
                    ],
                }
            )
            self.assertEqual(result["evidence_set_sha256"], expected_seal)

    def test_quality_chain_rejects_checkpoint_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            paths["fewshot_checkpoint"].write_bytes(b"tampered checkpoint")
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_requires_dataset_and_exact_quality_thresholds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            paths.pop("dataset_root")
            with self.assertRaisesRegex(ValueError, "requires the current dataset_root"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["quality_thresholds"]["novel_min_f1"] = 0.31
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "quality_thresholds"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_failed_or_nonboolean_audit_checks(self) -> None:
        for bad_value in (False, "PASS"):
            with self.subTest(bad_value=bad_value):
                with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
                    paths = _make_quality_chain(Path(temp))
                    audit = _read_json(paths["train_rollout_audit_file"])
                    audit["checks"]["synthetic_audit_check"] = bad_value
                    _write_json(paths["train_rollout_audit_file"], audit)
                    with self.assertRaises((TypeError, ValueError)):
                        validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_inexact_prediction_mean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            prediction_file = paths["development_predictions_file"]
            with np.load(prediction_file, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["fewshot"] = arrays["fewshot"].copy()
            arrays["fewshot"][0, 0, 0] += np.float32(0.01)
            np.savez_compressed(prediction_file, **arrays)
            development = _read_json(paths["development_summary_file"])
            development["predictions"]["sha256"] = sha256_file(prediction_file)
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "not exact mean"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_prediction_sample_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            prediction_file = paths["train_rollout_predictions_file"]
            with np.load(prediction_file, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["sample_ids"] = arrays["sample_ids"][::-1]
            np.savez_compressed(prediction_file, **arrays)
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["predictions"]["sha256"] = sha256_file(prediction_file)
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "IDs/order mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_wrong_split_cardinality_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            split = _read_json(paths["split_file"])
            split["cdm_fewshot"]["test"] = split["cdm_fewshot"]["test"][:-1]
            _write_json(paths["split_file"], split)
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_diffusion_or_draw_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            development = _read_json(paths["development_summary_file"])
            development["diffusion_steps"] = 499
            _write_json(paths["development_summary_file"], development)
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=6)

    def test_quality_chain_rejects_failed_shortlist_checks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            shortlist = _read_json(paths["shortlist_status_file"])
            shortlist["best_one_step"]["selection"]["loss_gate"]["passed"] = False
            _write_json(paths["shortlist_status_file"], shortlist)
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_rollout_selection_marked_not_passed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            rollout = _read_json(paths["rollout_selection_file"])
            rollout["selected"]["passed"] = False
            _write_json(paths["rollout_selection_file"], rollout)
            train = _read_json(paths["train_summary_file"])
            train["checkpoint_selection"]["rollout_selection"][
                "selection_file_sha256"
            ] = sha256_file(paths["rollout_selection_file"])
            _write_json(paths["train_summary_file"], train)
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["train_summary_sha256"] = sha256_file(paths["train_summary_file"])
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_prediction_bundle_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            split = _read_json(paths["split_file"])
            train_ids = tuple(split["cdm_fewshot"]["train"])
            _write_prediction_bundle(
                paths["train_rollout_predictions_file"],
                train_ids,
                5,
                offset=0.20,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_binds_train_rows_to_exact_split_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["per_sample"][0]["sample_id"] = "fabricated_sample"
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_recomputes_train_target_counts_from_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["per_sample"][0]["target"] = "bed"
            audit["per_sample"][0]["text"] = PROMPT_BY_TARGET["bed"]
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaises(ValueError):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_dataset_snapshot_order_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            audit = _read_json(paths["train_rollout_audit_file"])
            snapshot = audit["dataset_snapshot"]
            fingerprints = list(snapshot["sample_fingerprints"].items())
            snapshot["sample_fingerprints"] = dict(reversed(fingerprints))
            snapshot_payload = dict(snapshot)
            snapshot_payload.pop("sha256")
            snapshot["sha256"] = rollout_sha256_json(snapshot_payload)
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "sample IDs/order mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_binds_prediction_gt_to_dataset_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            prediction_file = paths["development_predictions_file"]
            with np.load(prediction_file, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["gt"] = arrays["gt"].copy()
            arrays["gt"][0, 0, 0] += np.float32(0.01)
            np.savez_compressed(prediction_file, **arrays)
            development = _read_json(paths["development_summary_file"])
            development["predictions"]["sha256"] = sha256_file(prediction_file)
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "GT/dataset hash mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_recorded_pass_checks_cannot_hide_failing_prediction_quality(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            prediction_file = paths["development_predictions_file"]
            with np.load(prediction_file, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            development = _read_json(paths["development_summary_file"])
            bed_index = next(
                index
                for index, row in enumerate(development["per_sample"])
                if row["target"] == "bed"
            )
            arrays["fewshot_draws"][bed_index] = np.float32(0.05)
            arrays["fewshot"][bed_index] = arrays["fewshot_draws"][
                bed_index
            ].mean(axis=0, dtype=np.float32)
            np.savez_compressed(prediction_file, **arrays)

            fingerprints = development["dataset_snapshot"]["sample_fingerprints"]
            with np.load(
                paths["dataset_root"]
                / "scenes"
                / "synthetic_test_scene"
                / "adm_input"
                / "sidecar.npz",
                allow_pickle=False,
            ) as source:
                instance_ids = source["instance_ids"].astype(np.int64)
            instance_ids_by_sample = {
                sample_id: instance_ids.copy() for sample_id in fingerprints
            }
            development.update(
                _development_metric_fields(
                    arrays=arrays,
                    fingerprints=fingerprints,
                    instance_ids_by_sample=instance_ids_by_sample,
                )
            )
            development["predictions"]["sha256"] = sha256_file(prediction_file)
            self.assertTrue(all(development["checks"].values()))
            _write_json(paths["development_summary_file"], development)

            with self.assertRaisesRegex(
                ValueError, "development check differs from recomputation"
            ):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_noncanonical_prediction_seed_table(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            prediction_file = paths["train_rollout_predictions_file"]
            with np.load(prediction_file, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["initial_noise_seeds"] = arrays[
                "initial_noise_seeds"
            ].copy()
            arrays["initial_noise_seeds"][0, 0] += np.int64(1)
            np.savez_compressed(prediction_file, **arrays)
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["predictions"]["sha256"] = sha256_file(prediction_file)
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "seed table differs"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_rejects_prediction_extra_keys_and_wrong_dtypes(self) -> None:
        for mutation in ("extra_key", "float64_draws"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory(
                    prefix="base_teacher_quality_"
                ) as temp:
                    paths = _make_quality_chain(Path(temp))
                    prediction_file = paths["development_predictions_file"]
                    with np.load(prediction_file, allow_pickle=False) as source:
                        arrays = {name: source[name] for name in source.files}
                    if mutation == "extra_key":
                        arrays["unexpected"] = np.asarray([1], dtype=np.int64)
                        expected_exception = KeyError
                        message = "key set mismatch"
                    else:
                        arrays["fewshot_draws"] = arrays[
                            "fewshot_draws"
                        ].astype(np.float64)
                        expected_exception = TypeError
                        message = "must be float32"
                    np.savez_compressed(prediction_file, **arrays)
                    development = _read_json(paths["development_summary_file"])
                    development["predictions"]["sha256"] = sha256_file(
                        prediction_file
                    )
                    _write_json(paths["development_summary_file"], development)
                    with self.assertRaisesRegex(expected_exception, message):
                        validate_v5r4_quality_chain(
                            **paths, minimum_k_draws=5
                        )

    def test_quality_chain_rejects_incomplete_check_and_protocol_sets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            rollout = _read_json(paths["rollout_selection_file"])
            removed_check = sorted(REQUIRED_ROLLOUT_SELECTION_CHECKS)[0]
            rollout["selected"]["checks"].pop(removed_check)
            rollout["evaluated"][0]["checks"].pop(removed_check)
            _write_json(paths["rollout_selection_file"], rollout)
            train = _read_json(paths["train_summary_file"])
            train["checkpoint_selection"]["rollout_selection"][
                "selection_file_sha256"
            ] = sha256_file(paths["rollout_selection_file"])
            _write_json(paths["train_summary_file"], train)
            with self.assertRaisesRegex(ValueError, "check set is incomplete"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["checks"].pop(sorted(REQUIRED_AUDIT_CHECKS)[0])
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "check set mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            development = _read_json(paths["development_summary_file"])
            development["checks"]["unexpected_check"] = True
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "check set mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["rollout_protocol_hashes"].pop(ROLLOUT_PROTOCOL_FILES[0])
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "protocol file set mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

    def test_quality_chain_binds_rollout_probe_and_summary_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            train = _read_json(paths["train_summary_file"])
            train["checkpoint_selection"]["rollout_selection"][
                "seed_partition"
            ] = "wrong_partition"
            _write_json(paths["train_summary_file"], train)
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["train_summary_sha256"] = sha256_file(paths["train_summary_file"])
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            train = _read_json(paths["train_summary_file"])
            train["checkpoint_selection"]["rollout_selection"][
                "evaluated_candidate_count"
            ] = 2
            _write_json(paths["train_summary_file"], train)
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["train_summary_sha256"] = sha256_file(
                paths["train_summary_file"]
            )
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(
                ValueError,
                "evaluated_candidate_count vs candidate_count",
            ):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)

        with tempfile.TemporaryDirectory(prefix="base_teacher_quality_") as temp:
            paths = _make_quality_chain(Path(temp))
            rollout = _read_json(paths["rollout_selection_file"])
            rollout["probe_sample_ids"] = list(
                reversed(rollout["probe_sample_ids"])
            )
            _write_json(paths["rollout_selection_file"], rollout)
            train = _read_json(paths["train_summary_file"])
            recorded = train["checkpoint_selection"]["rollout_selection"]
            recorded["probe_sample_ids"] = list(rollout["probe_sample_ids"])
            recorded["selection_file_sha256"] = sha256_file(
                paths["rollout_selection_file"]
            )
            _write_json(paths["train_summary_file"], train)
            audit = _read_json(paths["train_rollout_audit_file"])
            audit["train_summary_sha256"] = sha256_file(paths["train_summary_file"])
            _write_json(paths["train_rollout_audit_file"], audit)
            development = _read_json(paths["development_summary_file"])
            development["train_rollout_audit"]["sha256"] = sha256_file(
                paths["train_rollout_audit_file"]
            )
            _write_json(paths["development_summary_file"], development)
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                validate_v5r4_quality_chain(**paths, minimum_k_draws=5)


class ArtifactTests(unittest.TestCase):
    def test_write_and_validate_identity_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, scene, draws = _write_valid_artifact(Path(temp))
            result = validate_base_teacher_artifact(
                manifest_file=manifest, artifact_file=artifact
            )
            self.assertEqual(result["status"], "INTEGRITY_PASS")
            self.assertEqual(result["draw_count"], 5)
            record = load_json(manifest)
            self.assertEqual(record["status"], "STAGED")
            self.assertIs(record["promotion_authorized"], False)
            self.assertEqual(record["canonical_representation"], "affordance")
            self.assertEqual(record["draw_count"], 5)
            self.assertEqual(record["draw_shape"], [5, NUM_POINTS, NUM_CHANNELS])
            self.assertEqual(record["cache_key_payload"]["k_draws"], 5)
            self.assertEqual(record["cache_key_payload"]["diffusion_steps"], 500)
            self.assertEqual(record["cache_key_payload"]["base_seed"], 20260815)
            self.assertEqual(record["quality_gate"]["evidence_set_sha256"], "b" * 64)
            self.assertEqual(record["transform"]["name"], "identity")
            self.assertIs(record["transform"]["distance_kernel_reapplied"], False)
            with np.load(artifact, allow_pickle=False) as source:
                self.assertTrue(np.array_equal(source["affordance_draws"], draws))
                self.assertTrue(np.array_equal(source["xyz"], scene["xyz"]))
                self.assertTrue(np.array_equal(source["rgb01"], scene["rgb01"]))
                self.assertEqual(source["channel_names"].astype(str).tolist(), list(CHANNEL_ORDER))
                self.assertEqual(source["channel_joint_indices"].tolist(), list(CHANNEL_JOINT_INDICES))

    def test_artifact_requires_exactly_five_draws_and_500_steps(self) -> None:
        for name, overrides in (
            ("four_draws", {"k_draws": 4}),
            ("wrong_diffusion", {"diffusion_steps": 499}),
        ):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(
                    prefix="base_teacher_artifact_policy_"
                ) as temp:
                    with self.assertRaises(ValueError):
                        _write_valid_artifact(Path(temp), **overrides)

    def test_artifact_rejects_self_consistent_noncanonical_base_seed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="base_teacher_artifact_wrong_seed_"
        ) as temp:
            with self.assertRaisesRegex(ValueError, "base seed 20260815"):
                _write_valid_artifact(Path(temp), base_seed=20260820)

    def test_writer_rejects_overwrite_bad_status_shape_seed_and_range(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            root = Path(temp)
            manifest, artifact, scene, draws = _write_valid_artifact(root)
            record = load_json(manifest)
            seeds = [
                (row["initial_xT_seed"], row["reverse_noise_seed"])
                for row in record["draw_seeds"]
            ]
            common = {
                "output_dir": root / "artifact",
                "scene": scene,
                "prompt_id": VALID_PROMPT_ID,
                "text": VALID_TEXT,
                "affordance_draws": draws,
                "transform": record["transform"],
                "quality_gate": {"status": "PASS"},
                "runtime_provenance": _valid_runtime(),
                "cache_key_payload": record["cache_key_payload"],
                "draw_seeds": seeds,
                "source_representation": "affordance",
            }
            with self.assertRaises(FileExistsError):
                write_base_teacher_artifact(**common)

            for name, delta in (
                ("quality", {"quality_gate": {"status": "FAIL"}}),
                ("runtime", {"runtime_provenance": {"status": "FAIL"}}),
                ("seeds", {"draw_seeds": seeds[:-1]}),
                ("shape", {"affordance_draws": draws[:, :-1, :]}),
            ):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        write_base_teacher_artifact(
                            **{**common, **delta, "output_dir": root / ("bad_" + name)}
                        )
            bad = draws.copy()
            bad[0, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                write_base_teacher_artifact(
                    **{**common, "affordance_draws": bad, "output_dir": root / "bad_nan"}
                )
            self.assertTrue(manifest.is_file())
            self.assertTrue(artifact.is_file())

    def test_writer_forbids_distance_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            root = Path(temp)
            scene_files = _make_scene_files(root)
            scene = load_scene_contract(
                scene_files["dataset_root"], scene_files["scene_id"]
            )
            draws = _valid_draws()
            payload = build_cache_key_payload(
                scene_sha256=scene["scene_sha256"],
                prompt_id=VALID_PROMPT_ID,
                text=VALID_TEXT,
                checkpoint_sha256="a" * 64,
                stats_sha256=SYNTHETIC_STATS_SHA256,
                teacher_runtime_sha256=SYNTHETIC_RUNTIME_SHA256,
                diffusion_steps=500,
                k_draws=draws.shape[0],
                base_seed=1,
            )
            with self.assertRaises(ValueError):
                write_base_teacher_artifact(
                    output_dir=root / "artifact",
                    scene=scene,
                    prompt_id=VALID_PROMPT_ID,
                    text=VALID_TEXT,
                    affordance_draws=draws,
                    transform={
                        "name": "gaussian_distance_to_affordance",
                        "gaussian_applied": True,
                        "denormalization_applied": False,
                        "distance_kernel_reapplied": True,
                    },
                    quality_gate={"status": "PASS"},
                    runtime_provenance=_valid_runtime(),
                    cache_key_payload=payload,
                    draw_seeds=[(1, 2), (3, 4), (5, 6)],
                    source_representation="distance",
                )

    def test_validator_rejects_file_and_array_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
            with np.load(artifact, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["affordance_draws"] = arrays["affordance_draws"].copy()
            arrays["affordance_draws"][0, 0, 0] += np.float32(0.01)
            np.savez_compressed(artifact, **arrays)
            with self.assertRaisesRegex(ValueError, "artifact file SHA-256 mismatch"):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )
            record = load_json(manifest)
            record["artifact_file_sha256"] = sha256_file(artifact)
            atomic_write_json(manifest, record)
            with self.assertRaisesRegex(ValueError, "array SHA-256 mismatch"):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )

    def test_validator_rejects_manifest_semantic_tampering(self) -> None:
        mutations = {
            "status": lambda record: record.update({"status": "PASS"}),
            "promotion": lambda record: record.update({"promotion_authorized": True}),
            "channel_order": lambda record: record.update(
                {"channel_order": list(reversed(CHANNEL_ORDER))}
            ),
            "joint_order": lambda record: record.update(
                {"channel_joint_indices": list(reversed(CHANNEL_JOINT_INDICES))}
            ),
            "sigma": lambda record: record.update({"kernel_sigma_m": 0.5}),
            "quality": lambda record: record["quality_gate"].update({"status": "FAIL"}),
            "quality_seal": lambda record: record["quality_gate"].update(
                {"evidence_set_sha256": "not-a-sha256"}
            ),
            "runtime": lambda record: record["runtime_provenance"].update(
                {"all_parameters_frozen": False}
            ),
            "runtime_registry": lambda record: record["runtime_provenance"][
                "runtime_file_sha256"
            ].pop("utils/registry.py"),
            "runtime_pointops_binary": lambda record: record[
                "runtime_provenance"
            ]["runtime_file_sha256"].pop("binary/pointops_cuda"),
            "runtime_torch_build": lambda record: record[
                "runtime_provenance"
            ].pop("torch_build_config_sha256"),
            "runtime_torch_build_tampered": lambda record: record[
                "runtime_provenance"
            ].update({"torch_build_config_sha256": "0" * 64}),
            "gaussian": lambda record: record["transform"].update(
                {"gaussian_applied": True}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
                    manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
                    record = load_json(manifest)
                    mutate(record)
                    atomic_write_json(manifest, record)
                    with self.assertRaises(ValueError):
                        validate_base_teacher_artifact(
                            manifest_file=manifest, artifact_file=artifact
                        )

    def test_validator_rejects_extra_npz_key(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
            with np.load(artifact, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["unexpected"] = np.asarray([1], dtype=np.int64)
            np.savez_compressed(artifact, **arrays)
            record = load_json(manifest)
            record["artifact_file_sha256"] = sha256_file(artifact)
            atomic_write_json(manifest, record)
            with self.assertRaisesRegex(KeyError, "forbidden extra keys"):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )

    def test_writer_rejects_bad_instance_and_source_index_shapes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            root = Path(temp)
            scene_files = _make_scene_files(root)
            scene = load_scene_contract(
                scene_files["dataset_root"], scene_files["scene_id"]
            )
            with self.assertRaises(ValueError):
                _write_valid_artifact(
                    root / "bad",
                    scene_override={
                        "instance_ids": scene["instance_ids"][:-1],
                        "source_indices": scene["source_indices"][:-1],
                    },
                )

    def test_writer_rejects_self_consistent_malformed_index_arrays(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            root = Path(temp)
            fixture = _make_scene_files(root)
            scene = load_scene_contract(fixture["dataset_root"], fixture["scene_id"])
            bad_instance_ids = scene["instance_ids"][:-1]
            bad_source_indices = np.zeros(NUM_POINTS - 1, dtype=np.int64)
            hash_payload = dict(scene["hash_payload"])
            hash_payload["instance_ids_sha256"] = sha256_array(bad_instance_ids)
            hash_payload["source_indices_sha256"] = sha256_array(bad_source_indices)
            bad_scene = {
                **scene,
                "instance_ids": bad_instance_ids,
                "source_indices": bad_source_indices,
                "hash_payload": hash_payload,
                "scene_sha256": canonical_json_sha256(hash_payload),
            }
            with self.assertRaises(ValueError):
                _write_valid_artifact(root / "bad", scene_override=bad_scene)

    def test_validator_binds_text_and_text_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
            record = load_json(manifest)
            record["text"] = "Lie down somewhere."
            atomic_write_json(manifest, record)
            with self.assertRaises(ValueError):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )

    def test_validator_binds_manifest_condition_to_cache_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
            record = load_json(manifest)
            record["cache_key_payload"]["prompt_id"] = "different.prompt.v5"
            record["cache_key"] = canonical_json_sha256(record["cache_key_payload"])
            with np.load(artifact, allow_pickle=False) as source:
                affordance = source["affordance"]
                draws = source["affordance_draws"]
            record["artifact_id"] = canonical_json_sha256(
                {
                    "schema": ARTIFACT_SCHEMA,
                    "cache_key": record["cache_key"],
                    "affordance_sha256": sha256_array(affordance),
                    "draws_sha256": sha256_array(draws),
                    "scene_sha256": record["scene_sha256"],
                }
            )
            atomic_write_json(manifest, record)
            with self.assertRaises(ValueError):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )

    def test_validator_checks_declared_shapes_policy_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
            record = load_json(manifest)
            record["shape"] = [1, 1]
            record["draw_shape"] = [1, 1, 1]
            record["draw_policy"] = "untrusted"
            record["normalization_state"] = "normalized_contact"
            atomic_write_json(manifest, record)
            with self.assertRaises(ValueError):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )

    def test_validator_rederives_and_requires_unique_draw_seeds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_artifact_") as temp:
            manifest, artifact, _, _ = _write_valid_artifact(Path(temp))
            record = load_json(manifest)
            duplicate = {
                "draw_index": 0,
                "initial_xT_seed": 1,
                "reverse_noise_seed": 1,
            }
            record["draw_seeds"] = [copy.deepcopy(duplicate) for _ in record["draw_seeds"]]
            with np.load(artifact, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["initial_noise_seeds"] = np.ones_like(arrays["initial_noise_seeds"])
            arrays["reverse_noise_seeds"] = np.ones_like(arrays["reverse_noise_seeds"])
            np.savez_compressed(artifact, **arrays)
            record["artifact_file_sha256"] = sha256_file(artifact)
            record["array_sha256"] = {
                name: sha256_array(value) for name, value in arrays.items()
            }
            atomic_write_json(manifest, record)
            with self.assertRaises(ValueError):
                validate_base_teacher_artifact(
                    manifest_file=manifest, artifact_file=artifact
                )


class JsonAtomicTests(unittest.TestCase):
    def test_atomic_json_and_load_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_json_") as temp:
            path = Path(temp) / "nested" / "value.json"
            value = {"status": "PASS", "unicode": "의미", "number": 7}
            atomic_write_json(path, value)
            self.assertEqual(load_json(path), value)
            self.assertEqual(sha256_text("abc"), sha256_text("abc"))
            self.assertNotEqual(sha256_text("abc"), sha256_text("abd"))

    def test_load_json_rejects_non_object_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="base_teacher_json_") as temp:
            root = Path(temp)
            sequence = root / "sequence.json"
            sequence.write_text("[]\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_json(sequence)
            with self.assertRaises(FileNotFoundError):
                load_json(root / "missing.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Strict artifact contract for a frozen history-affordance Base teacher.

The v2 pipeline treats the selected CDM+LoRA prediction as an affordance
tensor, never as an implicitly typed array.  This module deliberately keeps
quality-gate validation and artifact serialization independent of PyTorch so
they can be tested on a CPU-only machine before any server-side sampling.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ARTIFACT_SCHEMA = "history_affordance_v2_base_teacher_v1"
TRAIN_SCHEMA = "history_affordance_v1_fewshot_cdm_train_v5r4"
AUDIT_SCHEMA = "history_affordance_v1_fewshot_cdm_train_rollout_audit_v6"
DEVELOPMENT_SCHEMA = (
    "history_affordance_v1_fewshot_cdm_development_eval_v5r4_sealed_v2"
)
SHORTLIST_SCHEMA = "history_affordance_v1_fewshot_cdm_v5r4_shortlist_status_v1"
ROLLOUT_SELECTION_SCHEMA = (
    "history_affordance_v1_fewshot_cdm_candidate_rollout_v1"
)
ROLLOUT_PROTOCOL_FILES = (
    "prepare/evaluate_fewshot_cdm.py",
    "prepare/audit_fewshot_cdm_train_rollout.py",
    "prepare/fewshot_cdm_common.py",
    "prepare/fewshot_cdm_lora.py",
    "prepare/fewshot_cdm_rollout_cache.py",
    "prepare/train_fewshot_cdm.py",
    "prepare/fewshot_cdm_v5_semantics.py",
    "models/base.py",
    "models/cdm.py",
    "models/functions.py",
    "models/modules.py",
    "models/scene_models/pointops.py",
    "models/scene_models/pointtransformer.py",
    "diffusion/gaussian_diffusion.py",
    "diffusion/losses.py",
    "diffusion/nn.py",
    "diffusion/resample.py",
    "diffusion/respace.py",
    "utils/misc.py",
    "utils/training.py",
    "configs/default.yaml",
    "configs/model/cdm.yaml",
    "configs/task/contact_gen.yaml",
)
NUM_POINTS = 8192
NUM_CHANNELS = 6
CHANNEL_ORDER = (
    "pelvis",
    "left_foot",
    "right_foot",
    "neck",
    "left_wrist",
    "right_wrist",
)
CHANNEL_JOINT_INDICES = (0, 10, 11, 12, 20, 21)
PROMPT_POLICY_ID = "object_agnostic_action_v5"
PROMPT_BY_TARGET = {
    "chair": "Sit somewhere.",
    "bed": "Lie down somewhere.",
    "whiteboard": (
        "Write on a nearby vertical surface with the right hand."
    ),
}
TARGET_INSTANCE_IDS = {"chair": 1, "bed": 2, "whiteboard": 3}
COMMON_QUALITY_THRESHOLDS = {
    "schema": "fewshot_cdm_rollout_quality_thresholds_v1",
    "active_threshold": 0.7,
    "instance_top_fraction": 0.1,
    "chair_mae_degradation_limit": 0.05,
    "novel_min_f1": 0.30,
    "dominance_margin": 0.01,
    "per_draw_min_rate": 0.80,
    "sit_chair_primary_margin": 0.15,
    "sit_bed_any_joint_band": [0.12, 0.40],
    "sit_bed_pelvis_band": [0.06, 0.25],
    "candidate_instance_ids": dict(TARGET_INSTANCE_IDS),
    "channel_indices": {
        "pelvis": 0,
        "right_wrist": 5,
        "any_joint": "max_over_6",
    },
}
AUDIT_QUALITY_THRESHOLDS = {
    **COMMON_QUALITY_THRESHOLDS,
    "chair_min_dominance_rate": 0.90,
    "sit_bed_candidate_min_rate": 0.80,
    "novel_min_dominance_rate": 1.0,
}
DEVELOPMENT_QUALITY_THRESHOLDS = dict(COMMON_QUALITY_THRESHOLDS)
REQUIRED_ROLLOUT_SELECTION_CHECKS = {
    "chair_mae_retained",
    "chair_pelvis_dominance",
    "chair_any_joint_dominance",
    "chair_per_draw_dominance",
    "sit_chair_remains_primary",
    "sit_bed_candidate_visible_and_bounded",
    "sit_per_draw_chair_primary",
    "sit_per_draw_bed_candidate_visible_and_bounded",
    "bed_f1_improved",
    "bed_f1_active",
    "bed_mae_beats_zero",
    "bed_pelvis_dominance",
    "bed_any_joint_dominance",
    "bed_per_draw_dominance",
    "whiteboard_f1_improved",
    "whiteboard_f1_active",
    "whiteboard_mae_beats_zero",
    "whiteboard_any_joint_dominance",
    "whiteboard_right_wrist_dominance",
    "whiteboard_per_draw_any_joint_dominance",
    "whiteboard_per_draw_right_wrist_dominance",
}
REQUIRED_AUDIT_CHECKS = {
    "training_schema_is_strict_v5r4",
    "training_status_pass",
    "checkpoint_hash_matches_training",
    "original_checkpoint_hash_matches_training",
    "split_hash_matches_training",
    "training_was_train_only",
    "zero_init_lora_and_multinoise_teacher_contract_recorded",
    "rollout_aligned_sparse_objective_recorded",
    "v5r4_prompt_and_semantic_contract_recorded",
    "one_step_and_semantic_shortlist_gate_passed",
    "train_only_full_rollout_selected_checkpoint",
    "fewshot_checkpoint_keys_exact",
    "constructor_loaded_frozen_state_bitwise_equal",
    "partition_is_train_only",
    "heldout_sample_tensors_never_read",
    "paired_noise_full_diffusion_k_at_least_5",
    "full_trajectory_repeatability_canary",
    "chair_rollout_mae_retained",
    "bed_rollout_mae_improved",
    "whiteboard_rollout_mae_improved",
    "bed_rollout_mae_beats_zero",
    "whiteboard_rollout_mae_beats_zero",
    "bed_rollout_f1_improved",
    "whiteboard_rollout_f1_improved",
    "bed_rollout_f1_active",
    "whiteboard_rollout_f1_active",
    "chair_rollout_any_joint_dominance",
    "chair_rollout_pelvis_dominance",
    "chair_per_draw_dominance_rate",
    "sit_chair_remains_primary",
    "sit_bed_candidate_visible_and_bounded",
    "sit_per_draw_contract_rate",
    "bed_rollout_any_joint_dominance",
    "bed_rollout_pelvis_dominance",
    "bed_per_draw_dominance_rate",
    "whiteboard_rollout_any_joint_dominance",
    "whiteboard_rollout_right_wrist_dominance",
    "whiteboard_per_draw_any_joint_dominance_rate",
    "whiteboard_per_draw_right_wrist_dominance_rate",
    "v5_prompts_exact_and_object_agnostic",
}
REQUIRED_DEVELOPMENT_CHECKS = {
    "training_schema_is_strict_v5r4",
    "original_checkpoint_hash_matches_training",
    "fewshot_checkpoint_hash_matches_training",
    "split_hash_matches_training",
    "training_selected_without_test",
    "one_sample_checkpoint_absent",
    "chair3_bed1_whiteboard1_replay_recorded",
    "zero_init_lora_multinoise_frozen_base_recorded",
    "rollout_aligned_sparse_objective_recorded",
    "v5r4_prompt_policy_recorded",
    "sit_multicandidate_v5r4_policy_recorded",
    "one_step_shortlist_gate_passed",
    "train_only_full_rollout_selected_checkpoint",
    "heldout_sample_tensors_never_read_during_training",
    "train_rollout_audit_schema_valid",
    "train_rollout_audit_status_pass",
    "train_rollout_partition_is_train",
    "train_rollout_read_no_development_tensors",
    "train_rollout_complete_train_partition",
    "train_rollout_used_paired_full_diffusion",
    "train_rollout_checkpoint_hash_matches",
    "train_rollout_original_hash_matches",
    "train_rollout_stats_hash_matches",
    "train_rollout_protocol_hashes_match",
    "train_rollout_pairing_contract_valid",
    "train_rollout_summary_hash_matches",
    "train_rollout_split_hash_matches",
    "every_train_rollout_gate_passed",
    "heldout_count_is_12",
    "heldout_target_counts_exact",
    "bed_target_region_mae_improved_vs_original",
    "whiteboard_target_region_mae_improved_vs_original",
    "bed_target_region_mae_beats_zero_contact",
    "whiteboard_target_region_mae_beats_zero_contact",
    "bed_target_region_f1_improved_and_active",
    "whiteboard_target_region_f1_improved_and_active",
    "chair_replay_target_region_mae_retained",
    "chair_bed_dominate_pelvis",
    "all_targets_dominate_any_joint",
    "whiteboard_any_joint_dominance",
    "whiteboard_right_wrist_dominance",
    "sit_chair_remains_primary",
    "sit_bed_candidate_visible_and_bounded",
    "per_draw_target_dominance_rates_pass",
    "sit_per_draw_chair_primary_rate_pass",
    "sit_per_draw_bed_candidate_rate_pass",
    "v5_prompts_exact_and_object_agnostic",
}
ALLOWED_REPRESENTATIONS = (
    "affordance",
    "normalized_contact",
    "distance",
)
PROMPT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def load_json(path: Path) -> Dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(str(path) + " must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rollout_sha256_json(value: object) -> str:
    """Match the v5r4 rollout-cache canonical JSON hashing contract."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rollout_sha256_array(value: np.ndarray) -> str:
    """Match fewshot_cdm_rollout_cache.sha256_array exactly."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(
            list(array.shape),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def rollout_draw_seeds(
    *, base_seed: int, partition: str, sample_id: str, draw_index: int
) -> Tuple[int, int]:
    """Match evaluate_fewshot_cdm.stable_rollout_seeds exactly."""

    _require_exact_int(base_seed, "rollout base_seed", minimum=0)
    _require_exact_int(draw_index, "rollout draw_index", minimum=0)
    if partition not in {"train_audit", "development"}:
        raise ValueError("unsupported rollout seed partition")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("rollout sample_id is empty")
    payload = (
        str(base_seed)
        + "|"
        + partition
        + "|"
        + sample_id
        + "|"
        + str(draw_index)
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


def _validate_dataset_snapshot(
    *,
    snapshot: object,
    partition: str,
    expected_sample_ids: Sequence[str],
    split_sha256: str,
    stats_sha256: str,
) -> Dict[str, Mapping[str, object]]:
    if not isinstance(snapshot, Mapping):
        raise TypeError(partition + " dataset snapshot is not a mapping")
    if snapshot.get("schema") != "history_affordance_v1_rollout_dataset_snapshot_v1":
        raise ValueError(partition + " dataset snapshot schema mismatch")
    if snapshot.get("partition") != partition:
        raise ValueError(partition + " dataset snapshot partition mismatch")
    if snapshot.get("split_sha256") != split_sha256:
        raise ValueError(partition + " dataset snapshot split hash mismatch")
    if snapshot.get("stats_file_sha256") != stats_sha256:
        raise ValueError(partition + " dataset snapshot stats hash mismatch")
    fingerprints = snapshot.get("sample_fingerprints", {})
    if not isinstance(fingerprints, Mapping) or list(fingerprints) != list(
        expected_sample_ids
    ):
        raise ValueError(partition + " dataset snapshot sample IDs/order mismatch")
    required_fingerprint_fields = {
        "scene_id",
        "target",
        "target_instance_id",
        "text",
        "gt_sha256",
        "xyz_sha256",
        "feat_sha256",
        "instance_ids_sha256",
        "source_indices_sha256",
    }
    for sample_id, fingerprint in fingerprints.items():
        if not isinstance(fingerprint, Mapping) or set(fingerprint) != (
            required_fingerprint_fields
        ):
            raise ValueError(partition + " dataset fingerprint field mismatch")
        if not all(
            isinstance(fingerprint[name], str)
            and bool(str(fingerprint[name]).strip())
            for name in ("scene_id", "target", "text")
        ):
            raise TypeError(partition + " dataset fingerprint text field is invalid")
        _require_exact_int(
            fingerprint["target_instance_id"],
            partition + " dataset target_instance_id",
            minimum=1,
        )
        for name in required_fingerprint_fields:
            if name.endswith("_sha256"):
                _require_sha256(
                    fingerprint[name], partition + " dataset fingerprint " + name
                )
    payload = dict(snapshot)
    recorded_sha256 = payload.pop("sha256", None)
    if recorded_sha256 != rollout_sha256_json(payload):
        raise ValueError(partition + " dataset snapshot SHA-256 mismatch")
    return {str(name): value for name, value in fingerprints.items()}


def _load_current_dataset_contract(
    *,
    dataset_root: Path,
    split: Mapping[str, object],
    sample_ids: Sequence[str],
) -> Tuple[Dict[str, Mapping[str, object]], Dict[str, np.ndarray]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    split_meta = {
        str(row.get("sample_id")): row
        for row in split.get("samples", [])
        if isinstance(row, Mapping)
    }
    entries: Dict[str, Mapping[str, object]] = {}
    scene_indexes: Dict[str, Mapping[str, object]] = {}
    for sample_id in sample_ids:
        meta = split_meta.get(sample_id)
        if meta is None:
            raise KeyError("split metadata missing current sample: " + sample_id)
        scene_id = str(meta.get("scene_id"))
        if scene_id not in scene_indexes:
            index_file = dataset_root / ("index_" + scene_id + ".json")
            scene_indexes[scene_id] = load_json(index_file)
        for entry in scene_indexes[scene_id].get("samples", []):
            if str(entry.get("sample_id")) == sample_id:
                entries[sample_id] = entry
                break
    missing = sorted(set(sample_ids).difference(entries))
    if missing:
        raise KeyError("current dataset index misses samples: " + str(missing))

    scenes: Dict[str, Dict[str, np.ndarray]] = {}
    result: Dict[str, Mapping[str, object]] = {}
    sample_instance_ids: Dict[str, np.ndarray] = {}
    for sample_id in sample_ids:
        entry = entries[sample_id]
        meta = split_meta[sample_id]
        scene_id = str(entry.get("scene_id"))
        if scene_id != str(meta.get("scene_id")):
            raise ValueError("current dataset scene metadata mismatch: " + sample_id)
        if scene_id not in scenes:
            index = scene_indexes[scene_id]
            if str(index.get("scene_id")) != scene_id:
                raise ValueError("current dataset index scene ID mismatch")
            scene_dir = dataset_root / str(index["scene_adm_input"])
            with np.load(scene_dir / "points.npz", allow_pickle=False) as source:
                points = source["points"].astype(np.float32)
            with np.load(scene_dir / "sidecar.npz", allow_pickle=False) as source:
                instance_ids = source["instance_ids"].astype(np.int64)
                source_indices = source["source_indices"].astype(np.int64)
            if points.shape != (NUM_POINTS, 6):
                raise ValueError("current dataset scene point shape mismatch")
            if instance_ids.shape != (NUM_POINTS,) or source_indices.shape != (
                NUM_POINTS,
            ):
                raise ValueError("current dataset sidecar shape mismatch")
            if not np.isfinite(points).all() or len(
                np.unique(source_indices)
            ) != NUM_POINTS:
                raise ValueError("current dataset scene values/order are invalid")
            scenes[scene_id] = {
                "points": points,
                "instance_ids": instance_ids,
                "source_indices": source_indices,
            }
        scene = scenes[scene_id]
        manifest_file = dataset_root / str(entry["sample_manifest"])
        gt_file = manifest_file.resolve().parent / "affordance_gt" / "full_affordance_gt.npz"
        with np.load(gt_file, allow_pickle=False) as source:
            gt = source["affordance"].astype(np.float32)
            gt_instance_ids = source["instance_ids"].astype(np.int64)
            gt_source_indices = source["source_indices"].astype(np.int64)
        if not np.array_equal(gt_instance_ids, scene["instance_ids"]):
            raise ValueError("current GT/scene instance order mismatch: " + sample_id)
        if not np.array_equal(gt_source_indices, scene["source_indices"]):
            raise ValueError("current GT/scene point order mismatch: " + sample_id)
        target = str(meta.get("target"))
        if target not in PROMPT_BY_TARGET:
            raise ValueError("current dataset target is unsupported: " + target)
        target_instance_id = int(entry["target_instance_id"])
        if target_instance_id != {"chair": 1, "bed": 2, "whiteboard": 3}[
            target
        ]:
            raise ValueError("current dataset target instance mismatch: " + sample_id)
        if gt.shape != (NUM_POINTS, NUM_CHANNELS) or not np.isfinite(gt).all():
            raise ValueError("current dataset GT is invalid: " + sample_id)
        points = scene["points"]
        result[sample_id] = {
            "scene_id": scene_id,
            "target": target,
            "target_instance_id": target_instance_id,
            "text": PROMPT_BY_TARGET[target],
            "gt_sha256": rollout_sha256_array(gt),
            "xyz_sha256": rollout_sha256_array(
                points[:, :3].astype(np.float32)
            ),
            "feat_sha256": rollout_sha256_array(
                (points[:, 3:6] / 255.0).astype(np.float32)
            ),
            "instance_ids_sha256": rollout_sha256_array(
                scene["instance_ids"]
            ),
            "source_indices_sha256": rollout_sha256_array(
                scene["source_indices"]
            ),
        }
        sample_instance_ids[sample_id] = np.ascontiguousarray(
            scene["instance_ids"].copy()
        )
    return result, sample_instance_ids


def _current_dataset_fingerprints(
    *,
    dataset_root: Path,
    split: Mapping[str, object],
    sample_ids: Sequence[str],
) -> Dict[str, Mapping[str, object]]:
    fingerprints, _ = _load_current_dataset_contract(
        dataset_root=dataset_root,
        split=split,
        sample_ids=sample_ids,
    )
    return fingerprints


def _resolved_file(path: Path, label: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(label + ": " + str(result))
    return result


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: Optional[int] = None,
    expected: Optional[int] = None,
) -> int:
    """Reject bools, floats, and numeric strings at JSON boundaries."""

    if type(value) is not int:
        raise TypeError(label + " must be a JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(label + " is below its minimum")
    if expected is not None and value != expected:
        raise ValueError(label + " differs from the required value")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(label + " must be a lowercase hexadecimal SHA-256")
    return value


def _require_json_number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(label + " must be a finite JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(label + " must be finite")
    if positive and result <= 0.0:
        raise ValueError(label + " must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(label + " is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(label + " is above its maximum")
    return result


def _require_all_boolean_checks(summary: Mapping[str, object], label: str) -> None:
    checks = summary.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError(label + " has no non-empty checks mapping")
    non_boolean = sorted(
        str(name) for name, value in checks.items() if not isinstance(value, bool)
    )
    failed = sorted(str(name) for name, value in checks.items() if value is not True)
    if non_boolean:
        raise TypeError(label + " has non-boolean checks: " + str(non_boolean))
    if failed:
        raise ValueError(label + " failed checks: " + str(failed))


def _validate_rollout_prediction_bundle(
    *,
    path: Path,
    expected_sample_ids: Sequence[str],
    expected_k_draws: int,
    expected_base_seed: int,
    seed_partition: str,
    label: str,
    expected_fingerprints: Optional[Mapping[str, Mapping[str, object]]] = None,
    return_arrays: bool = False,
) -> Dict[str, object]:
    required = (
        "sample_ids",
        "gt",
        "original",
        "fewshot",
        "original_draws",
        "fewshot_draws",
        "initial_noise_seeds",
        "reverse_noise_seeds",
    )
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != set(required):
            raise KeyError(label + " prediction bundle key set mismatch")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(label + " prediction bundle misses " + str(missing))
        arrays = {name: source[name] for name in required}
    sample_ids = arrays["sample_ids"].astype(str).tolist()
    if arrays["sample_ids"].dtype.kind not in {"U", "S"}:
        raise TypeError(label + " prediction sample IDs must be strings")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(label + " prediction bundle has duplicate sample IDs")
    if sample_ids != list(expected_sample_ids):
        raise ValueError(label + " prediction sample IDs/order mismatch")
    sample_count = len(sample_ids)
    expected_map_shape = (sample_count, NUM_POINTS, NUM_CHANNELS)
    for name in ("gt", "original", "fewshot"):
        value = arrays[name]
        if value.shape != expected_map_shape:
            raise ValueError(label + " " + name + " shape mismatch")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(label + " " + name + " must be float32")
        if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(label + " " + name + " lies outside [0,1]")
    expected_draw_shape = (
        sample_count,
        expected_k_draws,
        NUM_POINTS,
        NUM_CHANNELS,
    )
    for name in ("original_draws", "fewshot_draws"):
        value = arrays[name]
        if value.shape != expected_draw_shape:
            raise ValueError(label + " " + name + " shape mismatch")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(label + " " + name + " must be float32")
        if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(label + " " + name + " lies outside [0,1]")
    expected_seed_shape = (sample_count, expected_k_draws)
    for name in ("initial_noise_seeds", "reverse_noise_seeds"):
        value = arrays[name]
        if value.shape != expected_seed_shape:
            raise ValueError(label + " " + name + " shape mismatch")
        if value.dtype != np.dtype(np.int64):
            raise TypeError(label + " " + name + " must be int64")
        if np.any(value < 0):
            raise ValueError(label + " " + name + " contains a negative seed")
    expected_seed_pairs = np.asarray(
        [
            [
                rollout_draw_seeds(
                    base_seed=expected_base_seed,
                    partition=seed_partition,
                    sample_id=sample_id,
                    draw_index=draw_index,
                )
                for draw_index in range(expected_k_draws)
            ]
            for sample_id in sample_ids
        ],
        dtype=np.int64,
    )
    if not np.array_equal(
        arrays["initial_noise_seeds"], expected_seed_pairs[:, :, 0]
    ) or not np.array_equal(
        arrays["reverse_noise_seeds"], expected_seed_pairs[:, :, 1]
    ):
        raise ValueError(label + " seed table differs from canonical derivation")
    for mean_name, draw_name in (
        ("original", "original_draws"),
        ("fewshot", "fewshot_draws"),
    ):
        expected = arrays[draw_name].mean(axis=1).astype(np.float32)
        actual = arrays[mean_name].astype(np.float32)
        if not np.array_equal(actual, expected):
            raise ValueError(
                label + " " + mean_name + " is not exact mean of draws"
            )
    if expected_fingerprints is not None:
        for index, sample_id in enumerate(sample_ids):
            recorded_gt_sha256 = expected_fingerprints[sample_id].get(
                "gt_sha256"
            )
            if rollout_sha256_array(arrays["gt"][index]) != recorded_gt_sha256:
                raise ValueError(label + " prediction GT/dataset hash mismatch")
    result: Dict[str, object] = {
        "status": "PASS",
        "sample_count": sample_count,
        "k_draws": expected_k_draws,
        "sample_ids_sha256": sha256_array(arrays["sample_ids"]),
        "array_sha256": {
            name: sha256_array(value) for name, value in arrays.items()
        },
    }
    if return_arrays:
        result["_arrays"] = arrays
    return result


def _binary_metrics_numpy(
    prediction: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("independent binary metric inputs are invalid")
    pred_active = prediction >= 0.7
    target_active = target >= 0.7
    true_positive = int(np.logical_and(pred_active, target_active).sum())
    false_positive = int(np.logical_and(pred_active, ~target_active).sum())
    false_negative = int(np.logical_and(~pred_active, target_active).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    foreground_mae = (
        float(np.abs(prediction[target_active] - target[target_active]).mean())
        if target_active.any()
        else 0.0
    )
    flat_prediction = prediction.reshape(-1).astype(np.float64)
    flat_target = target.reshape(-1).astype(np.float64)
    if float(flat_prediction.std()) == 0.0 or float(flat_target.std()) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flat_prediction, flat_target)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    return {
        "mae": float(np.abs(prediction - target).mean()),
        "foreground_mae": foreground_mae,
        "f1_at_0_7": float(f1),
        "correlation": correlation,
    }


def _top_fraction_mean_numpy(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    count = max(1, int(np.ceil(flat.size * 0.1)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def _instance_scores_numpy(
    affordance: np.ndarray, instance_ids: np.ndarray, channel: str
) -> Dict[str, float]:
    affordance = np.asarray(affordance, dtype=np.float32)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    if affordance.shape != (NUM_POINTS, NUM_CHANNELS):
        raise ValueError("independent instance-score affordance shape mismatch")
    if instance_ids.shape != (NUM_POINTS,):
        raise ValueError("independent instance-score ID shape mismatch")
    if channel == "pelvis":
        scalar = affordance[:, 0]
    elif channel == "any_joint":
        scalar = affordance.max(axis=1)
    elif channel == "right_wrist":
        scalar = affordance[:, 5]
    else:
        raise ValueError("unsupported independent instance-score channel")
    return {
        target: _top_fraction_mean_numpy(
            scalar[instance_ids == target_instance_id]
        )
        for target, target_instance_id in TARGET_INSTANCE_IDS.items()
    }


def _assert_tree_close(actual: object, expected: object, label: str) -> None:
    """Compare a recorded JSON metric tree with an independent NumPy result."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(label + " mapping keys differ from recomputation")
        for name, value in expected.items():
            _assert_tree_close(actual[name], value, label + "." + str(name))
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + " list shape differs from recomputation")
        for index, value in enumerate(expected):
            _assert_tree_close(actual[index], value, label + "[" + str(index) + "]")
        return
    if isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(label + " boolean differs from recomputation")
        return
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError(label + " integer differs from recomputation")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise TypeError(label + " is not a JSON number")
        if not np.isfinite(float(actual)) or not np.isclose(
            float(actual), expected, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(label + " number differs from recomputation")
        return
    if actual != expected:
        raise ValueError(label + " differs from recomputation")


def _aggregate_metric_rows(
    rows: Sequence[Mapping[str, object]], metric_field: str
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for target in ("chair", "bed", "whiteboard", "novel"):
        selected = [
            row
            for row in rows
            if row["target"] == target
            or (target == "novel" and row["target"] in {"bed", "whiteboard"})
        ]
        if not selected:
            raise ValueError("independent quality target partition is empty")
        result[target] = {}
        for model_name in ("original", "fewshot", "zero_contact"):
            result[target][model_name] = {
                metric: float(
                    np.mean(
                        [
                            row[metric_field][model_name][metric]
                            for row in selected
                        ]
                    )
                )
                for metric in (
                    "mae",
                    "foreground_mae",
                    "f1_at_0_7",
                    "correlation",
                )
            }
    return result


def _aggregate_per_draw_metric_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for target in ("chair", "bed", "whiteboard", "novel"):
        selected = [
            row
            for row in rows
            if row["target"] == target
            or (target == "novel" and row["target"] in {"bed", "whiteboard"})
        ]
        if not selected:
            raise ValueError("independent per-draw target partition is empty")
        result[target] = {}
        for model_name in ("original", "fewshot", "zero_contact"):
            result[target][model_name] = {
                metric: float(
                    np.mean(
                        [
                            draw[metric]
                            for row in selected
                            for draw in row["per_draw_target_region_metrics"][
                                model_name
                            ]
                        ]
                    )
                )
                for metric in (
                    "mae",
                    "foreground_mae",
                    "f1_at_0_7",
                    "correlation",
                )
            }
    return result


def _recompute_rollout_rows(
    *,
    arrays: Mapping[str, np.ndarray],
    fingerprints: Mapping[str, Mapping[str, object]],
    instance_ids_by_sample: Mapping[str, np.ndarray],
) -> Sequence[Dict[str, object]]:
    sample_ids = arrays["sample_ids"].astype(str).tolist()
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        fingerprint = fingerprints[sample_id]
        target = str(fingerprint["target"])
        if target not in TARGET_INSTANCE_IDS:
            raise ValueError("independent quality target is unsupported")
        target_instance_id = _require_exact_int(
            fingerprint["target_instance_id"],
            "independent quality target_instance_id",
            expected=TARGET_INSTANCE_IDS[target],
        )
        instance_ids = np.asarray(
            instance_ids_by_sample[sample_id], dtype=np.int64
        )
        if instance_ids.shape != (NUM_POINTS,):
            raise ValueError("independent quality instance IDs have wrong shape")
        target_mask = instance_ids == target_instance_id
        if not np.any(target_mask):
            raise ValueError("independent quality target instance mask is empty")
        gt = arrays["gt"][sample_index]
        original = arrays["original"][sample_index]
        fewshot = arrays["fewshot"][sample_index]
        original_draws = arrays["original_draws"][sample_index]
        fewshot_draws = arrays["fewshot_draws"][sample_index]
        zero_contact = np.zeros_like(gt)

        original_scores = {
            channel: _instance_scores_numpy(original, instance_ids, channel)
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        fewshot_scores = {
            channel: _instance_scores_numpy(fewshot, instance_ids, channel)
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        dominance_margin = {
            channel: float(
                fewshot_scores[channel][target]
                - max(
                    fewshot_scores[channel][candidate]
                    for candidate in TARGET_INSTANCE_IDS
                    if candidate != target
                )
            )
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        dominance = {
            channel: bool(value >= 0.01)
            for channel, value in dominance_margin.items()
        }
        sit_chair_primary = bool(
            fewshot_scores["any_joint"]["chair"]
            >= fewshot_scores["any_joint"]["bed"] + 0.15
            and fewshot_scores["pelvis"]["chair"]
            >= fewshot_scores["pelvis"]["bed"] + 0.15
        )
        sit_bed_candidate_visible = bool(
            0.12 <= fewshot_scores["any_joint"]["bed"] <= 0.40
            and 0.06 <= fewshot_scores["pelvis"]["bed"] <= 0.25
        )

        per_draw_metrics = {
            "original": [],
            "fewshot": [],
            "zero_contact": [],
        }
        per_draw_dominance = []
        per_draw_dominance_margin = []
        for original_draw, fewshot_draw in zip(original_draws, fewshot_draws):
            per_draw_metrics["original"].append(
                _binary_metrics_numpy(original_draw[target_mask], gt[target_mask])
            )
            per_draw_metrics["fewshot"].append(
                _binary_metrics_numpy(fewshot_draw[target_mask], gt[target_mask])
            )
            per_draw_metrics["zero_contact"].append(
                _binary_metrics_numpy(zero_contact[target_mask], gt[target_mask])
            )
            draw_scores = {
                channel: _instance_scores_numpy(
                    fewshot_draw, instance_ids, channel
                )
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            draw_margins = {
                channel: float(
                    draw_scores[channel][target]
                    - max(
                        draw_scores[channel][candidate]
                        for candidate in TARGET_INSTANCE_IDS
                        if candidate != target
                    )
                )
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            draw_flags = {
                channel: bool(value >= 0.01)
                for channel, value in draw_margins.items()
            }
            draw_flags["sit_chair_primary"] = bool(
                draw_scores["any_joint"]["chair"]
                >= draw_scores["any_joint"]["bed"] + 0.15
                and draw_scores["pelvis"]["chair"]
                >= draw_scores["pelvis"]["bed"] + 0.15
            )
            draw_flags["sit_bed_candidate_visible"] = bool(
                0.12 <= draw_scores["any_joint"]["bed"] <= 0.40
                and 0.06 <= draw_scores["pelvis"]["bed"] <= 0.25
            )
            per_draw_dominance_margin.append(draw_margins)
            per_draw_dominance.append(draw_flags)

        rows.append(
            {
                "sample_id": sample_id,
                "scene_id": str(fingerprint["scene_id"]),
                "target": target,
                "text": str(fingerprint["text"]),
                "metrics": {
                    "original": _binary_metrics_numpy(original, gt),
                    "fewshot": _binary_metrics_numpy(fewshot, gt),
                    "zero_contact": _binary_metrics_numpy(zero_contact, gt),
                },
                "target_region_metrics": {
                    "original": _binary_metrics_numpy(
                        original[target_mask], gt[target_mask]
                    ),
                    "fewshot": _binary_metrics_numpy(
                        fewshot[target_mask], gt[target_mask]
                    ),
                    "zero_contact": _binary_metrics_numpy(
                        zero_contact[target_mask], gt[target_mask]
                    ),
                },
                "per_draw_target_region_metrics": per_draw_metrics,
                "target_top10": {
                    "original_pelvis": original_scores["pelvis"][target],
                    "fewshot_pelvis": fewshot_scores["pelvis"][target],
                    "original_any_joint": original_scores["any_joint"][target],
                    "fewshot_any_joint": fewshot_scores["any_joint"][target],
                    "original_right_wrist": original_scores["right_wrist"][target],
                    "fewshot_right_wrist": fewshot_scores["right_wrist"][target],
                },
                "fewshot_candidate_scores": {
                    channel: {
                        candidate: fewshot_scores[channel][candidate]
                        for candidate in TARGET_INSTANCE_IDS
                    }
                    for channel in ("pelvis", "any_joint", "right_wrist")
                },
                "target_dominates_candidates": dominance,
                "target_dominance_margin": dominance_margin,
                "per_draw_target_dominance": per_draw_dominance,
                "per_draw_target_dominance_margin": per_draw_dominance_margin,
                "sit_chair_primary": sit_chair_primary,
                "sit_bed_candidate_visible": sit_bed_candidate_visible,
            }
        )
    return rows


def _dominance_rates(
    rows: Sequence[Mapping[str, object]], *, include_ensemble: bool
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for target in TARGET_INSTANCE_IDS:
        selected = [row for row in rows if row["target"] == target]
        if not selected:
            raise ValueError("independent dominance target partition is empty")
        per_draw = {
            channel: float(
                np.mean(
                    [
                        draw[channel]
                        for row in selected
                        for draw in row["per_draw_target_dominance"]
                    ]
                )
            )
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        if include_ensemble:
            result[target] = {
                channel: float(
                    np.mean(
                        [
                            row["target_dominates_candidates"][channel]
                            for row in selected
                        ]
                    )
                )
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            result[target]["per_draw"] = per_draw
        else:
            result[target] = per_draw
    return result


def _sit_rates(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    chair_rows = [row for row in rows if row["target"] == "chair"]
    if not chair_rows:
        raise ValueError("independent Sit partition has no Chair samples")
    return {
        "ensemble_chair_primary_rate": float(
            np.mean([row["sit_chair_primary"] for row in chair_rows])
        ),
        "ensemble_bed_candidate_rate": float(
            np.mean([row["sit_bed_candidate_visible"] for row in chair_rows])
        ),
        "per_draw_chair_primary_rate": float(
            np.mean(
                [
                    draw["sit_chair_primary"]
                    for row in chair_rows
                    for draw in row["per_draw_target_dominance"]
                ]
            )
        ),
        "per_draw_bed_candidate_rate": float(
            np.mean(
                [
                    draw["sit_bed_candidate_visible"]
                    for row in chair_rows
                    for draw in row["per_draw_target_dominance"]
                ]
            )
        ),
    }


def _audit_row_projection(row: Mapping[str, object]) -> Dict[str, object]:
    fields = (
        "sample_id",
        "scene_id",
        "target",
        "text",
        "target_region_metrics",
        "per_draw_target_region_metrics",
        "fewshot_candidate_scores",
        "target_dominates_candidates",
        "target_dominance_margin",
        "per_draw_target_dominance",
        "per_draw_target_dominance_margin",
        "sit_chair_primary",
        "sit_bed_candidate_visible",
    )
    return {name: row[name] for name in fields}


def _development_row_projection(row: Mapping[str, object]) -> Dict[str, object]:
    fields = (
        "sample_id",
        "scene_id",
        "target",
        "text",
        "metrics",
        "target_region_metrics",
        "per_draw_target_region_metrics",
        "target_top10",
        "fewshot_candidate_scores",
        "target_dominates_candidates",
        "target_dominance_margin",
        "per_draw_target_dominance",
        "per_draw_target_dominance_margin",
        "sit_chair_primary",
        "sit_bed_candidate_visible",
    )
    return {name: row[name] for name in fields}


def _validate_independent_audit_quality(
    *,
    audit: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    fingerprints: Mapping[str, Mapping[str, object]],
    instance_ids_by_sample: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    _assert_tree_close(
        audit.get("quality_thresholds"),
        AUDIT_QUALITY_THRESHOLDS,
        "train audit quality_thresholds",
    )
    rows = _recompute_rollout_rows(
        arrays=arrays,
        fingerprints=fingerprints,
        instance_ids_by_sample=instance_ids_by_sample,
    )
    recorded_rows = audit.get("per_sample")
    expected_rows = [_audit_row_projection(row) for row in rows]
    _assert_tree_close(recorded_rows, expected_rows, "train audit per_sample")

    aggregate_target_region = _aggregate_metric_rows(
        rows, "target_region_metrics"
    )
    per_draw_aggregate = _aggregate_per_draw_metric_rows(rows)
    dominance_rate = _dominance_rates(rows, include_ensemble=True)
    sit_rates = _sit_rates(rows)
    chair_original_mae = float(
        per_draw_aggregate["chair"]["original"]["mae"]
    )
    chair_degradation = (
        float(per_draw_aggregate["chair"]["fewshot"]["mae"])
        - chair_original_mae
    ) / max(chair_original_mae, 1e-12)
    expected_sit_contract = {
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
    }
    _assert_tree_close(
        audit.get("aggregate_target_region"),
        aggregate_target_region,
        "train audit aggregate_target_region",
    )
    _assert_tree_close(
        audit.get("per_draw_aggregate_target_region"),
        per_draw_aggregate,
        "train audit per_draw_aggregate_target_region",
    )
    _assert_tree_close(
        audit.get("dominance_rate"),
        dominance_rate,
        "train audit dominance_rate",
    )
    _assert_tree_close(
        audit.get("sit_contract"),
        expected_sit_contract,
        "train audit sit_contract",
    )
    _assert_tree_close(
        audit.get("chair_mae_relative_degradation"),
        float(chair_degradation),
        "train audit chair_mae_relative_degradation",
    )
    _assert_tree_close(
        audit.get("dominance_margin_required"),
        0.01,
        "train audit dominance_margin_required",
    )

    checks = {
        "chair_rollout_mae_retained": chair_degradation <= 0.05,
        "bed_rollout_mae_improved": (
            per_draw_aggregate["bed"]["fewshot"]["mae"]
            < per_draw_aggregate["bed"]["original"]["mae"]
        ),
        "whiteboard_rollout_mae_improved": (
            per_draw_aggregate["whiteboard"]["fewshot"]["mae"]
            < per_draw_aggregate["whiteboard"]["original"]["mae"]
        ),
        "bed_rollout_mae_beats_zero": (
            per_draw_aggregate["bed"]["fewshot"]["mae"]
            < per_draw_aggregate["bed"]["zero_contact"]["mae"]
        ),
        "whiteboard_rollout_mae_beats_zero": (
            per_draw_aggregate["whiteboard"]["fewshot"]["mae"]
            < per_draw_aggregate["whiteboard"]["zero_contact"]["mae"]
        ),
        "bed_rollout_f1_improved": (
            per_draw_aggregate["bed"]["fewshot"]["f1_at_0_7"]
            > per_draw_aggregate["bed"]["original"]["f1_at_0_7"]
        ),
        "whiteboard_rollout_f1_improved": (
            per_draw_aggregate["whiteboard"]["fewshot"]["f1_at_0_7"]
            > per_draw_aggregate["whiteboard"]["original"]["f1_at_0_7"]
        ),
        "bed_rollout_f1_active": (
            per_draw_aggregate["bed"]["fewshot"]["f1_at_0_7"] >= 0.30
        ),
        "whiteboard_rollout_f1_active": (
            per_draw_aggregate["whiteboard"]["fewshot"]["f1_at_0_7"]
            >= 0.30
        ),
        "chair_rollout_any_joint_dominance": (
            dominance_rate["chair"]["any_joint"] >= 0.90
        ),
        "chair_rollout_pelvis_dominance": (
            dominance_rate["chair"]["pelvis"] >= 0.90
        ),
        "chair_per_draw_dominance_rate": (
            dominance_rate["chair"]["per_draw"]["pelvis"] >= 0.80
            and dominance_rate["chair"]["per_draw"]["any_joint"] >= 0.80
        ),
        "sit_chair_remains_primary": (
            sit_rates["ensemble_chair_primary_rate"] >= 0.90
        ),
        "sit_bed_candidate_visible_and_bounded": (
            sit_rates["ensemble_bed_candidate_rate"] >= 0.80
        ),
        "sit_per_draw_contract_rate": (
            sit_rates["per_draw_chair_primary_rate"] >= 0.90
            and sit_rates["per_draw_bed_candidate_rate"] >= 0.80
        ),
        "bed_rollout_any_joint_dominance": (
            dominance_rate["bed"]["any_joint"] >= 1.0
        ),
        "bed_rollout_pelvis_dominance": (
            dominance_rate["bed"]["pelvis"] >= 1.0
        ),
        "bed_per_draw_dominance_rate": (
            dominance_rate["bed"]["per_draw"]["pelvis"] >= 0.80
            and dominance_rate["bed"]["per_draw"]["any_joint"] >= 0.80
        ),
        "whiteboard_rollout_any_joint_dominance": (
            dominance_rate["whiteboard"]["any_joint"] >= 1.0
        ),
        "whiteboard_rollout_right_wrist_dominance": (
            dominance_rate["whiteboard"]["right_wrist"] >= 1.0
        ),
        "whiteboard_per_draw_any_joint_dominance_rate": (
            dominance_rate["whiteboard"]["per_draw"]["any_joint"] >= 0.80
        ),
        "whiteboard_per_draw_right_wrist_dominance_rate": (
            dominance_rate["whiteboard"]["per_draw"]["right_wrist"] >= 0.80
        ),
        "v5_prompts_exact_and_object_agnostic": all(
            row["text"] == PROMPT_BY_TARGET[row["target"]] for row in rows
        ),
    }
    recorded_checks = audit.get("checks", {})
    for name, value in checks.items():
        if recorded_checks.get(name) is not value:
            raise ValueError("train audit check differs from recomputation: " + name)
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise ValueError("independent train audit quality failed: " + str(failed))
    return {
        "status": "PASS",
        "checks": checks,
        "chair_mae_relative_degradation": float(chair_degradation),
    }


def _validate_independent_development_quality(
    *,
    development: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    fingerprints: Mapping[str, Mapping[str, object]],
    instance_ids_by_sample: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    _assert_tree_close(
        development.get("quality_thresholds"),
        DEVELOPMENT_QUALITY_THRESHOLDS,
        "development quality_thresholds",
    )
    rows = _recompute_rollout_rows(
        arrays=arrays,
        fingerprints=fingerprints,
        instance_ids_by_sample=instance_ids_by_sample,
    )
    expected_rows = [_development_row_projection(row) for row in rows]
    _assert_tree_close(
        development.get("per_sample"),
        expected_rows,
        "development per_sample",
    )
    aggregate = _aggregate_metric_rows(rows, "metrics")
    aggregate_target_region = _aggregate_metric_rows(
        rows, "target_region_metrics"
    )
    per_draw_aggregate = _aggregate_per_draw_metric_rows(rows)
    per_draw_dominance_rate = _dominance_rates(
        rows, include_ensemble=False
    )
    sit_rates = _sit_rates(rows)
    chair_original_mae = float(
        per_draw_aggregate["chair"]["original"]["mae"]
    )
    chair_degradation = (
        float(per_draw_aggregate["chair"]["fewshot"]["mae"])
        - chair_original_mae
    ) / max(chair_original_mae, 1e-12)
    sit_per_draw_contract = {
        "chair_primary_rate": sit_rates["per_draw_chair_primary_rate"],
        "bed_candidate_visible_rate": sit_rates[
            "per_draw_bed_candidate_rate"
        ],
        "minimum_rate": 0.80,
    }
    for name, expected in (
        ("aggregate", aggregate),
        ("aggregate_target_region", aggregate_target_region),
        ("per_draw_aggregate_target_region", per_draw_aggregate),
        ("per_draw_dominance_rate", per_draw_dominance_rate),
        ("sit_per_draw_contract", sit_per_draw_contract),
        ("chair_mae_relative_degradation", float(chair_degradation)),
        ("dominance_margin_required", 0.01),
    ):
        _assert_tree_close(
            development.get(name), expected, "development " + name
        )

    chair_bed_rows = [
        row for row in rows if row["target"] in {"chair", "bed"}
    ]
    whiteboard_rows = [row for row in rows if row["target"] == "whiteboard"]
    chair_rows = [row for row in rows if row["target"] == "chair"]
    checks = {
        "bed_target_region_mae_improved_vs_original": (
            per_draw_aggregate["bed"]["fewshot"]["mae"]
            < per_draw_aggregate["bed"]["original"]["mae"]
        ),
        "whiteboard_target_region_mae_improved_vs_original": (
            per_draw_aggregate["whiteboard"]["fewshot"]["mae"]
            < per_draw_aggregate["whiteboard"]["original"]["mae"]
        ),
        "bed_target_region_mae_beats_zero_contact": (
            per_draw_aggregate["bed"]["fewshot"]["mae"]
            < per_draw_aggregate["bed"]["zero_contact"]["mae"]
        ),
        "whiteboard_target_region_mae_beats_zero_contact": (
            per_draw_aggregate["whiteboard"]["fewshot"]["mae"]
            < per_draw_aggregate["whiteboard"]["zero_contact"]["mae"]
        ),
        "bed_target_region_f1_improved_and_active": (
            per_draw_aggregate["bed"]["fewshot"]["f1_at_0_7"]
            > per_draw_aggregate["bed"]["original"]["f1_at_0_7"]
            and per_draw_aggregate["bed"]["fewshot"]["f1_at_0_7"] >= 0.30
        ),
        "whiteboard_target_region_f1_improved_and_active": (
            per_draw_aggregate["whiteboard"]["fewshot"]["f1_at_0_7"]
            > per_draw_aggregate["whiteboard"]["original"]["f1_at_0_7"]
            and per_draw_aggregate["whiteboard"]["fewshot"]["f1_at_0_7"]
            >= 0.30
        ),
        "chair_replay_target_region_mae_retained": chair_degradation <= 0.05,
        "chair_bed_dominate_pelvis": all(
            row["target_dominates_candidates"]["pelvis"]
            for row in chair_bed_rows
        ),
        "all_targets_dominate_any_joint": all(
            row["target_dominates_candidates"]["any_joint"] for row in rows
        ),
        "whiteboard_any_joint_dominance": all(
            row["target_dominates_candidates"]["any_joint"]
            for row in whiteboard_rows
        ),
        "whiteboard_right_wrist_dominance": all(
            row["target_dominates_candidates"]["right_wrist"]
            for row in whiteboard_rows
        ),
        "sit_chair_remains_primary": all(
            row["sit_chair_primary"] for row in chair_rows
        ),
        "sit_bed_candidate_visible_and_bounded": all(
            row["sit_bed_candidate_visible"] for row in chair_rows
        ),
        "per_draw_target_dominance_rates_pass": (
            per_draw_dominance_rate["chair"]["pelvis"] >= 0.80
            and per_draw_dominance_rate["chair"]["any_joint"] >= 0.80
            and per_draw_dominance_rate["bed"]["pelvis"] >= 0.80
            and per_draw_dominance_rate["bed"]["any_joint"] >= 0.80
            and per_draw_dominance_rate["whiteboard"]["any_joint"] >= 0.80
            and per_draw_dominance_rate["whiteboard"]["right_wrist"] >= 0.80
        ),
        "sit_per_draw_chair_primary_rate_pass": (
            sit_rates["per_draw_chair_primary_rate"] >= 0.80
        ),
        "sit_per_draw_bed_candidate_rate_pass": (
            sit_rates["per_draw_bed_candidate_rate"] >= 0.80
        ),
        "v5_prompts_exact_and_object_agnostic": all(
            row["text"] == PROMPT_BY_TARGET[row["target"]] for row in rows
        ),
    }
    recorded_checks = development.get("checks", {})
    for name, value in checks.items():
        if recorded_checks.get(name) is not value:
            raise ValueError(
                "development check differs from recomputation: " + name
            )
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise ValueError(
            "independent development quality failed: " + str(failed)
        )
    return {
        "status": "PASS",
        "checks": checks,
        "chair_mae_relative_degradation": float(chair_degradation),
    }


def validate_v5r4_quality_chain(
    *,
    fewshot_checkpoint: Path,
    original_checkpoint: Path,
    train_summary_file: Path,
    shortlist_status_file: Path,
    rollout_selection_file: Path,
    train_rollout_audit_file: Path,
    train_rollout_predictions_file: Path,
    development_summary_file: Path,
    development_predictions_file: Path,
    split_file: Path,
    stats_file: Path,
    minimum_k_draws: int = 5,
    expected_diffusion_steps: Optional[int] = None,
    runtime_repo_root: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Validate the complete v5r4 evidence chain without loading model tensors."""

    _require_exact_int(minimum_k_draws, "minimum_k_draws", minimum=1)
    if expected_diffusion_steps is not None:
        _require_exact_int(
            expected_diffusion_steps,
            "expected_diffusion_steps",
            minimum=1,
        )
    if dataset_root is None:
        raise ValueError(
            "strict v5r4 quality validation requires the current dataset_root"
        )

    files = {
        "fewshot_checkpoint": _resolved_file(
            fewshot_checkpoint, "fewshot checkpoint"
        ),
        "original_checkpoint": _resolved_file(
            original_checkpoint, "original checkpoint"
        ),
        "train_summary": _resolved_file(train_summary_file, "train summary"),
        "shortlist_status": _resolved_file(
            shortlist_status_file, "shortlist status"
        ),
        "rollout_selection": _resolved_file(
            rollout_selection_file, "rollout selection"
        ),
        "train_rollout_audit": _resolved_file(
            train_rollout_audit_file, "train rollout audit"
        ),
        "train_rollout_predictions": _resolved_file(
            train_rollout_predictions_file, "train rollout predictions"
        ),
        "development_summary": _resolved_file(
            development_summary_file, "development summary"
        ),
        "development_predictions": _resolved_file(
            development_predictions_file, "development predictions"
        ),
        "split": _resolved_file(split_file, "split"),
        "stats": _resolved_file(stats_file, "contact stats"),
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    experiment_root = files["train_summary"].parent
    for name in (
        "fewshot_checkpoint",
        "shortlist_status",
        "rollout_selection",
    ):
        if files[name].parent != experiment_root:
            raise ValueError(name + " is outside the explicit v5r4 experiment root")
    if files["train_rollout_predictions"].parent != files[
        "train_rollout_audit"
    ].parent:
        raise ValueError("train rollout summary/predictions are not colocated")
    if files["development_predictions"].parent != files[
        "development_summary"
    ].parent:
        raise ValueError("development summary/predictions are not colocated")

    train = load_json(files["train_summary"])
    if train.get("schema") != TRAIN_SCHEMA or train.get("status") != "PASS":
        raise ValueError("training summary is not strict v5r4 PASS")
    if train.get("checkpoint_sha256") != hashes["fewshot_checkpoint"]:
        raise ValueError("training summary/checkpoint SHA-256 mismatch")
    if train.get("initialization", {}).get("sha256") != hashes[
        "original_checkpoint"
    ]:
        raise ValueError("training summary/original checkpoint SHA-256 mismatch")
    if train.get("split", {}).get("sha256") != hashes["split"]:
        raise ValueError("training summary/split SHA-256 mismatch")
    if train.get("selection_data") != "train_only":
        raise ValueError("v5r4 checkpoint was not selected on train-only data")
    if train.get("test_partition_read_during_training") is not False:
        raise ValueError("training summary says test partition was read")
    if train.get("test_sample_data_read_during_training") is not False:
        raise ValueError("training summary says test tensors were read")
    if train.get("test_ids_used_only_for_exclusion_assertion") is not True:
        raise ValueError("training summary lacks held-out ID firewall evidence")
    initialization = train.get("initialization", {})
    if initialization.get("standard_original_path") is not True:
        raise ValueError("training did not use the standard original checkpoint")
    if initialization.get("one_sample_diagnostic_checkpoint_used") is not False:
        raise ValueError("training used a diagnostic initialization")
    objective = train.get("data_objective", {})
    if not isinstance(objective, Mapping):
        raise TypeError("training data_objective is not a mapping")
    if objective.get("prompt_policy_id") != PROMPT_POLICY_ID:
        raise ValueError("training summary uses the wrong prompt policy")
    if objective.get("prompt_by_target") != PROMPT_BY_TARGET:
        raise ValueError("training summary uses non-canonical v5 prompts")
    sampling = train.get("sampling", {})
    if not isinstance(sampling, Mapping) or sampling != {
        "method": "chair3_bed1_whiteboard1_shuffled_cycles",
        "batch_size": 5,
        "chair_replay_per_step": 3,
        "bed_per_step": 1,
        "whiteboard_per_step": 1,
    }:
        raise ValueError("training summary uses the wrong balanced replay contract")
    regularization = train.get("regularization", {})
    if not isinstance(regularization, Mapping):
        raise TypeError("training regularization is not a mapping")
    lora = regularization.get("lora", {})
    if (
        regularization.get("method")
        != (
            "frozen_original_cdm_plus_zero_init_lora_and_"
            "chair_region_multinoise_teacher_v5r4"
        )
        or regularization.get("zero_init_bitwise_original") is not True
        or regularization.get("original_checkpoint_sha256")
        != hashes["original_checkpoint"]
        or regularization.get("chair_teacher_region")
        != "target_chair_instance_only"
        or regularization.get(
            "bed_candidate_region_excluded_from_chair_teacher"
        )
        is not True
        or not isinstance(lora, Mapping)
        or lora.get("base_parameters_frozen") is not True
        or lora.get("zero_initialized_output_projection") is not True
        or lora.get("export_format") != "merged_legacy_partial_state_dict"
    ):
        raise ValueError("training frozen-base/LoRA regularization contract mismatch")
    for name in (
        "chair_teacher_weight",
        "chair_high_timestep_weight",
        "chair_high_teacher_weight",
    ):
        _require_json_number(
            regularization.get(name), "training regularization " + name, positive=True
        )
    if (
        objective.get("method")
        != "rollout_aligned_sparse_semantic_multinoise_start_x_v5r4"
        or objective.get("diffusion_prediction_target") != "START_X"
        or objective.get("chair_uses_legacy_uniform_weights") is not True
        or objective.get("background_remains_supervised") is not True
        or objective.get("novel_high_timestep_extra_forward") is not True
        or objective.get("chair_high_timestep_extra_forward") is not True
        or objective.get("whiteboard_semantic_channel")
        != "right_wrist_native_index_5"
        or objective.get("active_threshold") != 0.7
        or objective.get("base_weight") != 1.0
    ):
        raise ValueError("training sparse START_X objective contract mismatch")
    for name in (
        "target_instance_additive_weight",
        "target_foreground_additive_weight",
        "semantic_weight",
        "foreground_bce_weight",
        "dice_weight",
        "ranking_weight",
        "high_timestep_weight",
    ):
        _require_json_number(
            objective.get(name), "training data_objective " + name, positive=True
        )
    if _require_json_number(
        objective.get("chair_uniform_legacy_parity_max_abs_diff"),
        "training Chair legacy parity",
        minimum=0.0,
    ) > 1e-6:
        raise ValueError("training Chair legacy parity exceeds tolerance")
    sit_policy = objective.get("sit_multicandidate", {})
    if (
        not isinstance(sit_policy, Mapping)
        or sit_policy.get("supervision_type") != "weak_semantic_prior_only"
        or sit_policy.get("motion_gt_relabelled_or_copied") is not False
        or sit_policy.get("weight") != 0.50
        or sit_policy.get("bed_any_joint_band") != [0.12, 0.40]
        or sit_policy.get("bed_pelvis_band") != [0.06, 0.25]
        or sit_policy.get("chair_primary_margin") != 0.15
    ):
        raise ValueError("training Sit multicandidate contract mismatch")
    selection = train.get("checkpoint_selection", {})
    if not isinstance(selection, Mapping):
        raise TypeError("training checkpoint_selection is not a mapping")
    if selection.get("method") != "one_step_shortlist_then_train_full_rollout":
        raise ValueError("training checkpoint selection method mismatch")
    if selection.get("sit_bed_candidate_min_rate") != 0.80:
        raise ValueError("training Sit-Bed selection threshold mismatch")
    if selection.get("gate_passed") is not True:
        raise ValueError("one-step training gate did not pass")
    selection_best = selection.get("best", {})
    if (
        not isinstance(selection_best, Mapping)
        or selection_best.get("loss_gate", {}).get("passed") is not True
        or selection_best.get("semantic_gate", {}).get("passed") is not True
    ):
        raise ValueError("training summary best one-step gates did not pass")
    if selection.get("rollout_gate_passed") is not True:
        raise ValueError("train-only rollout selection gate did not pass")
    selected = selection.get("rollout_selection", {}).get("selected", {})
    if selected.get("passed") is not True:
        raise ValueError("rollout-selected checkpoint is not marked passed")
    if selected.get("checkpoint_sha256") != hashes["fewshot_checkpoint"]:
        raise ValueError("rollout selection/checkpoint SHA-256 mismatch")

    shortlist = load_json(files["shortlist_status"])
    if shortlist.get("schema") != SHORTLIST_SCHEMA or shortlist.get("status") != "PASS":
        raise ValueError("one-step shortlist is not strict v5r4 PASS")
    if shortlist.get("partition") != "train_only":
        raise ValueError("one-step shortlist is not train-only")
    if shortlist.get("heldout_sample_tensors_read") is not False:
        raise ValueError("one-step shortlist read held-out tensors")
    if shortlist.get("prompt_policy_id") != PROMPT_POLICY_ID:
        raise ValueError("one-step shortlist uses the wrong prompt policy")
    _require_exact_int(
        shortlist.get("candidate_count"),
        "one-step shortlist candidate_count",
        minimum=1,
    )
    shortlist_selection = shortlist.get("best_one_step", {}).get("selection", {})
    if shortlist_selection.get("loss_gate", {}).get("passed") is not True:
        raise ValueError("one-step shortlist loss gate did not pass")
    if shortlist_selection.get("semantic_gate", {}).get("passed") is not True:
        raise ValueError("one-step shortlist semantic gate did not pass")
    if "checks" in shortlist:
        _require_all_boolean_checks(shortlist, "one-step shortlist")

    rollout_selection = load_json(files["rollout_selection"])
    if (
        rollout_selection.get("schema") != ROLLOUT_SELECTION_SCHEMA
        or rollout_selection.get("status") != "PASS"
    ):
        raise ValueError("train rollout selection is not strict PASS")
    if rollout_selection.get("heldout_sample_tensors_read") is not False:
        raise ValueError("train rollout selection read held-out tensors")
    if rollout_selection.get("partition") != "train_complete":
        raise ValueError("train rollout selection used the wrong partition")
    if rollout_selection.get("complete_train_partition") is not True:
        raise ValueError("train rollout selection is not complete")
    rollout_selection_k = _require_exact_int(
        rollout_selection.get("k_samples"),
        "train rollout selection k_samples",
        minimum=minimum_k_draws,
    )
    if rollout_selection_k != 5:
        raise ValueError("strict v5r4 evidence requires exactly five draws")
    if rollout_selection_k < minimum_k_draws:
        raise ValueError("train rollout selection has too few draws")
    if rollout_selection.get("seed_partition") != "train_audit":
        raise ValueError("train rollout selection used the wrong seed partition")
    if len(rollout_selection.get("probe_sample_ids", [])) != 25:
        raise ValueError("train rollout selection did not cover 25 train samples")
    shortlisted_candidate_count = _require_exact_int(
        rollout_selection.get("shortlisted_candidate_count"),
        "train rollout shortlisted_candidate_count",
        minimum=1,
    )
    evaluated_candidate_count = _require_exact_int(
        rollout_selection.get("candidate_count"),
        "train rollout candidate_count",
        minimum=1,
    )
    if evaluated_candidate_count > shortlisted_candidate_count:
        raise ValueError("train rollout evaluated more candidates than shortlisted")
    evaluated_candidates = rollout_selection.get("evaluated")
    if (
        not isinstance(evaluated_candidates, list)
        or len(evaluated_candidates) != evaluated_candidate_count
    ):
        raise ValueError("train rollout candidate_count/evaluated rows mismatch")
    rollout_selected = rollout_selection.get("selected", {})
    if rollout_selected.get("passed") is not True:
        raise ValueError("train rollout selected candidate did not pass")
    if rollout_selected not in evaluated_candidates:
        raise ValueError("train rollout selected candidate is absent from evaluated rows")
    selected_checks = rollout_selected.get("checks", {})
    if (
        not isinstance(selected_checks, Mapping)
        or not selected_checks
        or any(value is not True for value in selected_checks.values())
    ):
        raise ValueError("train rollout selected candidate has failed checks")
    if set(selected_checks) != REQUIRED_ROLLOUT_SELECTION_CHECKS:
        raise ValueError("train rollout selected candidate check set is incomplete")
    if rollout_selected.get("checkpoint_sha256") != hashes["fewshot_checkpoint"]:
        raise ValueError("rollout selection/checkpoint hash mismatch")
    recorded_selection = selection.get("rollout_selection", {})
    if recorded_selection.get("selection_file_sha256") != hashes[
        "rollout_selection"
    ]:
        raise ValueError("train summary/rollout selection hash mismatch")

    audit = load_json(files["train_rollout_audit"])
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != "PASS":
        raise ValueError("train rollout audit is not strict PASS")
    if audit.get("decision") != "TRAIN_ROLLOUT_VALID":
        raise ValueError("train rollout audit decision is not valid")
    if audit.get("partition") != "train":
        raise ValueError("train rollout audit used the wrong partition")
    if audit.get("test_sample_tensors_read") is not False:
        raise ValueError("train rollout audit read test tensors")
    if audit.get("checkpoint_sha256") != hashes["fewshot_checkpoint"]:
        raise ValueError("audit/checkpoint SHA-256 mismatch")
    if audit.get("original_checkpoint_sha256") != hashes["original_checkpoint"]:
        raise ValueError("audit/original checkpoint SHA-256 mismatch")
    if audit.get("train_summary_sha256") != hashes["train_summary"]:
        raise ValueError("audit/train summary SHA-256 mismatch")
    if audit.get("split_sha256") != hashes["split"]:
        raise ValueError("audit/split SHA-256 mismatch")
    if audit.get("stats_file_sha256") != hashes["stats"]:
        raise ValueError("audit/stats SHA-256 mismatch")
    audit_predictions = audit.get("predictions", {})
    if audit_predictions.get("format") != "paired_rollout_affordance_npz_v2":
        raise ValueError("train audit predictions format is missing or stale")
    if audit_predictions.get("sha256") != hashes["train_rollout_predictions"]:
        raise ValueError("train audit/predictions SHA-256 mismatch")
    if Path(str(audit_predictions.get("file", ""))).name != files[
        "train_rollout_predictions"
    ].name:
        raise ValueError("train audit predictions filename mismatch")
    if audit.get("paired_noise") is not True:
        raise ValueError("train rollout audit is not paired-noise")
    audit_sampling = audit.get("sampling_provenance", {})
    expected_audit_sampling = {
        "base_seed": 20260815,
        "partition": "train_audit",
        "seed_derivation": "sha256(base|partition|sample_id|draw)",
        "seed_table_storage": "prediction_npz_int64_matrices",
    }
    if audit_sampling != expected_audit_sampling:
        raise ValueError("train audit sampling provenance mismatch")
    rollout_protocol_hashes = audit.get("rollout_protocol_hashes", {})
    if not isinstance(rollout_protocol_hashes, Mapping) or set(
        rollout_protocol_hashes
    ) != set(ROLLOUT_PROTOCOL_FILES):
        raise ValueError("train rollout protocol file set mismatch")
    invalid_protocol_hashes = sorted(
        name
        for name, value in rollout_protocol_hashes.items()
        if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
    )
    if invalid_protocol_hashes:
        raise ValueError(
            "train rollout protocol contains invalid hashes: "
            + str(invalid_protocol_hashes)
        )
    if runtime_repo_root is not None:
        runtime_repo_root = Path(runtime_repo_root).expanduser().resolve()
        current_protocol_hashes = {
            name: sha256_file(runtime_repo_root / name)
            for name in ROLLOUT_PROTOCOL_FILES
        }
        if current_protocol_hashes != dict(rollout_protocol_hashes):
            raise ValueError("current rollout protocol files differ from audit")
    pairing = audit.get("pairing_protocol", {})
    for name in (
        "initial_xT_and_all_reverse_step_noise_paired",
        "caller_rng_state_restored",
        "repeatability_canary_bitwise_equal",
        "metrics_computed_per_draw",
    ):
        if pairing.get(name) is not True:
            raise ValueError("train rollout pairing contract failed: " + name)
    audit_k_samples = _require_exact_int(
        audit.get("k_samples"),
        "train rollout audit k_samples",
        minimum=minimum_k_draws,
    )
    if audit_k_samples != 5:
        raise ValueError("strict train audit requires exactly five draws")
    if audit_k_samples < minimum_k_draws:
        raise ValueError("train rollout audit has too few draws")
    _require_all_boolean_checks(audit, "train rollout audit")
    actual_audit_checks = set(audit.get("checks", {}))
    if actual_audit_checks != REQUIRED_AUDIT_CHECKS:
        raise ValueError(
            "train rollout audit check set mismatch: missing="
            + str(sorted(REQUIRED_AUDIT_CHECKS - actual_audit_checks))
            + ", unexpected="
            + str(sorted(actual_audit_checks - REQUIRED_AUDIT_CHECKS))
        )
    audit_sample_ids = [str(value) for value in audit.get("train_sample_ids", [])]
    if audit_sample_ids != sorted(audit_sample_ids) or not audit_sample_ids:
        raise ValueError("train audit sample IDs are absent or unsorted")
    split = load_json(files["split"])
    if split.get("schema") != "affordance_source_disjoint_split_v1":
        raise ValueError("unsupported v5r4 split schema")
    split_checks = split.get("checks", {})
    required_split_checks = (
        "source_components_disjoint",
        "original_motion_ids_disjoint",
        "multistart_excluded_from_cdm",
        "object_names_absent_from_new_prompts",
    )
    if not isinstance(split_checks, Mapping) or any(
        split_checks.get(name) is not True for name in required_split_checks
    ):
        raise ValueError("v5r4 split safety checks are incomplete or failed")
    cdm_split = split.get("cdm_fewshot", {})
    train_split_ids = sorted(str(value) for value in cdm_split.get("train", []))
    development_split_ids = sorted(
        str(value) for value in cdm_split.get("test", [])
    )
    if len(train_split_ids) != 25 or len(development_split_ids) != 12:
        raise ValueError(
            "v5r4 split must contain exactly 25 train and 12 development IDs"
        )
    if len(train_split_ids) != len(set(train_split_ids)) or len(
        development_split_ids
    ) != len(set(development_split_ids)):
        raise ValueError("v5r4 split contains duplicate sample IDs")
    if set(train_split_ids).intersection(development_split_ids):
        raise ValueError("v5r4 train/development sample IDs overlap")
    if split.get("split_unit") != "connected raw-source component":
        raise ValueError("v5r4 split is not raw-source-component disjoint")
    sample_split = {
        sample_id: partition
        for partition, values in (
            ("train", train_split_ids),
            ("test", development_split_ids),
        )
        for sample_id in values
    }
    split_sample_meta = {
        str(row.get("sample_id")): row
        for row in split.get("samples", [])
        if isinstance(row, Mapping)
    }
    for sample_id, partition in sample_split.items():
        meta = split_sample_meta.get(sample_id)
        if meta is None:
            raise ValueError("v5r4 split sample metadata is incomplete")
        if meta.get("stage") != "base" or meta.get("split") != partition:
            raise ValueError("v5r4 split sample metadata is inconsistent")
        if str(meta.get("target")) not in PROMPT_BY_TARGET:
            raise ValueError("v5r4 split sample target is unsupported")
    component_partitions: Dict[str, set] = {}
    grouped_cdm_ids = set()
    for group in split.get("groups", []):
        if not isinstance(group, Mapping):
            raise TypeError("v5r4 split group is not a mapping")
        group_partition = str(group.get("split"))
        if group_partition not in {"train", "test"}:
            raise ValueError("v5r4 split group has an invalid partition")
        for component in group.get("source_components", []):
            component_partitions.setdefault(str(component), set()).add(
                group_partition
            )
        for sample_id in group.get("sample_ids", []):
            sample_id = str(sample_id)
            if sample_id in sample_split:
                if sample_split[sample_id] != group_partition:
                    raise ValueError("sample/group partition mismatch in v5r4 split")
                grouped_cdm_ids.add(sample_id)
    leaking_components = sorted(
        component
        for component, partitions in component_partitions.items()
        if len(partitions) != 1
    )
    if leaking_components:
        raise ValueError(
            "raw-source components cross train/development: "
            + str(leaking_components)
        )
    if grouped_cdm_ids != set(sample_split):
        raise ValueError("v5r4 split groups do not cover every CDM sample")
    if audit_sample_ids != train_split_ids:
        raise ValueError("train audit sample IDs differ from the exact split")
    if [str(value) for value in rollout_selection.get("probe_sample_ids", [])] != (
        train_split_ids
    ):
        raise ValueError("train rollout selection IDs differ from the exact split")
    recorded_rollout = selection.get("rollout_selection", {})
    for name in (
        "selected",
        "partition",
        "complete_train_partition",
        "k_samples",
        "seed_partition",
        "probe_sample_ids",
    ):
        if recorded_rollout.get(name) != rollout_selection.get(name):
            raise ValueError(
                "train summary/rollout selection content mismatch: " + name
            )
    recorded_evaluated_count = _require_exact_int(
        recorded_rollout.get("evaluated_candidate_count"),
        "train summary rollout evaluated_candidate_count",
        minimum=1,
    )
    if recorded_evaluated_count != evaluated_candidate_count:
        raise ValueError(
            "train summary/rollout selection content mismatch: "
            "evaluated_candidate_count vs candidate_count"
        )
    if _require_exact_int(
        audit.get("train_count"), "train audit train_count", minimum=1
    ) != len(train_split_ids):
        raise ValueError("train audit count differs from the exact split")
    expected_train_counts = {"chair": 18, "whiteboard": 6, "bed": 1}
    if audit.get("train_target_counts") != expected_train_counts:
        raise ValueError("train audit target counts differ from v5r4 contract")
    if train.get("split", {}).get("train_target_counts") != expected_train_counts:
        raise ValueError("train summary target counts differ from v5r4 contract")
    audit_rows = audit.get("per_sample", [])
    if not isinstance(audit_rows, list) or len(audit_rows) != len(train_split_ids):
        raise ValueError("train audit per-sample rows are incomplete")
    if [str(row.get("sample_id")) for row in audit_rows] != train_split_ids:
        raise ValueError("train audit per-sample row order differs from split")
    for row in audit_rows:
        if not isinstance(row, Mapping):
            raise TypeError("train audit per-sample row is not a mapping")
        target = str(row.get("target"))
        if row.get("text") != PROMPT_BY_TARGET.get(target):
            raise ValueError("train audit contains a non-canonical v5 prompt")
    actual_train_counts = {
        target: sum(str(row.get("target")) == target for row in audit_rows)
        for target in ("chair", "whiteboard", "bed")
    }
    if actual_train_counts != expected_train_counts:
        raise ValueError("train audit per-sample target counts are inconsistent")
    audit_fingerprints = _validate_dataset_snapshot(
        snapshot=audit.get("dataset_snapshot"),
        partition="train",
        expected_sample_ids=train_split_ids,
        split_sha256=hashes["split"],
        stats_sha256=hashes["stats"],
    )
    current_train_fingerprints, train_instance_ids = (
        _load_current_dataset_contract(
            dataset_root=dataset_root,
            split=split,
            sample_ids=train_split_ids,
        )
    )
    if current_train_fingerprints != audit_fingerprints:
        raise ValueError("current train dataset differs from sealed audit")
    train_prediction_contract = _validate_rollout_prediction_bundle(
        path=files["train_rollout_predictions"],
        expected_sample_ids=audit_sample_ids,
        expected_k_draws=audit_k_samples,
        expected_base_seed=20260815,
        seed_partition="train_audit",
        label="train rollout",
        expected_fingerprints=audit_fingerprints,
        return_arrays=True,
    )
    train_prediction_arrays = train_prediction_contract.pop("_arrays")
    independent_train_quality = _validate_independent_audit_quality(
        audit=audit,
        arrays=train_prediction_arrays,
        fingerprints=audit_fingerprints,
        instance_ids_by_sample=train_instance_ids,
    )

    development = load_json(files["development_summary"])
    if (
        development.get("schema") != DEVELOPMENT_SCHEMA
        or development.get("status") != "PASS"
    ):
        raise ValueError("development evaluation is not strict v5r4 PASS")
    if development.get("may_proceed_to_moe_iiw") is not True:
        raise ValueError("development summary forbids proceeding to MoE-IIW")
    if development.get("final_paper_test_requires_new_unseen_sources") is not True:
        raise ValueError("development is incorrectly presented as a final paper test")
    development_pairing = development.get("pairing_protocol", {})
    if development_pairing.get(
        "initial_xT_and_all_reverse_step_noise_paired"
    ) is not True or development_pairing.get("caller_rng_state_restored") is not True:
        raise ValueError("development paired-noise contract is incomplete")
    development_k_samples = _require_exact_int(
        development.get("k_samples"),
        "development k_samples",
        minimum=minimum_k_draws,
    )
    if development_k_samples != 5:
        raise ValueError("strict development evaluation requires exactly five draws")
    if development_k_samples < minimum_k_draws:
        raise ValueError("development evaluation has too few draws")
    development_predictions = development.get("predictions", {})
    if development_predictions.get("format") != (
        "paired_rollout_affordance_npz_v2"
    ):
        raise ValueError("development predictions format is missing or stale")
    if development_predictions.get("sha256") != hashes[
        "development_predictions"
    ]:
        raise ValueError("development/predictions SHA-256 mismatch")
    if Path(str(development_predictions.get("file", ""))).name != files[
        "development_predictions"
    ].name:
        raise ValueError("development predictions filename mismatch")
    development_sampling = development.get("sampling_provenance", {})
    expected_development_sampling = {
        "base_seed": 20260815,
        "partition": "development",
        "seed_derivation": "sha256(base|partition|sample_id|draw)",
        "seed_table_storage": "prediction_npz_int64_matrices",
    }
    if development_sampling != expected_development_sampling:
        raise ValueError("development sampling provenance mismatch")
    audit_ref = development.get("train_rollout_audit", {})
    if audit_ref.get("sha256") != hashes["train_rollout_audit"]:
        raise ValueError("development/audit SHA-256 mismatch")
    if audit_ref.get("schema") != AUDIT_SCHEMA or audit_ref.get("status") != "PASS":
        raise ValueError("development summary references a non-PASS audit")
    if audit_ref.get("test_sample_tensors_read") is not False:
        raise ValueError("development-referenced train audit read test tensors")
    audit_ref_k_samples = _require_exact_int(
        audit_ref.get("k_samples"),
        "development audit reference k_samples",
        minimum=1,
    )
    if audit_ref_k_samples < development_k_samples:
        raise ValueError("development uses more draws than the train audit")
    _require_all_boolean_checks(development, "development evaluation")
    actual_development_checks = set(development.get("checks", {}))
    if actual_development_checks != REQUIRED_DEVELOPMENT_CHECKS:
        raise ValueError(
            "development evaluation check set mismatch: missing="
            + str(sorted(REQUIRED_DEVELOPMENT_CHECKS - actual_development_checks))
            + ", unexpected="
            + str(sorted(actual_development_checks - REQUIRED_DEVELOPMENT_CHECKS))
        )
    development_sample_ids = [
        str(row.get("sample_id"))
        for row in development.get("per_sample", [])
        if isinstance(row, Mapping)
    ]
    if development_sample_ids != development_split_ids:
        raise ValueError(
            "development summary IDs/order differ from the exact split"
        )
    development_targets = [
        str(row.get("target"))
        for row in development.get("per_sample", [])
        if isinstance(row, Mapping)
    ]
    expected_development_counts = {"chair": 5, "whiteboard": 6, "bed": 1}
    actual_development_counts = {
        target: development_targets.count(target)
        for target in sorted(set(development_targets))
    }
    if actual_development_counts != expected_development_counts:
        raise ValueError("development target counts differ from v5r4 contract")
    for row in development.get("per_sample", []):
        target = str(row.get("target"))
        if row.get("text") != PROMPT_BY_TARGET.get(target):
            raise ValueError("development contains a non-canonical v5 prompt")
    development_fingerprints = _validate_dataset_snapshot(
        snapshot=development.get("dataset_snapshot"),
        partition="development",
        expected_sample_ids=development_split_ids,
        split_sha256=hashes["split"],
        stats_sha256=hashes["stats"],
    )
    current_development_fingerprints, development_instance_ids = (
        _load_current_dataset_contract(
            dataset_root=dataset_root,
            split=split,
            sample_ids=development_split_ids,
        )
    )
    if current_development_fingerprints != development_fingerprints:
        raise ValueError(
            "current development dataset differs from sealed evaluation"
        )
    development_prediction_contract = _validate_rollout_prediction_bundle(
        path=files["development_predictions"],
        expected_sample_ids=development_sample_ids,
        expected_k_draws=development_k_samples,
        expected_base_seed=20260815,
        seed_partition="development",
        label="development",
        expected_fingerprints=development_fingerprints,
        return_arrays=True,
    )
    development_prediction_arrays = development_prediction_contract.pop("_arrays")
    independent_development_quality = (
        _validate_independent_development_quality(
            development=development,
            arrays=development_prediction_arrays,
            fingerprints=development_fingerprints,
            instance_ids_by_sample=development_instance_ids,
        )
    )
    quality_replay_plan = {
        "schema": "history_affordance_v2_full_quality_replay_plan_v1",
        "mode": "all_samples_all_draws_both_models",
        "base_seed": 20260815,
        "roles": ["fewshot", "original"],
        "partitions": {
            "train_audit": {
                "sample_ids": list(train_split_ids),
                "sample_count": len(train_split_ids),
                "k_draws": audit_k_samples,
                "total_replayed_draws_per_role": (
                    len(train_split_ids) * audit_k_samples
                ),
            },
            "development": {
                "sample_ids": list(development_split_ids),
                "sample_count": len(development_split_ids),
                "k_draws": development_k_samples,
                "total_replayed_draws_per_role": (
                    len(development_split_ids) * development_k_samples
                ),
            },
        },
        "total_replayed_draws": 2
        * (
            len(train_split_ids) * audit_k_samples
            + len(development_split_ids) * development_k_samples
        ),
    }

    train_steps = _require_exact_int(
        train.get("diffusion_steps"), "training diffusion_steps", minimum=1
    )
    audit_steps = _require_exact_int(
        audit.get("diffusion_steps"), "train audit diffusion_steps", minimum=1
    )
    development_steps = _require_exact_int(
        development.get("diffusion_steps"),
        "development diffusion_steps",
        minimum=1,
    )
    selection_steps = _require_exact_int(
        rollout_selection.get("diffusion_steps"),
        "rollout selection diffusion_steps",
        minimum=1,
    )
    if len({train_steps, selection_steps, audit_steps, development_steps}) != 1:
        raise ValueError("diffusion-step mismatch across v5r4 evidence")
    if expected_diffusion_steps is not None and train_steps != expected_diffusion_steps:
        raise ValueError("v5r4 evidence uses an unexpected diffusion-step count")
    selection_k = rollout_selection_k
    audit_k = audit_k_samples
    development_k = development_k_samples
    if len({selection_k, audit_k, development_k}) != 1:
        raise ValueError("draw-count mismatch across v5r4 evidence")
    if selection_k < minimum_k_draws:
        raise ValueError("v5r4 evidence has too few paired draws")

    evidence_set_sha256 = canonical_json_sha256(
        {
            "schemas": {
                "train": TRAIN_SCHEMA,
                "shortlist": SHORTLIST_SCHEMA,
                "rollout_selection": ROLLOUT_SELECTION_SCHEMA,
                "train_rollout_audit": AUDIT_SCHEMA,
                "development": DEVELOPMENT_SCHEMA,
            },
            "file_sha256": hashes,
            "diffusion_steps": train_steps,
            "train_prediction_contract": train_prediction_contract,
            "development_prediction_contract": development_prediction_contract,
            "independent_train_quality": independent_train_quality,
            "independent_development_quality": independent_development_quality,
            "quality_replay_plan": quality_replay_plan,
            "rollout_protocol_hashes": dict(rollout_protocol_hashes),
        }
    )
    return {
        "status": "PASS",
        "schemas": {
            "train": TRAIN_SCHEMA,
            "shortlist": SHORTLIST_SCHEMA,
            "rollout_selection": ROLLOUT_SELECTION_SCHEMA,
            "train_rollout_audit": AUDIT_SCHEMA,
            "development": DEVELOPMENT_SCHEMA,
        },
        "files": {name: str(path) for name, path in files.items()},
        "sha256": hashes,
        "evidence_set_sha256": evidence_set_sha256,
        "rollout_protocol_sha256": dict(rollout_protocol_hashes),
        "diffusion_steps": train_steps,
        "train_audit_k_draws": audit_k_samples,
        "development_k_draws": development_k_samples,
        "prediction_contracts": {
            "train_rollout": train_prediction_contract,
            "development": development_prediction_contract,
        },
        "independent_quality": {
            "train_rollout": independent_train_quality,
            "development": independent_development_quality,
        },
        "quality_replay_plan": quality_replay_plan,
    }


def validate_prompt(prompt_id: str, text: str) -> None:
    if not PROMPT_ID_PATTERN.fullmatch(prompt_id):
        raise ValueError(
            "prompt_id must match " + PROMPT_ID_PATTERN.pattern + ": " + prompt_id
        )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if text != text.strip():
        raise ValueError("text must not contain leading/trailing whitespace")


def load_scene_contract(dataset_root: Path, scene_id: str) -> Dict[str, object]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    index_file = dataset_root / ("index_" + scene_id + ".json")
    index = load_json(index_file)
    if str(index.get("scene_id")) != scene_id:
        raise ValueError("scene index ID mismatch")
    scene_dir = dataset_root / str(index["scene_adm_input"])
    points_file = scene_dir / "points.npz"
    sidecar_file = scene_dir / "sidecar.npz"
    for path in (points_file, sidecar_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(points_file, allow_pickle=False) as source:
        if "points" not in source:
            raise KeyError(str(points_file) + " has no points array")
        points = source["points"].astype(np.float32)
    with np.load(sidecar_file, allow_pickle=False) as source:
        required = (
            "xyz_afford_z_up",
            "instance_ids",
            "source_indices",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(str(sidecar_file) + " misses " + str(missing))
        xyz = source["xyz_afford_z_up"].astype(np.float32)
        instance_ids = source["instance_ids"].astype(np.int64)
        source_indices = source["source_indices"].astype(np.int64)

    if points.shape != (NUM_POINTS, 6):
        raise ValueError("scene points must be [8192,6]")
    if xyz.shape != (NUM_POINTS, 3):
        raise ValueError("scene xyz must be [8192,3]")
    if instance_ids.shape != (NUM_POINTS,):
        raise ValueError("instance_ids must be [8192]")
    if source_indices.shape != (NUM_POINTS,):
        raise ValueError("source_indices must be [8192]")
    if not np.isfinite(points).all() or not np.isfinite(xyz).all():
        raise ValueError("scene points contain NaN/Inf")
    if not np.allclose(points[:, :3], xyz, atol=1e-6, rtol=0.0):
        raise ValueError("points.npz and sidecar.npz point ordering differs")
    if len(np.unique(source_indices)) != NUM_POINTS:
        raise ValueError("source_indices must be unique")
    if np.any(points[:, 3:6] < 0.0) or np.any(points[:, 3:6] > 255.0):
        raise ValueError("scene RGB must lie in [0,255]")
    rgb01 = np.ascontiguousarray(points[:, 3:6] / np.float32(255.0))

    scene_hash_payload = {
        "scene_id": scene_id,
        "points_file_sha256": sha256_file(points_file),
        "sidecar_file_sha256": sha256_file(sidecar_file),
        "points_array_sha256": sha256_array(points),
        "xyz_array_sha256": sha256_array(xyz),
        "rgb01_array_sha256": sha256_array(rgb01),
        "instance_ids_sha256": sha256_array(instance_ids),
        "source_indices_sha256": sha256_array(source_indices),
    }
    return {
        "scene_id": scene_id,
        "index_file": str(index_file),
        "index_file_sha256": sha256_file(index_file),
        "points_file": str(points_file),
        "sidecar_file": str(sidecar_file),
        "points": points,
        "xyz": xyz,
        "rgb01": rgb01,
        "instance_ids": instance_ids,
        "source_indices": source_indices,
        "hash_payload": scene_hash_payload,
        "scene_sha256": canonical_json_sha256(scene_hash_payload),
    }


def load_quality_replay_row(
    dataset_root: Path, split_file: Path, sample_id: str
) -> Dict[str, object]:
    """Load one canonical v5 row without importing the PyTorch dataset stack."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    split = load_json(split_file)
    metadata = {
        str(row.get("sample_id")): row
        for row in split.get("samples", [])
        if isinstance(row, Mapping)
    }.get(sample_id)
    if metadata is None:
        raise KeyError("quality replay sample is absent from split: " + sample_id)
    target = str(metadata.get("target"))
    if target not in PROMPT_BY_TARGET:
        raise ValueError("quality replay target is unsupported")
    scene_id = str(metadata.get("scene_id"))
    index = load_json(dataset_root / ("index_" + scene_id + ".json"))
    matching = [
        entry
        for entry in index.get("samples", [])
        if isinstance(entry, Mapping) and str(entry.get("sample_id")) == sample_id
    ]
    if len(matching) != 1:
        raise ValueError("quality replay dataset index match is not unique")
    scene_dir = dataset_root / str(index["scene_adm_input"])
    with np.load(scene_dir / "points.npz", allow_pickle=False) as source:
        points = source["points"].astype(np.float32)
    if points.shape != (NUM_POINTS, 6) or not np.isfinite(points).all():
        raise ValueError("quality replay scene points are invalid")
    if np.any(points[:, 3:6] < 0.0) or np.any(points[:, 3:6] > 255.0):
        raise ValueError("quality replay RGB lies outside [0,255]")
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "target": target,
        "xyz": np.ascontiguousarray(points[:, :3]),
        "feat": np.ascontiguousarray(
            (points[:, 3:6] / np.float32(255.0)).astype(np.float32)
        ),
        "text": PROMPT_BY_TARGET[target],
    }


def load_contact_stats(stats_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = _resolved_file(stats_file, "contact stats")
    with np.load(path, allow_pickle=False) as source:
        if "mean" not in source or "std" not in source:
            raise KeyError("contact stats must contain mean and std")
        mean = source["mean"].astype(np.float32)
        std = source["std"].astype(np.float32)
    if mean.shape != (1, NUM_CHANNELS) or std.shape != (1, NUM_CHANNELS):
        raise ValueError("contact mean/std must both be [1,6]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("contact stats contain NaN/Inf")
    if np.any(std <= 0.0):
        raise ValueError("contact std must be strictly positive")
    return mean, std


def convert_prediction_draws(
    draws: np.ndarray,
    *,
    representation: str,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Convert explicitly typed draws to canonical affordance in [0,1]."""

    if representation not in ALLOWED_REPRESENTATIONS:
        raise ValueError(
            "representation must be one of " + str(ALLOWED_REPRESENTATIONS)
        )
    value = np.asarray(draws, dtype=np.float32)
    if value.shape == (NUM_POINTS, NUM_CHANNELS):
        value = value[None, ...]
    if value.ndim != 3 or value.shape[1:] != (NUM_POINTS, NUM_CHANNELS):
        raise ValueError("draws must be [K,8192,6] or [8192,6]")
    if value.shape[0] <= 0:
        raise ValueError("draws must contain at least one sample")
    if not np.isfinite(value).all():
        raise ValueError("prediction draws contain NaN/Inf")

    if representation == "affordance":
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError("affordance draws must lie in [0,1]")
        converted = value.copy()
        transform = {
            "name": "identity",
            "gaussian_applied": False,
            "denormalization_applied": False,
        }
    elif representation == "normalized_contact":
        if mean is None or std is None:
            raise ValueError("normalized_contact requires explicit mean and std")
        mean_value = np.asarray(mean, dtype=np.float32)
        std_value = np.asarray(std, dtype=np.float32)
        if mean_value.shape != (1, NUM_CHANNELS) or std_value.shape != (
            1,
            NUM_CHANNELS,
        ):
            raise ValueError("mean/std must both be [1,6]")
        if not np.isfinite(mean_value).all() or not np.isfinite(std_value).all():
            raise ValueError("mean/std contains NaN/Inf")
        if np.any(std_value <= 0.0):
            raise ValueError("std must be strictly positive")
        converted = np.clip(
            value * std_value[None, :, :] + mean_value[None, :, :],
            1e-20,
            1.0,
        ).astype(np.float32)
        transform = {
            "name": "contact_denormalize_then_clip",
            "gaussian_applied": False,
            "denormalization_applied": True,
            "mean_sha256": sha256_array(mean_value),
            "std_sha256": sha256_array(std_value),
        }
    else:
        if sigma is None or not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
            raise ValueError("distance representation requires positive sigma")
        if np.any(value < 0.0):
            raise ValueError("distance draws must be non-negative")
        converted = np.exp(
            -0.5 * (value / float(sigma)) ** 2
        ).astype(np.float32)
        transform = {
            "name": "gaussian_distance_to_affordance",
            "gaussian_applied": True,
            "denormalization_applied": False,
            "sigma": float(sigma),
        }

    if not np.isfinite(converted).all():
        raise ValueError("converted affordance contains NaN/Inf")
    if np.any(converted < 0.0) or np.any(converted > 1.0):
        raise AssertionError("converted affordance lies outside [0,1]")
    return np.ascontiguousarray(converted), transform


def derive_draw_seeds(
    *, cache_key: str, base_seed: int, draw_index: int
) -> Tuple[int, int]:
    if type(draw_index) is not int or draw_index < 0:
        raise ValueError("draw_index must be non-negative")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")

    def one(kind: str) -> int:
        payload = (
            ARTIFACT_SCHEMA
            + "|"
            + cache_key
            + "|"
            + str(base_seed)
            + "|"
            + str(draw_index)
            + "|"
            + kind
        )
        value = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
        )
        return value & ((1 << 63) - 1)

    return one("initial_xT"), one("reverse_noise")


def build_cache_key_payload(
    *,
    scene_sha256: str,
    prompt_id: str,
    text: str,
    checkpoint_sha256: str,
    stats_sha256: str,
    teacher_runtime_sha256: str,
    diffusion_steps: int,
    k_draws: int,
    base_seed: int,
    prompt_policy_id: str = PROMPT_POLICY_ID,
) -> Dict[str, object]:
    validate_prompt(prompt_id, text)
    hash_values = {
        "scene": scene_sha256,
        "checkpoint": checkpoint_sha256,
        "stats": stats_sha256,
        "teacher_runtime": teacher_runtime_sha256,
    }
    invalid_hashes = [
        name
        for name, value in hash_values.items()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None
    ]
    if invalid_hashes:
        raise ValueError(
            "cache SHA-256 values must be lowercase hexadecimal: "
            + str(invalid_hashes)
        )
    if (
        type(diffusion_steps) is not int
        or type(k_draws) is not int
        or type(base_seed) is not int
        or diffusion_steps <= 0
        or k_draws <= 0
        or base_seed < 0
    ):
        raise ValueError("diffusion_steps/k_draws/base_seed are invalid")
    if prompt_policy_id != PROMPT_POLICY_ID:
        raise ValueError("Base v5r4 export requires prompt policy " + PROMPT_POLICY_ID)
    if text not in PROMPT_BY_TARGET.values():
        raise ValueError("text is not an exact object-agnostic v5 prompt")
    return {
        "schema": ARTIFACT_SCHEMA,
        "scene_sha256": scene_sha256,
        "prompt_id": prompt_id,
        "prompt_policy_id": prompt_policy_id,
        "text_sha256": sha256_text(text),
        "checkpoint_sha256": checkpoint_sha256,
        "stats_sha256": stats_sha256,
        "teacher_runtime_sha256": teacher_runtime_sha256,
        "diffusion_steps": int(diffusion_steps),
        "draw_policy": "mean_of_deterministic_denormalized_affordance_draws",
        "k_draws": int(k_draws),
        "base_seed": int(base_seed),
        "channel_order": list(CHANNEL_ORDER),
    }


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix="." + path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(path.parent),
        prefix="." + path.name + ".",
        suffix=".npz",
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb+") as reader:
            os.fsync(reader.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_base_teacher_artifact(
    *,
    output_dir: Path,
    scene: Mapping[str, object],
    prompt_id: str,
    text: str,
    affordance_draws: np.ndarray,
    transform: Mapping[str, object],
    quality_gate: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    cache_key_payload: Mapping[str, object],
    draw_seeds: Sequence[Tuple[int, int]],
    source_representation: str,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Write an immutable NPZ+JSON pair and validate it before returning."""

    validate_prompt(prompt_id, text)
    if source_representation not in ALLOWED_REPRESENTATIONS:
        raise ValueError("invalid source representation")
    if source_representation == "distance":
        raise ValueError(
            "production Base teacher artifacts forbid distance input; "
            "Gaussian distance conversion belongs only to legacy import"
        )
    draws = np.asarray(affordance_draws, dtype=np.float32)
    if draws.ndim != 3 or draws.shape[1:] != (NUM_POINTS, NUM_CHANNELS):
        raise ValueError("canonical draws must be [K,8192,6]")
    if len(draw_seeds) != draws.shape[0]:
        raise ValueError("draw_seeds count differs from affordance draws")
    if cache_key_payload.get("k_draws") != int(draws.shape[0]):
        raise ValueError("cache payload draw count differs from affordance draws")
    if quality_gate.get("status") != "PASS":
        raise ValueError("quality gate must be PASS")
    if runtime_provenance.get("status") != "PASS":
        raise ValueError("runtime provenance must be PASS")
    if cache_key_payload.get("base_seed") != 20260815:
        raise ValueError("strict Base teacher v1 requires base seed 20260815")
    if not np.isfinite(draws).all() or np.any(draws < 0.0) or np.any(draws > 1.0):
        raise ValueError("canonical draws must be finite in [0,1]")

    mean_affordance = draws.mean(axis=0, dtype=np.float64).astype(np.float32)
    std_affordance = draws.std(axis=0, dtype=np.float64).astype(np.float32)
    xyz = np.asarray(scene["xyz"], dtype=np.float32)
    rgb01 = np.asarray(scene["rgb01"], dtype=np.float32)
    instance_ids = np.asarray(scene["instance_ids"], dtype=np.int64)
    source_indices = np.asarray(scene["source_indices"], dtype=np.int64)
    if xyz.shape != (NUM_POINTS, 3) or rgb01.shape != (NUM_POINTS, 3):
        raise ValueError("scene xyz/RGB shape changed")
    if instance_ids.shape != (NUM_POINTS,) or source_indices.shape != (
        NUM_POINTS,
    ):
        raise ValueError("scene instance/source index shape changed")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb01).all():
        raise ValueError("scene xyz/RGB contains NaN/Inf")
    if np.any(rgb01 < 0.0) or np.any(rgb01 > 1.0):
        raise ValueError("scene RGB lies outside [0,1]")
    if np.any(instance_ids < 0) or np.any(source_indices < 0):
        raise ValueError("scene instance/source indices must be non-negative")
    if len(np.unique(source_indices)) != NUM_POINTS:
        raise ValueError("scene source indices must be unique")
    draw_indices = np.arange(draws.shape[0], dtype=np.int64)
    initial_noise_seeds = np.asarray(
        [int(pair[0]) for pair in draw_seeds], dtype=np.int64
    )
    reverse_noise_seeds = np.asarray(
        [int(pair[1]) for pair in draw_seeds], dtype=np.int64
    )

    output_dir = Path(output_dir).expanduser().resolve()
    artifact_file = output_dir / "base_teacher.npz"
    manifest_file = output_dir / "manifest.json"
    if not overwrite and (artifact_file.exists() or manifest_file.exists()):
        raise FileExistsError(
            "refusing to overwrite existing Base teacher artifact: "
            + str(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "affordance": mean_affordance,
        "affordance_std": std_affordance,
        "affordance_draws": np.ascontiguousarray(draws),
        "xyz": xyz,
        "rgb01": rgb01,
        "instance_ids": instance_ids,
        "source_indices": source_indices,
        "channel_names": np.asarray(CHANNEL_ORDER),
        "channel_joint_indices": np.asarray(
            CHANNEL_JOINT_INDICES, dtype=np.int64
        ),
        "draw_indices": draw_indices,
        "initial_noise_seeds": initial_noise_seeds,
        "reverse_noise_seeds": reverse_noise_seeds,
    }
    atomic_savez(artifact_file, **arrays)

    cache_key = canonical_json_sha256(cache_key_payload)
    artifact_id = canonical_json_sha256(
        {
            "schema": ARTIFACT_SCHEMA,
            "cache_key": cache_key,
            "affordance_sha256": sha256_array(mean_affordance),
            "draws_sha256": sha256_array(draws),
            "scene_sha256": str(scene["scene_sha256"]),
        }
    )
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "status": "STAGED",
        "promotion_authorized": False,
        "artifact_id": artifact_id,
        "artifact_file": str(artifact_file),
        "artifact_file_sha256": sha256_file(artifact_file),
        "scene_id": str(scene["scene_id"]),
        "scene_sha256": str(scene["scene_sha256"]),
        "scene_hash_payload": dict(scene["hash_payload"]),
        "scene_index_file": str(scene["index_file"]),
        "scene_index_file_sha256": str(scene["index_file_sha256"]),
        "prompt_id": prompt_id,
        "prompt_policy_id": str(cache_key_payload["prompt_policy_id"]),
        "text": text,
        "text_sha256": sha256_text(text),
        "conditioning": {
            "keys": ["scene_points", "text"],
            "history_conditioned": False,
        },
        "source_representation": source_representation,
        "canonical_representation": "affordance",
        "normalization_state": "denormalized_clipped",
        "affordance_definition": "exp(-0.5*(distance_m/sigma_m)^2)",
        "kernel_sigma_m": 0.8,
        "transform": dict(transform),
        "channel_order": list(CHANNEL_ORDER),
        "channel_joint_indices": list(CHANNEL_JOINT_INDICES),
        "shape": list(mean_affordance.shape),
        "draw_shape": list(draws.shape),
        "draw_count": int(draws.shape[0]),
        "draw_policy": "mean_of_deterministic_denormalized_affordance_draws",
        "draw_seeds": [
            {
                "draw_index": index,
                "initial_xT_seed": int(pair[0]),
                "reverse_noise_seed": int(pair[1]),
            }
            for index, pair in enumerate(draw_seeds)
        ],
        "cache_key": cache_key,
        "cache_key_payload": dict(cache_key_payload),
        "quality_gate": dict(quality_gate),
        "runtime_provenance": dict(runtime_provenance),
        "array_sha256": {
            name: sha256_array(value) for name, value in arrays.items()
        },
        "affordance_range": [
            float(mean_affordance.min()),
            float(mean_affordance.max()),
        ],
        "draw_range": [float(draws.min()), float(draws.max())],
    }
    atomic_write_json(manifest_file, manifest)
    validate_base_teacher_artifact(
        manifest_file=manifest_file,
        artifact_file=artifact_file,
    )
    return manifest_file, artifact_file


def validate_base_teacher_artifact(
    *, manifest_file: Path, artifact_file: Optional[Path] = None
) -> Dict[str, object]:
    manifest_path = _resolved_file(manifest_file, "Base teacher manifest")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("status") != "STAGED"
        or manifest.get("promotion_authorized") is not False
    ):
        raise ValueError("Base teacher manifest is not an unpromoted STAGED artifact")
    artifact_path = (
        _resolved_file(artifact_file, "Base teacher artifact")
        if artifact_file is not None
        else _resolved_file(Path(str(manifest["artifact_file"])), "Base teacher artifact")
    )
    if sha256_file(artifact_path) != manifest.get("artifact_file_sha256"):
        raise ValueError("Base teacher artifact file SHA-256 mismatch")
    validate_prompt(str(manifest.get("prompt_id", "")), str(manifest.get("text", "")))
    if manifest.get("prompt_policy_id") != PROMPT_POLICY_ID:
        raise ValueError("Base teacher prompt policy mismatch")
    if manifest.get("text") not in PROMPT_BY_TARGET.values():
        raise ValueError("Base teacher prompt is outside the v5 policy")
    if manifest.get("text_sha256") != sha256_text(str(manifest.get("text"))):
        raise ValueError("Base teacher prompt SHA-256 mismatch")
    conditioning = manifest.get("conditioning", {})
    if conditioning.get("keys") != ["scene_points", "text"]:
        raise ValueError("Base teacher conditioning keys changed")
    if conditioning.get("history_conditioned") is not False:
        raise ValueError("Base teacher must not contain history conditioning")
    if manifest.get("scene_sha256") != canonical_json_sha256(
        manifest.get("scene_hash_payload", {})
    ):
        raise ValueError("Base teacher scene SHA-256 mismatch")
    if manifest.get("canonical_representation") != "affordance":
        raise ValueError("canonical Base teacher representation is not affordance")
    if manifest.get("normalization_state") != "denormalized_clipped":
        raise ValueError("Base teacher normalization state mismatch")
    if manifest.get("affordance_definition") != (
        "exp(-0.5*(distance_m/sigma_m)^2)"
    ):
        raise ValueError("Base teacher affordance definition mismatch")
    if manifest.get("draw_policy") != (
        "mean_of_deterministic_denormalized_affordance_draws"
    ):
        raise ValueError("Base teacher manifest draw policy mismatch")
    source_representation = str(manifest.get("source_representation"))
    if source_representation == "distance":
        raise ValueError("production Base teacher source cannot be distance")
    transform = manifest.get("transform", {})
    if source_representation == "affordance":
        if transform.get("name") != "identity":
            raise ValueError("affordance source was not identity transformed")
        if transform.get("source_model_output") != "affordance":
            raise ValueError("affordance transform source declaration mismatch")
        if transform.get("canonical_output") != "affordance":
            raise ValueError("affordance transform output declaration mismatch")
        if transform.get("gaussian_applied") is not False:
            raise ValueError("affordance source was incorrectly Gaussian transformed")
    elif source_representation == "normalized_contact":
        if transform.get("name") != "contact_denormalize_then_clip":
            raise ValueError("normalized contact used an unexpected transform")
        if transform.get("gaussian_applied") is not False:
            raise ValueError("normalized contact was incorrectly Gaussian transformed")
        if transform.get("denormalization_applied") is not True:
            raise ValueError("normalized contact was not denormalized")
        if transform.get("source_model_output") != "normalized_contact":
            raise ValueError("normalized-contact source declaration mismatch")
        if transform.get("canonical_output") != (
            "denormalized_clipped_affordance"
        ):
            raise ValueError("normalized-contact output declaration mismatch")
    else:
        raise ValueError("Base teacher source representation is unsupported")
    if transform.get("distance_kernel_reapplied") is not False:
        raise ValueError("distance kernel reapplication is not explicitly false")
    if manifest.get("channel_order") != list(CHANNEL_ORDER):
        raise ValueError("Base teacher channel order mismatch")
    if manifest.get("channel_joint_indices") != list(CHANNEL_JOINT_INDICES):
        raise ValueError("Base teacher contact-joint indices mismatch")
    if type(manifest.get("kernel_sigma_m")) is not float or manifest.get(
        "kernel_sigma_m"
    ) != 0.8:
        raise ValueError("Base teacher kernel sigma mismatch")
    cache_payload = manifest.get("cache_key_payload", {})
    if manifest.get("cache_key") != canonical_json_sha256(cache_payload):
        raise ValueError("Base teacher cache key mismatch")
    expected_cache_fields = {
        "schema": ARTIFACT_SCHEMA,
        "scene_sha256": manifest.get("scene_sha256"),
        "prompt_id": manifest.get("prompt_id"),
        "prompt_policy_id": PROMPT_POLICY_ID,
        "text_sha256": manifest.get("text_sha256"),
        "channel_order": list(CHANNEL_ORDER),
    }
    for name, expected in expected_cache_fields.items():
        if cache_payload.get(name) != expected:
            raise ValueError("Base teacher cache payload mismatch: " + name)
    _require_exact_int(
        cache_payload.get("diffusion_steps"),
        "Base teacher diffusion_steps",
        expected=500,
    )
    _require_exact_int(
        cache_payload.get("k_draws"), "Base teacher k_draws", expected=5
    )
    _require_exact_int(
        cache_payload.get("base_seed"),
        "Base teacher base_seed",
        expected=20260815,
    )
    _require_exact_int(
        manifest.get("draw_count"), "Base teacher draw_count", expected=5
    )
    for hash_name in (
        "scene_sha256",
        "checkpoint_sha256",
        "stats_sha256",
        "teacher_runtime_sha256",
        "text_sha256",
    ):
        _require_sha256(
            cache_payload.get(hash_name), "Base teacher cache " + hash_name
        )
    if cache_payload.get("draw_policy") != (
        "mean_of_deterministic_denormalized_affordance_draws"
    ):
        raise ValueError("Base teacher draw policy mismatch")
    if manifest.get("quality_gate", {}).get("status") != "PASS":
        raise ValueError("Base teacher quality provenance is not PASS")
    if re.fullmatch(
        r"[0-9a-f]{64}",
        str(manifest.get("quality_gate", {}).get("evidence_set_sha256", "")),
    ) is None:
        raise ValueError("Base teacher quality evidence-set seal is missing")
    quality_checkpoint_sha256 = manifest.get("quality_gate", {}).get(
        "sha256", {}
    ).get("fewshot_checkpoint")
    if quality_checkpoint_sha256 != cache_payload.get("checkpoint_sha256"):
        raise ValueError("Base teacher cache/checkpoint quality hash mismatch")
    quality_stats_sha256 = manifest.get("quality_gate", {}).get(
        "sha256", {}
    ).get("stats")
    if quality_stats_sha256 != cache_payload.get("stats_sha256"):
        raise ValueError("Base teacher cache/stats quality hash mismatch")
    runtime = manifest.get("runtime_provenance", {})
    if runtime.get("status") != "PASS":
        raise ValueError("Base teacher runtime provenance is not PASS")
    state = runtime.get("effective_model_state", {})
    required_state_hashes = (
        "full_state_sha256",
        "cdm_core_state_sha256",
        "scene_model_state_sha256",
        "text_model_state_sha256",
    )
    for name in required_state_hashes:
        _require_sha256(state.get(name), "effective model state hash " + name)
    if runtime.get("all_parameters_frozen") is not True:
        raise ValueError("effective Base teacher parameters are not frozen")
    if runtime.get("model_eval_mode") is not True:
        raise ValueError("effective Base teacher model is not in eval mode")
    if runtime.get("state_unchanged_during_sampling") is not True:
        raise ValueError("effective Base teacher changed during sampling")
    if runtime.get("teacher_runtime_sha256") != cache_payload.get(
        "teacher_runtime_sha256"
    ):
        raise ValueError("Base teacher cache/runtime identity mismatch")
    if runtime.get("teacher_runtime_sha256") != canonical_json_sha256(
        runtime.get("teacher_runtime_identity", {})
    ):
        raise ValueError("Base teacher runtime identity SHA-256 mismatch")
    recorded_protocol = manifest.get("quality_gate", {}).get(
        "rollout_protocol_sha256", {}
    )
    current_runtime_files = runtime.get("runtime_file_sha256", {})
    if not isinstance(recorded_protocol, Mapping) or set(recorded_protocol) != set(
        ROLLOUT_PROTOCOL_FILES
    ):
        raise ValueError("Base teacher rollout protocol file set mismatch")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in recorded_protocol.values()
    ):
        raise ValueError("Base teacher rollout protocol hash is invalid")
    if not isinstance(current_runtime_files, Mapping) or any(
        current_runtime_files.get(name) != value
        for name, value in recorded_protocol.items()
    ):
        raise ValueError("Base teacher runtime differs from rollout protocol")
    for required_runtime_name in (
        "utils/registry.py",
        "binary/pointops_cuda",
    ):
        if required_runtime_name not in current_runtime_files:
            raise ValueError(
                "Base teacher runtime omits " + required_runtime_name
            )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in current_runtime_files.values()
    ):
        raise ValueError("Base teacher runtime contains an invalid file hash")
    _require_sha256(
        runtime.get("torch_build_config_sha256"),
        "Base teacher PyTorch build-config hash",
    )
    _require_sha256(
        runtime.get("resolved_config_sha256"),
        "Base teacher resolved-config hash",
    )
    if runtime.get("resolved_config_sha256") != canonical_json_sha256(
        runtime.get("resolved_config", {})
    ):
        raise ValueError("Base teacher resolved configuration hash mismatch")
    sampling_environment_names = (
        "python_version",
        "numpy_version",
        "torch_version",
        "torch_build_config_sha256",
        "cuda_version",
        "cudnn_version",
        "device",
        "device_name",
        "device_capability",
    )
    sampling_environment = {
        name: runtime.get(name) for name in sampling_environment_names
    }
    expected_runtime_identity = {
        "schema": "history_affordance_v2_teacher_runtime_identity_v1",
        "fewshot_checkpoint_sha256": quality_checkpoint_sha256,
        "contact_stats_sha256": quality_stats_sha256,
        "effective_model_state": state,
        "partial_checkpoint_coverage": runtime.get(
            "partial_checkpoint_coverage"
        ),
        "scene_model_pretrained_weight_sha256": runtime.get(
            "scene_model_pretrained_weight_sha256"
        ),
        "text_model_name": runtime.get("text_model_name"),
        "resolved_config_sha256": runtime.get("resolved_config_sha256"),
        "runtime_file_set_sha256": canonical_json_sha256(
            current_runtime_files
        ),
        "sampling_environment": sampling_environment,
    }
    if runtime.get("teacher_runtime_identity") != expected_runtime_identity:
        raise ValueError("Base teacher runtime identity fields are inconsistent")
    for name in (
        "python_version",
        "numpy_version",
        "torch_version",
        "torch_build_config_sha256",
        "device",
        "device_name",
        "resolved_config_sha256",
        "scene_model_pretrained_weight_sha256",
    ):
        if runtime.get(name) in (None, ""):
            raise ValueError("missing Base teacher runtime provenance: " + name)

    with np.load(artifact_path, allow_pickle=False) as source:
        required = (
            "affordance",
            "affordance_std",
            "affordance_draws",
            "xyz",
            "rgb01",
            "instance_ids",
            "source_indices",
            "channel_names",
            "channel_joint_indices",
            "draw_indices",
            "initial_noise_seeds",
            "reverse_noise_seeds",
        )
        extra = sorted(set(source.files).difference(required))
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError("Base teacher NPZ misses " + str(missing))
        if extra:
            raise KeyError("Base teacher NPZ has forbidden extra keys " + str(extra))
        arrays = {name: source[name] for name in required}

    expected_dtypes = {
        "affordance": np.dtype(np.float32),
        "affordance_std": np.dtype(np.float32),
        "affordance_draws": np.dtype(np.float32),
        "xyz": np.dtype(np.float32),
        "rgb01": np.dtype(np.float32),
        "instance_ids": np.dtype(np.int64),
        "source_indices": np.dtype(np.int64),
        "channel_joint_indices": np.dtype(np.int64),
        "draw_indices": np.dtype(np.int64),
        "initial_noise_seeds": np.dtype(np.int64),
        "reverse_noise_seeds": np.dtype(np.int64),
    }
    for name, expected_dtype in expected_dtypes.items():
        if arrays[name].dtype != expected_dtype:
            raise TypeError(name + " dtype mismatch")

    affordance = arrays["affordance"].astype(np.float32)
    affordance_std = arrays["affordance_std"].astype(np.float32)
    draws = arrays["affordance_draws"].astype(np.float32)
    if affordance.shape != (NUM_POINTS, NUM_CHANNELS):
        raise ValueError("Base teacher affordance must be [8192,6]")
    if manifest.get("shape") != list(affordance.shape):
        raise ValueError("Base teacher manifest shape declaration mismatch")
    if affordance_std.shape != affordance.shape:
        raise ValueError("Base teacher affordance_std shape mismatch")
    if draws.ndim != 3 or draws.shape[1:] != affordance.shape:
        raise ValueError("Base teacher draws must be [K,8192,6]")
    if manifest.get("draw_shape") != list(draws.shape):
        raise ValueError("Base teacher manifest draw-shape declaration mismatch")
    if arrays["xyz"].shape != (NUM_POINTS, 3) or arrays["rgb01"].shape != (
        NUM_POINTS,
        3,
    ):
        raise ValueError("Base teacher xyz/RGB shape mismatch")
    if arrays["instance_ids"].shape != (NUM_POINTS,) or arrays[
        "source_indices"
    ].shape != (NUM_POINTS,):
        raise ValueError("Base teacher instance/source index shape mismatch")
    if not np.isfinite(arrays["xyz"]).all() or not np.isfinite(
        arrays["rgb01"]
    ).all():
        raise ValueError("Base teacher xyz/RGB contains NaN/Inf")
    if np.any(arrays["rgb01"] < 0.0) or np.any(arrays["rgb01"] > 1.0):
        raise ValueError("Base teacher RGB lies outside [0,1]")
    if np.any(arrays["instance_ids"] < 0) or np.any(
        arrays["source_indices"] < 0
    ):
        raise ValueError("Base teacher instance/source indices are negative")
    if len(np.unique(arrays["source_indices"])) != NUM_POINTS:
        raise ValueError("Base teacher source indices are not unique")
    if manifest.get("draw_count") != int(draws.shape[0]):
        raise ValueError("Base teacher draw count mismatch")
    recorded_seed_rows = manifest.get("draw_seeds", [])
    if not isinstance(recorded_seed_rows, list) or len(recorded_seed_rows) != int(
        draws.shape[0]
    ):
        raise ValueError("Base teacher draw seed count mismatch")
    for row in recorded_seed_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "draw_index",
            "initial_xT_seed",
            "reverse_noise_seed",
        }:
            raise ValueError("Base teacher draw seed row schema mismatch")
        _require_exact_int(row.get("draw_index"), "draw seed index", minimum=0)
        _require_exact_int(
            row.get("initial_xT_seed"), "draw initial seed", minimum=0
        )
        _require_exact_int(
            row.get("reverse_noise_seed"), "draw reverse seed", minimum=0
        )
    expected_seed_rows = []
    for draw_index in range(int(draws.shape[0])):
        initial_seed, reverse_seed = derive_draw_seeds(
            cache_key=str(manifest["cache_key"]),
            base_seed=cache_payload["base_seed"],
            draw_index=draw_index,
        )
        expected_seed_rows.append(
            {
                "draw_index": draw_index,
                "initial_xT_seed": initial_seed,
                "reverse_noise_seed": reverse_seed,
            }
        )
    if manifest.get("draw_seeds") != expected_seed_rows:
        raise ValueError("Base teacher draw seed derivation mismatch")
    for name, value in arrays.items():
        recorded = manifest.get("array_sha256", {}).get(name)
        if sha256_array(value) != recorded:
            raise ValueError(name + " array SHA-256 mismatch")
    if set(manifest.get("array_sha256", {})) != set(arrays):
        raise ValueError("Base teacher manifest array hash key set mismatch")
    scene_hash_payload = manifest.get("scene_hash_payload", {})
    scene_array_bindings = {
        "xyz": "xyz_array_sha256",
        "rgb01": "rgb01_array_sha256",
        "instance_ids": "instance_ids_sha256",
        "source_indices": "source_indices_sha256",
    }
    for array_name, scene_hash_name in scene_array_bindings.items():
        if sha256_array(arrays[array_name]) != scene_hash_payload.get(
            scene_hash_name
        ):
            raise ValueError(array_name + " differs from scene hash payload")
    expected_mean = draws.mean(axis=0, dtype=np.float64).astype(np.float32)
    expected_std = draws.std(axis=0, dtype=np.float64).astype(np.float32)
    if not np.array_equal(affordance, expected_mean):
        raise ValueError("Base teacher mean is not the exact mean of draws")
    if not np.array_equal(affordance_std, expected_std):
        raise ValueError("Base teacher std is not the exact std of draws")
    if not np.isfinite(draws).all() or np.any(draws < 0.0) or np.any(draws > 1.0):
        raise ValueError("Base teacher draws are invalid")
    if arrays["channel_names"].astype(str).tolist() != list(CHANNEL_ORDER):
        raise ValueError("NPZ channel_names mismatch")
    if manifest.get("affordance_range") != [
        float(affordance.min()),
        float(affordance.max()),
    ]:
        raise ValueError("Base teacher affordance range declaration mismatch")
    if manifest.get("draw_range") != [float(draws.min()), float(draws.max())]:
        raise ValueError("Base teacher draw range declaration mismatch")
    if arrays["channel_joint_indices"].tolist() != list(CHANNEL_JOINT_INDICES):
        raise ValueError("NPZ channel_joint_indices mismatch")
    draw_count = int(draws.shape[0])
    if arrays["draw_indices"].tolist() != list(range(draw_count)):
        raise ValueError("NPZ draw_indices mismatch")
    recorded_seeds = manifest.get("draw_seeds", [])
    if arrays["initial_noise_seeds"].tolist() != [
        int(row["initial_xT_seed"]) for row in recorded_seeds
    ]:
        raise ValueError("NPZ initial noise seeds mismatch")
    if arrays["reverse_noise_seeds"].tolist() != [
        int(row["reverse_noise_seed"]) for row in recorded_seeds
    ]:
        raise ValueError("NPZ reverse noise seeds mismatch")
    expected_artifact_id = canonical_json_sha256(
        {
            "schema": ARTIFACT_SCHEMA,
            "cache_key": str(manifest["cache_key"]),
            "affordance_sha256": sha256_array(arrays["affordance"]),
            "draws_sha256": sha256_array(arrays["affordance_draws"]),
            "scene_sha256": str(manifest["scene_sha256"]),
        }
    )
    if manifest.get("artifact_id") != expected_artifact_id:
        raise ValueError("Base teacher artifact ID mismatch")

    return {
        "status": "INTEGRITY_PASS",
        "schema": ARTIFACT_SCHEMA,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_id": str(manifest["artifact_id"]),
        "scene_id": str(manifest["scene_id"]),
        "prompt_id": str(manifest["prompt_id"]),
        "cache_key": str(manifest["cache_key"]),
        "draw_count": int(draws.shape[0]),
        "affordance_min": float(affordance.min()),
        "affordance_max": float(affordance.max()),
    }


__all__ = [
    "ALLOWED_REPRESENTATIONS",
    "ARTIFACT_SCHEMA",
    "AUDIT_SCHEMA",
    "AUDIT_QUALITY_THRESHOLDS",
    "CHANNEL_JOINT_INDICES",
    "CHANNEL_ORDER",
    "COMMON_QUALITY_THRESHOLDS",
    "DEVELOPMENT_SCHEMA",
    "DEVELOPMENT_QUALITY_THRESHOLDS",
    "NUM_CHANNELS",
    "NUM_POINTS",
    "PROMPT_BY_TARGET",
    "PROMPT_POLICY_ID",
    "REQUIRED_AUDIT_CHECKS",
    "REQUIRED_DEVELOPMENT_CHECKS",
    "REQUIRED_ROLLOUT_SELECTION_CHECKS",
    "ROLLOUT_PROTOCOL_FILES",
    "ROLLOUT_SELECTION_SCHEMA",
    "SHORTLIST_SCHEMA",
    "TARGET_INSTANCE_IDS",
    "TRAIN_SCHEMA",
    "atomic_write_json",
    "build_cache_key_payload",
    "canonical_json_sha256",
    "convert_prediction_draws",
    "derive_draw_seeds",
    "load_contact_stats",
    "load_json",
    "load_quality_replay_row",
    "load_scene_contract",
    "rollout_draw_seeds",
    "sha256_array",
    "sha256_file",
    "sha256_text",
    "validate_base_teacher_artifact",
    "validate_prompt",
    "validate_v5r4_quality_chain",
    "write_base_teacher_artifact",
]

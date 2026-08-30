#!/usr/bin/env python3
"""Shared, leakage-safe contracts for few-shot CDM training/evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


CONTACT_JOINTS = np.asarray([0, 10, 11, 12, 20, 21], dtype=np.int64)
CONTACT_JOINT_NAMES = np.asarray(
    ["pelvis", "left_foot", "right_foot", "neck", "left_wrist", "right_wrist"]
)
EXPECTED_TARGETS = ("chair", "bed", "whiteboard")
DIAGNOSTIC_IDS = {
    "room_0003_lie_bed_0001",
    "room_0004_interact_whiteboard_0001",
}


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    """Write one NPZ completely before making it visible at its final path."""

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


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a JSON object after flushing it to stable storage."""

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
            handle.write(json.dumps(value, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> MutableMapping[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_split(split_file: Path) -> MutableMapping[str, object]:
    split = read_json(split_file)
    if split.get("schema") != "affordance_source_disjoint_split_v1":
        raise ValueError(f"{split_file}: unsupported split schema")
    checks = split.get("checks", {})
    required = (
        "source_components_disjoint",
        "original_motion_ids_disjoint",
        "multistart_excluded_from_cdm",
        "object_names_absent_from_new_prompts",
    )
    if not isinstance(checks, dict) or not all(bool(checks.get(k)) for k in required):
        raise RuntimeError(f"{split_file}: split safety checks are not all true")
    cdm = split.get("cdm_fewshot")
    if not isinstance(cdm, dict):
        raise ValueError(f"{split_file}: missing cdm_fewshot")
    train_ids = [str(v) for v in cdm.get("train", [])]
    test_ids = [str(v) for v in cdm.get("test", [])]
    if not train_ids or not test_ids:
        raise ValueError(f"{split_file}: empty CDM train/test partition")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"CDM train/test overlap: {sorted(overlap)}")
    if DIAGNOSTIC_IDS & (set(train_ids) | set(test_ids)):
        raise AssertionError("one-sample diagnostic IDs entered production split")
    sample_meta = {
        str(row["sample_id"]): row for row in split.get("samples", [])
    }
    for sample_id in train_ids + test_ids:
        row = sample_meta.get(sample_id)
        if row is None:
            raise ValueError(f"{split_file}: missing metadata for {sample_id}")
        if str(row.get("stage")) != "base":
            raise AssertionError(f"{sample_id}: non-base sample entered CDM split")
        expected = "train" if sample_id in train_ids else "test"
        if str(row.get("split")) != expected:
            raise AssertionError(f"{sample_id}: split metadata mismatch")
    return split


def load_index_entries(
    dataset_root: Path, sample_ids: Iterable[str]
) -> Dict[str, Dict[str, object]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    requested = {str(value) for value in sample_ids}
    scenes = sorted({sample_id.split("_")[1] for sample_id in requested})
    entries: Dict[str, Dict[str, object]] = {}
    for scene_number in scenes:
        scene_id = "room_" + scene_number
        index_file = dataset_root / f"index_{scene_id}.json"
        index = read_json(index_file)
        if str(index.get("scene_id")) != scene_id:
            raise ValueError(f"{index_file}: scene_id mismatch")
        for raw_entry in index.get("samples", []):
            sample_id = str(raw_entry["sample_id"])
            if sample_id not in requested:
                continue
            entry = dict(raw_entry)
            entry["_index_file"] = str(index_file)
            entry["_scene_adm_input"] = str(index["scene_adm_input"])
            entries[sample_id] = entry
    missing = requested - set(entries)
    if missing:
        raise KeyError(f"prepared index entries missing: {sorted(missing)}")
    return entries


def load_scene(dataset_root: Path, entry: Mapping[str, object]) -> Dict[str, np.ndarray]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    scene_dir = dataset_root / str(entry["_scene_adm_input"])
    points_file = scene_dir / "points.npz"
    sidecar_file = scene_dir / "sidecar.npz"
    points = np.load(points_file, allow_pickle=False)["points"].astype(np.float32)
    sidecar = np.load(sidecar_file, allow_pickle=False)
    xyz = sidecar["xyz_afford_z_up"].astype(np.float32)
    instance_ids = sidecar["instance_ids"].astype(np.int64)
    source_indices = sidecar["source_indices"].astype(np.int64)
    candidate_mask = sidecar["candidate_mask"].astype(bool)
    if points.shape != (8192, 6) or xyz.shape != (8192, 3):
        raise ValueError(f"{scene_dir}: expected 8192 scene points")
    if instance_ids.shape != (8192,) or source_indices.shape != (8192,):
        raise ValueError(f"{scene_dir}: sidecar vector shape mismatch")
    if candidate_mask.shape != (8192, 3):
        raise ValueError(f"{scene_dir}: candidate_mask shape mismatch")
    if not np.array_equal(points[:, :3], xyz):
        if not np.allclose(points[:, :3], xyz, atol=1e-6):
            raise ValueError(f"{scene_dir}: points/sidecar point order differs")
    if not np.isfinite(points).all():
        raise ValueError(f"{scene_dir}: scene contains NaN/Inf")
    return {
        "points": points,
        "xyz": xyz,
        "instance_ids": instance_ids,
        "source_indices": source_indices,
        "candidate_mask": candidate_mask,
    }


def sample_dir_from_entry(dataset_root: Path, entry: Mapping[str, object]) -> Path:
    manifest = Path(dataset_root) / str(entry["sample_manifest"])
    return manifest.resolve().parent


def gt_file_from_entry(dataset_root: Path, entry: Mapping[str, object]) -> Path:
    return sample_dir_from_entry(dataset_root, entry) / "affordance_gt/full_affordance_gt.npz"


def load_motion_xyz(dataset_root: Path, entry: Mapping[str, object]) -> np.ndarray:
    path = Path(dataset_root) / str(entry["cmdm_motion_input"])
    motion = np.load(path, allow_pickle=False)
    key = "joint_positions22_adm_chair_local_z_up"
    if key not in motion:
        raise KeyError(f"{path}: missing {key}")
    value = motion[key].astype(np.float32)
    valid_frames = int(entry["valid_frames"])
    if value.ndim != 3 or value.shape[1:] != (22, 3):
        raise ValueError(f"{path}: motion shape must be [T,22,3]")
    if value.shape[0] < valid_frames:
        raise ValueError(f"{path}: fewer motion frames than index")
    value = value[:valid_frames]
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: motion contains NaN/Inf")
    return value


def compute_distance_map(
    scene_xyz: np.ndarray,
    motion_xyz: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    if scene_xyz.ndim != 2 or scene_xyz.shape[1] != 3:
        raise ValueError("scene_xyz must be [N,3]")
    if motion_xyz.ndim != 3 or motion_xyz.shape[1:] != (22, 3):
        raise ValueError("motion_xyz must be [T,22,3]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    distance = np.empty((scene_xyz.shape[0], 6), dtype=np.float32)
    for output_channel, joint_index in enumerate(CONTACT_JOINTS):
        trajectory = motion_xyz[:, joint_index, :]
        for start in range(0, scene_xyz.shape[0], chunk_size):
            end = min(start + chunk_size, scene_xyz.shape[0])
            delta = scene_xyz[start:end, None, :] - trajectory[None, :, :]
            distance[start:end, output_channel] = np.linalg.norm(
                delta, axis=-1
            ).min(axis=1)
    return distance


def distance_to_affordance(distance: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise ValueError("invalid distance map")
    return np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)


def load_stats(stats_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    stats = np.load(stats_file, allow_pickle=False)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    if mean.shape != (1, 6) or std.shape != (1, 6):
        raise ValueError("contact statistics must both be [1,6]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("contact statistics contain NaN/Inf")
    if np.any(std <= 0.0):
        raise ValueError("contact statistics contain non-positive std")
    return mean, std


def normalize_contact(contact: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if contact.shape[-1] != 6:
        raise ValueError("contact must have six channels")
    return ((contact - mean) / std).astype(np.float32)


def denormalize_contact(contact: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if contact.shape[-1] != 6:
        raise ValueError("contact must have six channels")
    return np.clip(contact * std + mean, 1e-20, 1.0).astype(np.float32)


def target_balanced_batches(
    sample_rows: Sequence[Mapping[str, object]],
    steps: int,
    seed: int,
    replay_counts: Mapping[str, int] | None = None,
) -> List[List[str]]:
    """Deterministic target replay with replacement across shuffled cycles.

    ``replay_counts=None`` preserves the v1 one-per-target contract.  The v3
    few-shot stage passes ``{"chair": 3, "bed": 1, "whiteboard": 1}`` so
    Chair occupies 60% of every optimization batch without ever reading the
    held-out partition.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if replay_counts is None:
        replay_counts = {target: 1 for target in EXPECTED_TARGETS}
    replay_counts = {str(key): int(value) for key, value in replay_counts.items()}
    if set(replay_counts) != set(EXPECTED_TARGETS):
        raise ValueError(f"replay_counts must have exactly {EXPECTED_TARGETS}")
    if any(replay_counts[target] <= 0 for target in EXPECTED_TARGETS):
        raise ValueError("all replay counts must be positive")
    by_target: Dict[str, List[str]] = defaultdict(list)
    for row in sample_rows:
        by_target[str(row["target"])].append(str(row["sample_id"]))
    if tuple(sorted(by_target)) != tuple(sorted(EXPECTED_TARGETS)):
        raise ValueError(f"training targets differ from {EXPECTED_TARGETS}")
    if any(not by_target[target] for target in EXPECTED_TARGETS):
        raise ValueError("each target must have a training sample")
    rng = random.Random(seed)
    positions = {target: 0 for target in EXPECTED_TARGETS}
    for values in by_target.values():
        values.sort()
        rng.shuffle(values)
    batches: List[List[str]] = []
    for _ in range(steps):
        batch = []
        for target in EXPECTED_TARGETS:
            for _ in range(replay_counts[target]):
                values = by_target[target]
                position = positions[target]
                if position >= len(values):
                    rng.shuffle(values)
                    position = 0
                batch.append(values[position])
                positions[target] = position + 1
        batches.append(batch)
    return batches


def checkpoint_selection_gate(
    initial_grid: Mapping[str, object],
    candidate_grid: Mapping[str, object],
    novel_min_relative_improvement: float,
    chair_max_relative_degradation: float,
) -> Dict[str, object]:
    """Pure train-only target loss gate used for the v5 one-step shortlist."""
    initial = {
        str(key): float(value)
        for key, value in initial_grid["per_target_mean"].items()
    }
    candidate = {
        str(key): float(value)
        for key, value in candidate_grid["per_target_mean"].items()
    }
    if set(initial) != {"chair", "bed", "whiteboard"} or set(candidate) != set(initial):
        raise ValueError("checkpoint grid must contain Chair, Bed, and Whiteboard")
    relative_improvement = {
        target: (initial[target] - candidate[target]) / max(initial[target], 1e-12)
        for target in initial
    }
    chair_degradation = -relative_improvement["chair"]
    checks = {
        "bed_train_grid_improved": (
            relative_improvement["bed"] >= novel_min_relative_improvement
        ),
        "whiteboard_train_grid_improved": (
            relative_improvement["whiteboard"] >= novel_min_relative_improvement
        ),
        "chair_train_replay_retained": (
            chair_degradation <= chair_max_relative_degradation
        ),
    }
    rank = [
        min(relative_improvement["bed"], relative_improvement["whiteboard"]),
        0.5 * (relative_improvement["bed"] + relative_improvement["whiteboard"]),
        -max(chair_degradation, 0.0),
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_improvement": relative_improvement,
        "chair_relative_degradation": chair_degradation,
        "rank": rank,
    }


def build_sparse_contact_weights(
    affordance: np.ndarray,
    instance_ids: np.ndarray,
    target_instance_id: int,
    target: str,
    target_instance_weight: float,
    target_foreground_weight: float,
    active_threshold: float = 0.7,
) -> np.ndarray:
    """Build a literal point/channel loss weight map for sparse novel contact.

    Chair replay deliberately keeps the legacy uniform MSE.  Bed and
    Whiteboard keep weight one everywhere, add a moderate weight to all
    channels of the target furniture, and add a stronger weight only where
    the motion-derived GT is active.  Therefore background still contributes,
    but cannot numerically drown the sparse target interaction.
    """
    affordance = np.asarray(affordance, dtype=np.float32)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    if affordance.shape != (8192, 6):
        raise ValueError("affordance must be [8192,6]")
    if instance_ids.shape != (8192,):
        raise ValueError("instance_ids must be [8192]")
    if target not in EXPECTED_TARGETS:
        raise ValueError(f"unknown target: {target}")
    if target_instance_weight < 0.0 or target_foreground_weight < 0.0:
        raise ValueError("sparse loss weights must be non-negative")
    if not 0.0 < active_threshold < 1.0:
        raise ValueError("active_threshold must be in (0,1)")
    if not np.isfinite(affordance).all():
        raise ValueError("affordance contains NaN/Inf")

    weights = np.ones_like(affordance, dtype=np.float32)
    if target == "chair":
        return weights

    target_points = instance_ids == int(target_instance_id)
    if not target_points.any():
        raise ValueError(f"target instance {target_instance_id} has no points")
    weights[target_points, :] += np.float32(target_instance_weight)
    active = target_points[:, None] & (affordance >= active_threshold)
    if not active.any():
        raise ValueError(
            f"{target}: no active GT contact inside target instance at "
            f"threshold {active_threshold}"
        )
    weights[active] += np.float32(target_foreground_weight)
    if not np.isfinite(weights).all() or np.any(weights < 1.0):
        raise ValueError("invalid sparse contact weights")
    return weights


def semantic_checkpoint_gate(
    semantic: Mapping[str, object],
    novel_min_f1: float,
    novel_min_dominance_rate: float,
    chair_min_dominance_rate: float,
    sit_bed_candidate_min_rate: float,
) -> Dict[str, object]:
    """Train-only semantic gate used in addition to diffusion loss.

    This gate prevents selecting a low-amplitude checkpoint merely because its
    average MSE is small.  It never consumes held-out rows.
    """
    per_target_raw = semantic.get("per_target")
    if not isinstance(per_target_raw, Mapping):
        raise ValueError("semantic metrics must contain per_target")
    per_target = {
        str(name): dict(value) for name, value in per_target_raw.items()
    }
    if set(per_target) != set(EXPECTED_TARGETS):
        raise ValueError("semantic metrics must contain all three targets")
    checks = {
        "bed_train_f1_active": (
            float(per_target["bed"]["f1_at_0_7"]) >= novel_min_f1
        ),
        "whiteboard_train_f1_active": (
            float(per_target["whiteboard"]["f1_at_0_7"]) >= novel_min_f1
        ),
        "chair_train_any_joint_dominance": (
            float(per_target["chair"]["any_joint_dominance_rate"])
            >= chair_min_dominance_rate
        ),
        "chair_train_pelvis_dominance": (
            float(per_target["chair"]["pelvis_dominance_rate"])
            >= chair_min_dominance_rate
        ),
        "bed_train_any_joint_dominance": (
            float(per_target["bed"]["any_joint_dominance_rate"])
            >= novel_min_dominance_rate
        ),
        "bed_train_pelvis_dominance": (
            float(per_target["bed"]["pelvis_dominance_rate"])
            >= novel_min_dominance_rate
        ),
        "whiteboard_train_any_joint_dominance": (
            float(per_target["whiteboard"]["any_joint_dominance_rate"])
            >= novel_min_dominance_rate
        ),
        "whiteboard_train_right_wrist_dominance": (
            float(per_target["whiteboard"]["right_wrist_dominance_rate"])
            >= novel_min_dominance_rate
        ),
        "sit_train_chair_remains_primary": (
            float(per_target["chair"]["sit_chair_primary_rate"])
            >= chair_min_dominance_rate
        ),
        "sit_train_bed_candidate_is_visible_and_bounded": (
            float(per_target["chair"]["sit_bed_candidate_visible_rate"])
            >= sit_bed_candidate_min_rate
        ),
    }
    rank = [
        min(
            float(per_target["bed"]["f1_at_0_7"]),
            float(per_target["whiteboard"]["f1_at_0_7"]),
        ),
        min(
            float(per_target["bed"]["any_joint_dominance_rate"]),
            float(per_target["whiteboard"]["any_joint_dominance_rate"]),
        ),
        float(per_target["chair"]["any_joint_dominance_rate"]),
        float(per_target["whiteboard"]["right_wrist_dominance_rate"]),
        float(per_target["chair"]["sit_bed_candidate_visible_rate"]),
    ]
    return {"passed": all(checks.values()), "checks": checks, "rank": rank}


def target_name_for_entry(entry: Mapping[str, object]) -> str:
    value = str(entry.get("target_name", "")).lower()
    if value:
        return value
    index = int(entry["target_index"])
    if index not in (0, 1, 2):
        raise ValueError(f"invalid target index: {index}")
    return EXPECTED_TARGETS[index]


def top_fraction_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    count = max(1, int(np.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def instance_scores(
    affordance: np.ndarray, instance_ids: np.ndarray, channel: str
) -> Dict[str, float]:
    if affordance.shape != (8192, 6):
        raise ValueError("affordance must be [8192,6]")
    if channel == "pelvis":
        scalar = affordance[:, 0]
    elif channel == "any_joint":
        scalar = affordance.max(axis=1)
    elif channel == "right_wrist":
        scalar = affordance[:, 5]
    elif channel == "left_wrist":
        scalar = affordance[:, 4]
    else:
        raise ValueError(
            "channel must be pelvis, any_joint, right_wrist, or left_wrist"
        )
    return {
        name: top_fraction_mean(scalar[instance_ids == instance_id])
        for instance_id, name in enumerate(
            ("environment", "chair", "bed", "whiteboard", "tv")
        )
    }

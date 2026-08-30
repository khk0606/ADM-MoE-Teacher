#!/usr/bin/env python3
"""Generate phase-level point-aligned IIW targets for prepared AMDM data.

This script transfers the distance kernel used by InterFaceRays IIW to the
fixed 8,192-point ADM scene representation.  The result is deliberately named
``point_aligned_iiw_proxy``: original InterFaceRays IIW is defined on 540
character-centric ray-hit points that change every frame, while this script
uses one fixed scene point cloud shared by all samples in a scene.

Expected input (relative to ``--dataset-root``):

* ``index_<scene_id>.json``;
* ``scenes/<scene_id>/adm_input/sidecar.npz``;
* each sample's prepared ``cmdm_motion_input.npz``.

Each output contains both the native IIW body-part order and the deterministic
CMDM proxy order.  The native tensor is the supervision target for a future
``IIWPlanner``.  The proxy is only for the first Oracle-IIW integration test;
``IIWAdapter`` will later learn this native-to-CMDM conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


METHOD_NAME = "point_aligned_iiw_proxy"
METHOD_VERSION = 1
DEFAULT_NUM_PHASES = 8
DEFAULT_CHARACTER_HEIGHT_M = 1.8
DEFAULT_LOWER_BOUND_M = 0.1
DEFAULT_POINT_CHUNK_SIZE = 1024
EXPECTED_CMDM_HORIZON = 196

NATIVE_BODY_PART_NAMES = (
    "base",
    "spine",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
)

# SMPL/HumanML 22-joint indices.  The weight kernel is evaluated independently
# at every indicator joint and then averaged inside a body group, matching the
# nonlinear order used by the Unity implementation.
NATIVE_BODY_JOINT_GROUPS: Mapping[str, Tuple[int, ...]] = {
    "base": (0, 1, 2),                 # pelvis, left hip, right hip
    # Unity's serialized IIW sensor is YBot ``Spine2``.  In the production
    # YBot25 -> HumanML22 conversion that source bone maps to HML22 index 9
    # (named ``spine3`` in the prepared tensor contract).
    "spine": (9,),
    "right_hand": (19, 21),            # right elbow, right wrist
    "left_hand": (18, 20),             # left elbow, left wrist
    "right_foot": (8, 11),             # right ankle, right foot/toe
    "left_foot": (7, 10),              # left ankle, left foot/toe
}

# Native [base, spine, RH, LH, RF, LF] -> CMDM contact slots
# [pelvis, LF, RF, neck, LH, RH].  This is a semantic proxy, not an equality:
# base != pelvis and spine != neck.  The manifest records that limitation.
CMDM_PROXY_FROM_NATIVE = np.asarray([0, 5, 4, 1, 3, 2], dtype=np.int64)
CMDM_CONTACT_JOINT_INDICES = np.asarray(
    [0, 10, 11, 12, 20, 21], dtype=np.int64
)
CMDM_PROXY_CHANNEL_SOURCES = (
    "base_as_pelvis",
    "left_foot",
    "right_foot",
    "spine_as_neck",
    "left_hand_as_left_wrist",
    "right_hand_as_right_wrist",
)


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_prefix_mask(mask: np.ndarray, valid_frames: int) -> None:
    if mask.ndim != 1:
        raise ValueError(f"x_mask must be one-dimensional, got {mask.shape}")
    if mask.shape != (EXPECTED_CMDM_HORIZON,):
        raise ValueError(
            f"x_mask must have the production CMDM horizon "
            f"({EXPECTED_CMDM_HORIZON},), got {mask.shape}"
        )
    if valid_frames <= 0 or valid_frames > len(mask):
        raise ValueError(f"invalid valid-frame count: {valid_frames}")
    expected = np.ones_like(mask, dtype=bool)
    expected[:valid_frames] = False
    if not np.array_equal(mask.astype(bool), expected):
        raise ValueError(
            "x_mask must be prefix-contiguous with False=valid and True=padding"
        )


def phase_ranges(valid_frames: int, num_phases: int) -> List[Tuple[int, int]]:
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")
    if valid_frames < num_phases:
        raise ValueError(
            f"valid_frames={valid_frames} must be >= num_phases={num_phases}"
        )
    result = []
    for phase_index in range(num_phases):
        start = phase_index * valid_frames // num_phases
        end = (phase_index + 1) * valid_frames // num_phases
        if end <= start:
            raise AssertionError("empty IIW phase")
        result.append((start, end))
    if result[0][0] != 0 or result[-1][1] != valid_frames:
        raise AssertionError("phase ranges do not cover every valid frame")
    return result


def distance_kernel(
    distance_m: np.ndarray,
    lower_bound_m: float,
    upper_bound_m: float,
) -> np.ndarray:
    """Exact piecewise kernel used by Unity CalcRelativeWeightsWithBounds."""
    distance = np.asarray(distance_m, dtype=np.float32)
    if not np.isfinite(distance).all():
        raise ValueError("distance contains NaN/Inf")
    if lower_bound_m <= 0.0 or upper_bound_m <= lower_bound_m:
        raise ValueError("distance bounds must satisfy 0 < lower < upper")

    middle = (lower_bound_m + upper_bound_m) / (
        distance + upper_bound_m
    )
    weight = np.where(
        distance < lower_bound_m,
        1.0,
        np.where(distance <= upper_bound_m, middle, 0.0),
    )
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def padded_body_joint_groups() -> Tuple[np.ndarray, np.ndarray]:
    groups = [NATIVE_BODY_JOINT_GROUPS[name] for name in NATIVE_BODY_PART_NAMES]
    max_group_size = max(len(group) for group in groups)
    padded = np.full(
        (len(groups), max_group_size), -1, dtype=np.int64
    )
    counts = np.zeros((len(groups),), dtype=np.int64)
    for group_index, group in enumerate(groups):
        padded[group_index, : len(group)] = np.asarray(group, dtype=np.int64)
        counts[group_index] = len(group)
    return padded, counts


def compute_phase_iiw(
    joint_positions22: np.ndarray,
    scene_xyz: np.ndarray,
    num_phases: int,
    lower_bound_m: float,
    upper_bound_m: float,
    point_chunk_size: int,
    save_framewise: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute native-order IIW with bounded point-memory usage.

    Returns:
        native_phase_max: [Q,N,6]
        native_phase_mean: [Q,N,6]
        frame_to_phase: [T]
        native_frame: optional [T,N,6]
    """
    joints = np.asarray(joint_positions22, dtype=np.float32)
    points = np.asarray(scene_xyz, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"joint positions must be [T,22,3], got {joints.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"scene_xyz must be [N,3], got {points.shape}")
    if len(joints) == 0 or len(points) == 0:
        raise ValueError("motion and scene must not be empty")
    if not np.isfinite(joints).all() or not np.isfinite(points).all():
        raise ValueError("motion/scene contains NaN/Inf")
    if point_chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")

    ranges = phase_ranges(len(joints), num_phases)
    num_points = len(points)
    num_parts = len(NATIVE_BODY_PART_NAMES)
    phase_max = np.zeros(
        (num_phases, num_points, num_parts), dtype=np.float32
    )
    phase_mean = np.zeros_like(phase_max)
    framewise = (
        np.zeros((len(joints), num_points, num_parts), dtype=np.float32)
        if save_framewise
        else None
    )
    frame_to_phase = np.empty((len(joints),), dtype=np.int64)

    for phase_index, (frame_start, frame_end) in enumerate(ranges):
        frame_to_phase[frame_start:frame_end] = phase_index
        phase_joints = joints[frame_start:frame_end]
        for point_start in range(0, num_points, point_chunk_size):
            point_end = min(point_start + point_chunk_size, num_points)
            point_chunk = points[None, None, point_start:point_end, :]
            part_weights = []
            for part_name in NATIVE_BODY_PART_NAMES:
                group = NATIVE_BODY_JOINT_GROUPS[part_name]
                group_joints = phase_joints[:, group, None, :]
                distance = np.linalg.norm(
                    group_joints - point_chunk, axis=-1
                )  # [Tp,G,P]
                joint_weights = distance_kernel(
                    distance, lower_bound_m, upper_bound_m
                )
                part_weights.append(joint_weights.mean(axis=1))  # [Tp,P]
            weights = np.stack(part_weights, axis=-1)  # [Tp,P,6]
            phase_max[
                phase_index, point_start:point_end
            ] = weights.max(axis=0)
            phase_mean[
                phase_index, point_start:point_end
            ] = weights.mean(axis=0)
            if framewise is not None:
                framewise[
                    frame_start:frame_end, point_start:point_end
                ] = weights

    if not np.array_equal(
        np.bincount(frame_to_phase, minlength=num_phases),
        np.asarray([end - start for start, end in ranges], dtype=np.int64),
    ):
        raise AssertionError("frame-to-phase assignment mismatch")
    for name, value in (
        ("phase max", phase_max),
        ("phase mean", phase_mean),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf")
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(f"{name} lies outside [0,1]")
    return phase_max, phase_mean, frame_to_phase, framewise


def load_scene(index: Dict, dataset_root: Path) -> Dict[str, np.ndarray | Path]:
    scene_dir = dataset_root / index["scene_adm_input"]
    sidecar_file = scene_dir / "sidecar.npz"
    if not sidecar_file.is_file():
        raise FileNotFoundError(sidecar_file)
    sidecar = np.load(sidecar_file, allow_pickle=False)
    scene_xyz = sidecar["xyz_afford_z_up"].astype(np.float32)
    instance_ids = sidecar["instance_ids"].astype(np.int64)
    source_indices = sidecar["source_indices"].astype(np.int64)
    candidate_mask = sidecar["candidate_mask"].astype(bool)
    if scene_xyz.shape != (8192, 3):
        raise ValueError(f"scene XYZ must be (8192,3), got {scene_xyz.shape}")
    if instance_ids.shape != (8192,) or source_indices.shape != (8192,):
        raise ValueError("scene point sidecar shape mismatch")
    if candidate_mask.shape != (8192, 3):
        raise ValueError("candidate mask must be (8192,3)")
    if not np.isfinite(scene_xyz).all():
        raise ValueError("scene XYZ contains NaN/Inf")
    return {
        "sidecar_file": sidecar_file,
        "scene_xyz": scene_xyz,
        "instance_ids": instance_ids,
        "source_indices": source_indices,
        "candidate_mask": candidate_mask,
    }


def generate_sample(
    entry: Dict,
    dataset_root: Path,
    scene: Dict[str, np.ndarray | Path],
    num_phases: int,
    character_height_m: float,
    lower_bound_m: float,
    point_chunk_size: int,
    save_framewise: bool,
    overwrite: bool,
) -> Tuple[Dict, Dict]:
    sample_id = str(entry["sample_id"])
    motion_file = dataset_root / entry["cmdm_motion_input"]
    if not motion_file.is_file():
        raise FileNotFoundError(motion_file)
    motion = np.load(motion_file, allow_pickle=False)
    joints = motion["joint_positions22_adm_chair_local_z_up"].astype(
        np.float32
    )
    mask = motion["x_mask"].astype(bool)
    valid_frames = int(entry["valid_frames"])
    validate_prefix_mask(mask, valid_frames)
    if joints.shape != (valid_frames, 22, 3):
        raise ValueError(
            f"{sample_id}: joint positions {joints.shape} != "
            f"({valid_frames},22,3)"
        )

    upper_bound_m = lower_bound_m + 0.25 * character_height_m
    phase_max, phase_mean, frame_to_phase, framewise = compute_phase_iiw(
        joint_positions22=joints,
        scene_xyz=scene["scene_xyz"],
        num_phases=num_phases,
        lower_bound_m=lower_bound_m,
        upper_bound_m=upper_bound_m,
        point_chunk_size=point_chunk_size,
        save_framewise=save_framewise,
    )
    cmdm_phase_max = phase_max[..., CMDM_PROXY_FROM_NATIVE]
    cmdm_phase_mean = phase_mean[..., CMDM_PROXY_FROM_NATIVE]
    ranges = phase_ranges(valid_frames, num_phases)
    phase_frame_counts = np.asarray(
        [end - start for start, end in ranges], dtype=np.int64
    )
    phase_mask = np.zeros((num_phases,), dtype=bool)
    body_joint_groups, body_group_counts = padded_body_joint_groups()

    out_dir = dataset_root / "samples" / sample_id / "iiw_gt"
    out_file = out_dir / "iiw_gt.npz"
    manifest_file = out_dir / "manifest.json"
    if (out_file.exists() or manifest_file.exists()) and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists; pass --overwrite to replace it"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "iiw_native_phase_max": phase_max.astype(np.float16),
        "iiw_native_phase_mean": phase_mean.astype(np.float16),
        "iiw_cmdm_proxy_phase_max": cmdm_phase_max.astype(np.float16),
        "iiw_cmdm_proxy_phase_mean": cmdm_phase_mean.astype(np.float16),
        "phase_mask": phase_mask,
        "phase_frame_counts": phase_frame_counts,
        "frame_to_phase": frame_to_phase,
        "valid_frames": np.asarray(valid_frames, dtype=np.int64),
        "scene_xyz_adm": np.asarray(scene["scene_xyz"], dtype=np.float32),
        "instance_ids": np.asarray(scene["instance_ids"], dtype=np.int64),
        "source_indices": np.asarray(scene["source_indices"], dtype=np.int64),
        "body_joint_groups": body_joint_groups,
        "body_group_counts": body_group_counts,
        "body_part_names": np.asarray(NATIVE_BODY_PART_NAMES),
        "cmdm_contact_joint_indices": CMDM_CONTACT_JOINT_INDICES,
        "cmdm_proxy_from_native": CMDM_PROXY_FROM_NATIVE,
        "cmdm_proxy_channel_sources": np.asarray(CMDM_PROXY_CHANNEL_SOURCES),
        "lower_bound_m": np.asarray(lower_bound_m, dtype=np.float32),
        "upper_bound_m": np.asarray(upper_bound_m, dtype=np.float32),
        "character_height_m": np.asarray(
            character_height_m, dtype=np.float32
        ),
    }
    if framewise is not None:
        arrays["iiw_native_frame"] = framewise.astype(np.float16)
        arrays["iiw_cmdm_proxy_frame"] = framewise[
            ..., CMDM_PROXY_FROM_NATIVE
        ].astype(np.float16)
    np.savez_compressed(out_file, **arrays)

    high_intensity_fraction = float(np.mean(phase_max >= 0.7))
    manifest = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "sample_id": sample_id,
        "scene_id": str(entry["scene_id"]),
        "coordinate_frame": "chair_local_z_up",
        "source_motion_input": str(motion_file.resolve()),
        "source_scene_sidecar": str(Path(scene["sidecar_file"]).resolve()),
        "output_file": str(out_file.resolve()),
        "representation": "phase_level_point_aligned_iiw_proxy",
        "approximation_disclosure": (
            "Uses the InterFaceRays distance kernel on fixed ADM scene points; "
            "it is not original 540-ray character-centric InterFaceRays IIW. "
            "Unity Bodypart_Contact_Label and contacted-furniture masking are "
            "not available in the prepared CMDM tensor and are not applied."
        ),
        "target_semantics": "continuous_distance_kernel_without_contact_mask",
        "unity_contact_label_masking_applied": False,
        "num_phases": num_phases,
        "valid_frames": valid_frames,
        "phase_frame_ranges": [list(value) for value in ranges],
        "phase_frame_counts": phase_frame_counts.tolist(),
        "fps": 20.0,
        "num_scene_points": int(len(scene["scene_xyz"])),
        "native_body_part_order": list(NATIVE_BODY_PART_NAMES),
        "native_body_joint_groups_smpl22": {
            name: list(NATIVE_BODY_JOINT_GROUPS[name])
            for name in NATIVE_BODY_PART_NAMES
        },
        "cmdm_proxy_channel_sources": list(CMDM_PROXY_CHANNEL_SOURCES),
        "cmdm_proxy_from_native": CMDM_PROXY_FROM_NATIVE.tolist(),
        "cmdm_proxy_disclosure": (
            "base->pelvis and spine->neck are semantic proxies; IIWAdapter "
            "must replace this deterministic mapping in the learned model."
        ),
        "character_height_m": character_height_m,
        "lower_bound_m": lower_bound_m,
        "upper_bound_m": upper_bound_m,
        "temporal_reductions": ["max", "mean"],
        "primary_planner_target": "iiw_native_phase_max",
        "storage_dtype": "float16",
        "loss_dtype": "float32",
        "scene_source_indices_sha256": sha256_array(
            np.asarray(scene["source_indices"], dtype=np.int64)
        ),
        "native_phase_min": float(phase_max.min()),
        "native_phase_max": float(phase_max.max()),
        "native_phase_mean": float(phase_max.mean()),
        "high_intensity_threshold_diagnostic_only": 0.7,
        "high_intensity_fraction": high_intensity_fraction,
        "framewise_saved": bool(save_framewise),
        "iiw_gt_ready": True,
    }
    write_json(manifest_file, manifest)

    updated_entry = dict(entry)
    updated_entry.update(
        {
            "iiw_gt": relative_to_root(out_file, dataset_root),
            "iiw_gt_manifest": relative_to_root(
                manifest_file, dataset_root
            ),
            "iiw_gt_ready": True,
        }
    )
    print(
        f"[PASS] {sample_id}: frames={valid_frames}, phases={num_phases}, "
        f"range={phase_max.min():.6f}..{phase_max.max():.6f}, "
        f"active@0.7={high_intensity_fraction:.6f}"
    )
    return updated_entry, manifest


def validate_saved_sample(
    entry: Dict,
    dataset_root: Path,
    scene: Dict[str, np.ndarray | Path],
    num_phases: int,
) -> None:
    sample_id = str(entry["sample_id"])
    target_file = dataset_root / entry["iiw_gt"]
    manifest_file = dataset_root / entry["iiw_gt_manifest"]
    if not target_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"{sample_id}: IIW target files are missing")
    target = np.load(target_file, allow_pickle=False)
    required_shapes = {
        "iiw_native_phase_max": (num_phases, 8192, 6),
        "iiw_native_phase_mean": (num_phases, 8192, 6),
        "iiw_cmdm_proxy_phase_max": (num_phases, 8192, 6),
        "iiw_cmdm_proxy_phase_mean": (num_phases, 8192, 6),
        "phase_mask": (num_phases,),
        "scene_xyz_adm": (8192, 3),
        "instance_ids": (8192,),
        "source_indices": (8192,),
    }
    for key, shape in required_shapes.items():
        if key not in target:
            raise ValueError(f"{sample_id}: missing IIW key {key}")
        if target[key].shape != shape:
            raise ValueError(
                f"{sample_id}: {key} has {target[key].shape}, expected {shape}"
            )
    if not np.array_equal(target["source_indices"], scene["source_indices"]):
        raise ValueError(f"{sample_id}: IIW/scene point order mismatch")
    if not np.array_equal(target["instance_ids"], scene["instance_ids"]):
        raise ValueError(f"{sample_id}: IIW/scene instance order mismatch")
    if not np.array_equal(target["scene_xyz_adm"], scene["scene_xyz"]):
        raise ValueError(f"{sample_id}: IIW/scene XYZ mismatch")
    native = target["iiw_native_phase_max"].astype(np.float32)
    proxy = target["iiw_cmdm_proxy_phase_max"].astype(np.float32)
    if not np.isfinite(native).all() or not np.isfinite(proxy).all():
        raise ValueError(f"{sample_id}: saved IIW contains NaN/Inf")
    if np.any(native < 0.0) or np.any(native > 1.0):
        raise ValueError(f"{sample_id}: saved IIW outside [0,1]")
    if not np.array_equal(proxy, native[..., CMDM_PROXY_FROM_NATIVE]):
        raise ValueError(f"{sample_id}: CMDM proxy channel permutation mismatch")
    manifest = load_json(manifest_file)
    if manifest.get("method") != METHOD_NAME:
        raise ValueError(f"{sample_id}: IIW method mismatch")
    if not bool(manifest.get("iiw_gt_ready", False)):
        raise ValueError(f"{sample_id}: IIW manifest is not ready")


def write_updated_index(
    source_index_file: Path,
    output_index_file: Path,
    dataset_root: Path,
    index: Dict,
    updated_entries: Sequence[Dict],
    num_phases: int,
    character_height_m: float,
    lower_bound_m: float,
) -> None:
    updated = dict(index)
    updated["samples"] = list(updated_entries)
    updated["num_samples"] = len(updated_entries)
    updated["iiw_gt_ready"] = True
    updated["iiw_gt_method"] = METHOD_NAME
    updated["iiw_gt_method_version"] = METHOD_VERSION
    updated["iiw_num_phases"] = num_phases
    updated["iiw_character_height_m"] = character_height_m
    updated["iiw_lower_bound_m"] = lower_bound_m
    updated["iiw_upper_bound_m"] = (
        lower_bound_m + 0.25 * character_height_m
    )
    updated["iiw_source_index"] = str(source_index_file.resolve())
    write_json(output_index_file, updated)

    jsonl_file = output_index_file.with_suffix(".jsonl")
    jsonl_file.write_text(
        "".join(json.dumps(entry) + "\n" for entry in updated_entries)
    )
    print(f"[OK] updated index: {output_index_file}")
    print(f"[OK] updated JSONL: {jsonl_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-index", default=None)
    parser.add_argument("--num-phases", type=int, default=DEFAULT_NUM_PHASES)
    parser.add_argument(
        "--character-height-m",
        type=float,
        default=DEFAULT_CHARACTER_HEIGHT_M,
    )
    parser.add_argument(
        "--lower-bound-m", type=float, default=DEFAULT_LOWER_BOUND_M
    )
    parser.add_argument(
        "--point-chunk-size",
        type=int,
        default=DEFAULT_POINT_CHUNK_SIZE,
    )
    parser.add_argument("--save-framewise", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    source_index_file = Path(args.index).expanduser()
    if not source_index_file.is_absolute():
        source_index_file = dataset_root / source_index_file
    source_index_file = source_index_file.resolve()
    index = load_json(source_index_file)
    entries = index.get("samples", [])
    if not entries or len(entries) != int(index.get("num_samples", -1)):
        raise ValueError("dataset index has no samples or num_samples mismatch")
    if args.character_height_m <= 0.0:
        raise ValueError("character height must be positive")
    upper_bound_m = args.lower_bound_m + 0.25 * args.character_height_m
    if args.lower_bound_m <= 0.0 or upper_bound_m <= args.lower_bound_m:
        raise ValueError("invalid IIW distance bounds")

    if args.output_index is None:
        output_index_file = source_index_file.with_name(
            source_index_file.stem + "_iiw.json"
        )
    else:
        output_index_file = Path(args.output_index).expanduser()
        if not output_index_file.is_absolute():
            output_index_file = dataset_root / output_index_file
        output_index_file = output_index_file.resolve()
    if output_index_file == source_index_file:
        raise ValueError(
            "refusing to overwrite the source index; choose a distinct "
            "--output-index"
        )
    if output_index_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_index_file} exists; pass --overwrite to replace it"
        )

    scene = load_scene(index, dataset_root)
    updated_entries = []
    for entry in entries:
        if str(entry["scene_id"]) != str(index["scene_id"]):
            raise ValueError(f"{entry['sample_id']}: scene ID mismatch")
        updated_entry, _ = generate_sample(
            entry=entry,
            dataset_root=dataset_root,
            scene=scene,
            num_phases=args.num_phases,
            character_height_m=args.character_height_m,
            lower_bound_m=args.lower_bound_m,
            point_chunk_size=args.point_chunk_size,
            save_framewise=args.save_framewise,
            overwrite=args.overwrite,
        )
        validate_saved_sample(
            updated_entry, dataset_root, scene, args.num_phases
        )
        updated_entries.append(updated_entry)

    write_updated_index(
        source_index_file=source_index_file,
        output_index_file=output_index_file,
        dataset_root=dataset_root,
        index=index,
        updated_entries=updated_entries,
        num_phases=args.num_phases,
        character_height_m=args.character_height_m,
        lower_bound_m=args.lower_bound_m,
    )
    print(
        f"[PASS] generated point-aligned IIW targets for "
        f"{len(updated_entries)} samples"
    )


if __name__ == "__main__":
    main()

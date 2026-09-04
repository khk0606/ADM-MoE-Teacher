#!/usr/bin/env python3
"""Read-only Viser audit for Teacher-v9 scene-level all-sittable GT.

The viewer never loads a model and never runs training or diffusion.  It reads
only the immutable Teacher-v9 target artifact and its bound Teacher-v7 point
cloud, verifies their hashes and array contracts, then presents six aligned
panels.  ``viser`` is imported only after argument parsing so ``--help`` works
even on hosts where the optional viewer dependency is not installed.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from relational_teacher_v9_all_sittable_contract import (
    ACTIVE_THRESHOLD,
    CONTACT_DIM,
    EXPECTED_SCENE_TARGETS,
    NON_SIT_OBJECT_CATEGORY_IDS,
    OUTPUT_DATASET_SCHEMA,
    POINT_COUNT,
    PROMPTS,
    SCENE_MANIFEST_SCHEMA,
    SITTABLE_CATEGORY_IDS,
    SOURCE_DATASET_SCHEMA,
    TEACHER_FORWARD_INPUTS,
    read_json,
    resolve_under,
    sha256_file,
)
from relational_teacher_v9_all_sittable_viewer_common import (
    CHANNEL_ORDER,
    affordance_colors,
    build_xy_interpolation_plan,
    interpolate_xy_heatmap,
    rgba_heatmap,
    scalar_channel,
)


EXPECTED_SOURCE_SCENES = {
    "room_0101": "train",
    "room_0102": "train",
    "room_0201": "development",
}
EXPECTED_SCENES = {
    "room_0101": "train",
    "room_0102": "train",
}
EXPECTED_BUNDLE_KEYS = {
    "xyz",
    "source_indices",
    "instance_ids",
    "category_ids",
    "eligible_instance_ids",
    "eligible_numeric_instance_ids",
    "verified_object_mask",
    "verified_positive_mask",
    "instance_affordance",
    "instance_active_mask",
    "all_sittable_affordance",
    "unknown_sittable_mask",
    "environment_aux_mask",
    "explicit_negative_mask",
    "winner_instance_slot",
}

ROLE_VERIFIED = np.asarray((45, 235, 90), dtype=np.uint8)
ROLE_UNKNOWN = np.asarray((255, 185, 35), dtype=np.uint8)
ROLE_NEGATIVE = np.asarray((255, 60, 195), dtype=np.uint8)
ROLE_OTHER = np.asarray((135, 145, 155), dtype=np.uint8)
ROLE_ENVIRONMENT = np.asarray((35, 45, 65), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only six-panel audit of Teacher-v9 Bed + normal Chair + "
            "High Chair all-sittable ground truth."
        )
    )
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--initial-scene", choices=tuple(EXPECTED_SCENES), default="room_0101")
    parser.add_argument("--heatmap-resolution", type=int, default=128)
    parser.add_argument("--heatmap-neighbors", type=int, default=8)
    return parser.parse_args()


def _exact(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError(
            "{} dtype/shape changed: {}/{} != {}/{}".format(
                label, actual.dtype, actual.shape, expected.dtype, expected.shape
            )
        )
    if not np.array_equal(actual, expected):
        raise ValueError(label + " changed")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _rgb_u8(points: np.ndarray) -> np.ndarray:
    rgb = np.asarray(points[:, 3:6], dtype=np.float32)
    if not np.isfinite(rgb).all() or float(rgb.min()) < 0.0:
        raise ValueError("scene RGB values are non-finite or negative")
    if float(rgb.max()) <= 1.0 + 1e-6:
        rgb = rgb * np.float32(255.0)
    elif float(rgb.max()) > 255.0 + 1e-4:
        raise ValueError("scene RGB range exceeds 255")
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def _validate_top_index(index: Mapping[str, object]) -> None:
    if (
        index.get("schema") != OUTPUT_DATASET_SCHEMA
        or index.get("status") != "ALL_SITTABLE_DATASET_PASS"
        or index.get("authorization") != "teacher_v9_all_sittable_cuda_preflight_only"
        or index.get("teacher_lora_v9_training_authorized") is not False
        or index.get("relation_or_distance_used_as_forward_input") is not False
        or index.get("motion_or_target_id_used_as_forward_input") is not False
        or index.get("unverified_sittable_instances_are_negative") is not False
        or index.get("primary_loss_policy")
        != "equal_instance_macro_on_verified_object_masks"
        or index.get("environment_contact_is_auxiliary_only") is not True
        or index.get("development_used_for_parameter_selection") is not False
        or index.get("teacher_forward_inputs")
        != list(TEACHER_FORWARD_INPUTS)
        or index.get("prompt_policy") != PROMPTS
        or index.get("development_scene_id") != "room_0201"
        or index.get("development_materialization_deferred_until_checkpoint_lock")
        is not True
        or index.get("development_arrays_read") is not False
        or index.get("scene_count") != 2
        or index.get("replica_count") != 12
        or index.get("prompt_expanded_row_count") != 4
        or index.get("split_scene_counts") != {"train": 2}
        or index.get("split_row_counts") != {"train": 4}
        or index.get("failed_checks")
    ):
        raise ValueError("Teacher-v9 viewer rejected the top-level dataset contract")


def _source_record_subset(record: Mapping[str, object]) -> Dict[str, object]:
    keys = (
        "scene_id",
        "split",
        "points_file",
        "points_sha256",
        "sidecar_file",
        "sidecar_sha256",
        "instances_file",
        "instances_sha256",
        "motion_index_file",
        "motion_index_sha256",
        "dense_index_file",
        "dense_index_sha256",
    )
    return {key: record[key] for key in keys}


def _load_scene(
    source_root: Path,
    output_root: Path,
    source_record: Mapping[str, object],
    output_record: Mapping[str, object],
) -> Dict[str, object]:
    scene_id = str(output_record.get("scene_id", ""))
    split = str(output_record.get("split", ""))
    if EXPECTED_SCENES.get(scene_id) != split:
        raise ValueError(scene_id + ": scene split changed")
    if output_record.get("source_scene_record") != _source_record_subset(source_record):
        raise ValueError(scene_id + ": source scene binding changed")
    if (
        output_record.get("replica_count") != 6
        or output_record.get("row_count") != 2
        or output_record.get("verified_target_instances")
        != sorted(set(EXPECTED_SCENE_TARGETS[scene_id].values()))
        or output_record.get("unknown_sittable_instances_are_negative") is not False
    ):
        raise ValueError(scene_id + ": scene count/role contract changed")

    points_path = resolve_under(source_root, source_record["points_file"])
    if sha256_file(points_path) != str(source_record["points_sha256"]):
        raise ValueError(scene_id + ": source point-cloud hash changed")
    points_payload = _load_npz(points_path)
    if "points" not in points_payload:
        raise ValueError(scene_id + ": source point array is absent")
    points = np.asarray(points_payload["points"], dtype=np.float32)
    if points.shape != (POINT_COUNT, 6) or not np.isfinite(points).all():
        raise ValueError(scene_id + ": source point tensor changed")
    sidecar_path = resolve_under(source_root, source_record["sidecar_file"])
    if sha256_file(sidecar_path) != str(source_record["sidecar_sha256"]):
        raise ValueError(scene_id + ": source sidecar hash changed")
    sidecar = _load_npz(sidecar_path)
    required_sidecar = {
        "xyz_afford_z_up",
        "source_indices",
        "instance_ids",
        "category_ids",
    }
    if not required_sidecar.issubset(sidecar):
        raise ValueError(scene_id + ": source sidecar arrays changed")
    source_xyz = np.asarray(sidecar["xyz_afford_z_up"], dtype=np.float32)
    source_indices = np.asarray(sidecar["source_indices"], dtype=np.int64)
    source_instance_ids = np.asarray(sidecar["instance_ids"], dtype=np.int64)
    source_category_ids = np.asarray(sidecar["category_ids"], dtype=np.int64)
    if (
        source_xyz.shape != (POINT_COUNT, 3)
        or source_indices.shape != (POINT_COUNT,)
        or source_instance_ids.shape != (POINT_COUNT,)
        or source_category_ids.shape != (POINT_COUNT,)
        or not np.isfinite(source_xyz).all()
        or not np.allclose(points[:, :3], source_xyz, rtol=0.0, atol=1e-6)
        or np.unique(source_indices).size != POINT_COUNT
    ):
        raise ValueError(scene_id + ": source point/sidecar order changed")

    manifest_path = resolve_under(output_root, output_record["manifest_file"])
    if sha256_file(manifest_path) != str(output_record["manifest_sha256"]):
        raise ValueError(scene_id + ": scene manifest hash changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != SCENE_MANIFEST_SCHEMA
        or manifest.get("status") != "ALL_SITTABLE_SCENE_PASS"
        or manifest.get("scene_id") != scene_id
        or manifest.get("split") != split
        or manifest.get("verified_target_instances")
        != sorted(set(EXPECTED_SCENE_TARGETS[scene_id].values()))
        or manifest.get("relation_or_distance_used") is not False
        or manifest.get("unknown_sittable_instances_are_negative") is not False
        or manifest.get("primary_loss_policy")
        != "equal_instance_macro_on_verified_object_masks"
        or manifest.get("environment_contact_is_auxiliary_only") is not True
        or manifest.get("development_used_for_parameter_selection") is not False
        or manifest.get("teacher_forward_inputs")
        != list(TEACHER_FORWARD_INPUTS)
        or manifest.get("failed_checks")
    ):
        raise ValueError(scene_id + ": scene manifest contract changed")

    targets = manifest.get("group_targets")
    if targets != EXPECTED_SCENE_TARGETS[scene_id]:
        raise ValueError(scene_id + ": group target inventory changed")
    role_targets = {
        "bed": str(targets["bed"]),
        "normal_chair": str(targets["normal_chair"]),
        "high_chair": str(targets["high_desk_motion"]),
    }
    if str(targets["high_chair_legacy_motion"]) != role_targets["high_chair"]:
        raise ValueError(scene_id + ": High-Desk and legacy motions changed target")
    if len(set(role_targets.values())) != 3:
        raise ValueError(scene_id + ": expected three unique physical Sit targets")

    bundle_path = resolve_under(output_root, manifest["all_sittable_consensus_file"])
    if sha256_file(bundle_path) != str(manifest["all_sittable_consensus_sha256"]):
        raise ValueError(scene_id + ": all-sittable consensus hash changed")
    arrays = _load_npz(bundle_path)
    if set(arrays) != EXPECTED_BUNDLE_KEYS:
        raise ValueError(scene_id + ": consensus bundle keys changed")

    expected_shapes = {
        "xyz": (POINT_COUNT, 3),
        "source_indices": (POINT_COUNT,),
        "instance_ids": (POINT_COUNT,),
        "category_ids": (POINT_COUNT,),
        "eligible_instance_ids": (3,),
        "eligible_numeric_instance_ids": (3,),
        "verified_object_mask": (3, POINT_COUNT),
        "verified_positive_mask": (POINT_COUNT,),
        "instance_affordance": (3, POINT_COUNT, CONTACT_DIM),
        "instance_active_mask": (3, POINT_COUNT, CONTACT_DIM),
        "all_sittable_affordance": (POINT_COUNT, CONTACT_DIM),
        "unknown_sittable_mask": (POINT_COUNT,),
        "environment_aux_mask": (POINT_COUNT,),
        "explicit_negative_mask": (POINT_COUNT,),
        "winner_instance_slot": (POINT_COUNT, CONTACT_DIM),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError("{}/{} shape changed: {}".format(scene_id, key, arrays[key].shape))
    if arrays["xyz"].dtype != np.float32:
        raise ValueError(scene_id + ": XYZ dtype changed")
    if arrays["instance_affordance"].dtype != np.float32:
        raise ValueError(scene_id + ": instance affordance dtype changed")
    if arrays["all_sittable_affordance"].dtype != np.float32:
        raise ValueError(scene_id + ": all-sittable affordance dtype changed")
    for key in ("source_indices", "instance_ids", "category_ids", "eligible_numeric_instance_ids"):
        if arrays[key].dtype != np.int64:
            raise ValueError("{}/{} dtype changed".format(scene_id, key))
    for key in (
        "verified_object_mask",
        "verified_positive_mask",
        "instance_active_mask",
        "unknown_sittable_mask",
        "environment_aux_mask",
        "explicit_negative_mask",
    ):
        if arrays[key].dtype != np.bool_:
            raise ValueError("{}/{} dtype changed".format(scene_id, key))
    if arrays["winner_instance_slot"].dtype != np.int8:
        raise ValueError(scene_id + ": winner slot dtype changed")
    if arrays["eligible_instance_ids"].dtype.kind not in {"U", "S"}:
        raise ValueError(scene_id + ": stable target IDs are not strings")

    _exact(arrays["xyz"], source_xyz, scene_id + "/xyz")
    _exact(arrays["source_indices"], source_indices, scene_id + "/source-indices")
    _exact(arrays["instance_ids"], source_instance_ids, scene_id + "/instance-ids")
    _exact(arrays["category_ids"], source_category_ids, scene_id + "/category-ids")
    instance_affordance = arrays["instance_affordance"]
    all_affordance = arrays["all_sittable_affordance"]
    if (
        not np.isfinite(instance_affordance).all()
        or not np.isfinite(all_affordance).all()
        or np.any(instance_affordance < 0.0)
        or np.any(instance_affordance > 1.0)
        or np.any(all_affordance < 0.0)
        or np.any(all_affordance > 1.0)
    ):
        raise ValueError(scene_id + ": affordance range changed")
    _exact(
        all_affordance,
        np.maximum.reduce(instance_affordance),
        scene_id + "/exact-instance-union",
    )
    _exact(
        arrays["instance_active_mask"],
        instance_affordance >= np.float32(ACTIVE_THRESHOLD),
        scene_id + "/active-mask",
    )

    target_names = tuple(str(value) for value in arrays["eligible_instance_ids"].tolist())
    if set(target_names) != set(role_targets.values()):
        raise ValueError(scene_id + ": eligible target names changed")
    target_slots = {name: target_names.index(name) for name in role_targets.values()}
    numeric_ids = arrays["eligible_numeric_instance_ids"]
    instance_ids = arrays["instance_ids"]
    category_ids = arrays["category_ids"]
    expected_verified = np.stack(
        [instance_ids == numeric_id for numeric_id in numeric_ids], axis=0
    )
    _exact(arrays["verified_object_mask"], expected_verified, scene_id + "/verified-mask")
    _exact(
        arrays["verified_positive_mask"],
        expected_verified.any(axis=0),
        scene_id + "/verified-positive-mask",
    )
    expected_unknown = np.isin(category_ids, list(SITTABLE_CATEGORY_IDS)) & ~expected_verified.any(axis=0)
    expected_negative = np.isin(category_ids, list(NON_SIT_OBJECT_CATEGORY_IDS))
    _exact(arrays["unknown_sittable_mask"], expected_unknown, scene_id + "/unknown-mask")
    _exact(
        arrays["environment_aux_mask"],
        instance_ids == 0,
        scene_id + "/environment-aux-mask",
    )
    _exact(arrays["explicit_negative_mask"], expected_negative, scene_id + "/negative-mask")
    if np.any(expected_unknown & expected_negative):
        raise ValueError(scene_id + ": unknown and explicit-negative roles overlap")

    winner = np.argmax(instance_affordance, axis=0).astype(np.int8)
    winner[np.max(instance_affordance, axis=0) <= 0.0] = -1
    _exact(arrays["winner_instance_slot"], winner, scene_id + "/winner-slot")
    for name, slot in target_slots.items():
        numeric = int(numeric_ids[slot])
        own = instance_ids == numeric
        if not np.any(own):
            raise ValueError(scene_id + "/" + name + ": no sampled object points")
        if float(instance_affordance[slot, own, 0].max()) < 0.75:
            raise ValueError(scene_id + "/" + name + ": robust pelvis evidence changed")
        other_objects = (instance_ids != 0) & ~own
        if np.any(instance_affordance[slot, other_objects] != 0.0):
            raise ValueError(scene_id + "/" + name + ": cross-object heat leaked into GT")

    # The separately stored instance maps must exactly match the bundle.
    instance_manifest = manifest.get("instance_consensus")
    if not isinstance(instance_manifest, dict) or set(instance_manifest) != set(target_names):
        raise ValueError(scene_id + ": instance-consensus inventory changed")
    for name in target_names:
        row = instance_manifest[name]
        path = resolve_under(output_root, row["file"])
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(scene_id + "/" + name + ": instance file hash changed")
        payload = _load_npz(path)
        if set(payload) != {"affordance"}:
            raise ValueError(scene_id + "/" + name + ": instance file keys changed")
        _exact(payload["affordance"], instance_affordance[target_slots[name]], scene_id + "/" + name)

    rows = output_record.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError(scene_id + ": prompt rows changed")
    if {str(row.get("prompt_id")) for row in rows} != set(PROMPTS):
        raise ValueError(scene_id + ": watch/write prompt pair changed")
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if (
            row.get("row_id") != f"{scene_id}_{prompt_id}"
            or row.get("text") != PROMPTS[prompt_id]
            or row.get("all_sittable_gt_file") != manifest["all_sittable_consensus_file"]
            or row.get("all_sittable_gt_sha256") != manifest["all_sittable_consensus_sha256"]
            or row.get("all_sittable_array_key") != "all_sittable_affordance"
            or row.get("supervision") != "scene_level_all_verified_sittable_consensus"
        ):
            raise ValueError(scene_id + ": primary prompt/GT binding changed")

    return {
        "scene_id": scene_id,
        "split": split,
        "points": points,
        "rgb": _rgb_u8(points),
        "arrays": arrays,
        "manifest": manifest,
        "role_targets": role_targets,
        "target_slots": target_slots,
    }


def load_viewer_dataset(
    source_root: Path, output_root: Path, index_path: Path
) -> Dict[str, Dict[str, object]]:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve()
    try:
        index_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Teacher-v9 index must be inside its dataset root") from exc
    index = read_json(index_path)
    _validate_top_index(index)
    if str(source_root) != str(index.get("source_dataset_root", "")):
        raise ValueError("Teacher-v9 source dataset root binding changed")
    source_index_path = resolve_under(source_root, index["source_index_file"])
    if sha256_file(source_index_path) != str(index["source_index_sha256"]):
        raise ValueError("Teacher-v9 source index hash changed")
    source_index = read_json(source_index_path)
    if (
        source_index.get("schema") != SOURCE_DATASET_SCHEMA
        or source_index.get("status") != "DENSE_DATASET_PASS"
        or source_index.get("authorization")
        != "teacher_lora_v7_cuda_preflight_only"
        or source_index.get("prompt_policy_id")
        != "purpose_sit_object_agnostic_v1"
        or source_index.get("point_count") != POINT_COUNT
        or source_index.get("contact_dim") != CONTACT_DIM
        or source_index.get("num_scenes") != 3
        or source_index.get("relation_or_distance_used_as_forward_input") is not False
        or source_index.get("target_instance_gt_is_supervision_only") is not True
        or source_index.get("heldout_dense_arrays_read_during_build") is not False
        or source_index.get("teacher_lora_v7_training_authorized") is not False
        or source_index.get("num_dense_motions") != 72
        or source_index.get("num_rows") != 144
        or source_index.get("split_scene_counts")
        != {"train": 2, "development": 1}
        or source_index.get("split_row_counts")
        != {"train": 96, "development": 48}
        or source_index.get("failed_checks")
    ):
        raise ValueError("Teacher-v9 source dataset top-level contract changed")
    source_scene_rows = source_index.get("scenes")
    if not isinstance(source_scene_rows, list) or len(source_scene_rows) != 3:
        raise ValueError("source Teacher-v7 scene inventory changed")
    source_records = {
        str(row["scene_id"]): row for row in source_scene_rows
    }
    if len(source_records) != 3:
        raise ValueError("source Teacher-v7 contains duplicate scene IDs")
    output_rows = index.get("scenes")
    if not isinstance(output_rows, list) or len(output_rows) != 2:
        raise ValueError("Teacher-v9 scene rows are absent")
    output_records = {str(row["scene_id"]): row for row in output_rows}
    if len(output_records) != 2:
        raise ValueError("Teacher-v9 contains duplicate scene IDs")
    row_ids = [
        str(row.get("row_id", ""))
        for scene_record in output_rows
        for row in scene_record.get("rows", [])
    ]
    if len(row_ids) != 4 or len(set(row_ids)) != 4 or any(not value for value in row_ids):
        raise ValueError("Teacher-v9 primary row IDs must be four globally unique values")
    source_splits = {
        scene_id: str(row["split"]) for scene_id, row in source_records.items()
    }
    if source_splits != EXPECTED_SOURCE_SCENES or set(output_records) != set(EXPECTED_SCENES):
        raise ValueError("Teacher-v9 viewer scene inventory changed")
    return {
        scene_id: _load_scene(
            source_root,
            output_root,
            source_records[scene_id],
            output_records[scene_id],
        )
        for scene_id in EXPECTED_SCENES
    }


def _blend_heatmap_rgb(heatmap: np.ndarray, rgb: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.60))
    return np.rint((1.0 - amount) * heatmap + amount * rgb).clip(0, 255).astype(np.uint8)


def _role_point_colors(
    rgb: np.ndarray,
    verified: np.ndarray,
    unknown: np.ndarray,
    negative: np.ndarray,
    instance_ids: np.ndarray,
) -> np.ndarray:
    colors = np.rint(0.22 * rgb + 0.78 * ROLE_OTHER).astype(np.uint8)
    colors[instance_ids == 0] = ROLE_ENVIRONMENT
    colors[verified] = ROLE_VERIFIED
    colors[unknown] = ROLE_UNKNOWN
    colors[negative] = ROLE_NEGATIVE
    return colors


def _role_heatmap(
    plan: Mapping[str, np.ndarray],
    verified: np.ndarray,
    unknown: np.ndarray,
    negative: np.ndarray,
    opacity: float,
) -> np.ndarray:
    scores = np.stack(
        (
            interpolate_xy_heatmap(verified.astype(np.float32), plan),
            interpolate_xy_heatmap(unknown.astype(np.float32), plan),
            interpolate_xy_heatmap(negative.astype(np.float32), plan),
        ),
        axis=-1,
    )
    strength = scores.max(axis=-1)
    total = scores.sum(axis=-1, keepdims=True)
    weights = np.divide(scores, total, out=np.zeros_like(scores), where=total > 1e-8)
    palette = np.stack((ROLE_VERIFIED, ROLE_UNKNOWN, ROLE_NEGATIVE), axis=0).astype(np.float32)
    rgb = np.rint(weights @ palette).clip(0, 255).astype(np.uint8)
    alpha = np.rint(255.0 * float(np.clip(opacity, 0.0, 1.0)) * strength)
    return np.concatenate((rgb, alpha[..., None].clip(0, 255).astype(np.uint8)), axis=-1)


def main() -> None:
    args = parse_args()
    # Optional UI dependency is intentionally lazy: ``--help`` has already
    # completed before this import is attempted.
    try:
        import viser
    except ImportError as exc:
        raise SystemExit(
            "viser is required only to launch the viewer; install it in the active environment"
        ) from exc

    scenes = load_viewer_dataset(args.source_dataset_root, args.dataset_root, args.index)
    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Teacher-v9: all-sittable GT audit",
    )
    server.gui.add_markdown(
        "## Teacher-v9 all-sittable GT audit\n"
        "**저장된 GT를 읽기만 하며 학습·추론을 실행하지 않습니다.**  \n"
        "위: `Input RGB → Bed consensus → Normal Chair consensus`  \n"
        "아래: `High Chair consensus → All-sittable consensus → Object roles`  \n"
        "All-sittable은 세 물체 map의 정확한 `max`입니다. `watch/write`는 같은 GT를 "
        "공유하며 TV/Desk 거리와 관계는 Teacher 입력이 아닙니다.  \n"
        "연속 면은 원본 8192개 값을 바꾸지 않는 표시 전용 XY 보간입니다.  \n"
        "역할 색: **초록=motion 검증 Sit**, **노랑=미검증 Sit(무시)**, "
        "**분홍=명시적 non-Sit**, 회색/남색=기타/환경."
    )
    scene_control = server.gui.add_dropdown(
        "Scene",
        options=tuple(EXPECTED_SCENES),
        initial_value=args.initial_scene,
    )
    channel_control = server.gui.add_dropdown(
        "Affordance channel",
        options=("any_joint",) + CHANNEL_ORDER,
        initial_value="any_joint",
    )
    rgb_blend = server.gui.add_slider(
        "RGB blend (affordance panels)",
        min=0.0,
        max=0.60,
        step=0.05,
        initial_value=0.10,
    )
    render_mode = server.gui.add_dropdown(
        "Map rendering",
        options=(
            "continuous heatmap + points",
            "continuous heatmap only",
            "points only",
        ),
        initial_value="continuous heatmap + points",
    )
    heatmap_opacity = server.gui.add_slider(
        "Continuous heatmap opacity",
        min=0.10,
        max=1.00,
        step=0.05,
        initial_value=0.85,
    )
    point_size = server.gui.add_slider(
        "Point size", min=0.008, max=0.060, step=0.002, initial_value=0.020
    )
    highlight = server.gui.add_dropdown(
        "Highlight role",
        options=("verified Sit", "unknown Sit", "explicit non-Sit", "none"),
        initial_value="none",
    )
    status = server.gui.add_markdown("")
    lock = threading.Lock()
    interpolation_cache: Dict[str, Mapping[str, np.ndarray]] = {}

    def render() -> None:
        with lock:
            scene_id = str(scene_control.value)
            scene = scenes[scene_id]
            arrays = scene["arrays"]
            xyz = np.asarray(arrays["xyz"], dtype=np.float32)
            rgb = np.asarray(scene["rgb"], dtype=np.uint8)
            center_xy = 0.5 * (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0))
            local_xyz = xyz.copy()
            local_xyz[:, :2] -= center_xy
            if scene_id not in interpolation_cache:
                interpolation_cache[scene_id] = build_xy_interpolation_plan(
                    local_xyz,
                    resolution=args.heatmap_resolution,
                    neighbors=args.heatmap_neighbors,
                )
            plan = interpolation_cache[scene_id]
            width = float(np.ptp(local_xyz[:, 0]))
            depth = float(np.ptp(local_xyz[:, 1]))
            spacing_x = max(width + 1.0, 4.0)
            spacing_y = max(depth + 1.0, 4.0)
            offsets = (
                np.asarray((-spacing_x, 0.5 * spacing_y, 0.0), dtype=np.float32),
                np.asarray((0.0, 0.5 * spacing_y, 0.0), dtype=np.float32),
                np.asarray((spacing_x, 0.5 * spacing_y, 0.0), dtype=np.float32),
                np.asarray((-spacing_x, -0.5 * spacing_y, 0.0), dtype=np.float32),
                np.asarray((0.0, -0.5 * spacing_y, 0.0), dtype=np.float32),
                np.asarray((spacing_x, -0.5 * spacing_y, 0.0), dtype=np.float32),
            )
            role_targets = scene["role_targets"]
            target_slots = scene["target_slots"]
            instance_maps = arrays["instance_affordance"]
            maps = (
                None,
                instance_maps[target_slots[role_targets["bed"]]],
                instance_maps[target_slots[role_targets["normal_chair"]]],
                instance_maps[target_slots[role_targets["high_chair"]]],
                arrays["all_sittable_affordance"],
                None,
            )
            names = (
                "Input 3D Scene (RGB)",
                "Bed consensus: " + role_targets["bed"],
                "Normal Chair consensus: " + role_targets["normal_chair"],
                "High Chair consensus: " + role_targets["high_chair"],
                "All-sittable consensus = max(Bed, Normal, High)",
                "Object roles: verified / unknown / explicit non-Sit",
            )
            verified = arrays["verified_object_mask"].any(axis=0)
            unknown = arrays["unknown_sittable_mask"]
            negative = arrays["explicit_negative_mask"]
            instance_ids = arrays["instance_ids"]
            role_colors = _role_point_colors(rgb, verified, unknown, negative, instance_ids)
            selected_channel = str(channel_control.value)
            mode = str(render_mode.value)
            show_points = mode != "continuous heatmap only"
            show_surface = mode != "points only"
            selection = str(highlight.value)
            if selection == "verified Sit":
                highlight_mask, highlight_color = verified, ROLE_VERIFIED
            elif selection == "unknown Sit":
                highlight_mask, highlight_color = unknown, ROLE_UNKNOWN
            elif selection == "explicit non-Sit":
                highlight_mask, highlight_color = negative, ROLE_NEGATIVE
            else:
                highlight_mask = np.zeros(POINT_COUNT, dtype=bool)
                highlight_color = ROLE_OTHER

            xy_min = np.asarray(plan["xy_min"], dtype=np.float32)
            xy_max = np.asarray(plan["xy_max"], dtype=np.float32)
            for panel, (name, affordance, offset) in enumerate(zip(names, maps, offsets)):
                panel_xyz = local_xyz + offset
                if panel == 0:
                    colors = rgb
                elif panel == 5:
                    colors = role_colors
                else:
                    scalar = scalar_channel(affordance, selected_channel)
                    colors = _blend_heatmap_rgb(
                        affordance_colors(scalar), rgb, float(rgb_blend.value)
                    )
                server.scene.add_point_cloud(
                    "/panel_{}/points".format(panel),
                    points=panel_xyz,
                    colors=colors,
                    point_size=float(point_size.value),
                    point_shape="circle",
                    precision="float32",
                    visible=(panel == 0 or show_points),
                )

                if panel != 0:
                    if panel == 5:
                        image = _role_heatmap(
                            plan,
                            verified,
                            unknown,
                            negative,
                            float(heatmap_opacity.value),
                        )
                    else:
                        scalar = scalar_channel(affordance, selected_channel)
                        image = rgba_heatmap(
                            affordance_colors(interpolate_xy_heatmap(scalar, plan)),
                            float(heatmap_opacity.value),
                        )
                    position = offset + np.asarray(
                        (
                            0.5 * float(xy_min[0] + xy_max[0]),
                            0.5 * float(xy_min[1] + xy_max[1]),
                            float(np.asarray(plan["floor_z"]).item()) - 0.015,
                        ),
                        dtype=np.float32,
                    )
                    server.scene.add_image(
                        "/panel_{}/continuous_heatmap".format(panel),
                        image=image,
                        render_width=float(xy_max[0] - xy_min[0]),
                        render_height=float(xy_max[1] - xy_min[1]),
                        position=position,
                        visible=show_surface,
                        cast_shadow=False,
                        receive_shadow=False,
                    )

                overlay_xyz = panel_xyz[highlight_mask]
                overlay_visible = bool(overlay_xyz.shape[0])
                if not overlay_visible:
                    overlay_xyz = panel_xyz[:1]
                overlay_colors = np.repeat(
                    np.asarray(highlight_color, dtype=np.uint8)[None],
                    len(overlay_xyz),
                    axis=0,
                )
                server.scene.add_point_cloud(
                    "/panel_{}/highlight".format(panel),
                    points=overlay_xyz,
                    colors=overlay_colors,
                    point_size=min(float(point_size.value) * 1.8, 0.10),
                    point_shape="circle",
                    precision="float32",
                    visible=overlay_visible,
                )
                label_position = offset + np.asarray(
                    (0.0, 0.0, float(local_xyz[:, 2].max()) + 0.55),
                    dtype=np.float32,
                )
                server.scene.add_label(
                    "/panel_{}/label".format(panel), text=name, position=label_position
                )

            all_scalar = scalar_channel(arrays["all_sittable_affordance"], selected_channel)
            per_instance = []
            for role in ("bed", "normal_chair", "high_chair"):
                name = role_targets[role]
                slot = target_slots[name]
                own = arrays["verified_object_mask"][slot]
                value = scalar_channel(instance_maps[slot], selected_channel)
                per_instance.append(
                    "`{}` max `{:.6f}` · active `{}/{}`".format(
                        name,
                        float(value[own].max()),
                        int((value[own] >= ACTIVE_THRESHOLD).sum()),
                        int(own.sum()),
                    )
                )
            status.content = (
                "**선택 scene** `{}` · split `{}` · channel `{}`  \n"
                "검증된 Sit: {}  \n"
                "미검증 Sit(negative 아님): `{}`개 points · 명시적 non-Sit: `{}`개 points  \n"
                "All-sittable active: `{}`/`{}` points  \n\n"
                "{}  \n{}  \n{}  \n\n"
                "`Sit anywhere to watch.`와 `Sit anywhere to write.`는 이 동일한 GT를 공유합니다."
            ).format(
                scene_id,
                scene["split"],
                selected_channel,
                ", ".join("`{}`".format(value) for value in role_targets.values()),
                int(unknown.sum()),
                int(negative.sum()),
                int((all_scalar >= ACTIVE_THRESHOLD).sum()),
                POINT_COUNT,
                per_instance[0],
                per_instance[1],
                per_instance[2],
            )
            server.flush()

    for control in (
        scene_control,
        channel_control,
        rgb_blend,
        render_mode,
        heatmap_opacity,
        point_size,
        highlight,
    ):
        control.on_update(lambda _event: render())

    @server.on_client_connect
    def _on_client_connect(client) -> None:
        scene = scenes[str(scene_control.value)]
        xyz = np.asarray(scene["arrays"]["xyz"], dtype=np.float32)
        radius = max(float(3.2 * np.ptp(xyz[:, 0]) + 2.2 * np.ptp(xyz[:, 1])), 12.0)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.look_at = (0.0, 0.0, float(np.median(xyz[:, 2])))
        client.camera.position = (0.0, -0.85 * radius, 0.72 * radius)

    render()
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print("[READY] Teacher-v9 all-sittable GT viewer")
    print("[PASS] all scene/bundle/source hashes and exact three-instance unions verified")
    print("[INFO] read-only GT audit; no model, training, diffusion, relation or distance")
    print("[OPEN] http://{}:{}".format(display_host, args.port), flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOPPED] Teacher-v9 all-sittable GT viewer")


if __name__ == "__main__":
    main()

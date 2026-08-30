#!/usr/bin/env python3
"""Strict contracts for the multi-instance relational Teacher-v6 dataset.

This module deliberately separates per-object ``instance_ids`` from semantic
``category_ids``.  Category/instance arrays are supervision-only metadata and
must never be included in the CDM forward kwargs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DATASET_SCHEMA = "relational_teacher_v6_dataset_v1"
SCENE_SPLIT_SCHEMA = "relational_teacher_v6_scene_split_v1"
INSTANCE_SCHEMA = "relational_affordance_unity_instances_v1"
PROMPT_POLICY_ID = "purpose_sit_object_agnostic_v1"

POINT_COUNT = 8192
CONTACT_DIM = 6

CATEGORY_IDS = {
    "environment": 0,
    "chair": 1,
    "bed": 2,
    "whiteboard": 3,
    "tv": 4,
    "desk": 5,
}
ID_TO_CATEGORY = {value: key for key, value in CATEGORY_IDS.items()}
SITTABLE_CATEGORY_IDS = frozenset(
    (CATEGORY_IDS["chair"], CATEGORY_IDS["bed"])
)
SIT_NEGATIVE_OBJECT_CATEGORY_IDS = frozenset(
    (
        CATEGORY_IDS["whiteboard"],
        CATEGORY_IDS["tv"],
        CATEGORY_IDS["desk"],
    )
)

PROMPTS = {
    "sit_watch_v1": "Sit anywhere to watch.",
    "sit_write_v1": "Sit anywhere to write.",
}
FORBIDDEN_PROMPT_WORDS = (
    "chair",
    "bed",
    "television",
    "tv",
    "desk",
    "whiteboard",
)
FORWARD_INPUT_KEYS = frozenset(("c_pc_xyz", "c_pc_feat", "c_text"))
SUPERVISION_ONLY_KEYS = frozenset(
    (
        "instance_ids",
        "category_ids",
        "candidate_mask",
        "purpose_mask",
        "nearest_instance_id",
        "distance_gt",
        "relation_gt",
    )
)


def read_json(path: Path) -> MutableMapping[str, object]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


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
            handle.write(json.dumps(value, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_prompt_policy() -> None:
    if len(PROMPTS) != 2:
        raise AssertionError("Teacher-v6 requires exactly the watch/write pair")
    for prompt_id, text in PROMPTS.items():
        lowered = text.lower()
        leaked = [word for word in FORBIDDEN_PROMPT_WORDS if word in lowered]
        if leaked:
            raise AssertionError(f"{prompt_id}: object-name leakage: {leaked}")
    if set(FORWARD_INPUT_KEYS) & set(SUPERVISION_ONLY_KEYS):
        raise AssertionError("Teacher forward and supervision keys overlap")


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: Tuple[int, ...],
    dtype_kind: Optional[str] = None,
) -> np.ndarray:
    if name not in arrays:
        raise KeyError(f"missing array: {name}")
    value = np.asarray(arrays[name])
    if value.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {value.shape}")
    if dtype_kind is not None and value.dtype.kind not in dtype_kind:
        raise ValueError(f"{name}: unexpected dtype {value.dtype}")
    return value


def validate_instance_manifest(
    manifest: Mapping[str, object], scene_id: str
) -> Dict[int, int]:
    if manifest.get("schema") != INSTANCE_SCHEMA:
        raise ValueError(f"{scene_id}: unsupported instance manifest schema")
    if str(manifest.get("scene_id")) != scene_id:
        raise ValueError(f"{scene_id}: instance manifest scene mismatch")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"{scene_id}: empty object manifest")
    instance_to_category: Dict[int, int] = {}
    names = set()
    for raw in objects:
        if not isinstance(raw, dict):
            raise ValueError(f"{scene_id}: object entry is not a mapping")
        instance_id = int(raw["instance_id"])
        category_id = int(raw["category_id"])
        category = str(raw["category"]).lower()
        name = str(raw["name"])
        if instance_id <= 0:
            raise ValueError(f"{scene_id}/{name}: instance_id must be positive")
        if instance_id in instance_to_category:
            raise ValueError(f"{scene_id}: duplicate instance_id {instance_id}")
        if name in names:
            raise ValueError(f"{scene_id}: duplicate object name {name}")
        if category not in CATEGORY_IDS:
            raise ValueError(f"{scene_id}/{name}: unknown category {category}")
        if CATEGORY_IDS[category] != category_id:
            raise ValueError(f"{scene_id}/{name}: category ID mismatch")
        if category in {"chair", "bed", "tv", "desk"}:
            anchor = np.asarray(raw.get("anchor_unity_xz"), dtype=np.float32)
            if anchor.shape != (2,) or not np.isfinite(anchor).all():
                raise ValueError(f"{scene_id}/{name}: invalid anchor_unity_xz")
        instance_to_category[instance_id] = category_id
        names.add(name)
    categories = set(instance_to_category.values())
    for required in ("chair", "bed", "tv", "desk"):
        if CATEGORY_IDS[required] not in categories:
            raise ValueError(f"{scene_id}: required category absent: {required}")
    return instance_to_category


def load_and_validate_scene(
    dataset_root: Path, scene_record: Mapping[str, object]
) -> Dict[str, np.ndarray]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    scene_id = str(scene_record["scene_id"])
    points_file = (dataset_root / str(scene_record["points_file"])).resolve()
    sidecar_file = (dataset_root / str(scene_record["sidecar_file"])).resolve()
    instances_file = (dataset_root / str(scene_record["instances_file"])).resolve()
    for label, path in (
        ("points", points_file),
        ("sidecar", sidecar_file),
        ("instances", instances_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{scene_id}: missing {label}: {path}")
        expected_hash = str(scene_record.get(label + "_sha256", ""))
        if expected_hash and sha256_file(path) != expected_hash:
            raise ValueError(f"{scene_id}: {label} hash changed")

    with np.load(points_file, allow_pickle=False) as source:
        points = _require_array(source, "points", (POINT_COUNT, 6), "f").astype(
            np.float32
        )
    with np.load(sidecar_file, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    xyz = _require_array(
        arrays, "xyz_afford_z_up", (POINT_COUNT, 3), "f"
    ).astype(np.float32)
    source_indices = _require_array(
        arrays, "source_indices", (POINT_COUNT,), "iu"
    ).astype(np.int64)
    instance_ids = _require_array(
        arrays, "instance_ids", (POINT_COUNT,), "iu"
    ).astype(np.int64)
    category_ids = _require_array(
        arrays, "category_ids", (POINT_COUNT,), "iu"
    ).astype(np.int64)

    if not np.isfinite(points).all() or not np.isfinite(xyz).all():
        raise ValueError(f"{scene_id}: point arrays contain NaN/Inf")
    if not np.allclose(points[:, :3], xyz, atol=1e-6):
        raise ValueError(f"{scene_id}: point/sidecar order differs")
    if np.unique(source_indices).size != POINT_COUNT:
        raise ValueError(f"{scene_id}: source_indices are not unique")
    if np.any(instance_ids < 0):
        raise ValueError(f"{scene_id}: negative instance ID")
    if not set(np.unique(category_ids)).issubset(set(ID_TO_CATEGORY)):
        raise ValueError(f"{scene_id}: unknown category ID in point sidecar")

    manifest = read_json(instances_file)
    instance_to_category = validate_instance_manifest(manifest, scene_id)
    point_instances = set(int(v) for v in np.unique(instance_ids) if int(v) != 0)
    if point_instances != set(instance_to_category):
        raise ValueError(
            f"{scene_id}: point/manifest instances differ: "
            f"points={sorted(point_instances)}, manifest={sorted(instance_to_category)}"
        )
    for instance_id, category_id in instance_to_category.items():
        mask = instance_ids == instance_id
        if not np.any(mask):
            raise ValueError(f"{scene_id}: empty instance {instance_id}")
        if not np.all(category_ids[mask] == category_id):
            raise ValueError(f"{scene_id}: mixed category for instance {instance_id}")

    sittable_instances = sorted(
        instance_id
        for instance_id, category_id in instance_to_category.items()
        if category_id in SITTABLE_CATEGORY_IDS
    )
    if len(sittable_instances) < 2:
        raise ValueError(f"{scene_id}: fewer than two sittable candidates")
    return {
        "points": points,
        "xyz": xyz,
        "source_indices": source_indices,
        "instance_ids": instance_ids,
        "category_ids": category_ids,
        "sittable_mask": np.isin(category_ids, list(SITTABLE_CATEGORY_IDS)),
        "sit_negative_object_mask": np.isin(
            category_ids, list(SIT_NEGATIVE_OBJECT_CATEGORY_IDS)
        ),
    }


def validate_scene_split(split: Mapping[str, object]) -> Tuple[Sequence[str], Sequence[str]]:
    if split.get("schema") != SCENE_SPLIT_SCHEMA:
        raise ValueError("unsupported Teacher-v6 scene split schema")
    train = tuple(str(v) for v in split.get("train", []))
    development = tuple(str(v) for v in split.get("development", []))
    if not train:
        raise ValueError("Teacher-v6 train split is empty")
    if not development:
        raise ValueError("Teacher-v6 development split is empty")
    if len(set(train)) != len(train) or len(set(development)) != len(development):
        raise ValueError("duplicate scene ID in split")
    overlap = set(train) & set(development)
    if overlap:
        raise ValueError(f"scene leakage across split: {sorted(overlap)}")
    return train, development


def assert_forward_kwargs(kwargs: Mapping[str, object]) -> None:
    keys = set(kwargs)
    forbidden = keys & set(SUPERVISION_ONLY_KEYS)
    if forbidden:
        raise AssertionError(
            "supervision-only metadata entered Teacher forward: "
            + ", ".join(sorted(forbidden))
        )
    if keys != set(FORWARD_INPUT_KEYS):
        raise AssertionError(
            f"Teacher forward keys changed: expected {sorted(FORWARD_INPUT_KEYS)}, "
            f"got {sorted(keys)}"
        )


validate_prompt_policy()

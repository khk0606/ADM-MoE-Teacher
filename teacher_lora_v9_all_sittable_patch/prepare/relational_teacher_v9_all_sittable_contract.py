#!/usr/bin/env python3
"""Pure NumPy contracts for scene-level all-sittable Teacher-v9 targets.

Teacher-v7 stores one dense contact map per motion.  Those rows are valid
motion-contact observations, but they are conflicting primary targets for an
object-agnostic prompt when used one at a time.  This module deterministically
groups the 24 rows in each scene, builds six balanced scene-level unions, and
builds a robust fixed consensus target for auditing.

No relation, distance, TV, Desk, or selected-instance metadata is a Teacher
forward input.  Untested Chair/Bed instances are explicitly *unknown*; they
must never be converted into negative labels by this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


SOURCE_DATASET_SCHEMA = "relational_teacher_v7_hd_dataset_v1"
SOURCE_DENSE_SCHEMA = "relational_teacher_v7_hd_dense_contact_v2"
OUTPUT_DATASET_SCHEMA = "relational_teacher_v9_all_sittable_dataset_v1"
SCENE_MANIFEST_SCHEMA = "relational_teacher_v9_all_sittable_scene_v1"
INSTANCE_SCHEMA = "relational_affordance_unity_instances_v1"

POINT_COUNT = 8192
CONTACT_DIM = 6
REPLICA_COUNT = 6
ACTIVE_THRESHOLD = 0.30
CONSENSUS_NUMERATOR = 3
CONSENSUS_DENOMINATOR = 4

CATEGORY_ENVIRONMENT = 0
CATEGORY_CHAIR = 1
CATEGORY_BED = 2
SITTABLE_CATEGORY_IDS = frozenset((CATEGORY_CHAIR, CATEGORY_BED))
NON_SIT_OBJECT_CATEGORY_IDS = frozenset((3, 4, 5))
CATEGORY_NAMES = {
    0: "environment",
    1: "chair",
    2: "bed",
    3: "whiteboard",
    4: "tv",
    5: "desk",
}

PROMPTS = {
    "sit_watch_v1": "Sit anywhere to watch.",
    "sit_write_v1": "Sit anywhere to write.",
}
TEACHER_FORWARD_INPUTS = ("c_pc_feat", "c_pc_xyz", "c_text")

GROUP_ORDER = (
    "bed",
    "normal_chair",
    "high_desk_motion",
    "high_chair_legacy_motion",
)
EXPECTED_SCENE_TARGETS = {
    "room_0101": {
        "bed": "bed_01",
        "normal_chair": "chair_01",
        "high_desk_motion": "chair_06",
        "high_chair_legacy_motion": "chair_06",
    },
    "room_0102": {
        "bed": "bed_01",
        "normal_chair": "chair_05",
        "high_desk_motion": "chair_06",
        "high_chair_legacy_motion": "chair_06",
    },
    "room_0201": {
        "bed": "bed_01",
        "normal_chair": "chair_03",
        "high_desk_motion": "chair_02",
        "high_chair_legacy_motion": "chair_02",
    },
}
HIGH_DESK_TOKENS = ("_hc_hd_", "_hcw_hdw_")
SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def assert_teacher_forward_inputs(keys: Sequence[str]) -> None:
    actual = tuple(str(value) for value in keys)
    if actual != TEACHER_FORWARD_INPUTS:
        raise ValueError(
            "Teacher forward inputs must be exactly text + scene; "
            f"got {actual}"
        )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> MutableMapping[str, object]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
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
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=str(path.parent),
        prefix="." + path.name + ".",
        suffix=".npz",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_under(root: Path, raw: object) -> Path:
    root = Path(root).expanduser().resolve()
    path = (root / str(raw)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {raw}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def is_high_desk_motion(motion_id: str) -> bool:
    return any(token in motion_id for token in HIGH_DESK_TOKENS)


def row_is_high_desk(row: Mapping[str, object]) -> bool:
    collection = str(row.get("source_collection_id", ""))
    if collection:
        return collection in {"hc_hd", "hcw_hdw"}
    return is_high_desk_motion(str(row["motion_id"]))


def classify_dense_rows(
    rows: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, List[Mapping[str, object]]], Dict[str, str]]:
    """Return the sealed four motion strata and their target instances.

    Every scene must contain six Bed motions, six normal-Chair motions, six
    High-Desk motions, and six legacy Sit motions bound to that same high
    chair.  The last two are different motion sources but one physical object.
    """

    if len(rows) != 24:
        raise ValueError(f"expected exactly 24 dense rows, got {len(rows)}")
    motion_ids = [str(row.get("motion_id", "")) for row in rows]
    if len(set(motion_ids)) != 24 or any(not value for value in motion_ids):
        raise ValueError("dense motion IDs must be 24 unique non-empty strings")

    high_rows = [row for row in rows if row_is_high_desk(row)]
    if len(high_rows) != 6:
        raise ValueError(f"expected six High-Desk motion rows, got {len(high_rows)}")
    high_targets = {str(row["target_instance_id"]) for row in high_rows}
    if len(high_targets) != 1:
        raise ValueError(f"High-Desk rows have mixed targets: {sorted(high_targets)}")
    high_target = next(iter(high_targets))

    bed_rows = [
        row for row in rows
        if str(row["target_instance_id"]).startswith("bed_")
    ]
    if len(bed_rows) != 6:
        raise ValueError(f"expected six Bed rows, got {len(bed_rows)}")
    bed_targets = {str(row["target_instance_id"]) for row in bed_rows}
    if len(bed_targets) != 1:
        raise ValueError(f"Bed rows have mixed targets: {sorted(bed_targets)}")
    bed_target = next(iter(bed_targets))

    legacy_rows = [
        row for row in rows
        if str(row["target_instance_id"]) == high_target
        and not row_is_high_desk(row)
    ]
    if len(legacy_rows) != 6:
        raise ValueError(
            "expected six non-High-Desk legacy motions on the High-Chair target, "
            f"got {len(legacy_rows)}"
        )

    used = {id(row) for row in high_rows + bed_rows + legacy_rows}
    normal_rows = [row for row in rows if id(row) not in used]
    if len(normal_rows) != 6:
        raise ValueError(f"expected six normal-Chair rows, got {len(normal_rows)}")
    normal_targets = {str(row["target_instance_id"]) for row in normal_rows}
    if len(normal_targets) != 1:
        raise ValueError(f"normal-Chair rows have mixed targets: {sorted(normal_targets)}")
    normal_target = next(iter(normal_targets))

    if len({bed_target, normal_target, high_target}) != 3:
        raise ValueError("Bed, normal Chair, and High Chair targets must be distinct")
    if not normal_target.startswith("chair_") or not high_target.startswith("chair_"):
        raise ValueError("normal/High targets must be stable Chair instance IDs")

    grouped = {
        "bed": sorted(bed_rows, key=lambda row: str(row["motion_id"])),
        "normal_chair": sorted(normal_rows, key=lambda row: str(row["motion_id"])),
        "high_desk_motion": sorted(high_rows, key=lambda row: str(row["motion_id"])),
        "high_chair_legacy_motion": sorted(
            legacy_rows, key=lambda row: str(row["motion_id"])
        ),
    }
    targets = {
        "bed": bed_target,
        "normal_chair": normal_target,
        "high_desk_motion": high_target,
        "high_chair_legacy_motion": high_target,
    }
    if tuple(grouped) != GROUP_ORDER or any(len(grouped[key]) != 6 for key in GROUP_ORDER):
        raise AssertionError("four-stratum grouping contract changed")
    return grouped, targets


def validate_source_metadata_without_arrays(
    source_root: Path, scene_records: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    """Bind the three-scene inventory and High-Desk source split using JSON only.

    This function deliberately does not open points, sidecars, motions,
    positions, or dense-contact NPZ payloads.  It is safe to run before the
    development checkpoint lock and records that room_0201 arrays stayed
    unread.
    """

    source_root = Path(source_root).expanduser().resolve()
    metadata = {}
    for record in scene_records:
        scene_id = str(record.get("scene_id", ""))
        motion_path = resolve_under(source_root, record["motion_index_file"])
        dense_path = resolve_under(source_root, record["dense_index_file"])
        if (
            sha256_file(motion_path) != str(record.get("motion_index_sha256", ""))
            or sha256_file(dense_path) != str(record.get("dense_index_sha256", ""))
        ):
            raise ValueError(f"{scene_id}: source metadata hash changed")
        motion_index = read_json(motion_path)
        dense_index = read_json(dense_path)
        if (
            motion_index.get("schema")
            != "relational_teacher_v7_hd_motion_inventory_v1"
            or motion_index.get("status") != "HD_MOTION_BINDING_PASS"
            or motion_index.get("relation_or_distance_used_as_forward_input")
            is not False
            or dense_index.get("schema") != SOURCE_DENSE_SCHEMA
            or dense_index.get("status") != "DENSE_CONTACT_V2_PASS"
            or dense_index.get("relation_or_distance_used") is not False
            or dense_index.get("motion_count") != 24
        ):
            raise ValueError(f"{scene_id}: source metadata contract changed")
        motion_rows = motion_index.get("rows")
        dense_rows_raw = dense_index.get("rows")
        if not isinstance(motion_rows, list) or not isinstance(dense_rows_raw, list):
            raise ValueError(f"{scene_id}: source metadata rows absent")
        motion_by_id = {str(row["motion_id"]): row for row in motion_rows}
        if len(motion_by_id) != len(motion_rows):
            raise ValueError(f"{scene_id}: duplicate source motion ID")
        dense_rows = []
        for raw in dense_rows_raw:
            row = dict(raw)
            motion_id = str(row["motion_id"])
            if motion_id not in motion_by_id:
                raise ValueError(f"{scene_id}: dense/source motion metadata differs")
            source = motion_by_id[motion_id]
            if (
                "target_instance_id" not in source
                or str(source["target_instance_id"])
                != str(row.get("target_instance_id", ""))
            ):
                raise ValueError(
                    f"{scene_id}/{motion_id}: dense and motion targets differ"
                )
            for key in (
                "source_sample_id",
                "source_group_id",
                "source_collection_id",
                "source_split_role",
                "motion_sha256",
            ):
                if key in source:
                    row[key] = source[key]
            dense_rows.append(row)
        grouped, targets = classify_dense_rows(dense_rows)
        if targets != EXPECTED_SCENE_TARGETS.get(scene_id):
            raise ValueError(f"{scene_id}: exact source target metadata changed")
        high_rows = grouped["high_desk_motion"]
        collection = {str(row.get("source_collection_id", "")) for row in high_rows}
        roles = {str(row.get("source_split_role", "")) for row in high_rows}
        groups = sorted(str(row.get("source_group_id", "")) for row in high_rows)
        samples = sorted(str(row.get("source_sample_id", "")) for row in high_rows)
        hashes = sorted(str(row.get("motion_sha256", "")) for row in high_rows)
        identities = sorted(
            [
                str(row.get("source_collection_id", "")),
                str(row.get("source_group_id", "")),
                str(row.get("source_sample_id", "")),
            ]
            for row in high_rows
        )
        if (
            len(collection) != 1
            or len(roles) != 1
            or len(set(groups)) != 6
            or len(set(samples)) != 6
            or len(set(hashes)) != 6
            or any(not value for value in groups + samples + hashes)
        ):
            raise ValueError(f"{scene_id}: High-Desk provenance metadata changed")
        metadata[scene_id] = {
            "split": str(record["split"]),
            "verified_targets": sorted(set(targets.values())),
            "source_collection_id": next(iter(collection)),
            "source_split_role": next(iter(roles)),
            "source_group_ids": groups,
            "source_sample_ids": samples,
            "source_identities": identities,
            "source_motion_sha256": hashes,
        }

    if set(metadata) != set(EXPECTED_SCENE_TARGETS):
        raise ValueError("three-scene source metadata inventory changed")
    train_a = metadata["room_0101"]
    train_b = metadata["room_0102"]
    development = metadata["room_0201"]
    if (
        train_a["source_collection_id"] != "hc_hd"
        or train_b["source_collection_id"] != "hc_hd"
        or development["source_collection_id"] != "hcw_hdw"
        or train_a["source_split_role"] != "train"
        or train_b["source_split_role"] != "train"
        or development["source_split_role"] != "development"
        or train_a["source_identities"] != train_b["source_identities"]
        or set(train_a["source_group_ids"])
        & set(development["source_group_ids"])
        or set(train_a["source_sample_ids"])
        & set(development["source_sample_ids"])
        or {
            tuple(value) for value in train_a["source_identities"]
        }
        & {
            tuple(value) for value in development["source_identities"]
        }
    ):
        raise ValueError("train/development High-Desk source separation changed")
    return {
        "scene_metadata": metadata,
        "train_rooms_reuse_same_hc_hd_six": True,
        "room_0201_uses_independent_hcw_hdw_six": True,
        "high_desk_train_development_source_overlap": False,
        "source_identity_keys": ["source_collection_id", "source_group_id", "source_sample_id"],
        "scene_local_motion_sha256_used_as_source_identity": False,
        "non_high_desk_development_protocol": (
            "scene_held_out; legacy source families may share canonical recordings"
        ),
        "development_array_payloads_read": False,
    }


def replica_schedule(
    grouped: Mapping[str, Sequence[Mapping[str, object]]]
) -> List[Dict[str, Mapping[str, object]]]:
    """Latin-shift pairing: every source row is used exactly once per stratum."""

    if tuple(grouped) != GROUP_ORDER:
        raise ValueError("group order changed")
    if any(len(grouped[key]) != REPLICA_COUNT for key in GROUP_ORDER):
        raise ValueError("every stratum must contain exactly six rows")
    result: List[Dict[str, Mapping[str, object]]] = []
    for replica in range(REPLICA_COUNT):
        selection = {}
        for offset, group in enumerate(GROUP_ORDER):
            selection[group] = grouped[group][(replica + offset) % REPLICA_COUNT]
        result.append(selection)
    for group in GROUP_ORDER:
        used = [str(row[group]["motion_id"]) for row in result]
        expected = [str(row["motion_id"]) for row in grouped[group]]
        if Counter(used) != Counter(expected):
            raise AssertionError(f"{group}: replica schedule did not use each source once")
    return result


def nearest_rank_quantile(
    arrays: Sequence[np.ndarray],
    numerator: int = CONSENSUS_NUMERATOR,
    denominator: int = CONSENSUS_DENOMINATOR,
) -> np.ndarray:
    """Version-stable nearest-rank quantile without interpolation."""

    if not arrays:
        raise ValueError("cannot aggregate an empty map sequence")
    if not (0 < numerator <= denominator):
        raise ValueError("invalid quantile fraction")
    stack = np.stack([np.asarray(value, dtype=np.float32) for value in arrays], axis=0)
    if stack.ndim != 3 or stack.shape[-1] != CONTACT_DIM:
        raise ValueError(f"expected stacked [K,N,6] maps, got {stack.shape}")
    if not np.isfinite(stack).all() or np.any(stack < 0.0) or np.any(stack > 1.0):
        raise ValueError("source affordance maps must be finite in [0,1]")
    count = stack.shape[0]
    rank = int(math.ceil(float(numerator * count) / float(denominator))) - 1
    result = np.partition(stack, rank, axis=0)[rank]
    return np.ascontiguousarray(result, dtype=np.float32)


def aggregate_all_sittable(
    grouped_rows: Mapping[str, Sequence[Mapping[str, object]]],
    targets: Mapping[str, str],
    maps_by_motion: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    """Build robust group/instance consensus plus six all-target replicas."""

    schedule = replica_schedule(grouped_rows)
    group_consensus: Dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        group_consensus[group] = nearest_rank_quantile(
            [maps_by_motion[str(row["motion_id"])] for row in grouped_rows[group]]
        )

    # Aggregate the four independently recorded strata first.  In particular,
    # the High-Desk and legacy sources share one physical Chair instance; a
    # direct 12-map quantile could erase one motion family.  Their two robust
    # six-map consensuses are therefore joined only after each has passed the
    # two-source (second-largest) consensus rule.
    groups_by_instance: Dict[str, List[np.ndarray]] = defaultdict(list)
    for group in GROUP_ORDER:
        groups_by_instance[str(targets[group])].append(group_consensus[group])
    if sorted(len(values) for values in groups_by_instance.values()) != [1, 1, 2]:
        raise ValueError("per-instance stratum inventory changed")
    instance_consensus = {
        target: np.ascontiguousarray(np.maximum.reduce(values), dtype=np.float32)
        for target, values in sorted(groups_by_instance.items())
    }
    all_consensus = np.maximum.reduce(list(instance_consensus.values())).astype(
        np.float32, copy=False
    )
    replicas = []
    replica_motion_ids = []
    for selection in schedule:
        selected_maps = [
            maps_by_motion[str(selection[group]["motion_id"])] for group in GROUP_ORDER
        ]
        replicas.append(
            np.ascontiguousarray(np.maximum.reduce(selected_maps), dtype=np.float32)
        )
        replica_motion_ids.append(
            {group: str(selection[group]["motion_id"]) for group in GROUP_ORDER}
        )
    return {
        "group_consensus": group_consensus,
        "instance_consensus": instance_consensus,
        "all_consensus": np.ascontiguousarray(all_consensus, dtype=np.float32),
        "replicas": replicas,
        "replica_motion_ids": replica_motion_ids,
    }


def mask_maps_to_target_and_environment(
    grouped_rows: Mapping[str, Sequence[Mapping[str, object]]],
    targets: Mapping[str, str],
    maps_by_motion: Mapping[str, np.ndarray],
    stable: Mapping[str, Tuple[int, int]],
    instance_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Remove accidental heat on other objects while retaining valid floor paths.

    A motion-derived map is evidence for its bound target object and for the
    environment traversed by that motion.  It is not evidence that a nearby TV,
    Desk, unverified Chair, or other object is sittable.  Those other object
    points are zeroed before any union.  Unverified Chair/Bed instances remain
    unknown/ignored, never negative.
    """

    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    if instance_ids.ndim != 1:
        raise ValueError("instance_ids must be a vector")
    result = {}
    for group in GROUP_ORDER:
        target = str(targets[group])
        if target not in stable:
            raise ValueError(f"target absent from stable instance map: {target}")
        numeric = int(stable[target][0])
        keep = (instance_ids == 0) | (instance_ids == numeric)
        for row in grouped_rows[group]:
            motion_id = str(row["motion_id"])
            source = np.asarray(maps_by_motion[motion_id], dtype=np.float32)
            if source.shape != (instance_ids.shape[0], CONTACT_DIM):
                raise ValueError(f"{motion_id}: map/instance shape differs")
            masked = source.copy()
            masked[~keep] = 0.0
            result[motion_id] = np.ascontiguousarray(masked, dtype=np.float32)
    if set(result) != set(maps_by_motion):
        raise ValueError("not every source motion was assigned exactly one target mask")
    return result


def stable_instance_map(
    instances_manifest: Mapping[str, object], scene_id: str
) -> Dict[str, Tuple[int, int]]:
    if instances_manifest.get("schema") != INSTANCE_SCHEMA:
        raise ValueError(f"{scene_id}: unsupported instance manifest schema")
    if str(instances_manifest.get("scene_id")) != str(scene_id):
        raise ValueError(f"{scene_id}: instance manifest scene mismatch")
    objects = instances_manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("instance manifest has no objects")
    result = {}
    numeric_ids = set()
    for row in objects:
        if not isinstance(row, dict):
            raise ValueError("instance manifest row is not an object")
        name = str(row["name"])
        if not SAFE_ID.match(name):
            raise ValueError(f"unsafe stable instance ID: {name}")
        value = (int(row["instance_id"]), int(row["category_id"]))
        category = str(row.get("category", "")).lower()
        if name in result or value[0] <= 0 or value[0] in numeric_ids:
            raise ValueError(f"duplicate/invalid instance: {name}")
        if value[1] not in CATEGORY_NAMES or CATEGORY_NAMES[value[1]] != category:
            raise ValueError(f"{scene_id}/{name}: category name/ID mismatch")
        if category in {"chair", "bed", "tv", "desk"}:
            anchor = np.asarray(row.get("anchor_unity_xz"), dtype=np.float32)
            if anchor.shape != (2,) or not np.isfinite(anchor).all():
                raise ValueError(f"{scene_id}/{name}: invalid relation anchor")
        result[name] = value
        numeric_ids.add(value[0])
    categories = {category for _, category in result.values()}
    if not {CATEGORY_CHAIR, CATEGORY_BED, 4, 5}.issubset(categories):
        raise ValueError(f"{scene_id}: required Chair/Bed/TV/Desk category absent")
    return result


def partition_sittable_instances(
    stable: Mapping[str, Tuple[int, int]], verified: Iterable[str]
) -> Dict[str, List[str]]:
    verified_set = {str(value) for value in verified}
    unknown = sorted(
        name for name, (_, category) in stable.items()
        if category in SITTABLE_CATEGORY_IDS and name not in verified_set
    )
    invalid = sorted(name for name in verified_set if name not in stable)
    if invalid:
        raise ValueError(f"verified targets absent from instance manifest: {invalid}")
    return {
        "verified_sittable": sorted(verified_set),
        "unknown_sittable_not_negative": unknown,
    }


def scene_metrics(
    aggregate: Mapping[str, object],
    targets: Mapping[str, str],
    stable: Mapping[str, Tuple[int, int]],
    instance_ids: np.ndarray,
) -> Dict[str, object]:
    all_consensus = np.asarray(aggregate["all_consensus"], dtype=np.float32)
    instance_consensus = aggregate["instance_consensus"]
    if all_consensus.shape != (POINT_COUNT, CONTACT_DIM):
        raise ValueError(f"all-sittable consensus shape changed: {all_consensus.shape}")
    unique_targets = sorted(set(str(value) for value in targets.values()))
    rows = {}
    for target in unique_targets:
        numeric, category = stable[target]
        expected = CATEGORY_BED if target.startswith("bed_") else CATEGORY_CHAIR
        if category != expected:
            raise ValueError(f"{target}: target semantic category changed")
        mask = np.asarray(instance_ids) == numeric
        if not np.any(mask):
            raise ValueError(f"{target}: target has no sampled points")
        value = np.asarray(instance_consensus[target], dtype=np.float32)
        any_value = value[mask].max(axis=-1)
        pelvis = value[mask, 0]
        other = [
            np.asarray(instance_consensus[name], dtype=np.float32)
            for name in unique_targets if name != target
        ]
        other_max = np.maximum.reduce(other)
        contribution = value > other_max + np.float32(1e-7)
        rows[target] = {
            "numeric_instance_id": int(numeric),
            "category_id": int(category),
            "target_point_count": int(mask.sum()),
            "target_any_max": float(any_value.max()),
            "target_pelvis_max": float(pelvis.max()),
            "target_active_point_count": int((any_value >= ACTIVE_THRESHOLD).sum()),
            "global_active_value_count": int((value >= ACTIVE_THRESHOLD).sum()),
            "unique_union_contribution_count": int(contribution.sum()),
        }
        if rows[target]["target_pelvis_max"] < 0.75:
            raise ValueError(f"{target}: robust pelvis response is below 0.75")
        if rows[target]["target_active_point_count"] <= 0:
            raise ValueError(f"{target}: robust target support is empty")
        if rows[target]["unique_union_contribution_count"] <= 0:
            raise ValueError(f"{target}: target contributes nothing unique to all-sit GT")
    return {
        "active_threshold": ACTIVE_THRESHOLD,
        "instances": rows,
        "all_consensus_active_value_count": int(
            (all_consensus >= ACTIVE_THRESHOLD).sum()
        ),
    }


def assert_all_sittable_contract(
    aggregate: Mapping[str, object], targets: Mapping[str, str]
) -> None:
    instance_consensus = aggregate["instance_consensus"]
    all_consensus = np.asarray(aggregate["all_consensus"])
    expected = np.maximum.reduce(
        [np.asarray(instance_consensus[key]) for key in sorted(instance_consensus)]
    )
    if not np.array_equal(all_consensus, expected):
        raise ValueError("all-sittable consensus is not exact instance-wise max")
    if len(set(targets.values())) != 3:
        raise ValueError("all-sittable target must contain three physical instances")
    replicas = aggregate["replicas"]
    if len(replicas) != REPLICA_COUNT:
        raise ValueError("all-sittable replica count changed")
    for replica in replicas:
        value = np.asarray(replica)
        if value.shape[-1] != CONTACT_DIM:
            raise ValueError("replica contact dimension changed")
        if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError("replica range changed")


def load_source_scene(
    source_root: Path, scene_record: Mapping[str, object]
) -> Dict[str, object]:
    """Deep-load one sealed v7 scene and bind every dense source hash."""

    source_root = Path(source_root).expanduser().resolve()
    scene_id = str(scene_record.get("scene_id", ""))
    if not scene_id:
        raise ValueError("source scene ID is empty")
    bound_files = {}
    for key in ("points", "sidecar", "instances", "motion_index", "dense_index"):
        file_key = key + "_file"
        hash_key = key + "_sha256"
        path = resolve_under(source_root, scene_record[file_key])
        digest = sha256_file(path)
        if digest != str(scene_record.get(hash_key, "")):
            raise ValueError(f"{scene_id}: {key} hash changed")
        bound_files[key] = path

    with np.load(bound_files["points"], allow_pickle=False) as source:
        if "points" not in source.files:
            raise ValueError(f"{scene_id}: points array absent")
        points = source["points"].astype(np.float32)
    if points.shape != (POINT_COUNT, 6) or not np.isfinite(points).all():
        raise ValueError(f"{scene_id}: point tensor contract changed")

    with np.load(bound_files["sidecar"], allow_pickle=False) as source:
        required = {"xyz_afford_z_up", "source_indices", "instance_ids", "category_ids"}
        if not required.issubset(set(source.files)):
            raise ValueError(f"{scene_id}: sidecar arrays changed")
        xyz = source["xyz_afford_z_up"].astype(np.float32)
        source_indices = source["source_indices"].astype(np.int64)
        instance_ids = source["instance_ids"].astype(np.int64)
        category_ids = source["category_ids"].astype(np.int64)
    if (
        xyz.shape != (POINT_COUNT, 3)
        or source_indices.shape != (POINT_COUNT,)
        or instance_ids.shape != (POINT_COUNT,)
        or category_ids.shape != (POINT_COUNT,)
        or not np.allclose(points[:, :3], xyz, rtol=0.0, atol=1e-6)
        or np.unique(source_indices).size != POINT_COUNT
    ):
        raise ValueError(f"{scene_id}: point/sidecar order contract changed")

    instances_manifest = read_json(bound_files["instances"])
    stable = stable_instance_map(instances_manifest, scene_id)
    if np.any(instance_ids < 0) or not set(np.unique(category_ids)).issubset(
        set(CATEGORY_NAMES)
    ):
        raise ValueError(f"{scene_id}: invalid sampled instance/category ID")
    if np.any(category_ids[instance_ids == 0] != CATEGORY_ENVIRONMENT):
        raise ValueError(f"{scene_id}: environment points have a non-environment category")
    numeric_to_category = {numeric: category for numeric, category in stable.values()}
    sampled_instances = {int(value) for value in np.unique(instance_ids) if int(value) != 0}
    if sampled_instances != set(numeric_to_category):
        raise ValueError(
            f"{scene_id}: sampled/manifest instance inventory differs"
        )
    for numeric, category in numeric_to_category.items():
        if not np.all(category_ids[instance_ids == numeric] == category):
            raise ValueError(f"{scene_id}: category mismatch for instance {numeric}")
    motion_index = read_json(bound_files["motion_index"])
    if (
        motion_index.get("schema") != "relational_teacher_v7_hd_motion_inventory_v1"
        or motion_index.get("status") != "HD_MOTION_BINDING_PASS"
        or motion_index.get("relation_or_distance_used_as_forward_input") is not False
    ):
        raise ValueError(f"{scene_id}: motion inventory contract changed")
    motion_rows = motion_index.get("rows")
    if not isinstance(motion_rows, list):
        raise ValueError(f"{scene_id}: motion inventory rows absent")
    motion_by_id = {str(row["motion_id"]): row for row in motion_rows}
    if len(motion_by_id) != len(motion_rows):
        raise ValueError(f"{scene_id}: duplicate motion inventory ID")

    dense_index = read_json(bound_files["dense_index"])
    if (
        dense_index.get("schema") != SOURCE_DENSE_SCHEMA
        or dense_index.get("status") != "DENSE_CONTACT_V2_PASS"
        or str(dense_index.get("scene_id")) != scene_id
        or dense_index.get("relation_or_distance_used") is not False
        or dense_index.get("motion_count") != 24
        or dense_index.get("prompt_expanded_row_count") != 48
    ):
        raise ValueError(f"{scene_id}: sealed dense index contract changed")
    raw_dense_rows = dense_index.get("rows")
    if not isinstance(raw_dense_rows, list):
        raise ValueError(f"{scene_id}: dense rows absent")
    dense_rows = []
    for raw_row in raw_dense_rows:
        row = dict(raw_row)
        motion_id = str(row["motion_id"])
        if motion_id not in motion_by_id:
            raise ValueError(f"{scene_id}: dense motion absent from motion inventory")
        source = motion_by_id[motion_id]
        if (
            "target_instance_id" not in source
            or str(source["target_instance_id"])
            != str(row.get("target_instance_id", ""))
        ):
            raise ValueError(
                f"{scene_id}/{motion_id}: dense and motion targets differ"
            )
        for key in (
            "source_sample_id",
            "source_group_id",
            "source_collection_id",
            "source_split_role",
        ):
            if key in source:
                row[key] = source[key]
        dense_rows.append(row)
    grouped, targets = classify_dense_rows(dense_rows)
    expected_targets = EXPECTED_SCENE_TARGETS.get(scene_id)
    if expected_targets is None or targets != expected_targets:
        raise ValueError(
            f"{scene_id}: exact verified target binding changed: {targets}"
        )

    maps_by_motion = {}
    source_bindings = []
    for row in dense_rows:
        motion_id = str(row["motion_id"])
        manifest_path = resolve_under(source_root, row["manifest_file"])
        manifest_hash = sha256_file(manifest_path)
        if manifest_hash != str(row.get("manifest_sha256", "")):
            raise ValueError(f"{scene_id}/{motion_id}: manifest hash changed")
        manifest = read_json(manifest_path)
        target_instance_id = str(row["target_instance_id"])
        target_numeric_instance_id = stable[target_instance_id][0]
        if (
            manifest.get("schema") != SOURCE_DENSE_SCHEMA + "_item"
            or manifest.get("status") != "DENSE_CONTACT_ITEM_PASS"
            or str(manifest.get("scene_id")) != scene_id
            or str(manifest.get("motion_id")) != motion_id
            or str(manifest.get("target_instance_id"))
            != target_instance_id
            or type(manifest.get("target_numeric_instance_id")) is not int
            or manifest.get("target_numeric_instance_id")
            != target_numeric_instance_id
            or manifest.get("relation_or_distance_used") is not False
            or manifest.get("compatible_prompt_ids") != sorted(PROMPTS)
        ):
            raise ValueError(f"{scene_id}/{motion_id}: dense item contract changed")
        source_motion_path = resolve_under(source_root, manifest["source_motion_file"])
        positions_path = resolve_under(source_root, manifest["positions_file"])
        motion_row_path = (
            bound_files["motion_index"].parent
            / str(motion_by_id[motion_id].get("motion_file", ""))
        ).resolve()
        try:
            motion_row_path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"{scene_id}/{motion_id}: motion inventory path escapes source root"
            ) from exc
        if motion_row_path != source_motion_path:
            raise ValueError(
                f"{scene_id}/{motion_id}: dense and inventory motion paths differ"
            )
        source_motion_hash = sha256_file(source_motion_path)
        positions_hash = sha256_file(positions_path)
        if (
            source_motion_hash != str(manifest.get("source_motion_sha256", ""))
            or source_motion_hash != str(motion_by_id[motion_id].get("motion_sha256", ""))
            or positions_hash != str(manifest.get("positions_sha256", ""))
        ):
            raise ValueError(f"{scene_id}/{motion_id}: motion/position provenance changed")
        with np.load(positions_path, allow_pickle=False) as source:
            position_keys = {
                "joint_positions22_adm_chair_local_z_up",
                "source_times_s",
                "target_times_s",
                "source_frame_start_inclusive",
                "source_frame_end_inclusive",
                "source_fps",
                "target_fps",
            }
            if set(source.files) != position_keys:
                raise ValueError(f"{scene_id}/{motion_id}: position arrays changed")
            positions = source["joint_positions22_adm_chair_local_z_up"]
        if (
            positions.ndim != 3
            or positions.shape[1:] != (22, 3)
            or positions.shape[0] <= 0
            or not np.isfinite(positions).all()
        ):
            raise ValueError(f"{scene_id}/{motion_id}: position tensor changed")
        affordance_path = resolve_under(source_root, manifest["affordance_file"])
        affordance_hash = sha256_file(affordance_path)
        if affordance_hash != str(manifest.get("affordance_sha256", "")):
            raise ValueError(f"{scene_id}/{motion_id}: affordance hash changed")
        with np.load(affordance_path, allow_pickle=False) as source:
            required = {
                "xyz",
                "instance_ids",
                "source_indices",
                "distance",
                "affordance",
                "contact_joints",
                "contact_joint_names",
                "sigma",
                "gt_root_start_xyz",
            }
            if set(source.files) != required:
                raise ValueError(f"{scene_id}/{motion_id}: dense arrays changed")
            dense_xyz = source["xyz"].astype(np.float32)
            affordance = source["affordance"].astype(np.float32)
            dense_instances = source["instance_ids"].astype(np.int64)
            dense_sources = source["source_indices"].astype(np.int64)
            dense_distance = source["distance"].astype(np.float32)
            sigma = float(source["sigma"])
        if (
            affordance.shape != (POINT_COUNT, CONTACT_DIM)
            or dense_distance.shape != (POINT_COUNT, CONTACT_DIM)
            or not np.isfinite(affordance).all()
            or not np.isfinite(dense_distance).all()
            or np.any(dense_distance < 0.0)
            or np.any(affordance < 0.0)
            or np.any(affordance > 1.0)
            or not np.array_equal(dense_xyz, xyz)
            or not np.array_equal(dense_instances, instance_ids)
            or not np.array_equal(dense_sources, source_indices)
            or not np.isclose(sigma, 0.8, rtol=0.0, atol=1e-7)
        ):
            raise ValueError(f"{scene_id}/{motion_id}: dense tensor contract changed")
        maps_by_motion[motion_id] = affordance
        source_bindings.append(
            {
                "motion_id": motion_id,
                "target_instance_id": str(row["target_instance_id"]),
                "manifest_file": str(manifest_path.relative_to(source_root)),
                "manifest_sha256": manifest_hash,
                "affordance_file": str(affordance_path.relative_to(source_root)),
                "affordance_sha256": affordance_hash,
                "positions_file": str(positions_path.relative_to(source_root)),
                "positions_sha256": positions_hash,
                "source_motion_file": str(
                    source_motion_path.relative_to(source_root)
                ),
                "source_sample_id": str(
                    motion_by_id[motion_id].get("source_sample_id", "")
                ),
                "source_group_id": str(
                    motion_by_id[motion_id].get("source_group_id", "")
                ),
                "source_collection_id": str(
                    motion_by_id[motion_id].get("source_collection_id", "")
                ),
                "source_motion_sha256": source_motion_hash,
            }
        )

    verified = set(targets.values())
    partition = partition_sittable_instances(stable, verified)
    return {
        "scene_id": scene_id,
        "split": str(scene_record["split"]),
        "points": points,
        "xyz": xyz,
        "source_indices": source_indices,
        "instance_ids": instance_ids,
        "category_ids": category_ids,
        "stable_instances": stable,
        "grouped_rows": grouped,
        "targets": targets,
        "maps_by_motion": maps_by_motion,
        "source_bindings": sorted(source_bindings, key=lambda row: row["motion_id"]),
        "partition": partition,
        "source_files": {
            key + "_file": str(path.relative_to(source_root))
            for key, path in bound_files.items()
        },
        "source_hashes": {
            key + "_sha256": sha256_file(path)
            for key, path in bound_files.items()
        },
    }


__all__ = [
    "ACTIVE_THRESHOLD",
    "CONTACT_DIM",
    "EXPECTED_SCENE_TARGETS",
    "GROUP_ORDER",
    "INSTANCE_SCHEMA",
    "OUTPUT_DATASET_SCHEMA",
    "POINT_COUNT",
    "PROMPTS",
    "REPLICA_COUNT",
    "SCENE_MANIFEST_SCHEMA",
    "SOURCE_DATASET_SCHEMA",
    "SOURCE_DENSE_SCHEMA",
    "TEACHER_FORWARD_INPUTS",
    "aggregate_all_sittable",
    "assert_all_sittable_contract",
    "assert_teacher_forward_inputs",
    "atomic_savez",
    "atomic_write_json",
    "classify_dense_rows",
    "load_source_scene",
    "mask_maps_to_target_and_environment",
    "partition_sittable_instances",
    "read_json",
    "replica_schedule",
    "row_is_high_desk",
    "resolve_under",
    "scene_metrics",
    "sha256_file",
    "stable_instance_map",
    "validate_source_metadata_without_arrays",
]

#!/usr/bin/env python3
"""Deterministic unit and synthetic end-to-end tests for IIW validation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from validate_iiw_gt import (  # noqa: E402
    BODY_PART_NAMES,
    CMDM_PROXY_FROM_NATIVE,
    EXPECTED_BODY_GROUP_COUNTS,
    EXPECTED_BODY_JOINT_GROUPS,
    EXPECTED_CMDM_HORIZON,
    NUM_BODY_PARTS,
    NUM_POINTS,
    load_and_validate_sample,
    top_fraction_mean,
    validate_frame_partition,
)


NUM_PHASES = 8
VALID_FRAMES = 118


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def phase_contract() -> Tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(
        [
            (phase + 1) * VALID_FRAMES // NUM_PHASES
            - phase * VALID_FRAMES // NUM_PHASES
            for phase in range(NUM_PHASES)
        ],
        dtype=np.int64,
    )
    mapping = np.repeat(np.arange(NUM_PHASES, dtype=np.int64), counts)
    return counts, mapping


def synthetic_native_target(instance_ids: np.ndarray) -> np.ndarray:
    """Make sparse, temporally varying targets with stronger late chair/base."""
    value = np.zeros(
        (NUM_PHASES, NUM_POINTS, NUM_BODY_PARTS), dtype=np.float32
    )
    chair = np.flatnonzero(instance_ids == 1)[:256]
    environment = np.flatnonzero(instance_ids == 0)
    for phase in range(NUM_PHASES):
        value[phase, chair, 0] = 0.72 + 0.03 * phase
        for body in range(1, NUM_BODY_PARTS):
            offset = body * 32
            region = environment[offset : offset + 64]
            value[phase, region, body] = 0.15 + 0.02 * phase
    return value


def build_dataset(root: Path) -> Tuple[Path, Dict, np.ndarray, np.ndarray, np.ndarray]:
    sample_id = "synthetic_sit_chair_0001"
    scene_rel = Path("scenes/room_synthetic/adm_input")
    scene_dir = root / scene_rel
    sample_dir = root / "samples" / sample_id
    target_rel = Path("samples") / sample_id / "iiw_gt" / "iiw_gt.npz"
    manifest_rel = Path("samples") / sample_id / "iiw_gt" / "manifest.json"
    motion_rel = Path("samples") / sample_id / "cmdm_motion_input.npz"
    scene_dir.mkdir(parents=True)
    (root / target_rel).parent.mkdir(parents=True)

    rng = np.random.default_rng(20260812)
    xyz = rng.normal(size=(NUM_POINTS, 3)).astype(np.float32)
    quotas = (2934, 1503, 1502, 1502, 751)
    instance_ids = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(quotas)]
    )
    source_indices = rng.permutation(NUM_POINTS).astype(np.int64)
    np.savez_compressed(
        scene_dir / "sidecar.npz",
        xyz_afford_z_up=xyz,
        instance_ids=instance_ids,
        source_indices=source_indices,
    )

    native_max = synthetic_native_target(instance_ids).astype(np.float16)
    native_mean = (native_max.astype(np.float32) * 0.5).astype(np.float16)
    counts, frame_to_phase = phase_contract()
    np.savez_compressed(
        root / target_rel,
        iiw_native_phase_max=native_max,
        iiw_native_phase_mean=native_mean,
        iiw_cmdm_proxy_phase_max=native_max[..., CMDM_PROXY_FROM_NATIVE],
        iiw_cmdm_proxy_phase_mean=native_mean[..., CMDM_PROXY_FROM_NATIVE],
        phase_frame_counts=counts,
        frame_to_phase=frame_to_phase,
        phase_mask=np.zeros((NUM_PHASES,), dtype=bool),
        scene_xyz_adm=xyz,
        instance_ids=instance_ids,
        source_indices=source_indices,
        valid_frames=np.asarray(VALID_FRAMES, dtype=np.int64),
        body_joint_groups=EXPECTED_BODY_JOINT_GROUPS,
        body_group_counts=EXPECTED_BODY_GROUP_COUNTS,
        body_part_names=np.asarray(BODY_PART_NAMES),
        cmdm_proxy_from_native=CMDM_PROXY_FROM_NATIVE,
    )
    mask = np.ones((EXPECTED_CMDM_HORIZON,), dtype=bool)
    mask[:VALID_FRAMES] = False
    np.savez_compressed(root / motion_rel, x_mask=mask)

    manifest = {
        "method": "point_aligned_iiw_proxy",
        "sample_id": sample_id,
        "coordinate_frame": "chair_local_z_up",
        "approximation_disclosure": (
            "Synthetic target is not original InterFaceRays IIW."
        ),
        "target_semantics": "continuous_distance_kernel_without_contact_mask",
        "unity_contact_label_masking_applied": False,
        "num_phases": NUM_PHASES,
        "valid_frames": VALID_FRAMES,
        "num_scene_points": NUM_POINTS,
        "native_body_part_order": list(BODY_PART_NAMES),
        "cmdm_proxy_from_native": CMDM_PROXY_FROM_NATIVE.tolist(),
        "phase_frame_counts": counts.tolist(),
        "phase_frame_ranges": [
            [phase * VALID_FRAMES // NUM_PHASES,
             (phase + 1) * VALID_FRAMES // NUM_PHASES]
            for phase in range(NUM_PHASES)
        ],
        "iiw_gt_ready": True,
    }
    write_json(root / manifest_rel, manifest)
    entry = {
        "sample_id": sample_id,
        "scene_id": "room_synthetic",
        "target_name": "chair",
        "target_instance_id": 1,
        "valid_frames": VALID_FRAMES,
        "cmdm_motion_input": str(motion_rel),
        "iiw_gt": str(target_rel),
        "iiw_gt_manifest": str(manifest_rel),
        "iiw_gt_ready": True,
    }
    index = {
        "scene_id": "room_synthetic",
        "scene_adm_input": str(scene_rel),
        "num_samples": 1,
        "samples": [entry],
        "iiw_gt_ready": True,
        "iiw_gt_method": "point_aligned_iiw_proxy",
        "iiw_num_phases": NUM_PHASES,
    }
    index_file = root / "index_room_synthetic_iiw.json"
    write_json(index_file, index)
    return index_file, entry, xyz, instance_ids, source_indices


def test_small_units() -> None:
    actual = top_fraction_mean(np.arange(10, dtype=np.float32), 0.2)
    assert actual == 8.5
    counts, mapping = phase_contract()
    validate_frame_partition(mapping, counts, VALID_FRAMES, NUM_PHASES)
    broken = mapping.copy()
    boundary = int(counts[0])
    broken[boundary - 1], broken[boundary] = (
        broken[boundary],
        broken[boundary - 1],
    )
    try:
        validate_frame_partition(broken, counts, VALID_FRAMES, NUM_PHASES)
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous temporal-bin mapping was accepted")
    print("[PASS] IIW diagnostic reducers and temporal-bin contract")


def test_synthetic_end_to_end() -> None:
    with tempfile.TemporaryDirectory(prefix="iiw_validate_") as directory:
        root = Path(directory)
        index_file, entry, xyz, instance_ids, source_indices = build_dataset(root)
        output_dir = root / "validation_output"
        completed = subprocess.run(
            [
                sys.executable,
                str(THIS_DIR / "validate_iiw_gt.py"),
                "--dataset-root",
                str(root),
                "--index",
                index_file.name,
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if "[PASS] IIW GT structural contract" not in completed.stdout:
            raise AssertionError(completed.stdout)
        summary = json.loads((output_dir / "summary.json").read_text())
        assert summary["status"] == "PASS"
        assert summary["num_samples"] == 1
        assert summary["semantic_checks"][
            "all_chair_targets_have_stronger_late_base_chair_signal"
        ]
        assert (output_dir / "per_phase_body_region.csv").is_file()
        print("[PASS] synthetic 8x8192x6 IIW validation end-to-end")

        target_file = root / entry["iiw_gt"]
        with np.load(target_file, allow_pickle=False) as target:
            arrays = {key: target[key] for key in target.files}
        corrupted = arrays["iiw_cmdm_proxy_phase_max"].copy()
        corrupted[0, 0, 0] = np.float16(1.0)
        arrays["iiw_cmdm_proxy_phase_max"] = corrupted
        np.savez_compressed(target_file, **arrays)
        try:
            load_and_validate_sample(
                root=root,
                entry=entry,
                scene_xyz=xyz,
                instance_ids=instance_ids,
                source_indices=source_indices,
                num_phases=NUM_PHASES,
                threshold=0.7,
            )
        except ValueError as error:
            assert "proxy permutation mismatch" in str(error)
        else:
            raise AssertionError("corrupted native-to-proxy mapping was accepted")
        print("[PASS] corrupted native-to-CMDM channel mapping is rejected")


def main() -> None:
    test_small_units()
    test_synthetic_end_to_end()
    print("[PASS] validate_iiw_gt unit and synthetic E2E contract")


if __name__ == "__main__":
    main()

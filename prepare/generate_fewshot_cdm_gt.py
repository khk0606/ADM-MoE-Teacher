#!/usr/bin/env python3
"""Generate point-aligned CDM GT for leakage-free Base samples only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fewshot_cdm_common import (
    CONTACT_JOINT_NAMES,
    CONTACT_JOINTS,
    compute_distance_map,
    distance_to_affordance,
    gt_file_from_entry,
    load_index_entries,
    load_motion_xyz,
    load_scene,
    load_split,
    sample_dir_from_entry,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/history_affordance_v1")
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(
            "data/history_affordance_v1/splits/"
            "chair23_bed2_whiteboard12_multistart24_v1.json"
        ),
    )
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--start-tolerance", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    split_file = args.split.expanduser().resolve()
    split = load_split(split_file)
    train_ids = [str(v) for v in split["cdm_fewshot"]["train"]]
    test_ids = [str(v) for v in split["cdm_fewshot"]["test"]]
    sample_ids = train_ids + test_ids
    entries = load_index_entries(dataset_root, sample_ids)
    split_sha256 = sha256_file(split_file)
    scene_cache = {}
    catalog = []

    for sample_id in sorted(sample_ids):
        entry = entries[sample_id]
        scene_id = str(entry["scene_id"])
        scene = scene_cache.get(scene_id)
        if scene is None:
            scene = load_scene(dataset_root, entry)
            scene_cache[scene_id] = scene
        motion_xyz = load_motion_xyz(dataset_root, entry)
        start = np.asarray(
            entry["start_position_adm_chair_local_xyz"], dtype=np.float32
        )
        start_error = float(np.linalg.norm(motion_xyz[0, 0] - start))
        if start_error > args.start_tolerance:
            raise RuntimeError(
                f"{sample_id}: GT/startpoint mismatch {start_error:.6f}m > "
                f"{args.start_tolerance:.6f}m"
            )
        distance = compute_distance_map(
            scene["xyz"], motion_xyz, chunk_size=args.chunk_size
        )
        affordance = distance_to_affordance(distance, args.sigma)
        output = gt_file_from_entry(dataset_root, entry)
        if output.exists() and not args.overwrite:
            existing = np.load(output, allow_pickle=False)
            required_keys = {"source_indices", "instance_ids", "affordance", "sigma"}
            exact = (
                required_keys.issubset(existing.files)
                and np.array_equal(existing["source_indices"], scene["source_indices"])
                and np.array_equal(existing["instance_ids"], scene["instance_ids"])
                and existing["affordance"].shape == (8192, 6)
                and float(existing["sigma"]) == float(args.sigma)
            )
            if not exact:
                raise FileExistsError(
                    f"{output}: existing GT does not match current scene/sigma; "
                    "rerun with --overwrite only after reviewing provenance"
                )
            print(f"[SKIP] verified existing GT: {sample_id}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                xyz=scene["xyz"],
                instance_ids=scene["instance_ids"],
                source_indices=scene["source_indices"],
                distance=distance,
                affordance=affordance,
                contact_joints=CONTACT_JOINTS,
                contact_joint_names=CONTACT_JOINT_NAMES,
                sigma=np.asarray(args.sigma, dtype=np.float32),
                gt_root_start_xyz=motion_xyz[0, 0],
                expected_start_xyz=start,
                start_position_error_m=np.asarray(start_error, dtype=np.float32),
            )
            manifest = {
                "schema": "history_affordance_v1_fewshot_cdm_gt_v1",
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": "train" if sample_id in set(train_ids) else "test",
                "sigma": float(args.sigma),
                "point_count": 8192,
                "motion_frame_count": int(motion_xyz.shape[0]),
                "contact_joints": CONTACT_JOINTS.tolist(),
                "contact_joint_names": CONTACT_JOINT_NAMES.tolist(),
                "start_position_error_m": start_error,
                "split_file": str(split_file),
                "split_sha256": split_sha256,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            (output.parent / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[PASS] {sample_id}: frames={motion_xyz.shape[0]}, "
                f"range={affordance.min():.6f}..{affordance.max():.6f}, "
                f"start_error={start_error:.6f}m"
            )
        catalog.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "partition": "train" if sample_id in set(train_ids) else "test",
                "gt_file": str(output.relative_to(dataset_root)),
            }
        )

    catalog_file = dataset_root / "fewshot_cdm_gt_catalog.json"
    catalog_file.write_text(
        json.dumps(
            {
                "schema": "history_affordance_v1_fewshot_cdm_gt_catalog_v1",
                "split_file": str(split_file),
                "split_sha256": split_sha256,
                "sigma": float(args.sigma),
                "num_train": len(train_ids),
                "num_test": len(test_ids),
                "samples": catalog,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] generated/verified {len(catalog)} Base-sample CDM targets")
    print(f"[OK] train/test = {len(train_ids)}/{len(test_ids)}")
    print(f"[OK] saved: {catalog_file}")


if __name__ == "__main__":
    main()

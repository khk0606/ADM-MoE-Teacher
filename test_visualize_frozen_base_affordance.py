#!/usr/bin/env python3
"""Synthetic end-to-end test for the frozen Base visualization tool."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import visualize_frozen_base_affordance as visualizer


PAIRS = (
    ("room_0001", "sit_generic_v1", "chair", "Sit somewhere."),
    ("room_0002", "sit_generic_v1", "chair", "Sit somewhere."),
    ("room_0003", "lie_generic_v1", "bed", "Lie down somewhere."),
    (
        "room_0004",
        "write_right_hand_generic_v1",
        "whiteboard",
        "Write on a nearby vertical surface with the right hand.",
    ),
)
PRODUCTION_ARTIFACT_SCHEMA = "history_affordance_v2_base_teacher_v1"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    assert visualizer.ARTIFACT_SCHEMA == PRODUCTION_ARTIFACT_SCHEMA
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact_root = root / "base_teacher_v2_staging"
        pair_rows = []
        x, y = np.meshgrid(
            np.linspace(-2.0, 2.0, 128, dtype=np.float32),
            np.linspace(-1.0, 1.0, 64, dtype=np.float32),
        )
        xyz = np.stack(
            [x.reshape(-1), y.reshape(-1), np.zeros(8192, dtype=np.float32)],
            axis=1,
        ).astype(np.float32)
        rgb = np.stack(
            [
                (x.reshape(-1) + 2.0) / 4.0,
                (y.reshape(-1) + 1.0) / 2.0,
                np.full(8192, 0.4, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        instances = np.zeros(8192, dtype=np.int64)
        instances[x.reshape(-1) < -0.7] = 1
        instances[(x.reshape(-1) >= -0.7) & (x.reshape(-1) < 0.7)] = 2
        instances[x.reshape(-1) >= 0.7] = 3
        source_indices = np.arange(8192, dtype=np.int64)
        for pair_index, (scene_id, prompt_id, target, text) in enumerate(PAIRS):
            cache_key = ("{:064x}".format(pair_index + 1))[-64:]
            output = artifact_root / cache_key
            output.mkdir(parents=True)
            center = -1.2 + 0.8 * pair_index
            distance = np.abs(x.reshape(-1) - center)
            base = np.exp(-0.5 * (distance / 0.55) ** 2).astype(np.float32)
            channel_scale = np.linspace(0.55, 1.0, 6, dtype=np.float32)
            nominal = np.clip(base[:, None] * channel_scale[None, :], 0.0, 1.0)
            draws = np.stack(
                [np.clip(nominal + (draw - 2) * 0.002, 0.0, 1.0) for draw in range(5)],
                axis=0,
            ).astype(np.float32)
            affordance = draws.mean(axis=0, dtype=np.float64).astype(np.float32)
            affordance_std = draws.std(axis=0, dtype=np.float64).astype(np.float32)
            arrays = {
                "affordance": affordance,
                "affordance_std": affordance_std,
                "affordance_draws": draws,
                "xyz": xyz,
                "rgb01": rgb,
                "instance_ids": instances,
                "source_indices": source_indices,
                "channel_names": np.asarray(visualizer.CHANNEL_ORDER),
                "channel_joint_indices": np.asarray(
                    visualizer.CHANNEL_JOINT_INDICES, dtype=np.int64
                ),
                "draw_indices": np.arange(5, dtype=np.int64),
                "initial_noise_seeds": np.arange(10, 15, dtype=np.int64),
                "reverse_noise_seeds": np.arange(20, 25, dtype=np.int64),
            }
            artifact_file = output / "base_teacher.npz"
            np.savez_compressed(artifact_file, **arrays)
            manifest = {
                "schema": PRODUCTION_ARTIFACT_SCHEMA,
                "status": "STAGED",
                "cache_key": cache_key,
                "artifact_file_sha256": visualizer.sha256_file(artifact_file),
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "text": text,
                "channel_order": list(visualizer.CHANNEL_ORDER),
                "channel_joint_indices": list(visualizer.CHANNEL_JOINT_INDICES),
                "shape": [8192, 6],
                "draw_shape": [5, 8192, 6],
                "draw_count": 5,
                "normalization_state": "denormalized_clipped",
                "conditioning": {"history_conditioned": False},
                "array_sha256": {
                    name: visualizer.sha256_array(value)
                    for name, value in arrays.items()
                },
            }
            manifest_file = output / "manifest.json"
            write_json(manifest_file, manifest)
            pair_rows.append(
                {
                    "cache_key": cache_key,
                    "scene_id": scene_id,
                    "target": target,
                    "prompt_id": prompt_id,
                    "text": text,
                    "manifest_file": str(manifest_file.relative_to(root)),
                    "manifest_sha256": visualizer.sha256_file(manifest_file),
                    "artifact_file": str(artifact_file.relative_to(root)),
                    "artifact_sha256": visualizer.sha256_file(artifact_file),
                }
            )
        index = {
            "schema": visualizer.INDEX_SCHEMA,
            "status": "PASS",
            "promotion_authorized": True,
            "dataset_root": str(root),
            "index_id": "synthetic",
            "num_pairs": 4,
            "num_samples": 61,
            "channel_order": list(visualizer.CHANNEL_ORDER),
            "channel_joint_indices": list(visualizer.CHANNEL_JOINT_INDICES),
            "checks": {"synthetic": True},
            "pairs": pair_rows,
        }
        index_file = root / "index.json"
        write_json(index_file, index)
        output_dir = root / "visualizations"
        subprocess.run(
            [
                str(Path(__import__("sys").executable)),
                str(Path(visualizer.__file__).resolve()),
                "--index",
                str(index_file),
                "--output-dir",
                str(output_dir),
                "--point-size",
                "0.5",
                "--dpi",
                "72",
            ],
            check=True,
        )
        report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert report["status"] == "PASS"
        assert report["num_pairs"] == 4
        assert report["gpu_inference_used"] is False
        assert (output_dir / "base_affordance_overview.png").stat().st_size > 0
        assert len(list(output_dir.glob("*__six_channels.png"))) == 4
        assert len(list((output_dir / "ply").rglob("*.ply"))) == 28
        print("[PASS] synthetic four-pair PNG/PLY visualization round trip")


if __name__ == "__main__":
    main()

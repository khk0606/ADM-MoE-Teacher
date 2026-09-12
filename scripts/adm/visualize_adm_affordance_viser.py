#!/usr/bin/env python3
"""Inspect original ADM contact-map test outputs in an interactive Viser scene."""

from __future__ import annotations

import sys
from pathlib import Path as _RepositoryPath
sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser


CHANNEL_NAMES = (
    "pelvis",
    "left_foot",
    "right_foot",
    "neck",
    "left_wrist",
    "right_wrist",
)


@dataclass(frozen=True)
class Sample:
    index: int
    prediction_path: Path
    points_path: Path
    scene_id: str
    utterance: str

    @property
    def label(self) -> str:
        text = self.utterance.strip().replace("\n", " ")
        if len(text) > 58:
            text = text[:55] + "..."
        suffix = " · " + text if text else ""
        return f"{self.index:05d} · {self.scene_id}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize original ADM test outputs as point-aligned affordance maps."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="ADM test directory containing custom/pred_contact/*.npy",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--initial-sample", type=int, default=None)
    return parser.parse_args()


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"scene point file is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "points" not in payload:
            raise ValueError(f"{path} does not contain a 'points' array")
        points = np.asarray(payload["points"])
    if points.ndim != 2 or points.shape[1] < 6:
        raise ValueError(f"{path} must have shape [N, >=6], got {points.shape}")
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    rgb = np.asarray(points[:, 3:6], dtype=np.float32)
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError(f"{path} contains non-finite point or color values")
    if float(rgb.max(initial=0.0)) <= 1.0:
        rgb = rgb * 255.0
    return xyz, np.rint(rgb).clip(0, 255).astype(np.uint8)


def load_distances(path: Path, point_count: int) -> np.ndarray:
    distances = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if distances.ndim == 2:
        distances = distances[None, ...]
    if distances.ndim != 3:
        raise ValueError(
            f"{path} must have shape [N, 6] or [K, N, 6], got {distances.shape}"
        )
    if distances.shape[1] != point_count or distances.shape[2] != len(CHANNEL_NAMES):
        raise ValueError(
            f"{path} must match [{point_count}, {len(CHANNEL_NAMES)}] per generation, "
            f"got {distances.shape}"
        )
    if np.isnan(distances).any() or np.any(distances < 0.0):
        raise ValueError(f"{path} contains NaN or negative distance values")
    return distances


def distance_to_affordance(distances: np.ndarray, sigma: float) -> np.ndarray:
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a finite positive number")
    safe = np.minimum(distances.astype(np.float64), 1.0e6)
    values = np.exp(-(safe * safe) / (2.0 * sigma * sigma))
    return values.astype(np.float32)


def discover_samples(eval_dir: Path, data_root: Path) -> list[Sample]:
    prediction_dir = eval_dir / "custom" / "pred_contact"
    prediction_paths = sorted(prediction_dir.glob("*.npy"))
    if not prediction_paths:
        raise FileNotFoundError(f"no ADM prediction files found in {prediction_dir}")

    annotation_path = data_root / "custom" / "anno.csv"
    annotations: list[dict[str, str]] | None = None
    if annotation_path.is_file():
        with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
            annotations = list(csv.DictReader(handle))
    samples: list[Sample] = []
    for prediction_path in prediction_paths:
        try:
            index = int(prediction_path.stem)
        except ValueError as error:
            raise ValueError(
                f"prediction filename must be a numeric sample ID: {prediction_path.name}"
            ) from error
        scene_id = "unknown-scene"
        utterance = ""
        if annotations is not None and index < len(annotations):
            row = annotations[index]
            scene_id = row.get("scene_id", scene_id).strip() or scene_id
            utterance = row.get("utterance", utterance).strip()
        samples.append(
            Sample(
                index=index,
                prediction_path=prediction_path,
                points_path=data_root / "custom" / "points" / f"{index:04d}.npz",
                scene_id=scene_id,
                utterance=utterance,
            )
        )
    labels = [sample.label for sample in samples]
    if len(labels) != len(set(labels)):
        raise ValueError("sample labels are not unique")
    return samples


def affordance_colors(values: np.ndarray) -> np.ndarray:
    scalar = np.asarray(values, dtype=np.float32).clip(0.0, 1.0)
    stops = np.asarray([0.0, 0.18, 0.38, 0.58, 0.78, 1.0], dtype=np.float32)
    palette = np.asarray(
        [
            (48, 18, 92),
            (32, 91, 204),
            (25, 190, 220),
            (66, 205, 93),
            (251, 218, 60),
            (215, 38, 36),
        ],
        dtype=np.float32,
    )
    colors = np.stack(
        [np.interp(scalar, stops, palette[:, channel]) for channel in range(3)],
        axis=1,
    )
    return np.rint(colors).clip(0, 255).astype(np.uint8)


def blend_colors(heatmap: np.ndarray, rgb: np.ndarray, rgb_amount: float) -> np.ndarray:
    amount = float(np.clip(rgb_amount, 0.0, 0.8))
    return np.rint((1.0 - amount) * heatmap + amount * rgb).clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    samples = discover_samples(eval_dir, data_root)
    sample_by_label = {sample.label: sample for sample in samples}
    initial_sample = samples[0]
    if args.initial_sample is not None:
        matches = [sample for sample in samples if sample.index == args.initial_sample]
        if not matches:
            raise ValueError(f"initial sample {args.initial_sample} is absent from {eval_dir}")
        initial_sample = matches[0]

    generation_counts: dict[str, int] = {}
    for sample in samples:
        xyz, _ = load_points(sample.points_path)
        generation_counts[sample.label] = load_distances(
            sample.prediction_path, xyz.shape[0]
        ).shape[0]
    max_generations = max(generation_counts.values())

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Original ADM affordance-map viewer",
    )
    server.gui.add_markdown(
        "## Original ADM test output\n"
        "Left: input RGB point cloud. Right: the selected ADM generation converted "
        "from contact distance to **0–1 affordance**.  \n"
        "Purple means low affordance; red means high affordance."
    )
    sample_control = server.gui.add_dropdown(
        "Sample / scene / prompt",
        options=tuple(sample_by_label),
        initial_value=initial_sample.label,
    )
    generation_control = server.gui.add_dropdown(
        "Generation",
        options=tuple(f"generation_{index}" for index in range(max_generations)),
        initial_value="generation_0",
    )
    channel_control = server.gui.add_dropdown(
        "Affordance channel",
        options=("any_joint",) + CHANNEL_NAMES,
        initial_value="any_joint",
    )
    rgb_blend_control = server.gui.add_slider(
        "RGB blend", min=0.0, max=0.8, step=0.05, initial_value=0.1
    )
    point_size_control = server.gui.add_slider(
        "Point size", min=0.005, max=0.06, step=0.001, initial_value=0.02
    )
    status = server.gui.add_markdown("")
    lock = threading.Lock()

    def render() -> None:
        with lock:
            sample = sample_by_label[str(sample_control.value)]
            xyz, rgb = load_points(sample.points_path)
            distances = load_distances(sample.prediction_path, xyz.shape[0])
            requested_generation = int(str(generation_control.value).rsplit("_", 1)[1])
            generation_index = min(requested_generation, distances.shape[0] - 1)
            affordance = distance_to_affordance(distances[generation_index], args.sigma)
            channel_name = str(channel_control.value)
            if channel_name == "any_joint":
                scalar = affordance.max(axis=1)
            else:
                scalar = affordance[:, CHANNEL_NAMES.index(channel_name)]
            colors = blend_colors(
                affordance_colors(scalar), rgb, float(rgb_blend_control.value)
            )

            local_xyz = xyz.copy()
            local_xyz[:, :2] -= 0.5 * (
                local_xyz[:, :2].min(axis=0) + local_xyz[:, :2].max(axis=0)
            )
            spacing = max(float(np.ptp(local_xyz[:, 0])) + 1.0, 4.0)
            input_offset = np.asarray((-0.55 * spacing, 0.0, 0.0), dtype=np.float32)
            output_offset = np.asarray((0.55 * spacing, 0.0, 0.0), dtype=np.float32)
            point_size = float(point_size_control.value)
            server.scene.add_point_cloud(
                "/input/points",
                points=local_xyz + input_offset,
                colors=rgb,
                point_size=point_size,
                point_shape="circle",
                precision="float32",
            )
            server.scene.add_point_cloud(
                "/affordance/points",
                points=local_xyz + output_offset,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
                precision="float32",
            )
            label_height = float(local_xyz[:, 2].max()) + 0.5
            server.scene.add_label(
                "/input/label",
                text="Input 3D Scene (RGB)",
                position=input_offset + np.asarray((0.0, 0.0, label_height)),
            )
            server.scene.add_label(
                "/affordance/label",
                text="ADM Affordance Map",
                position=output_offset + np.asarray((0.0, 0.0, label_height)),
            )
            clamped_note = ""
            if requested_generation != generation_index:
                clamped_note = (
                    f"  \nRequested generation `{requested_generation}` is unavailable for this "
                    f"sample; showing generation `{generation_index}`."
                )
            status.content = (
                f"**Sample:** `{sample.index:05d}` · scene `{sample.scene_id}`  \n"
                f"**Prompt:** {sample.utterance or '(missing from anno.csv)'}  \n"
                f"**Prediction:** `{distances.shape[0]} × {distances.shape[1]} × "
                f"{distances.shape[2]}` (generations × points × channels)  \n"
                f"**Displayed:** generation `{generation_index}` · channel `{channel_name}` · "
                f"affordance range `{float(scalar.min()):.4f}`–`{float(scalar.max()):.4f}`"
                f"{clamped_note}"
            )
            server.flush()

    for control in (
        sample_control,
        generation_control,
        channel_control,
        rgb_blend_control,
        point_size_control,
    ):
        control.on_update(lambda _: render())

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        xyz, _ = load_points(initial_sample.points_path)
        radius = max(float(np.ptp(xyz[:, 0]) + np.ptp(xyz[:, 1])), 8.0)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.look_at = (0.0, 0.0, float(np.median(xyz[:, 2])))
        client.camera.position = (0.0, -1.15 * radius, 0.85 * radius)

    render()
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"[READY] ADM affordance-map viewer: {len(samples)} samples")
    print(f"[OPEN] http://{display_host}:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOPPED] ADM affordance-map viewer")


if __name__ == "__main__":
    main()

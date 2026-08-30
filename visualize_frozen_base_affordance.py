#!/usr/bin/env python3
"""Render immutable Base-teacher affordance artifacts without GPU inference.

The promoted cache index points to four frozen ``base_teacher.npz`` files.
Each artifact already contains the point-aligned XYZ/RGB scene, the K=5 mean
affordance, its standard deviation and all six Base-native contact channels.
This tool verifies those bindings, renders top-down PNG heatmaps, and optionally
writes scalar-colored PLY files for CloudCompare, MeshLab, or Open3D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


INDEX_SCHEMA = "history_affordance_v2_base_teacher_cache_index_v1"
ARTIFACT_SCHEMA = "history_affordance_v2_base_teacher_v1"
REPORT_SCHEMA = "history_affordance_v2_base_visualization_v1"
CHANNEL_ORDER = (
    "pelvis",
    "left_foot",
    "right_foot",
    "neck",
    "left_wrist",
    "right_wrist",
)
CHANNEL_JOINT_INDICES = (0, 10, 11, 12, 20, 21)
REQUIRED_ARRAYS = (
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
INSTANCE_NAMES = {
    0: "environment",
    1: "chair",
    2: "bed",
    3: "whiteboard",
    4: "tv",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(str(path) + ": expected a JSON object")
    return value


def resolve_under(root: Path, value: object, label: str) -> Path:
    path = Path(str(value)).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(label + " lies outside dataset root: " + str(path)) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_artifact(
    dataset_root: Path,
    pair: Mapping[str, object],
) -> Dict[str, object]:
    manifest_file = resolve_under(dataset_root, pair.get("manifest_file"), "manifest")
    artifact_file = resolve_under(dataset_root, pair.get("artifact_file"), "artifact")
    if sha256_file(manifest_file) != pair.get("manifest_sha256"):
        raise ValueError(str(manifest_file) + ": manifest SHA-256 mismatch")
    if sha256_file(artifact_file) != pair.get("artifact_sha256"):
        raise ValueError(str(artifact_file) + ": artifact SHA-256 mismatch")
    manifest = load_json(manifest_file)
    expected_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "status": "STAGED",
        "cache_key": pair.get("cache_key"),
        "artifact_file_sha256": pair.get("artifact_sha256"),
        "scene_id": pair.get("scene_id"),
        "prompt_id": pair.get("prompt_id"),
        "text": pair.get("text"),
        "channel_order": list(CHANNEL_ORDER),
        "channel_joint_indices": list(CHANNEL_JOINT_INDICES),
        "shape": [8192, 6],
        "draw_shape": [5, 8192, 6],
        "draw_count": 5,
        "normalization_state": "denormalized_clipped",
    }
    mismatches = {
        name: {"expected": expected, "actual": manifest.get(name)}
        for name, expected in expected_manifest.items()
        if manifest.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            str(manifest_file) + ": Base manifest mismatch: " + str(mismatches)
        )
    conditioning = manifest.get("conditioning")
    if not isinstance(conditioning, Mapping) or (
        conditioning.get("history_conditioned") is not False
    ):
        raise ValueError(str(manifest_file) + ": Base artifact contains history")

    with np.load(artifact_file, allow_pickle=False) as source:
        if set(source.files) != set(REQUIRED_ARRAYS):
            raise ValueError(str(artifact_file) + ": NPZ key set mismatch")
        arrays = {name: source[name] for name in REQUIRED_ARRAYS}
    expected_shapes = {
        "affordance": (8192, 6),
        "affordance_std": (8192, 6),
        "affordance_draws": (5, 8192, 6),
        "xyz": (8192, 3),
        "rgb01": (8192, 3),
        "instance_ids": (8192,),
        "source_indices": (8192,),
        "channel_joint_indices": (6,),
        "draw_indices": (5,),
        "initial_noise_seeds": (5,),
        "reverse_noise_seeds": (5,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(name + " shape mismatch: " + str(arrays[name].shape))
    float_names = ("affordance", "affordance_std", "affordance_draws", "xyz", "rgb01")
    if any(arrays[name].dtype != np.float32 for name in float_names):
        raise TypeError("Base floating arrays must be float32")
    integer_names = (
        "instance_ids",
        "source_indices",
        "channel_joint_indices",
        "draw_indices",
        "initial_noise_seeds",
        "reverse_noise_seeds",
    )
    if any(arrays[name].dtype != np.int64 for name in integer_names):
        raise TypeError("Base integer arrays must be int64")
    if arrays["channel_names"].astype(str).tolist() != list(CHANNEL_ORDER):
        raise ValueError("Base channel name order changed")
    if arrays["channel_joint_indices"].tolist() != list(CHANNEL_JOINT_INDICES):
        raise ValueError("Base joint index order changed")
    if arrays["draw_indices"].tolist() != list(range(5)):
        raise ValueError("Base draw index order changed")
    for name in float_names:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    if np.any(arrays["affordance"] < 0.0) or np.any(arrays["affordance"] > 1.0):
        raise ValueError("Base affordance lies outside [0,1]")
    if np.any(arrays["affordance_draws"] < 0.0) or np.any(
        arrays["affordance_draws"] > 1.0
    ):
        raise ValueError("Base affordance draws lie outside [0,1]")
    if np.any(arrays["affordance_std"] < 0.0):
        raise ValueError("Base affordance std is negative")
    if np.any(arrays["rgb01"] < 0.0) or np.any(arrays["rgb01"] > 1.0):
        raise ValueError("Base RGB lies outside [0,1]")
    mean = arrays["affordance_draws"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = arrays["affordance_draws"].std(axis=0, dtype=np.float64).astype(np.float32)
    if not np.array_equal(arrays["affordance"], mean):
        raise ValueError("Base affordance is not the exact K=5 mean")
    if not np.array_equal(arrays["affordance_std"], std):
        raise ValueError("Base affordance std is not the exact K=5 std")
    hashes = manifest.get("array_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(REQUIRED_ARRAYS):
        raise ValueError("Base manifest array hash inventory mismatch")
    for name, value in arrays.items():
        if sha256_array(value) != hashes.get(name):
            raise ValueError(name + " array SHA-256 mismatch")
    return {
        "manifest_file": manifest_file,
        "artifact_file": artifact_file,
        "manifest": manifest,
        "arrays": arrays,
    }


def task_channel(target: str) -> str:
    if target in {"chair", "bed"}:
        return "pelvis"
    if target == "whiteboard":
        return "right_wrist"
    raise ValueError("unsupported target: " + target)


def scalar_map(affordance: np.ndarray, name: str) -> np.ndarray:
    if name == "any_joint":
        return affordance.max(axis=1)
    return affordance[:, CHANNEL_ORDER.index(name)]


def annotate_instances(axis, xyz: np.ndarray, instance_ids: np.ndarray) -> None:
    for instance_id in sorted(np.unique(instance_ids).tolist()):
        mask = instance_ids == instance_id
        if not np.any(mask):
            continue
        center = np.median(xyz[mask, :2], axis=0)
        axis.text(
            float(center[0]),
            float(center[1]),
            INSTANCE_NAMES.get(int(instance_id), "id=" + str(instance_id)),
            fontsize=6,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68},
        )


def style_axis(axis, title: str, limits: Sequence[float]) -> None:
    axis.set_title(title, fontsize=9)
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("ADM X (m)", fontsize=7)
    axis.set_ylabel("ADM Y (m)", fontsize=7)
    axis.tick_params(labelsize=6)


def scene_limits(xyz: np.ndarray) -> Sequence[float]:
    x_min, y_min = np.min(xyz[:, :2], axis=0)
    x_max, y_max = np.max(xyz[:, :2], axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    pad = 0.04 * span
    return (
        float(x_min - pad),
        float(x_max + pad),
        float(y_min - pad),
        float(y_max + pad),
    )


def render_pair(
    output: Path,
    pair: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    point_size: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = arrays["xyz"]
    rgb = arrays["rgb01"]
    affordance = arrays["affordance"]
    instances = arrays["instance_ids"]
    limits = scene_limits(xyz)
    panel_names = ("rgb", "any_joint") + CHANNEL_ORDER
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    heat_scatter = None
    for axis, name in zip(axes.flat, panel_names):
        if name == "rgb":
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                c=np.clip(rgb, 0.0, 1.0),
                s=point_size,
                linewidths=0,
                rasterized=True,
            )
            annotate_instances(axis, xyz, instances)
            title = "Scene RGB"
        else:
            values = np.clip(scalar_map(affordance, name), 0.0, 1.0)
            heat_scatter = axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                c=values,
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                s=point_size,
                linewidths=0,
                rasterized=True,
            )
            title = (
                "Any joint (max)"
                if name == "any_joint"
                else name.replace("_", " ").title()
            )
        style_axis(axis, title, limits)
    if heat_scatter is None:
        raise AssertionError("no affordance heatmap was rendered")
    fig.colorbar(heat_scatter, ax=axes, label="Base affordance [0,1]", shrink=0.82)
    fig.suptitle(
        "{} | {} | {}".format(pair["scene_id"], pair["prompt_id"], pair["text"]),
        fontsize=13,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def render_overview(
    output: Path,
    pair_rows: Sequence[Mapping[str, object]],
    point_size: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(pair_rows), 3, figsize=(13.5, 4.0 * len(pair_rows)), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(pair_rows), 3)
    heat_scatter = None
    for row_index, row in enumerate(pair_rows):
        pair = row["pair"]
        arrays = row["arrays"]
        xyz = arrays["xyz"]
        rgb = arrays["rgb01"]
        affordance = arrays["affordance"]
        instances = arrays["instance_ids"]
        limits = scene_limits(xyz)
        channel = task_channel(str(pair["target"]))
        for column, name in enumerate(("rgb", "any_joint", channel)):
            axis = axes[row_index, column]
            if name == "rgb":
                axis.scatter(
                    xyz[:, 0],
                    xyz[:, 1],
                    c=np.clip(rgb, 0.0, 1.0),
                    s=point_size,
                    linewidths=0,
                    rasterized=True,
                )
                annotate_instances(axis, xyz, instances)
                title = "{} | RGB".format(pair["scene_id"])
            else:
                values = np.clip(scalar_map(affordance, name), 0.0, 1.0)
                heat_scatter = axis.scatter(
                    xyz[:, 0],
                    xyz[:, 1],
                    c=values,
                    cmap="turbo",
                    vmin=0.0,
                    vmax=1.0,
                    s=point_size,
                    linewidths=0,
                    rasterized=True,
                )
                label = (
                    "Any joint"
                    if name == "any_joint"
                    else "Task: " + name.replace("_", " ")
                )
                title = "{} | {}".format(pair["target"], label)
            style_axis(axis, title, limits)
    if heat_scatter is None:
        raise AssertionError("no overview heatmap was rendered")
    fig.colorbar(heat_scatter, ax=axes, label="Base affordance [0,1]", shrink=0.88)
    fig.suptitle(
        "Frozen Base affordance maps | promoted 4 scene-prompt pairs", fontsize=14
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def write_ply(
    output: Path,
    xyz: np.ndarray,
    scalar: np.ndarray,
    color_map,
) -> None:
    values = np.clip(np.asarray(scalar, dtype=np.float32), 0.0, 1.0)
    colors = np.rint(color_map(values)[:, :3] * 255.0).astype(np.uint8)
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment scalar is frozen Base affordance in [0,1]\n")
        handle.write("element vertex {}\n".format(len(xyz)))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property float affordance\nend_header\n")
        for point, color, value in zip(xyz, colors, values):
            handle.write(
                "{:.8g} {:.8g} {:.8g} {} {} {} {:.8g}\n".format(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    float(value),
                )
            )


def channel_summary(values: np.ndarray, threshold: float) -> Dict[str, object]:
    return {
        "min": float(values.min()),
        "mean": float(values.mean(dtype=np.float64)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "active_threshold": threshold,
        "active_point_count": int((values >= threshold).sum()),
        "active_point_fraction": float((values >= threshold).mean()),
    }


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-threshold", type=float, default=0.7)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-ply", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.active_threshold < 1.0:
        raise ValueError("--active-threshold must lie in (0,1)")
    if args.point_size <= 0.0 or args.dpi < 72:
        raise ValueError("point size/DPI is invalid")
    index_file = args.index.expanduser().resolve()
    index = load_json(index_file)
    if (
        index.get("schema") != INDEX_SCHEMA
        or index.get("status") != "PASS"
        or index.get("promotion_authorized") is not True
        or index.get("num_pairs") != 4
        or index.get("num_samples") != 61
        or index.get("channel_order") != list(CHANNEL_ORDER)
        or index.get("channel_joint_indices") != list(CHANNEL_JOINT_INDICES)
    ):
        raise ValueError("promoted Base cache index contract mismatch")
    checks = index.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("promoted Base cache index has a failed check")
    pairs = index.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        raise ValueError("promoted Base cache pair inventory mismatch")
    keys = [str(pair.get("cache_key", "")) for pair in pairs]
    if len(set(keys)) != 4 or any(
        SHA256_PATTERN.fullmatch(key) is None for key in keys
    ):
        raise ValueError("promoted Base cache keys are invalid")
    recorded_root = Path(str(index.get("dataset_root", ""))).expanduser().resolve()
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else recorded_root
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite visualization: " + str(output_dir))

    validated = []
    source_hashes_before = {}
    for pair in sorted(
        pairs, key=lambda row: (str(row["scene_id"]), str(row["prompt_id"]))
    ):
        result = validate_artifact(dataset_root, pair)
        source_hashes_before[str(result["manifest_file"])] = sha256_file(
            result["manifest_file"]
        )
        source_hashes_before[str(result["artifact_file"])] = sha256_file(
            result["artifact_file"]
        )
        validated.append({"pair": pair, **result})

    stage = Path(
        tempfile.mkdtemp(
            prefix=output_dir.name + ".staging.", dir=str(output_dir.parent)
        )
    )
    try:
        overview_file = stage / "base_affordance_overview.png"
        render_overview(overview_file, validated, args.point_size, args.dpi)
        import matplotlib.pyplot as plt

        turbo = plt.get_cmap("turbo")
        rows = []
        for result in validated:
            pair = result["pair"]
            arrays = result["arrays"]
            slug = "{}__{}".format(pair["scene_id"], pair["prompt_id"])
            png_file = stage / (slug + "__six_channels.png")
            render_pair(png_file, pair, arrays, args.point_size, args.dpi)
            ply_files = {}
            names = ("any_joint",) + CHANNEL_ORDER
            if not args.skip_ply:
                ply_dir = stage / "ply" / slug
                ply_dir.mkdir(parents=True)
                for name in names:
                    path = ply_dir / (name + ".ply")
                    write_ply(
                        path,
                        arrays["xyz"],
                        scalar_map(arrays["affordance"], name),
                        turbo,
                    )
                    ply_files[name] = str(path.relative_to(stage))
            summaries = {
                name: channel_summary(
                    scalar_map(arrays["affordance"], name),
                    args.active_threshold,
                )
                for name in names
            }
            rows.append(
                {
                    "cache_key": pair["cache_key"],
                    "scene_id": pair["scene_id"],
                    "target": pair["target"],
                    "prompt_id": pair["prompt_id"],
                    "text": pair["text"],
                    "manifest_file": str(result["manifest_file"]),
                    "manifest_sha256": pair["manifest_sha256"],
                    "artifact_file": str(result["artifact_file"]),
                    "artifact_sha256": pair["artifact_sha256"],
                    "six_channel_png": str(png_file.relative_to(stage)),
                    "ply": ply_files,
                    "channel_statistics": summaries,
                    "mean_teacher_std": float(
                        arrays["affordance_std"].mean(dtype=np.float64)
                    ),
                    "max_teacher_std": float(arrays["affordance_std"].max()),
                }
            )
        source_hashes_after = {
            path: sha256_file(Path(path)) for path in source_hashes_before
        }
        if source_hashes_after != source_hashes_before:
            raise RuntimeError("Base source artifact changed during visualization")
        report = {
            "schema": REPORT_SCHEMA,
            "status": "PASS",
            "index": str(index_file),
            "index_sha256": sha256_file(index_file),
            "index_id": index.get("index_id"),
            "dataset_root": str(dataset_root),
            "num_pairs": 4,
            "num_samples_bound_by_index": 61,
            "gpu_inference_used": False,
            "diffusion_sampling_used": False,
            "source_artifacts_modified": False,
            "source_artifacts_unchanged": True,
            "channel_order": list(CHANNEL_ORDER),
            "channel_joint_indices": list(CHANNEL_JOINT_INDICES),
            "active_threshold": args.active_threshold,
            "overview_png": str(overview_file.relative_to(stage)),
            "pairs": rows,
            "source_sha256": source_hashes_before,
        }
        atomic_json(stage / "summary.json", report)
        os.replace(stage, output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    print("[PASS] frozen Base affordance visualization")
    print("[PASS] 4 promoted artifacts hash-verified and unchanged")
    print("[OK] overview:", output_dir / "base_affordance_overview.png")
    print("[OK] per-pair six-channel PNGs: 4")
    print("[OK] PLY files:", 0 if args.skip_ply else 4 * 7)
    print("[OK] report:", output_dir / "summary.json")


if __name__ == "__main__":
    main()

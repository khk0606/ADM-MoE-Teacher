#!/usr/bin/env python3
"""Validate structural and semantic diagnostics of generated IIW targets.

The hard PASS checks only facts that must hold for every dataset: tensor
shape, point order, finite range, channel mapping, and temporal-bin coverage.
Motion-specific expectations (for example, pelvis/base intensity moving toward
the chair in later bins) are reported separately as diagnostics so that a new
action category is never rejected by a chair-only heuristic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


NUM_POINTS = 8192
NUM_BODY_PARTS = 6
EXPECTED_CMDM_HORIZON = 196
DEFAULT_THRESHOLD = 0.7

BODY_PART_NAMES = (
    "base",
    "spine",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
)
INSTANCE_NAMES: Mapping[int, str] = {
    0: "environment",
    1: "chair",
    2: "bed",
    3: "whiteboard",
    4: "tv",
}
CMDM_PROXY_FROM_NATIVE = np.asarray([0, 5, 4, 1, 3, 2], dtype=np.int64)
EXPECTED_BODY_JOINT_GROUPS = np.asarray(
    [
        [0, 1, 2],
        [9, -1, -1],
        [19, 21, -1],
        [18, 20, -1],
        [8, 11, -1],
        [7, 10, -1],
    ],
    dtype=np.int64,
)
EXPECTED_BODY_GROUP_COUNTS = np.asarray([3, 1, 2, 2, 2, 2], dtype=np.int64)


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def resolve_under(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def top_fraction_mean(value: np.ndarray, fraction: float = 0.1) -> float:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot summarize an empty array")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    count = max(1, int(np.ceil(flat.size * fraction)))
    start = flat.size - count
    selected = np.partition(flat, start)[start:]
    return float(selected.mean())


def validate_frame_partition(
    frame_to_phase: np.ndarray,
    phase_counts: np.ndarray,
    valid_frames: int,
    num_phases: int,
) -> None:
    if frame_to_phase.shape != (valid_frames,):
        raise ValueError(
            f"frame_to_phase {frame_to_phase.shape} != ({valid_frames},)"
        )
    if phase_counts.shape != (num_phases,):
        raise ValueError(
            f"phase_frame_counts {phase_counts.shape} != ({num_phases},)"
        )
    if int(phase_counts.sum()) != valid_frames or np.any(phase_counts <= 0):
        raise ValueError("temporal-bin counts do not cover valid frames")
    if np.any(frame_to_phase < 0) or np.any(frame_to_phase >= num_phases):
        raise ValueError("frame_to_phase contains an invalid bin index")
    actual = np.bincount(frame_to_phase, minlength=num_phases)
    if not np.array_equal(actual, phase_counts):
        raise ValueError("frame_to_phase/count mismatch")
    if np.any(np.diff(frame_to_phase) < 0):
        raise ValueError("temporal-bin order is not monotonic")
    expected = np.repeat(np.arange(num_phases, dtype=np.int64), phase_counts)
    if not np.array_equal(frame_to_phase, expected):
        raise ValueError("temporal bins are not prefix-contiguous")


def validate_motion_mask(
    motion_file: Path,
    valid_frames: int,
) -> None:
    if valid_frames <= 0 or valid_frames > EXPECTED_CMDM_HORIZON:
        raise ValueError(
            f"{motion_file}: valid_frames={valid_frames} is outside "
            f"1..{EXPECTED_CMDM_HORIZON}"
        )
    with np.load(motion_file, allow_pickle=False) as motion:
        if "x_mask" not in motion:
            raise KeyError(f"{motion_file}: missing x_mask")
        mask = motion["x_mask"].astype(bool)
    if mask.shape != (EXPECTED_CMDM_HORIZON,):
        raise ValueError(f"{motion_file}: x_mask shape is {mask.shape}")
    expected = np.ones((EXPECTED_CMDM_HORIZON,), dtype=bool)
    expected[:valid_frames] = False
    if not np.array_equal(mask, expected):
        raise ValueError(f"{motion_file}: x_mask is not a valid prefix mask")


def summarize_tensor(
    sample_id: str,
    value: np.ndarray,
    instance_ids: np.ndarray,
    threshold: float,
) -> Tuple[List[Dict], Dict]:
    num_phases = value.shape[0]
    rows: List[Dict] = []
    global_active = np.zeros(
        (num_phases, NUM_BODY_PARTS), dtype=np.float64
    )
    global_mean = np.zeros_like(global_active)
    instance_active: Dict[str, np.ndarray] = {}
    instance_top10: Dict[str, np.ndarray] = {}

    for instance_id in sorted(INSTANCE_NAMES):
        name = INSTANCE_NAMES[instance_id]
        if np.any(instance_ids == instance_id):
            instance_active[name] = np.zeros_like(global_active)
            instance_top10[name] = np.zeros_like(global_active)

    for phase_index in range(num_phases):
        for body_index, body_name in enumerate(BODY_PART_NAMES):
            channel = value[phase_index, :, body_index]
            base_row = {
                "sample_id": sample_id,
                "phase": phase_index,
                "body_index": body_index,
                "body_part": body_name,
                "region": "all",
                "point_count": int(len(channel)),
                "mean": float(channel.mean()),
                "max": float(channel.max()),
                "top10_mean": top_fraction_mean(channel),
                "active_fraction": float(np.mean(channel >= threshold)),
            }
            rows.append(base_row)
            global_active[phase_index, body_index] = base_row[
                "active_fraction"
            ]
            global_mean[phase_index, body_index] = base_row["mean"]

            for instance_id in sorted(INSTANCE_NAMES):
                mask = instance_ids == instance_id
                if not np.any(mask):
                    continue
                name = INSTANCE_NAMES[instance_id]
                region_value = channel[mask]
                row = {
                    "sample_id": sample_id,
                    "phase": phase_index,
                    "body_index": body_index,
                    "body_part": body_name,
                    "region": name,
                    "point_count": int(mask.sum()),
                    "mean": float(region_value.mean()),
                    "max": float(region_value.max()),
                    "top10_mean": top_fraction_mean(region_value),
                    "active_fraction": float(
                        np.mean(region_value >= threshold)
                    ),
                }
                rows.append(row)
                instance_active[name][phase_index, body_index] = row[
                    "active_fraction"
                ]
                instance_top10[name][phase_index, body_index] = row[
                    "top10_mean"
                ]

    adjacent_delta = np.abs(np.diff(value.astype(np.float32), axis=0))
    body_mass = value.astype(np.float64).sum(axis=(0, 1))
    total_mass = float(body_mass.sum())
    body_mass_fraction = (
        body_mass / total_mass
        if total_mass > 0.0
        else np.zeros_like(body_mass)
    )
    diagnostics = {
        "global_active_fraction": float(np.mean(value >= threshold)),
        "global_mean": float(value.mean()),
        "global_max": float(value.max()),
        "adjacent_bin_mean_absolute_delta": float(adjacent_delta.mean()),
        "adjacent_bin_max_absolute_delta": float(adjacent_delta.max()),
        "body_mass_fraction": {
            name: float(body_mass_fraction[index])
            for index, name in enumerate(BODY_PART_NAMES)
        },
        "global_active_by_phase_body": global_active.tolist(),
        "global_mean_by_phase_body": global_mean.tolist(),
        "instance_active_by_phase_body": {
            key: value.tolist() for key, value in instance_active.items()
        },
        "instance_top10_by_phase_body": {
            key: value.tolist() for key, value in instance_top10.items()
        },
    }

    chair_top10 = instance_top10.get("chair")
    environment_top10 = instance_top10.get("environment")
    if chair_top10 is not None:
        early = float(chair_top10[:2, 0].mean())
        late = float(chair_top10[-2:, 0].mean())
        diagnostics["chair_base_top10_early"] = early
        diagnostics["chair_base_top10_late"] = late
        diagnostics["chair_base_late_minus_early"] = late - early
        diagnostics["chair_base_late_exceeds_early"] = bool(late > early)
    if environment_top10 is not None:
        foot_indices = (4, 5)
        diagnostics["environment_feet_top10_mean"] = float(
            environment_top10[:, foot_indices].mean()
        )
    return rows, diagnostics


def load_and_validate_sample(
    root: Path,
    entry: Dict,
    scene_xyz: np.ndarray,
    instance_ids: np.ndarray,
    source_indices: np.ndarray,
    num_phases: int,
    threshold: float,
) -> Tuple[List[Dict], Dict]:
    if num_phases < 2:
        raise ValueError("IIW validation requires at least two temporal bins")
    sample_id = str(entry["sample_id"])
    if not bool(entry.get("iiw_gt_ready", False)):
        raise ValueError(f"{sample_id}: iiw_gt_ready is not true")
    target_file = resolve_under(root, entry["iiw_gt"])
    manifest_file = resolve_under(root, entry["iiw_gt_manifest"])
    motion_file = resolve_under(root, entry["cmdm_motion_input"])
    manifest = load_json(manifest_file)
    if manifest.get("sample_id") != sample_id:
        raise ValueError(f"{sample_id}: target manifest ID mismatch")
    if manifest.get("method") != "point_aligned_iiw_proxy":
        raise ValueError(f"{sample_id}: unexpected IIW method")
    if manifest.get("target_semantics") != (
        "continuous_distance_kernel_without_contact_mask"
    ):
        raise ValueError(f"{sample_id}: target semantics are not explicit")
    if bool(manifest.get("unity_contact_label_masking_applied", True)):
        raise ValueError(f"{sample_id}: contact-mask disclosure mismatch")
    disclosure = str(manifest.get("approximation_disclosure", ""))
    if "not original" not in disclosure:
        raise ValueError(f"{sample_id}: approximation disclosure is missing")

    with np.load(target_file, allow_pickle=False) as target_archive:
        target = {key: target_archive[key] for key in target_archive.files}
    expected_shape = (num_phases, NUM_POINTS, NUM_BODY_PARTS)
    required = (
        "iiw_native_phase_max",
        "iiw_native_phase_mean",
        "iiw_cmdm_proxy_phase_max",
        "iiw_cmdm_proxy_phase_mean",
        "phase_frame_counts",
        "frame_to_phase",
        "phase_mask",
        "scene_xyz_adm",
        "instance_ids",
        "source_indices",
        "valid_frames",
        "body_joint_groups",
        "body_group_counts",
        "body_part_names",
        "cmdm_proxy_from_native",
    )
    for key in required:
        if key not in target:
            raise KeyError(f"{sample_id}: missing {key}")
    for key in required[:4]:
        if target[key].shape != expected_shape:
            raise ValueError(
                f"{sample_id}: {key} {target[key].shape} != {expected_shape}"
            )

    native_max = target["iiw_native_phase_max"].astype(np.float32)
    native_mean = target["iiw_native_phase_mean"].astype(np.float32)
    proxy_max = target["iiw_cmdm_proxy_phase_max"].astype(np.float32)
    proxy_mean = target["iiw_cmdm_proxy_phase_mean"].astype(np.float32)
    for name, value in (
        ("native max", native_max),
        ("native mean", native_mean),
        ("proxy max", proxy_max),
        ("proxy mean", proxy_mean),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{sample_id}: {name} contains NaN/Inf")
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(f"{sample_id}: {name} is outside [0,1]")
    if np.any(native_mean > native_max + 1e-3):
        raise ValueError(f"{sample_id}: phase mean exceeds phase max")
    if not np.array_equal(proxy_max, native_max[..., CMDM_PROXY_FROM_NATIVE]):
        raise ValueError(f"{sample_id}: max proxy permutation mismatch")
    if not np.array_equal(
        proxy_mean, native_mean[..., CMDM_PROXY_FROM_NATIVE]
    ):
        raise ValueError(f"{sample_id}: mean proxy permutation mismatch")
    if not np.array_equal(target["scene_xyz_adm"], scene_xyz):
        raise ValueError(f"{sample_id}: scene XYZ order/content mismatch")
    if not np.array_equal(target["instance_ids"], instance_ids):
        raise ValueError(f"{sample_id}: instance order mismatch")
    if not np.array_equal(target["source_indices"], source_indices):
        raise ValueError(f"{sample_id}: source point order mismatch")
    saved_valid_frames = int(np.asarray(target["valid_frames"]).item())
    valid_frames = int(entry["valid_frames"])
    if saved_valid_frames != valid_frames:
        raise ValueError(f"{sample_id}: saved valid-frame count mismatch")
    if not np.array_equal(
        target["body_joint_groups"].astype(np.int64),
        EXPECTED_BODY_JOINT_GROUPS,
    ):
        raise ValueError(f"{sample_id}: native body-joint groups mismatch")
    if not np.array_equal(
        target["body_group_counts"].astype(np.int64),
        EXPECTED_BODY_GROUP_COUNTS,
    ):
        raise ValueError(f"{sample_id}: native body-group counts mismatch")
    if target["body_part_names"].astype(str).tolist() != list(BODY_PART_NAMES):
        raise ValueError(f"{sample_id}: native body-part order mismatch")
    if not np.array_equal(
        target["cmdm_proxy_from_native"].astype(np.int64),
        CMDM_PROXY_FROM_NATIVE,
    ):
        raise ValueError(f"{sample_id}: saved proxy permutation mismatch")
    phase_mask = target["phase_mask"].astype(bool)
    if phase_mask.shape != (num_phases,) or np.any(phase_mask):
        raise ValueError(f"{sample_id}: all generated temporal bins must be valid")

    phase_counts = target["phase_frame_counts"].astype(np.int64)
    frame_to_phase = target["frame_to_phase"].astype(np.int64)
    validate_frame_partition(
        frame_to_phase,
        phase_counts,
        valid_frames,
        num_phases,
    )
    expected_ranges = []
    phase_start = 0
    for count in phase_counts:
        phase_end = phase_start + int(count)
        expected_ranges.append([phase_start, phase_end])
        phase_start = phase_end
    manifest_contract = {
        "coordinate_frame": "chair_local_z_up",
        "num_phases": num_phases,
        "valid_frames": valid_frames,
        "num_scene_points": NUM_POINTS,
        "native_body_part_order": list(BODY_PART_NAMES),
        "cmdm_proxy_from_native": CMDM_PROXY_FROM_NATIVE.tolist(),
        "phase_frame_counts": phase_counts.tolist(),
        "phase_frame_ranges": expected_ranges,
        "iiw_gt_ready": True,
    }
    for key, expected in manifest_contract.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{sample_id}: manifest {key}={manifest.get(key)!r}, "
                f"expected {expected!r}"
            )
    validate_motion_mask(motion_file, valid_frames)
    rows, diagnostics = summarize_tensor(
        sample_id, native_max, instance_ids, threshold
    )
    semantic_alerts = []
    if diagnostics["global_active_fraction"] <= 0.0:
        raise ValueError(f"{sample_id}: IIW target is all inactive")
    if diagnostics["global_active_fraction"] >= 0.25:
        semantic_alerts.append("active_fraction_at_least_0.25")
    if diagnostics["adjacent_bin_mean_absolute_delta"] <= 1e-7:
        semantic_alerts.append("temporal_bins_nearly_invariant")

    diagnostics.update(
        {
            "sample_id": sample_id,
            "valid_frames": valid_frames,
            "num_phases": num_phases,
            "target_name": str(entry.get("target_name", "")),
            "target_instance_id": int(entry.get("target_instance_id", -1)),
            "semantic_alerts": semantic_alerts,
            "structural_pass": True,
        }
    )
    return rows, diagnostics


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty diagnostics CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, sample_diagnostics: Sequence[Dict]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CHECK] matplotlib unavailable; skipped IIW diagnostic plot")
        return False

    sample_ids = [item["sample_id"] for item in sample_diagnostics]
    active = np.asarray(
        [item["global_active_by_phase_body"] for item in sample_diagnostics],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for body_index, body_name in enumerate(BODY_PART_NAMES):
        axis = axes.flat[body_index]
        image = axis.imshow(
            active[:, :, body_index],
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
        )
        axis.set_title(body_name)
        axis.set_xlabel("Temporal bin")
        axis.set_xticks(np.arange(active.shape[1]))
        axis.set_yticks(np.arange(len(sample_ids)))
        axis.set_yticklabels(sample_ids, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("IIW active fraction by sample / temporal bin / body part")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def mean_and_std(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    index_file = resolve_under(root, args.index)
    index = load_json(index_file)
    if not bool(index.get("iiw_gt_ready", False)):
        raise ValueError("dataset index is not IIW-ready")
    if index.get("iiw_gt_method") != "point_aligned_iiw_proxy":
        raise ValueError("dataset IIW method mismatch")
    num_phases = int(index["iiw_num_phases"])
    if num_phases < 2:
        raise ValueError("dataset must contain at least two temporal bins")
    entries = index.get("samples", [])
    if not entries or len(entries) != int(index.get("num_samples", -1)):
        raise ValueError("dataset index sample count mismatch")
    sample_ids = [str(entry["sample_id"]) for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs")

    scene_dir = resolve_under(root, index["scene_adm_input"])
    sidecar_file = scene_dir / "sidecar.npz"
    with np.load(sidecar_file, allow_pickle=False) as sidecar:
        scene_xyz = sidecar["xyz_afford_z_up"].astype(np.float32)
        instance_ids = sidecar["instance_ids"].astype(np.int64)
        source_indices = sidecar["source_indices"].astype(np.int64)
    if scene_xyz.shape != (NUM_POINTS, 3):
        raise ValueError("scene XYZ shape mismatch")
    if instance_ids.shape != (NUM_POINTS,):
        raise ValueError("scene instance-ID shape mismatch")
    if source_indices.shape != (NUM_POINTS,):
        raise ValueError("scene source-index shape mismatch")
    if not np.isfinite(scene_xyz).all():
        raise ValueError("scene XYZ contains NaN/Inf")
    unknown_instance_ids = sorted(
        set(np.unique(instance_ids).tolist()) - set(INSTANCE_NAMES)
    )
    if unknown_instance_ids:
        raise ValueError(
            f"scene contains unknown v1 instance IDs: {unknown_instance_ids}"
        )

    if args.output_dir is None:
        output_dir = root / "diagnostics" / (index_file.stem + "_validation")
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    sample_diagnostics: List[Dict] = []
    for entry in entries:
        rows, diagnostics = load_and_validate_sample(
            root=root,
            entry=entry,
            scene_xyz=scene_xyz,
            instance_ids=instance_ids,
            source_indices=source_indices,
            num_phases=num_phases,
            threshold=args.threshold,
        )
        all_rows.extend(rows)
        sample_diagnostics.append(diagnostics)
        chair_delta = diagnostics.get("chair_base_late_minus_early")
        chair_text = (
            f", chair_base_late-early={chair_delta:+.6f}"
            if chair_delta is not None
            else ""
        )
        print(
            f"[PASS] {diagnostics['sample_id']}: "
            f"active@{args.threshold:.2f}="
            f"{diagnostics['global_active_fraction']:.6f}, "
            f"temporal_delta="
            f"{diagnostics['adjacent_bin_mean_absolute_delta']:.6f}"
            f"{chair_text}"
        )

    chair_diagnostics = [
        item
        for item in sample_diagnostics
        if str(item.get("target_name", "")).lower() == "chair"
    ]
    semantic_checks = {
        "all_chair_targets_have_stronger_late_base_chair_signal": bool(
            chair_diagnostics
        )
        and all(
            bool(item.get("chair_base_late_exceeds_early", False))
            for item in chair_diagnostics
        ),
        "all_body_channels_have_mass": all(
            all(value > 0.0 for value in item["body_mass_fraction"].values())
            for item in sample_diagnostics
        ),
        "no_semantic_alerts": all(
            not item["semantic_alerts"] for item in sample_diagnostics
        ),
    }
    summary = {
        "status": "PASS",
        "scope": "IIW GT structural contract and semantic diagnostics",
        "index": str(index_file),
        "scene_id": str(index["scene_id"]),
        "num_samples": len(entries),
        "num_phases": num_phases,
        "num_points": NUM_POINTS,
        "num_body_parts": NUM_BODY_PARTS,
        "threshold": args.threshold,
        "target_semantics": "continuous_distance_kernel_without_contact_mask",
        "global_active_fraction": mean_and_std(
            item["global_active_fraction"] for item in sample_diagnostics
        ),
        "temporal_delta": mean_and_std(
            item["adjacent_bin_mean_absolute_delta"]
            for item in sample_diagnostics
        ),
        "semantic_checks": semantic_checks,
        "semantic_interpretation": (
            "Diagnostics are action-specific evidence, not hard structural "
            "validity conditions. Temporal bins are uniform normalized bins, "
            "not contact-onset/hold semantic phases."
        ),
        "per_sample": sample_diagnostics,
    }
    csv_file = output_dir / "per_phase_body_region.csv"
    summary_file = output_dir / "summary.json"
    plot_file = output_dir / "iiw_phase_body_heatmaps.png"
    write_csv(csv_file, all_rows)
    plot_saved = save_plot(plot_file, sample_diagnostics)
    summary["plot_saved"] = plot_saved
    summary["outputs"] = {
        "summary": str(summary_file),
        "per_phase_body_region": str(csv_file),
        "phase_body_heatmaps": str(plot_file) if plot_saved else None,
    }
    write_json(summary_file, summary)

    for item in sample_diagnostics:
        if item["semantic_alerts"]:
            print(
                f"[CHECK] {item['sample_id']}: semantic alerts="
                + ",".join(item["semantic_alerts"])
            )
    if chair_diagnostics and not semantic_checks[
        "all_chair_targets_have_stronger_late_base_chair_signal"
    ]:
        print(
            "[CHECK] at least one chair sample does not have stronger late-bin "
            "base/chair intensity; inspect the saved diagnostics"
        )
    print(
        f"[PASS] IIW GT structural contract: samples={len(entries)}, "
        f"bins={num_phases}, points={NUM_POINTS}, bodies={NUM_BODY_PARTS}"
    )
    print(f"[OK] saved: {summary_file}")
    print(f"[OK] saved: {csv_file}")
    if plot_saved:
        print(f"[OK] saved: {plot_file}")


if __name__ == "__main__":
    main()

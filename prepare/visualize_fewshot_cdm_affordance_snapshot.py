#!/usr/bin/env python3
"""Visualize the completed train-only v4 CDM rollout cache.

This script performs no diffusion sampling and never reads the held-out CDM
partition.  It validates the crash-safe rollout contract, averages the five
paired train-audit draws, and compares GT, the untouched original CDM, and one
explicit LoRA candidate checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot_cdm_common import (  # noqa: E402
    instance_scores,
    load_split,
    load_stats,
    sha256_file,
)
from fewshot_cdm_rollout_cache import (  # noqa: E402
    CONTRACT_SCHEMA,
    load_prediction,
    prediction_path,
    sha256_json,
)
TARGETS = ("chair", "bed", "whiteboard")
INSTANCE_NAMES = ("environment", "chair", "bed", "whiteboard", "tv")
SNAPSHOT_SCHEMA = "history_affordance_v1_fewshot_cdm_map_snapshot_v1"


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
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path(
            "data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_"
            "contact_cont_joints_0.8_fur.npz"
        ),
    )
    parser.add_argument(
        "--fewshot-dir",
        type=Path,
        default=Path(
            "data/history_affordance_v1/experiments/"
            "fewshot_cdm_chair23_bed2_whiteboard12_v4"
        ),
    )
    parser.add_argument("--candidate-step", type=int, default=3750)
    parser.add_argument("--k-samples", type=int, default=5)
    parser.add_argument("--active-threshold", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <fewshot-dir>/affordance_snapshot_step_<step>.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def binary_f1(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction_active = np.asarray(prediction) >= threshold
    target_active = np.asarray(target) >= threshold
    true_positive = int(np.logical_and(prediction_active, target_active).sum())
    false_positive = int(np.logical_and(prediction_active, ~target_active).sum())
    false_negative = int(np.logical_and(~prediction_active, target_active).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else float(2 * true_positive / denominator)


def reduce_affordance(affordance: np.ndarray, channel: str) -> np.ndarray:
    value = np.asarray(affordance, dtype=np.float32)
    if value.shape != (8192, 6):
        raise ValueError(f"affordance shape {value.shape} != (8192, 6)")
    if channel == "pelvis":
        return value[:, 0]
    if channel == "any_joint":
        return value.max(axis=1)
    raise ValueError(f"unknown channel: {channel}")


def deterministic_representatives(
    rows: Mapping[str, Mapping[str, object]],
) -> Dict[str, str]:
    result = {}
    for target in TARGETS:
        candidates = sorted(
            sample_id
            for sample_id, row in rows.items()
            if str(row["target"]) == target
        )
        if not candidates:
            raise ValueError(f"train partition has no {target} sample")
        result[target] = candidates[0]
    return result


def validate_contract(
    fewshot_dir: Path,
    split_file: Path,
    train_ids: Sequence[str],
    candidate_step: int,
    k_samples: int,
) -> Tuple[Path, Dict[str, object], Dict[str, object]]:
    selection_file = fewshot_dir / "rollout_candidate_selection.json"
    selection = read_json(selection_file)
    if selection.get("schema") != "history_affordance_v1_fewshot_cdm_candidate_rollout_v1":
        raise ValueError(f"{selection_file}: unsupported schema")
    if selection.get("partition") != "train_complete":
        raise AssertionError("rollout selection is not train_complete")
    if selection.get("heldout_sample_tensors_read") is not False:
        raise AssertionError("rollout selection touched held-out tensors")
    if selection.get("complete_train_partition") is not True:
        raise AssertionError("rollout selection did not cover complete train partition")
    if int(selection.get("k_samples", -1)) != k_samples or k_samples < 5:
        raise AssertionError("snapshot requires the exact K>=5 rollout protocol")
    probe_ids = [str(value) for value in selection.get("probe_sample_ids", [])]
    if probe_ids != sorted(train_ids):
        raise AssertionError("rollout probe IDs do not equal the complete train partition")

    rollout_cache = selection.get("rollout_cache")
    if not isinstance(rollout_cache, dict):
        raise ValueError(f"{selection_file}: missing rollout_cache")
    cache_dir = Path(str(rollout_cache["directory"])).expanduser().resolve()
    contract_file = Path(str(rollout_cache["contract_file"])).expanduser().resolve()
    contract = read_json(contract_file)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"{contract_file}: unsupported cache schema")
    if sha256_json(contract) != str(rollout_cache.get("contract_sha256")):
        raise AssertionError("rollout cache contract hash mismatch")
    if contract.get("heldout_sample_tensors_read") is not False:
        raise AssertionError("rollout contract touched held-out tensors")
    if contract.get("partition") != "train_complete":
        raise AssertionError("rollout contract partition mismatch")
    if contract.get("probe_sample_ids") != probe_ids:
        raise AssertionError("selection/contract probe IDs differ")
    if str(contract.get("split_sha256")) != sha256_file(split_file):
        raise AssertionError("split file changed after rollout")

    candidates = {
        int(row["step"]): row for row in contract.get("shortlisted_candidates", [])
    }
    if candidate_step not in candidates:
        raise KeyError(f"candidate step {candidate_step} is not bound by rollout contract")
    candidate = candidates[candidate_step]
    checkpoint = Path(str(candidate["checkpoint"])).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != str(candidate["checkpoint_sha256"]):
        raise AssertionError("candidate checkpoint changed after rollout")

    evaluated = {
        int(row["step"]): row for row in selection.get("evaluated", [])
    }
    if candidate_step not in evaluated:
        raise KeyError(f"candidate step {candidate_step} was not evaluated")
    result = evaluated[candidate_step]
    if str(result.get("contract_sha256")) != sha256_json(contract):
        raise AssertionError("candidate result is bound to a different cache contract")
    if str(result.get("checkpoint_sha256")) != str(candidate["checkpoint_sha256"]):
        raise AssertionError("candidate result/checkpoint hash mismatch")
    return cache_dir, contract, result


def load_ensemble(
    cache_dir: Path,
    role: str,
    sample_id: str,
    k_samples: int,
    candidate_step: Optional[int] = None,
) -> np.ndarray:
    draws: List[np.ndarray] = []
    for draw in range(k_samples):
        path = prediction_path(
            cache_dir,
            role,
            sample_id,
            draw,
            candidate_step=candidate_step,
        )
        draws.append(load_prediction(path))
    return np.mean(np.stack(draws, axis=0), axis=0, dtype=np.float32).astype(
        np.float32
    )


def target_metrics(
    prediction: np.ndarray,
    gt: np.ndarray,
    instance_ids: np.ndarray,
    target: str,
    target_instance_id: int,
    threshold: float,
) -> Dict[str, object]:
    mask = instance_ids == target_instance_id
    if not np.any(mask):
        raise AssertionError(f"target instance {target_instance_id} has no points")
    scores = {
        channel: instance_scores(prediction, instance_ids, channel)
        for channel in ("pelvis", "any_joint")
    }
    return {
        "target_region_mae": float(np.abs(prediction[mask] - gt[mask]).mean()),
        "target_region_f1_at_0_7": binary_f1(
            prediction[mask], gt[mask], threshold
        ),
        "instance_top10_scores": scores,
        "target_dominates_pelvis": bool(
            scores["pelvis"][target]
            == max(scores["pelvis"][name] for name in TARGETS)
        ),
        "target_dominates_any_joint": bool(
            scores["any_joint"][target]
            == max(scores["any_joint"][name] for name in TARGETS)
        ),
    }


def annotate_instances(axis, xyz: np.ndarray, instance_ids: np.ndarray) -> None:
    for instance_id, name in enumerate(INSTANCE_NAMES):
        mask = instance_ids == instance_id
        if not np.any(mask):
            continue
        center = np.median(xyz[mask, :2], axis=0)
        axis.text(
            float(center[0]),
            float(center[1]),
            name.capitalize(),
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.75},
        )


def plot_snapshot(
    output: Path,
    channel: str,
    representatives: Mapping[str, str],
    rows: Mapping[str, Mapping[str, object]],
    maps: Mapping[str, Mapping[str, np.ndarray]],
    candidate_step: int,
    k_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = (
        ("gt", "Motion GT contact"),
        ("original", "Original pretrained CDM"),
        ("lora", f"LoRA CDM step {candidate_step}"),
    )
    fig, axes = plt.subplots(
        len(TARGETS), len(conditions), figsize=(15.5, 14.0), constrained_layout=True
    )
    scatter = None
    for row_index, target in enumerate(TARGETS):
        sample_id = representatives[target]
        row = rows[sample_id]
        xyz = np.asarray(row["xyz"], dtype=np.float32)
        instance_ids = np.asarray(row["instance_ids"], dtype=np.int64)
        for column_index, (key, title) in enumerate(conditions):
            axis = axes[row_index, column_index]
            scalar = np.clip(
                reduce_affordance(maps[sample_id][key], channel), 0.0, 1.0
            )
            scatter = axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                c=scalar,
                s=4,
                cmap="turbo",
                vmin=0.0,
                vmax=1.0,
                rasterized=True,
            )
            annotate_instances(axis, xyz, instance_ids)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{target.capitalize()} | {title}")
            axis.set_xlabel("Chair-local X (m)")
            axis.set_ylabel("Chair-local Y = Unity local Z (m)")
            axis.text(
                0.01,
                0.01,
                sample_id,
                transform=axis.transAxes,
                fontsize=6,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none"},
            )
    if scatter is None:
        raise AssertionError("nothing plotted")
    label = "Pelvis affordance" if channel == "pelvis" else "Any-joint affordance"
    fig.colorbar(scatter, ax=axes, label=label, shrink=0.86)
    fig.suptitle(
        f"Train-only K={k_samples} paired rollout snapshot | {channel}\n"
        "Deterministic first train sample per target; no held-out tensor read",
        fontsize=15,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    # Keep the pure cache/plot helpers importable on CPU-only machines.  The
    # production loader lives in train_fewshot_cdm and imports torch, so load
    # it only for the actual Ubuntu-side snapshot command.
    from train_fewshot_cdm import load_rows

    args = parse_args()
    if args.k_samples < 5:
        raise ValueError("--k-samples must be at least 5")
    if args.candidate_step <= 0:
        raise ValueError("--candidate-step must be positive")
    if not 0.0 < args.active_threshold < 1.0:
        raise ValueError("--active-threshold must be in (0,1)")

    dataset_root = args.dataset_root.expanduser().resolve()
    split_file = args.split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    fewshot_dir = args.fewshot_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fewshot_dir / f"affordance_snapshot_step_{args.candidate_step}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(split_file)
    train_ids = [str(value) for value in split["cdm_fewshot"]["train"]]
    test_ids = {str(value) for value in split["cdm_fewshot"]["test"]}
    if set(train_ids) & test_ids:
        raise AssertionError("train/test overlap")
    mean, std = load_stats(stats_file)
    train_rows = load_rows(
        dataset_root,
        split,
        "train",
        mean,
        std,
        target_instance_weight=4.0,
        target_foreground_weight=16.0,
        active_threshold=args.active_threshold,
    )
    if set(train_rows) != set(train_ids):
        raise AssertionError("loaded rows do not equal the train partition")
    if set(train_rows) & test_ids:
        raise AssertionError("held-out row entered visualization")

    cache_dir, contract, candidate_result = validate_contract(
        fewshot_dir,
        split_file,
        train_ids,
        args.candidate_step,
        args.k_samples,
    )
    representatives = deterministic_representatives(train_rows)

    all_maps: Dict[str, Dict[str, np.ndarray]] = {}
    per_sample = []
    aggregate_values = defaultdict(lambda: defaultdict(list))
    for sample_id in sorted(train_rows):
        row = train_rows[sample_id]
        gt = np.asarray(row["gt"], dtype=np.float32)
        original = load_ensemble(
            cache_dir, "original", sample_id, args.k_samples
        )
        lora = load_ensemble(
            cache_dir,
            "candidate",
            sample_id,
            args.k_samples,
            candidate_step=args.candidate_step,
        )
        all_maps[sample_id] = {"gt": gt, "original": original, "lora": lora}
        target = str(row["target"])
        conditions = {
            "original": target_metrics(
                original,
                gt,
                np.asarray(row["instance_ids"]),
                target,
                int(row["target_instance_id"]),
                args.active_threshold,
            ),
            "lora": target_metrics(
                lora,
                gt,
                np.asarray(row["instance_ids"]),
                target,
                int(row["target_instance_id"]),
                args.active_threshold,
            ),
        }
        per_sample.append(
            {"sample_id": sample_id, "target": target, "conditions": conditions}
        )
        for condition, metrics in conditions.items():
            aggregate_values[target][f"{condition}_mae"].append(
                metrics["target_region_mae"]
            )
            aggregate_values[target][f"{condition}_f1"].append(
                metrics["target_region_f1_at_0_7"]
            )
            aggregate_values[target][f"{condition}_pelvis_dominance"].append(
                metrics["target_dominates_pelvis"]
            )
            aggregate_values[target][f"{condition}_any_joint_dominance"].append(
                metrics["target_dominates_any_joint"]
            )

    aggregate = {
        target: {
            key: float(np.mean(np.asarray(values, dtype=np.float64)))
            for key, values in metrics.items()
        }
        for target, metrics in aggregate_values.items()
    }
    pelvis_file = output_dir / "affordance_snapshot_pelvis.png"
    any_joint_file = output_dir / "affordance_snapshot_any_joint.png"
    plot_snapshot(
        pelvis_file,
        "pelvis",
        representatives,
        train_rows,
        all_maps,
        args.candidate_step,
        args.k_samples,
    )
    plot_snapshot(
        any_joint_file,
        "any_joint",
        representatives,
        train_rows,
        all_maps,
        args.candidate_step,
        args.k_samples,
    )

    representative_ids = [representatives[target] for target in TARGETS]
    np.savez_compressed(
        output_dir / "affordance_snapshot_arrays.npz",
        sample_ids=np.asarray(representative_ids),
        targets=np.asarray(TARGETS),
        gt=np.stack([all_maps[sample_id]["gt"] for sample_id in representative_ids]),
        original=np.stack(
            [all_maps[sample_id]["original"] for sample_id in representative_ids]
        ),
        lora=np.stack(
            [all_maps[sample_id]["lora"] for sample_id in representative_ids]
        ),
        xyz=np.stack(
            [np.asarray(train_rows[sample_id]["xyz"]) for sample_id in representative_ids]
        ),
        instance_ids=np.stack(
            [
                np.asarray(train_rows[sample_id]["instance_ids"])
                for sample_id in representative_ids
            ]
        ),
    )
    summary = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "PASS",
        "scientific_scope": "train_only_diagnostic",
        "partition": "train_complete",
        "heldout_sample_tensors_read": False,
        "cache_only_no_diffusion_sampling": True,
        "k_samples": args.k_samples,
        "candidate_step": args.candidate_step,
        "candidate_passed_strict_rollout_gate": bool(candidate_result["passed"]),
        "candidate_failed_checks": [
            key for key, value in candidate_result["checks"].items() if not value
        ],
        "rollout_contract_sha256": sha256_json(contract),
        "representative_policy": "lexicographically_first_train_sample_per_target",
        "representatives": representatives,
        "aggregate_definition": (
            "mean of per-sample metrics computed from each sample's K=5 ensemble"
        ),
        "aggregate_over_all_train_samples": aggregate,
        "per_sample": per_sample,
        "outputs": {
            "pelvis_png": str(pelvis_file),
            "any_joint_png": str(any_joint_file),
            "arrays_npz": str(output_dir / "affordance_snapshot_arrays.npz"),
        },
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("[PASS] train-only cached affordance snapshot")
    print(f"[OK] candidate step: {args.candidate_step}")
    print(f"[OK] representatives: {representatives}")
    for target in TARGETS:
        values = aggregate[target]
        print(
            f"[AGG] {target:10s} "
            f"F1={values['original_f1']:.4f}->{values['lora_f1']:.4f} "
            f"MAE={values['original_mae']:.4f}->{values['lora_mae']:.4f} "
            f"dominance(p/a)={values['lora_pelvis_dominance']:.3f}/"
            f"{values['lora_any_joint_dominance']:.3f}"
        )
    print(f"[OK] saved: {pelvis_file}")
    print(f"[OK] saved: {any_joint_file}")
    print(f"[OK] saved: {summary_file}")


if __name__ == "__main__":
    main()

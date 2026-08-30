#!/usr/bin/env python3
"""Train-only full-diffusion audit for a few-shot CDM checkpoint.

This is deliberately placed between training and development evaluation.  It
uses exactly the train partition and the same full reverse-diffusion sampler as
the evaluator, so a fixed-timestep x0 checkpoint cannot be mistaken for a
usable generative checkpoint.  No held-out sample tensor is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_fewshot_cdm import (  # noqa: E402
    binary_metrics,
    create_model,
    rollout_protocol_hashes,
    sample_contact_deterministic,
    stable_rollout_seeds,
)
from fewshot_cdm_common import (  # noqa: E402
    atomic_savez_compressed,
    atomic_write_json,
    denormalize_contact,
    instance_scores,
    load_split,
    load_stats,
    sha256_file,
)
from train_fewshot_cdm import (  # noqa: E402
    assert_original_checkpoint,
    compose_cdm_config,
    configure_reproducibility,
    load_rows,
)
from fewshot_cdm_v5_semantics import (  # noqa: E402
    PROMPT_POLICY_ID,
    prompt_for_target,
)
from fewshot_cdm_rollout_cache import (  # noqa: E402
    fingerprint_rows,
    sha256_json,
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
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path(
            "data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_"
            "contact_cont_joints_0.8_fur.npz"
        ),
    )
    parser.add_argument(
        "--original-checkpoint",
        type=Path,
        default=Path("outputs/CDM-Perceiver-ALL/ckpt/model300000.pt"),
    )
    parser.add_argument(
        "--fewshot-dir",
        type=Path,
        default=Path(
            "data/history_affordance_v1/experiments/"
            "fewshot_cdm_chair23_bed2_whiteboard12_v5r4"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--k-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chair-mae-degradation-limit", type=float, default=0.05)
    parser.add_argument("--novel-min-f1", type=float, default=0.30)
    parser.add_argument("--chair-min-dominance-rate", type=float, default=0.90)
    parser.add_argument(
        "--sit-bed-candidate-min-rate",
        type=float,
        default=0.80,
        help=(
            "Weak Sit-to-Bed candidate coverage; intentionally lower than "
            "strict Chair-primary dominance."
        ),
    )
    parser.add_argument("--novel-min-dominance-rate", type=float, default=1.0)
    parser.add_argument("--dominance-margin", type=float, default=0.01)
    return parser.parse_args()


def validate_train_summary(
    summary: Dict[str, object],
    summary_file: Path,
    checkpoint: Path,
    original_checkpoint: Path,
    split_file: Path,
) -> Dict[str, bool]:
    regularization = summary.get("regularization", {})
    objective = summary.get("data_objective", {})
    selection = summary.get("checkpoint_selection", {})
    rollout = selection.get("rollout_selection", {})
    selected = rollout.get("selected", {})
    selection_file = Path(str(rollout.get("selection_file", ""))).expanduser()
    selection_file_valid = (
        selection_file.is_file()
        and sha256_file(selection_file) == rollout.get("selection_file_sha256")
    )
    checks = {
        "training_schema_is_strict_v5r4": (
            summary.get("schema")
            == "history_affordance_v1_fewshot_cdm_train_v5r4"
        ),
        "training_status_pass": summary.get("status") == "PASS",
        "checkpoint_hash_matches_training": (
            sha256_file(checkpoint) == summary.get("checkpoint_sha256")
        ),
        "original_checkpoint_hash_matches_training": (
            sha256_file(original_checkpoint)
            == summary.get("initialization", {}).get("sha256")
        ),
        "split_hash_matches_training": (
            sha256_file(split_file) == summary.get("split", {}).get("sha256")
        ),
        "training_was_train_only": (
            summary.get("selection_data") == "train_only"
            and summary.get("test_sample_data_read_during_training") is False
        ),
        "zero_init_lora_and_multinoise_teacher_contract_recorded": (
            regularization.get("method")
            == (
                "frozen_original_cdm_plus_zero_init_lora_and_"
                "chair_region_multinoise_teacher_v5r4"
            )
            and regularization.get("zero_init_bitwise_original") is True
            and regularization.get("lora", {}).get("base_parameters_frozen") is True
            and regularization.get("lora", {}).get(
                "zero_initialized_output_projection"
            )
            is True
            and regularization.get("lora", {}).get("export_format")
            == "merged_legacy_partial_state_dict"
            and float(regularization.get("chair_teacher_weight", 0.0)) > 0.0
            and float(regularization.get("chair_high_timestep_weight", 0.0))
            > 0.0
            and float(regularization.get("chair_high_teacher_weight", 0.0))
            > 0.0
            and regularization.get("chair_teacher_region")
            == "target_chair_instance_only"
            and regularization.get(
                "bed_candidate_region_excluded_from_chair_teacher"
            )
            is True
        ),
        "rollout_aligned_sparse_objective_recorded": (
            objective.get("method")
            == "rollout_aligned_sparse_semantic_multinoise_start_x_v5r4"
            and objective.get("diffusion_prediction_target") == "START_X"
            and objective.get("novel_high_timestep_extra_forward") is True
            and objective.get("chair_high_timestep_extra_forward") is True
            and float(objective.get("semantic_weight", 0.0)) > 0.0
            and float(objective.get("foreground_bce_weight", 0.0)) > 0.0
            and float(objective.get("dice_weight", 0.0)) > 0.0
            and float(objective.get("ranking_weight", 0.0)) > 0.0
            and float(objective.get("high_timestep_weight", 0.0)) > 0.0
        ),
        "v5r4_prompt_and_semantic_contract_recorded": (
            objective.get("prompt_policy_id") == PROMPT_POLICY_ID
            and objective.get("prompt_by_target")
            == {
                target: prompt_for_target(target)
                for target in ("chair", "bed", "whiteboard")
            }
            and objective.get("whiteboard_semantic_channel")
            == "right_wrist_native_index_5"
            and objective.get("sit_multicandidate", {}).get("supervision_type")
            == "weak_semantic_prior_only"
            and objective.get("sit_multicandidate", {}).get(
                "motion_gt_relabelled_or_copied"
            )
            is False
            and abs(
                float(objective.get("sit_multicandidate", {}).get("weight", 0.0))
                - 0.50
            )
            <= 1e-12
            and objective.get("sit_multicandidate", {}).get("bed_any_joint_band")
            == [0.12, 0.40]
            and objective.get("sit_multicandidate", {}).get("bed_pelvis_band")
            == [0.06, 0.25]
            and abs(
                float(
                    objective.get("sit_multicandidate", {}).get(
                        "chair_primary_margin", 0.0
                    )
                )
                - 0.15
            )
            <= 1e-12
            and abs(
                float(
                    summary.get("checkpoint_selection", {}).get(
                        "sit_bed_candidate_min_rate", 0.0
                    )
                )
                - 0.80
            )
            <= 1e-12
        ),
        "one_step_and_semantic_shortlist_gate_passed": (
            selection.get("gate_passed") is True
            and selection.get("best", {}).get("loss_gate", {}).get("passed")
            is True
            and selection.get("best", {}).get("semantic_gate", {}).get("passed")
            is True
        ),
        "train_only_full_rollout_selected_checkpoint": (
            selection.get("method")
            == "one_step_shortlist_then_train_full_rollout"
            and selection.get("rollout_gate_passed") is True
            and selected.get("passed") is True
            and selected.get("checkpoint_sha256") == sha256_file(checkpoint)
            and rollout.get("partition") == "train_complete"
            and rollout.get("complete_train_partition") is True
            and rollout.get("seed_partition") == "train_audit"
            and int(rollout.get("k_samples", 0)) >= 5
            and len(rollout.get("probe_sample_ids", [])) == 25
            and int(rollout.get("evaluated_candidate_count", 0)) > 0
            and selection_file_valid
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{summary_file}: checkpoint provenance failed: {checks}"
        )
    return checks


def aggregate_metrics(per_sample: List[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {}
    for target in ("chair", "bed", "whiteboard", "novel"):
        selected = [
            row
            for row in per_sample
            if row["target"] == target
            or (target == "novel" and row["target"] in {"bed", "whiteboard"})
        ]
        aggregate[target] = {}
        for model_name in ("original", "fewshot", "zero_contact"):
            aggregate[target][model_name] = {
                metric: float(
                    np.mean(
                        [
                            row["target_region_metrics"][model_name][metric]
                            for row in selected
                        ]
                    )
                )
                for metric in ("mae", "foreground_mae", "f1_at_0_7", "correlation")
            }
    return aggregate


def validate_effective_checkpoint_state(
    original_model: torch.nn.Module,
    fewshot_model: torch.nn.Module,
    checkpoint: Path,
) -> Dict[str, bool]:
    """Compensate for the upstream loader's intentionally non-strict behavior."""
    saved = torch.load(checkpoint, map_location="cpu")
    if not isinstance(saved, dict):
        raise TypeError(f"{checkpoint}: expected a state-dict checkpoint")
    model_keys = set(fewshot_model.state_dict())
    expected_saved_keys = {
        key
        for key in model_keys
        if not any(
            token in key
            for token in ("scene_model", "clip_model", "text_model", "bert_model")
        )
    }
    actual_saved_keys = set(saved)
    checkpoint_keys_exact = actual_saved_keys == expected_saved_keys
    original_state = original_model.state_dict()
    fewshot_state = fewshot_model.state_dict()
    frozen_keys = sorted(model_keys - expected_saved_keys)
    frozen_state_equal = all(
        torch.equal(original_state[key], fewshot_state[key]) for key in frozen_keys
    )
    checks = {
        "fewshot_checkpoint_keys_exact": checkpoint_keys_exact,
        "constructor_loaded_frozen_state_bitwise_equal": frozen_state_equal,
    }
    if not all(checks.values()):
        missing = sorted(expected_saved_keys - actual_saved_keys)
        unexpected = sorted(actual_saved_keys - expected_saved_keys)
        raise RuntimeError(
            f"effective checkpoint reconstruction failed: {checks}; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return checks


def main() -> None:
    args = parse_args()
    strict_gate_values = {
        "diffusion_steps": 500,
        "chair_mae_degradation_limit": 0.05,
        "novel_min_f1": 0.30,
        "chair_min_dominance_rate": 0.90,
        "sit_bed_candidate_min_rate": 0.80,
        "novel_min_dominance_rate": 1.0,
        "dominance_margin": 0.01,
    }
    mismatched_gate_values = {
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in strict_gate_values.items()
        if getattr(args, name) != expected
    }
    if mismatched_gate_values:
        raise ValueError(
            "strict train rollout audit gate values changed: "
            + str(mismatched_gate_values)
        )
    if args.k_samples != 5:
        raise ValueError("strict train rollout audit requires exactly --k-samples 5")
    if args.seed != 20260815:
        raise ValueError("strict train rollout audit requires --seed 20260815")
    if torch.device(args.device).type != "cuda":
        raise ValueError(
            "strict train rollout audit requires CUDA/pointops_cuda"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 0.0 <= args.chair_mae_degradation_limit <= 0.20:
        raise ValueError("invalid Chair degradation limit")
    if not 0.0 < args.novel_min_f1 <= 1.0:
        raise ValueError("--novel-min-f1 must be in (0,1]")
    if not 0.0 < args.sit_bed_candidate_min_rate <= 1.0:
        raise ValueError("--sit-bed-candidate-min-rate must be in (0,1]")
    if args.sit_bed_candidate_min_rate >= args.chair_min_dominance_rate:
        raise ValueError(
            "weak Sit-Bed coverage must be lower than strict Chair dominance"
        )
    if args.dominance_margin <= 0.0:
        raise ValueError("--dominance-margin must be positive")
    configure_reproducibility(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    split_file = args.split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    fewshot_dir = args.fewshot_dir.expanduser().resolve()
    checkpoint = fewshot_dir / "fewshot_cdm.pt"
    train_summary_file = fewshot_dir / "summary.json"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fewshot_dir / f"train_full_rollout_k{args.k_samples}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_original_checkpoint(original_checkpoint, False)
    if not checkpoint.is_file() or not train_summary_file.is_file():
        raise FileNotFoundError("few-shot checkpoint/summary is incomplete")
    train_summary = json.loads(train_summary_file.read_text(encoding="utf-8"))
    provenance_checks = validate_train_summary(
        train_summary,
        train_summary_file,
        checkpoint,
        original_checkpoint,
        split_file,
    )

    split = load_split(split_file)
    train_ids = set(str(value) for value in split["cdm_fewshot"]["train"])
    test_ids = set(str(value) for value in split["cdm_fewshot"]["test"])
    if train_ids & test_ids:
        raise AssertionError("train/test IDs overlap")
    mean, std = load_stats(stats_file)
    objective = train_summary["data_objective"]
    sit_policy = objective["sit_multicandidate"]
    sit_bed_any_min, sit_bed_any_max = map(
        float, sit_policy["bed_any_joint_band"]
    )
    sit_bed_pelvis_min, sit_bed_pelvis_max = map(
        float, sit_policy["bed_pelvis_band"]
    )
    sit_chair_bed_margin = float(sit_policy["chair_primary_margin"])
    rows = load_rows(
        dataset_root,
        split,
        "train",
        mean,
        std,
        float(objective["target_instance_additive_weight"]),
        float(objective["target_foreground_additive_weight"]),
        float(objective["active_threshold"]),
    )
    if set(rows) != train_ids:
        raise AssertionError("train rollout loader returned unexpected sample IDs")
    counts = Counter(str(row["target"]) for row in rows.values())
    if counts != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise AssertionError(f"unexpected train counts: {counts}")

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    original_model, original_diffusion = create_model(
        cfg, original_checkpoint, args.device
    )
    fewshot_model, fewshot_diffusion = create_model(cfg, checkpoint, args.device)
    if original_diffusion.num_timesteps != fewshot_diffusion.num_timesteps:
        raise AssertionError("original/few-shot diffusion schedules differ")
    checkpoint_structure_checks = validate_effective_checkpoint_state(
        original_model, fewshot_model, checkpoint
    )

    per_sample: List[Dict[str, object]] = []
    saved_ids = []
    saved_gt = []
    saved_original = []
    saved_fewshot = []
    saved_original_draws = []
    saved_fewshot_draws = []
    repeatability_canary = None
    candidate_names = ("chair", "bed", "whiteboard")
    for sample_id in sorted(rows):
        row = rows[sample_id]
        original_samples = []
        fewshot_samples = []
        for k in range(args.k_samples):
            initial_seed, reverse_seed = stable_rollout_seeds(
                args.seed, "train_audit", sample_id, k
            )
            original_samples.append(
                denormalize_contact(
                    sample_contact_deterministic(
                        original_model,
                        original_diffusion,
                        row,
                        initial_seed,
                        reverse_seed,
                        args.device,
                    ),
                    mean,
                    std,
                )
            )
            fewshot_samples.append(
                denormalize_contact(
                    sample_contact_deterministic(
                        fewshot_model,
                        fewshot_diffusion,
                        row,
                        initial_seed,
                        reverse_seed,
                        args.device,
                    ),
                    mean,
                    std,
                )
            )
            if repeatability_canary is None:
                repeated = denormalize_contact(
                    sample_contact_deterministic(
                        fewshot_model,
                        fewshot_diffusion,
                        row,
                        initial_seed,
                        reverse_seed,
                        args.device,
                    ),
                    mean,
                    std,
                )
                repeatability_canary = bool(
                    np.array_equal(fewshot_samples[-1], repeated)
                )
                if not repeatability_canary:
                    raise RuntimeError(
                        "full-diffusion repeatability canary failed; paired "
                        "model comparison is invalid"
                    )
        original = np.mean(original_samples, axis=0).astype(np.float32)
        fewshot = np.mean(fewshot_samples, axis=0).astype(np.float32)
        gt = np.asarray(row["gt"], dtype=np.float32)
        zero = np.zeros_like(gt)
        instance_ids = np.asarray(row["instance_ids"], dtype=np.int64)
        target_instance_id = int(row["target_instance_id"])
        target_mask = instance_ids == target_instance_id
        target = str(row["target"])
        scores = {
            channel: instance_scores(fewshot, instance_ids, channel)
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        dominance_margin = {
            channel: float(
                scores[channel][target]
                - max(
                    scores[channel][name]
                    for name in candidate_names
                    if name != target
                )
            )
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        dominance = {
            channel: dominance_margin[channel] >= args.dominance_margin
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        sit_chair_primary = (
            scores["any_joint"]["chair"]
            >= scores["any_joint"]["bed"] + sit_chair_bed_margin
            and scores["pelvis"]["chair"]
            >= scores["pelvis"]["bed"] + sit_chair_bed_margin
        )
        sit_bed_candidate_visible = (
            sit_bed_any_min <= scores["any_joint"]["bed"] <= sit_bed_any_max
            and sit_bed_pelvis_min
            <= scores["pelvis"]["bed"]
            <= sit_bed_pelvis_max
        )
        draw_metrics = {"original": [], "fewshot": [], "zero_contact": []}
        draw_dominance = []
        draw_margins = []
        for original_draw, fewshot_draw in zip(original_samples, fewshot_samples):
            draw_metrics["original"].append(
                binary_metrics(original_draw[target_mask], gt[target_mask])
            )
            draw_metrics["fewshot"].append(
                binary_metrics(fewshot_draw[target_mask], gt[target_mask])
            )
            draw_metrics["zero_contact"].append(
                binary_metrics(zero[target_mask], gt[target_mask])
            )
            draw_scores = {
                channel: instance_scores(fewshot_draw, instance_ids, channel)
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            margins = {
                channel: float(
                    draw_scores[channel][target]
                    - max(
                        draw_scores[channel][name]
                        for name in candidate_names
                        if name != target
                    )
                )
                for channel in ("pelvis", "any_joint", "right_wrist")
            }
            draw_margins.append(margins)
            draw_dominance.append(
                {
                    channel: margins[channel] >= args.dominance_margin
                    for channel in ("pelvis", "any_joint", "right_wrist")
                }
            )
            draw_dominance[-1]["sit_chair_primary"] = (
                draw_scores["any_joint"]["chair"]
                >= draw_scores["any_joint"]["bed"] + sit_chair_bed_margin
                and draw_scores["pelvis"]["chair"]
                >= draw_scores["pelvis"]["bed"] + sit_chair_bed_margin
            )
            draw_dominance[-1]["sit_bed_candidate_visible"] = (
                sit_bed_any_min
                <= draw_scores["any_joint"]["bed"]
                <= sit_bed_any_max
                and sit_bed_pelvis_min
                <= draw_scores["pelvis"]["bed"]
                <= sit_bed_pelvis_max
            )
        record = {
            "sample_id": sample_id,
            "scene_id": row["scene_id"],
            "target": target,
            "text": row["text"],
            "target_region_metrics": {
                "original": binary_metrics(original[target_mask], gt[target_mask]),
                "fewshot": binary_metrics(fewshot[target_mask], gt[target_mask]),
                "zero_contact": binary_metrics(zero[target_mask], gt[target_mask]),
            },
            "per_draw_target_region_metrics": draw_metrics,
            "fewshot_candidate_scores": {
                channel: {name: scores[channel][name] for name in candidate_names}
                for channel in ("pelvis", "any_joint", "right_wrist")
            },
            "target_dominates_candidates": dominance,
            "target_dominance_margin": dominance_margin,
            "per_draw_target_dominance": draw_dominance,
            "per_draw_target_dominance_margin": draw_margins,
            "sit_chair_primary": sit_chair_primary,
            "sit_bed_candidate_visible": sit_bed_candidate_visible,
        }
        per_sample.append(record)
        saved_ids.append(sample_id)
        saved_gt.append(gt)
        saved_original.append(original)
        saved_fewshot.append(fewshot)
        saved_original_draws.append(np.stack(original_samples))
        saved_fewshot_draws.append(np.stack(fewshot_samples))
        print(
            f"[TRAIN-ROLLOUT] {sample_id} target={target} "
            f"F1={record['target_region_metrics']['original']['f1_at_0_7']:.4f}->"
            f"{record['target_region_metrics']['fewshot']['f1_at_0_7']:.4f} "
            f"dominance={dominance}"
        )

    aggregate = aggregate_metrics(per_sample)
    per_draw_aggregate: Dict[str, object] = {}
    for aggregate_target in ("chair", "bed", "whiteboard", "novel"):
        selected = [
            row
            for row in per_sample
            if row["target"] == aggregate_target
            or (
                aggregate_target == "novel"
                and row["target"] in {"bed", "whiteboard"}
            )
        ]
        per_draw_aggregate[aggregate_target] = {}
        for model_name in ("original", "fewshot", "zero_contact"):
            per_draw_aggregate[aggregate_target][model_name] = {
                metric: float(
                    np.mean(
                        [
                            draw[metric]
                            for row in selected
                            for draw in row["per_draw_target_region_metrics"][
                                model_name
                            ]
                        ]
                    )
                )
                for metric in (
                    "mae",
                    "foreground_mae",
                    "f1_at_0_7",
                    "correlation",
                )
            }
    dominance_rate = {}
    for target in candidate_names:
        selected = [row for row in per_sample if row["target"] == target]
        dominance_rate[target] = {
            channel: float(
                np.mean(
                    [row["target_dominates_candidates"][channel] for row in selected]
                )
            )
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        dominance_rate[target]["per_draw"] = {
            channel: float(
                np.mean(
                    [
                        draw[channel]
                        for row in selected
                        for draw in row["per_draw_target_dominance"]
                    ]
                )
            )
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
    chair_original_draw_mae = float(
        per_draw_aggregate["chair"]["original"]["mae"]
    )
    chair_draw_degradation = (
        float(per_draw_aggregate["chair"]["fewshot"]["mae"])
        - chair_original_draw_mae
    ) / max(chair_original_draw_mae, 1e-12)
    chair_samples = [row for row in per_sample if row["target"] == "chair"]
    sit_ensemble_chair_primary_rate = float(
        np.mean([row["sit_chair_primary"] for row in chair_samples])
    )
    sit_ensemble_bed_candidate_rate = float(
        np.mean([row["sit_bed_candidate_visible"] for row in chair_samples])
    )
    sit_per_draw_chair_primary_rate = float(
        np.mean(
            [
                draw["sit_chair_primary"]
                for row in chair_samples
                for draw in row["per_draw_target_dominance"]
            ]
        )
    )
    sit_per_draw_bed_candidate_rate = float(
        np.mean(
            [
                draw["sit_bed_candidate_visible"]
                for row in chair_samples
                for draw in row["per_draw_target_dominance"]
            ]
        )
    )
    checks = {
        **provenance_checks,
        **checkpoint_structure_checks,
        "partition_is_train_only": set(rows) == train_ids,
        "heldout_sample_tensors_never_read": True,
        "paired_noise_full_diffusion_k_at_least_5": args.k_samples >= 5,
        "full_trajectory_repeatability_canary": repeatability_canary is True,
        "chair_rollout_mae_retained": (
            chair_draw_degradation <= args.chair_mae_degradation_limit
        ),
        "bed_rollout_mae_improved": (
            float(per_draw_aggregate["bed"]["fewshot"]["mae"])
            < float(per_draw_aggregate["bed"]["original"]["mae"])
        ),
        "whiteboard_rollout_mae_improved": (
            float(per_draw_aggregate["whiteboard"]["fewshot"]["mae"])
            < float(per_draw_aggregate["whiteboard"]["original"]["mae"])
        ),
        "bed_rollout_mae_beats_zero": (
            float(per_draw_aggregate["bed"]["fewshot"]["mae"])
            < float(per_draw_aggregate["bed"]["zero_contact"]["mae"])
        ),
        "whiteboard_rollout_mae_beats_zero": (
            float(per_draw_aggregate["whiteboard"]["fewshot"]["mae"])
            < float(per_draw_aggregate["whiteboard"]["zero_contact"]["mae"])
        ),
        "bed_rollout_f1_improved": (
            float(per_draw_aggregate["bed"]["fewshot"]["f1_at_0_7"])
            > float(per_draw_aggregate["bed"]["original"]["f1_at_0_7"])
        ),
        "whiteboard_rollout_f1_improved": (
            float(per_draw_aggregate["whiteboard"]["fewshot"]["f1_at_0_7"])
            > float(
                per_draw_aggregate["whiteboard"]["original"]["f1_at_0_7"]
            )
        ),
        "bed_rollout_f1_active": (
            float(per_draw_aggregate["bed"]["fewshot"]["f1_at_0_7"])
            >= args.novel_min_f1
        ),
        "whiteboard_rollout_f1_active": (
            float(
                per_draw_aggregate["whiteboard"]["fewshot"]["f1_at_0_7"]
            )
            >= args.novel_min_f1
        ),
        "chair_rollout_any_joint_dominance": (
            dominance_rate["chair"]["any_joint"]
            >= args.chair_min_dominance_rate
        ),
        "chair_rollout_pelvis_dominance": (
            dominance_rate["chair"]["pelvis"]
            >= args.chair_min_dominance_rate
        ),
        "chair_per_draw_dominance_rate": (
            dominance_rate["chair"]["per_draw"]["pelvis"] >= 0.80
            and dominance_rate["chair"]["per_draw"]["any_joint"] >= 0.80
        ),
        "sit_chair_remains_primary": (
            sit_ensemble_chair_primary_rate
            >= args.chair_min_dominance_rate
        ),
        "sit_bed_candidate_visible_and_bounded": (
            sit_ensemble_bed_candidate_rate
            >= args.sit_bed_candidate_min_rate
        ),
        "sit_per_draw_contract_rate": (
            sit_per_draw_chair_primary_rate
            >= args.chair_min_dominance_rate
            and sit_per_draw_bed_candidate_rate
            >= args.sit_bed_candidate_min_rate
        ),
        "bed_rollout_any_joint_dominance": (
            dominance_rate["bed"]["any_joint"]
            >= args.novel_min_dominance_rate
        ),
        "bed_rollout_pelvis_dominance": (
            dominance_rate["bed"]["pelvis"]
            >= args.novel_min_dominance_rate
        ),
        "bed_per_draw_dominance_rate": (
            dominance_rate["bed"]["per_draw"]["pelvis"] >= 0.80
            and dominance_rate["bed"]["per_draw"]["any_joint"] >= 0.80
        ),
        "whiteboard_rollout_any_joint_dominance": (
            dominance_rate["whiteboard"]["any_joint"]
            >= args.novel_min_dominance_rate
        ),
        "whiteboard_rollout_right_wrist_dominance": (
            dominance_rate["whiteboard"]["right_wrist"]
            >= args.novel_min_dominance_rate
        ),
        "whiteboard_per_draw_any_joint_dominance_rate": (
            dominance_rate["whiteboard"]["per_draw"]["any_joint"] >= 0.80
        ),
        "whiteboard_per_draw_right_wrist_dominance_rate": (
            dominance_rate["whiteboard"]["per_draw"]["right_wrist"] >= 0.80
        ),
        "v5_prompts_exact_and_object_agnostic": all(
            str(row["text"]) == prompt_for_target(str(row["target"]))
            for row in per_sample
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    sample_fingerprints = fingerprint_rows(rows, sorted(rows))
    dataset_snapshot_payload = {
        "schema": "history_affordance_v1_rollout_dataset_snapshot_v1",
        "partition": "train",
        "split_sha256": sha256_file(split_file),
        "stats_file_sha256": sha256_file(stats_file),
        "sample_fingerprints": sample_fingerprints,
    }
    summary = {
        "schema": "history_affordance_v1_fewshot_cdm_train_rollout_audit_v6",
        "status": status,
        "decision": (
            "TRAIN_ROLLOUT_VALID"
            if status == "PASS"
            else "FIXED_X0_GATE_DOES_NOT_ESTABLISH_FULL_SAMPLING_QUALITY"
        ),
        "partition": "train",
        "test_sample_tensors_read": False,
        "train_count": len(rows),
        "train_target_counts": dict(counts),
        "train_sample_ids": sorted(rows),
        "k_samples": args.k_samples,
        "paired_noise": True,
        "sampling_provenance": {
            "base_seed": args.seed,
            "partition": "train_audit",
            "seed_derivation": "sha256(base|partition|sample_id|draw)",
            "seed_table_storage": "prediction_npz_int64_matrices",
        },
        "pairing_protocol": {
            "seed_derivation": "sha256(base|partition|sample_id|draw)",
            "initial_xT_and_all_reverse_step_noise_paired": True,
            "caller_rng_state_restored": True,
            "repeatability_canary_bitwise_equal": repeatability_canary,
            "metrics_computed_per_draw": True,
            "ensemble_map_metrics_are_secondary": True,
        },
        "diffusion_steps": args.diffusion_steps,
        "quality_thresholds": {
            "schema": "fewshot_cdm_rollout_quality_thresholds_v1",
            "active_threshold": 0.7,
            "instance_top_fraction": 0.1,
            "chair_mae_degradation_limit": args.chair_mae_degradation_limit,
            "novel_min_f1": args.novel_min_f1,
            "chair_min_dominance_rate": args.chair_min_dominance_rate,
            "sit_bed_candidate_min_rate": args.sit_bed_candidate_min_rate,
            "novel_min_dominance_rate": args.novel_min_dominance_rate,
            "dominance_margin": args.dominance_margin,
            "per_draw_min_rate": 0.80,
            "sit_chair_primary_margin": sit_chair_bed_margin,
            "sit_bed_any_joint_band": [sit_bed_any_min, sit_bed_any_max],
            "sit_bed_pelvis_band": [
                sit_bed_pelvis_min,
                sit_bed_pelvis_max,
            ],
            "candidate_instance_ids": {
                "chair": 1,
                "bed": 2,
                "whiteboard": 3,
            },
            "channel_indices": {
                "pelvis": 0,
                "right_wrist": 5,
                "any_joint": "max_over_6",
            },
        },
        "rollout_protocol_hashes": rollout_protocol_hashes(),
        "stats_file": str(stats_file),
        "stats_file_sha256": sha256_file(stats_file),
        "original_checkpoint": str(original_checkpoint),
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_summary": str(train_summary_file),
        "train_summary_sha256": sha256_file(train_summary_file),
        "split": str(split_file),
        "split_sha256": sha256_file(split_file),
        "dataset_snapshot": {
            **dataset_snapshot_payload,
            "sha256": sha256_json(dataset_snapshot_payload),
        },
        "checks": checks,
        "sit_contract": {
            "chair_primary_min_rate": args.chair_min_dominance_rate,
            "bed_candidate_min_rate": args.sit_bed_candidate_min_rate,
            "ensemble_chair_primary_rate": sit_ensemble_chair_primary_rate,
            "ensemble_bed_candidate_rate": sit_ensemble_bed_candidate_rate,
            "per_draw_chair_primary_rate": sit_per_draw_chair_primary_rate,
            "per_draw_bed_candidate_rate": sit_per_draw_bed_candidate_rate,
        },
        "chair_mae_relative_degradation": chair_draw_degradation,
        "dominance_margin_required": args.dominance_margin,
        "dominance_rate": dominance_rate,
        "aggregate_target_region": aggregate,
        "per_draw_aggregate_target_region": per_draw_aggregate,
        "per_sample": per_sample,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    predictions_file = output_dir / "train_rollout_predictions.npz"
    saved_seed_pairs = np.asarray(
        [
            [
                stable_rollout_seeds(args.seed, "train_audit", sample_id, draw)
                for draw in range(args.k_samples)
            ]
            for sample_id in saved_ids
        ],
        dtype=np.int64,
    )
    atomic_savez_compressed(
        predictions_file,
        sample_ids=np.asarray(saved_ids),
        gt=np.stack(saved_gt),
        original=np.stack(saved_original),
        fewshot=np.stack(saved_fewshot),
        original_draws=np.stack(saved_original_draws),
        fewshot_draws=np.stack(saved_fewshot_draws),
        initial_noise_seeds=saved_seed_pairs[:, :, 0],
        reverse_noise_seeds=saved_seed_pairs[:, :, 1],
    )
    summary["predictions"] = {
        "file": str(predictions_file),
        "sha256": sha256_file(predictions_file),
        "format": "paired_rollout_affordance_npz_v2",
    }
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, summary)
    print(f"[{status}] train-only full-diffusion rollout audit")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"[OK] saved: {summary_file}")
    print(f"[OK] saved: {predictions_file}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

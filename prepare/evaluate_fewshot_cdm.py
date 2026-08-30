#!/usr/bin/env python3
"""Held-out paired-noise Base-ADM evaluation for few-shot CDM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot_cdm_common import (  # noqa: E402
    atomic_savez_compressed,
    atomic_write_json,
    denormalize_contact,
    gt_file_from_entry,
    instance_scores,
    load_index_entries,
    load_scene,
    load_split,
    load_stats,
    normalize_contact,
    sha256_file,
)
from train_fewshot_cdm import (  # noqa: E402
    assert_original_checkpoint,
    compose_cdm_config,
    configure_reproducibility,
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
    parser.add_argument(
        "--train-rollout-audit",
        type=Path,
        required=True,
        help=(
            "PASS summary from audit_fewshot_cdm_train_rollout.py. "
            "The evaluator refuses to read development tensors without it."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--k-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chair-mae-degradation-limit", type=float, default=0.05)
    parser.add_argument("--novel-min-f1", type=float, default=0.30)
    parser.add_argument("--dominance-margin", type=float, default=0.01)
    return parser.parse_args()


def validate_train_rollout_audit(
    audit: Mapping[str, object],
    audit_file: Path,
    train_summary_file: Path,
    fewshot_checkpoint: Path,
    original_checkpoint: Path,
    stats_file: Path,
    split_file: Path,
    split: Mapping[str, object],
    diffusion_steps: int,
    evaluation_k_samples: int,
) -> Dict[str, bool]:
    """Reject stale, weak, or non-train-only rollout evidence.

    This function must run before ``load_test_rows``.  File hashes bind the
    audit to the exact checkpoint and inputs being evaluated; sample IDs bind
    it to the complete train partition rather than a convenient subset.
    """
    expected_train_ids = sorted(str(value) for value in split["cdm_fewshot"]["train"])
    expected_counts = {"chair": 18, "whiteboard": 6, "bed": 1}
    recorded_checks = audit.get("checks", {})
    checks = {
        "train_rollout_audit_schema_valid": (
            audit.get("schema")
            == "history_affordance_v1_fewshot_cdm_train_rollout_audit_v6"
        ),
        "train_rollout_audit_status_pass": audit.get("status") == "PASS",
        "train_rollout_partition_is_train": audit.get("partition") == "train",
        "train_rollout_read_no_development_tensors": (
            audit.get("test_sample_tensors_read") is False
        ),
        "train_rollout_complete_train_partition": (
            int(audit.get("train_count", -1)) == len(expected_train_ids)
            and sorted(str(value) for value in audit.get("train_sample_ids", []))
            == expected_train_ids
            and audit.get("train_target_counts") == expected_counts
        ),
        "train_rollout_used_paired_full_diffusion": (
            audit.get("paired_noise") is True
            and int(audit.get("diffusion_steps", -1)) == diffusion_steps
            and int(audit.get("k_samples", -1)) >= max(5, evaluation_k_samples)
        ),
        "train_rollout_checkpoint_hash_matches": (
            audit.get("checkpoint_sha256") == sha256_file(fewshot_checkpoint)
        ),
        "train_rollout_original_hash_matches": (
            audit.get("original_checkpoint_sha256")
            == sha256_file(original_checkpoint)
        ),
        "train_rollout_stats_hash_matches": (
            audit.get("stats_file_sha256") == sha256_file(stats_file)
        ),
        "train_rollout_protocol_hashes_match": (
            audit.get("rollout_protocol_hashes") == rollout_protocol_hashes()
        ),
        "train_rollout_pairing_contract_valid": (
            audit.get("pairing_protocol", {}).get(
                "initial_xT_and_all_reverse_step_noise_paired"
            )
            is True
            and audit.get("pairing_protocol", {}).get(
                "caller_rng_state_restored"
            )
            is True
            and audit.get("pairing_protocol", {}).get(
                "repeatability_canary_bitwise_equal"
            )
            is True
            and audit.get("pairing_protocol", {}).get(
                "metrics_computed_per_draw"
            )
            is True
        ),
        "train_rollout_summary_hash_matches": (
            audit.get("train_summary_sha256") == sha256_file(train_summary_file)
        ),
        "train_rollout_split_hash_matches": (
            audit.get("split_sha256") == sha256_file(split_file)
        ),
        "every_train_rollout_gate_passed": (
            isinstance(recorded_checks, dict)
            and bool(recorded_checks)
            and all(value is True for value in recorded_checks.values())
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{audit_file}: train-only full-diffusion audit is not valid for "
            f"this evaluation: {checks}. Development tensors were not read."
        )
    return checks


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    threshold = 0.7
    pred_active = prediction >= threshold
    target_active = target >= threshold
    tp = int(np.logical_and(pred_active, target_active).sum())
    fp = int(np.logical_and(pred_active, ~target_active).sum())
    fn = int(np.logical_and(~pred_active, target_active).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    active = target_active
    foreground_mae = float(np.abs(prediction[active] - target[active]).mean()) if active.any() else 0.0
    flat_pred = prediction.reshape(-1).astype(np.float64)
    flat_target = target.reshape(-1).astype(np.float64)
    if float(flat_pred.std()) == 0.0 or float(flat_target.std()) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flat_pred, flat_target)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    return {
        "mae": float(np.abs(prediction - target).mean()),
        "foreground_mae": foreground_mae,
        "f1_at_0_7": float(f1),
        "correlation": correlation,
    }


def load_test_rows(dataset_root, split, mean, std):
    test_ids = [str(v) for v in split["cdm_fewshot"]["test"]]
    train_ids = set(str(v) for v in split["cdm_fewshot"]["train"])
    if train_ids & set(test_ids):
        raise AssertionError("train/test overlap")
    entries = load_index_entries(dataset_root, test_ids)
    split_meta = {str(v["sample_id"]): v for v in split["samples"]}
    scene_cache = {}
    rows = {}
    for sample_id in test_ids:
        entry = entries[sample_id]
        scene_id = str(entry["scene_id"])
        scene = scene_cache.get(scene_id)
        if scene is None:
            scene = load_scene(dataset_root, entry)
            scene_cache[scene_id] = scene
        gt_file = gt_file_from_entry(dataset_root, entry)
        gt_npz = np.load(gt_file, allow_pickle=False)
        gt = gt_npz["affordance"].astype(np.float32)
        if not np.array_equal(gt_npz["source_indices"], scene["source_indices"]):
            raise ValueError(f"{sample_id}: GT point order mismatch")
        target = str(split_meta[sample_id]["target"])
        source_text = str(entry["text"])
        text = prompt_for_target(target)
        rows[sample_id] = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "target": target,
            "target_instance_id": int(entry["target_instance_id"]),
            "text": text,
            "source_text": source_text,
            "prompt_policy": PROMPT_POLICY_ID,
            "gt": gt,
            "x": normalize_contact(gt, mean, std),
            "xyz": scene["points"][:, :3].astype(np.float32),
            "feat": (scene["points"][:, 3:6] / 255.0).astype(np.float32),
            "instance_ids": scene["instance_ids"],
            "source_indices": scene["source_indices"],
        }
    return rows


def create_model(cfg, checkpoint: Path, device: str):
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)
    load_ckpt(model, str(checkpoint))
    model.eval()
    return model, diffusion


@torch.no_grad()
def sample_contact(model, diffusion, row, noise: torch.Tensor, device: str):
    xyz = torch.from_numpy(row["xyz"][None]).to(device).contiguous()
    feat = torch.from_numpy(row["feat"][None]).to(device).contiguous()
    sample = diffusion.p_sample_loop(
        model,
        (1, 8192, 6),
        clip_denoised=False,
        noise=noise,
        model_kwargs={
            "c_pc_xyz": xyz,
            "c_pc_feat": feat,
            "c_text": [str(row["text"])],
        },
        device=device,
        progress=False,
    )
    return sample[0].detach().cpu().numpy().astype(np.float32)


def stable_rollout_seeds(
    base_seed: int, partition: str, sample_id: str, draw_index: int
) -> tuple[int, int]:
    """Return row-order-independent seeds for x_T and reverse-step noise."""
    payload = f"{base_seed}|{partition}|{sample_id}|{draw_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    limit = 2**63 - 1
    return (
        int.from_bytes(digest[:8], "big") % limit,
        int.from_bytes(digest[8:16], "big") % limit,
    )


def rollout_protocol_hashes() -> Dict[str, str]:
    """Hash the code/config files that define rollout comparison semantics."""
    relative_files = (
        "prepare/evaluate_fewshot_cdm.py",
        "prepare/audit_fewshot_cdm_train_rollout.py",
        "prepare/fewshot_cdm_common.py",
        "prepare/fewshot_cdm_lora.py",
        "prepare/fewshot_cdm_rollout_cache.py",
        "prepare/train_fewshot_cdm.py",
        "prepare/fewshot_cdm_v5_semantics.py",
        "models/base.py",
        "models/cdm.py",
        "models/functions.py",
        "models/modules.py",
        "models/scene_models/pointops.py",
        "models/scene_models/pointtransformer.py",
        "diffusion/gaussian_diffusion.py",
        "diffusion/losses.py",
        "diffusion/nn.py",
        "diffusion/resample.py",
        "diffusion/respace.py",
        "utils/misc.py",
        "utils/training.py",
        "configs/default.yaml",
        "configs/model/cdm.yaml",
        "configs/task/contact_gen.yaml",
    )
    files = {name: REPO_ROOT / name for name in relative_files}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"rollout protocol files missing: {missing}")
    return {name: sha256_file(path) for name, path in files.items()}


def validate_v5_training_provenance(
    train_summary: Mapping[str, object],
    original_checkpoint: Path,
    fewshot_checkpoint: Path,
    split_file: Path,
) -> Dict[str, bool]:
    """Validate the train-only LoRA/full-rollout contract before test I/O."""

    regularization = train_summary.get("regularization", {})
    objective = train_summary.get("data_objective", {})
    selection = train_summary.get("checkpoint_selection", {})
    rollout = selection.get("rollout_selection", {})
    selected = rollout.get("selected", {})
    selection_file = Path(str(rollout.get("selection_file", ""))).expanduser()
    selection_file_valid = (
        selection_file.is_file()
        and sha256_file(selection_file) == rollout.get("selection_file_sha256")
    )
    checks = {
        "training_schema_is_strict_v5r4": (
            train_summary.get("schema")
            == "history_affordance_v1_fewshot_cdm_train_v5r4"
        ),
        "original_checkpoint_hash_matches_training": (
            sha256_file(original_checkpoint)
            == train_summary.get("initialization", {}).get("sha256")
        ),
        "fewshot_checkpoint_hash_matches_training": (
            sha256_file(fewshot_checkpoint) == train_summary.get("checkpoint_sha256")
        ),
        "split_hash_matches_training": (
            sha256_file(split_file) == train_summary.get("split", {}).get("sha256")
        ),
        "training_selected_without_test": (
            train_summary.get("selection_data") == "train_only"
            and train_summary.get("test_partition_read_during_training") is False
            and train_summary.get("test_sample_data_read_during_training") is False
        ),
        "one_sample_checkpoint_absent": (
            train_summary.get("initialization", {}).get(
                "one_sample_diagnostic_checkpoint_used"
            )
            is False
        ),
        "chair3_bed1_whiteboard1_replay_recorded": (
            train_summary.get("sampling", {}).get("batch_size") == 5
            and train_summary.get("sampling", {}).get("chair_replay_per_step") == 3
            and train_summary.get("sampling", {}).get("bed_per_step") == 1
            and train_summary.get("sampling", {}).get("whiteboard_per_step") == 1
        ),
        "zero_init_lora_multinoise_frozen_base_recorded": (
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
            and objective.get("chair_uses_legacy_uniform_weights") is True
            and objective.get("background_remains_supervised") is True
            and objective.get("novel_high_timestep_extra_forward") is True
            and objective.get("chair_high_timestep_extra_forward") is True
            and float(objective.get("target_instance_additive_weight", 0.0)) > 0.0
            and float(objective.get("target_foreground_additive_weight", 0.0))
            > 0.0
            and float(objective.get("semantic_weight", 0.0)) > 0.0
            and float(objective.get("foreground_bce_weight", 0.0)) > 0.0
            and float(objective.get("dice_weight", 0.0)) > 0.0
            and float(objective.get("ranking_weight", 0.0)) > 0.0
            and float(objective.get("high_timestep_weight", 0.0)) > 0.0
            and float(
                objective.get("chair_uniform_legacy_parity_max_abs_diff", float("inf"))
            )
            <= 1e-6
        ),
        "v5r4_prompt_policy_recorded": (
            objective.get("prompt_policy_id") == PROMPT_POLICY_ID
            and objective.get("prompt_by_target")
            == {
                target: prompt_for_target(target)
                for target in ("chair", "bed", "whiteboard")
            }
            and objective.get("whiteboard_semantic_channel")
            == "right_wrist_native_index_5"
        ),
        "sit_multicandidate_v5r4_policy_recorded": (
            objective.get("sit_multicandidate", {}).get("supervision_type")
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
                float(selection.get("sit_bed_candidate_min_rate", 0.0)) - 0.80
            )
            <= 1e-12
        ),
        "one_step_shortlist_gate_passed": (
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
            and selected.get("checkpoint_sha256") == sha256_file(fewshot_checkpoint)
            and rollout.get("partition") == "train_complete"
            and rollout.get("complete_train_partition") is True
            and rollout.get("seed_partition") == "train_audit"
            and int(rollout.get("k_samples", 0)) >= 5
            and len(rollout.get("probe_sample_ids", [])) == 25
            and int(rollout.get("evaluated_candidate_count", 0)) > 0
            and selection_file_valid
        ),
        "heldout_sample_tensors_never_read_during_training": (
            train_summary.get("test_sample_data_read_during_training") is False
            and train_summary.get("test_ids_used_only_for_exclusion_assertion") is True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"training provenance failed: {checks}")
    return checks


@torch.no_grad()
def sample_contact_deterministic(
    model,
    diffusion,
    row,
    initial_noise_seed: int,
    reverse_noise_seed: int,
    device: str,
):
    """Run a complete rollout without consuming the caller's RNG stream.

    Passing the same two seeds gives both models the same x_T and the same
    random stream at every stochastic reverse-diffusion step.  Merely cloning
    x_T is insufficient because ``p_sample`` draws fresh noise internally.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(initial_noise_seed)
    noise = torch.randn((1, 8192, 6), generator=generator).to(device)
    torch_device = torch.device(device)
    # torch.manual_seed seeds every visible CUDA default generator even when
    # the sampled tensor targets CPU. Fork every visible device so neither a
    # CUDA nor CPU rollout can perturb caller CUDA RNG state.
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(reverse_noise_seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed(reverse_noise_seed)
        return sample_contact(model, diffusion, row, noise, device)


def plot_sample(output: Path, row, original, fewshot, gt) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = row["xyz"]
    maps = [gt.max(axis=1), original.max(axis=1), fewshot.max(axis=1)]
    titles = ["GT contact", "Untouched original CDM", "Few-shot CDM"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, scalar, title in zip(axes, maps, titles):
        scatter = axis.scatter(
            xyz[:, 0], xyz[:, 1], c=scalar, s=3, cmap="turbo", vmin=0.0, vmax=1.0
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("Chair-local X (m)")
        axis.set_ylabel("Chair-local Y = Unity local Z (m)")
    fig.colorbar(scatter, ax=axes, label="Any-joint affordance")
    fig.suptitle(f"{row['sample_id']} | target={row['target']} | object-agnostic prompt")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    strict_gate_values = {
        "diffusion_steps": 500,
        "chair_mae_degradation_limit": 0.05,
        "novel_min_f1": 0.30,
        "dominance_margin": 0.01,
    }
    mismatched_gate_values = {
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in strict_gate_values.items()
        if getattr(args, name) != expected
    }
    if mismatched_gate_values:
        raise ValueError(
            "strict development evaluation gate values changed: "
            + str(mismatched_gate_values)
        )
    if args.k_samples != 5:
        raise ValueError("strict development evaluation requires exactly --k-samples 5")
    if args.seed != 20260815:
        raise ValueError("strict development evaluation requires --seed 20260815")
    if not 0.0 < args.novel_min_f1 <= 1.0:
        raise ValueError("--novel-min-f1 must be in (0,1]")
    if args.dominance_margin <= 0.0:
        raise ValueError("--dominance-margin must be positive")
    if torch.device(args.device).type != "cuda":
        raise ValueError(
            "strict development evaluation requires CUDA/pointops_cuda"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configure_reproducibility(args.seed)
    dataset_root = args.dataset_root.expanduser().resolve()
    split_file = args.split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    fewshot_dir = args.fewshot_dir.expanduser().resolve()
    train_rollout_audit_file = args.train_rollout_audit.expanduser().resolve()
    fewshot_checkpoint = fewshot_dir / "fewshot_cdm.pt"
    train_summary_file = fewshot_dir / "summary.json"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fewshot_dir / "heldout_base_adm_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    assert_original_checkpoint(original_checkpoint, False)
    if not fewshot_checkpoint.is_file() or not train_summary_file.is_file():
        raise FileNotFoundError("few-shot checkpoint/summary is incomplete")
    if not train_rollout_audit_file.is_file():
        raise FileNotFoundError(
            f"Required train-only rollout audit is missing: "
            f"{train_rollout_audit_file}. Development tensors were not read."
        )
    train_summary = json.loads(train_summary_file.read_text(encoding="utf-8"))
    split = load_split(split_file)
    if train_summary.get("test_partition_read_during_training") is not False:
        raise RuntimeError("training summary does not prove held-out isolation")
    provenance_checks = validate_v5_training_provenance(
        train_summary,
        original_checkpoint,
        fewshot_checkpoint,
        split_file,
    )

    # Scientific firewall: this validation intentionally precedes
    # ``load_test_rows`` so a weak/stale checkpoint cannot consume the
    # development partition merely to discover that train-time x0 metrics did
    # not translate to actual reverse-diffusion samples.
    train_rollout_audit = json.loads(
        train_rollout_audit_file.read_text(encoding="utf-8")
    )
    rollout_checks = validate_train_rollout_audit(
        train_rollout_audit,
        train_rollout_audit_file,
        train_summary_file,
        fewshot_checkpoint,
        original_checkpoint,
        stats_file,
        split_file,
        split,
        args.diffusion_steps,
        args.k_samples,
    )
    provenance_checks.update(rollout_checks)
    sit_policy = train_summary["data_objective"]["sit_multicandidate"]
    sit_bed_any_min, sit_bed_any_max = map(
        float, sit_policy["bed_any_joint_band"]
    )
    sit_bed_pelvis_min, sit_bed_pelvis_max = map(
        float, sit_policy["bed_pelvis_band"]
    )
    sit_chair_bed_margin = float(sit_policy["chair_primary_margin"])

    mean, std = load_stats(stats_file)
    rows = load_test_rows(dataset_root, split, mean, std)
    counts = Counter(str(row["target"]) for row in rows.values())
    if counts != Counter({"chair": 5, "whiteboard": 6, "bed": 1}):
        raise AssertionError(f"unexpected held-out target counts: {counts}")
    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    original_model, original_diffusion = create_model(
        cfg, original_checkpoint, args.device
    )
    fewshot_model, fewshot_diffusion = create_model(
        cfg, fewshot_checkpoint, args.device
    )

    per_sample = []
    saved_sample_ids: List[str] = []
    saved_gt = []
    saved_original = []
    saved_fewshot = []
    saved_original_draws = []
    saved_fewshot_draws = []
    for sample_id in sorted(rows):
        row = rows[sample_id]
        original_samples = []
        fewshot_samples = []
        for k in range(args.k_samples):
            initial_seed, reverse_seed = stable_rollout_seeds(
                args.seed, "development", sample_id, k
            )
            original_normalized = sample_contact_deterministic(
                original_model,
                original_diffusion,
                row,
                initial_seed,
                reverse_seed,
                args.device,
            )
            fewshot_normalized = sample_contact_deterministic(
                fewshot_model,
                fewshot_diffusion,
                row,
                initial_seed,
                reverse_seed,
                args.device,
            )
            original_samples.append(
                denormalize_contact(original_normalized, mean, std)
            )
            fewshot_samples.append(
                denormalize_contact(fewshot_normalized, mean, std)
            )
        original = np.mean(original_samples, axis=0).astype(np.float32)
        fewshot = np.mean(fewshot_samples, axis=0).astype(np.float32)
        gt = row["gt"]
        zero = np.zeros_like(gt)
        original_scores = {
            channel: instance_scores(original, row["instance_ids"], channel)
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        fewshot_scores = {
            channel: instance_scores(fewshot, row["instance_ids"], channel)
            for channel in ("pelvis", "any_joint", "right_wrist")
        }
        target = str(row["target"])
        candidate_names = ("chair", "bed", "whiteboard")
        dominance_margin = {
            channel: float(
                fewshot_scores[channel][target]
                - max(
                    fewshot_scores[channel][name]
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
            fewshot_scores["any_joint"]["chair"]
            >= fewshot_scores["any_joint"]["bed"] + sit_chair_bed_margin
            and fewshot_scores["pelvis"]["chair"]
            >= fewshot_scores["pelvis"]["bed"] + sit_chair_bed_margin
        )
        sit_bed_candidate_visible = (
            sit_bed_any_min
            <= fewshot_scores["any_joint"]["bed"]
            <= sit_bed_any_max
            and sit_bed_pelvis_min
            <= fewshot_scores["pelvis"]["bed"]
            <= sit_bed_pelvis_max
        )
        target_mask = row["instance_ids"] == row["target_instance_id"]
        per_draw_metrics = {"original": [], "fewshot": [], "zero_contact": []}
        per_draw_dominance = []
        per_draw_dominance_margin = []
        for original_draw, fewshot_draw in zip(original_samples, fewshot_samples):
            per_draw_metrics["original"].append(
                binary_metrics(original_draw[target_mask], gt[target_mask])
            )
            per_draw_metrics["fewshot"].append(
                binary_metrics(fewshot_draw[target_mask], gt[target_mask])
            )
            per_draw_metrics["zero_contact"].append(
                binary_metrics(zero[target_mask], gt[target_mask])
            )
            draw_scores = {
                channel: instance_scores(
                    fewshot_draw, row["instance_ids"], channel
                )
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
            per_draw_dominance_margin.append(margins)
            per_draw_dominance.append(
                {
                    channel: margins[channel] >= args.dominance_margin
                    for channel in ("pelvis", "any_joint", "right_wrist")
                }
            )
            per_draw_dominance[-1]["sit_chair_primary"] = (
                draw_scores["any_joint"]["chair"]
                >= draw_scores["any_joint"]["bed"] + sit_chair_bed_margin
                and draw_scores["pelvis"]["chair"]
                >= draw_scores["pelvis"]["bed"] + sit_chair_bed_margin
            )
            per_draw_dominance[-1]["sit_bed_candidate_visible"] = (
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
            "metrics": {
                "original": binary_metrics(original, gt),
                "fewshot": binary_metrics(fewshot, gt),
                "zero_contact": binary_metrics(zero, gt),
            },
            "target_region_metrics": {
                "original": binary_metrics(
                    original[row["instance_ids"] == row["target_instance_id"]],
                    gt[row["instance_ids"] == row["target_instance_id"]],
                ),
                "fewshot": binary_metrics(
                    fewshot[row["instance_ids"] == row["target_instance_id"]],
                    gt[row["instance_ids"] == row["target_instance_id"]],
                ),
                "zero_contact": binary_metrics(
                    zero[row["instance_ids"] == row["target_instance_id"]],
                    gt[row["instance_ids"] == row["target_instance_id"]],
                ),
            },
            "per_draw_target_region_metrics": per_draw_metrics,
            "target_top10": {
                "original_pelvis": original_scores["pelvis"][target],
                "fewshot_pelvis": fewshot_scores["pelvis"][target],
                "original_any_joint": original_scores["any_joint"][target],
                "fewshot_any_joint": fewshot_scores["any_joint"][target],
                "original_right_wrist": original_scores["right_wrist"][target],
                "fewshot_right_wrist": fewshot_scores["right_wrist"][target],
            },
            "fewshot_candidate_scores": {
                channel: {name: fewshot_scores[channel][name] for name in candidate_names}
                for channel in ("pelvis", "any_joint", "right_wrist")
            },
            "target_dominates_candidates": dominance,
            "target_dominance_margin": dominance_margin,
            "per_draw_target_dominance": per_draw_dominance,
            "per_draw_target_dominance_margin": per_draw_dominance_margin,
            "sit_chair_primary": sit_chair_primary,
            "sit_bed_candidate_visible": sit_bed_candidate_visible,
        }
        per_sample.append(record)
        saved_sample_ids.append(sample_id)
        saved_gt.append(gt)
        saved_original.append(original)
        saved_fewshot.append(fewshot)
        saved_original_draws.append(np.stack(original_samples))
        saved_fewshot_draws.append(np.stack(fewshot_samples))
        plot_sample(
            output_dir / "visualizations" / f"{sample_id}.png",
            row,
            original,
            fewshot,
            gt,
        )
        print(
            f"[EVAL] {sample_id} target={target} "
            f"MAE={record['metrics']['original']['mae']:.6f}->"
            f"{record['metrics']['fewshot']['mae']:.6f} "
            f"dominance={dominance}"
        )

    aggregate = {}
    aggregate_target_region = {}
    per_draw_aggregate_target_region = {}
    for target in ("chair", "bed", "whiteboard", "novel"):
        selected = [
            row for row in per_sample
            if (row["target"] == target or (target == "novel" and row["target"] in {"bed", "whiteboard"}))
        ]
        aggregate[target] = {}
        aggregate_target_region[target] = {}
        per_draw_aggregate_target_region[target] = {}
        for model_name in ("original", "fewshot", "zero_contact"):
            aggregate[target][model_name] = {
                metric: float(np.mean([row["metrics"][model_name][metric] for row in selected]))
                for metric in ("mae", "foreground_mae", "f1_at_0_7", "correlation")
            }
            aggregate_target_region[target][model_name] = {
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
            per_draw_aggregate_target_region[target][model_name] = {
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

    chair_original_mae = per_draw_aggregate_target_region["chair"]["original"][
        "mae"
    ]
    chair_degradation = (
        per_draw_aggregate_target_region["chair"]["fewshot"]["mae"]
        - chair_original_mae
    ) / max(chair_original_mae, 1e-12)
    chair_bed_rows = [row for row in per_sample if row["target"] in {"chair", "bed"}]
    whiteboard_rows = [row for row in per_sample if row["target"] == "whiteboard"]
    per_draw_dominance_rate = {}
    for target in ("chair", "bed", "whiteboard"):
        selected = [row for row in per_sample if row["target"] == target]
        per_draw_dominance_rate[target] = {
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
    sit_per_draw_chair_primary_rate = float(
        np.mean(
            [
                draw["sit_chair_primary"]
                for row in per_sample
                if row["target"] == "chair"
                for draw in row["per_draw_target_dominance"]
            ]
        )
    )
    sit_per_draw_bed_candidate_rate = float(
        np.mean(
            [
                draw["sit_bed_candidate_visible"]
                for row in per_sample
                if row["target"] == "chair"
                for draw in row["per_draw_target_dominance"]
            ]
        )
    )
    checks = {
        **provenance_checks,
        "heldout_count_is_12": len(per_sample) == 12,
        "heldout_target_counts_exact": counts == Counter({"chair": 5, "whiteboard": 6, "bed": 1}),
        "bed_target_region_mae_improved_vs_original": (
            per_draw_aggregate_target_region["bed"]["fewshot"]["mae"]
            < per_draw_aggregate_target_region["bed"]["original"]["mae"]
        ),
        "whiteboard_target_region_mae_improved_vs_original": (
            per_draw_aggregate_target_region["whiteboard"]["fewshot"]["mae"]
            < per_draw_aggregate_target_region["whiteboard"]["original"]["mae"]
        ),
        "bed_target_region_mae_beats_zero_contact": (
            per_draw_aggregate_target_region["bed"]["fewshot"]["mae"]
            < per_draw_aggregate_target_region["bed"]["zero_contact"]["mae"]
        ),
        "whiteboard_target_region_mae_beats_zero_contact": (
            per_draw_aggregate_target_region["whiteboard"]["fewshot"]["mae"]
            < per_draw_aggregate_target_region["whiteboard"]["zero_contact"]["mae"]
        ),
        "bed_target_region_f1_improved_and_active": (
            per_draw_aggregate_target_region["bed"]["fewshot"]["f1_at_0_7"]
            > per_draw_aggregate_target_region["bed"]["original"]["f1_at_0_7"]
            and per_draw_aggregate_target_region["bed"]["fewshot"]["f1_at_0_7"]
            >= args.novel_min_f1
        ),
        "whiteboard_target_region_f1_improved_and_active": (
            per_draw_aggregate_target_region["whiteboard"]["fewshot"][
                "f1_at_0_7"
            ]
            > per_draw_aggregate_target_region["whiteboard"]["original"][
                "f1_at_0_7"
            ]
            and per_draw_aggregate_target_region["whiteboard"]["fewshot"][
                "f1_at_0_7"
            ]
            >= args.novel_min_f1
        ),
        "chair_replay_target_region_mae_retained": chair_degradation <= args.chair_mae_degradation_limit,
        "chair_bed_dominate_pelvis": all(
            row["target_dominates_candidates"]["pelvis"]
            for row in chair_bed_rows
        ),
        "all_targets_dominate_any_joint": all(
            row["target_dominates_candidates"]["any_joint"]
            for row in per_sample
        ),
        "whiteboard_any_joint_dominance": all(
            row["target_dominates_candidates"]["any_joint"]
            for row in whiteboard_rows
        ),
        "whiteboard_right_wrist_dominance": all(
            row["target_dominates_candidates"]["right_wrist"]
            for row in whiteboard_rows
        ),
        "sit_chair_remains_primary": all(
            row["sit_chair_primary"]
            for row in per_sample
            if row["target"] == "chair"
        ),
        "sit_bed_candidate_visible_and_bounded": all(
            row["sit_bed_candidate_visible"]
            for row in per_sample
            if row["target"] == "chair"
        ),
        "per_draw_target_dominance_rates_pass": (
            per_draw_dominance_rate["chair"]["pelvis"] >= 0.80
            and per_draw_dominance_rate["chair"]["any_joint"] >= 0.80
            and per_draw_dominance_rate["bed"]["pelvis"] >= 0.80
            and per_draw_dominance_rate["bed"]["any_joint"] >= 0.80
            and per_draw_dominance_rate["whiteboard"]["any_joint"] >= 0.80
            and per_draw_dominance_rate["whiteboard"]["right_wrist"] >= 0.80
        ),
        "sit_per_draw_chair_primary_rate_pass": (
            sit_per_draw_chair_primary_rate >= 0.80
        ),
        "sit_per_draw_bed_candidate_rate_pass": (
            sit_per_draw_bed_candidate_rate >= 0.80
        ),
        "v5_prompts_exact_and_object_agnostic": all(
            row["text"] == prompt_for_target(row["target"])
            and row["prompt_policy"] == PROMPT_POLICY_ID
            for row in rows.values()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    sample_fingerprints = fingerprint_rows(rows, sorted(rows))
    dataset_snapshot_payload = {
        "schema": "history_affordance_v1_rollout_dataset_snapshot_v1",
        "partition": "development",
        "split_sha256": sha256_file(split_file),
        "stats_file_sha256": sha256_file(stats_file),
        "sample_fingerprints": sample_fingerprints,
    }
    summary = {
        "schema": (
            "history_affordance_v1_fewshot_cdm_"
            "development_eval_v5r4_sealed_v2"
        ),
        "status": status,
        "scientific_scope": (
            "development raw-source evaluation; these sources were inspected "
            "during v1/v2 diagnosis and are not a final paper test"
        ),
        "final_paper_test_requires_new_unseen_sources": True,
        "may_proceed_to_moe_iiw": status == "PASS",
        "k_samples": args.k_samples,
        "diffusion_steps": args.diffusion_steps,
        "quality_thresholds": {
            "schema": "fewshot_cdm_rollout_quality_thresholds_v1",
            "active_threshold": 0.7,
            "instance_top_fraction": 0.1,
            "chair_mae_degradation_limit": args.chair_mae_degradation_limit,
            "novel_min_f1": args.novel_min_f1,
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
        "pairing_protocol": {
            "seed_derivation": "sha256(base|partition|sample_id|draw)",
            "initial_xT_and_all_reverse_step_noise_paired": True,
            "caller_rng_state_restored": True,
        },
        "sampling_provenance": {
            "base_seed": args.seed,
            "partition": "development",
            "seed_derivation": "sha256(base|partition|sample_id|draw)",
            "seed_table_storage": "prediction_npz_int64_matrices",
        },
        "train_rollout_audit": {
            "file": str(train_rollout_audit_file),
            "sha256": sha256_file(train_rollout_audit_file),
            "schema": train_rollout_audit["schema"],
            "status": train_rollout_audit["status"],
            "k_samples": train_rollout_audit["k_samples"],
            "test_sample_tensors_read": train_rollout_audit[
                "test_sample_tensors_read"
            ],
        },
        "dataset_snapshot": {
            **dataset_snapshot_payload,
            "sha256": sha256_json(dataset_snapshot_payload),
        },
        "checks": checks,
        "chair_mae_relative_degradation": chair_degradation,
        "dominance_margin_required": args.dominance_margin,
        "per_draw_dominance_rate": per_draw_dominance_rate,
        "sit_per_draw_contract": {
            "chair_primary_rate": sit_per_draw_chair_primary_rate,
            "bed_candidate_visible_rate": sit_per_draw_bed_candidate_rate,
            "minimum_rate": 0.80,
        },
        "aggregate": aggregate,
        "aggregate_target_region": aggregate_target_region,
        "per_draw_aggregate_target_region": per_draw_aggregate_target_region,
        "per_sample": per_sample,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    predictions_file = output_dir / "predictions.npz"
    saved_seed_pairs = np.asarray(
        [
            [
                stable_rollout_seeds(args.seed, "development", sample_id, draw)
                for draw in range(args.k_samples)
            ]
            for sample_id in saved_sample_ids
        ],
        dtype=np.int64,
    )
    atomic_savez_compressed(
        predictions_file,
        sample_ids=np.asarray(saved_sample_ids),
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
    atomic_write_json(output_dir / "summary.json", summary)
    print(f"[{status}] held-out Base ADM quality gate")
    for key, value in checks.items():
        print(f"[{'PASS' if value else 'FAIL'}] {key}")
    print(f"[OK] saved: {output_dir / 'summary.json'}")
    print(f"[OK] saved: {predictions_file}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

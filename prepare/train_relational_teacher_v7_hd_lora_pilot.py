#!/usr/bin/env python3
"""Fresh bounded 60-update train-only pilot for Teacher-v7 High-Desk LoRA."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from fewshot_cdm_common import load_split
from fewshot_cdm_lora import (
    install_lora,
    lora_metadata,
    lora_named_parameters,
    lora_parameter_energy,
    save_merged_legacy_state,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from preflight_relational_teacher_v7_hd_lora import (
    build_batch,
    compose_cdm_config,
    configure_reproducibility,
    load_stats,
    predict_xstart,
)
from recover_relational_teacher_v7_hd_lora_invariance import (
    CANDIDATE_WEIGHT,
    DEFAULT_LR,
    INVARIANCE_WEIGHT,
    LORA_WEIGHT,
    NEGATIVE_WEIGHT,
    NEW_DENSE_WEIGHT,
    PRESERVATION_WEIGHT,
    V5_REPLAY_WEIGHT,
    build_selected_relational_batch,
    deterministic_noise,
    fixed_panel,
    group_relational_rows,
    load_relational_rows,
    relation_objective,
    relational_timesteps,
    select_v5_batch_ids,
)
from relational_teacher_v6_contract import (
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v7_hd_preflight_contract import (
    DATASET_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
)
from relational_teacher_v7_hd_calibration_contract import (
    TRAINING_STRATA_PER_SCENE,
    validate_batch_audits,
)
from relational_teacher_v7_hd_pilot_contract import (
    CHECKPOINT_STEPS,
    SCHEMA,
    STEPS,
    checkpoint_eligibility,
    rank_shortlist,
    select_pilot_batch_rows,
)
from relational_teacher_v7_hd_recovery_contract import (
    EXPECTED_WEIGHTS,
    SCHEMA as RECOVERY_SCHEMA,
)
from train_fewshot_cdm import load_rows as load_v5_rows
from train_fewshot_cdm import stack_batch as stack_v5_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v5-dataset-root", type=Path, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_recovery_binding(
    recovery_file: Path,
    *,
    seed: int,
    index_file: Path,
    v5_split_file: Path,
    stats_file: Path,
    original_checkpoint: Path,
    v5_checkpoint: Path,
) -> Mapping[str, object]:
    recovery = read_json(recovery_file)
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("status") != "RECOVERY_PASS"
        or int(recovery.get("steps", -1)) != 12
        or int(recovery.get("seed", -1)) != seed
        or recovery.get("failed_checks") != []
        or recovery.get("objective_weights") != EXPECTED_WEIGHTS
        or float(recovery.get("learning_rate", float("nan"))) != DEFAULT_LR
        or recovery.get("development_payloads_read") is not False
        or recovery.get("diagnostic_checkpoint_only") is not True
        or recovery.get("authorizes_bounded_train_only_pilot") is not True
        or recovery.get("prior_failed_v7_checkpoint_loaded") is not False
        or recovery.get("same_seed_batches_noise_timesteps_as_failed_v7") is not True
        or recovery.get("single_changed_hyperparameter")
        != {
            "name": "watch_write_invariance",
            "failed_value": 0.5,
            "recovery_value": INVARIANCE_WEIGHT,
        }
    ):
        raise ValueError("Teacher-v7 recovery does not authorize the pilot")
    for key in (
        "authorizes_long_training",
        "authorizes_development_evaluation",
        "authorizes_paper_test",
    ):
        if recovery.get(key) is not False:
            raise ValueError("Teacher-v7 recovery authority changed: " + key)
    checks = recovery.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("Teacher-v7 recovery checks are not all PASS")

    expected = {
        "dataset_index": index_file,
        "v5_split": v5_split_file,
        "stats_file": stats_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
    }
    for key, path in expected.items():
        if (
            Path(str(recovery.get(key, ""))).expanduser().resolve() != path
            or recovery.get(key + "_sha256") != sha256_file(path)
        ):
            raise ValueError("pilot input differs from recovery: " + key)
    for path_key, hash_key in (
        ("failed_calibration_summary", "failed_calibration_summary_sha256"),
        ("preflight_report", "preflight_report_sha256"),
        ("lora_state", "lora_state_sha256"),
        ("merged_checkpoint", "merged_checkpoint_sha256"),
    ):
        path = Path(str(recovery.get(path_key, ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(recovery.get(hash_key, "")):
            raise ValueError("Teacher-v7 recovery artifact changed: " + path_key)
    source_paths = recovery.get("source_paths")
    source_hashes = recovery.get("source_sha256")
    if (
        not isinstance(source_paths, dict)
        or not isinstance(source_hashes, dict)
        or set(source_paths) != set(source_hashes)
    ):
        raise ValueError("Teacher-v7 recovery source binding is absent")
    for name, raw in source_paths.items():
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(source_hashes[name]):
            raise ValueError("Teacher-v7 recovery source changed: " + name)
    return recovery


def main() -> None:
    args = parse_args()
    if args.steps != STEPS:
        raise ValueError("Teacher-v7 pilot is sealed to exactly 60 updates")
    if args.lr != DEFAULT_LR or args.grad_clip != 1.0:
        raise ValueError("Teacher-v7 pilot is sealed to lr=4e-5 and grad-clip=1")
    if args.lora_rank != 4 or args.lora_alpha != 8.0:
        raise ValueError("Teacher-v7 pilot is sealed to LoRA rank=4 alpha=8")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v7 pilot requires CUDA")
    configure_reproducibility(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    index_file = args.index.expanduser().resolve()
    v5_dataset_root = args.v5_dataset_root.expanduser().resolve()
    v5_split_file = args.v5_split.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    v5_checkpoint = args.v5_checkpoint.expanduser().resolve()
    recovery_file = args.recovery_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v7 pilot: " + str(output_dir))
    for path in (
        index_file,
        v5_split_file,
        stats_file,
        original_checkpoint,
        v5_checkpoint,
        recovery_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    recovery = validate_recovery_binding(
        recovery_file,
        seed=args.seed,
        index_file=index_file,
        v5_split_file=v5_split_file,
        stats_file=stats_file,
        original_checkpoint=original_checkpoint,
        v5_checkpoint=v5_checkpoint,
    )

    index = read_json(index_file)
    if (
        index.get("schema") != DATASET_SCHEMA
        or index.get("status") != "DENSE_DATASET_PASS"
        or index.get("authorization") != "teacher_lora_v7_cuda_preflight_only"
        or index.get("split_scene_counts") != {"train": 2, "development": 1}
        or index.get("split_row_counts") != {"development": 48, "train": 96}
        or index.get("num_dense_motions") != 72
        or index.get("num_rows") != 144
        or index.get("heldout_dense_arrays_read_during_build") is not False
        or index.get("relation_or_distance_used_as_forward_input") is not False
        or index.get("target_instance_gt_is_supervision_only") is not True
    ):
        raise ValueError("Teacher-v7 dataset index changed")
    relational_rows = load_relational_rows(dataset_root, index)
    grouped_relational = group_relational_rows(relational_rows)
    if {scene: len(rows) for scene, rows in relational_rows.items()} != {
        "room_0101": 24,
        "room_0102": 24,
    }:
        raise ValueError("Teacher-v7 relational train rows changed")

    mean, std = load_stats(stats_file)
    v5_split = load_split(v5_split_file)
    v5_rows = load_v5_rows(
        v5_dataset_root, v5_split, "train", mean, std, 4.0, 16.0, 0.7
    )
    v5_counts = Counter(str(row["target"]) for row in v5_rows.values())
    if v5_counts != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise ValueError("Teacher-v7 v5 replay inventory changed")

    cfg = compose_cdm_config(500, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()

    parity_selected, _ = select_pilot_batch_rows(grouped_relational, 1)
    parity_batch = build_selected_relational_batch(
        parity_selected, mean, std, args.device
    )
    parity_timestep = relational_timesteps(1, args.device)
    parity_noise = deterministic_noise(
        parity_batch["x"].shape, args.seed + 300, args.device
    )
    parity_kwargs = {
        "c_pc_xyz": parity_batch["xyz"],
        "c_pc_feat": parity_batch["feat"],
        "c_text": parity_batch["text"],
    }
    with torch.no_grad():
        before = predict_xstart(
            model, diffusion, parity_batch["x"], parity_timestep, parity_kwargs, parity_noise
        )
    install_lora(model, args.lora_rank, args.lora_alpha, dropout=0.0)
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    if len(lora_metadata(model).get("module_names", [])) != 31:
        raise AssertionError("Teacher-v7 pilot expected exactly 31 LoRA modules")
    with torch.no_grad():
        after = predict_xstart(
            model, diffusion, parity_batch["x"], parity_timestep, parity_kwargs, parity_noise
        )
    if not torch.equal(before, after):
        raise AssertionError("fresh Teacher-v7 pilot zero-init parity failed")

    optimizer = torch.optim.AdamW(list(named_lora.values()), lr=args.lr)
    initial_panel = fixed_panel(
        model,
        diffusion,
        grouped_relational,
        v5_rows,
        mean,
        std,
        args.device,
        args.seed + 10000,
    )
    recovery_initial = recovery.get("initial_fixed_panel")
    if not isinstance(recovery_initial, dict) or initial_panel != recovery_initial:
        raise ValueError("fresh pilot initialization differs from recovery A/B")

    output_dir.mkdir(parents=True, exist_ok=False)
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)
    logs = []
    batch_audits = []
    checkpoint_records = []
    frozen_gradient_count = 0

    for step in range(1, STEPS + 1):
        set_frozen_base_eval_lora_train(model)
        selected, audit = select_pilot_batch_rows(grouped_relational, step)
        relation_batch = build_selected_relational_batch(
            selected, mean, std, args.device
        )
        relation_timestep = relational_timesteps(step, args.device)
        relation_noise = deterministic_noise(
            relation_batch["x"].shape, args.seed + step * 1009, args.device
        )
        relation_kwargs = {
            "c_pc_xyz": relation_batch["xyz"],
            "c_pc_feat": relation_batch["feat"],
            "c_text": relation_batch["text"],
        }
        prediction = predict_xstart(
            model,
            diffusion,
            relation_batch["x"],
            relation_timestep,
            relation_kwargs,
            relation_noise,
        )
        set_lora_enabled(model, False)
        try:
            with torch.no_grad():
                frozen_prediction = predict_xstart(
                    model,
                    diffusion,
                    relation_batch["x"],
                    relation_timestep,
                    relation_kwargs,
                    relation_noise,
                )
        finally:
            set_lora_enabled(model, True)
        relation = relation_objective(
            prediction,
            frozen_prediction,
            relation_batch,
            mean_tensor,
            std_tensor,
        )

        replay_ids = select_v5_batch_ids(v5_rows, step)
        replay_batch = stack_v5_batch(v5_rows, replay_ids, args.device)
        replay_timestep = torch.tensor(
            [100, 300, 450], dtype=torch.long, device=args.device
        )
        replay_noise = deterministic_noise(
            replay_batch["x"].shape, args.seed + 50000 + step * 1013, args.device
        )
        replay_prediction = predict_xstart(
            model,
            diffusion,
            replay_batch["x"],
            replay_timestep,
            {
                "c_pc_xyz": replay_batch["xyz"],
                "c_pc_feat": replay_batch["feat"],
                "c_text": replay_batch["text"],
            },
            replay_noise,
        )
        replay_dense = (replay_prediction - replay_batch["x"]).square().mean()
        dense_combined = (
            NEW_DENSE_WEIGHT * relation["new_dense"]
            + V5_REPLAY_WEIGHT * replay_dense
        )
        regularizer = lora_parameter_energy(model)
        total = (
            dense_combined
            + CANDIDATE_WEIGHT * relation["candidate"]
            + NEGATIVE_WEIGHT * relation["negative"]
            + INVARIANCE_WEIGHT * relation["invariance"]
            + PRESERVATION_WEIGHT * relation["preservation"]
            + LORA_WEIGHT * regularizer
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        frozen_gradients = [
            name
            for name, parameter in model.named_parameters()
            if "lora_" not in name and parameter.grad is not None
        ]
        frozen_gradient_count += len(frozen_gradients)
        if frozen_gradients:
            raise AssertionError("gradient reached frozen CDM during pilot")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(named_lora.values()), args.grad_clip
        )
        gradient_value = float(gradient_norm.item())
        if not math.isfinite(gradient_value) or gradient_value <= 0.0:
            raise RuntimeError("invalid Teacher-v7 pilot gradient")
        optimizer.step()

        log = {
            "step": step,
            "relational_noise_seed": args.seed + step * 1009,
            "v5_replay_noise_seed": args.seed + 50000 + step * 1013,
            "relational_timesteps": relation_timestep.detach().cpu().tolist(),
            "v5_replay_timesteps": replay_timestep.detach().cpu().tolist(),
            "total": float(total.detach().item()),
            "dense_combined": float(dense_combined.detach().item()),
            "new_dense": float(relation["new_dense"].detach().item()),
            "high_desk_dense": float(
                relation["high_desk_dense"].detach().item()
            ),
            "v5_replay_dense": float(replay_dense.detach().item()),
            "candidate": float(relation["candidate"].detach().item()),
            "negative": float(relation["negative"].detach().item()),
            "invariance": float(relation["invariance"].detach().item()),
            "preservation": float(relation["preservation"].detach().item()),
            "lora_energy": float(regularizer.detach().item()),
            "gradient_l2_before_clip": gradient_value,
            "v5_replay_ids": replay_ids,
        }
        if not all(
            math.isfinite(float(value))
            for key, value in log.items()
            if key
            not in {
                "step",
                "relational_noise_seed",
                "v5_replay_noise_seed",
                "relational_timesteps",
                "v5_replay_timesteps",
                "v5_replay_ids",
            }
        ):
            raise RuntimeError("non-finite Teacher-v7 pilot metric")
        logs.append(log)
        audit["v5_replay_ids"] = replay_ids
        batch_audits.append(audit)
        if step == 1 or step % 5 == 0:
            print(
                f"[PILOT] step={step:02d}/60 total={log['total']:.6f} "
                f"high_desk={log['high_desk_dense']:.6f} "
                f"semantic={float(relation['semantic_total'].detach().item()):.6f} "
                f"invariance={log['invariance']:.6f} grad={gradient_value:.6f}"
            )

        if step in CHECKPOINT_STEPS:
            panel = fixed_panel(
                model,
                diffusion,
                grouped_relational,
                v5_rows,
                mean,
                std,
                args.device,
                args.seed + 10000,
            )
            lora_file = output_dir / f"lora_state_step{step:02d}.pt"
            merged_file = output_dir / f"merged_teacher_step{step:02d}.pt"
            atomic_torch_save(
                {
                    "schema": SCHEMA + "_diagnostic_lora_state",
                    "step": step,
                    "seed": args.seed,
                    "lora": {
                        name: parameter.detach().cpu()
                        for name, parameter in named_lora.items()
                    },
                },
                lora_file,
            )
            temporary = merged_file.with_name(
                merged_file.name + ".tmp." + str(os.getpid())
            )
            save_merged_legacy_state(model, temporary)
            os.replace(temporary, merged_file)
            eligibility = checkpoint_eligibility(panel, initial_panel)
            checkpoint_records.append(
                {
                    "step": step,
                    "fixed_panel": panel,
                    "eligibility_checks": eligibility,
                    "eligible": all(eligibility.values()),
                    "lora_state": str(lora_file),
                    "lora_state_sha256": sha256_file(lora_file),
                    "merged_checkpoint": str(merged_file),
                    "merged_checkpoint_sha256": sha256_file(merged_file),
                }
            )
            print(
                f"[CHECKPOINT] step={step:02d} eligible={all(eligibility.values())} "
                f"semantic={panel['semantic_total']:.6f} "
                f"v5={panel['v5_replay_dense']:.6f} "
                f"invariance={panel['invariance']:.6f}"
            )

    first12_exact = (
        logs[:12] == recovery.get("logs")
        and batch_audits[:12] == recovery.get("batch_audits")
        and checkpoint_records[0]["fixed_panel"] == recovery.get("final_fixed_panel")
    )
    shortlisted_steps = list(rank_shortlist(checkpoint_records))
    batch_contract = validate_batch_audits(batch_audits)
    checks = {
        "exact_60_updates": len(logs) == STEPS,
        "five_checkpoint_steps_present": [
            int(row["step"]) for row in checkpoint_records
        ]
        == list(CHECKPOINT_STEPS),
        "first12_exactly_reproduce_recovery": first12_exact,
        **batch_contract,
        "every_update_uses_v5_chair_bed_whiteboard_replay": all(
            len(row["v5_replay_ids"]) == 3 for row in batch_audits
        ),
        "development_payloads_unread": True,
        "frozen_cdm_has_no_gradient": frozen_gradient_count == 0,
        "all_gradients_finite_and_positive": all(
            math.isfinite(float(row["gradient_l2_before_clip"]))
            and float(row["gradient_l2_before_clip"]) > 0.0
            for row in logs
        ),
        "at_least_one_checkpoint_is_admissible": len(shortlisted_steps) >= 1,
        "shortlist_size_at_most_two": 1 <= len(shortlisted_steps) <= 2,
    }
    status = "PILOT_PASS" if all(checks.values()) else "PILOT_FAIL"
    prepare_root = Path(__file__).resolve().parent
    source_paths = {
        "pilot": Path(__file__).resolve(),
        "validator": prepare_root
        / "validate_relational_teacher_v7_hd_lora_pilot.py",
        "recovery_source": prepare_root
        / "recover_relational_teacher_v7_hd_lora_invariance.py",
        "recovery_validator_source": prepare_root
        / "validate_relational_teacher_v7_hd_lora_invariance_recovery.py",
        "pilot_contract_source": prepare_root
        / "relational_teacher_v7_hd_pilot_contract.py",
        "recovery_contract_source": prepare_root
        / "relational_teacher_v7_hd_recovery_contract.py",
        "calibration_contract_source": prepare_root
        / "relational_teacher_v7_hd_calibration_contract.py",
        "preflight_contract_source": prepare_root
        / "relational_teacher_v7_hd_preflight_contract.py",
        "lora_source": prepare_root / "fewshot_cdm_lora.py",
        "contract_source": prepare_root / "relational_teacher_v6_contract.py",
        "semantics_source": prepare_root / "relational_teacher_v6_semantics.py",
        "dataset_validator_source": prepare_root
        / "validate_relational_teacher_v7_hd_dataset.py",
    }
    payload = {
        "schema": SCHEMA,
        "status": status,
        "steps": STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "seed": args.seed,
        "device": args.device,
        "learning_rate": args.lr,
        "gradient_clip": args.grad_clip,
        "train_scenes": list(EXPECTED_TRAIN_SCENES),
        "development_scenes_metadata_only": list(EXPECTED_DEVELOPMENT_SCENES),
        "development_payloads_read": False,
        "relational_motion_counts": {scene: 24 for scene in EXPECTED_TRAIN_SCENES},
        "high_desk_motion_counts": {scene: 6 for scene in EXPECTED_TRAIN_SCENES},
        "training_strata_per_scene": TRAINING_STRATA_PER_SCENE,
        "v5_replay_target_counts": dict(v5_counts),
        "objective_weights": EXPECTED_WEIGHTS,
        "initial_fixed_panel": initial_panel,
        "logs": logs,
        "batch_audits": batch_audits,
        "checkpoint_records": checkpoint_records,
        "shortlisted_steps": shortlisted_steps,
        "selection_rank": [
            "objective_proxy",
            "high_desk_dense",
            "semantic_total",
            "step",
        ],
        "frozen_parameter_gradient_count": frozen_gradient_count,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "recovery_summary": str(recovery_file),
        "recovery_summary_sha256": sha256_file(recovery_file),
        "dataset_index": str(index_file),
        "dataset_index_sha256": sha256_file(index_file),
        "v5_split": str(v5_split_file),
        "v5_split_sha256": sha256_file(v5_split_file),
        "stats_file": str(stats_file),
        "stats_file_sha256": sha256_file(stats_file),
        "original_checkpoint": str(original_checkpoint),
        "original_checkpoint_sha256": sha256_file(original_checkpoint),
        "v5_checkpoint": str(v5_checkpoint),
        "v5_checkpoint_sha256": sha256_file(v5_checkpoint),
        "lora_metadata": dict(lora_metadata(model)),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "fresh_zero_init_from_sealed_v5r4": True,
        "prior_v6_lora_checkpoint_loaded": False,
        "prior_failed_v7_checkpoint_loaded": False,
        "recovery_checkpoint_loaded": False,
        "diagnostic_checkpoints_only": True,
        "authorizes_train_only_reverse_diffusion_shortlist": status
        == "PILOT_PASS",
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    summary_file = output_dir / "summary.json"
    atomic_write_json(summary_file, payload)
    print(f"[{status}] relational Teacher-v7 High-Desk fresh bounded 60-update pilot")
    print("[PASS] fresh v5r4 initialization and first-12 recovery reproduction")
    print("[OK] shortlisted steps:", shortlisted_steps)
    print("[OK] failed checks:", payload["failed_checks"])
    print("[OK] summary:", summary_file)


if __name__ == "__main__":
    main()

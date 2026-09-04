#!/usr/bin/env python3
"""Fresh CUDA response grid for the Teacher-v9.1 support/ranking objective."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from fewshot_cdm_common import load_split  # noqa: E402
from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    lora_parameter_energy,
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    atomic_savez,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v9_lora_objective import preservation_loss  # noqa: E402
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
    OBJECTIVE_WEIGHTS,
    canonical_sha256,
    load_train_scene_bundle,
    validate_top_index,
)
from relational_teacher_v9_lora_runtime import (  # noqa: E402
    compose_cdm_config,
    configure_reproducibility,
    deterministic_noise,
    load_stats,
    predict_xstart,
    tensor_sha256,
)
from relational_teacher_v9_overfit_smoke_contract import sanitize_metric_nonfinite  # noqa: E402
from relational_teacher_v91_active_support_objective import corrected_objective  # noqa: E402
from relational_teacher_v91_loss_response_contract import (  # noqa: E402
    CANDIDATES,
    FAILED_SCHEMA,
    GRAD_CLIP,
    LEARNING_RATE,
    SCHEMA,
    SEED,
    STEPS_PER_CANDIDATE,
    TRAIN_SCENE,
    rank_candidates,
    response_checks,
)
from run_relational_teacher_v9_one_scene_overfit_smoke import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
    _metrics,
    _physical,
    _prompt_invariance,
    _require_same_path,
    _restore_lora_state,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-smoke-summary", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v5-dataset-root", type=Path, required=True)
    parser.add_argument("--v5-split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-evidence-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_tree(nested, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(label + " contains a non-finite value")


def _validate_failure(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != FAILED_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("steps") != 120
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("authorizes_fresh_response3") is not False
        or value.get("failed_checks")
        != [
            "one_step_three_object_panel_passes",
            "reverse_diffusion_three_object_panel_passes",
        ]
    ):
        raise ValueError("Teacher-v9 failed smoke binding changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("failed smoke paths are absent")
    for name, raw in paths.items():
        source = Path(str(raw)).resolve()
        if not source.is_file() or sha256_file(source) != hashes.get(name):
            raise ValueError("failed-smoke file changed: " + str(name))
    one_step = value["monitor_panels"][-1]
    rollout = value["rollout"]["panel"]
    required_one_step = {
        "prompt_invariance_retained",
        "v5_replay_dense_retained",
        f"{TRAIN_SCENE}|sit_watch_v1|absolute_presence",
        f"{TRAIN_SCENE}|sit_write_v1|absolute_presence",
    }
    if not required_one_step.issubset(one_step["failed_checks"]):
        raise ValueError("failed smoke one-step diagnosis changed")
    for row in rollout["absolute_presence"].values():
        if (
            row.get("every_verified_instance_has_soft_recall") is not False
            or row.get("every_verified_instance_has_topk_overlap") is not False
            or row.get("every_verified_instance_has_bounded_active_support_mae")
            is not False
            or row.get("every_verified_instance_has_bounded_hotspot_centroid")
            is not True
            or row.get("explicit_negative_mean_bounded") is not True
        ):
            raise ValueError("failed reverse-diffusion diagnosis changed")
    return value


def _objective_values(
    objective: Mapping[str, torch.Tensor],
    physical: np.ndarray,
    bundle: Mapping[str, object],
) -> Dict[str, object]:
    metrics = [_metrics(bundle, physical[index]) for index in range(2)]
    verified = np.asarray(bundle["verified_positive_mask"], bool)
    return {
        "active_support_macro": float(objective["active_support_macro"].item()),
        "within_object_ranking": float(objective["within_object_ranking"].item()),
        "per_instance_active_support": [
            float(value)
            for value in objective["per_instance_active_support"].mean(dim=0).tolist()
        ],
        "per_instance_ranking": [
            float(value)
            for value in objective["per_instance_ranking"].mean(dim=0).tolist()
        ],
        "prompt_invariance": _prompt_invariance(physical, verified),
        "negative_mean": float(
            np.mean([row["explicit_negative_mean"] for row in metrics])
        ),
        "per_prompt_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.1 loss response requires CUDA")
    if args.diffusion_steps != 500 or args.seed != SEED:
        raise ValueError("Teacher-v9.1 response protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.1 response grid")
    configure_reproducibility(args.seed)

    failed_file = args.failed_smoke_summary.expanduser().resolve()
    failed = _validate_failure(failed_file)
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, policy = _validate_preflight(preflight_file)
    if failed.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("failed smoke/preflight binding differs")
    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _require_same_path(args.index, preflight, "dataset_index")
    split_file = _require_same_path(args.v5_split, preflight, "v5_split")
    stats_file = _require_same_path(args.stats_file, preflight, "stats_file")
    original_checkpoint = _require_same_path(
        args.original_checkpoint, preflight, "original_checkpoint"
    )
    v5_checkpoint = _require_same_path(args.v5_checkpoint, preflight, "v5_checkpoint")
    evidence_file = _require_same_path(
        args.v5_evidence_report, preflight, "v5_evidence_report"
    )
    index = validate_top_index(dataset_root, source_root, index_file)
    records = {str(row["scene_id"]): row for row in index["scenes"]}
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    mean, std = load_stats(stats_file)
    batch = _build_batch(bundle, mean, std, args.device)
    kwargs = _kwargs(batch)
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    split = load_split(split_file)
    v5_rows = load_v5_rows(v5_root, split, "train", mean, std, 4.0, 16.0, 0.7)
    if Counter(str(row["target"]) for row in v5_rows.values()) != Counter(
        {"chair": 18, "whiteboard": 6, "bed": 1}
    ):
        raise ValueError("v5 replay inventory changed")
    v5_ids = list(policy["v5_probe_ids"])
    v5_batch = stack_v5_batch(v5_rows, v5_ids, args.device)
    v5_kwargs = {
        "c_pc_xyz": v5_batch["xyz"],
        "c_pc_feat": v5_batch["feat"],
        "c_text": v5_batch["text"],
    }
    monitor_t = torch.tensor([175, 175], dtype=torch.long, device=args.device)
    monitor_noise = deterministic_noise(
        torch.Size((1, 8192, 6)), args.seed + 1000, args.device
    ).repeat(2, 1, 1)
    v5_t = torch.tensor([100, 300, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, args.seed + 2000, args.device)

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    with torch.no_grad():
        base_normal = predict_xstart(
            model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise
        )
        base_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    base_physical = _physical(base_normal, mean, std)
    base_v5_dense = (
        (base_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    )
    modules = install_lora(model, 4, 8.0, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named = lora_named_parameters(model)
    initial_state = _lora_cpu_state(named)
    with torch.no_grad():
        zero = predict_xstart(
            model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise
        )
        zero_v5 = predict_xstart(
            model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
        )
    if not torch.equal(zero, base_normal):
        raise AssertionError("zero-init parity changed")
    if not torch.equal(zero_v5, base_v5):
        raise AssertionError("zero-init v5 parity changed")
    before_objective = corrected_objective(
        zero, base_normal, batch, mean_tensor, std_tensor
    )
    before = _objective_values(before_objective, base_physical, bundle)
    del before_objective

    rows = []
    saved_predictions = []
    saved_v5 = []
    for candidate in CANDIDATES:
        _restore_lora_state(named, initial_state)
        optimizer = torch.optim.AdamW(
            list(named.values()), lr=LEARNING_RATE, weight_decay=0.0
        )
        logs = []
        for step in range(1, STEPS_PER_CANDIDATE + 1):
            set_frozen_base_eval_lora_train(model)
            set_lora_enabled(model, True)
            optimizer.zero_grad(set_to_none=True)
            step_t = torch.full(
                (2,), (100, 275, 450)[step - 1], dtype=torch.long, device=args.device
            )
            step_noise = deterministic_noise(
                torch.Size((1, 8192, 6)), args.seed + step * 1009, args.device
            ).repeat(2, 1, 1)
            set_lora_enabled(model, False)
            with torch.no_grad():
                frozen = predict_xstart(
                    model, diffusion, batch["x"], step_t, kwargs, step_noise
                )
                frozen_v5 = predict_xstart(
                    model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
                )
            set_lora_enabled(model, True)
            prediction = predict_xstart(
                model, diffusion, batch["x"], step_t, kwargs, step_noise
            )
            objective = corrected_objective(
                prediction, frozen, batch, mean_tensor, std_tensor
            )
            current_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
            replay = preservation_loss(current_v5, frozen_v5)
            total = (
                OBJECTIVE_WEIGHTS["instance_macro_primary"]
                * objective["instance_macro_primary"]
                + OBJECTIVE_WEIGHTS["verified_union"] * objective["verified_union"]
                + OBJECTIVE_WEIGHTS["environment_auxiliary"]
                * objective["environment_auxiliary"]
                + float(candidate["active_support_weight"])
                * objective["active_support_macro"]
                + float(candidate["ranking_weight"])
                * objective["within_object_ranking"]
                + float(candidate["negative_weight"])
                * objective["explicit_negative_addition"]
                + float(candidate["prompt_weight"])
                * objective["paired_prompt_invariance"]
                + float(candidate["v5_preservation_weight"]) * replay
                + OBJECTIVE_WEIGHTS["lora_regularizer"] * lora_parameter_energy(model)
            )
            total.backward()
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if "lora_" not in name
            ):
                raise AssertionError("gradient reached frozen CDM")
            gradient = float(
                torch.nn.utils.clip_grad_norm_(list(named.values()), GRAD_CLIP).item()
            )
            if not math.isfinite(gradient) or gradient <= 0.0:
                raise RuntimeError("candidate produced invalid gradient")
            optimizer.step()
            logs.append(
                {
                    "step": step,
                    "total": float(total.detach().item()),
                    "active_support": float(
                        objective["active_support_macro"].detach().item()
                    ),
                    "ranking": float(
                        objective["within_object_ranking"].detach().item()
                    ),
                    "v5_preservation": float(replay.detach().item()),
                    "gradient_l2_before_clip": gradient,
                }
            )
            del total, replay, current_v5, objective, prediction, frozen_v5, frozen
        model.eval()
        with torch.no_grad():
            candidate_normal = predict_xstart(
                model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise
            )
            candidate_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        physical = _physical(candidate_normal, mean, std)
        after_objective = corrected_objective(
            candidate_normal, base_normal, batch, mean_tensor, std_tensor
        )
        after = _objective_values(after_objective, physical, bundle)
        candidate_v5_dense = (
            (candidate_v5 - v5_batch["x"])
            .square()
            .mean(dim=(1, 2))
            .cpu()
            .tolist()
        )
        checks = response_checks(
            before=before,
            after=after,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=candidate_v5_dense,
        )
        row = {
            **dict(candidate),
            "steps": STEPS_PER_CANDIDATE,
            "learning_rate": LEARNING_RATE,
            "before": copy.deepcopy(before),
            "after": after,
            "base_v5_dense": [float(value) for value in base_v5_dense],
            "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
            "prediction_sha256": tensor_sha256(candidate_normal),
            "v5_prediction_sha256": tensor_sha256(candidate_v5),
            "update_log": logs,
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        _finite_tree(row, "candidate response")
        rows.append(row)
        saved_predictions.append(physical)
        saved_v5.append(candidate_v5.detach().cpu().numpy().astype(np.float32))
        print(
            f"[RESPONSE {candidate['name']}] eligible={row['eligible']} "
            f"support={before['active_support_macro']:.6f}->{after['active_support_macro']:.6f} "
            f"rank={before['within_object_ranking']:.6f}->{after['within_object_ranking']:.6f} "
            f"v5={np.mean(base_v5_dense):.6f}->{np.mean(candidate_v5_dense):.6f}",
            flush=True,
        )
        del optimizer, after_objective, candidate_normal, candidate_v5
        torch.cuda.empty_cache()

    order = rank_candidates(rows)
    selected_name = order[0] if order else None
    status = "PASS" if selected_name is not None else "FAIL"
    output_dir.mkdir(parents=True)
    arrays_file = output_dir / "response_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(bundle["xyz"], np.float32),
        verified_object_mask=np.asarray(bundle["verified_object_mask"], bool),
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        base=base_physical,
        candidates=np.stack(saved_predictions).astype(np.float32),
        candidate_names=np.asarray([row["name"] for row in rows]),
        v5_gt_normalized=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_normalized=base_v5.detach().cpu().numpy().astype(np.float32),
        candidate_v5_normalized=np.stack(saved_v5).astype(np.float32),
    )
    paths = {
        "failure_summary": failed_file,
        "failure_maps": Path(str(failed["paths"]["rollout_maps"])).resolve(),
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "dataset_index": index_file,
        "source_dataset_index": Path(
            str(preflight["paths"]["source_dataset_index"])
        ).resolve(),
        "v5_split": split_file,
        "stats_file": stats_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "v5_evidence_report": evidence_file,
        "objective": PREPARE_ROOT / "relational_teacher_v91_active_support_objective.py",
        "contract": PREPARE_ROOT / "relational_teacher_v91_loss_response_contract.py",
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v91_loss_response.py",
        "response_maps": arrays_file,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in paths.items()}
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "steps_per_candidate": STEPS_PER_CANDIDATE,
        "learning_rate": LEARNING_RATE,
        "grad_clip": GRAD_CLIP,
        "candidate_grid": list(CANDIDATES),
        "candidates": rows,
        "eligible_selection_order": order,
        "selected_candidate": selected_name,
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": "room_0102",
        "development_scene_metadata_only": "room_0201",
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "fresh_zero_lora_state_restored_per_candidate": True,
        "failed_smoke_checkpoint_loaded": False,
        "serialized_model_state": False,
        "lora": dict(lora_metadata(model)),
        "preflight_binding_id": preflight["binding_id"],
        "failed_smoke_binding_id": failed["binding_id"],
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": {
            "failed_smoke_diagnosis_bound": True,
            "fresh_v5r4_zero_init": True,
            "exact_candidate_grid": True,
            "three_updates_per_candidate": True,
            "only_lora_updated": True,
            "room_0102_arrays_unread": True,
            "room_0201_arrays_unread": True,
            "paper_test_unread": True,
            "no_checkpoint_saved": True,
            "at_least_one_candidate_admissible": selected_name is not None,
        },
        "authorizes_corrected_one_scene_overfit": status == "PASS",
        "authorizes_response3": False,
        "authorizes_calibration": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    report["failed_checks"] = sorted(
        name for name, passed in report["checks"].items() if not passed
    )
    report["binding_id"] = canonical_sha256(
        {
            "failed_smoke_binding_id": report["failed_smoke_binding_id"],
            "preflight_binding_id": report["preflight_binding_id"],
            "candidate_grid": report["candidate_grid"],
            "selected_candidate": selected_name,
            "response_maps_sha256": path_hashes["response_maps"],
        }
    )
    _finite_tree(report, "response report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    print(f"[LOSS_RESPONSE_{status}] Teacher-v9.1 active-support/ranking grid")
    print("[OK] selected candidate:", selected_name)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)


if __name__ == "__main__":
    main()

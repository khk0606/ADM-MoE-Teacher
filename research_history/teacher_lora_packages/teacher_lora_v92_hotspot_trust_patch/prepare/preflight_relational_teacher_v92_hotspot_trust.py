#!/usr/bin/env python3
"""Fresh, train-only Teacher-v9.2 hotspot/trust response grid.

This is a diagnostic gate, not a training run.  Every candidate starts from
sealed v5r4 plus zero-output LoRA, uses the same six room_0101 updates, and
persists predictions only.  No checkpoint is written.
"""

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
from relational_teacher_v92_hotspot_trust_contract import (  # noqa: E402
    CANDIDATES,
    GRAD_CLIP,
    LEARNING_RATE,
    SCHEMA,
    SEED,
    STEPS_PER_CANDIDATE,
    TRAIN_SCENE,
    V91_FAILURE_SCHEMA,
    V91_RESPONSE_SCHEMA,
    rank_candidates,
    response_checks,
)
from relational_teacher_v92_hotspot_trust_objective import (  # noqa: E402
    corrected_v92_objective,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
    _metrics,
    _physical,
    _prompt_invariance,
    _require_same_path,
    _restore_lora_state,
    _validate_loss_response,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-corrected-overfit-summary", type=Path, required=True)
    parser.add_argument("--selected-response-report", type=Path, required=True)
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


def _validate_failed_corrected(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V91_FAILURE_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("steps") != 120
        or value.get("selected_monitor_step") != 120
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("failed_checks")
        != [
            "one_step_three_object_panel_passes",
            "reverse_diffusion_three_object_panel_passes",
        ]
    ):
        raise ValueError("Teacher-v9.1 corrected failure binding changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("corrected failure path binding is absent")
    if set(paths) != set(hashes):
        raise ValueError("corrected failure path/hash inventory differs")
    for name, raw in paths.items():
        source = Path(str(raw)).expanduser().resolve()
        if not source.is_file() or sha256_file(source) != hashes[name]:
            raise ValueError("corrected-failure file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("failed corrected smoke contains forbidden model state")
    last = value.get("monitor_panels", [])[-1]
    if int(last.get("step", -1)) != 120 or last.get("eligible") is not False:
        raise ValueError("corrected failure terminal monitor changed")
    rollout = value.get("rollout", {})
    if not isinstance(rollout, Mapping) or rollout.get("panel", {}).get("eligible") is not False:
        raise ValueError("corrected failure rollout diagnosis changed")
    return value


def _objective_values(
    objective: Mapping[str, torch.Tensor],
    physical: np.ndarray,
    frozen_physical: np.ndarray,
    bundle: Mapping[str, object],
) -> Dict[str, object]:
    metrics = [_metrics(bundle, physical[index]) for index in range(2)]
    verified = np.asarray(bundle["verified_positive_mask"], bool)
    background_mask = ~(
        np.asarray(bundle["verified_object_mask"], bool).any(axis=0)
        | np.asarray(bundle["unknown_sittable_mask"], bool)
    )
    return {
        "active_support_macro": float(objective["active_support_macro"].item()),
        "hotspot_margin_macro": float(objective["hotspot_margin_macro"].item()),
        "hotspot_listwise_macro": float(objective["hotspot_listwise_macro"].item()),
        "per_instance_active_support": [
            float(item)
            for item in objective["per_instance_active_support"].mean(dim=0).tolist()
        ],
        "per_instance_hotspot_margin": [
            float(item)
            for item in objective["per_instance_hotspot_margin"].mean(dim=0).tolist()
        ],
        "per_instance_hotspot_listwise": [
            float(item)
            for item in objective["per_instance_hotspot_listwise"].mean(dim=0).tolist()
        ],
        "background_trust": float(
            np.square(physical[:, background_mask] - frozen_physical[:, background_mask]).mean()
        ),
        "prompt_invariance": _prompt_invariance(physical, verified),
        "negative_mean": float(np.mean([row["explicit_negative_mean"] for row in metrics])),
        "negative_max": float(max(row["explicit_negative_max"] for row in metrics)),
        "per_prompt_metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.2 hotspot/trust response requires CUDA")
    if args.diffusion_steps != 500 or args.seed != SEED:
        raise ValueError("Teacher-v9.2 hotspot/trust protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.2 response grid")
    configure_reproducibility(args.seed)

    corrected_file = args.failed_corrected_overfit_summary.expanduser().resolve()
    corrected_failure = _validate_failed_corrected(corrected_file)
    response_file = args.selected_response_report.expanduser().resolve()
    selected_response, _ = _validate_loss_response(response_file)
    if selected_response.get("schema") != V91_RESPONSE_SCHEMA:
        raise ValueError("Teacher-v9.1 response schema changed")
    if corrected_failure.get("loss_response_binding_id") != selected_response.get("binding_id"):
        raise ValueError("corrected failure/selected response binding differs")
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, policy = _validate_preflight(preflight_file)
    if corrected_failure.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("corrected failure/preflight binding differs")

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
    if set(records) != {"room_0101", "room_0102"}:
        raise ValueError("train scene metadata inventory changed")
    # The only v9 scene-array load. room_0102 and room_0201 remain unread.
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
    if [str(v5_rows[name]["target"]) for name in v5_ids] != [
        "chair", "bed", "whiteboard"
    ]:
        raise ValueError("v5 probe order changed")
    v5_batch = stack_v5_batch(v5_rows, v5_ids, args.device)
    v5_kwargs = {
        "c_pc_xyz": v5_batch["xyz"],
        "c_pc_feat": v5_batch["feat"],
        "c_text": v5_batch["text"],
    }
    monitor_t = torch.tensor([125, 125], dtype=torch.long, device=args.device)
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
        base_normal = predict_xstart(model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise)
        base_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    base_physical = _physical(base_normal, mean, std)
    base_v5_dense = (base_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()

    modules = install_lora(model, 4, 8.0, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named = lora_named_parameters(model)
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    initial_state = _lora_cpu_state(named)
    with torch.no_grad():
        zero = predict_xstart(model, diffusion, batch["x"], monitor_t, kwargs, monitor_noise)
        zero_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    if not torch.equal(zero, base_normal) or not torch.equal(zero_v5, base_v5):
        raise AssertionError("fresh zero-init parity changed")
    before_objective = corrected_v92_objective(
        zero, base_normal, batch, mean_tensor, std_tensor
    )
    before = _objective_values(before_objective, base_physical, base_physical, bundle)
    del before_objective, zero, zero_v5

    rows = []
    saved_predictions = []
    saved_normalized = []
    saved_v5 = []
    timestep_values = (50, 125, 200, 275, 350, 425)
    for candidate in CANDIDATES:
        _restore_lora_state(named, initial_state)
        optimizer = torch.optim.AdamW(
            list(named.values()), lr=LEARNING_RATE, weight_decay=0.0
        )
        logs = []
        for step, timestep_value in enumerate(timestep_values, start=1):
            set_frozen_base_eval_lora_train(model)
            optimizer.zero_grad(set_to_none=True)
            step_t = torch.full((2,), timestep_value, dtype=torch.long, device=args.device)
            step_noise = deterministic_noise(
                torch.Size((1, 8192, 6)), args.seed + step * 1009, args.device
            ).repeat(2, 1, 1)
            set_lora_enabled(model, False)
            with torch.no_grad():
                frozen = predict_xstart(model, diffusion, batch["x"], step_t, kwargs, step_noise)
                frozen_v5 = predict_xstart(
                    model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
                )
            set_lora_enabled(model, True)
            prediction = predict_xstart(model, diffusion, batch["x"], step_t, kwargs, step_noise)
            objective = corrected_v92_objective(
                prediction, frozen, batch, mean_tensor, std_tensor
            )
            current_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
            replay = preservation_loss(current_v5, frozen_v5)
            total = (
                OBJECTIVE_WEIGHTS["instance_macro_primary"] * objective["instance_macro_primary"]
                + OBJECTIVE_WEIGHTS["verified_union"] * objective["verified_union"]
                + OBJECTIVE_WEIGHTS["environment_auxiliary"] * objective["environment_auxiliary"]
                + float(candidate["active_support_weight"]) * objective["active_support_macro"]
                + float(candidate["hotspot_margin_weight"]) * objective["hotspot_margin_macro"]
                + float(candidate["hotspot_listwise_weight"]) * objective["hotspot_listwise_macro"]
                + float(candidate["absolute_negative_weight"]) * objective["absolute_negative"]
                + float(candidate["background_trust_weight"]) * objective["background_trust"]
                + float(candidate["prompt_weight"]) * objective["paired_prompt_invariance"]
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
                    "timestep": timestep_value,
                    "total": float(total.detach().item()),
                    "active_support": float(objective["active_support_macro"].detach().item()),
                    "hotspot_margin": float(objective["hotspot_margin_macro"].detach().item()),
                    "hotspot_listwise": float(objective["hotspot_listwise_macro"].detach().item()),
                    "absolute_negative": float(objective["absolute_negative"].detach().item()),
                    "background_trust": float(objective["background_trust"].detach().item()),
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
        after_objective = corrected_v92_objective(
            candidate_normal, base_normal, batch, mean_tensor, std_tensor
        )
        after = _objective_values(after_objective, physical, base_physical, bundle)
        candidate_v5_dense = (
            (candidate_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
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
        saved_normalized.append(candidate_normal.detach().cpu().numpy().astype(np.float32))
        saved_v5.append(candidate_v5.detach().cpu().numpy().astype(np.float32))
        before_instances = {
            name: np.mean([
                item["instances"][name]["topk_overlap"]
                for item in before["per_prompt_metrics"]
            ])
            for name in ("bed_01", "chair_01", "chair_06")
        }
        after_instances = {
            name: np.mean([
                item["instances"][name]["topk_overlap"]
                for item in after["per_prompt_metrics"]
            ])
            for name in ("bed_01", "chair_01", "chair_06")
        }
        print(
            f"[HOTSPOT {candidate['name']}] eligible={row['eligible']} "
            f"topk bed={before_instances['bed_01']:.6f}->{after_instances['bed_01']:.6f} "
            f"chair={before_instances['chair_01']:.6f}->{after_instances['chair_01']:.6f} "
            f"high={before_instances['chair_06']:.6f}->{after_instances['chair_06']:.6f}",
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
        verified_positive_mask=np.asarray(bundle["verified_positive_mask"], bool),
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        base=base_physical,
        candidates=np.stack(saved_predictions).astype(np.float32),
        base_normalized=base_normal.detach().cpu().numpy().astype(np.float32),
        candidates_normalized=np.stack(saved_normalized).astype(np.float32),
        candidate_names=np.asarray([row["name"] for row in rows]),
        v5_gt_normalized=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_normalized=base_v5.detach().cpu().numpy().astype(np.float32),
        candidate_v5_normalized=np.stack(saved_v5).astype(np.float32),
    )
    paths = {
        "failed_corrected_overfit_summary": corrected_file,
        "failed_corrected_overfit_maps": Path(
            str(corrected_failure["paths"]["rollout_maps"])
        ).resolve(),
        "selected_response_report": response_file,
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
        "objective": PREPARE_ROOT / "relational_teacher_v92_hotspot_trust_objective.py",
        "contract": PREPARE_ROOT / "relational_teacher_v92_hotspot_trust_contract.py",
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v92_hotspot_trust.py",
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
        "training_timesteps": list(timestep_values),
        "monitor_timestep": 125,
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
        "failed_corrected_checkpoint_loaded": False,
        "serialized_model_state": False,
        "lora": dict(lora_metadata(model)),
        "preflight_binding_id": preflight["binding_id"],
        "selected_response_binding_id": selected_response["binding_id"],
        "failed_corrected_binding_id": corrected_failure["binding_id"],
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": {
            "failed_corrected_diagnosis_bound": True,
            "fresh_v5r4_zero_init": True,
            "exact_ratio_varying_candidate_grid": True,
            "six_identical_updates_per_candidate": True,
            "three_object_topk_gate": True,
            "background_and_negative_trust_gate": True,
            "only_lora_updated": True,
            "room_0102_arrays_unread": True,
            "room_0201_arrays_unread": True,
            "paper_test_unread": True,
            "no_checkpoint_saved": True,
            "at_least_one_candidate_admissible": selected_name is not None,
        },
        "authorizes_early_rollout_canary": status == "PASS",
        "authorizes_overfit120": False,
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
            "preflight_binding_id": report["preflight_binding_id"],
            "selected_response_binding_id": report["selected_response_binding_id"],
            "failed_corrected_binding_id": report["failed_corrected_binding_id"],
            "candidate_grid": report["candidate_grid"],
            "selected_candidate": selected_name,
            "response_maps_sha256": path_hashes["response_maps"],
        }
    )
    _finite_tree(report, "hotspot/trust report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    print(f"[HOTSPOT_TRUST_{status}] Teacher-v9.2 six-update response gate")
    print("[OK] selected candidate:", selected_name)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)


if __name__ == "__main__":
    main()

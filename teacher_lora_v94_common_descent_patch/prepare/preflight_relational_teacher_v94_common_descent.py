#!/usr/bin/env python3
"""Train-only Teacher-v9.4 six-task gradient-geometry preflight.

This gate does not train a candidate checkpoint.  It measures the six exact
top-k task gradients at the fresh v5r4 initialization, constructs the
minimum-norm common-descent direction, and tests four small Euclidean steps.
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
    set_frozen_base_eval_lora_train,
    set_lora_enabled,
)
from relational_teacher_v9_all_sittable_contract import (  # noqa: E402
    atomic_savez,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v9_lora_preflight_contract import (  # noqa: E402
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
from relational_teacher_v93_exact_topk_objective import (  # noqa: E402
    corrected_v93_objective,
)
from relational_teacher_v94_common_descent import (  # noqa: E402
    apply_flat_direction,
    flattened_task_gradients,
    frank_wolfe_min_norm_weights,
)
from relational_teacher_v94_common_descent_contract import (  # noqa: E402
    EXPECTED_V93_FAILURES,
    FW_ITERATIONS,
    MIN_DIRECTIONAL_DERIVATIVE,
    PROBE_TIMESTEP,
    SCHEMA,
    SEED,
    STEP_RADII,
    TASK_ORDER,
    TRAIN_SCENE,
    V93_SCHEMA,
    candidate_checks,
    rank_candidates,
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
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v93-report", type=Path, required=True)
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


def _validate_failed_v93(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V93_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("selected_candidate") is not None
        or value.get("eligible_selection_order") != []
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("failed_checks") != ["at_least_one_candidate_admissible"]
        or value.get("authorizes_exact_topk_rollout_canary") is not False
    ):
        raise ValueError("Teacher-v9.3 exact-topk failure binding changed")
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Teacher-v9.3 failure candidate inventory changed")
    observed = {str(row.get("name")): row.get("failed_checks") for row in rows}
    if observed != EXPECTED_V93_FAILURES or any(row.get("eligible") is not False for row in rows):
        raise ValueError("Teacher-v9.3 per-candidate failure diagnosis changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.3 failure path binding is absent")
    for name, raw in paths.items():
        source = Path(str(raw)).expanduser().resolve()
        if not source.is_file() or sha256_file(source) != hashes[name]:
            raise ValueError("Teacher-v9.3 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.3 failure contains forbidden model state")
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
        | np.asarray(bundle["explicit_negative_mask"], bool)
    )
    overlap_rows = objective["per_instance_exact_topk_overlap"].detach().cpu().numpy()
    for prompt in range(2):
        for slot, name in enumerate(("bed_01", "chair_01", "chair_06")):
            metric = float(metrics[prompt]["instances"][name]["topk_overlap"])
            if abs(float(overlap_rows[prompt, slot]) - metric) > 2e-7:
                raise AssertionError("training/evaluation exact top-k sets differ")
    task_rows = objective["per_instance_exact_topk_swap"].detach().cpu().numpy()
    return {
        "per_task_exact_topk_swap": task_rows.astype(float).tolist(),
        "per_task_exact_topk_overlap": overlap_rows.astype(float).tolist(),
        "per_instance_exact_topk_swap": task_rows.mean(axis=0).astype(float).tolist(),
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
        raise RuntimeError("Teacher-v9.4 common-descent preflight requires CUDA")
    if args.diffusion_steps != 500 or args.seed != SEED:
        raise ValueError("Teacher-v9.4 common-descent protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.4 common-descent preflight")
    configure_reproducibility(args.seed)

    failed_v93_file = args.failed_v93_report.expanduser().resolve()
    failed_v93 = _validate_failed_v93(failed_v93_file)
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, policy = _validate_preflight(preflight_file)
    if failed_v93.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("v9.3 failure/preflight binding differs")

    source_root = args.source_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    v5_root = args.v5_dataset_root.expanduser().resolve()
    index_file = _require_same_path(args.index, preflight, "dataset_index")
    split_file = _require_same_path(args.v5_split, preflight, "v5_split")
    stats_file = _require_same_path(args.stats_file, preflight, "stats_file")
    original_checkpoint = _require_same_path(args.original_checkpoint, preflight, "original_checkpoint")
    v5_checkpoint = _require_same_path(args.v5_checkpoint, preflight, "v5_checkpoint")
    evidence_file = _require_same_path(args.v5_evidence_report, preflight, "v5_evidence_report")
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
    if [str(v5_rows[name]["target"]) for name in v5_ids] != ["chair", "bed", "whiteboard"]:
        raise ValueError("v5 probe order changed")
    v5_batch = stack_v5_batch(v5_rows, v5_ids, args.device)
    v5_kwargs = {
        "c_pc_xyz": v5_batch["xyz"],
        "c_pc_feat": v5_batch["feat"],
        "c_text": v5_batch["text"],
    }
    probe_t = torch.tensor([PROBE_TIMESTEP, PROBE_TIMESTEP], dtype=torch.long, device=args.device)
    probe_noise = deterministic_noise(
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
        base_normal = predict_xstart(model, diffusion, batch["x"], probe_t, kwargs, probe_noise)
        base_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    base_physical = _physical(base_normal, mean, std)
    base_v5_dense = (base_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()

    modules = install_lora(model, 4, 8.0, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named = lora_named_parameters(model)
    parameters = list(named.values())
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    initial_state = _lora_cpu_state(named)
    with torch.no_grad():
        zero = predict_xstart(model, diffusion, batch["x"], probe_t, kwargs, probe_noise)
        zero_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    if not torch.equal(zero, base_normal) or not torch.equal(zero_v5, base_v5):
        raise AssertionError("fresh zero-init parity changed")
    zero_objective = corrected_v93_objective(zero, base_normal, batch, mean_tensor, std_tensor)
    before = _objective_values(zero_objective, base_physical, base_physical, bundle)
    del zero_objective, zero, zero_v5

    # Compute the six live gradients on the exact fixed evaluation panel.
    set_lora_enabled(model, True)
    prediction = predict_xstart(model, diffusion, batch["x"], probe_t, kwargs, probe_noise)
    objective = corrected_v93_objective(prediction, base_normal, batch, mean_tensor, std_tensor)
    task_losses = objective["per_instance_exact_topk_swap"].reshape(-1)
    task_gradients = flattened_task_gradients(task_losses, parameters)
    gram_tensor = task_gradients @ task_gradients.T
    gram = gram_tensor.detach().cpu().double().numpy()
    weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
    weight_tensor = torch.from_numpy(weights).to(task_gradients.device, task_gradients.dtype)
    common = (weight_tensor[:, None] * task_gradients).sum(dim=0)
    common_norm = float(torch.linalg.vector_norm(common).item())
    if not math.isfinite(common_norm) or common_norm <= 0.0:
        common_direction = torch.zeros_like(common)
        directional = np.zeros(len(TASK_ORDER), dtype=np.float64)
        common_direction_valid = False
    else:
        common_direction = common / common_norm
        directional = (task_gradients @ common_direction).detach().cpu().double().numpy()
        common_direction_valid = bool(
            np.isfinite(directional).all()
            and float(directional.min()) >= MIN_DIRECTIONAL_DERIVATIVE
        )
    cosine = gram.copy()
    conflict_pairs = sum(
        1
        for left in range(len(TASK_ORDER))
        for right in range(left + 1, len(TASK_ORDER))
        if cosine[left, right] < -1e-8
    )
    del objective, prediction, task_losses, gram_tensor

    rows = []
    saved_predictions = []
    saved_normalized = []
    saved_v5 = []
    for radius in STEP_RADII:
        _restore_lora_state(named, initial_state)
        if common_direction_valid:
            apply_flat_direction(parameters, common_direction, radius)
        model.eval()
        with torch.no_grad():
            candidate_normal = predict_xstart(
                model, diffusion, batch["x"], probe_t, kwargs, probe_noise
            )
            candidate_v5 = predict_xstart(
                model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise
            )
        physical = _physical(candidate_normal, mean, std)
        after_objective = corrected_v93_objective(
            candidate_normal, base_normal, batch, mean_tensor, std_tensor
        )
        after = _objective_values(after_objective, physical, base_physical, bundle)
        candidate_v5_dense = (
            (candidate_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
        )
        checks = candidate_checks(
            before=before,
            after=after,
            base_v5_dense=base_v5_dense,
            candidate_v5_dense=candidate_v5_dense,
            common_direction_valid=common_direction_valid,
        )
        name = "common_descent_radius_" + str(radius).replace(".", "p")
        row = {
            "name": name,
            "step_radius": float(radius),
            "before": copy.deepcopy(before),
            "after": after,
            "base_v5_dense": [float(value) for value in base_v5_dense],
            "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
            "prediction_sha256": tensor_sha256(candidate_normal),
            "v5_prediction_sha256": tensor_sha256(candidate_v5),
            "checks": checks,
            "eligible": all(checks.values()),
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
        _finite_tree(row, "common-descent candidate")
        rows.append(row)
        saved_predictions.append(physical)
        saved_normalized.append(candidate_normal.detach().cpu().numpy().astype(np.float32))
        saved_v5.append(candidate_v5.detach().cpu().numpy().astype(np.float32))
        before_task = np.asarray(before["per_task_exact_topk_swap"], dtype=np.float64)
        after_task = np.asarray(after["per_task_exact_topk_swap"], dtype=np.float64)
        print(
            f"[COMMON-DESCENT R={radius:.4g}] eligible={row['eligible']} "
            f"task-change-min/max={(before_task-after_task).min():.9f}/"
            f"{(before_task-after_task).max():.9f} failed={row['failed_checks']}",
            flush=True,
        )
        del after_objective, candidate_normal, candidate_v5
        torch.cuda.empty_cache()

    order = rank_candidates(rows)
    selected_name = order[0] if order else None
    status = "PASS" if selected_name is not None else "FAIL"
    output_dir.mkdir(parents=True)
    arrays_file = output_dir / "common_descent_maps.npz"
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
        "failed_v93_report": failed_v93_file,
        "failed_v93_maps": Path(str(failed_v93["paths"]["response_maps"])).resolve(),
        "preflight_report": preflight_file,
        "metric_policy": Path(str(preflight["paths"]["metric_policy"])).resolve(),
        "dataset_index": index_file,
        "source_dataset_index": Path(str(preflight["paths"]["source_dataset_index"])).resolve(),
        "v5_split": split_file,
        "stats_file": stats_file,
        "original_checkpoint": original_checkpoint,
        "v5_checkpoint": v5_checkpoint,
        "v5_evidence_report": evidence_file,
        "v93_objective": PREPARE_ROOT / "relational_teacher_v93_exact_topk_objective.py",
        "common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "contract": PREPARE_ROOT / "relational_teacher_v94_common_descent_contract.py",
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v94_common_descent.py",
        "response_maps": arrays_file,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in paths.items()}
    gradient_audit = {
        "task_order": list(TASK_ORDER),
        "normalized_gram": gram.astype(float).tolist(),
        "pairwise_conflict_count": int(conflict_pairs),
        "frank_wolfe_iterations": FW_ITERATIONS,
        "minimum_norm_weights": weights.astype(float).tolist(),
        "common_vector_norm": common_norm,
        "task_directional_derivatives": directional.astype(float).tolist(),
        "minimum_task_directional_derivative": float(directional.min()),
        "common_direction_valid": common_direction_valid,
    }
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "probe_timestep": PROBE_TIMESTEP,
        "step_radius_grid": list(STEP_RADII),
        "gradient_audit": gradient_audit,
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
        "failed_v93_checkpoint_loaded": False,
        "serialized_model_state": False,
        "lora": dict(lora_metadata(model)),
        "preflight_binding_id": preflight["binding_id"],
        "failed_v93_binding_id": failed_v93["binding_id"],
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": {
            "failed_v93_diagnosis_bound": True,
            "fresh_v5r4_zero_init": True,
            "six_exact_task_gradients_measured": True,
            "common_direction_exists": common_direction_valid,
            "exact_radius_grid": True,
            "only_lora_perturbed": True,
            "room_0102_arrays_unread": True,
            "room_0201_arrays_unread": True,
            "paper_test_unread": True,
            "no_checkpoint_saved": True,
            "at_least_one_trust_region_step_admissible": selected_name is not None,
        },
        "authorizes_common_descent_response6": status == "PASS",
        "authorizes_rollout": False,
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
            "failed_v93_binding_id": report["failed_v93_binding_id"],
            "gradient_audit": gradient_audit,
            "step_radius_grid": report["step_radius_grid"],
            "selected_candidate": selected_name,
            "response_maps_sha256": path_hashes["response_maps"],
        }
    )
    _finite_tree(report, "common-descent report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    print(f"[COMMON_DESCENT_{status}] Teacher-v9.4 gradient-geometry preflight")
    print("[OK] conflict pairs:", conflict_pairs, "/ 15")
    print("[OK] minimum directional derivative:", float(directional.min()))
    print("[OK] selected candidate:", selected_name)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)


if __name__ == "__main__":
    main()

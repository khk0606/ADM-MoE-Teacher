#!/usr/bin/env python3
"""Noise/timestep-disjoint metric replication of the sealed v9.4 direction."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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
from preflight_relational_teacher_v94_common_descent import (  # noqa: E402
    _objective_values,
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
from relational_teacher_v93_exact_topk_objective import corrected_v93_objective  # noqa: E402
from relational_teacher_v94_common_descent import (  # noqa: E402
    apply_flat_direction,
    flattened_task_gradients,
    frank_wolfe_min_norm_weights,
)
from relational_teacher_v94_common_descent_contract import (  # noqa: E402
    FW_ITERATIONS,
    PROBE_TIMESTEP,
    STEP_RADII,
)
from relational_teacher_v941_metric_replication_contract import (  # noqa: E402
    DIRECTION_SEED,
    OBJECTS,
    REPLICATION_PANELS,
    SCHEMA,
    SEED,
    SELECTED_RADIUS,
    TRAIN_SCENE,
    V94_SCHEMA,
    replication_checks,
)
from run_relational_teacher_v91_corrected_one_scene_overfit import (  # noqa: E402
    _build_batch,
    _kwargs,
    _lora_cpu_state,
    _physical,
    _require_same_path,
    _restore_lora_state,
    _validate_preflight,
)
from train_fewshot_cdm import load_rows as load_v5_rows  # noqa: E402
from train_fewshot_cdm import stack_batch as stack_v5_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-v94-report", type=Path, required=True)
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
        raise ValueError(label + " contains NaN/Inf")


def _mean_topk(row: Mapping[str, object], name: str) -> float:
    return sum(
        float(prompt["instances"][name]["topk_overlap"])
        for prompt in row["per_prompt_metrics"]
    ) / 2.0


def _validate_failed_v94(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V94_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_candidate") is not None
        or value.get("eligible_selection_order") != []
        or value.get("train_scene") != TRAIN_SCENE
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("failed_checks") != ["at_least_one_trust_region_step_admissible"]
        or value.get("authorizes_common_descent_response6") is not False
    ):
        raise ValueError("Teacher-v9.4 common-descent failure binding changed")
    audit = value.get("gradient_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("common_direction_valid") is not True
        or audit.get("pairwise_conflict_count") != 8
        or float(audit.get("minimum_task_directional_derivative", 0.0)) <= 0.49
    ):
        raise ValueError("Teacher-v9.4 gradient diagnosis changed")
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(STEP_RADII):
        raise ValueError("Teacher-v9.4 radius inventory changed")
    if [float(row.get("step_radius")) for row in rows] != list(STEP_RADII):
        raise ValueError("Teacher-v9.4 radius grid changed")
    if any(
        row.get("eligible") is not False
        or row.get("failed_checks") != ["all_six_same_panel_swap_losses_decrease"]
        for row in rows
    ):
        raise ValueError("Teacher-v9.4 single-surrogate failure changed")
    selected = rows[-1]
    if float(selected["step_radius"]) != SELECTED_RADIUS:
        raise ValueError("Teacher-v9.4 metric replication radius changed")
    strict_improvements = 0
    for name in OBJECTS:
        if _mean_topk(selected["after"], name) <= _mean_topk(selected["before"], name):
            raise ValueError("selected radius does not improve every object mean top-k")
        for prompt in range(2):
            before = float(selected["before"]["per_prompt_metrics"][prompt]["instances"][name]["topk_overlap"])
            after = float(selected["after"]["per_prompt_metrics"][prompt]["instances"][name]["topk_overlap"])
            if after < before:
                raise ValueError("selected radius regresses a fixed-panel top-k case")
            strict_improvements += int(after > before)
    if strict_improvements != 5:
        raise ValueError("selected radius fixed-panel improvement count changed")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.4 path binding is absent")
    for name, raw in paths.items():
        source = Path(str(raw)).expanduser().resolve()
        if not source.is_file() or sha256_file(source) != hashes[name]:
            raise ValueError("Teacher-v9.4 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.4 failure contains forbidden model state")
    return value


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.4.1 metric replication requires CUDA")
    if args.diffusion_steps != 500 or args.seed != SEED:
        raise ValueError("Teacher-v9.4.1 metric replication protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.4.1 metric replication")
    # LoRA A-matrix initialization must reproduce the sealed v9.4 direction.
    # Replication noise is independently fixed by REPLICATION_PANELS below.
    configure_reproducibility(DIRECTION_SEED)

    failed_v94_file = args.failed_v94_report.expanduser().resolve()
    failed_v94 = _validate_failed_v94(failed_v94_file)
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, policy = _validate_preflight(preflight_file)
    if failed_v94.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("v9.4 failure/preflight binding differs")

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
    # The only scene-array load. room_0102 and room_0201 remain unread.
    bundle = load_train_scene_bundle(dataset_root, source_root, records[TRAIN_SCENE])
    mean, std = load_stats(stats_file)
    batch = _build_batch(bundle, mean, std, args.device)
    kwargs = _kwargs(batch)
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)

    split = load_split(split_file)
    v5_rows = load_v5_rows(v5_root, split, "train", mean, std, 4.0, 16.0, 0.7)
    if Counter(str(row["target"]) for row in v5_rows.values()) != Counter({"chair": 18, "whiteboard": 6, "bed": 1}):
        raise ValueError("v5 replay inventory changed")
    v5_ids = list(policy["v5_probe_ids"])
    if [str(v5_rows[name]["target"]) for name in v5_ids] != ["chair", "bed", "whiteboard"]:
        raise ValueError("v5 probe order changed")
    v5_batch = stack_v5_batch(v5_rows, v5_ids, args.device)
    v5_kwargs = {"c_pc_xyz": v5_batch["xyz"], "c_pc_feat": v5_batch["feat"], "c_text": v5_batch["text"]}

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    modules = install_lora(model, 4, 8.0, dropout=0.0)
    if len(modules) != 31:
        raise AssertionError("LoRA module inventory changed")
    set_frozen_base_eval_lora_train(model)
    named = lora_named_parameters(model)
    parameters = list(named.values())
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name):
        raise AssertionError("a frozen CDM parameter became trainable")
    initial_state = _lora_cpu_state(named)

    # Recreate the sealed v9.4 common direction on its original panel.
    direction_t = torch.tensor([PROBE_TIMESTEP, PROBE_TIMESTEP], dtype=torch.long, device=args.device)
    direction_noise = deterministic_noise(torch.Size((1, 8192, 6)), DIRECTION_SEED + 1000, args.device).repeat(2, 1, 1)
    set_lora_enabled(model, False)
    with torch.no_grad():
        direction_base = predict_xstart(model, diffusion, batch["x"], direction_t, kwargs, direction_noise)
    set_lora_enabled(model, True)
    direction_prediction = predict_xstart(model, diffusion, batch["x"], direction_t, kwargs, direction_noise)
    if not torch.equal(direction_prediction.detach(), direction_base):
        raise AssertionError("fresh zero-init direction panel differs from v5r4")
    direction_objective = corrected_v93_objective(
        direction_prediction, direction_base, batch, mean_tensor, std_tensor
    )
    task_gradients = flattened_task_gradients(
        direction_objective["per_instance_exact_topk_swap"].reshape(-1), parameters
    )
    gram = (task_gradients @ task_gradients.T).detach().cpu().double().numpy()
    weights = frank_wolfe_min_norm_weights(gram, FW_ITERATIONS)
    weight_tensor = torch.from_numpy(weights).to(task_gradients.device, task_gradients.dtype)
    common = (weight_tensor[:, None] * task_gradients).sum(dim=0)
    common_direction = common / torch.linalg.vector_norm(common)
    directional = (task_gradients @ common_direction).detach().cpu().double().numpy()
    audit = failed_v94["gradient_audit"]
    if not np.allclose(gram, np.asarray(audit["normalized_gram"], np.float64), rtol=2e-6, atol=2e-6):
        raise ValueError("v9.4 gradient Gram matrix did not reproduce")
    if not np.allclose(weights, np.asarray(audit["minimum_norm_weights"], np.float64), rtol=2e-6, atol=2e-6):
        raise ValueError("v9.4 common-descent weights did not reproduce")
    if not np.allclose(directional, np.asarray(audit["task_directional_derivatives"], np.float64), rtol=2e-6, atol=2e-6):
        raise ValueError("v9.4 directional derivatives did not reproduce")
    del direction_objective, direction_prediction, direction_base

    _restore_lora_state(named, initial_state)
    apply_flat_direction(parameters, common_direction, SELECTED_RADIUS)
    model.eval()
    base_rows = []
    candidate_rows = []
    base_maps = []
    candidate_maps = []
    base_normalized = []
    candidate_normalized = []
    for panel in REPLICATION_PANELS:
        timestep = int(panel["timestep"])
        noise_seed = int(panel["noise_seed"])
        panel_t = torch.tensor([timestep, timestep], dtype=torch.long, device=args.device)
        panel_noise = deterministic_noise(torch.Size((1, 8192, 6)), noise_seed, args.device).repeat(2, 1, 1)
        set_lora_enabled(model, False)
        with torch.no_grad():
            base = predict_xstart(model, diffusion, batch["x"], panel_t, kwargs, panel_noise)
        set_lora_enabled(model, True)
        with torch.no_grad():
            candidate = predict_xstart(model, diffusion, batch["x"], panel_t, kwargs, panel_noise)
        base_physical = _physical(base, mean, std)
        candidate_physical = _physical(candidate, mean, std)
        base_objective = corrected_v93_objective(base, base, batch, mean_tensor, std_tensor)
        candidate_objective = corrected_v93_objective(candidate, base, batch, mean_tensor, std_tensor)
        base_value = _objective_values(base_objective, base_physical, base_physical, bundle)
        candidate_value = _objective_values(candidate_objective, candidate_physical, base_physical, bundle)
        base_rows.append(base_value)
        candidate_rows.append(candidate_value)
        base_maps.append(base_physical)
        candidate_maps.append(candidate_physical)
        base_normalized.append(base.detach().cpu().numpy().astype(np.float32))
        candidate_normalized.append(candidate.detach().cpu().numpy().astype(np.float32))
        print(
            f"[REPLICA {panel['name']}] t={timestep} "
            + " ".join(
                f"{name}={_mean_topk(base_value, name):.6f}->{_mean_topk(candidate_value, name):.6f}"
                for name in OBJECTS
            ),
            flush=True,
        )
        del base_objective, candidate_objective, base, candidate
        torch.cuda.empty_cache()

    v5_t = torch.tensor([75, 250, 450], dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, SEED + 5000, args.device)
    set_lora_enabled(model, False)
    with torch.no_grad():
        base_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    set_lora_enabled(model, True)
    with torch.no_grad():
        candidate_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    base_v5_dense = (base_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    candidate_v5_dense = (candidate_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
    checks = replication_checks(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        base_v5_dense=base_v5_dense,
        candidate_v5_dense=candidate_v5_dense,
    )
    status = "PASS" if all(checks.values()) else "FAIL"

    output_dir.mkdir(parents=True)
    arrays_file = output_dir / "metric_replication_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(bundle["xyz"], np.float32),
        verified_object_mask=np.asarray(bundle["verified_object_mask"], bool),
        verified_positive_mask=np.asarray(bundle["verified_positive_mask"], bool),
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        base=np.stack(base_maps).astype(np.float32),
        candidate=np.stack(candidate_maps).astype(np.float32),
        base_normalized=np.stack(base_normalized).astype(np.float32),
        candidate_normalized=np.stack(candidate_normalized).astype(np.float32),
        v5_gt_normalized=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_normalized=base_v5.detach().cpu().numpy().astype(np.float32),
        candidate_v5_normalized=candidate_v5.detach().cpu().numpy().astype(np.float32),
    )
    paths = {
        "failed_v94_report": failed_v94_file,
        "failed_v94_maps": Path(str(failed_v94["paths"]["response_maps"])).resolve(),
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
        "v94_common_descent": PREPARE_ROOT / "relational_teacher_v94_common_descent.py",
        "contract": PREPARE_ROOT / "relational_teacher_v941_metric_replication_contract.py",
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v941_metric_replication.py",
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
        "direction_seed": DIRECTION_SEED,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "selected_radius": SELECTED_RADIUS,
        "replication_panels": list(REPLICATION_PANELS),
        "base_rows": base_rows,
        "candidate_rows": candidate_rows,
        "base_v5_dense": [float(value) for value in base_v5_dense],
        "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
        "base_v5_prediction_sha256": tensor_sha256(base_v5),
        "candidate_v5_prediction_sha256": tensor_sha256(candidate_v5),
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": "room_0102",
        "development_scene_metadata_only": "room_0201",
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "serialized_model_state": False,
        "failed_v94_checkpoint_loaded": False,
        "lora": dict(lora_metadata(model)),
        "preflight_binding_id": preflight["binding_id"],
        "failed_v94_binding_id": failed_v94["binding_id"],
        "paths": path_strings,
        "path_sha256": path_hashes,
        "authorizes_common_descent_response6": status == "PASS",
        "authorizes_checkpoint": False,
        "authorizes_rollout": False,
        "authorizes_overfit120": False,
        "authorizes_calibration": False,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    report["binding_id"] = canonical_sha256(
        {
            "preflight_binding_id": report["preflight_binding_id"],
            "failed_v94_binding_id": report["failed_v94_binding_id"],
            "selected_radius": SELECTED_RADIUS,
            "replication_panels": report["replication_panels"],
            "response_maps_sha256": path_hashes["response_maps"],
            "checks": checks,
        }
    )
    _finite_tree(report, "metric replication report")
    report_file = output_dir / "summary.json"
    atomic_write_json(report_file, report)
    print(f"[METRIC_REPLICATION_{status}] Teacher-v9.4.1 noise/timestep-disjoint gate")
    print("[OK] selected radius:", SELECTED_RADIUS)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] summary:", report_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Teacher-v9.6 object-balanced multi-timestep consensus preflight."""

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
from preflight_relational_teacher_v94_common_descent import _objective_values  # noqa: E402
from preflight_relational_teacher_v95_multitimestep_consensus import _panel_prediction  # noqa: E402
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
)
from relational_teacher_v95_multitimestep_consensus_contract import (  # noqa: E402
    AUDIT_PANELS as V95_AUDIT_PANELS,
    DESIGN_PANELS as V95_DESIGN_PANELS,
)
from relational_teacher_v96_object_balanced_consensus_contract import (  # noqa: E402
    AUDIT_PANELS,
    DESIGN_PANELS,
    DIRECTION_NAMES,
    EXPECTED_V95_FAILURES,
    FW_ITERATIONS,
    INITIALIZATION_SEED,
    OBJECTS,
    SCHEMA,
    SEED,
    STEP_RADII,
    TRAIN_SCENE,
    V5_NOISE_SEED,
    V5_TIMESTEPS,
    V95_SCHEMA,
    build_direction_specs,
    candidate_checks,
    rank_candidates,
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
    parser.add_argument("--failed-v95-report", type=Path, required=True)
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
    return sum(float(prompt["instances"][name]["topk_overlap"]) for prompt in row["per_prompt_metrics"]) / 2.0


def _validate_failed_v95(path: Path) -> Mapping[str, object]:
    value = read_json(path)
    if (
        value.get("schema") != V95_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("selected_candidate") is not None
        or value.get("eligible_selection_order") != []
        or value.get("failed_checks") != ["at_least_one_consensus_candidate_admissible"]
        or value.get("authorizes_consensus_response6") is not False
        or value.get("serialized_model_state") is not False
        or value.get("heldout_train_arrays_read") is not False
        or value.get("development_arrays_read") is not False
        or value.get("paper_test_access") is not False
        or value.get("train_scene") != TRAIN_SCENE
    ):
        raise ValueError("Teacher-v9.5 failure authority changed")
    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Teacher-v9.5 candidate inventory changed")
    observed = {str(row.get("name")): row.get("failed_checks") for row in rows}
    if observed != EXPECTED_V95_FAILURES or any(row.get("eligible") is not False for row in rows):
        raise ValueError("Teacher-v9.5 candidate failure anatomy changed")
    near = rows[0]
    if near.get("name") != "multitimestep_radius_0p001" or float(near.get("step_radius")) != 0.001:
        raise ValueError("Teacher-v9.5 nearest candidate changed")
    regressions = []
    for panel_index, panel in enumerate(value["audit_panels"]):
        for prompt, prompt_name in enumerate(("watch", "write")):
            for name in OBJECTS:
                base = float(near["base_rows"][panel_index]["per_prompt_metrics"][prompt]["instances"][name]["topk_overlap"])
                candidate = float(near["candidate_rows"][panel_index]["per_prompt_metrics"][prompt]["instances"][name]["topk_overlap"])
                if candidate < base:
                    regressions.append((str(panel["name"]), prompt_name, name))
    if regressions != [
        ("audit_t475", "watch", "bed_01"),
        ("audit_t475", "watch", "chair_01"),
    ]:
        raise ValueError("Teacher-v9.5 t475 regression diagnosis changed")
    for panel_index in range(len(value["audit_panels"])):
        for prompt in range(2):
            base = float(near["base_rows"][panel_index]["per_prompt_metrics"][prompt]["instances"]["chair_06"]["topk_overlap"])
            candidate = float(near["candidate_rows"][panel_index]["per_prompt_metrics"][prompt]["instances"]["chair_06"]["topk_overlap"])
            if candidate != base:
                raise ValueError("Teacher-v9.5 nearest-candidate High-Chair plateau changed")
    expected_design_keys = {(int(row["timestep"]), int(row["noise_seed"])) for row in V95_DESIGN_PANELS}
    expected_design_keys |= {(int(row["timestep"]), int(row["noise_seed"])) for row in V95_AUDIT_PANELS}
    new_design_keys = {(int(row["timestep"]), int(row["noise_seed"])) for row in DESIGN_PANELS}
    if expected_design_keys != new_design_keys:
        raise ValueError("Teacher-v9.6 design panels do not exactly reuse all disclosed panels")
    paths = value.get("paths")
    hashes = value.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
        raise ValueError("Teacher-v9.5 path binding is absent")
    for name, raw in paths.items():
        source = Path(str(raw)).expanduser().resolve()
        if not source.is_file() or sha256_file(source) != hashes[name]:
            raise ValueError("Teacher-v9.5 bound file changed: " + str(name))
    if list(path.parent.glob("*.pt")) or list(path.parent.glob("*.pth")):
        raise ValueError("Teacher-v9.5 failure contains forbidden model state")
    return value


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v9.6 object-balanced preflight requires CUDA")
    if args.diffusion_steps != 500 or args.seed != SEED:
        raise ValueError("Teacher-v9.6 protocol is sealed")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite Teacher-v9.6 preflight")
    configure_reproducibility(INITIALIZATION_SEED)

    failed_file = args.failed_v95_report.expanduser().resolve()
    failed = _validate_failed_v95(failed_file)
    preflight_file = args.preflight_report.expanduser().resolve()
    preflight, policy = _validate_preflight(preflight_file)
    if failed.get("preflight_binding_id") != preflight.get("binding_id"):
        raise ValueError("v9.5 failure/preflight binding differs")

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

    gradient_blocks = []
    task_labels = []
    for panel in DESIGN_PANELS:
        set_lora_enabled(model, False)
        with torch.no_grad():
            frozen = _panel_prediction(model, diffusion, batch, kwargs, panel, args.device)
        set_lora_enabled(model, True)
        prediction = _panel_prediction(model, diffusion, batch, kwargs, panel, args.device)
        if not torch.equal(prediction.detach(), frozen):
            raise AssertionError("fresh zero-init design panel differs from v5r4")
        objective = corrected_v93_objective(prediction, frozen, batch, mean_tensor, std_tensor)
        block = flattened_task_gradients(objective["per_instance_exact_topk_swap"].reshape(-1), parameters).detach().cpu()
        gradient_blocks.append(block)
        task_labels.extend(
            f"{panel['name']}|{prompt}|{name}"
            for prompt in ("watch", "write") for name in OBJECTS
        )
        del objective, prediction, frozen, block
        torch.cuda.empty_cache()
    task_gradients = torch.cat(gradient_blocks, dim=0)
    gram = (task_gradients.double() @ task_gradients.double().T).numpy()
    direction_specs = build_direction_specs(gram, task_labels)
    conflict_pairs = sum(
        1 for left in range(len(task_labels)) for right in range(left + 1, len(task_labels))
        if gram[left, right] < -1e-8
    )

    directions = {}
    for spec in direction_specs:
        coefficients = torch.tensor(spec["task_coefficients"], dtype=task_gradients.dtype)
        direction = coefficients @ task_gradients
        norm = torch.linalg.vector_norm(direction)
        if not torch.isclose(norm, norm.new_tensor(1.0), rtol=2e-5, atol=2e-6):
            raise AssertionError("object-balanced direction norm changed")
        directions[str(spec["name"])] = direction.to(args.device)
    del gradient_blocks

    base_rows = []
    base_maps = []
    base_normalized = []
    set_lora_enabled(model, False)
    model.eval()
    for panel in AUDIT_PANELS:
        with torch.no_grad():
            base = _panel_prediction(model, diffusion, batch, kwargs, panel, args.device)
        base_physical = _physical(base, mean, std)
        base_objective = corrected_v93_objective(base, base, batch, mean_tensor, std_tensor)
        base_rows.append(_objective_values(base_objective, base_physical, base_physical, bundle))
        base_maps.append(base_physical)
        base_normalized.append(base.detach().cpu().numpy().astype(np.float32))
        del base_objective, base
        torch.cuda.empty_cache()
    v5_t = torch.tensor(V5_TIMESTEPS, dtype=torch.long, device=args.device)
    v5_noise = deterministic_noise(v5_batch["x"].shape, V5_NOISE_SEED, args.device)
    with torch.no_grad():
        base_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
    base_v5_dense = (base_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()

    rows = []
    saved_candidates = []
    saved_candidates_normalized = []
    saved_v5 = []
    spec_by_name = {str(row["name"]): row for row in direction_specs}
    for direction_name in DIRECTION_NAMES:
        spec = spec_by_name[direction_name]
        for radius in STEP_RADII:
            _restore_lora_state(named, initial_state)
            if spec["valid"] is True:
                apply_flat_direction(parameters, directions[direction_name], radius)
            model.eval()
            set_lora_enabled(model, True)
            candidate_rows = []
            candidate_maps = []
            candidate_normalized = []
            for panel_index, panel in enumerate(AUDIT_PANELS):
                with torch.no_grad():
                    candidate = _panel_prediction(model, diffusion, batch, kwargs, panel, args.device)
                candidate_physical = _physical(candidate, mean, std)
                frozen_normalized = torch.from_numpy(base_normalized[panel_index]).to(args.device)
                objective = corrected_v93_objective(candidate, frozen_normalized, batch, mean_tensor, std_tensor)
                values = _objective_values(objective, candidate_physical, base_maps[panel_index], bundle)
                candidate_rows.append(values)
                candidate_maps.append(candidate_physical)
                candidate_normalized.append(candidate.detach().cpu().numpy().astype(np.float32))
                del objective, candidate, frozen_normalized
                torch.cuda.empty_cache()
            with torch.no_grad():
                candidate_v5 = predict_xstart(model, diffusion, v5_batch["x"], v5_t, v5_kwargs, v5_noise)
            candidate_v5_dense = (candidate_v5 - v5_batch["x"]).square().mean(dim=(1, 2)).cpu().tolist()
            checks = candidate_checks(
                base_rows=base_rows,
                candidate_rows=candidate_rows,
                base_v5_dense=base_v5_dense,
                candidate_v5_dense=candidate_v5_dense,
                direction_valid=bool(spec["valid"]),
            )
            name = direction_name + "_radius_" + str(radius).replace(".", "p")
            row = {
                "name": name,
                "direction_name": direction_name,
                "step_radius": float(radius),
                "base_rows": base_rows,
                "candidate_rows": candidate_rows,
                "base_v5_dense": [float(value) for value in base_v5_dense],
                "candidate_v5_dense": [float(value) for value in candidate_v5_dense],
                "candidate_v5_prediction_sha256": tensor_sha256(candidate_v5),
                "checks": checks,
                "eligible": all(checks.values()),
                "failed_checks": sorted(key for key, passed in checks.items() if not passed),
            }
            _finite_tree(row, "object-balanced candidate")
            rows.append(row)
            saved_candidates.append(np.stack(candidate_maps).astype(np.float32))
            saved_candidates_normalized.append(np.stack(candidate_normalized).astype(np.float32))
            saved_v5.append(candidate_v5.detach().cpu().numpy().astype(np.float32))
            gains = {
                object_name: sum(
                    _mean_topk(candidate_rows[index], object_name) - _mean_topk(base_rows[index], object_name)
                    for index in range(len(AUDIT_PANELS))
                ) / len(AUDIT_PANELS)
                for object_name in OBJECTS
            }
            print(
                f"[BALANCED {direction_name} R={radius:.4g}] eligible={row['eligible']} "
                f"topk-gain bed={gains['bed_01']:.6f} chair={gains['chair_01']:.6f} "
                f"high={gains['chair_06']:.6f} failed={row['failed_checks']}",
                flush=True,
            )
            del candidate_v5

    order = rank_candidates(rows)
    selected = order[0] if order else None
    status = "PASS" if selected is not None else "FAIL"
    output_dir.mkdir(parents=True)
    arrays_file = output_dir / "object_balanced_consensus_maps.npz"
    atomic_savez(
        arrays_file,
        xyz=np.asarray(bundle["xyz"], np.float32),
        verified_object_mask=np.asarray(bundle["verified_object_mask"], bool),
        verified_positive_mask=np.asarray(bundle["verified_positive_mask"], bool),
        unknown_sittable_mask=np.asarray(bundle["unknown_sittable_mask"], bool),
        explicit_negative_mask=np.asarray(bundle["explicit_negative_mask"], bool),
        instance_targets=np.asarray(bundle["instance_targets"], np.float32),
        base=np.stack(base_maps).astype(np.float32),
        candidates=np.stack(saved_candidates).astype(np.float32),
        base_normalized=np.stack(base_normalized).astype(np.float32),
        candidates_normalized=np.stack(saved_candidates_normalized).astype(np.float32),
        candidate_names=np.asarray([row["name"] for row in rows]),
        v5_gt_normalized=v5_batch["x"].detach().cpu().numpy().astype(np.float32),
        base_v5_normalized=base_v5.detach().cpu().numpy().astype(np.float32),
        candidate_v5_normalized=np.stack(saved_v5).astype(np.float32),
    )
    paths = {
        "failed_v95_report": failed_file,
        "failed_v95_maps": Path(str(failed["paths"]["response_maps"])).resolve(),
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
        "v95_contract": PREPARE_ROOT / "relational_teacher_v95_multitimestep_consensus_contract.py",
        "v95_runner": PREPARE_ROOT / "preflight_relational_teacher_v95_multitimestep_consensus.py",
        "contract": PREPARE_ROOT / "relational_teacher_v96_object_balanced_consensus_contract.py",
        "runner": Path(__file__).resolve(),
        "validator": PREPARE_ROOT / "validate_relational_teacher_v96_object_balanced_consensus.py",
        "response_maps": arrays_file,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    path_strings = {name: str(path.resolve()) for name, path in paths.items()}
    path_hashes = {name: sha256_file(path.resolve()) for name, path in paths.items()}
    gradient_audit = {
        "task_labels": task_labels,
        "task_count": len(task_labels),
        "normalized_gram": gram.astype(float).tolist(),
        "pairwise_conflict_count": int(conflict_pairs),
        "pair_count": len(task_labels) * (len(task_labels) - 1) // 2,
        "frank_wolfe_iterations": FW_ITERATIONS,
        "direction_specs": direction_specs,
    }
    checks = {
        "sealed_v95_failure_bound": True,
        "all_eight_disclosed_panels_used_for_design": True,
        "forty_eight_design_task_gradients_measured": True,
        "exact_direction_and_radius_grid": True,
        "new_audit_panels_disjoint": True,
        "fresh_v5r4_zero_init": True,
        "only_lora_perturbed": True,
        "room_0102_arrays_unread": True,
        "room_0201_arrays_unread": True,
        "paper_test_unread": True,
        "no_checkpoint_saved": True,
        "at_least_one_object_balanced_candidate_admissible": selected is not None,
    }
    report = {
        "schema": SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "initialization_seed": INITIALIZATION_SEED,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "design_panels": list(DESIGN_PANELS),
        "audit_panels": list(AUDIT_PANELS),
        "direction_names": list(DIRECTION_NAMES),
        "step_radius_grid": list(STEP_RADII),
        "v5_timesteps": list(V5_TIMESTEPS),
        "v5_noise_seed": V5_NOISE_SEED,
        "gradient_audit": gradient_audit,
        "candidates": rows,
        "eligible_selection_order": order,
        "selected_candidate": selected,
        "train_scene": TRAIN_SCENE,
        "heldout_train_scene_metadata_only": "room_0102",
        "development_scene_metadata_only": "room_0201",
        "heldout_train_arrays_read": False,
        "development_arrays_read": False,
        "paper_test_access": False,
        "serialized_model_state": False,
        "failed_checkpoint_loaded": False,
        "lora": dict(lora_metadata(model)),
        "preflight_binding_id": preflight["binding_id"],
        "failed_v95_binding_id": failed["binding_id"],
        "paths": path_strings,
        "path_sha256": path_hashes,
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        "authorizes_balanced_response6": status == "PASS",
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
            "failed_v95_binding_id": report["failed_v95_binding_id"],
            "gradient_audit": gradient_audit,
            "design_panels": report["design_panels"],
            "audit_panels": report["audit_panels"],
            "direction_names": report["direction_names"],
            "step_radius_grid": report["step_radius_grid"],
            "v5_timesteps": report["v5_timesteps"],
            "v5_noise_seed": report["v5_noise_seed"],
            "selected_candidate": selected,
            "response_maps_sha256": path_hashes["response_maps"],
        }
    )
    _finite_tree(report, "object-balanced consensus report")
    report_file = output_dir / "preflight.json"
    atomic_write_json(report_file, report)
    print(f"[OBJECT_BALANCED_CONSENSUS_{status}] Teacher-v9.6 CUDA gate")
    print("[OK] conflicts:", conflict_pairs, "/", gradient_audit["pair_count"])
    print("[OK] direction minima:", {row["name"]: row["minimum_task_directional_derivative"] for row in direction_specs})
    print("[OK] selected candidate:", selected)
    print("[OK] failed checks:", report["failed_checks"])
    print("[OK] report:", report_file)


if __name__ == "__main__":
    main()

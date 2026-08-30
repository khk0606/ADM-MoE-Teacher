#!/usr/bin/env python3
"""Strictly preflight a frozen v2 Base teacher before MoE can consume it.

``full`` mode reconstructs the exact effective CDM, checks every state hash,
reproduces all Base draws, repeats draw 0, and replays every sealed quality
draw for both the original and selected models. ``integrity-only`` is useful
after transporting an artifact, but deliberately exits non-zero and never
authorizes promotion.
"""

from __future__ import annotations

import argparse
import platform
import random
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_DIR = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from base_teacher_contract import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    convert_prediction_draws,
    load_contact_stats,
    load_json,
    load_quality_replay_row,
    load_scene_contract,
    sha256_array,
    sha256_file,
    validate_base_teacher_artifact,
    validate_v5r4_quality_chain,
)
from export_base_teacher import (  # noqa: E402
    create_model_strict,
    effective_model_state_hashes,
    pin_gate0_runtime_output_paths,
    resolved_gate0_cdm_config,
    runtime_file_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fewshot-checkpoint", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--shortlist-status", type=Path, required=True)
    parser.add_argument("--rollout-selection", type=Path, required=True)
    parser.add_argument("--train-rollout-audit", type=Path, required=True)
    parser.add_argument("--train-rollout-predictions", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--development-predictions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "integrity-only"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _numpy_rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _rng_snapshot(torch) -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _rng_equal(torch, left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    if left["python"] != right["python"]:
        return False
    if not _numpy_rng_equal(left["numpy"], right["numpy"]):
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = left["torch_cuda"]
    right_cuda = right["torch_cuda"]
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(left_value, right_value)
        for left_value, right_value in zip(left_cuda, right_cuda)
    )


def _restore_rng_snapshot(torch, snapshot: Mapping[str, object]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _device_fingerprint(torch, device: str) -> Dict[str, object]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(
            "full preflight requires CUDA because pointops_cuda has no CPU "
            "execution path"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("full preflight requested CUDA but CUDA is unavailable")
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name(torch_device),
        "device_capability": list(torch.cuda.get_device_capability(torch_device)),
    }


def _assert_files_unchanged(
    *,
    manifest_file: Path,
    artifact_file: Path,
    manifest_sha256: str,
    artifact_sha256: str,
    quality_result: Mapping[str, object],
) -> None:
    if sha256_file(manifest_file) != manifest_sha256:
        raise RuntimeError("Base teacher manifest changed during preflight")
    if sha256_file(artifact_file) != artifact_sha256:
        raise RuntimeError("Base teacher artifact changed during preflight")
    for name, path in quality_result["files"].items():
        if sha256_file(Path(str(path))) != quality_result["sha256"][name]:
            raise RuntimeError("quality evidence changed during preflight: " + name)


def _expected_quality_replay_plan(
    prediction_bundles: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    plan = {
        "schema": "history_affordance_v2_full_quality_replay_plan_v1",
        "mode": "all_samples_all_draws_both_models",
        "base_seed": 20260815,
        "roles": ["fewshot", "original"],
        "partitions": {},
        "total_replayed_draws": 0,
    }
    for partition in ("train_audit", "development"):
        if partition not in prediction_bundles:
            raise ValueError("quality replay bundle is missing " + partition)
        bundle = prediction_bundles[partition]
        sample_ids = [str(value) for value in bundle["sample_ids"]]
        fewshot_draws = np.asarray(bundle["fewshot_draws"])
        original_draws = np.asarray(bundle["original_draws"])
        if fewshot_draws.ndim < 2 or original_draws.shape != fewshot_draws.shape:
            raise ValueError("quality replay role draw shapes differ")
        if fewshot_draws.shape[0] != len(sample_ids):
            raise ValueError("quality replay sample/draw cardinality differs")
        sample_count = len(sample_ids)
        k_draws = int(fewshot_draws.shape[1])
        plan["partitions"][partition] = {
            "sample_ids": sample_ids,
            "sample_count": sample_count,
            "k_draws": k_draws,
            "total_replayed_draws_per_role": sample_count * k_draws,
        }
        plan["total_replayed_draws"] += 2 * sample_count * k_draws
    return plan


def _replay_quality_role(
    *,
    replay_model,
    replay_diffusion,
    role: str,
    prediction_bundles: Mapping[str, Mapping[str, object]],
    quality_rows: Dict[str, object],
    load_row,
    sample_contact,
    convert_prediction,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    show_progress: bool = True,
):
    if role not in {"fewshot", "original"}:
        raise ValueError("quality replay role must be fewshot or original")
    role_results = []
    for partition in ("train_audit", "development"):
        bundle = prediction_bundles[partition]
        expected_draws = np.asarray(bundle[role + "_draws"])
        sample_ids = [str(value) for value in bundle["sample_ids"]]
        initial_seeds = np.asarray(bundle["initial_noise_seeds"])
        reverse_seeds = np.asarray(bundle["reverse_noise_seeds"])
        expected_seed_shape = expected_draws.shape[:2]
        if (
            initial_seeds.shape != expected_seed_shape
            or reverse_seeds.shape != expected_seed_shape
            or expected_draws.shape[0] != len(sample_ids)
        ):
            raise ValueError("quality replay bundle seed/cardinality mismatch")
        for sample_index, sample_id in enumerate(sample_ids):
            if sample_id not in quality_rows:
                quality_rows[sample_id] = load_row(sample_id)
            quality_row = quality_rows[sample_id]
            for draw_index in range(expected_draws.shape[1]):
                normalized_quality = sample_contact(
                    replay_model,
                    replay_diffusion,
                    quality_row,
                    int(initial_seeds[sample_index, draw_index]),
                    int(reverse_seeds[sample_index, draw_index]),
                    device,
                )
                quality_affordance, _ = convert_prediction(
                    normalized_quality,
                    representation="normalized_contact",
                    mean=mean,
                    std=std,
                )
                if not np.array_equal(
                    quality_affordance[0],
                    expected_draws[sample_index, draw_index],
                ):
                    raise ValueError(
                        "quality prediction differs from checkpoint replay: "
                        + role
                        + "/"
                        + partition
                        + "/"
                        + sample_id
                        + "/draw_"
                        + str(draw_index)
                    )
            if show_progress:
                print(
                    "[QUALITY-REPLAY] role="
                    + role
                    + " partition="
                    + partition
                    + " sample="
                    + sample_id,
                    flush=True,
                )
        role_results.append(
            {
                "role": role,
                "partition": partition,
                "sample_count": len(sample_ids),
                "draw_count": int(expected_draws.shape[1]),
                "total_replayed_draws": int(
                    expected_draws.shape[0] * expected_draws.shape[1]
                ),
                "stored_draws_sha256": sha256_array(expected_draws),
                "all_reproduced_bitwise": True,
            }
        )
    return role_results


def _validate_quality_replay_coverage(
    replay_results, replay_plan: Mapping[str, object]
) -> int:
    expected = {}
    for role in replay_plan.get("roles", []):
        for partition, row in replay_plan.get("partitions", {}).items():
            expected[(role, partition)] = {
                "sample_count": row["sample_count"],
                "draw_count": row["k_draws"],
                "total_replayed_draws": row[
                    "total_replayed_draws_per_role"
                ],
            }
    actual = {}
    for row in replay_results:
        key = (row.get("role"), row.get("partition"))
        if key in actual:
            raise RuntimeError("duplicate quality replay coverage row: " + str(key))
        if row.get("all_reproduced_bitwise") is not True:
            raise RuntimeError("quality replay contains a non-bitwise result")
        actual[key] = {
            "sample_count": row.get("sample_count"),
            "draw_count": row.get("draw_count"),
            "total_replayed_draws": row.get("total_replayed_draws"),
        }
    if actual != expected:
        raise RuntimeError("full quality replay coverage differs from sealed plan")
    total = sum(row["total_replayed_draws"] for row in actual.values())
    if total != replay_plan.get("total_replayed_draws"):
        raise RuntimeError("full quality replay total differs from sealed plan")
    return int(total)


def _full_replay(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, object],
    scene: Mapping[str, object],
    quality_result: Mapping[str, object],
) -> Dict[str, object]:
    import torch
    from evaluate_fewshot_cdm import sample_contact_deterministic
    from train_fewshot_cdm import compose_cdm_config, configure_reproducibility

    runtime = manifest["runtime_provenance"]
    current_environment = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_build_config_sha256": canonical_json_sha256(
            {"torch_build_config": torch.__config__.show()}
        ),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        **_device_fingerprint(torch, args.device),
    }
    environment_mismatches = {
        name: {"expected": runtime.get(name), "actual": value}
        for name, value in current_environment.items()
        if runtime.get(name) != value
    }
    if environment_mismatches:
        raise ValueError(
            "full preflight environment differs from export: "
            + str(environment_mismatches)
        )

    current_runtime_files = runtime_file_hashes()
    if current_runtime_files != runtime.get("runtime_file_sha256"):
        raise ValueError("Base teacher runtime source/config files changed")

    diffusion_steps = int(manifest["cache_key_payload"]["diffusion_steps"])
    base_seed = int(manifest["cache_key_payload"]["base_seed"])
    configure_reproducibility(base_seed)
    cfg = compose_cdm_config(diffusion_steps, args.device)
    pin_gate0_runtime_output_paths(cfg)
    resolved_config = resolved_gate0_cdm_config(cfg)
    if canonical_json_sha256(resolved_config) != runtime.get(
        "resolved_config_sha256"
    ):
        raise ValueError("resolved CDM configuration changed")
    if resolved_config != runtime.get("resolved_config"):
        raise ValueError("resolved CDM configuration content changed")

    scene_weight = Path(str(cfg.model.scene_model.pretrained_weight)).expanduser()
    if not scene_weight.is_absolute():
        scene_weight = (REPO_ROOT / scene_weight).resolve()
    else:
        scene_weight = scene_weight.resolve()
    recorded_scene_weight = Path(
        str(runtime.get("scene_model_pretrained_weight", ""))
    ).expanduser()
    if not recorded_scene_weight.is_absolute():
        recorded_scene_weight = (REPO_ROOT / recorded_scene_weight).resolve()
    else:
        recorded_scene_weight = recorded_scene_weight.resolve()
    if recorded_scene_weight != scene_weight:
        raise ValueError(
            "recorded scene-model weight path differs from resolved config"
        )
    if not scene_weight.is_file():
        raise FileNotFoundError(scene_weight)
    if sha256_file(scene_weight) != runtime.get(
        "scene_model_pretrained_weight_sha256"
    ):
        raise ValueError("scene-model pretrained weight changed")

    expected_checkpoint_sha256 = manifest["quality_gate"]["sha256"][
        "fewshot_checkpoint"
    ]
    model, diffusion, checkpoint_coverage, loaded_checkpoint_sha256 = (
        create_model_strict(
            cfg,
            args.fewshot_checkpoint,
            args.device,
            expected_sha256=expected_checkpoint_sha256,
        )
    )
    if loaded_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("loaded few-shot checkpoint snapshot hash changed")
    if checkpoint_coverage != runtime.get("partial_checkpoint_coverage"):
        raise ValueError("partial checkpoint coverage changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("reconstructed Base teacher is not frozen/eval")
    state_before = effective_model_state_hashes(model)
    if state_before != runtime.get("effective_model_state"):
        raise ValueError("effective Base teacher state hashes changed")
    teacher_runtime_identity = {
        "schema": "history_affordance_v2_teacher_runtime_identity_v1",
        "fewshot_checkpoint_sha256": sha256_file(args.fewshot_checkpoint),
        "contact_stats_sha256": sha256_file(args.stats_file),
        "effective_model_state": state_before,
        "partial_checkpoint_coverage": checkpoint_coverage,
        "scene_model_pretrained_weight_sha256": sha256_file(scene_weight),
        "text_model_name": str(cfg.model.text_model.version),
        "resolved_config_sha256": canonical_json_sha256(resolved_config),
        "runtime_file_set_sha256": canonical_json_sha256(current_runtime_files),
        "sampling_environment": current_environment,
    }
    if teacher_runtime_identity != runtime.get("teacher_runtime_identity"):
        raise ValueError("effective Base teacher runtime identity changed")
    teacher_runtime_sha256 = canonical_json_sha256(teacher_runtime_identity)
    if teacher_runtime_sha256 != runtime.get("teacher_runtime_sha256"):
        raise ValueError("effective Base teacher runtime SHA-256 changed")
    if teacher_runtime_sha256 != manifest.get("cache_key_payload", {}).get(
        "teacher_runtime_sha256"
    ):
        raise ValueError("cache key is not bound to effective teacher runtime")

    artifact_path = args.artifact.expanduser().resolve()
    expected_artifact_sha256 = str(manifest["artifact_file_sha256"])
    if sha256_file(artifact_path) != expected_artifact_sha256:
        raise ValueError("Base teacher artifact changed before full replay")
    with np.load(artifact_path, allow_pickle=False) as source:
        stored_draws = source["affordance_draws"].copy()
    if sha256_file(artifact_path) != expected_artifact_sha256:
        raise ValueError("Base teacher artifact changed while loading replay draws")
    seed_rows = manifest["draw_seeds"]
    if len(seed_rows) != stored_draws.shape[0]:
        raise ValueError("stored draw/seed count mismatch during full replay")
    points = np.asarray(scene["points"], dtype=np.float32)
    row = {
        "xyz": np.ascontiguousarray(points[:, :3]),
        "feat": np.ascontiguousarray(scene["rgb01"]),
        "text": str(manifest["text"]),
    }
    mean, std = load_contact_stats(args.stats_file)
    regenerated = []
    rng_before = _rng_snapshot(torch)
    for seed_row in seed_rows:
        normalized = sample_contact_deterministic(
            model,
            diffusion,
            row,
            int(seed_row["initial_xT_seed"]),
            int(seed_row["reverse_noise_seed"]),
            args.device,
        )
        converted, transform = convert_prediction_draws(
            normalized,
            representation="normalized_contact",
            mean=mean,
            std=std,
        )
        regenerated.append(converted[0])
    first_seed_row = seed_rows[0]
    repeated_normalized = sample_contact_deterministic(
        model,
        diffusion,
        row,
        int(first_seed_row["initial_xT_seed"]),
        int(first_seed_row["reverse_noise_seed"]),
        args.device,
    )
    repeated_converted, repeated_transform = convert_prediction_draws(
        repeated_normalized,
        representation="normalized_contact",
        mean=mean,
        std=std,
    )
    prediction_files = {
        "train_audit": Path(
            str(quality_result["files"]["train_rollout_predictions"])
        ),
        "development": Path(
            str(quality_result["files"]["development_predictions"])
        ),
    }
    prediction_bundles = {}
    for partition, path in prediction_files.items():
        expected_hash_name = (
            "train_rollout_predictions"
            if partition == "train_audit"
            else "development_predictions"
        )
        if sha256_file(path) != quality_result["sha256"][expected_hash_name]:
            raise RuntimeError("quality prediction bundle changed before replay")
        with np.load(path, allow_pickle=False) as source:
            prediction_bundles[partition] = {
                "sample_ids": source["sample_ids"].astype(str).tolist(),
                "original_draws": source["original_draws"].copy(),
                "fewshot_draws": source["fewshot_draws"].copy(),
                "initial_noise_seeds": source["initial_noise_seeds"].copy(),
                "reverse_noise_seeds": source["reverse_noise_seeds"].copy(),
            }
        if sha256_file(path) != quality_result["sha256"][expected_hash_name]:
            raise RuntimeError("quality prediction bundle changed while loading")
    expected_replay_plan = _expected_quality_replay_plan(prediction_bundles)
    if quality_result.get("quality_replay_plan") != expected_replay_plan:
        raise ValueError("quality replay plan differs from sealed prediction bundles")
    quality_rows = {}

    def load_quality_row(sample_id):
        return load_quality_replay_row(args.dataset_root, args.split, sample_id)

    fewshot_replay_results = _replay_quality_role(
        replay_model=model,
        replay_diffusion=diffusion,
        role="fewshot",
        prediction_bundles=prediction_bundles,
        quality_rows=quality_rows,
        load_row=load_quality_row,
        sample_contact=sample_contact_deterministic,
        convert_prediction=convert_prediction_draws,
        mean=mean,
        std=std,
        device=args.device,
    )
    rng_after = _rng_snapshot(torch)
    if not _rng_equal(torch, rng_before, rng_after):
        raise RuntimeError("deterministic sampler changed caller RNG state")
    if not np.array_equal(regenerated[0], repeated_converted[0]):
        raise ValueError("draw-0 repeatability canary is not bitwise equal")
    regenerated_draws = np.stack(regenerated).astype(np.float32)
    if not np.array_equal(regenerated_draws, stored_draws):
        raise ValueError("one or more regenerated draws differ from stored draws")
    if transform.get("mean_sha256") != manifest.get("transform", {}).get(
        "mean_sha256"
    ) or transform.get("std_sha256") != manifest.get("transform", {}).get(
        "std_sha256"
    ):
        raise ValueError("contact stats tensors differ from export transform")
    if repeated_transform != transform:
        raise ValueError("draw-0 repeated transform differs from full replay")
    state_after = effective_model_state_hashes(model)
    if state_before != state_after:
        raise RuntimeError("effective Base teacher state changed during preflight")

    # Release the selected model before constructing the original baseline so
    # strict replay does not require both full encoders to fit on one GPU.
    model.to("cpu")
    del model
    del diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rng_before_original_construction = _rng_snapshot(torch)
    expected_original_sha256 = manifest["quality_gate"]["sha256"][
        "original_checkpoint"
    ]
    (
        original_model,
        original_diffusion,
        original_coverage,
        loaded_original_sha256,
    ) = create_model_strict(
        cfg,
        args.original_checkpoint,
        args.device,
        expected_sha256=expected_original_sha256,
    )
    _restore_rng_snapshot(torch, rng_before_original_construction)
    if loaded_original_sha256 != expected_original_sha256:
        raise RuntimeError("loaded original checkpoint snapshot hash changed")
    original_model.eval()
    for parameter in original_model.parameters():
        parameter.requires_grad_(False)
    if original_model.training or any(
        parameter.requires_grad for parameter in original_model.parameters()
    ):
        raise RuntimeError("reconstructed original CDM is not frozen/eval")
    original_state_before = effective_model_state_hashes(original_model)
    original_rng_before = _rng_snapshot(torch)
    original_replay_results = _replay_quality_role(
        replay_model=original_model,
        replay_diffusion=original_diffusion,
        role="original",
        prediction_bundles=prediction_bundles,
        quality_rows=quality_rows,
        load_row=load_quality_row,
        sample_contact=sample_contact_deterministic,
        convert_prediction=convert_prediction_draws,
        mean=mean,
        std=std,
        device=args.device,
    )
    original_rng_after = _rng_snapshot(torch)
    if not _rng_equal(torch, original_rng_before, original_rng_after):
        raise RuntimeError("original deterministic sampler changed caller RNG state")
    original_state_after = effective_model_state_hashes(original_model)
    if original_state_before != original_state_after:
        raise RuntimeError("original CDM state changed during quality replay")
    quality_replay_results = fewshot_replay_results + original_replay_results
    actual_quality_replayed_draws = _validate_quality_replay_coverage(
        quality_replay_results, expected_replay_plan
    )
    final_runtime_files = runtime_file_hashes()
    if final_runtime_files != current_runtime_files:
        raise RuntimeError("Base teacher runtime files changed during full replay")
    final_scene_weight_sha256 = sha256_file(scene_weight)
    if final_scene_weight_sha256 != runtime.get(
        "scene_model_pretrained_weight_sha256"
    ):
        raise RuntimeError("scene-model weight changed during full replay")

    return {
        "status": "PASS",
        "promotion_authorized": True,
        "environment": current_environment,
        "runtime_file_sha256": current_runtime_files,
        "resolved_config_sha256": canonical_json_sha256(resolved_config),
        "scene_model_pretrained_weight_sha256": final_scene_weight_sha256,
        "scene_model_pretrained_weight": str(scene_weight),
        "partial_checkpoint_coverage": checkpoint_coverage,
        "effective_model_state": state_before,
        "teacher_runtime_sha256": teacher_runtime_sha256,
        "all_draws_reproduced_bitwise": True,
        "reproduced_draw_count": int(stored_draws.shape[0]),
        "draw_zero_repeatability_bitwise": True,
        "quality_prediction_replay": quality_replay_results,
        "quality_prediction_models_replayed": ["fewshot", "original"],
        "quality_replay_plan": expected_replay_plan,
        "quality_prediction_total_replayed_draws": actual_quality_replayed_draws,
        "quality_prediction_all_reproduced_bitwise": True,
        "original_partial_checkpoint_coverage": original_coverage,
        "original_effective_model_state": original_state_before,
        "sampling_calls_preserved_caller_rng_state": True,
        "state_unchanged_during_preflight": True,
    }


def _validate_quality_for_manifest(
    args: argparse.Namespace, manifest: Mapping[str, object]
) -> Dict[str, object]:
    quality = manifest["quality_gate"]
    result = validate_v5r4_quality_chain(
        fewshot_checkpoint=args.fewshot_checkpoint,
        original_checkpoint=args.original_checkpoint,
        train_summary_file=args.train_summary,
        shortlist_status_file=args.shortlist_status,
        rollout_selection_file=args.rollout_selection,
        train_rollout_audit_file=args.train_rollout_audit,
        train_rollout_predictions_file=args.train_rollout_predictions,
        development_summary_file=args.development_summary,
        development_predictions_file=args.development_predictions,
        split_file=args.split,
        stats_file=args.stats_file,
        minimum_k_draws=int(manifest["draw_count"]),
        expected_diffusion_steps=int(
            manifest["cache_key_payload"]["diffusion_steps"]
        ),
        runtime_repo_root=REPO_ROOT,
        dataset_root=args.dataset_root,
    )
    if result["sha256"] != quality.get("sha256"):
        raise ValueError("current v5r4 quality files differ from artifact provenance")
    if result["schemas"] != quality.get("schemas"):
        raise ValueError("current v5r4 schemas differ from artifact provenance")
    if result["evidence_set_sha256"] != quality.get("evidence_set_sha256"):
        raise ValueError("current v5r4 evidence-set seal differs from artifact")
    for name in (
        "prediction_contracts",
        "independent_quality",
        "quality_replay_plan",
        "rollout_protocol_sha256",
    ):
        if result.get(name) != quality.get(name):
            raise ValueError("current v5r4 quality provenance differs: " + name)
    return result


def _assert_full_runtime_inputs_unchanged(
    *,
    manifest: Mapping[str, object],
    scene: Mapping[str, object],
    dataset_root: Path,
    replay: Mapping[str, object],
) -> None:
    runtime = manifest["runtime_provenance"]
    current_runtime_files = runtime_file_hashes()
    if (
        current_runtime_files != runtime.get("runtime_file_sha256")
        or current_runtime_files != replay.get("runtime_file_sha256")
    ):
        raise RuntimeError("Base teacher runtime changed before report commit")
    scene_weight = Path(
        str(replay["scene_model_pretrained_weight"])
    ).expanduser()
    if not scene_weight.is_absolute():
        scene_weight = (REPO_ROOT / scene_weight).resolve()
    else:
        scene_weight = scene_weight.resolve()
    recorded_scene_weight = Path(
        str(runtime["scene_model_pretrained_weight"])
    ).expanduser()
    if not recorded_scene_weight.is_absolute():
        recorded_scene_weight = (REPO_ROOT / recorded_scene_weight).resolve()
    else:
        recorded_scene_weight = recorded_scene_weight.resolve()
    if recorded_scene_weight != scene_weight:
        raise RuntimeError("scene-model weight path changed before report commit")
    expected_scene_weight_sha256 = runtime.get(
        "scene_model_pretrained_weight_sha256"
    )
    if (
        sha256_file(scene_weight) != expected_scene_weight_sha256
        or replay.get("scene_model_pretrained_weight_sha256")
        != expected_scene_weight_sha256
    ):
        raise RuntimeError("scene-model weight changed before report commit")
    final_scene = load_scene_contract(dataset_root, str(manifest["scene_id"]))
    if final_scene["scene_sha256"] != scene["scene_sha256"]:
        raise RuntimeError("Base teacher scene changed before report commit")


def main() -> None:
    args = parse_args()
    if args.report.expanduser().resolve().exists():
        raise FileExistsError(
            "refusing to overwrite an existing Base teacher preflight report: "
            + str(args.report.expanduser().resolve())
        )
    artifact_result = validate_base_teacher_artifact(
        manifest_file=args.manifest,
        artifact_file=args.artifact,
    )
    manifest_path = args.manifest.expanduser().resolve()
    artifact_path = args.artifact.expanduser().resolve()
    if sha256_file(manifest_path) != artifact_result["manifest_sha256"]:
        raise RuntimeError("Base teacher manifest changed after integrity validation")
    manifest = load_json(args.manifest)
    if sha256_file(manifest_path) != artifact_result["manifest_sha256"]:
        raise RuntimeError("Base teacher manifest changed while being loaded")
    quality_result = _validate_quality_for_manifest(args, manifest)

    scene = load_scene_contract(args.dataset_root, str(manifest["scene_id"]))
    if scene["scene_sha256"] != manifest.get("scene_sha256"):
        raise ValueError("current scene differs from Base teacher scene")
    with np.load(args.artifact.expanduser().resolve(), allow_pickle=False) as source:
        exact_scene_arrays = {
            "xyz": scene["xyz"],
            "rgb01": scene["rgb01"],
            "instance_ids": scene["instance_ids"],
            "source_indices": scene["source_indices"],
        }
        for name, expected in exact_scene_arrays.items():
            if not np.array_equal(source[name], expected):
                raise ValueError("artifact/current scene differs: " + name)

    common_report = {
        "artifact": artifact_result,
        "manifest_sha256": artifact_result["manifest_sha256"],
        "artifact_id": artifact_result["artifact_id"],
        "artifact_sha256": artifact_result["artifact_sha256"],
        "cache_key": artifact_result["cache_key"],
        "quality_chain": {
            "status": quality_result["status"],
            "schemas": quality_result["schemas"],
            "sha256": quality_result["sha256"],
            "evidence_set_sha256": quality_result["evidence_set_sha256"],
        },
        "scene_sha256": scene["scene_sha256"],
        "checks": {
            "artifact_integrity_pass": True,
            "quality_hash_chain_pass": True,
            "scene_point_order_exact": True,
            "distance_kernel_not_reapplied": (
                manifest.get("transform", {}).get("distance_kernel_reapplied")
                is False
            ),
            "artifact_contains_no_history_conditioning": (
                manifest.get("conditioning", {}).get("history_conditioned")
                is False
            ),
        },
    }
    if not all(common_report["checks"].values()):
        raise AssertionError("Base teacher preflight common check failed")

    if args.mode == "integrity-only":
        report = {
            "schema": "history_affordance_v2_base_teacher_preflight_v1",
            "status": "INTEGRITY_ONLY",
            "promotion_authorized": False,
            "mode": args.mode,
            **common_report,
        }
        _assert_files_unchanged(
            manifest_file=manifest_path,
            artifact_file=artifact_path,
            manifest_sha256=artifact_result["manifest_sha256"],
            artifact_sha256=artifact_result["artifact_sha256"],
            quality_result=quality_result,
        )
        atomic_write_json(args.report, report)
        print("[INTEGRITY_ONLY] artifact transport integrity passed")
        print("[BLOCKED] promotion requires --mode full on the export runtime")
        raise SystemExit(3)

    replay = _full_replay(
        args=args,
        manifest=manifest,
        scene=scene,
        quality_result=quality_result,
    )
    report = {
        "schema": "history_affordance_v2_base_teacher_preflight_v1",
        "status": "PASS",
        "promotion_authorized": True,
        "mode": args.mode,
        **common_report,
        "full_replay": replay,
    }
    final_scene = load_scene_contract(
        args.dataset_root, str(manifest["scene_id"])
    )
    if final_scene["scene_sha256"] != scene["scene_sha256"]:
        raise RuntimeError("current scene changed during full preflight")
    final_quality_result = _validate_quality_for_manifest(args, manifest)
    if final_quality_result != quality_result:
        raise RuntimeError("quality evidence changed during full preflight")
    _assert_files_unchanged(
        manifest_file=manifest_path,
        artifact_file=artifact_path,
        manifest_sha256=artifact_result["manifest_sha256"],
        artifact_sha256=artifact_result["artifact_sha256"],
        quality_result=quality_result,
    )
    _assert_full_runtime_inputs_unchanged(
        manifest=manifest,
        scene=scene,
        dataset_root=args.dataset_root,
        replay=replay,
    )
    atomic_write_json(args.report, report)
    print("[PASS] frozen v2 Base teacher full preflight")
    print("[OK] cache key: " + artifact_result["cache_key"])
    print("[OK] artifact SHA-256: " + artifact_result["artifact_sha256"])
    print("[OK] report: " + str(args.report.expanduser().resolve()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate one hash-bound v2 Base affordance teacher artifact on AMDM.

Install this file and ``base_teacher_contract.py`` into ``~/AMDM/prepare``.
The command refuses to load the CDM unless the complete v5r4 train/audit/
development evidence chain is PASS and bound to the exact checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_DIR = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from base_teacher_contract import (  # noqa: E402
    PROMPT_POLICY_ID,
    build_cache_key_payload,
    canonical_json_sha256,
    convert_prediction_draws,
    derive_draw_seeds,
    load_contact_stats,
    load_scene_contract,
    sha256_file,
    validate_v5r4_quality_chain,
    write_base_teacher_artifact,
)


EXCLUDED_CHECKPOINT_PREFIXES = (
    "scene_model.",
    "text_model.",
    "clip_model.",
    "bert_model.",
)

# ``configs/default.yaml`` derives exp_dir from Hydra's ``${now:...}``
# resolver. That value is operational logging metadata, not part of the CDM
# teacher, and differs between the export and validation processes. Pin all
# derived output paths before hashing the resolved model configuration so a
# same-runtime replay compares scientific configuration rather than wall time.
GATE0_RUNTIME_EXP_DIR = "outputs/gate0_base_teacher_runtime"


def pin_gate0_runtime_output_paths(cfg):
    """Remove Hydra wall-clock output paths from the Gate 0 config identity."""

    cfg.exp_dir = GATE0_RUNTIME_EXP_DIR
    cfg.log_dir = GATE0_RUNTIME_EXP_DIR + "/log"
    cfg.ckpt_dir = GATE0_RUNTIME_EXP_DIR + "/ckpt"
    cfg.eval_dir = GATE0_RUNTIME_EXP_DIR + "/eval"
    return cfg


def resolved_gate0_cdm_config(cfg) -> Dict[str, object]:
    """Resolve and validate the timestamp-free Gate 0 CDM configuration."""

    from omegaconf import OmegaConf

    pin_gate0_runtime_output_paths(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved CDM configuration is not a mapping")
    expected_paths = {
        "exp_dir": GATE0_RUNTIME_EXP_DIR,
        "log_dir": GATE0_RUNTIME_EXP_DIR + "/log",
        "ckpt_dir": GATE0_RUNTIME_EXP_DIR + "/ckpt",
        "eval_dir": GATE0_RUNTIME_EXP_DIR + "/eval",
    }
    actual_paths = {name: resolved.get(name) for name in expected_paths}
    if actual_paths != expected_paths:
        raise ValueError(
            "Gate 0 runtime output paths were not pinned: " + str(actual_paths)
        )
    return resolved


def _tensor_bytes(tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    try:
        return value.numpy().tobytes(order="C")
    except TypeError:
        # NumPy cannot represent every torch dtype (for example bfloat16).
        return value.view(__import__("torch").uint8).numpy().tobytes(order="C")


def _state_hash(
    state: Mapping[str, object], predicate: Callable[[str], bool]
) -> Tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name in sorted(state):
        if not predicate(name):
            continue
        tensor = state[name]
        header = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(header)
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
        digest.update(b"\0")
        count += 1
    if count == 0:
        raise RuntimeError("effective model state group is empty")
    return digest.hexdigest(), count


def effective_model_state_hashes(model) -> Dict[str, object]:
    state = model.state_dict()

    def scene(name: str) -> bool:
        return name.startswith("scene_model.")

    def text(name: str) -> bool:
        return name.startswith(("text_model.", "clip_model.", "bert_model."))

    def core(name: str) -> bool:
        return not scene(name) and not text(name)

    full_hash, full_count = _state_hash(state, lambda _: True)
    core_hash, core_count = _state_hash(state, core)
    scene_hash, scene_count = _state_hash(state, scene)
    text_hash, text_count = _state_hash(state, text)
    return {
        "full_state_sha256": full_hash,
        "full_state_tensor_count": full_count,
        "cdm_core_state_sha256": core_hash,
        "cdm_core_tensor_count": core_count,
        "scene_model_state_sha256": scene_hash,
        "scene_model_tensor_count": scene_count,
        "text_model_state_sha256": text_hash,
        "text_model_tensor_count": text_count,
    }


def _load_partial_checkpoint_snapshot(
    checkpoint_file: Path, *, expected_sha256: Optional[str] = None
):
    import torch

    checkpoint_file = Path(checkpoint_file).expanduser().resolve()
    checkpoint_bytes = checkpoint_file.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError("checkpoint snapshot SHA-256 differs from provenance")
    try:
        loaded = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except TypeError:
        # PyTorch before weights_only support. The checkpoint is still bound by
        # the validated v5r4 SHA before this compatibility path is reached.
        loaded = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    if sha256_file(checkpoint_file) != checkpoint_sha256:
        raise RuntimeError("checkpoint path changed while snapshotting bytes")
    if not isinstance(loaded, Mapping) or not loaded:
        raise TypeError("few-shot checkpoint is not a non-empty state mapping")
    normalized = {}
    for raw_name, value in loaded.items():
        raw_name = str(raw_name)
        name = raw_name[7:] if raw_name.startswith("module.") else raw_name
        if name in normalized:
            raise ValueError(
                "few-shot checkpoint has duplicate key after module-prefix "
                "normalization: "
                + name
            )
        normalized[name] = value
    if not all(isinstance(value, torch.Tensor) for value in normalized.values()):
        raise TypeError("few-shot checkpoint contains non-tensor state values")
    excluded_present = sorted(
        name for name in normalized if name.startswith(EXCLUDED_CHECKPOINT_PREFIXES)
    )
    if excluded_present:
        raise ValueError(
            "merged v5r4 checkpoint unexpectedly embeds frozen encoder state: "
            + str(excluded_present)
        )
    return normalized, checkpoint_sha256


def _partial_checkpoint_coverage_from_state(
    model, normalized: Mapping[str, object]
) -> Dict[str, object]:
    import torch

    model_state = model.state_dict()
    unexpected = sorted(set(normalized).difference(model_state))
    if unexpected:
        raise ValueError("few-shot checkpoint has unexpected keys: " + str(unexpected))
    missing = sorted(set(model_state).difference(normalized))
    illegal_missing = [
        name
        for name in missing
        if not name.startswith(EXCLUDED_CHECKPOINT_PREFIXES)
    ]
    if illegal_missing:
        raise ValueError(
            "few-shot checkpoint misses non-frozen CDM state: "
            + str(illegal_missing)
        )
    mismatched = []
    for name, checkpoint_value in normalized.items():
        model_value = model_state[name].detach().cpu()
        checkpoint_value = checkpoint_value.detach().cpu()
        if (
            model_value.shape != checkpoint_value.shape
            or model_value.dtype != checkpoint_value.dtype
            or not torch.equal(model_value, checkpoint_value)
        ):
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            "loaded effective model differs from checkpoint: " + str(mismatched)
        )
    return {
        "status": "PASS",
        "checkpoint_tensor_count": len(normalized),
        "effective_model_tensor_count": len(model_state),
        "allowed_missing_frozen_tensor_count": len(missing),
        "allowed_missing_prefixes": list(EXCLUDED_CHECKPOINT_PREFIXES),
        "unexpected_keys": [],
        "illegal_missing_keys": [],
        "loaded_tensor_values_exact": True,
    }


def validate_partial_checkpoint_coverage(
    model, checkpoint_file: Path, *, expected_sha256: Optional[str] = None
) -> Dict[str, object]:
    normalized, _ = _load_partial_checkpoint_snapshot(
        checkpoint_file, expected_sha256=expected_sha256
    )
    return _partial_checkpoint_coverage_from_state(model, normalized)


def create_model_strict(
    cfg,
    checkpoint_file: Path,
    device: str,
    *,
    expected_sha256: Optional[str] = None,
):
    """Construct CDM and load only the exact validated partial state mapping."""

    from models.base import create_model_and_diffusion

    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)
    normalized, loaded_checkpoint_sha256 = _load_partial_checkpoint_snapshot(
        checkpoint_file, expected_sha256=expected_sha256
    )
    current = model.state_dict()
    unexpected = sorted(set(normalized).difference(current))
    if unexpected:
        raise ValueError("few-shot checkpoint has unexpected keys: " + str(unexpected))
    missing = sorted(set(current).difference(normalized))
    illegal_missing = [
        name for name in missing if not name.startswith(EXCLUDED_CHECKPOINT_PREFIXES)
    ]
    if illegal_missing:
        raise ValueError(
            "few-shot checkpoint misses non-frozen CDM state: "
            + str(illegal_missing)
        )
    for name, value in normalized.items():
        if current[name].shape != value.shape or current[name].dtype != value.dtype:
            raise ValueError("few-shot checkpoint tensor metadata mismatch: " + name)
        current[name] = value
    model.load_state_dict(current, strict=True)
    model.eval()
    coverage = _partial_checkpoint_coverage_from_state(model, normalized)
    return model, diffusion, coverage, loaded_checkpoint_sha256


def runtime_file_hashes() -> Dict[str, str]:
    required_files = (
        "prepare/base_teacher_contract.py",
        "prepare/export_base_teacher.py",
        "prepare/validate_base_teacher.py",
        "prepare/evaluate_fewshot_cdm.py",
        "prepare/audit_fewshot_cdm_train_rollout.py",
        "prepare/fewshot_cdm_common.py",
        "prepare/fewshot_cdm_lora.py",
        "prepare/fewshot_cdm_rollout_cache.py",
        "prepare/train_fewshot_cdm.py",
        "prepare/fewshot_cdm_v5_semantics.py",
        "utils/misc.py",
        "utils/registry.py",
        "utils/training.py",
        "configs/default.yaml",
        "configs/model/cdm.yaml",
        "configs/task/contact_gen.yaml",
    )
    relative_files = set(required_files)
    for directory in (REPO_ROOT / "models", REPO_ROOT / "diffusion"):
        if not directory.is_dir():
            raise FileNotFoundError("Base teacher runtime directory: " + str(directory))
        relative_files.update(
            str(path.relative_to(REPO_ROOT))
            for path in directory.rglob("*.py")
            if path.is_file()
        )
    files = {name: REPO_ROOT / name for name in sorted(relative_files)}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Base teacher runtime files missing: " + str(missing))
    hashes = {name: sha256_file(path) for name, path in files.items()}
    try:
        import pointops_cuda
    except ImportError as exc:
        raise RuntimeError(
            "the exact pointops_cuda extension must be importable before "
            "Base teacher export or replay"
        ) from exc
    pointops_origin = getattr(pointops_cuda, "__file__", None)
    if not pointops_origin:
        raise RuntimeError("pointops_cuda has no hashable binary origin")
    pointops_binary = Path(str(pointops_origin)).expanduser().resolve()
    if not pointops_binary.is_file():
        raise FileNotFoundError("pointops_cuda binary: " + str(pointops_binary))
    hashes["binary/pointops_cuda"] = sha256_file(pointops_binary)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--text", required=True)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-policy-id", default=PROMPT_POLICY_ID)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--k-draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_type = args.device.split(":", 1)[0]
    if device_type != "cuda":
        raise ValueError(
            "strict Base teacher export requires CUDA because pointops_cuda "
            "has no CPU execution path"
        )
    if args.diffusion_steps != 500:
        raise ValueError("strict Base teacher v1 requires exactly 500 diffusion steps")
    if args.k_draws != 5:
        raise ValueError("strict Base teacher v1 requires exactly five draws")
    if args.seed != 20260815:
        raise ValueError("strict Base teacher v1 requires --seed 20260815")

    dataset_root = args.dataset_root.expanduser().resolve()
    fewshot_checkpoint = args.fewshot_checkpoint.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()

    # This runs before importing or constructing any torch model.  A weak or
    # stale checkpoint cannot consume GPU time or create a teacher artifact.
    quality_gate = validate_v5r4_quality_chain(
        fewshot_checkpoint=fewshot_checkpoint,
        original_checkpoint=original_checkpoint,
        train_summary_file=args.train_summary,
        shortlist_status_file=args.shortlist_status,
        rollout_selection_file=args.rollout_selection,
        train_rollout_audit_file=args.train_rollout_audit,
        train_rollout_predictions_file=args.train_rollout_predictions,
        development_summary_file=args.development_summary,
        development_predictions_file=args.development_predictions,
        split_file=args.split,
        stats_file=stats_file,
        minimum_k_draws=args.k_draws,
        expected_diffusion_steps=args.diffusion_steps,
        runtime_repo_root=REPO_ROOT,
        dataset_root=dataset_root,
    )
    scene = load_scene_contract(dataset_root, args.scene_id)
    mean, std = load_contact_stats(stats_file)

    # Import the validated v5r4 sampler only after every CPU-side contract has
    # passed.  The sampler returns normalized contact, not distance.
    from evaluate_fewshot_cdm import sample_contact_deterministic  # noqa: E402
    from train_fewshot_cdm import (  # noqa: E402
        compose_cdm_config,
        configure_reproducibility,
    )
    import numpy
    import torch

    torch_device = torch.device(args.device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Base teacher export but unavailable")
    configure_reproducibility(args.seed)
    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    pin_gate0_runtime_output_paths(cfg)
    resolved_config = resolved_gate0_cdm_config(cfg)
    resolved_config_sha256 = canonical_json_sha256(resolved_config)
    expected_checkpoint_sha256 = quality_gate["sha256"]["fewshot_checkpoint"]
    model, diffusion, checkpoint_coverage, loaded_checkpoint_sha256 = (
        create_model_strict(
            cfg,
            fewshot_checkpoint,
            args.device,
            expected_sha256=expected_checkpoint_sha256,
        )
    )
    if loaded_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("loaded few-shot checkpoint snapshot hash changed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Base teacher model did not freeze completely")
    state_before = effective_model_state_hashes(model)

    if resolved_gate0_cdm_config(cfg) != resolved_config:
        raise RuntimeError("CDM configuration changed during model construction")
    scene_weight_value = cfg.model.scene_model.pretrained_weight
    scene_weight_file = Path(str(scene_weight_value)).expanduser()
    if not scene_weight_file.is_absolute():
        scene_weight_file = (REPO_ROOT / scene_weight_file).resolve()
    if not scene_weight_file.is_file():
        raise FileNotFoundError(
            "effective scene-model pretrained weight: " + str(scene_weight_file)
        )
    scene_weight_sha256 = sha256_file(scene_weight_file)
    runtime_files = runtime_file_hashes()
    sampling_environment = {
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
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
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else "cpu"
        ),
        "device_capability": (
            list(torch.cuda.get_device_capability(torch.device(args.device)))
            if torch.device(args.device).type == "cuda"
            else None
        ),
    }
    teacher_runtime_identity = {
        "schema": "history_affordance_v2_teacher_runtime_identity_v1",
        "fewshot_checkpoint_sha256": quality_gate["sha256"][
            "fewshot_checkpoint"
        ],
        "contact_stats_sha256": quality_gate["sha256"]["stats"],
        "effective_model_state": state_before,
        "partial_checkpoint_coverage": checkpoint_coverage,
        "scene_model_pretrained_weight_sha256": scene_weight_sha256,
        "text_model_name": str(cfg.model.text_model.version),
        "resolved_config_sha256": resolved_config_sha256,
        "runtime_file_set_sha256": canonical_json_sha256(runtime_files),
        "sampling_environment": sampling_environment,
    }
    teacher_runtime_sha256 = canonical_json_sha256(teacher_runtime_identity)
    cache_payload = build_cache_key_payload(
        scene_sha256=str(scene["scene_sha256"]),
        prompt_id=args.prompt_id,
        text=args.text,
        checkpoint_sha256=str(
            quality_gate["sha256"]["fewshot_checkpoint"]
        ),
        stats_sha256=str(quality_gate["sha256"]["stats"]),
        teacher_runtime_sha256=teacher_runtime_sha256,
        diffusion_steps=args.diffusion_steps,
        k_draws=args.k_draws,
        base_seed=args.seed,
        prompt_policy_id=args.prompt_policy_id,
    )
    cache_key = canonical_json_sha256(cache_payload)
    draw_seeds = [
        derive_draw_seeds(
            cache_key=cache_key,
            base_seed=args.seed,
            draw_index=draw_index,
        )
        for draw_index in range(args.k_draws)
    ]

    points = np.asarray(scene["points"], dtype=np.float32)
    row = {
        "xyz": np.ascontiguousarray(points[:, :3]),
        "feat": np.ascontiguousarray(scene["rgb01"]),
        "text": args.text,
    }
    normalized_draws = []
    for initial_seed, reverse_seed in draw_seeds:
        normalized_draws.append(
            sample_contact_deterministic(
                model,
                diffusion,
                row,
                int(initial_seed),
                int(reverse_seed),
                args.device,
            )
        )
    normalized = np.stack(normalized_draws).astype(np.float32)
    affordance_draws, transform = convert_prediction_draws(
        normalized,
        representation="normalized_contact",
        mean=mean,
        std=std,
    )
    transform = {
        **transform,
        "source_model_output": "normalized_contact",
        "canonical_output": "denormalized_clipped_affordance",
        "distance_kernel_reapplied": False,
    }
    state_after = effective_model_state_hashes(model)
    if state_before != state_after:
        raise RuntimeError("effective Base teacher state changed during sampling")
    final_runtime_files = runtime_file_hashes()
    if final_runtime_files != runtime_files:
        raise RuntimeError("Base teacher runtime files changed during export")
    if sha256_file(scene_weight_file) != scene_weight_sha256:
        raise RuntimeError("scene-model pretrained weight changed during export")
    final_scene = load_scene_contract(dataset_root, args.scene_id)
    if final_scene["scene_sha256"] != scene["scene_sha256"]:
        raise RuntimeError("Base teacher scene changed during export")
    for name, path in quality_gate["files"].items():
        if sha256_file(Path(str(path))) != quality_gate["sha256"][name]:
            raise RuntimeError("quality evidence changed during export: " + name)

    runtime_provenance = {
        "status": "PASS",
        **sampling_environment,
        "model_eval_mode": model.training is False,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "state_unchanged_during_sampling": True,
        "effective_model_state": state_before,
        "partial_checkpoint_coverage": checkpoint_coverage,
        "scene_model_pretrained_weight": str(scene_weight_file),
        "scene_model_pretrained_weight_sha256": scene_weight_sha256,
        "text_model_name": str(cfg.model.text_model.version),
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "runtime_file_sha256": runtime_files,
        "teacher_runtime_identity": teacher_runtime_identity,
        "teacher_runtime_sha256": teacher_runtime_sha256,
    }

    manifest_file, artifact_file = write_base_teacher_artifact(
        output_dir=args.output_root.expanduser().resolve() / cache_key,
        scene=scene,
        prompt_id=args.prompt_id,
        text=args.text,
        affordance_draws=affordance_draws,
        transform=transform,
        quality_gate=quality_gate,
        runtime_provenance=runtime_provenance,
        cache_key_payload=cache_payload,
        draw_seeds=draw_seeds,
        source_representation="normalized_contact",
    )
    print("[STAGED] frozen v2 Base teacher artifact; full preflight still required")
    print("[OK] cache key: " + cache_key)
    print("[OK] manifest: " + str(manifest_file))
    print("[OK] artifact: " + str(artifact_file))


if __name__ == "__main__":
    main()

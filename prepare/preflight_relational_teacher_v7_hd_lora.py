#!/usr/bin/env python3
"""CUDA zero-init and gradient preflight for relational Teacher-v7 High Desk LoRA.

This gate reads only the two training scenes.  The development scene remains
metadata-only until a train-selected checkpoint has been locked.  The dense
probe in each training scene is required to be one of the newly added
``hc_hd`` High-Desk motions, rather than a legacy v6 replay row.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from fewshot_cdm_lora import (  # noqa: E402
    install_lora,
    lora_metadata,
    lora_named_parameters,
    lora_parameter_energy,
    set_frozen_base_eval_lora_train,
)
from relational_teacher_v6_contract import (  # noqa: E402
    FORWARD_INPUT_KEYS,
    PROMPTS,
    atomic_write_json,
    read_json,
    sha256_file,
)
from relational_teacher_v6_semantics import (  # noqa: E402
    all_sittable_semantic_loss,
    frozen_teacher_preservation_loss,
    paired_prompt_invariance_loss,
)
from relational_teacher_v7_hd_preflight_contract import (  # noqa: E402
    DATASET_SCHEMA,
    EXPECTED_DEVELOPMENT_SCENES,
    EXPECTED_TRAIN_SCENES,
    SCHEMA,
    canonical_sha256,
    select_train_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-evidence-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compose_cdm_config(diffusion_steps: int, device: str):
    from hydra import compose, initialize_config_dir
    from utils.misc import compute_repr_dimesion

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(
            config_name="default",
            overrides=[
                "task=contact_gen",
                "model=cdm",
                "model.arch=Perceiver",
                "task.dataset.sigma=0.8",
                f"diffusion.steps={diffusion_steps}",
            ],
        )
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    cfg.gpu = int(device.split(":", 1)[1]) if device.startswith("cuda:") else None
    return cfg


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        mean = source["mean"].astype(np.float32)
        std = source["std"].astype(np.float32)
    if mean.shape != (1, 6) or std.shape != (1, 6):
        raise ValueError("Teacher-v7 contact statistics must be [1,6]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("contact statistics contain NaN/Inf")
    if np.any(std <= 0.0):
        raise ValueError("contact statistics contain non-positive std")
    return mean, std


def predict_xstart(
    model,
    diffusion,
    x_start: torch.Tensor,
    timestep: torch.Tensor,
    kwargs: Mapping[str, object],
    noise: torch.Tensor,
) -> torch.Tensor:
    if set(kwargs) != set(FORWARD_INPUT_KEYS):
        raise AssertionError("Teacher-v7 forward keys changed")
    if getattr(diffusion.model_mean_type, "name", "") != "START_X":
        raise RuntimeError("Teacher-v7 requires START_X diffusion prediction")
    x_t = diffusion.q_sample(x_start, timestep, noise=noise)
    wrapped = (
        diffusion._wrap_model(model)
        if hasattr(diffusion, "_wrap_model")
        else model
    )
    prediction = wrapped(
        x_t,
        diffusion._scale_timesteps(timestep),
        **dict(kwargs),
    )
    if prediction.shape != x_start.shape:
        raise ValueError("CDM prediction shape changed")
    return prediction


def build_batch(
    probes: Sequence[Mapping[str, object]],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> Dict[str, object]:
    values = []
    for probe in probes:
        for prompt_id in sorted(PROMPTS):
            values.append((probe, prompt_id, PROMPTS[prompt_id]))
    points = np.stack(
        [np.asarray(value[0]["points"], dtype=np.float32) for value in values]
    )
    affordance = np.stack(
        [np.asarray(value[0]["affordance"], dtype=np.float32) for value in values]
    )
    normalized = (affordance - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)
    return {
        "x": torch.from_numpy(normalized.astype(np.float32)).to(device),
        "xyz": torch.from_numpy(points[:, :, :3]).to(device).contiguous(),
        "feat": torch.from_numpy(points[:, :, 3:6] / 255.0).to(device).contiguous(),
        "text": [value[2] for value in values],
        "prompt_ids": [value[1] for value in values],
        "scene_ids": [str(value[0]["scene_id"]) for value in values],
        "instance_ids": torch.from_numpy(
            np.stack(
                [
                    np.asarray(value[0]["instance_ids"], dtype=np.int64)
                    for value in values
                ]
            )
        ).to(device),
        "category_ids": torch.from_numpy(
            np.stack(
                [
                    np.asarray(value[0]["category_ids"], dtype=np.int64)
                    for value in values
                ]
            )
        ).to(device),
    }


def gradient_summary(
    loss: torch.Tensor,
    named_parameters: Mapping[str, torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> Dict[str, float]:
    parameters = list(named_parameters.values())
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total_sq = 0.0
    a_sq = 0.0
    b_sq = 0.0
    finite = True
    used = 0
    for (name, _), gradient in zip(named_parameters.items(), gradients):
        if gradient is None:
            continue
        used += 1
        finite = finite and bool(torch.isfinite(gradient).all().item())
        value = float(gradient.detach().float().square().sum().item())
        total_sq += value
        if name.endswith("lora_A"):
            a_sq += value
        elif name.endswith("lora_B"):
            b_sq += value
    return {
        "loss": float(loss.detach().item()),
        "gradient_l2": math.sqrt(total_sq),
        "lora_A_gradient_l2": math.sqrt(a_sq),
        "lora_B_gradient_l2": math.sqrt(b_sq),
        "parameters_with_gradient": used,
        "all_finite": finite,
    }


def require_positive_gradient(name: str, summary: Mapping[str, object]) -> None:
    value = float(summary["gradient_l2"])
    if (
        not math.isfinite(value)
        or value <= 0.0
        or summary.get("all_finite") is not True
    ):
        raise RuntimeError(f"{name} does not provide a finite nonzero LoRA gradient")


def perturb_lora_output(
    named_parameters: Mapping[str, torch.nn.Parameter], amount: float
) -> Dict[str, torch.Tensor]:
    saved = {name: value.detach().clone() for name, value in named_parameters.items()}
    with torch.no_grad():
        for name, value in named_parameters.items():
            if name.endswith("lora_B"):
                value.add_(amount)
    return saved


def restore_lora(
    named_parameters: Mapping[str, torch.nn.Parameter], saved: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, value in named_parameters.items():
            value.copy_(saved[name])


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Teacher-v7 LoRA preflight requires CUDA")
    if args.diffusion_steps != 500:
        raise ValueError("Teacher-v7 preflight is sealed to 500 diffusion steps")
    configure_reproducibility(args.seed)

    dataset_root = args.dataset_root.expanduser().resolve()
    index_file = args.index.expanduser().resolve()
    stats_file = args.stats_file.expanduser().resolve()
    original_checkpoint = args.original_checkpoint.expanduser().resolve()
    v5_checkpoint = args.v5_checkpoint.expanduser().resolve()
    evidence_file = args.v5_evidence_report.expanduser().resolve()
    report_file = args.report.expanduser().resolve()
    if report_file.exists():
        raise FileExistsError(
            "refusing to overwrite preflight report: " + str(report_file)
        )
    for path in (
        index_file,
        stats_file,
        original_checkpoint,
        v5_checkpoint,
        evidence_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    index = read_json(index_file)
    if (
        index.get("schema") != DATASET_SCHEMA
        or index.get("status") != "DENSE_DATASET_PASS"
        or index.get("authorization") != "teacher_lora_v7_cuda_preflight_only"
        or index.get("split_scene_counts") != {"train": 2, "development": 1}
        or index.get("split_row_counts")
        != {"development": 48, "train": 96}
        or index.get("num_dense_motions") != 72
        or index.get("num_rows") != 144
        or index.get("heldout_dense_arrays_read_during_build") is not False
        or index.get("relation_or_distance_used_as_forward_input") is not False
        or index.get("target_instance_gt_is_supervision_only") is not True
        or index.get("teacher_lora_v7_training_authorized") is not False
    ):
        raise ValueError("Teacher-v7 three-scene dataset index is not sealed")
    records = index.get("scenes")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("Teacher-v7 index must contain exactly three scenes")
    train_records = [record for record in records if record.get("split") == "train"]
    development_records = [
        record for record in records if record.get("split") == "development"
    ]
    train_ids = tuple(sorted(str(record["scene_id"]) for record in train_records))
    development_ids = tuple(
        sorted(str(record["scene_id"]) for record in development_records)
    )
    if (
        train_ids != EXPECTED_TRAIN_SCENES
        or development_ids != EXPECTED_DEVELOPMENT_SCENES
    ):
        raise ValueError("Teacher-v7 train/development scene split changed")

    evidence = read_json(evidence_file)
    quality = evidence.get("quality_chain")
    if (
        evidence.get("schema") != "history_affordance_v2_gate0a_evidence_report_v1"
        or evidence.get("status") != "PASS"
        or evidence.get("may_export_base_teacher") is not True
        or not isinstance(quality, dict)
    ):
        raise ValueError("v5r4 evidence report is not Gate 0A PASS")
    quality_hashes = quality.get("sha256")
    if not isinstance(quality_hashes, dict):
        raise ValueError("v5r4 evidence hashes are absent")
    if quality_hashes.get("fewshot_checkpoint") != sha256_file(v5_checkpoint):
        raise ValueError("v5r4 checkpoint differs from Gate 0A evidence")
    if quality_hashes.get("original_checkpoint") != sha256_file(original_checkpoint):
        raise ValueError("original checkpoint differs from Gate 0A evidence")

    mean, std = load_stats(stats_file)
    probes = [select_train_probe(dataset_root, record) for record in train_records]
    batch = build_batch(probes, mean, std, args.device)
    if batch["scene_ids"] != [
        "room_0101",
        "room_0101",
        "room_0102",
        "room_0102",
    ]:
        raise AssertionError("train probe order changed")
    if batch["prompt_ids"] != ["sit_watch_v1", "sit_write_v1"] * 2:
        raise AssertionError("prompt-pair order changed")

    cfg = compose_cdm_config(args.diffusion_steps, args.device)
    from models.base import create_model_and_diffusion
    from utils.training import load_ckpt

    model, diffusion = create_model_and_diffusion(cfg, device=args.device)
    model.to(args.device)
    load_ckpt(model, str(original_checkpoint))
    load_ckpt(model, str(v5_checkpoint))
    model.eval()
    timesteps = torch.tensor(
        [50, 350, 50, 350], dtype=torch.long, device=args.device
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1000)
    noise = torch.randn(batch["x"].shape, generator=generator).to(args.device)
    kwargs = {
        "c_pc_xyz": batch["xyz"],
        "c_pc_feat": batch["feat"],
        "c_text": batch["text"],
    }
    with torch.no_grad():
        v5_prediction = predict_xstart(
            model, diffusion, batch["x"], timesteps, kwargs, noise
        )

    module_names = install_lora(
        model, args.lora_rank, args.lora_alpha, dropout=0.0
    )
    if len(module_names) != 31:
        raise AssertionError(
            f"Teacher-v7 expected 31 LoRA modules, got {len(module_names)}"
        )
    set_frozen_base_eval_lora_train(model)
    named_lora = lora_named_parameters(model)
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise AssertionError("a frozen CDM parameter became trainable")
    with torch.no_grad():
        zero_prediction = predict_xstart(
            model, diffusion, batch["x"], timesteps, kwargs, noise
        )
    parity_max_abs = float((zero_prediction - v5_prediction).abs().max().item())
    if not torch.equal(zero_prediction, v5_prediction):
        raise AssertionError("zero-init v6 LoRA changed the v5r4 Teacher output")

    prediction = predict_xstart(
        model, diffusion, batch["x"], timesteps, kwargs, noise
    )
    mean_tensor = torch.from_numpy(mean.reshape(1, 1, 6)).to(args.device)
    std_tensor = torch.from_numpy(std.reshape(1, 1, 6)).to(args.device)
    dense_loss = (prediction - batch["x"]).square().mean()
    semantic = all_sittable_semantic_loss(
        prediction,
        batch["instance_ids"],
        batch["category_ids"],
        mean_tensor,
        std_tensor,
    )
    watch = prediction[0::2]
    write = prediction[1::2]
    invariance_loss = paired_prompt_invariance_loss(
        watch, write, batch["category_ids"][0::2]
    )
    zero_preservation = frozen_teacher_preservation_loss(
        prediction, v5_prediction, batch["category_ids"]
    )
    zero_lora_energy = lora_parameter_energy(model)
    zero_preservation_value = float(zero_preservation.detach().item())
    zero_lora_energy_value = float(zero_lora_energy.detach().item())
    components = {
        "dense_replay": gradient_summary(
            dense_loss, named_lora, retain_graph=True
        ),
        "all_sittable_semantic": gradient_summary(
            semantic["total"], named_lora, retain_graph=True
        ),
        "watch_write_invariance": gradient_summary(
            invariance_loss, named_lora, retain_graph=False
        ),
    }
    for name, summary in components.items():
        require_positive_gradient(name, summary)
    if zero_preservation_value != 0.0:
        raise AssertionError("zero-init preservation loss is not exactly zero")
    if zero_lora_energy_value != 0.0:
        raise AssertionError("zero-init LoRA energy is not exactly zero")

    saved_lora = perturb_lora_output(named_lora, 1e-5)
    perturbed_prediction = predict_xstart(
        model, diffusion, batch["x"], timesteps, kwargs, noise
    )
    perturbed_preservation = frozen_teacher_preservation_loss(
        perturbed_prediction, v5_prediction, batch["category_ids"]
    )
    perturbed_lora_energy = lora_parameter_energy(model)
    preservation_summary = gradient_summary(
        perturbed_preservation, named_lora, retain_graph=False
    )
    lora_summary = gradient_summary(
        perturbed_lora_energy, named_lora, retain_graph=False
    )
    require_positive_gradient("perturbed preservation", preservation_summary)
    require_positive_gradient("perturbed LoRA regularizer", lora_summary)
    restore_lora(named_lora, saved_lora)

    for parameter in model.parameters():
        parameter.grad = None
    prediction = predict_xstart(
        model, diffusion, batch["x"], timesteps, kwargs, noise
    )
    semantic = all_sittable_semantic_loss(
        prediction,
        batch["instance_ids"],
        batch["category_ids"],
        mean_tensor,
        std_tensor,
    )
    invariance_loss = paired_prompt_invariance_loss(
        prediction[0::2], prediction[1::2], batch["category_ids"][0::2]
    )
    preservation = frozen_teacher_preservation_loss(
        prediction, v5_prediction, batch["category_ids"]
    )
    total_loss = (
        (prediction - batch["x"]).square().mean()
        + 0.5 * semantic["total"]
        + 0.25 * invariance_loss
        + preservation
        + 1e-4 * lora_parameter_energy(model)
    )
    total_loss.backward()
    frozen_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if "lora_" not in name and parameter.grad is not None
    ]
    if frozen_gradient_names:
        raise AssertionError("gradient reached frozen CDM parameters")
    total_gradient_sq = 0.0
    for parameter in named_lora.values():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("total objective produced non-finite LoRA gradient")
            total_gradient_sq += float(parameter.grad.float().square().sum().item())
    total_gradient_l2 = math.sqrt(total_gradient_sq)
    if total_gradient_l2 <= 0.0:
        raise RuntimeError("total objective has no LoRA gradient")

    checks = {
        "cuda_used": True,
        "exact_train_scene_split": True,
        "sealed_v7_72_motion_144_row_index": True,
        "development_payloads_unread": True,
        "train_probes_are_new_high_desk_hc_hd": all(
            probe["probe_role"] == "new_high_desk_hc_hd_train_motion"
            and "_hc_hd_" in str(probe["motion_id"])
            and probe["target_instance_id"] == "chair_06"
            for probe in probes
        ),
        "gate0a_v5_checkpoint_bound": True,
        "original_checkpoint_bound": True,
        "zero_init_v5_output_bitwise_equal": True,
        "only_lora_parameters_trainable": True,
        "dense_replay_gradient_reaches_lora": True,
        "semantic_gradient_reaches_lora": True,
        "invariance_gradient_reaches_lora": True,
        "perturbed_preservation_gradient_reaches_lora": True,
        "perturbed_regularizer_gradient_reaches_lora": True,
        "total_objective_gradient_reaches_lora": True,
        "frozen_cdm_has_no_gradient": True,
        "relation_distance_absent_from_forward": set(kwargs)
        == set(FORWARD_INPUT_KEYS),
    }
    if not all(checks.values()):
        raise AssertionError("Teacher-v7 preflight check failed")
    source_paths = {
        "preflight": Path(__file__).resolve(),
        "validator": PREPARE_ROOT
        / "validate_relational_teacher_v7_hd_lora_preflight.py",
        "lora": PREPARE_ROOT / "fewshot_cdm_lora.py",
        "contract": PREPARE_ROOT / "relational_teacher_v6_contract.py",
        "semantics": PREPARE_ROOT / "relational_teacher_v6_semantics.py",
        "preflight_contract": PREPARE_ROOT
        / "relational_teacher_v7_hd_preflight_contract.py",
        "dataset_validator": PREPARE_ROOT
        / "validate_relational_teacher_v7_hd_dataset.py",
    }
    paths = {
        "dataset_index": str(index_file),
        "stats_file": str(stats_file),
        "original_checkpoint": str(original_checkpoint),
        "v5_checkpoint": str(v5_checkpoint),
        "v5_evidence_report": str(evidence_file),
        **{name: str(path) for name, path in source_paths.items()},
    }
    path_sha256 = {
        "dataset_index": sha256_file(index_file),
        "stats_file": sha256_file(stats_file),
        "original_checkpoint": sha256_file(original_checkpoint),
        "v5_checkpoint": sha256_file(v5_checkpoint),
        "v5_evidence_report": sha256_file(evidence_file),
        **{name: sha256_file(path) for name, path in source_paths.items()},
    }
    probe_records = [
        {
            key: probe[key]
            for key in (
                "scene_id",
                "motion_id",
                "target_instance_id",
                "probe_role",
                "points_sha256",
                "sidecar_sha256",
                "dense_index_sha256",
                "affordance_sha256",
            )
        }
        for probe in probes
    ]
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "seed": args.seed,
        "device": args.device,
        "diffusion_steps": args.diffusion_steps,
        "train_scenes": list(train_ids),
        "development_scenes_metadata_only": list(development_ids),
        "development_payloads_read": False,
        "dataset_contract": {
            "schema": DATASET_SCHEMA,
            "status": "DENSE_DATASET_PASS",
            "dense_motions": 72,
            "prompt_expanded_rows": 144,
            "train_rows": 96,
            "development_rows_metadata_only": 48,
        },
        "forward_input_keys": sorted(kwargs),
        "prompts": PROMPTS,
        "probe_records": probe_records,
        "paths": paths,
        "path_sha256": path_sha256,
        "lora": dict(lora_metadata(model)),
        "zero_init_parity_max_abs": parity_max_abs,
        "zero_init_preservation_loss": zero_preservation_value,
        "zero_init_lora_energy": zero_lora_energy_value,
        "component_gradients": {
            **components,
            "preservation_perturbed": preservation_summary,
            "lora_regularizer_perturbed": lora_summary,
        },
        "total_objective": {
            "loss": float(total_loss.detach().item()),
            "gradient_l2": total_gradient_l2,
            "weights": {
                "dense_replay": 1.0,
                "all_sittable_semantic": 0.5,
                "watch_write_invariance": 0.25,
                "frozen_v5_preservation": 1.0,
                "lora": 1e-4,
            },
        },
        "frozen_parameter_gradient_count": len(frozen_gradient_names),
        "checks": checks,
        "authorizes_12_update_calibration": True,
        "authorizes_long_training": False,
        "authorizes_development_evaluation": False,
        "authorizes_paper_test": False,
    }
    payload["binding_id"] = canonical_sha256(
        {
            "paths": path_sha256,
            "probes": probe_records,
            "train_scenes": list(train_ids),
            "development_scenes": list(development_ids),
            "seed": args.seed,
        }
    )
    atomic_write_json(report_file, payload)
    print("[PASS] relational Teacher-v7 High-Desk LoRA CUDA preflight")
    print("[PASS] zero-init output is bitwise equal to sealed v5r4 Teacher")
    print("[PASS] new High-Desk dense/semantic/invariance gradients reach LoRA only")
    print("[PASS] preservation and regularizer gradients reach LoRA only")
    print("[PASS] room_0201 development payloads unread")
    print("[OK] LoRA modules:", len(module_names))
    print("[OK] total gradient L2:", total_gradient_l2)
    print("[OK] binding ID:", payload["binding_id"])
    print("[OK] report:", report_file)


if __name__ == "__main__":
    main()

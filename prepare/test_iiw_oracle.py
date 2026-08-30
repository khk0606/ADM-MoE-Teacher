#!/usr/bin/env python3
"""Oracle IIW -> frozen CMDM integration and overfit test.

This runner answers one deliberately narrow question: if the *ground-truth*
phase-level IIW plan is supplied to a small :class:`IIWAdapter`, can motion
supervision learn to use it while the official pretrained CMDM remains bitwise
unchanged?

The IIW target contains future motion information, so results from this script
are an oracle upper-bound / wiring test, not deployable inference performance.
The existing base ADM is still passed to CMDM as ``c_pc_contact``.  IIW is a
separate residual condition and never replaces, multiplies, or renormalizes the
pretrained contact-map input.

Only ``IIWAdapter`` is optimized.  Evaluation uses identical diffusion noise
and timesteps for the correct temporal plan and five controls:

* phase-reversed plan;
* phasewise-static max plan;
* all-zero plan;
* another sample's plan (same-scene cyclic sample shuffle);
* the unmodified legacy CMDM path with no IIW residual.

The hard PASS contract is integration-focused: exact zero-residual legacy
parity, a nonzero adapter gradient, an unchanged frozen CMDM state, and an
improved fixed-timestep loss grid.  Ranking the correct plan above every
control is reported as a CHECK rather than a hard failure because six HC
samples are an overfit set, not a statistically meaningful benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models  # noqa: E402,F401
from models.base import create_model_and_diffusion  # noqa: E402
from models.iiw_adapter import IIWAdapter  # noqa: E402
from models.iiw_cmdm import (  # noqa: E402
    IIWConditionedCMDM,
    _inject_iiw_residual,
)
from utils.misc import compute_repr_dimesion  # noqa: E402
from utils.training import load_ckpt  # noqa: E402


NUM_POINTS = 8192
NUM_BODY_PARTS = 6
MAX_HORIZON = 196
DEFAULT_GRID = (0, 100, 250, 500, 750, 999)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(name + " contains NaN/Inf")


def parameter_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def state_digest(module: torch.nn.Module) -> str:
    """Hash every parameter and persistent buffer without retaining a copy."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_coverage(
    module: torch.nn.Module,
    checkpoint_file: Path,
    require_trainable_complete: bool = True,
) -> Dict[str, object]:
    """Verify that every trainable CMDM tensor is supplied by the checkpoint.

    AMDM's legacy ``load_ckpt`` intentionally permits missing keys (for
    transfer-learning experiments).  An Oracle freeze test must be stricter:
    otherwise randomly initialized CMDM layers could be mislabeled as fully
    pretrained. External text-encoder tensors intentionally excluded by
    AMDM's checkpoint writer are reported as frozen/missing instead.
    """
    saved = torch.load(str(checkpoint_file), map_location="cpu")
    if not isinstance(saved, dict):
        raise TypeError("CMDM checkpoint must contain a state-dict mapping")
    current = module.state_dict()
    trainable_names = {
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    frozen_external_prefixes = ("text_model.", "clip_model.", "bert_model.")
    loaded_keys = []
    missing_trainable = []
    mismatched_shapes = []
    missing_frozen = []
    for key, value in current.items():
        saved_key = key if key in saved else "module." + key
        if saved_key not in saved:
            if key in trainable_names and not key.startswith(
                frozen_external_prefixes
            ):
                missing_trainable.append(key)
            else:
                missing_frozen.append(key)
            continue
        saved_value = saved[saved_key]
        if not torch.is_tensor(saved_value):
            mismatched_shapes.append(
                {"key": key, "reason": "checkpoint value is not a tensor"}
            )
            continue
        if tuple(saved_value.shape) != tuple(value.shape):
            mismatched_shapes.append(
                {
                    "key": key,
                    "model_shape": list(value.shape),
                    "checkpoint_shape": list(saved_value.shape),
                }
            )
            continue
        loaded_keys.append(key)
    if mismatched_shapes or (
        require_trainable_complete and missing_trainable
    ):
        raise ValueError(
            "CMDM checkpoint coverage failed: missing_trainable={} "
            "shape_mismatches={}".format(
                len(missing_trainable), len(mismatched_shapes)
            )
        )
    return {
        "model_state_tensors": len(current),
        "loaded_shape_matched_tensors": len(loaded_keys),
        "missing_trainable_tensors": missing_trainable,
        "missing_frozen_tensors": missing_frozen,
        "shape_mismatches": mismatched_shapes,
    }


def load_cmdm_config(repo_root: Path):
    default_cfg = OmegaConf.load(repo_root / "configs/default.yaml")
    cfg = OmegaConf.create(
        {"seed": int(default_cfg.seed), "diffusion": default_cfg.diffusion}
    )
    cfg.task = OmegaConf.load(
        repo_root / "configs/task/contact_motion_gen.yaml"
    )
    cfg.model = OmegaConf.load(repo_root / "configs/model/cmdm.yaml")
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    OmegaConf.resolve(cfg)
    if cfg.model.data_repr != "pos" or int(cfg.model.input_feats) != 66:
        raise ValueError("oracle test requires CMDM pos representation (66D)")
    if int(cfg.task.dataset.num_points) != NUM_POINTS:
        raise ValueError("oracle test requires exactly 8192 scene points")
    if int(cfg.task.dataset.max_horizon) != MAX_HORIZON:
        raise ValueError("oracle test requires max_horizon=196")
    if int(cfg.model.latent_dim) != 512:
        raise ValueError("oracle patch was designed for CMDM latent_dim=512")
    return cfg


def resolve_under(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def run_lightweight_contract_tests() -> None:
    """CPU-only tests that do not require AMDM data or a checkpoint."""
    torch.manual_seed(17)
    adapter = IIWAdapter(
        latent_dim=32,
        num_phases=3,
        num_body_parts=NUM_BODY_PARTS,
        body_hidden_dim=12,
        hidden_dim=20,
        embedding_dim=4,
        zero_init=False,
    )
    batch_size, num_points, horizon = 2, 31, 7
    xyz = torch.randn(batch_size, num_points, 3)
    plan = torch.rand(batch_size, 3, num_points, NUM_BODY_PARTS)
    mapping = torch.tensor(
        [[0, 0, 1, 1, 2, -1, -1], [0, 1, 1, 2, 2, 2, -1]],
        dtype=torch.long,
    )
    mask = mapping < 0
    residual = adapter(xyz, plan, mapping, mask)
    assert residual.shape == (batch_size, horizon, 32)
    assert int(torch.count_nonzero(residual[mask]).item()) == 0
    assert not torch.allclose(residual, torch.zeros_like(residual))
    reversed_residual = adapter(xyz, plan.flip(1), mapping, mask)
    assert not torch.allclose(residual, reversed_residual)
    null_residual = adapter(xyz, torch.zeros_like(plan), mapping, mask)
    assert int(torch.count_nonzero(null_residual).item()) == 0

    motion = torch.randn(batch_size, horizon, 32)
    zero = torch.zeros_like(motion)
    assert torch.equal(_inject_iiw_residual(motion, zero, mask), motion)
    try:
        bad = zero.clone()
        bad[mask] = 1.0
        _inject_iiw_residual(motion, bad, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("CMDM accepted a nonzero padded IIW residual")
    print("[PASS] IIWAdapter CPU tensor contract")
    print("[PASS] trained/random adapter keeps zero IIW an exact no-op")
    print("[PASS] phase order changes frame conditioning")
    print("[PASS] CMDM zero/padded residual contract")


class IIWOracleDataset(Dataset):
    """Strict loader for an IIW-augmented production index."""

    def __init__(self, dataset_root: Path, index_names: Sequence[str]):
        self.root = dataset_root.expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not index_names:
            raise ValueError("at least one --index is required")
        self.scenes: Dict[str, Dict[str, np.ndarray]] = {}
        self.samples: List[Dict] = []
        seen_ids = set()

        for name in index_names:
            index_file = resolve_under(self.root, name)
            if not index_file.is_file():
                raise FileNotFoundError(index_file)
            index = json.loads(index_file.read_text())
            if not bool(index.get("base_adm_cached", False)):
                raise ValueError(str(index_file) + ": base ADM is not cached")
            if not bool(index.get("iiw_gt_ready", False)):
                raise ValueError(str(index_file) + ": IIW GT is not ready")
            if index.get("iiw_gt_method") != "point_aligned_iiw_proxy":
                raise ValueError(str(index_file) + ": unexpected IIW method")
            num_phases = int(index.get("iiw_num_phases", -1))
            if num_phases <= 1:
                raise ValueError(str(index_file) + ": invalid phase count")

            scene_id = str(index["scene_id"])
            if scene_id in self.scenes:
                raise ValueError("duplicate scene index: " + scene_id)
            self.scenes[scene_id] = self._load_scene(index, scene_id)

            entries = index.get("samples", [])
            if len(entries) != int(index.get("num_samples", -1)):
                raise ValueError(str(index_file) + ": num_samples mismatch")
            for entry in entries:
                sample = self._load_sample(entry, scene_id, num_phases)
                if sample["sample_id"] in seen_ids:
                    raise ValueError("duplicate sample: " + sample["sample_id"])
                seen_ids.add(sample["sample_id"])
                self.samples.append(sample)

        if not self.samples:
            raise ValueError("IIW oracle dataset contains no samples")
        phase_counts = {sample["num_phases"] for sample in self.samples}
        if len(phase_counts) != 1:
            raise ValueError("all oracle indices must use one phase count")
        self.num_phases = next(iter(phase_counts))

        # A point-aligned plan may only be shuffled between motions that use
        # the exact same scene point cloud.  Build a deterministic cyclic peer
        # inside every scene rather than taking the next global sample.
        scene_to_indices: Dict[str, List[int]] = {}
        for sample_index, sample in enumerate(self.samples):
            scene_to_indices.setdefault(sample["scene_id"], []).append(
                sample_index
            )
        self.shuffle_peer_indices: List[int] = list(
            range(len(self.samples))
        )
        for scene_id, indices in scene_to_indices.items():
            if len(indices) < 2:
                raise ValueError(
                    scene_id
                    + ": at least two samples are required for the "
                    + "same-scene shuffled-IIW control"
                )
            for offset, sample_index in enumerate(indices):
                self.shuffle_peer_indices[sample_index] = indices[
                    (offset + 1) % len(indices)
                ]

    def _load_scene(self, index: Dict, scene_id: str) -> Dict[str, np.ndarray]:
        adm_dir = resolve_under(self.root, index["scene_adm_input"])
        points_file = adm_dir / "points.npz"
        sidecar_file = adm_dir / "sidecar.npz"
        base_file = resolve_under(self.root, index["base_affordance"])
        for path in (points_file, sidecar_file, base_file):
            if not path.is_file():
                raise FileNotFoundError(path)

        with np.load(points_file, allow_pickle=False) as archive:
            points = archive["points"].astype(np.float32)
        with np.load(sidecar_file, allow_pickle=False) as archive:
            instance_ids = archive["instance_ids"].astype(np.int64)
            source_indices = archive["source_indices"].astype(np.int64)
        with np.load(base_file, allow_pickle=False) as archive:
            base = archive["affordance"].astype(np.float32)
            base_instances = archive["instance_ids"].astype(np.int64)
            base_sources = archive["source_indices"].astype(np.int64)

        if points.shape != (NUM_POINTS, 6):
            raise ValueError(scene_id + ": scene point shape mismatch")
        if base.shape != (NUM_POINTS, NUM_BODY_PARTS):
            raise ValueError(scene_id + ": base affordance shape mismatch")
        if not np.array_equal(instance_ids, base_instances):
            raise ValueError(scene_id + ": base/sidecar instance mismatch")
        if not np.array_equal(source_indices, base_sources):
            raise ValueError(scene_id + ": base/sidecar point-order mismatch")
        if not np.isfinite(points).all() or not np.isfinite(base).all():
            raise ValueError(scene_id + ": scene/base contains NaN/Inf")
        if np.any(base < 0.0) or np.any(base > 1.0):
            raise ValueError(scene_id + ": base affordance outside [0,1]")

        scene_points = points.copy()
        scene_points[:, 3:6] /= 255.0
        return {
            "scene_points": scene_points,
            "scene_xyz": scene_points[:, :3].copy(),
            "instance_ids": instance_ids,
            "source_indices": source_indices,
            "base_affordance": base,
        }

    def _load_sample(
        self, entry: Dict, scene_id: str, num_phases: int
    ) -> Dict:
        sample_id = str(entry["sample_id"])
        if str(entry["scene_id"]) != scene_id:
            raise ValueError(sample_id + ": scene_id mismatch")
        if not bool(entry.get("iiw_gt_ready", False)):
            raise ValueError(sample_id + ": IIW target is not ready")
        motion_file = resolve_under(self.root, entry["cmdm_motion_input"])
        iiw_file = resolve_under(self.root, entry["iiw_gt"])
        for path in (motion_file, iiw_file):
            if not path.is_file():
                raise FileNotFoundError(path)

        with np.load(motion_file, allow_pickle=False) as archive:
            motion = archive["motion_normalized"].astype(np.float32)
            x_mask = archive["x_mask"].astype(bool)
        with np.load(iiw_file, allow_pickle=False) as archive:
            plan = archive["iiw_native_phase_max"].astype(np.float32)
            frame_to_phase_valid = archive["frame_to_phase"].astype(np.int64)
            saved_xyz = archive["scene_xyz_adm"].astype(np.float32)
            saved_instances = archive["instance_ids"].astype(np.int64)
            saved_sources = archive["source_indices"].astype(np.int64)
            phase_mask = archive["phase_mask"].astype(bool)

        scene = self.scenes[scene_id]
        expected_plan_shape = (
            num_phases, NUM_POINTS, NUM_BODY_PARTS
        )
        if motion.shape != (MAX_HORIZON, 66):
            raise ValueError(sample_id + ": normalized motion shape mismatch")
        if x_mask.shape != (MAX_HORIZON,):
            raise ValueError(sample_id + ": x_mask shape mismatch")
        if plan.shape != expected_plan_shape:
            raise ValueError(sample_id + ": IIW plan shape mismatch")
        if phase_mask.shape != (num_phases,) or bool(phase_mask.any()):
            raise ValueError(sample_id + ": generated phases must all be valid")
        if not np.array_equal(saved_xyz, scene["scene_xyz"]):
            raise ValueError(sample_id + ": IIW/ADM XYZ order mismatch")
        if not np.array_equal(saved_instances, scene["instance_ids"]):
            raise ValueError(sample_id + ": IIW/ADM instance order mismatch")
        if not np.array_equal(saved_sources, scene["source_indices"]):
            raise ValueError(sample_id + ": IIW/ADM source order mismatch")
        if not np.isfinite(motion).all() or not np.isfinite(plan).all():
            raise ValueError(sample_id + ": motion/IIW contains NaN/Inf")
        if np.any(plan < 0.0) or np.any(plan > 1.0):
            raise ValueError(sample_id + ": IIW outside [0,1]")

        valid_frames = int((~x_mask).sum())
        if valid_frames != int(entry["valid_frames"]):
            raise ValueError(sample_id + ": valid-frame count mismatch")
        expected_mask = np.arange(MAX_HORIZON) >= valid_frames
        if not np.array_equal(x_mask, expected_mask):
            raise ValueError(sample_id + ": x_mask is not prefix-contiguous")
        if frame_to_phase_valid.shape != (valid_frames,):
            raise ValueError(sample_id + ": frame_to_phase length mismatch")
        if np.any(frame_to_phase_valid < 0) or np.any(
            frame_to_phase_valid >= num_phases
        ):
            raise ValueError(sample_id + ": phase index outside range")
        if np.any(np.diff(frame_to_phase_valid) < 0):
            raise ValueError(sample_id + ": phase order is not monotonic")
        if set(frame_to_phase_valid.tolist()) != set(range(num_phases)):
            raise ValueError(sample_id + ": not all phases cover valid frames")

        frame_to_phase = np.full((MAX_HORIZON,), -1, dtype=np.int64)
        frame_to_phase[:valid_frames] = frame_to_phase_valid
        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "text": str(entry["text"]),
            "motion": motion,
            "x_mask": x_mask,
            "iiw_plan": plan,
            "frame_to_phase": frame_to_phase,
            "valid_frames": valid_frames,
            "num_phases": num_phases,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        shuffled = self.samples[self.shuffle_peer_indices[index]]
        scene = self.scenes[sample["scene_id"]]
        if shuffled["scene_id"] != sample["scene_id"]:
            raise AssertionError("shuffled IIW peer crossed scene boundaries")
        if shuffled["iiw_plan"].shape != sample["iiw_plan"].shape:
            raise ValueError("sample-shuffle IIW shape mismatch")
        return {
            "x": sample["motion"].copy(),
            "x_mask": sample["x_mask"].copy(),
            "c_pc_xyz": scene["scene_xyz"].copy(),
            "c_pc_contact": scene["base_affordance"].copy(),
            "iiw_plan": sample["iiw_plan"].copy(),
            "iiw_shuffled": shuffled["iiw_plan"].copy(),
            "frame_to_phase": sample["frame_to_phase"].copy(),
            "c_text": sample["text"],
            "sample_id": sample["sample_id"],
            "shuffled_sample_id": shuffled["sample_id"],
        }


def make_plan_variant(plan: torch.Tensor, shuffled: torch.Tensor, mode: str):
    if mode == "temporal":
        return plan
    if mode == "reversed":
        return torch.flip(plan, dims=(1,))
    if mode == "static_max":
        return plan.max(dim=1, keepdim=True)[0].expand_as(plan)
    if mode == "zero":
        return torch.zeros_like(plan)
    if mode == "sample_shuffled":
        return shuffled
    raise ValueError("unknown IIW mode: " + mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--index", action="append")
    parser.add_argument("--cmdm-checkpoint", type=Path)
    parser.add_argument(
        "--allow-partial-cmdm-checkpoint",
        action="store_true",
        help=(
            "Allow missing trainable CMDM keys for deliberate transfer "
            "experiments; shape mismatches still fail and are reported."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--eval-timesteps", type=int, nargs="+", default=list(DEFAULT_GRID)
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_lightweight_contract_tests()
    if args.unit_only:
        print("[PASS] oracle IIW lightweight unit tests")
        return
    if args.dataset_root is None:
        parser.error("--dataset-root is required unless --unit-only is used")
    if not args.index:
        parser.error("--index is required unless --unit-only is used")
    if args.cmdm_checkpoint is None:
        parser.error(
            "--cmdm-checkpoint is required unless --unit-only is used"
        )

    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if args.adapter_lr <= 0.0:
        raise ValueError("adapter-lr must be positive")
    if not args.eval_timesteps:
        raise ValueError("eval-timesteps must not be empty")
    if any(value < 0 or value >= 1000 for value in args.eval_timesteps):
        raise ValueError("evaluation timesteps must lie in [0,999]")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset_root = args.dataset_root.expanduser().resolve()
    cmdm_checkpoint = args.cmdm_checkpoint.expanduser().resolve()
    if not cmdm_checkpoint.is_file():
        raise FileNotFoundError(cmdm_checkpoint)

    dataset = IIWOracleDataset(dataset_root, args.index)
    if args.batch_size > len(dataset):
        raise ValueError("batch-size exceeds dataset size")
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=train_generator,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    cfg = load_cmdm_config(REPO_ROOT)
    cmdm_backbone, diffusion = create_model_and_diffusion(
        cfg, device=str(device)
    )
    if any(
        value >= int(diffusion.num_timesteps)
        for value in args.eval_timesteps
    ):
        raise ValueError(
            "evaluation timestep exceeds diffusion.num_timesteps="
            + str(diffusion.num_timesteps)
        )
    cmdm_backbone.to(device)
    checkpoint_coverage_report = checkpoint_coverage(
        cmdm_backbone,
        cmdm_checkpoint,
        require_trainable_complete=(
            not args.allow_partial_cmdm_checkpoint
        ),
    )
    load_ckpt(cmdm_backbone, str(cmdm_checkpoint))
    if checkpoint_coverage_report["missing_trainable_tensors"]:
        print(
            "[CHECK] partial CMDM checkpoint explicitly allowed: "
            + str(
                len(
                    checkpoint_coverage_report[
                        "missing_trainable_tensors"
                    ]
                )
            )
            + " trainable tensors were not restored"
        )
    else:
        print(
            "[PASS] checkpoint covers every trainable CMDM tensor: "
            + str(
                checkpoint_coverage_report[
                    "loaded_shape_matched_tensors"
                ]
            )
            + "/"
            + str(checkpoint_coverage_report["model_state_tensors"])
            + " total state tensors matched"
        )
    cmdm_backbone.eval()
    for parameter in cmdm_backbone.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in cmdm_backbone.parameters()):
        raise AssertionError("CMDM freezing failed")
    frozen_digest_before = state_digest(cmdm_backbone)
    cmdm = IIWConditionedCMDM(cmdm_backbone).to(device)
    cmdm.eval()

    adapter = IIWAdapter(
        latent_dim=int(cfg.model.latent_dim),
        num_phases=dataset.num_phases,
        num_body_parts=NUM_BODY_PARTS,
        zero_init=True,
    ).to(device)
    adapter_parameters = [p for p in adapter.parameters() if p.requires_grad]
    if not adapter_parameters:
        raise RuntimeError("IIWAdapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=args.adapter_lr,
        weight_decay=args.weight_decay,
    )

    def prepare_batch(raw: Dict) -> Dict:
        result = {
            "x": raw["x"].to(device, torch.float32).contiguous(),
            "x_mask": raw["x_mask"].to(device, torch.bool).contiguous(),
            "scene_xyz": raw["c_pc_xyz"].to(
                device, torch.float32
            ).contiguous(),
            "base_contact": raw["c_pc_contact"].to(
                device, torch.float32
            ).contiguous(),
            "iiw_plan": raw["iiw_plan"].to(
                device, torch.float32
            ).contiguous(),
            "iiw_shuffled": raw["iiw_shuffled"].to(
                device, torch.float32
            ).contiguous(),
            "frame_to_phase": raw["frame_to_phase"].to(
                device, torch.long
            ).contiguous(),
            "texts": [str(value) for value in raw["c_text"]],
            "sample_ids": [str(value) for value in raw["sample_id"]],
            "shuffled_ids": [
                str(value) for value in raw["shuffled_sample_id"]
            ],
        }
        batch_size = int(result["x"].shape[0])
        expected_plan = (
            batch_size, dataset.num_phases, NUM_POINTS, NUM_BODY_PARTS
        )
        if result["x"].shape != (batch_size, MAX_HORIZON, 66):
            raise ValueError("motion batch shape mismatch")
        if result["scene_xyz"].shape != (batch_size, NUM_POINTS, 3):
            raise ValueError("scene XYZ batch shape mismatch")
        if result["base_contact"].shape != (
            batch_size, NUM_POINTS, NUM_BODY_PARTS
        ):
            raise ValueError("base contact batch shape mismatch")
        if result["iiw_plan"].shape != expected_plan:
            raise ValueError("IIW batch shape mismatch")
        if result["frame_to_phase"].shape != (
            batch_size, MAX_HORIZON
        ):
            raise ValueError("frame-to-phase batch shape mismatch")
        for name in ("x", "scene_xyz", "base_contact", "iiw_plan"):
            assert_finite(name, result[name])
        return result

    def residual_for(batch: Dict, mode: str) -> torch.Tensor:
        plan = make_plan_variant(
            batch["iiw_plan"], batch["iiw_shuffled"], mode
        )
        frame_to_phase = batch["frame_to_phase"]
        if mode == "static_max":
            # Remove both plan order and phase timing: every valid frame uses
            # phase 0, whose spatial content is the max over all GT phases.
            frame_to_phase = torch.where(
                batch["x_mask"],
                torch.full_like(frame_to_phase, -1),
                torch.zeros_like(frame_to_phase),
            )
        residual = adapter(
            batch["scene_xyz"],
            plan,
            frame_to_phase,
            batch["x_mask"],
        )
        if residual.shape != (
            batch["x"].shape[0], MAX_HORIZON, int(cfg.model.latent_dim)
        ):
            raise ValueError("adapter residual shape mismatch")
        assert_finite("adapter residual", residual)
        if not torch.equal(
            residual.masked_select(batch["x_mask"].unsqueeze(-1)),
            torch.zeros_like(
                residual.masked_select(batch["x_mask"].unsqueeze(-1))
            ),
        ):
            raise AssertionError("adapter residual is nonzero on padding")
        return residual

    def cmdm_forward(
        batch: Dict,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        mode: str = "temporal",
        include_residual: bool = True,
    ) -> torch.Tensor:
        """Return the direct CMDM prediction under a fixed noisy input."""
        kwargs = {
            "x_mask": batch["x_mask"],
            "c_pc_xyz": batch["scene_xyz"],
            "c_pc_contact": batch["base_contact"],
            "c_text": batch["texts"],
        }
        if include_residual:
            kwargs["c_iiw_residual"] = residual_for(batch, mode)
        x_t = diffusion.q_sample(batch["x"], timesteps, noise=noise)
        prediction = cmdm(
            x_t,
            diffusion._scale_timesteps(timesteps),
            **kwargs
        )
        assert_finite("CMDM prediction", prediction)
        return prediction

    def motion_loss(
        batch: Dict,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        mode: str = "temporal",
        include_residual: bool = True,
    ) -> torch.Tensor:
        kwargs = {
            "x_mask": batch["x_mask"],
            "c_pc_xyz": batch["scene_xyz"],
            "c_pc_contact": batch["base_contact"],
            "c_text": batch["texts"],
        }
        if include_residual:
            kwargs["c_iiw_residual"] = residual_for(batch, mode)
        terms = diffusion.training_losses(
            cmdm,
            batch["x"],
            timesteps,
            model_kwargs=kwargs,
            noise=noise,
        )
        loss = terms["loss"].mean()
        assert_finite("motion loss", loss)
        return loss

    modes = (
        "temporal",
        "reversed",
        "static_max",
        "zero",
        "sample_shuffled",
        "legacy",
    )
    control_modes = (
        "reversed", "static_max", "zero", "sample_shuffled"
    )

    def evaluate_grid(include_controls: bool) -> Dict:
        previous_training = adapter.training
        adapter.eval()
        cmdm.eval()
        requested_modes = modes if include_controls else ("temporal",)
        mode_losses = {mode: [] for mode in requested_modes}
        per_timestep = []
        with torch.no_grad():
            for timestep in args.eval_timesteps:
                # Resetting to the same seed makes initial/final evaluation
                # use exactly the same per-sample Gaussian noise.
                set_seed(args.seed + 100000 + int(timestep))
                timestep_values = {mode: [] for mode in requested_modes}
                for raw in eval_loader:
                    batch = prepare_batch(raw)
                    noise = torch.randn_like(batch["x"])
                    timesteps = torch.full(
                        (batch["x"].shape[0],),
                        int(timestep),
                        dtype=torch.long,
                        device=device,
                    )
                    for mode in requested_modes:
                        include_residual = mode != "legacy"
                        value = float(
                            motion_loss(
                                batch,
                                timesteps,
                                noise,
                                mode=mode,
                                include_residual=include_residual,
                            ).item()
                        )
                        timestep_values[mode].append(value)
                row = {"timestep": int(timestep)}
                for mode in requested_modes:
                    mean_value = float(np.mean(timestep_values[mode]))
                    row[mode] = mean_value
                    mode_losses[mode].append(mean_value)
                per_timestep.append(row)
        means = {
            mode: float(np.mean(values))
            for mode, values in mode_losses.items()
        }
        result = {
            "timesteps": [int(value) for value in args.eval_timesteps],
            "per_timestep": per_timestep,
            "motion_losses": mode_losses,
            "mean_motion_loss": means,
        }
        adapter.train(previous_training)
        return result

    # The new optional CMDM branch must be an exact no-op at zero residual.
    cmdm.eval()
    adapter.eval()
    parity_batch = prepare_batch(next(iter(eval_loader)))
    parity_timestep = torch.full(
        (1,), 500, dtype=torch.long, device=device
    )
    set_seed(args.seed + 7)
    parity_noise = torch.randn_like(parity_batch["x"])
    with torch.no_grad():
        baseline_prediction = cmdm_forward(
            parity_batch,
            parity_timestep,
            parity_noise,
            include_residual=False,
        )
        zero_residual = torch.zeros(
            (
                1, MAX_HORIZON, int(cfg.model.latent_dim)
            ),
            dtype=parity_batch["x"].dtype,
            device=device,
        )
        x_t = diffusion.q_sample(
            parity_batch["x"], parity_timestep, noise=parity_noise
        )
        explicit_zero_prediction = cmdm(
            x_t,
            diffusion._scale_timesteps(parity_timestep),
            x_mask=parity_batch["x_mask"],
            c_pc_xyz=parity_batch["scene_xyz"],
            c_pc_contact=parity_batch["base_contact"],
            c_text=parity_batch["texts"],
            c_iiw_residual=zero_residual,
        )
    if not torch.equal(baseline_prediction, explicit_zero_prediction):
        raise AssertionError("zero IIW residual changes legacy CMDM output")
    print("[PASS] zero residual preserves exact legacy CMDM output")

    initial_grid = evaluate_grid(include_controls=False)

    # Confirm a pure motion loss reaches the adapter through frozen CMDM.
    # Keep frozen CMDM deterministic.  The adapter contains no dropout, so it
    # can remain in eval mode while gradients still flow normally.
    adapter.eval()
    gradient_batch = prepare_batch(next(iter(eval_loader)))
    optimizer.zero_grad(set_to_none=True)
    set_seed(args.seed + 11)
    gradient_noise = torch.randn_like(gradient_batch["x"])
    gradient_timesteps = torch.full(
        (1,), 500, dtype=torch.long, device=device
    )
    gradient_loss = motion_loss(
        gradient_batch, gradient_timesteps, gradient_noise
    )
    gradient_loss.backward()
    initial_gradient_l2 = parameter_l2(adapter_parameters)
    if not np.isfinite(initial_gradient_l2) or initial_gradient_l2 <= 0.0:
        raise AssertionError("motion loss gradient to IIWAdapter is zero")
    optimizer.zero_grad(set_to_none=True)
    print(
        "[PASS] motion loss reaches IIWAdapter: gradient_l2="
        + "{:.9g}".format(initial_gradient_l2)
    )

    set_seed(args.seed + 1)
    adapter.train()
    cmdm.eval()
    records = []
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        batch = prepare_batch(raw)
        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (batch["x"].shape[0],),
            device=device,
        )
        noise = torch.randn_like(batch["x"])
        optimizer.zero_grad(set_to_none=True)
        loss = motion_loss(batch, timesteps, noise)
        loss.backward()
        gradient_l2 = parameter_l2(adapter_parameters)
        if not np.isfinite(gradient_l2) or gradient_l2 <= 0.0:
            raise AssertionError("non-finite/zero adapter gradient during training")
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters, args.grad_clip
            )
        optimizer.step()
        record = {
            "step": step,
            "sample_ids": batch["sample_ids"],
            "timesteps": timesteps.detach().cpu().tolist(),
            "motion_loss": float(loss.detach().item()),
            "adapter_gradient_l2": gradient_l2,
        }
        records.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(record))

    final_grid = evaluate_grid(include_controls=True)

    # Null-plan anchoring must remain an exact no-op after optimization, not
    # merely at initialization.  This prevents the adapter from solving the
    # task with phase embeddings or biases while ignoring IIW content.
    adapter.eval()
    with torch.no_grad():
        for raw in eval_loader:
            zero_batch = prepare_batch(raw)
            zero_plan_residual = residual_for(zero_batch, "zero")
            if int(torch.count_nonzero(zero_plan_residual).item()) != 0:
                raise AssertionError(
                    "trained zero-IIW plan produced a nonzero residual"
                )
    print("[PASS] trained zero-IIW plan remains an exact zero residual")
    adapter.train()

    zero_losses = np.asarray(
        final_grid["motion_losses"]["zero"], dtype=np.float64
    )
    legacy_losses = np.asarray(
        final_grid["motion_losses"]["legacy"], dtype=np.float64
    )
    zero_legacy_max_abs_diff = float(
        np.max(np.abs(zero_losses - legacy_losses))
    )
    if not np.allclose(
        zero_losses, legacy_losses, rtol=1e-6, atol=1e-8
    ):
        raise AssertionError(
            "trained zero-IIW losses differ from the legacy CMDM path: "
            + "max_abs_diff={:.9g}".format(zero_legacy_max_abs_diff)
        )
    print(
        "[PASS] zero-IIW and legacy losses agree: max_abs_diff="
        + "{:.9g}".format(zero_legacy_max_abs_diff)
    )

    frozen_digest_after = state_digest(cmdm_backbone)
    cmdm_unchanged = frozen_digest_after == frozen_digest_before
    if not cmdm_unchanged:
        raise AssertionError("frozen CMDM state changed during adapter training")
    print("[PASS] official pretrained CMDM state is bitwise unchanged")

    initial_mean = initial_grid["mean_motion_loss"]["temporal"]
    final_mean = final_grid["mean_motion_loss"]["temporal"]
    grid_improved = final_mean < initial_mean
    if not grid_improved:
        raise AssertionError(
            "fixed-timestep oracle loss did not improve: "
            + "{:.8f} -> {:.8f}".format(initial_mean, final_mean)
        )
    print(
        "[PASS] fixed-timestep grid improved: "
        + "{:.8f} -> {:.8f}".format(initial_mean, final_mean)
    )

    final_means = final_grid["mean_motion_loss"]
    temporal_beats = {
        mode: bool(final_means["temporal"] < final_means[mode])
        for mode in control_modes
    }
    all_controls_worse = all(temporal_beats.values())
    print(
        "[PASS] temporal plan beats every corruption"
        if all_controls_worse
        else "[CHECK] inspect oracle/control ranking on the six-sample set"
    )
    for mode in modes:
        print(
            "[OK] final {} grid mean: {:.8f}".format(
                mode, final_means[mode]
            )
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "experiments/iiw_oracle"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS",
        "scope": "oracle IIW adapter overfit and frozen-CMDM contract",
        "deployable_inference": False,
        "oracle_disclosure": (
            "Ground-truth phase IIW is computed from the complete target "
            "motion and therefore contains future information."
        ),
        "dataset_root": str(dataset_root),
        "index_files": list(args.index),
        "num_samples": len(dataset),
        "num_phases": dataset.num_phases,
        "source_cmdm_checkpoint": str(cmdm_checkpoint),
        "cmdm_checkpoint_coverage": checkpoint_coverage_report,
        "partial_cmdm_checkpoint_allowed": bool(
            args.allow_partial_cmdm_checkpoint
        ),
        "cmdm_frozen": True,
        "cmdm_static_contact": "original cached base_affordance",
        "iiw_condition_path": "separate c_iiw_residual",
        "zero_residual_legacy_parity": True,
        "trained_zero_plan_exact_zero_residual": True,
        "zero_legacy_loss_max_abs_diff": zero_legacy_max_abs_diff,
        "cmdm_state_digest_before": frozen_digest_before,
        "cmdm_state_digest_after": frozen_digest_after,
        "cmdm_state_unchanged": cmdm_unchanged,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eval_timesteps": [int(value) for value in args.eval_timesteps],
        "adapter_lr": args.adapter_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "adapter_architecture": {
            "latent_dim": adapter.latent_dim,
            "num_phases": adapter.num_phases,
            "num_body_parts": adapter.num_body_parts,
            "body_hidden_dim": adapter.body_hidden_dim,
            "hidden_dim": adapter.hidden_dim,
            "embedding_dim": adapter.embedding_dim,
            "null_plan_anchored": True,
        },
        "initial_motion_only_adapter_gradient_l2": initial_gradient_l2,
        "initial_timestep_grid": initial_grid,
        "final_timestep_grid": final_grid,
        "grid_improved": grid_improved,
        "temporal_beats_controls": temporal_beats,
        "temporal_beats_all_controls": all_controls_worse,
        "control_ranking_is_hard_pass_requirement": False,
        "optimization_records": records,
    }
    summary_file = output_dir / "summary.json"
    checkpoint_file = output_dir / "iiw_oracle_adapter.pt"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "iiw_adapter_state_dict": adapter.state_dict(),
            "source_cmdm_checkpoint": str(cmdm_checkpoint),
            "cmdm_state_digest": frozen_digest_after,
            "num_phases": dataset.num_phases,
            "native_body_part_count": NUM_BODY_PARTS,
            "adapter_architecture": summary["adapter_architecture"],
            "summary": summary,
        },
        checkpoint_file,
    )
    print("[PASS] oracle IIW -> frozen CMDM integration and overfit")
    print("[OK] saved: " + str(summary_file))
    print("[OK] saved: " + str(checkpoint_file))


if __name__ == "__main__":
    main()

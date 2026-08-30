"""Production datasets for Unity ``history_affordance_v1`` exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from datasets.base import DATASET
from datasets.motionx import ContactMapDataset
from utils.misc import compute_repr_dimesion


@DATASET.register()
class HistoryAffordanceV1ContactMapDataset(ContactMapDataset):
    """Load one prepared production scene for frozen base-ADM inference.

    The scene/text pair is evaluated exactly once.  Motion histories are not
    inputs to the base ADM and are intentionally absent here.
    """

    def __init__(self, cfg, phase: str, **kwargs):
        if phase != "test":
            raise ValueError(
                "HistoryAffordanceV1ContactMapDataset supports test only"
            )
        self.dataset_root = Path(cfg.dataset_root).expanduser().resolve()
        self.scene_id = str(cfg.scene_id)
        self.stats_file = Path(cfg.stats_file).expanduser().resolve()
        self.contact_dim = compute_repr_dimesion(cfg.data_repr)
        super().__init__(cfg, phase, **kwargs)

    def _load_datasets(self):
        index_file = self.dataset_root / ("index_" + self.scene_id + ".json")
        if not index_file.is_file():
            raise FileNotFoundError(index_file)
        self.dataset_index = json.loads(index_file.read_text())
        if self.dataset_index.get("scene_id") != self.scene_id:
            raise ValueError("production index scene_id mismatch")
        samples = self.dataset_index.get("samples", [])
        if not samples:
            raise ValueError("production index contains no samples")
        texts = {str(sample["text"]) for sample in samples}
        if len(texts) != 1:
            raise ValueError(
                "base ADM cache requires one fixed text per scene evaluation"
            )
        self.text = next(iter(texts))
        self.scene_adm_dir = self.dataset_root / self.dataset_index[
            "scene_adm_input"
        ]
        self.all_data = [0]
        self.indices = [0]

    def _prepare_statistics(self):
        if not self.stats_file.is_file():
            raise FileNotFoundError(self.stats_file)
        stats = np.load(self.stats_file, allow_pickle=False)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)
        expected = (1, self.contact_dim)
        if self.mean.shape != expected or self.std.shape != expected:
            raise ValueError(
                "ADM statistics shape mismatch: "
                + str((self.mean.shape, self.std.shape))
            )
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("ADM statistics contain NaN/Inf")
        if np.any(self.std <= 0.0):
            raise ValueError("ADM statistics contain non-positive std")

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        points_file = self.scene_adm_dir / "points.npz"
        sidecar_file = self.scene_adm_dir / "sidecar.npz"
        points = np.load(points_file, allow_pickle=False)["points"].astype(
            np.float32
        )
        sidecar = np.load(sidecar_file, allow_pickle=False)
        instance_ids = sidecar["instance_ids"].astype(np.int64)
        candidate_mask = sidecar["candidate_mask"].astype(bool)
        if points.shape != (self.num_points, 6):
            raise ValueError(
                "expected scene points "
                + str((self.num_points, 6))
                + ", got "
                + str(points.shape)
            )
        if instance_ids.shape != (self.num_points,):
            raise ValueError("instance_ids shape mismatch")
        if candidate_mask.shape != (self.num_points, 3):
            raise ValueError("candidate_mask shape mismatch")
        if not np.isfinite(points).all():
            raise ValueError("scene points contain NaN/Inf")

        xyz = points[:, 0:3]
        feat = np.zeros((self.num_points, 0), dtype=np.float32)
        if self.use_color:
            feat = points[:, 3:6] / 255.0
        contact = np.zeros(
            (self.num_points, self.contact_dim), dtype=np.float32
        )
        info_set = "history_affordance_v1_" + self.scene_id
        data = {
            "x": contact,
            "c_pc_xyz": xyz,
            "c_pc_feat": feat,
            "c_text": self.text,
            "info_set": info_set,
            "info_index": 0,
            "info_scene_trans": np.eye(4, dtype=np.float32),
            "info_scene_mesh": "",
            "info_instance_ids": instance_ids,
            "info_candidate_mask": candidate_mask,
        }
        if self.transform is not None:
            data = self.transform(data)
        data["x"] = self.normalize(data["x"])
        return data


@DATASET.register()
class HistoryAffordanceV1ContactMotionDataset(Dataset):
    """Prepared multi-sample HistoryMoE/CMDM training dataset.

    Scene point clouds and frozen base ADM predictions are cached once per
    scene.  Motion targets and history states remain sample-specific.  More
    index files can be supplied later when LC, bed, and board data are ready.
    """

    def __init__(self, cfg, phase: str, **kwargs):
        if phase not in ("train", "all", "test"):
            raise ValueError("unsupported phase: " + str(phase))
        self.phase = phase
        self.dataset_root = Path(cfg.dataset_root).expanduser().resolve()
        self.num_points = int(getattr(cfg, "num_points", 8192))
        self.max_horizon = int(getattr(cfg, "max_horizon", 196))
        configured_indices = list(cfg.index_files)
        if not configured_indices:
            raise ValueError("index_files must contain at least one index")
        self.index_files = []
        for value in configured_indices:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.dataset_root / path
            self.index_files.append(path.resolve())
        self.scene_cache = {}
        self.samples = []
        self._load_and_validate()

    def _load_scene(self, index, index_file):
        scene_id = str(index["scene_id"])
        if not bool(index.get("base_adm_cached", False)):
            raise RuntimeError(
                f"{index_file}: base_adm_cached is not true"
            )
        scene_adm_dir = self.dataset_root / index["scene_adm_input"]
        base_file = self.dataset_root / index["base_affordance"]
        required = (
            scene_adm_dir / "points.npz",
            scene_adm_dir / "sidecar.npz",
            base_file,
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)

        points = np.load(required[0], allow_pickle=False)["points"].astype(
            np.float32
        )
        sidecar = np.load(required[1], allow_pickle=False)
        instance_ids = sidecar["instance_ids"].astype(np.int64)
        candidate_mask = sidecar["candidate_mask"].astype(bool)
        source_indices = sidecar["source_indices"].astype(np.int64)
        base = np.load(base_file, allow_pickle=False)
        affordance = base["affordance"].astype(np.float32)
        if points.shape != (self.num_points, 6):
            raise ValueError(f"{scene_id}: point shape mismatch")
        if instance_ids.shape != (self.num_points,):
            raise ValueError(f"{scene_id}: instance ID shape mismatch")
        if candidate_mask.shape != (self.num_points, 3):
            raise ValueError(f"{scene_id}: candidate mask shape mismatch")
        if affordance.shape != (self.num_points, 6):
            raise ValueError(f"{scene_id}: base affordance shape mismatch")
        if not np.array_equal(base["instance_ids"], instance_ids):
            raise ValueError(f"{scene_id}: base/sidecar instance order mismatch")
        if not np.array_equal(base["source_indices"], source_indices):
            raise ValueError(f"{scene_id}: base/sidecar point order mismatch")
        for name, value in (
            ("points", points),
            ("base affordance", affordance),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{scene_id}: {name} contains NaN/Inf")
        if not np.all((affordance >= 0.0) & (affordance <= 1.0)):
            raise ValueError(f"{scene_id}: base affordance outside [0,1]")

        scene_points = points.copy()
        scene_points[:, 3:6] /= 255.0
        self.scene_cache[scene_id] = {
            "scene_points": scene_points,
            "instance_ids": instance_ids,
            "candidate_mask": candidate_mask,
            "base_affordance": affordance,
            "base_file": str(base_file),
        }

    def _load_sample(self, entry, expected_scene_id, index_file):
        sample_id = str(entry["sample_id"])
        scene_id = str(entry["scene_id"])
        if scene_id != expected_scene_id:
            raise ValueError(f"{sample_id}: scene_id differs from its index")
        manifest_file = self.dataset_root / entry["sample_manifest"]
        motion_file = self.dataset_root / entry["cmdm_motion_input"]
        for path in (manifest_file, motion_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("sample_id") != sample_id:
            raise ValueError(f"{sample_id}: output manifest ID mismatch")
        if not bool(manifest.get("spatial_pairing_ready", False)):
            raise RuntimeError(f"{sample_id}: spatial pairing is not ready")
        if not bool(manifest.get("cmdm_motion_ready", False)):
            raise RuntimeError(f"{sample_id}: CMDM motion is not ready")

        motion = np.load(motion_file, allow_pickle=False)
        normalized = motion["motion_normalized"].astype(np.float32)
        padded = motion["motion_padded"].astype(np.float32)
        mask = motion["x_mask"].astype(bool)
        if normalized.shape != (self.max_horizon, 66):
            raise ValueError(f"{sample_id}: normalized motion shape mismatch")
        if padded.shape != (self.max_horizon, 66):
            raise ValueError(f"{sample_id}: raw motion shape mismatch")
        if mask.shape != (self.max_horizon,):
            raise ValueError(f"{sample_id}: motion mask shape mismatch")
        valid_frames = int((~mask).sum())
        if valid_frames != int(entry["valid_frames"]):
            raise ValueError(f"{sample_id}: valid-frame count mismatch")
        if not np.isfinite(normalized).all() or not np.isfinite(padded).all():
            raise ValueError(f"{sample_id}: motion contains NaN/Inf")

        start_xyz = np.asarray(
            entry["start_position_adm_chair_local_xyz"], dtype=np.float32
        )
        history_xy = np.asarray(
            entry["history_direction_adm_xy"], dtype=np.float32
        )
        if start_xyz.shape != (3,) or history_xy.shape != (2,):
            raise ValueError(f"{sample_id}: history-state shape mismatch")
        history_norm = float(np.linalg.norm(history_xy))
        if history_norm < 1e-8:
            raise ValueError(f"{sample_id}: history direction is zero")
        history_xy /= history_norm
        history_state = np.concatenate(
            [start_xyz[0:2], history_xy]
        ).astype(np.float32)

        return {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "text": str(entry["text"]),
            "target_index": int(entry["target_index"]),
            "motion_normalized": normalized,
            "motion_padded": padded,
            "x_mask": mask,
            "history_state": history_state,
            "valid_frames": valid_frames,
            "source_index_file": str(index_file),
        }

    def _load_and_validate(self):
        seen_sample_ids = set()
        for index_file in self.index_files:
            if not index_file.is_file():
                raise FileNotFoundError(index_file)
            index = json.loads(index_file.read_text())
            scene_id = str(index["scene_id"])
            if scene_id in self.scene_cache:
                raise ValueError(f"duplicate scene index: {scene_id}")
            self._load_scene(index, index_file)
            entries = index.get("samples", [])
            if len(entries) != int(index["num_samples"]):
                raise ValueError(f"{index_file}: num_samples mismatch")
            for entry in entries:
                sample = self._load_sample(entry, scene_id, index_file)
                if sample["sample_id"] in seen_sample_ids:
                    raise ValueError(
                        "duplicate sample ID: " + sample["sample_id"]
                    )
                seen_sample_ids.add(sample["sample_id"])
                self.samples.append(sample)
        if not self.samples:
            raise ValueError("production dataset contains no samples")

    def __len__(self):
        return len(self.samples)

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scene = self.scene_cache[sample["scene_id"]]
        return {
            "x": sample["motion_normalized"].copy(),
            "x_raw": sample["motion_padded"].copy(),
            "x_mask": sample["x_mask"].copy(),
            "c_pc_xyz": scene["scene_points"][:, 0:3].copy(),
            "c_scene_points": scene["scene_points"].copy(),
            "c_instance_ids": scene["instance_ids"].copy(),
            "c_candidate_mask": scene["candidate_mask"].copy(),
            "c_base_affordance": scene["base_affordance"].copy(),
            "c_history_state": sample["history_state"].copy(),
            "c_target_index": np.asarray(
                sample["target_index"], dtype=np.int64
            ),
            "c_text": sample["text"],
            "info_sample_id": sample["sample_id"],
            "info_scene_id": sample["scene_id"],
            "info_valid_frames": np.asarray(
                sample["valid_frames"], dtype=np.int64
            ),
            "info_index": idx,
            "info_set": "history_affordance_v1",
            "info_scene_trans": np.eye(4, dtype=np.float32),
            "info_scene_mesh": "",
        }

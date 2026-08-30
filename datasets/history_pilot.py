"""
Dataset classes for the Unity history-affordance pilot.
데이터를 PyTorch가 읽도록 하는 Dataset/Loader
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from datasets.base import DATASET
from datasets.motionx import ContactMapDataset
from utils.misc import compute_repr_dimesion


CONTACT_JOINTS = np.asarray([0, 10, 11, 12, 20, 21], dtype=np.int64)


@DATASET.register()
class HistoryPilotContactMapDataset(ContactMapDataset):
    """Load one Unity pilot sample for frozen ADM inference."""

    def __init__(self, cfg, phase: str, **kwargs):
        if phase != "test":
            raise ValueError("HistoryPilotContactMapDataset supports test only")

        self.pilot_dir = Path(cfg.pilot_dir).expanduser().resolve()
        self.pilot_stats_file = Path(cfg.pilot_stats_file).expanduser().resolve()
        self.contact_dim = compute_repr_dimesion(cfg.data_repr)

        super().__init__(cfg, phase, **kwargs)

    def _load_datasets(self):
        self.all_data = [0]
        self.indices = [0]

    def _prepare_statistics(self):
        if not self.pilot_stats_file.is_file():
            raise FileNotFoundError(
                f"Missing ADM normalization statistics: {self.pilot_stats_file}"
            )

        stats = np.load(self.pilot_stats_file)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

        expected_shape = (1, self.contact_dim)

        if self.mean.shape != expected_shape:
            raise ValueError(
                f"mean shape must be {expected_shape}, got {self.mean.shape}"
            )

        if self.std.shape != expected_shape:
            raise ValueError(
                f"std shape must be {expected_shape}, got {self.std.shape}"
            )

        if not np.isfinite(self.mean).all():
            raise ValueError("ADM normalization mean contains NaN/Inf")

        if not np.isfinite(self.std).all():
            raise ValueError("ADM normalization std contains NaN/Inf")

        if np.any(self.std <= 0.0):
            raise ValueError("ADM normalization std must be positive")

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)

        input_dir = self.pilot_dir / "adm_input"
        points_file = input_dir / "points.npz"
        sidecar_file = input_dir / "sidecar.npz"
        manifest_file = input_dir / "manifest.json"

        for path in (points_file, sidecar_file, manifest_file):
            if not path.is_file():
                raise FileNotFoundError(path)

        points = np.load(points_file)["points"].astype(np.float32)
        sidecar = np.load(sidecar_file)
        manifest = json.loads(manifest_file.read_text())

        if points.shape != (self.num_points, 6):
            raise ValueError(
                f"Expected ({self.num_points},6), got {points.shape}"
            )

        if not np.isfinite(points).all():
            raise ValueError("ADM input points contain NaN/Inf")

        instance_ids = sidecar["instance_ids"].astype(np.int64)
        candidate_mask = sidecar["candidate_mask"].astype(bool)

        if instance_ids.shape != (self.num_points,):
            raise ValueError("instance_ids shape mismatch")

        if candidate_mask.shape != (self.num_points, 3):
            raise ValueError("candidate_mask shape mismatch")

        xyz = points[:, 0:3]

        if self.use_color:
            feat = (points[:, 3:6] / 255.0).astype(np.float32)
        else:
            feat = np.zeros((self.num_points, 0), dtype=np.float32)

        # ADM inference에서는 GT contact map을 사용하지 않으므로
        # 입력 x에는 zero dummy map을 전달한다.
        x = np.zeros(
            (self.num_points, self.contact_dim),
            dtype=np.float32,
        )

        data = {
            "x": x,
            "c_pc_xyz": xyz,
            "c_pc_feat": feat,
            "c_text": manifest["text"],
            "info_set": "history_pilot",
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
class HistoryPilotContactMotionDataset(Dataset):
    """Load a paired Unity motion and full-scene GT affordance sample.

    Returned data includes:

    - normalized GT CMDM motion;
    - Unity scene point cloud;
    - frozen ADM base affordance;
    - full-motion GT affordance;
    - history state;
    - target furniture index.
    """

    def __init__(self, cfg, phase: str, **kwargs):
        if phase not in ("train", "all", "test"):
            raise ValueError(f"unsupported phase: {phase}")

        self.phase = phase
        self.pilot_dir = Path(cfg.pilot_dir).expanduser().resolve()
        self.base_affordance_file = (
            Path(cfg.base_affordance_file).expanduser().resolve()
        )
        self.motion_id = str(getattr(cfg, "motion_id", "chair_0001"))
        self.num_points = int(getattr(cfg, "num_points", 8192))
        self.max_horizon = int(getattr(cfg, "max_horizon", 196))

        self._load_and_validate()

    def _load_and_validate(self):
        pilot_manifest_file = self.pilot_dir / "pilot_manifest.json"
        adm_manifest_file = self.pilot_dir / "adm_input/manifest.json"
        point_file = self.pilot_dir / "adm_input/points.npz"
        sidecar_file = self.pilot_dir / "adm_input/sidecar.npz"

        motion_input_file = (
            self.pilot_dir
            / "motion_gt"
            / self.motion_id
            / "cmdm_pos"
            / "motion_cmdm_input.npz"
        )

        gt_affordance_file = (
            self.pilot_dir
            / "motion_gt"
            / self.motion_id
            / "affordance_gt"
            / "full_affordance_gt.npz"
        )

        required_files = (
            pilot_manifest_file,
            adm_manifest_file,
            point_file,
            sidecar_file,
            motion_input_file,
            self.base_affordance_file,
            gt_affordance_file,
        )

        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(path)

        self.pilot_manifest = json.loads(pilot_manifest_file.read_text())
        self.adm_manifest = json.loads(adm_manifest_file.read_text())

        if not bool(self.pilot_manifest.get("spatial_pairing_ready", False)):
            raise RuntimeError("spatial_pairing_ready is not true")

        if not bool(self.pilot_manifest.get("cmdm_motion_ready", False)):
            raise RuntimeError("cmdm_motion_ready is not true")

        # Scene point cloud
        self.points = np.load(point_file)["points"].astype(np.float32)

        sidecar = np.load(sidecar_file)
        self.instance_ids = sidecar["instance_ids"].astype(np.int64)
        self.candidate_mask = sidecar["candidate_mask"].astype(bool)
        sidecar_xyz = sidecar["xyz_afford_z_up"].astype(np.float32)

        # GT CMDM motion
        motion_input = np.load(motion_input_file)
        self.motion_normalized = motion_input["motion_normalized"].astype(
            np.float32
        )
        self.motion_padded = motion_input["motion_padded"].astype(np.float32)
        self.x_mask = motion_input["x_mask"].astype(bool)

        # Frozen ADM base affordance
        base_data = np.load(self.base_affordance_file)

        if "affordance" not in base_data:
            raise KeyError("base affordance file has no 'affordance' key")

        self.base_affordance = base_data["affordance"].astype(np.float32)

        # Unity full-motion GT affordance
        gt_data = np.load(gt_affordance_file)

        required_gt_keys = ("xyz", "distance", "affordance", "contact_joints")

        for key in required_gt_keys:
            if key not in gt_data:
                raise KeyError(f"GT affordance file has no '{key}' key")

        self.gt_affordance_xyz = gt_data["xyz"].astype(np.float32)
        self.gt_distance = gt_data["distance"].astype(np.float32)
        self.gt_affordance = gt_data["affordance"].astype(np.float32)
        self.gt_contact_joints = gt_data["contact_joints"].astype(np.int64)

        self._validate_shapes(sidecar_xyz)
        self._validate_point_order(sidecar_xyz, base_data)
        self._validate_values()
        self._prepare_condition_data()

    def _validate_shapes(self, sidecar_xyz):
        if self.points.shape != (self.num_points, 6):
            raise ValueError(
                f"points must be ({self.num_points},6), "
                f"got {self.points.shape}"
            )

        if sidecar_xyz.shape != (self.num_points, 3):
            raise ValueError(
                f"sidecar xyz shape mismatch: {sidecar_xyz.shape}"
            )

        if self.instance_ids.shape != (self.num_points,):
            raise ValueError("instance_ids shape mismatch")

        if self.candidate_mask.shape != (self.num_points, 3):
            raise ValueError("candidate_mask shape mismatch")

        if self.base_affordance.shape != (self.num_points, 6):
            raise ValueError(
                f"base_affordance shape mismatch: "
                f"{self.base_affordance.shape}"
            )

        if self.gt_affordance_xyz.shape != (self.num_points, 3):
            raise ValueError(
                f"GT affordance xyz shape mismatch: "
                f"{self.gt_affordance_xyz.shape}"
            )

        if self.gt_distance.shape != (self.num_points, 6):
            raise ValueError(
                f"GT distance shape mismatch: {self.gt_distance.shape}"
            )

        if self.gt_affordance.shape != (self.num_points, 6):
            raise ValueError(
                f"GT affordance shape mismatch: "
                f"{self.gt_affordance.shape}"
            )

        if not np.array_equal(self.gt_contact_joints, CONTACT_JOINTS):
            raise ValueError(
                "GT contact-joint order mismatch: "
                f"{self.gt_contact_joints.tolist()}"
            )

        if self.motion_normalized.shape != (self.max_horizon, 66):
            raise ValueError(
                "normalized CMDM motion shape mismatch: "
                f"{self.motion_normalized.shape}"
            )

        if self.motion_padded.shape != (self.max_horizon, 66):
            raise ValueError(
                "raw padded CMDM motion shape mismatch: "
                f"{self.motion_padded.shape}"
            )

        if self.x_mask.shape != (self.max_horizon,):
            raise ValueError(f"x_mask shape mismatch: {self.x_mask.shape}")

        expected_valid_frames = int(
            self.pilot_manifest["cmdm_motion_num_frames"]
        )
        actual_valid_frames = int((~self.x_mask).sum())

        if actual_valid_frames != expected_valid_frames:
            raise ValueError(
                "valid-frame count disagrees with pilot manifest: "
                f"{actual_valid_frames} != {expected_valid_frames}"
            )

    def _validate_point_order(self, sidecar_xyz, base_data):
        scene_xyz = self.points[:, 0:3]

        if not np.allclose(sidecar_xyz, scene_xyz, atol=1e-6):
            raise ValueError(
                "sidecar point order does not match points.npz"
            )

        if not np.allclose(
            self.gt_affordance_xyz,
            scene_xyz,
            atol=1e-6,
        ):
            raise ValueError(
                "GT affordance point order does not match ADM point order"
            )

        if "xyz" in base_data:
            base_xyz = base_data["xyz"].astype(np.float32)

            if base_xyz.shape != (self.num_points, 3):
                raise ValueError(
                    f"base affordance xyz shape mismatch: {base_xyz.shape}"
                )

            if not np.allclose(base_xyz, scene_xyz, atol=1e-6):
                raise ValueError(
                    "base affordance point order does not match ADM point order"
                )

    def _validate_values(self):
        arrays = {
            "points": self.points,
            "motion_normalized": self.motion_normalized,
            "motion_padded": self.motion_padded,
            "base_affordance": self.base_affordance,
            "gt_affordance_xyz": self.gt_affordance_xyz,
            "gt_distance": self.gt_distance,
            "gt_affordance": self.gt_affordance,
        }

        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")

        if not np.all(
            (self.base_affordance >= 0.0)
            & (self.base_affordance <= 1.0)
        ):
            raise ValueError("base affordance must be in [0,1]")

        if not np.all(
            (self.gt_affordance >= 0.0)
            & (self.gt_affordance <= 1.0)
        ):
            raise ValueError("GT affordance must be in [0,1]")

        if np.any(self.gt_distance < 0.0):
            raise ValueError("GT distance contains negative values")

    def _prepare_condition_data(self):
        self.scene_points = self.points.copy()

        # RGB: [0,255] -> [0,1]
        self.scene_points[:, 3:6] /= 255.0

        start_xy = np.asarray(
            self.adm_manifest["start_position_xyz"][0:2],
            dtype=np.float32,
        )

        direction_xy = np.asarray(
            self.adm_manifest["history_direction_xy"],
            dtype=np.float32,
        )

        direction_norm = float(np.linalg.norm(direction_xy))

        if direction_norm < 1e-8:
            raise ValueError("history direction is zero")

        direction_xy /= direction_norm

        self.history_state = np.concatenate(
            [start_xy, direction_xy],
            axis=0,
        ).astype(np.float32)

        if self.history_state.shape != (4,):
            raise ValueError(
                f"history_state shape mismatch: {self.history_state.shape}"
            )

        if not np.isfinite(self.history_state).all():
            raise ValueError("history_state contains NaN/Inf")

        self.target_index = np.asarray(
            int(self.pilot_manifest["target_index"]),
            dtype=np.int64,
        )

        if not 0 <= int(self.target_index) < 3:
            raise ValueError("target_index must be 0, 1 or 2")

        self.text = str(self.adm_manifest["text"])

        if not self.text.strip():
            raise ValueError("text prompt is empty")

    def __len__(self):
        return 1

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)

        return {
            # GT motion
            "x": self.motion_normalized.copy(),
            "x_raw": self.motion_padded.copy(),
            "x_mask": self.x_mask.copy(),

            # Scene
            "c_pc_xyz": self.scene_points[:, 0:3].copy(),
            "c_scene_points": self.scene_points.copy(),
            "c_instance_ids": self.instance_ids.copy(),
            "c_candidate_mask": self.candidate_mask.copy(),

            # Affordance
            "c_base_affordance": self.base_affordance.copy(),
            "c_gt_affordance": self.gt_affordance.copy(),
            "c_gt_distance": self.gt_distance.copy(),

            # Conditions
            "c_history_state": self.history_state.copy(),
            "c_target_index": self.target_index.copy(),
            "c_text": self.text,

            # Metadata
            "info_set": "history_pilot",
            "info_index": 0,
            "info_motion_id": self.motion_id,
            "info_scene_trans": np.eye(4, dtype=np.float32),
            "info_scene_mesh": "",
        }
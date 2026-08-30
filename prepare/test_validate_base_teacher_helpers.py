#!/usr/bin/env python3
"""CPU tests for the control flow around the CUDA-only full replay."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PREPARE_DIR = Path(__file__).resolve().parent
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from validate_base_teacher import (  # noqa: E402
    _device_fingerprint,
    _expected_quality_replay_plan,
    _replay_quality_role,
    _validate_quality_replay_coverage,
)
from export_base_teacher import (  # noqa: E402
    GATE0_RUNTIME_EXP_DIR,
    pin_gate0_runtime_output_paths,
    resolved_gate0_cdm_config,
)


def _bundle(sample_ids, k_draws: int = 3):
    sample_ids = list(sample_ids)
    initial = np.arange(
        1, len(sample_ids) * k_draws + 1, dtype=np.int64
    ).reshape(len(sample_ids), k_draws)
    reverse = initial + np.int64(1000)

    def draws(offset: int) -> np.ndarray:
        values = initial.astype(np.float32) + np.float32(offset)
        return np.broadcast_to(
            values[:, :, None, None],
            (len(sample_ids), k_draws, 2, 1),
        ).copy()

    return {
        "sample_ids": sample_ids,
        "fewshot_draws": draws(10),
        "original_draws": draws(20),
        "initial_noise_seeds": initial,
        "reverse_noise_seeds": reverse,
    }


class QualityReplayHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundles = {
            "train_audit": _bundle(["train_a", "train_b"]),
            "development": _bundle(["dev_a"]),
        }
        self.calls = []
        self.loads = []

    def sample_contact(
        self, model, diffusion, row, initial_seed, reverse_seed, device
    ) -> np.ndarray:
        del diffusion
        self.calls.append(
            (model["role"], row["sample_id"], initial_seed, reverse_seed, device)
        )
        value = np.float32(initial_seed + model["offset"])
        return np.full((1, 2, 1), value, dtype=np.float32)

    @staticmethod
    def convert_prediction(value, **kwargs):
        del kwargs
        return np.asarray(value, dtype=np.float32), {"name": "fake"}

    def load_row(self, sample_id):
        self.loads.append(sample_id)
        return {"sample_id": sample_id}

    def replay(self, role, cache):
        return _replay_quality_role(
            replay_model={
                "role": role,
                "offset": 10 if role == "fewshot" else 20,
            },
            replay_diffusion=object(),
            role=role,
            prediction_bundles=self.bundles,
            quality_rows=cache,
            load_row=self.load_row,
            sample_contact=self.sample_contact,
            convert_prediction=self.convert_prediction,
            mean=np.zeros(1, dtype=np.float32),
            std=np.ones(1, dtype=np.float32),
            device="cuda:0",
            show_progress=False,
        )

    def test_all_roles_samples_and_draws_are_covered(self) -> None:
        cache = {}
        results = self.replay("fewshot", cache) + self.replay("original", cache)
        plan = _expected_quality_replay_plan(self.bundles)
        self.assertEqual(plan["total_replayed_draws"], 18)
        self.assertEqual(_validate_quality_replay_coverage(results, plan), 18)
        expected_calls = []
        for role in ("fewshot", "original"):
            for partition in ("train_audit", "development"):
                bundle = self.bundles[partition]
                for sample_index, sample_id in enumerate(bundle["sample_ids"]):
                    for draw_index in range(
                        bundle[role + "_draws"].shape[1]
                    ):
                        expected_calls.append(
                            (
                                role,
                                sample_id,
                                int(
                                    bundle["initial_noise_seeds"][
                                        sample_index, draw_index
                                    ]
                                ),
                                int(
                                    bundle["reverse_noise_seeds"][
                                        sample_index, draw_index
                                    ]
                                ),
                                "cuda:0",
                            )
                        )
        self.assertEqual(self.calls, expected_calls)
        self.assertEqual(
            set((role, sample_id) for role, sample_id, *_ in self.calls),
            {
                ("fewshot", "train_a"),
                ("fewshot", "train_b"),
                ("fewshot", "dev_a"),
                ("original", "train_a"),
                ("original", "train_b"),
                ("original", "dev_a"),
            },
        )
        self.assertEqual(self.loads, ["train_a", "train_b", "dev_a"])

    def test_any_changed_stored_draw_is_rejected(self) -> None:
        self.bundles["development"]["fewshot_draws"][0, 2, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "dev_a/draw_2"):
            self.replay("fewshot", {})

    def test_coverage_rejects_missing_duplicate_or_failed_rows(self) -> None:
        cache = {}
        results = self.replay("fewshot", cache) + self.replay("original", cache)
        plan = _expected_quality_replay_plan(self.bundles)
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            _validate_quality_replay_coverage(results[:-1], plan)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _validate_quality_replay_coverage(results + [results[0]], plan)
        failed = copy.deepcopy(results)
        failed[0]["all_reproduced_bitwise"] = False
        with self.assertRaisesRegex(RuntimeError, "non-bitwise"):
            _validate_quality_replay_coverage(failed, plan)

    def test_plan_rejects_missing_partition_or_role_shape_mismatch(self) -> None:
        missing = {"train_audit": self.bundles["train_audit"]}
        with self.assertRaisesRegex(ValueError, "development"):
            _expected_quality_replay_plan(missing)
        mismatched = copy.deepcopy(self.bundles)
        mismatched["development"]["original_draws"] = np.zeros(
            (1, 2, 2, 1), dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "role draw shapes"):
            _expected_quality_replay_plan(mismatched)


class _FakeCuda:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self):
        return self.available

    @staticmethod
    def get_device_name(device):
        del device
        return "Fake CUDA"

    @staticmethod
    def get_device_capability(device):
        del device
        return (8, 0)


class _FakeTorch:
    def __init__(self, device_type: str, available: bool):
        self.device_type = device_type
        self.cuda = _FakeCuda(available)

    def device(self, value):
        del value
        return type("Device", (), {"type": self.device_type})()


class DeviceContractTests(unittest.TestCase):
    def test_full_replay_is_cuda_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            _device_fingerprint(_FakeTorch("cpu", False), "cpu")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            _device_fingerprint(_FakeTorch("cuda", False), "cuda:0")
        self.assertEqual(
            _device_fingerprint(_FakeTorch("cuda", True), "cuda:0"),
            {
                "device": "cuda:0",
                "device_name": "Fake CUDA",
                "device_capability": [8, 0],
            },
        )


class RuntimeConfigIdentityTests(unittest.TestCase):
    @staticmethod
    def _resolve(cfg):
        fake_omegaconf = types.ModuleType("omegaconf")

        class FakeOmegaConf:
            @staticmethod
            def to_container(value, resolve):
                if resolve is not True:
                    raise AssertionError("configuration must be fully resolved")
                return dict(vars(value))

        fake_omegaconf.OmegaConf = FakeOmegaConf
        with mock.patch.dict(sys.modules, {"omegaconf": fake_omegaconf}):
            return resolved_gate0_cdm_config(cfg)

    def test_wall_clock_output_paths_do_not_change_config_identity(self) -> None:
        first = types.SimpleNamespace(
            exp_dir="outputs/2026-08-21_10-00-00_default",
            log_dir="outputs/2026-08-21_10-00-00_default/log",
            ckpt_dir="outputs/2026-08-21_10-00-00_default/ckpt",
            eval_dir="outputs/2026-08-21_10-00-00_default/eval",
            scientific_value=500,
        )
        second = types.SimpleNamespace(
            exp_dir="outputs/2026-08-21_11-30-42_default",
            log_dir="outputs/2026-08-21_11-30-42_default/log",
            ckpt_dir="outputs/2026-08-21_11-30-42_default/ckpt",
            eval_dir="outputs/2026-08-21_11-30-42_default/eval",
            scientific_value=500,
        )
        first_resolved = self._resolve(first)
        second_resolved = self._resolve(second)
        self.assertEqual(first_resolved, second_resolved)
        self.assertEqual(first_resolved["exp_dir"], GATE0_RUNTIME_EXP_DIR)
        self.assertEqual(
            first_resolved["log_dir"], GATE0_RUNTIME_EXP_DIR + "/log"
        )

    def test_only_runtime_output_paths_are_changed(self) -> None:
        cfg = types.SimpleNamespace(
            exp_dir="volatile",
            log_dir="volatile/log",
            ckpt_dir="volatile/ckpt",
            eval_dir="volatile/eval",
            scientific_value={"steps": 500, "arch": "Perceiver"},
        )
        original_scientific_value = copy.deepcopy(cfg.scientific_value)
        returned = pin_gate0_runtime_output_paths(cfg)
        self.assertIs(returned, cfg)
        self.assertEqual(cfg.scientific_value, original_scientific_value)
        self.assertEqual(cfg.exp_dir, GATE0_RUNTIME_EXP_DIR)
        self.assertEqual(cfg.ckpt_dir, GATE0_RUNTIME_EXP_DIR + "/ckpt")
        self.assertEqual(cfg.eval_dir, GATE0_RUNTIME_EXP_DIR + "/eval")


if __name__ == "__main__":
    unittest.main()

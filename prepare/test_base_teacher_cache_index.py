#!/usr/bin/env python3
"""CPU-only unit contracts for promoted Base-cache indexing."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from build_base_teacher_cache_index import (
    PREFLIGHT_SCHEMA,
    _cache_index_id,
    _validate_promotion_report,
    expected_pairs,
)


class BaseCachePairTests(unittest.TestCase):
    def test_exact_four_scene_prompt_pairs_cover_61_samples(self) -> None:
        rows = {}
        specifications = (
            ("room_0001", "chair", 17),
            ("room_0002", "chair", 6),
            ("room_0003", "bed", 2),
            ("room_0004", "whiteboard", 36),
        )
        for scene_id, target, count in specifications:
            for index in range(count):
                sample_id = "{}_{}_{}".format(scene_id, target, index)
                rows[sample_id] = {
                    "scene_id": scene_id,
                    "target": target,
                    "split": "train" if index % 2 == 0 else "test",
                    "stage": "base",
                }
        pairs = expected_pairs(rows)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(sum(len(value) for value in pairs.values()), 61)
        self.assertIn(
            ("room_0003", "lie_generic_v1", "Lie down somewhere."), pairs
        )
        self.assertIn(
            (
                "room_0004",
                "write_right_hand_generic_v1",
                "Write on a nearby vertical surface with the right hand.",
            ),
            pairs,
        )

    def test_unknown_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported target"):
            expected_pairs(
                {
                    "desk_sample": {
                        "scene_id": "room_0005",
                        "target": "desk",
                    }
                }
            )


class PromotionReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache_key = "1" * 64
        self.result = {
            "manifest_sha256": "2" * 64,
            "artifact_id": "3" * 64,
            "artifact_sha256": "4" * 64,
        }
        self.manifest = {
            "runtime_provenance": {"teacher_runtime_sha256": "5" * 64}
        }
        self.report = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "PASS",
            "promotion_authorized": True,
            "mode": "full",
            "cache_key": self.cache_key,
            **self.result,
            "checks": {"artifact_integrity_pass": True},
            "quality_chain": {"status": "PASS"},
            "full_replay": {
                "status": "PASS",
                "promotion_authorized": True,
                "all_draws_reproduced_bitwise": True,
                "draw_zero_repeatability_bitwise": True,
                "quality_prediction_all_reproduced_bitwise": True,
                "quality_prediction_total_replayed_draws": 370,
                "sampling_calls_preserved_caller_rng_state": True,
                "state_unchanged_during_preflight": True,
                "teacher_runtime_sha256": "5" * 64,
            },
        }

    def validate(self, report):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full_preflight.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            return _validate_promotion_report(
                path,
                canary_cache_key=self.cache_key,
                canary_result=self.result,
                canary_manifest=self.manifest,
            )

    def test_full_370_draw_promotion_report_passes(self) -> None:
        self.assertEqual(self.validate(self.report)["status"], "PASS")

    def test_integrity_only_or_reduced_replay_is_rejected(self) -> None:
        integrity_only = copy.deepcopy(self.report)
        integrity_only["mode"] = "integrity-only"
        with self.assertRaisesRegex(ValueError, "report/canary"):
            self.validate(integrity_only)
        reduced = copy.deepcopy(self.report)
        reduced["full_replay"]["quality_prediction_total_replayed_draws"] = 6
        with self.assertRaisesRegex(ValueError, "replay contract"):
            self.validate(reduced)

    def test_index_id_detects_any_binding_change(self) -> None:
        index = {
            "schema": "synthetic",
            "pairs": [{"cache_key": "a" * 64}],
            "samples": [{"sample_id": "sample_a"}],
        }
        digest = _cache_index_id(index)
        sealed = {**index, "index_id": digest}
        self.assertEqual(_cache_index_id(sealed), digest)
        changed = copy.deepcopy(sealed)
        changed["samples"][0]["sample_id"] = "sample_b"
        self.assertNotEqual(_cache_index_id(changed), digest)


if __name__ == "__main__":
    unittest.main()

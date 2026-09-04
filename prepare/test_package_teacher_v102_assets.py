#!/usr/bin/env python3
"""Synthetic end-to-end contract for the Teacher-v10.2 asset packager."""

from __future__ import annotations

import tempfile
from pathlib import Path

from package_teacher_v102_assets import build_bundle
from teacher_v102_portable_assets import (
    V101_SUMMARY,
    extract_verified_archive,
    json_bytes,
    read_json,
    sha256_file,
    validate_installed_assets,
)


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _write_json(path: Path, value: dict) -> Path:
    return _write(path, json_bytes(value))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="teacher-v102-package-test-") as raw:
        root = Path(raw) / "source"
        root.mkdir()
        source_files = {
            key: _write(root / "prepare" / (key + ".py"), (key + "\n").encode())
            for key in ("runner", "validator", "contract", "objective")
        }
        source_index = _write_json(
            root / "data/history_affordance_relational_teacher_v7_hd/index.json",
            {"schema": "source", "repo": str(root)},
        )
        dataset_index = _write_json(
            root
            / "data/history_affordance_relational_teacher_v9_all_sittable_v1/index.json",
            {
                "schema": "teacher",
                "source_dataset_root": str(source_index.parent),
                "source_index_file": "index.json",
                "source_index_sha256": sha256_file(source_index),
            },
        )
        _write(source_index.parent / "room_0101/points.npz", b"POINTS\n")
        _write(dataset_index.parent / "room_0101/target.npz", b"TARGET\n")

        v5_root = root / "data/history_affordance_v1"
        v5_split = _write_json(v5_root / "splits/split.json", {"schema": "split"})
        _write(v5_root / "room_0001/points.npz", b"V5DATA\n")
        v5_dir = v5_root / "experiments/fewshot_v5"
        v5_checkpoint = _write(v5_dir / "fewshot_cdm.pt", b"V5CKPT\n")
        v5_evidence = _write_json(
            v5_dir / "gate0a_report.json", {"root": str(root)}
        )
        stats = _write(root / "data/stats.npz", b"STATS\n")
        original = _write(
            root / "outputs/CDM-Perceiver-ALL/ckpt/model300000.pt",
            b"ORIGINAL\n",
        )
        v10_failure = _write_json(
            dataset_index.parent / "experiments/teacher_lora_v10/run/summary.json",
            {"schema": "v10", "paths": {"root": str(root)}},
        )
        metric_policy = _write_json(
            dataset_index.parent / "experiments/teacher_lora_v9/policy.json",
            {"policy": True},
        )
        v101_dir = root / V101_SUMMARY.parent
        training_policy = _write_json(v101_dir / "policy.json", {"training": True})
        maps = _write(v101_dir / "maps.npz", b"MAPS\n")
        paths = {
            **source_files,
            "v10_failure": v10_failure,
            "dataset_index": dataset_index,
            "source_dataset_index": source_index,
            "stats_file": stats,
            "v5_split": v5_split,
            "v5_evidence_report": v5_evidence,
            "original_checkpoint": original,
            "v5_checkpoint": v5_checkpoint,
            "metric_policy": metric_policy,
            "training_policy": training_policy,
            "maps": maps,
        }
        summary = {
            "schema": "relational_teacher_v101_fullfield_supervision_v1",
            "status": "FAIL",
            "selected_step": None,
            "serialized_model_state": False,
            "failed_checks": ["at_least_one_fullfield_candidate_passes_actual_k3"],
            "paths": {key: str(path) for key, path in paths.items()},
            "path_sha256": {key: sha256_file(path) for key, path in paths.items()},
        }
        summary_file = _write_json(root / V101_SUMMARY, summary)
        archive_file = Path(raw) / "teacher-v102-assets-v1.tar.gz"
        build_bundle(root, summary_file, archive_file)

        clone = Path(raw) / "clone"
        clone.mkdir()
        for key, source in source_files.items():
            _write(clone / "prepare" / (key + ".py"), source.read_bytes())
        extract_verified_archive(archive_file, clone)
        manifest = validate_installed_assets(clone)
        if manifest["file_count"] != len(manifest["files"]):
            raise AssertionError("portable asset count changed")
        portable_summary = read_json(clone / V101_SUMMARY)
        if not portable_summary.get("portable_paths"):
            raise AssertionError("portable summary marker is absent")
        if any(Path(value).is_absolute() for value in portable_summary["paths"].values()):
            raise AssertionError("portable summary retained an absolute bound path")
        portable_index = read_json(
            clone
            / "data/history_affordance_relational_teacher_v9_all_sittable_v1/index.json"
        )
        if Path(str(portable_index["source_dataset_root"])).is_absolute():
            raise AssertionError("portable dataset index retained its absolute root")

    print("[PASS] Teacher-v10.2 asset packager synthetic end-to-end contract")
    print("[PASS] bound JSON paths are portable and payload hashes verify")


if __name__ == "__main__":
    main()

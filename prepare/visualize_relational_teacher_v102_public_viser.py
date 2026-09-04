#!/usr/bin/env python3
"""Portable read-only Viser for an exported Teacher-v10.2 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import visualize_relational_teacher_v10_supervised_capacity_viser as viewer
from relational_teacher_v102_dense_instance_contract import (
    PROMPT_IDS,
    SCENES,
    SCHEMA,
)


PUBLIC_SCHEMA = "teacher_v102_public_bundle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a path-free Teacher-v10.2 public bundle."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--heatmap-resolution", type=int, default=128)
    parser.add_argument("--heatmap-neighbors", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object: " + str(path))
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(bundle_dir: Path) -> tuple[Mapping[str, object], Path, Path]:
    summary_file = (bundle_dir / "public_summary.json").resolve()
    if not summary_file.is_file():
        raise FileNotFoundError("missing public_summary.json")
    report = read_json(summary_file)
    maps_name = report.get("maps_file")
    if (
        report.get("schema") != SCHEMA
        or report.get("public_schema") != PUBLIC_SCHEMA
        or report.get("train_scenes") != list(SCENES)
        or report.get("prompt_ids") != list(PROMPT_IDS)
        or report.get("machine_paths_removed") is not True
        or report.get("checkpoint_included") is not False
        or not isinstance(maps_name, str)
        or Path(maps_name).name != maps_name
    ):
        raise ValueError("public Teacher-v10.2 bundle authority changed")
    maps_file = (bundle_dir / maps_name).resolve()
    if (
        not maps_file.is_file()
        or maps_file.parent != bundle_dir.resolve()
        or sha256_file(maps_file) != report.get("maps_sha256")
    ):
        raise ValueError("public Teacher-v10.2 map hash changed")
    return report, maps_file, summary_file


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir.expanduser().resolve()
    report, maps_file, summary_file = validate_bundle(bundle_dir)

    viewer.SCHEMA = SCHEMA
    viewer.parse_args = lambda: SimpleNamespace(
        summary=summary_file,
        host=args.host,
        port=args.port,
        heatmap_resolution=args.heatmap_resolution,
        heatmap_neighbors=args.heatmap_neighbors,
    )
    viewer._validate_report = lambda _: (report, maps_file)
    viewer.main()


if __name__ == "__main__":
    main()

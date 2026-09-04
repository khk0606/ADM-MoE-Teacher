#!/usr/bin/env python3
"""Export a path-free, read-only Teacher-v10.2 visualization bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping


SCHEMA = "relational_teacher_v102_dense_instance_supervision_v1"
PUBLIC_SCHEMA = "teacher_v102_public_bundle_v1"
PUBLIC_SUMMARY = "public_summary.json"
PUBLIC_MAPS = "dense_instance_supervision_maps.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export saved Teacher-v10.2 maps without machine-local paths."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def public_report(report: Mapping[str, object]) -> dict[str, object]:
    required = (
        "status",
        "train_scenes",
        "prompt_ids",
        "shortlisted_steps",
        "rollout_seed_table",
        "rollout_rows",
        "selected_step",
        "failed_checks",
        "development_arrays_read",
        "paper_test_access",
        "maps_sha256",
    )
    missing = [name for name in required if name not in report]
    if report.get("schema") != SCHEMA or missing:
        raise ValueError("not a complete Teacher-v10.2 summary: " + ", ".join(missing))
    if report.get("development_arrays_read") is not False:
        raise ValueError("development-array access is not authorized")
    if report.get("paper_test_access") is not False:
        raise ValueError("paper-test access is not authorized")
    payload = {name: report[name] for name in required}
    payload.update(
        {
            "schema": SCHEMA,
            "public_schema": PUBLIC_SCHEMA,
            "maps_file": PUBLIC_MAPS,
            "machine_paths_removed": True,
            "checkpoint_included": False,
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    summary_file = args.summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "output already exists; preserve it and choose a new directory: "
            + str(output_dir)
        )
    report = read_json(summary_file)
    paths = report.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Teacher-v10.2 summary has no bound paths")
    maps_file = Path(str(paths.get("maps", ""))).expanduser().resolve()
    if not maps_file.is_file():
        raise FileNotFoundError("missing saved maps: " + str(maps_file))
    maps_hash = sha256_file(maps_file)
    if maps_hash != report.get("maps_sha256"):
        raise ValueError("saved map hash differs from Teacher-v10.2 summary")

    payload = public_report(report)
    output_dir.mkdir(parents=True)
    try:
        destination = output_dir / PUBLIC_MAPS
        shutil.copy2(maps_file, destination)
        if sha256_file(destination) != maps_hash:
            raise ValueError("copied map hash changed")
        with (output_dir / PUBLIC_SUMMARY).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print("[PUBLIC_BUNDLE_PASS] Teacher-v10.2")
    print("[OK] summary:", output_dir / PUBLIC_SUMMARY)
    print("[OK] maps:", output_dir / PUBLIC_MAPS)
    print("[OK] maps sha256:", maps_hash)
    print("[OK] checkpoint included: False")


if __name__ == "__main__":
    main()

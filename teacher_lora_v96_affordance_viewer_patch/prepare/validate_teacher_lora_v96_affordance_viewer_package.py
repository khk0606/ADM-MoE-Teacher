#!/usr/bin/env python3
"""Verify the immutable Teacher-v9.6 viewer delivery manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_file = ROOT / "PACKAGE_MANIFEST.json"
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "teacher_lora_v96_affordance_viewer_package_v1"
        or value.get("purpose") != "read_only_saved_one_step_map_visualization"
        or value.get("loads_model") is not False
        or value.get("runs_diffusion") is not False
        or value.get("writes_checkpoint") is not False
        or value.get("authorizes_training") is not False
    ):
        raise ValueError("Teacher-v9.6 viewer package policy changed")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.6 viewer package file inventory is absent")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json" and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.6 viewer package inventory changed")
    for relative, expected in files.items():
        path = ROOT / relative
        if sha256_file(path) != expected:
            raise ValueError("Teacher-v9.6 viewer package hash changed: " + relative)
    print("[PACKAGE_PASS] Teacher-v9.6 affordance viewer delivery")
    print("[PASS] {} files and read-only policy verified".format(len(files)))


if __name__ == "__main__":
    main()

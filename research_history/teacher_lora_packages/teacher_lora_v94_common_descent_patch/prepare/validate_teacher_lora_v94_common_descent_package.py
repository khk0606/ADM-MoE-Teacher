#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.4 delivery package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "teacher_lora_v94_common_descent_package_v1":
        raise ValueError("Teacher-v9.4 package schema changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Teacher-v9.4 package file inventory is absent")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts
    }
    if actual != set(files):
        raise ValueError("Teacher-v9.4 package file inventory changed")
    for name, expected in files.items():
        path = root / name
        if sha256_file(path) != expected:
            raise ValueError("Teacher-v9.4 package file changed: " + name)
    policy = manifest.get("authorization")
    expected_policy = {
        "common_descent_response6_on_pass": True,
        "checkpoint_write": False,
        "rollout": False,
        "overfit120": False,
        "calibration": False,
        "long_training": False,
        "room_0102_arrays": False,
        "room_0201_arrays": False,
        "paper_test": False,
    }
    if policy != expected_policy:
        raise ValueError("Teacher-v9.4 authorization policy changed")
    print("[PACKAGE_PASS] Teacher-v9.4 common-descent preflight delivery")
    print(f"[PASS] {len(files)} files and fail-closed authorization verified")


if __name__ == "__main__":
    main()

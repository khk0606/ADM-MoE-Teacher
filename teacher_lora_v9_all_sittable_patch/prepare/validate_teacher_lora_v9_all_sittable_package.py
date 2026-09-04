#!/usr/bin/env python3
"""Fail-closed integrity check for the extracted Teacher-v9 delivery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "PACKAGE_MANIFEST.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "teacher_lora_v9_all_sittable_data_gate_package_v1"
        or value.get("status") != "PACKAGE_PASS"
        or value.get("authorization", {}).get("lora_training") is not False
        or value.get("authorization", {}).get("development_array_access") is not False
        or value.get("authorization", {}).get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v9 package policy changed")
    expected = value.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Teacher-v9 package file inventory is absent")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual != set(expected):
        raise ValueError(
            "Teacher-v9 package inventory changed: "
            f"missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Teacher-v9 package hash mismatch: {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
    print("[PACKAGE_PASS] Teacher-v9 all-sittable delivery integrity")
    print(f"[PASS] {len(expected)} internal files and authorization policy verified")


if __name__ == "__main__":
    main()

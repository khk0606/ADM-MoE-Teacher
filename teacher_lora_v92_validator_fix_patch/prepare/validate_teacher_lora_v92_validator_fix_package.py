#!/usr/bin/env python3
"""Verify the Teacher-v9.2 validator-correction delivery."""

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
    manifest_file = root / "PACKAGE_MANIFEST.json"
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "teacher_lora_v92_validator_fix_package_v1"
        or value.get("status") != "PACKAGE_PASS"
        or value.get("changes")
        != ["cpu_cuda_reduction_tolerance", "read_only_failure_summary"]
        or value.get("rerun_cuda") is not False
        or value.get("changes_model_result") is not False
        or value.get("authorizes_training") is not False
    ):
        raise ValueError("Teacher-v9.2 validator-fix policy changed")
    expected = value.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("validator-fix inventory missing")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_file
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual != set(expected):
        raise ValueError("validator-fix package inventory changed")
    for name, digest in expected.items():
        if sha256_file(root / name) != digest:
            raise ValueError("validator-fix package hash mismatch: " + name)
    print("[PACKAGE_PASS] Teacher-v9.2 validator correction delivery")
    print(f"[PASS] {len(expected)} read-only files verified")


if __name__ == "__main__":
    main()

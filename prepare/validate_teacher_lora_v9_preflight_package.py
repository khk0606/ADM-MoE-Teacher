#!/usr/bin/env python3
"""Fail-closed integrity check for the Teacher-v9 CUDA preflight delivery."""

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
    authorization = value.get("authorization")
    if (
        value.get("schema")
        != "teacher_lora_v9_all_sittable_cuda_preflight_package_v3"
        or value.get("status") != "PACKAGE_PASS"
        or not isinstance(authorization, dict)
        or authorization.get("cuda_preflight") is not True
        or authorization.get("one_scene_overfit_after_validated_pass") is not True
        or authorization.get("optimizer_updates_in_this_package") != 0
        or authorization.get("response3") is not False
        or authorization.get("calibration") is not False
        or authorization.get("long_training") is not False
        or authorization.get("development_array_access") is not False
        or authorization.get("paper_test_access") is not False
        or value.get("teacher_forward_inputs")
        != ["c_pc_feat", "c_pc_xyz", "c_text"]
        or value.get("primary_loss")
        != "equal_macro_of_bed_normal_chair_high_chair"
    ):
        raise ValueError("Teacher-v9 preflight package policy changed")
    expected = value.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Teacher-v9 preflight package inventory is absent")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_file
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual != set(expected):
        raise ValueError(
            "Teacher-v9 preflight package inventory changed: "
            f"missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Teacher-v9 preflight hash mismatch: {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
    print("[PACKAGE_PASS] Teacher-v9 all-sittable CUDA preflight delivery")
    print(f"[PASS] {len(expected)} files and staged authorization policy verified")


if __name__ == "__main__":
    main()

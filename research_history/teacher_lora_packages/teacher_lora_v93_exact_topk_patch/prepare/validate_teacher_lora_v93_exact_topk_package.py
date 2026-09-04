#!/usr/bin/env python3
"""Verify the Teacher-v9.3 exact top-k response delivery."""

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
        value.get("schema") != "teacher_lora_v93_exact_topk_package_v1"
        or value.get("status") != "PACKAGE_PASS"
        or value.get("source_model") != "sealed_v5r4_teacher_plus_fresh_zero_init_lora"
        or value.get("train_scene") != "room_0101"
        or value.get("target_topk_definition")
        != "GT>=0.30_then_ceil(active_count*0.25)_stable_value_descending"
        or value.get("teacher_forward_inputs") != ["c_pc_feat", "c_pc_xyz", "c_text"]
        or not isinstance(authorization, dict)
        or authorization.get("response_grid") is not True
        or authorization.get("candidates") != 3
        or authorization.get("updates_per_candidate") != 6
        or authorization.get("learning_rate_grid") != [1e-5, 2e-5, 4e-5]
        or authorization.get("checkpoint_export") is not False
        or authorization.get("rollout_canary_after_validated_pass") is not True
        or authorization.get("overfit120") is not False
        or authorization.get("calibration") is not False
        or authorization.get("long_training") is not False
        or authorization.get("room_0102_array_access") is not False
        or authorization.get("room_0201_array_access") is not False
        or authorization.get("paper_test_access") is not False
    ):
        raise ValueError("Teacher-v9.3 package policy changed")
    expected = value.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Teacher-v9.3 package inventory is absent")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_file
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual != set(expected):
        raise ValueError("Teacher-v9.3 package inventory changed")
    for name, digest in expected.items():
        if sha256_file(root / name) != digest:
            raise ValueError("Teacher-v9.3 package hash mismatch: " + name)
    print("[PACKAGE_PASS] Teacher-v9.3 exact top-k response delivery")
    print(f"[PASS] {len(expected)} files and fail-closed authorization verified")


if __name__ == "__main__":
    main()


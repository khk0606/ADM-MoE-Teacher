#!/usr/bin/env python3
"""Validate the immutable Teacher-v9.8.12 viewer package."""

from __future__ import annotations

import ast
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
    if manifest.get("schema") != "teacher_lora_v9812_affordance_viewer_package_v1":
        raise ValueError("Teacher-v9.8.12 viewer package schema changed")
    files = manifest.get("files")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts
    }
    if not isinstance(files, dict) or actual != set(files):
        raise ValueError("Teacher-v9.8.12 viewer package inventory changed")
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Teacher-v9.8.12 viewer file changed: " + name)
    expected = {
        "saved_room0101_maps": True,
        "saved_room0102_maps": True,
        "all_sittable_and_per_object_gt": True,
        "start_and_six_updates": True,
        "continuous_xy_display_interpolation": True,
        "model_load": False,
        "diffusion": False,
        "optimizer": False,
        "checkpoint_write": False,
        "room_0201_arrays": False,
        "paper_test": False,
    }
    if manifest.get("authorization") != expected:
        raise ValueError("Teacher-v9.8.12 viewer authorization changed")
    viewer = root / "prepare/visualize_relational_teacher_v9812_affordance_viser.py"
    source = viewer.read_text(encoding="utf-8")
    ast.parse(source)
    common = root / "prepare/relational_teacher_v9812_affordance_viewer_common.py"
    test = root / "prepare/test_relational_teacher_v9812_affordance_viewer_common.py"
    ast.parse(common.read_text(encoding="utf-8"))
    ast.parse(test.read_text(encoding="utf-8"))
    for forbidden in (
        "import torch",
        "create_model_and_diffusion",
        "load_ckpt",
        "torch.optim",
        "torch.save",
        "load_train_scene_bundle",
    ):
        if forbidden in source:
            raise ValueError("viewer contains forbidden operation: " + forbidden)
    for required in (
        'report.get("status") != "FAIL"',
        'report.get("shortlisted_steps") != []',
        '"calibration_maps"',
        'arrays["candidates"]',
        'arrays["start"]',
        "interpolate_xy_heatmap(",
        "signed_difference_colors(",
        "[VIEWER_PASS]",
    ):
        if required not in source:
            raise ValueError("viewer guard changed: " + required)
    print("[PACKAGE_PASS] Teacher-v9.8.12 affordance viewer delivery")
    print("[PASS] {} files and saved-map-only authorization verified".format(len(files)))
    print("[PASS] two-scene GT/Start/updates, XY rendering and no-model guards")


if __name__ == "__main__":
    main()

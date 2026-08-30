#!/usr/bin/env python3
"""Synthetic contract test for build_fewshot_split.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "build_fewshot_split.py"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def add_sample(
    root: Path,
    sample_id: str,
    scene_id: str,
    target: str,
    text: str,
    stage: str = "base",
    original: str = "",
    components=None,
    parent: str = "",
) -> None:
    sample_dir = root / "samples" / sample_id
    write_json(
        sample_dir / "sample_manifest.json",
        {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "target_name": target,
            "text": text,
        },
    )
    if original:
        write_json(
            sample_dir / "derivation_manifest.json",
            {
                "stage": stage,
                "original_motion_id": original,
                "parent_base_id": parent,
                "source_component_ids": list(components or []),
            },
        )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fewshot_split_test_") as temp:
        root = Path(temp) / "unity"
        for index in range(1, 18):
            sample_id = f"room_0001_sit_chair_{index:04d}"
            add_sample(root, sample_id, "room_0001", "chair", "Sit somewhere")
        for index in range(1, 7):
            sample_id = f"room_0002_sit_chair_{index:04d}"
            add_sample(root, sample_id, "room_0002", "chair", "Sit somewhere")

        for index, component in ((2, "bed:b2"), (5, "bed:b1")):
            sample_id = f"room_0003_lie_bed_{index:04d}"
            add_sample(
                root,
                sample_id,
                "room_0003",
                "bed",
                "Lie down somewhere",
                original=component,
                components=[component],
            )

        for offset in range(12):
            base_index = offset + 2
            original = f"wb:route:{offset:02d}"
            interaction = "wb:interaction:w1" if offset % 2 == 0 else "wb:interaction:w2"
            base_id = f"room_0004_interact_whiteboard_{base_index:04d}"
            add_sample(
                root,
                base_id,
                "room_0004",
                "whiteboard",
                "Interact with something",
                original=original,
                components=[original, interaction],
            )
            for block in (0, 1):
                derived_index = 14 + offset + block * 12
                derived_id = f"room_0004_interact_whiteboard_{derived_index:04d}"
                add_sample(
                    root,
                    derived_id,
                    "room_0004",
                    "whiteboard",
                    "Interact with something",
                    stage="multistart",
                    original=original,
                    components=[original, interaction],
                    parent=base_id,
                )

        output = Path(temp) / "split.json"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--unity-root",
                str(root),
                "--output",
                str(output),
            ],
            check=True,
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        assert len(value["samples"]) == 61
        assert len(value["cdm_fewshot"]["train"]) + len(
            value["cdm_fewshot"]["test"]
        ) == 37
        assert len(value["moe_iiw"]["train"]) + len(value["moe_iiw"]["test"]) == 61
        cdm_ids = value["cdm_fewshot"]["train"] + value["cdm_fewshot"]["test"]
        assert all("_00" not in sample_id or int(sample_id[-4:]) < 14 for sample_id in cdm_ids if "whiteboard" in sample_id)
        assert all(value["checks"].values())
        print("[PASS] synthetic raw-source split contract")


if __name__ == "__main__":
    main()
